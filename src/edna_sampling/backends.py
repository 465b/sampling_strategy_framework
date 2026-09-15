"""Launching a run's chunks locally, and reporting progress.

A production run is one job per chunk (400 for the current experiments), started
as a bounded pool of `edna run --chunk N` child processes.

Chunk numbers are always an explicit list, so `--resume` is just "submit the
chunks that are not finished" rather than a special mode.

There is deliberately no scheduler backend. `edna run --chunk N` is the whole
integration point: one chunk, one process, no shared state between chunks, so a
cluster user wraps it in whatever their site expects - a SLURM array job, a PBS
script, GNU parallel - without this package needing to model any of them.
`edna status --format chunks --resume` prints the outstanding chunks as an array
spec, which is what makes that practical.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from edna_sampling.config import ConfigError, MachineProfile, SiteConfig
from edna_sampling.params import chunk_run_name, root_output_dir

__all__ = [
    "COMPLETE", "PARTIAL", "MISSING",
    "chunk_dir", "chunk_state", "run_status", "incomplete_chunks",
    "submit_local",
]

COMPLETE = "complete"
PARTIAL = "partial"     # directory exists but the expected output does not
MISSING = "missing"


def chunk_dir(site: SiteConfig, profile: MachineProfile, chunk_number: int) -> Path:
    """Directory OceanTracker writes one chunk into."""
    return root_output_dir(site, profile) / chunk_run_name(site, chunk_number)


def chunk_state(site: SiteConfig, profile: MachineProfile, chunk_number: int) -> str:
    """Whether a chunk has finished, started, or not been run.

    "Finished" means caseInfo.json plus one statistics file per configured
    statistic. Checking the stats files matters: a job that dies partway leaves
    a directory and a case file behind, and counting that as done would silently
    drop release groups from the analysis.
    """
    d = chunk_dir(site, profile, chunk_number)
    if not d.is_dir():
        return MISSING
    if not (d / "caseInfo.json").is_file():
        return PARTIAL
    for spec in site.stats:
        if not any(d.glob(f"stats_*{spec.name}.nc")):
            return PARTIAL
    return COMPLETE


def run_status(site: SiteConfig, profile: MachineProfile) -> dict[str, list[int]]:
    """Chunk numbers grouped by state."""
    out: dict[str, list[int]] = {COMPLETE: [], PARTIAL: [], MISSING: []}
    for n in range(1, site.n_chunks + 1):
        out[chunk_state(site, profile, n)].append(n)
    return out


def incomplete_chunks(site: SiteConfig, profile: MachineProfile) -> list[int]:
    """Chunks that still need running, in order."""
    status = run_status(site, profile)
    return sorted(status[PARTIAL] + status[MISSING])


def _interpreter(profile: MachineProfile) -> str:
    """Which python runs a chunk.

    Explicit `python:` in the profile always wins - which is how a cluster user
    points chunks at an environment that differs from the one submitting them.
    Otherwise use the interpreter we are running under, which is what the user
    has active.
    """
    if profile.python:
        return profile.python
    return sys.executable


def _run_command(site: SiteConfig, profile: MachineProfile, chunk_number: int) -> list[str]:
    """Command that runs one chunk.

    Invoked via `-m edna_sampling.cli` rather than the `edna` script so it works
    whether or not the environment's bin directory is on PATH - which it often
    is not inside a batch job.
    """
    return [_interpreter(profile), "-m", "edna_sampling.cli", "run",
            str(site.source_path), "--profile", str(_profile_ref(profile)),
            "--chunk", str(chunk_number)]


def _profile_ref(profile: MachineProfile) -> str:
    """How to name this profile in a generated command.

    The absolute path, not the short name: a batch job starts in an arbitrary
    working directory, where name lookup against ./profiles/ would not resolve.
    """
    return str(profile.source_path) if profile.source_path else profile.name


# --------------------------------------------------------------------------
# local backend
# --------------------------------------------------------------------------

def _tail(path: Path, n: int = 12) -> list[str]:
    """The last `n` non-blank lines of a log, for reporting a failure in place."""
    try:
        lines = [l.rstrip() for l in path.read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return []
    return lines[-n:]


@dataclass
class LocalResult:
    launched: int
    failed: list[int]


def submit_local(site: SiteConfig, profile: MachineProfile, chunks: list[int],
                 *, n_parallel: int | None = None, poll_seconds: float = 5.0,
                 dry_run: bool = False, verbose: bool = False, echo=print) -> LocalResult:
    """Run `chunks` as child processes, at most `n_parallel` at a time.

    Each chunk's console output goes to `<output>/logs/chunk_NNN.log` rather than
    to the terminal, which would otherwise carry the interleaved progress of
    `n_parallel` OceanTracker runs at once. Nothing is lost: OceanTracker writes
    its own `run_log.txt` into every chunk directory, and these files additionally
    catch anything that fails before that log exists. `verbose=True` restores the
    firehose.

    When a chunk fails, the tail of its log is printed - otherwise the only signal
    would be an exit code, with the reason in a file the caller has to go and
    find."""
    if n_parallel is None:
        n_parallel = int(profile.local.get("n_parallel_jobs", 1))
    if n_parallel < 1:
        raise ConfigError(f"profile {profile.name!r}: local.n_parallel_jobs must be >= 1")

    if dry_run:
        for n in chunks:
            echo(" ".join(shlex.quote(c) for c in _run_command(site, profile, n)))
        return LocalResult(launched=0, failed=[])

    log_dir = root_output_dir(site, profile) / "logs"
    if not verbose:
        log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(n: int) -> Path:
        return log_dir / f"chunk_{n:03d}.log"

    running: dict[int, subprocess.Popen] = {}
    handles: dict[int, object] = {}
    failed: list[int] = []
    pending = list(chunks)
    launched = 0

    while pending or running:
        for n, proc in list(running.items()):
            if proc.poll() is not None:
                del running[n]
                handle = handles.pop(n, None)
                if handle is not None:
                    handle.close()
                if proc.returncode != 0:
                    failed.append(n)
                    echo(f"chunk {n}: FAILED (exit {proc.returncode})")
                    if not verbose:
                        echo(f"  {_log_path(n)}")
                        for line in _tail(_log_path(n)):
                            echo(f"  | {line}")
                else:
                    echo(f"chunk {n}: done")
        while pending and len(running) < n_parallel:
            n = pending.pop(0)
            if verbose:
                running[n] = subprocess.Popen(_run_command(site, profile, n))
            else:
                handles[n] = _log_path(n).open("w")
                # Unbuffered, or the merged log lies about order: writing to a
                # file, the child's stdout block-buffers while its stderr does
                # not, so a traceback can land *before* the output it followed -
                # and the failure tail below would then show progress lines
                # instead of the error.
                env = dict(os.environ, PYTHONUNBUFFERED="1")
                running[n] = subprocess.Popen(_run_command(site, profile, n),
                                              stdout=handles[n],
                                              stderr=subprocess.STDOUT,
                                              env=env)
            launched += 1
            echo(f"chunk {n}: started (pid {running[n].pid}), "
                 f"{len(running)}/{n_parallel} running, {len(pending)} queued")
        if running:
            time.sleep(poll_seconds)

    return LocalResult(launched=launched, failed=sorted(failed))
