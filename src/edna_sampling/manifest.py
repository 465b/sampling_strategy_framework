"""Per-run provenance, written next to the output.

A run is 400 jobs spread over hours. Nothing stops someone editing the site
config while they are in flight, at which point early and late chunks belong to
different experiments and the resulting arrays are quietly inconsistent.

The manifest records what the run was started as. `edna run` compares the config
it was handed against it and refuses to add a chunk that does not belong, which
turns a silent inconsistency into an error at the point it happens.

It doubles as the provenance record: config, versions, git commit, and a hash of
the release points the run is pinned to.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from edna_sampling import __version__
from edna_sampling.config import ConfigError, MachineProfile, SiteConfig
from edna_sampling.params import root_output_dir

__all__ = ["manifest_path", "write_manifest", "read_manifest", "check_manifest"]

MANIFEST_NAME = "manifest.json"


def manifest_path(site: SiteConfig, profile: MachineProfile) -> Path:
    return root_output_dir(site, profile) / MANIFEST_NAME


def _git_state() -> dict:
    """Commit and dirtiness of this working tree, if it is a git checkout."""
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    if commit.returncode != 0:
        return {}
    return {"commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}


def _oceantracker_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("oceantracker")
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(site: SiteConfig, profile: MachineProfile,
                   release_points_file: Path) -> Path:
    """Record what this run is, at the moment it starts."""
    payload = {
        "run_name": site.run_name,
        "fingerprint": site.fingerprint(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_chunks": site.n_chunks,
        "release_groups_per_chunk": site.chunking.release_groups_per_chunk,
        "config_path": str(site.source_path),
        "config": json.loads(json.dumps(asdict(site), default=str)),
        "profile": {"name": profile.name,
                    "path": str(profile.source_path) if profile.source_path else None},
        "release_points": {
            "path": str(release_points_file),
            "n": site.release_points.n,
            "sha256": file_sha256(release_points_file),
        },
        "versions": {
            "edna_sampling": __version__,
            "oceantracker": _oceantracker_version(),
        },
        "git": _git_state(),
    }
    path = manifest_path(site, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(site: SiteConfig, profile: MachineProfile) -> dict | None:
    path = manifest_path(site, profile)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: manifest is not valid JSON - {exc}") from exc


def check_manifest(site: SiteConfig, profile: MachineProfile,
                   release_points_file: Path) -> None:
    """Raise if this config no longer matches the run already in progress.

    Absent manifest is fine: runs started before this existed, and the first
    chunk of a new run, both legitimately have none.
    """
    recorded = read_manifest(site, profile)
    if recorded is None:
        return

    current = site.fingerprint()
    if recorded.get("fingerprint") != current:
        raise ConfigError(
            f"{site.run_name}: the config has changed since this run was started.\n"
            f"  manifest {recorded.get('fingerprint')}  (written {recorded.get('created')})\n"
            f"  config   {current}\n"
            f"  Chunks already written belong to the earlier definition. Bump `version` to\n"
            f"  start a separate run, or pass --ignore-manifest if you know the change is\n"
            f"  immaterial."
        )

    recorded_points = (recorded.get("release_points") or {}).get("sha256")
    if recorded_points and release_points_file.is_file():
        actual = file_sha256(release_points_file)
        if actual != recorded_points:
            raise ConfigError(
                f"{site.run_name}: the release points have changed since this run was "
                f"started.\n  Chunks already written used different source locations.\n"
                f"  Bump `version` to start a separate run, or pass --ignore-manifest."
            )
