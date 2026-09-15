"""Scheduling: chunk-state detection and job-script generation.

No scheduler is invoked; these are about the decisions made before submission.
"""

from pathlib import Path

import sys

import pytest
import yaml

from edna_sampling.backends import (COMPLETE, MISSING, PARTIAL, chunk_dir, chunk_state,
                                    incomplete_chunks, run_status, submit_local)
from edna_sampling.cli import _parse_chunks
from edna_sampling.params import root_output_dir
from edna_sampling.config import ConfigError, MachineProfile, SiteConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_site(tmp_path, site_dict):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "poly.csv").write_text("")
    site_dict["release_points"]["n"] = 6
    site_dict["chunking"]["release_groups_per_chunk"] = 2   # -> 3 chunks
    p = site_dir / "v1.yaml"
    p.write_text(yaml.safe_dump(site_dict))
    return SiteConfig.load(p)


@pytest.fixture
def prof(tmp_path, profile_dict):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(profile_dict))
    return MachineProfile.load(p)


def _finish(site, profile, n, *, complete=True):
    d = chunk_dir(site, profile, n)
    d.mkdir(parents=True, exist_ok=True)
    (d / "caseInfo.json").write_text("{}")
    if complete:
        for spec in site.stats:
            (d / f"stats_gridded_time_3D_000_{spec.name}.nc").write_text("")


# --- chunk state -----------------------------------------------------------

def test_missing_when_nothing_written(small_site, prof):
    assert chunk_state(small_site, prof, 1) == MISSING


def test_partial_when_case_file_only(small_site, prof):
    """A job that dies partway leaves a directory and a case file. Counting that
    as done would silently drop release groups from the analysis."""
    _finish(small_site, prof, 1, complete=False)
    assert chunk_state(small_site, prof, 1) == PARTIAL


def test_partial_when_a_statistic_is_absent(small_site, prof):
    d = chunk_dir(small_site, prof, 1)
    d.mkdir(parents=True)
    (d / "caseInfo.json").write_text("{}")
    (d / "stats_gridded_time_3D_000_other_name.nc").write_text("")
    assert chunk_state(small_site, prof, 1) == PARTIAL


def test_complete_when_case_file_and_all_statistics_present(small_site, prof):
    _finish(small_site, prof, 1)
    assert chunk_state(small_site, prof, 1) == COMPLETE


def test_run_status_and_incomplete(small_site, prof):
    _finish(small_site, prof, 1)
    _finish(small_site, prof, 2, complete=False)
    status = run_status(small_site, prof)
    assert status[COMPLETE] == [1]
    assert status[PARTIAL] == [2]
    assert status[MISSING] == [3]
    assert incomplete_chunks(small_site, prof) == [2, 3]


# --- chunk selection -------------------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("5", [5]),
    ("1-3", [1, 2, 3]),
    ("1-3,7", [1, 2, 3, 7]),
    ("7,1-3", [1, 2, 3, 7]),
    ("2-3,3-4", [2, 3, 4]),
])
def test_parse_chunks(spec, expected):
    assert _parse_chunks(spec, 10) == expected


@pytest.mark.parametrize("spec", ["1-abc", "", "3-1"])
def test_parse_chunks_rejects_nonsense(spec):
    with pytest.raises(ConfigError):
        _parse_chunks(spec, 10)


def test_parse_chunks_rejects_out_of_range():
    with pytest.raises(ConfigError, match="outside 1-10"):
        _parse_chunks("9-12", 10)


# --- interpreter choice -----------------------------------------------------

def test_active_interpreter_is_used_by_default(small_site, prof):
    """A chunk runs under whatever environment the user has active."""
    import sys
    lines = []
    submit_local(small_site, prof, [1], dry_run=True, echo=lines.append)
    assert sys.executable in lines[0]


def test_explicit_python_wins(small_site, tmp_path):
    """`python:` pins the environment chunks run in, whatever submits them.

    This is the hook a cluster user needs: submit from anywhere, run the chunk
    under a named environment.
    """
    p = tmp_path / "pinned.yaml"
    p.write_text(yaml.safe_dump({
        "hindcasts": {"some_hindcast": "/h"},
        "output_root": str(tmp_path / "out"),
        "python": "/opt/env/bin/python",
    }))
    prof = MachineProfile.load(p)
    lines = []
    submit_local(small_site, prof, [1], dry_run=True, echo=lines.append)
    assert lines[0].startswith("/opt/env/bin/python -m edna_sampling.cli")


# --- local backend ---------------------------------------------------------

def test_local_dry_run_prints_one_command_per_chunk(small_site, prof):
    lines = []
    result = submit_local(small_site, prof, [1, 2, 3], dry_run=True, echo=lines.append)
    assert result.launched == 0
    assert len(lines) == 3
    assert all("edna_sampling.cli run" in line for line in lines)
    assert lines[0].endswith("--chunk 1")


def test_local_rejects_zero_parallelism(small_site, prof):
    with pytest.raises(ConfigError, match="n_parallel_jobs"):
        submit_local(small_site, prof, [1], n_parallel=0)


# --- chunk output goes to a log, not the terminal ---------------------------

def _noisy(monkeypatch, *, fail: set[int] = frozenset(), lines: int = 30):
    """Stand in for a chunk: prints to stdout, then to stderr, then maybe fails."""
    import edna_sampling.backends as backends

    def command(site, profile, n):
        return [sys.executable, "-c",
                f"import sys;[print('noise', i) for i in range({lines})];"
                f"print('BOOM {n}', file=sys.stderr);"
                f"sys.exit(1 if {n} in {set(fail) or set()} else 0)"]
    monkeypatch.setattr(backends, "_run_command", command)


def _run(small_site, prof, chunks, monkeypatch, **kw):
    said = []
    result = submit_local(small_site, prof, chunks, poll_seconds=0.05,
                          echo=said.append, **kw)
    return said, result


def test_chunk_output_does_not_reach_the_terminal(small_site, prof, monkeypatch):
    """n_parallel OceanTracker runs interleaved on one terminal is unreadable,
    and every line is already on disk."""
    _noisy(monkeypatch)
    said, _ = _run(small_site, prof, [1], monkeypatch)
    assert not any("noise" in line for line in said)
    assert any("chunk 1: done" in line for line in said)


def test_chunk_output_is_written_to_a_log(small_site, prof, monkeypatch):
    _noisy(monkeypatch, lines=30)
    _run(small_site, prof, [1], monkeypatch)
    log = root_output_dir(small_site, prof) / "logs" / "chunk_001.log"
    assert log.is_file()
    assert "noise 29" in log.read_text()


def test_a_failing_chunk_reports_why_without_being_hunted_for(small_site, prof, monkeypatch):
    """An exit code alone would leave the reason in a file the caller has to go
    and find. The tail is only trustworthy because the child runs unbuffered:
    otherwise stderr lands before block-buffered stdout and the tail shows
    progress lines instead of the error."""
    _noisy(monkeypatch, fail={1})
    said, result = _run(small_site, prof, [1], monkeypatch)
    assert result.failed == [1]
    joined = "\n".join(said)
    assert "FAILED" in joined
    assert "BOOM 1" in joined
    assert "chunk_001.log" in joined


def test_verbose_streams_instead_of_logging(small_site, prof, monkeypatch):
    """The firehose is still available when a single chunk is being debugged."""
    _noisy(monkeypatch)
    _run(small_site, prof, [1], monkeypatch, verbose=True)
    assert not (root_output_dir(small_site, prof) / "logs").exists()
