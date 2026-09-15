"""CLI behaviour, especially the error paths - for a new user those messages
are the whole interface."""

import json
from pathlib import Path

import pytest
import yaml

from edna_sampling.cli import main

REPO = Path(__file__).resolve().parents[1]
CAPE = str(REPO / "examples" / "cape_rodney" / "config.yml")


@pytest.fixture
def sandbox_profile(tmp_path):
    # the directory has to exist: `validate_inputs` checks it, so that a missing
    # hindcast is reported before 400 jobs are launched at it
    (tmp_path / "hindcast").mkdir()
    p = tmp_path / "sandbox.yaml"
    p.write_text(yaml.safe_dump({
        "hindcasts": {"hauraki_gulf": {"input_dir": str(tmp_path / "hindcast"),
                                               "file_mask": "schout_*.nc"}},
        "output_root": str(tmp_path / "out"),
    }))
    return str(p)


@pytest.fixture
def cape_sandbox(tmp_path, sandbox_profile):
    """A copy of the cape_rodney example with its release points already cached.

    `examples/` ships no point files - they are a deterministic function of the
    config and the hindcast, so committing them would pin a derived artifact.
    Anything exercising `run`, `params` or `status` therefore brings its own
    cache, which also keeps these tests independent of the example's contents.
    """
    site_dir = tmp_path / "cape"
    site_dir.mkdir()
    (site_dir / "poly.csv").write_text("")
    cfg = yaml.safe_load(Path(CAPE).read_text())
    cfg["domain"] = "poly.csv"
    (site_dir / "config.yml").write_text(yaml.safe_dump(cfg))

    # the cache name is keyed by run_name, so read it off the config rather than
    # spelling it out - the example's `version:` is not ours to depend on
    n = cfg["release_points"]["n"]
    run_name = f"{cfg['version']}_{cfg['name']}"
    (site_dir / f"{run_name}_lloyd_points.csv").write_text(
        "".join(f"{174.80 + 0.0001 * i},{-36.28 - 0.0001 * i}\n" for i in range(n)))
    return str(site_dir / "config.yml")


def test_info_without_profile(capsys):
    assert main(["info", CAPE]) == 0
    out = capsys.readouterr().out
    assert "cape_rodney" in out
    assert "100 x 1 release group(s)" in out
    # the vertical selection is surfaced, not hidden: the tow window is what
    # decides how much of the water column is sampled at all
    assert "[whole column]" in out


def test_info_fails_fast_on_bad_profile(capsys):
    assert main(["info", CAPE, "--profile", "does_not_exist"]) == 2
    cap = capsys.readouterr()
    assert "no machine profile" in cap.err
    # nothing printed before the failure
    assert cap.out == ""


def test_missing_profile_is_explained(capsys):
    assert main(["points", CAPE]) == 2
    assert "no machine profile given" in capsys.readouterr().err


def test_run_refuses_when_points_not_generated(capsys, tmp_path, sandbox_profile):
    """Every chunk must see identical sources, so `run` never generates them
    itself - that would race across jobs."""
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "poly.csv").write_text("")
    cfg = yaml.safe_load(Path(CAPE).read_text())
    cfg["domain"] = "poly.csv"
    path = site_dir / "config.yml"
    path.write_text(yaml.safe_dump(cfg))

    assert main(["run", str(path), "--profile", sandbox_profile, "--chunk", "1", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "release points not generated yet" in err
    assert "edna points" in err          # tells you the command to run


def test_chunk_out_of_range(capsys, sandbox_profile, cape_sandbox):
    assert main(["run", cape_sandbox, "--profile", sandbox_profile,
                 "--chunk", "401", "--dry-run"]) == 2
    assert "out of range" in capsys.readouterr().err


def test_params_emits_valid_json_for_the_right_chunk(capsys, sandbox_profile,
                                                     cape_sandbox):
    assert main(["params", cape_sandbox, "--profile", sandbox_profile,
                 "--chunk", "7"]) == 0
    params = json.loads(capsys.readouterr().out)
    assert params["output_file_base"].endswith("cape_rodney_chunk_007")
    # chunk 7 at 1 group per chunk is the 7th point, i.e. global index 6
    assert params["release_groups"][0]["name"] == "node_0006"


def test_points_loads_cache_without_regenerating(capsys, tmp_path, sandbox_profile):
    """A warm cache must not re-run the relaxation: every chunk of a run has to
    see the same sources, and regenerating would need the mesh and the hindcast.

    The cache is written here rather than read from `examples/`, which ships no
    point files - they are generated artifacts. A fixture that builds its own is
    also the only way this can assert the count it expects.
    """
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "poly.csv").write_text("")
    cfg = yaml.safe_load(Path(CAPE).read_text())
    cfg["domain"] = "poly.csv"
    cfg["release_points"]["n"] = 3
    (site_dir / "config.yml").write_text(yaml.safe_dump(cfg))
    (site_dir / f"{cfg['version']}_{cfg['name']}_lloyd_points.csv").write_text(
        "174.80,-36.28\n174.81,-36.29\n174.82,-36.30\n")

    assert main(["points", str(site_dir / "config.yml"),
                 "--profile", sandbox_profile, "--no-figure"]) == 0
    assert "loaded cached 3 release points" in capsys.readouterr().out


def test_dry_run_creates_output_dir_but_does_not_model(capsys, tmp_path, sandbox_profile,
                                                       cape_sandbox):
    assert main(["run", cape_sandbox, "--profile", sandbox_profile,
                 "--chunk", "3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert list((tmp_path / "out").glob("*_cape_rodney/*_cape_rodney_chunk_003"))


def test_status_format_chunks_is_the_whole_run_by_default(monkeypatch, capsys,
                                                         sandbox_profile):
    """Without --resume it is the spec to submit: every chunk."""
    from edna_sampling.backends import COMPLETE, MISSING, PARTIAL

    monkeypatch.setattr("edna_sampling.backends.run_status",
                        lambda site, profile: {COMPLETE: [1, 2], PARTIAL: [3],
                                               MISSING: list(range(4, 101))})
    rc = main(["status", CAPE, "--profile", sandbox_profile, "--format", "chunks"])
    assert capsys.readouterr().out.strip() == "1-100"
    assert rc == 1


def test_status_format_chunks_with_resume_is_only_unfinished(monkeypatch, capsys,
                                                             sandbox_profile):
    """With --resume it is the spec to re-submit, and nothing else on stdout -
    a cluster user pipes this straight into their job array."""
    from edna_sampling.backends import COMPLETE, MISSING, PARTIAL

    monkeypatch.setattr("edna_sampling.backends.run_status",
                        lambda site, profile: {COMPLETE: [1, 2], PARTIAL: [3],
                                               MISSING: [7, 8, 9]})
    rc = main(["status", CAPE, "--profile", sandbox_profile,
               "--format", "chunks", "--resume"])
    assert capsys.readouterr().out.strip() == "3,7-9"
    assert rc == 1          # unfinished work -> nonzero, so `||` chains work


def test_status_resume_spec_is_shown_in_text_output(monkeypatch, capsys,
                                                    sandbox_profile):
    from edna_sampling.backends import COMPLETE, MISSING, PARTIAL

    monkeypatch.setattr("edna_sampling.backends.run_status",
                        lambda site, profile: {COMPLETE: [1], PARTIAL: [], MISSING: [2, 3]})
    main(["status", CAPE, "--profile", sandbox_profile, "--resume"])
    assert "resume: 2-3" in capsys.readouterr().out


def test_status_format_chunks_with_resume_is_silent_when_complete(monkeypatch, capsys,
                                                                  sandbox_profile):
    from edna_sampling.backends import COMPLETE, MISSING, PARTIAL

    monkeypatch.setattr("edna_sampling.backends.run_status",
                        lambda site, profile: {COMPLETE: list(range(1, 401)),
                                               PARTIAL: [], MISSING: []})
    rc = main(["status", CAPE, "--profile", sandbox_profile,
               "--format", "chunks", "--resume"])
    assert capsys.readouterr().out.strip() == ""
    assert rc == 0


def test_figures_from_a_condensed_file_needs_no_profile(tmp_path, capsys):
    """The condensed product is meant to travel to a laptop with no hindcast and
    no output_root. Requiring a machine profile to read it would defeat that."""
    missing = tmp_path / "absent.nc"
    assert main(["figures", CAPE, "--condensed", str(missing)]) == 2
    err = capsys.readouterr().err
    assert "no condensed product" in err
    assert "machine profile" not in err, "should not ask for a profile it does not need"


def test_figures_without_condensed_still_needs_a_profile(monkeypatch, capsys):
    """Without an explicit file the run's output location must be resolved."""
    monkeypatch.delenv("EDNA_PROFILE", raising=False)
    assert main(["figures", CAPE]) == 2
    assert "no machine profile" in capsys.readouterr().err


def test_info_reports_the_derived_particle_buffer(capsys):
    """The buffer is computed from the decay physics rather than configured, so
    it should be visible rather than silently applied."""
    assert main(["info", CAPE]) == 0
    out = capsys.readouterr().out
    assert "alive per source at the end of the run" in out
    assert "(derived)" in out
