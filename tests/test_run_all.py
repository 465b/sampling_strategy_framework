"""`run_all` - submit + condense + figures as one act.

The stages themselves are tested elsewhere; what matters here is the wiring:
that they run in order, that the artifacts come back, and above all that a
failed chunk stops the pipeline instead of quietly producing a condensed
product with fewer sources in it than the config declares.
"""

from pathlib import Path

import pytest
import yaml

from edna_sampling.cli import main
from edna_sampling.config import ConfigError, MachineProfile, SiteConfig
from edna_sampling.pipeline import run_all

from test_cli import CAPE, cape_sandbox, sandbox_profile      # noqa: F401


class _Result:
    def __init__(self, failed=()):
        self.launched = 0
        self.failed = list(failed)


@pytest.fixture
def stub_stages(monkeypatch, tmp_path):
    """Replace the three heavy stages, recording the order they ran in."""
    calls = []
    condensed = tmp_path / "out" / "cond.nc"
    condensed.parent.mkdir(parents=True, exist_ok=True)
    condensed.write_bytes(b"x" * 2048)

    import edna_sampling.backends as backends
    import edna_sampling.condense as condense_mod
    import edna_sampling.figures as figures_mod
    import edna_sampling.manifest as manifest

    def fake_submit(site, profile, chunks, **kw):
        calls.append(("submit", list(chunks)))
        return _Result(failed=state["failed"])

    def fake_condense(site, profile, *, out=None, **kw):
        calls.append(("condense", out))
        if out is None:
            return condensed
        Path(out).write_bytes(b"x" * 2048)      # run_all reports its size
        return Path(out)

    def fake_figures(ds, results, out_dir, **kw):
        calls.append(("figures", Path(out_dir)))
        return [Path(out_dir) / "map.png"]

    state = {"failed": []}
    monkeypatch.setattr(backends, "submit_local", fake_submit)
    monkeypatch.setattr(backends, "incomplete_chunks", lambda s, p: state.get("incomplete", []))
    monkeypatch.setattr(condense_mod, "condense", fake_condense)
    monkeypatch.setattr(condense_mod, "open_condensed", lambda p: _FakeDS())
    monkeypatch.setattr(figures_mod, "write_figures", fake_figures)
    monkeypatch.setattr(figures_mod, "site_results", lambda ds, **kw: {
        "sources_covered": 3, "n_sources": 3, "coverage_fraction": 1.0,
        "useful_stations": 2, "time_index": 0})
    monkeypatch.setattr(manifest, "check_manifest", lambda *a, **k: None)
    monkeypatch.setattr(manifest, "write_manifest", lambda *a, **k: tmp_path / "manifest.json")
    return calls, state, condensed


class _FakeDS:
    def close(self):
        pass


def _load(cape_sandbox, sandbox_profile):      # noqa: F811
    return SiteConfig.load(cape_sandbox), MachineProfile.load(Path(sandbox_profile))


def test_the_three_stages_run_in_order(stub_stages, cape_sandbox, sandbox_profile):  # noqa: F811
    calls, _, condensed = stub_stages
    site, profile = _load(cape_sandbox, sandbox_profile)

    out = run_all(site, profile, echo=lambda *a: None)

    assert [c[0] for c in calls] == ["submit", "condense", "figures"]
    assert calls[0][1] == list(range(1, site.n_chunks + 1))
    assert out["condensed"] == condensed
    assert out["figures"] == [condensed.parent / "figures" / "map.png"]
    assert out["results"]["sources_covered"] == 3


def test_a_failed_chunk_stops_before_condensing(stub_stages, cape_sandbox,  # noqa: F811
                                                sandbox_profile):
    """The failure is worth more than the artifact.

    A condensed product built from a run with failed chunks opens perfectly
    happily; it just holds fewer sources than the config declares, which is
    exactly the kind of thing nobody notices until the figures are wrong.
    """
    calls, state, _ = stub_stages
    state["failed"] = [7, 12]
    site, profile = _load(cape_sandbox, sandbox_profile)

    with pytest.raises(ConfigError, match="2 chunk"):
        run_all(site, profile, echo=lambda *a: None)

    assert [c[0] for c in calls] == ["submit"]


def test_resume_skips_modelling_but_still_condenses(stub_stages, cape_sandbox,  # noqa: F811
                                                    sandbox_profile):
    """Re-running a finished site rebuilds the analysis without redoing the model."""
    calls, state, _ = stub_stages
    state["incomplete"] = []
    site, profile = _load(cape_sandbox, sandbox_profile)

    run_all(site, profile, resume=True, echo=lambda *a: None)

    assert [c[0] for c in calls] == ["condense", "figures"]


def test_an_explicit_chunk_selection_is_passed_through(stub_stages, cape_sandbox,  # noqa: F811
                                                       sandbox_profile):
    calls, _, _ = stub_stages
    site, profile = _load(cape_sandbox, sandbox_profile)

    run_all(site, profile, chunks=[2, 5, 9], echo=lambda *a: None)

    assert calls[0] == ("submit", [2, 5, 9])


def test_figures_land_beside_the_condensed_product(stub_stages, cape_sandbox,  # noqa: F811
                                                   sandbox_profile, tmp_path):
    """So that `--condensed-out elsewhere/` carries its figures with it."""
    calls, _, _ = stub_stages
    site, profile = _load(cape_sandbox, sandbox_profile)
    elsewhere = tmp_path / "elsewhere" / "c.nc"
    elsewhere.parent.mkdir(parents=True)

    run_all(site, profile, condensed_out=elsewhere, echo=lambda *a: None)

    assert calls[2] == ("figures", elsewhere.parent / "figures")


def test_missing_release_points_names_the_command_that_makes_them(
        stub_stages, cape_sandbox, sandbox_profile):      # noqa: F811
    """Generating them here would let a config typo move every source unseen."""
    site, profile = _load(cape_sandbox, sandbox_profile)
    from edna_sampling.pipeline import release_points_path
    release_points_path(site).unlink()

    with pytest.raises(ConfigError, match="edna points"):
        run_all(site, profile, echo=lambda *a: None)


def test_cli_all_wires_its_arguments_through(stub_stages, monkeypatch, cape_sandbox,  # noqa: F811
                                             sandbox_profile, tmp_path):
    seen = {}
    import edna_sampling.pipeline as pipeline
    monkeypatch.setattr(pipeline, "run_all", lambda site, profile, **kw: seen.update(kw))

    assert main(["all", cape_sandbox, "--profile", sandbox_profile,
                 "--chunks", "2-4", "--resume", "--n-parallel", "3",
                 "--partial-cells", "drop", "--subsample", "7",
                 "--time-index", "5",
                 "--figures-out", str(tmp_path / "f")]) == 0

    assert seen["chunks"] == [2, 3, 4]
    assert seen["resume"] is True
    assert seen["n_parallel"] == 3
    assert seen["policy"] == "drop"
    assert seen["subsample"] == 7
    assert seen["time_index"] == 5
    assert seen["figures_out"] == tmp_path / "f"


def test_all_is_listed_in_the_cli_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "all" in capsys.readouterr().out
