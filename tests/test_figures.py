"""The figures layer, on a synthetic condensed product.

Builds a dataset in memory rather than running the model, so the derived numbers
are checkable by hand and the test costs milliseconds.
"""

import json

import numpy as np
import pytest

from edna_sampling.figures import site_results, write_figures


@pytest.fixture
def fake_condensed():
    """Three sources on a 4x4 grid. Source 0 and 1 are detectable from cell (0,0),
    source 2 only from (3,3), so one station covers two and two cover all three."""
    import xarray as xr

    n_time, n_rel, n_y, n_x = 2, 3, 4, 4
    conc = np.zeros((n_time, n_rel, n_y, n_x), dtype=np.float32)
    conc[:, 0, 0, 0] = 10.0
    conc[:, 1, 0, 0] = 10.0
    conc[:, 2, 3, 3] = 10.0
    conc[1, 0, 1, 1] = 10.0            # t=1 has a larger footprint, so it is chosen
    return xr.Dataset(
        {"conc_vert_avg": (("time", "release_group", "y", "x"), conc),
         "cell_area": (("y", "x"), np.full((n_y, n_x), 1e6, dtype=np.float32))},
        coords={"time": np.arange(n_time), "release_group": ["a", "b", "c"],
                "x": np.array([0.0, 1.0, 2.0, 3.0]), "y": np.array([10.0, 11.0, 12.0, 13.0])},
        attrs={"run_name": "test_run", "format_version": 1},
    )


def _results(ds, **kw):
    kw.setdefault("filtered_volume", 1.0)
    kw.setdefault("threshold_copies", 1.0)
    kw.setdefault("n_stations", 2)
    kw.setdefault("k_max", 3)
    return site_results(ds, **kw)


def test_two_stations_cover_all_three_sources(fake_condensed):
    r = _results(fake_condensed)
    assert r["n_sources"] == 3
    assert r["sources_covered"] == 3
    assert r["coverage_fraction"] == pytest.approx(1.0)


def test_one_station_covers_only_the_pair(fake_condensed):
    r = _results(fake_condensed, n_stations=1)
    assert r["sources_covered"] == 2
    assert r["stations_grid"] == [(0, 0)]


def test_stations_are_reported_in_lon_lat_not_grid_indices(fake_condensed):
    """A station is only useful if you can go there."""
    r = _results(fake_condensed, n_stations=1)
    assert r["stations_lonlat"] == [(0.0, 10.0)]        # x[0], y[0]


def test_the_chosen_time_is_the_largest_footprint(fake_condensed):
    """t=1 has an extra detectable cell, so it is the favourable moment."""
    assert _results(fake_condensed)["time_index"] == 1


def test_an_explicit_time_index_is_respected(fake_condensed):
    assert _results(fake_condensed, time_index=0)["time_index"] == 0


def test_coverage_curve_length_and_saturation(fake_condensed):
    r = _results(fake_condensed, k_max=3)
    assert r["coverage_curve"] == [2, 3, 3]


def test_threshold_below_which_nothing_is_detectable(fake_condensed):
    r = _results(fake_condensed, threshold_copies=1e9)
    assert r["sources_covered"] == 0
    assert r["footprint_km2_mean"] == 0.0


def test_write_figures_produces_files_and_a_readable_summary(fake_condensed, tmp_path):
    r = _results(fake_condensed)
    written = write_figures(fake_condensed, r, tmp_path)
    names = {p.name for p in written}
    assert names == {"detectability_map.png", "footprint_timeseries.png",
                     "coverage_curve.png", "summary.json"}
    assert all(p.stat().st_size > 0 for p in written)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["sources_covered"] == 3
    assert summary["source"]["run_name"] == "test_run"
    assert "_det" not in summary, "internal arrays must not leak into the summary"


def test_summary_is_json_serialisable_without_numpy_leakage(fake_condensed, tmp_path):
    """The summary is what two runs get compared on, so it has to round-trip."""
    r = _results(fake_condensed)
    write_figures(fake_condensed, r, tmp_path)
    text = (tmp_path / "summary.json").read_text()
    assert "array(" not in text and "np." not in text
    json.loads(text)


def test_saturated_extra_stations_are_reported_as_adding_nothing(fake_condensed):
    """Optimisers always return exactly k. Beyond saturation the surplus lands on
    an arbitrary cell - often grid corner (0,0) - and must not be presented as a
    place worth sampling."""
    r = _results(fake_condensed, n_stations=3)     # 2 suffice for 3 sources
    assert r["sources_covered"] == 3
    assert r["useful_stations"] == 2
    assert len(r["stations_grid"]) == 3            # contract preserved
    assert r["station_gains"][-1] == 0
    assert sum(r["station_gains"]) == 3


def test_every_station_counts_when_none_are_wasted(fake_condensed):
    r = _results(fake_condensed, n_stations=2)
    assert r["useful_stations"] == 2
    assert all(g > 0 for g in r["station_gains"])


def test_the_map_marks_only_stations_that_add_coverage(fake_condensed, tmp_path,
                                                       monkeypatch):
    """A marker on a cell where nothing is detectable is an instruction to sail
    somewhere pointless."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = []
    real_plot = plt.Axes.plot

    def spy(self, *a, **kw):
        if len(a) >= 3 and a[2] == "r*":
            plotted.append((list(np.atleast_1d(a[0])), kw.get("label", "")))
        return real_plot(self, *a, **kw)

    monkeypatch.setattr(plt.Axes, "plot", spy)
    r = _results(fake_condensed, n_stations=3)
    write_figures(fake_condensed, r, tmp_path)

    assert len(plotted) == 1
    xs, label = plotted[0]
    assert len(xs) == 2, "only the two useful stations should be marked"
    assert "add nothing" in label
