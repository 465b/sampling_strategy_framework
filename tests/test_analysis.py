"""The detectability / footprint / coverage layer, on arrays small enough to check by hand."""

import numpy as np
import pytest

from edna_sampling.analysis import (coverage_curve, detectability_map, detectable,
                                    footprint_area_km2, representative_source,
                                    sources_detected_by)


def test_detection_threshold_is_concentration_times_filtered_volume():
    conc = np.array([[[[1.0, 2.0]]]])          # (t=1, rel=1, y=1, x=2)
    # filtered_volume 10 -> 10 and 20 copies; threshold 15 admits only the second
    det = detectable(conc, filtered_volume=10.0, threshold_copies=15.0)
    assert det.tolist() == [[[[False, True]]]]


def test_the_threshold_is_strict():
    """Exactly at threshold is not a detection - matching the notebook."""
    conc = np.array([[[[1.5]]]])
    assert not detectable(conc, 10.0, 15.0).any()
    assert detectable(conc, 10.0, 14.999).all()


def test_detectability_map_counts_sources_per_cell():
    det = np.zeros((1, 3, 2, 2), dtype=bool)
    det[0, 0, 0, 0] = True
    det[0, 1, 0, 0] = True
    det[0, 2, 1, 1] = True
    hot = detectability_map(det)
    assert hot.shape == (1, 2, 2)
    assert hot[0, 0, 0] == 2
    assert hot[0, 1, 1] == 1
    assert hot[0, 0, 1] == 0


def test_footprint_area_weights_each_cell_by_its_own_area():
    det = np.zeros((1, 1, 1, 2), dtype=bool)
    det[0, 0, 0, :] = True
    cell_area = np.array([[1e6, 3e6]])          # 1 and 3 km^2
    assert footprint_area_km2(det, cell_area)[0, 0] == pytest.approx(4.0)


def test_the_unweighted_form_agrees_only_where_cells_are_identical():
    """`weighted=False` multiplies the cell *count* by the mean cell area. It is
    kept for the archived notebook, which is why it is still reachable - but the
    weighted form is the default because the two part company as soon as cells
    differ in size, and the near-shore cells this analysis cares about do."""
    rng = np.random.default_rng(0)
    uniform = np.full((5, 5), 2e6)
    det = rng.random((3, 4, 5, 5)) < 0.3
    assert np.allclose(footprint_area_km2(det, uniform, weighted=True),
                       footprint_area_km2(det, uniform, weighted=False))

    det = np.zeros((1, 1, 1, 2), dtype=bool)
    det[0, 0, 0, 0] = True                      # only the 1 km^2 cell is detectable
    ragged = np.array([[1e6, 3e6]])
    assert footprint_area_km2(det, ragged, weighted=True)[0, 0] == pytest.approx(1.0)
    assert footprint_area_km2(det, ragged, weighted=False)[0, 0] == pytest.approx(2.0)


def test_representative_source_is_the_median_not_the_biggest():
    footprint = np.array([[1.0, 50.0, 5.0]])     # totals 1, 50, 5; median 5
    assert representative_source(footprint) == 2


def test_sources_detected_by_a_station_set():
    det = np.zeros((3, 2, 2), dtype=bool)
    det[0, 0, 0] = True
    det[1, 1, 1] = True
    det[2, 0, 0] = True
    assert sources_detected_by(det, [(0, 0)]) == {0, 2}
    assert sources_detected_by(det, [(1, 1)]) == {1}
    assert sources_detected_by(det, [(0, 0), (1, 1)]) == {0, 1, 2}
    assert sources_detected_by(det, []) == set()


def test_coverage_curve_is_nondecreasing_and_ends_at_full_coverage():
    rng = np.random.default_rng(1)
    det = rng.random((6, 4, 4)) < 0.4
    curve = coverage_curve(det, k_max=8)
    assert curve.shape == (8,)
    assert np.all(np.diff(curve) >= 0), curve
    assert curve[-1] <= det.shape[0]


def test_coverage_curve_saturates_when_one_station_sees_everything():
    det = np.zeros((4, 3, 3), dtype=bool)
    det[:, 1, 1] = True
    curve = coverage_curve(det, k_max=3)
    assert curve.tolist() == [4, 4, 4]
