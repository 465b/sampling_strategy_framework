"""Lloyd relaxation, on synthetic regions that need no hindcast or mesh.

The property that matters is that sources end up spread uniformly per unit
area of the admissible region. Nothing downstream checks this, and when it
broke it broke quietly: the depth distribution of the sources just shifted.
"""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from edna_sampling.release_points import (_far_field_frame, _lloyd_relax,
                                          _random_points_in_polygon)


def _dense_square(n_per_side=60):
    """A unit square whose boundary carries many vertices.

    Boundary sampling is the whole point: a coastline built as a union of mesh
    triangles has a vertex every few tens of metres, which is what used to
    distort the tessellation.
    """
    t = np.linspace(0, 1, n_per_side, endpoint=False)
    ones, zeros = np.ones_like(t), np.zeros_like(t)
    ring = np.concatenate([
        np.column_stack([t, zeros]), np.column_stack([ones, t]),
        np.column_stack([1 - t, ones]), np.column_stack([zeros, 1 - t]),
    ])
    return Polygon(ring)


def _inner_half_share(points):
    """Fraction of `points` inside the centred sub-square of half the area."""
    s = 1 / np.sqrt(2)
    lo, hi = (1 - s) / 2, (1 + s) / 2
    inside = ((points[:, 0] > lo) & (points[:, 0] < hi)
              & (points[:, 1] > lo) & (points[:, 1] < hi))
    return inside.mean()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_relaxation_does_not_push_points_away_from_the_boundary(seed):
    """Half the area must hold about half the points.

    The regression this guards: seeding the Voronoi diagram with the region's
    own boundary vertices makes them generators, so each one claims the ground
    around it and a nearby release point keeps only a sliver opening inward.
    Relaxation then walked points off the boundary - this sub-square held ~67%
    of the points instead of 50%, and at Cape Rodney the 0-10 m band lost three
    quarters of its sources.
    """
    region = _dense_square()
    rng = np.random.default_rng(seed)
    points = _random_points_in_polygon(region, 120, rng)
    points, _ = _lloyd_relax(points, region)

    assert _inner_half_share(points) == pytest.approx(0.5, abs=0.1)


def test_relaxation_spreads_a_clustered_start_over_the_whole_region():
    """It still does its job: points that start piled up end up spread out."""
    region = _dense_square()
    rng = np.random.default_rng(0)
    points = 0.45 + 0.1 * rng.random((80, 2))       # all inside the middle tenth
    assert _inner_half_share(points) == 1.0

    points, _ = _lloyd_relax(points, region)

    assert _inner_half_share(points) == pytest.approx(0.5, abs=0.15)
    assert points.min() >= 0.0 and points.max() <= 1.0


def test_far_field_frame_sits_well_outside_the_region():
    """A sentinel inside the region would compete for area, which is the bug."""
    region = _dense_square()
    frame = _far_field_frame(region)
    assert not any(region.buffer(0.5).contains(Point(*p)) for p in frame)
    minx, miny, maxx, maxy = region.bounds
    assert frame[:, 0].min() < minx and frame[:, 0].max() > maxx
    assert frame[:, 1].min() < miny and frame[:, 1].max() > maxy


def test_relaxation_reports_how_it_stopped():
    region = _dense_square()
    rng = np.random.default_rng(0)
    points = _random_points_in_polygon(region, 40, rng)

    _, info = _lloyd_relax(points, region, max_iter=5)

    assert info["stop"] == "max_iter"
    assert info["converged"] is False
    assert info["n_iter"] == 5
    assert len(info["cv_history"]) == 5
    assert info["final_cv"] == info["cv_history"][-1]


def test_a_reachable_cv_target_is_reported_as_converged():
    region = _dense_square()
    rng = np.random.default_rng(0)
    points = _random_points_in_polygon(region, 40, rng)

    _, info = _lloyd_relax(points, region, cv_target=10.0)

    assert info["stop"] == "cv_target"
    assert info["converged"] is True
    assert info["n_iter"] == 1


def test_relaxation_stops_early_once_the_cell_areas_stop_improving():
    """`cv_target` is unreachable on a ragged region; the plateau test ends it."""
    region = _dense_square()
    rng = np.random.default_rng(3)
    points = _random_points_in_polygon(region, 40, rng)

    _, info = _lloyd_relax(points, region, cv_target=0.0, max_iter=400,
                           plateau_tol=0.05, plateau_patience=5)

    assert info["stop"] == "plateau"
    assert info["converged"] is False
    assert info["n_iter"] < 400
