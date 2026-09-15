"""Detectability, footprints and station coverage, computed from a condensed product.

Pure array code, like `detection.py`: numpy only, no oceantracker, no config
objects. Functions take arrays and parameters so that a caller comparing two sites
composes them itself - one site at a time is the default path, and the multi-site
figures in the paper are one caller among several rather than the shape the API is
built around.

Lifted from archiv/2026_07_22_paper_figures.ipynb.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "detectable", "detectability_map", "footprint_area_km2",
    "representative_source", "coverage_curve", "sources_detected_by",
]


def detectable(conc_vert_avg, filtered_volume, threshold_copies):
    """Which (source, cell, time) combinations a sample would detect.

    A sample at a cell detects a source when the eDNA it filters carries at least
    `threshold_copies`:

        conc [copies/m^3] * filtered_volume [m^3] > threshold_copies

    `conc_vert_avg` is (time, release_group, y, x) as written by `edna condense`;
    the result has the same shape and is boolean.

    Strictly greater, not >=, matching the notebook this came from.
    """
    conc = np.asarray(conc_vert_avg)
    return conc * float(filtered_volume) > float(threshold_copies)


def detectability_map(det):
    """How many sources are detectable from each cell, per time: (time, y, x).

    This is the field the detectability figures shade.
    """
    return np.asarray(det).sum(axis=1)


def footprint_area_km2(det, cell_area, *, weighted=True):
    """Detection-footprint area per source per time, in km^2: (time, release_group).

    With `weighted=True` each detectable cell contributes its own area, which is
    what the condensed product's per-cell `cell_area` is for. `weighted=False`
    reproduces the notebook, which multiplied the cell *count* by the grid's mean
    cell area - fine on a grid whose cells are near-identical, and the two agree to
    that extent, but the weighted form needs no such assumption.
    """
    det = np.asarray(det)
    n_time, n_rel = det.shape[:2]
    if weighted:
        area = np.asarray(cell_area, dtype=float)
        flat = det.reshape(n_time, n_rel, -1)
        return flat @ area.ravel() / 1e6
    mean_area = float(np.nanmean(cell_area))
    return det.reshape(n_time, n_rel, -1).sum(axis=2) * mean_area / 1e6


def representative_source(footprint_km2):
    """Index of the source whose total footprint is closest to the median.

    The paper's "example source" panels use this rather than a hand-picked index,
    so the illustration is typical of the site instead of flattering to it.
    """
    total = np.asarray(footprint_km2).sum(axis=0)
    return int(np.argmin(np.abs(total - np.median(total))))


def sources_detected_by(det_at_time, stations):
    """Which sources a set of stations detects, at one time step.

    `det_at_time` is (release_group, y, x); `stations` is a sequence of (iy, ix)
    grid indices, the form the optimisers return.
    """
    det = np.asarray(det_at_time)
    if not len(stations):
        return set()
    rows = np.array([iy for iy, _ in stations])
    cols = np.array([ix for _, ix in stations])
    return {int(s) for s in np.where(det[:, rows, cols].any(axis=1))[0]}


def coverage_curve(det_at_time, k_max, *, method="greedy", random_seed=0):
    """Sources covered as a function of station count, for k = 1..k_max.

    Returns an integer array of length `k_max`. Each k is optimised independently,
    matching the notebook; for the greedy optimiser the result is nested anyway,
    but that is a property of greedy rather than something to rely on.
    """
    from edna_sampling.detection import optimize_stations

    det = np.asarray(det_at_time)
    out = np.zeros(int(k_max), dtype=int)
    for k in range(1, int(k_max) + 1):
        _, covered = optimize_stations(det, k, method=method, random_seed=random_seed)
        out[k - 1] = len(covered)
    return out
