"""One site's figures, and the derived numbers behind them.

Reads a condensed product (`edna condense`) and nothing else - no oceantracker, no
hindcast, no chunk directories - so this runs on a laptop.

One site at a time. A multi-site comparison like the paper's is written by calling
`site_results` twice and plotting both, which is the caller's business rather than
something this module models.

`summary.json` is the point of the whole exercise as much as the figures are: it is
the derived numbers - chosen stations, sources covered, footprint areas - whose
survival under the 2D switch is the open question in
docs/2d-stats-assessment.md section 7. Two runs are compared by diffing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edna_sampling.analysis import (coverage_curve, detectability_map, detectable,
                                    footprint_area_km2, representative_source,
                                    sources_detected_by)
from edna_sampling.detection import optimize_stations


def site_results(ds, *, filtered_volume, threshold_copies, n_stations,
                 optimizer="greedy", random_seed=0, k_max=10, time_index=None,
                 ):
    """Everything one site's figures need, computed once.

    `ds` is an open condensed dataset. `time_index` selects the snapshot the
    station choice is made on; the default is the time with the largest total
    detectable area, which is the most favourable moment to sample and so the one
    the headline station count refers to.
    """
    conc = ds["conc_vert_avg"].values
    cell_area = ds["cell_area"].values
    det = detectable(conc, filtered_volume, threshold_copies)

    footprint = footprint_area_km2(det, cell_area)
    if time_index is None:
        time_index = int(np.argmax(footprint.mean(axis=1)))

    stations, covered = optimize_stations(
        det[time_index], int(n_stations), method=optimizer, random_seed=random_seed)

    # How many sources each station adds over those before it. The optimisers
    # always return exactly k stations, even once no station can add anything -
    # the surplus then lands on whatever cell argmax happens to hit, typically
    # grid corner (0, 0). Reporting those as recommended sampling locations would
    # send someone to a cell where nothing is detectable, so the gains are
    # published alongside and `useful_stations` counts the ones that earn a visit.
    gains, running = [], set()
    for iy, ix in stations:
        adds = sources_detected_by(det[time_index], [(iy, ix)]) - running
        gains.append(len(adds))
        running |= adds
    n_useful = sum(1 for g in gains if g > 0)
    curve = coverage_curve(det[time_index], k_max,
                           method=optimizer, random_seed=random_seed)

    n_sources = det.shape[1]
    xs, ys = ds["x"].values, ds["y"].values
    return {
        "n_sources": n_sources,
        "n_times": det.shape[0],
        "time_index": time_index,
        "stations_grid": [(int(iy), int(ix)) for iy, ix in stations],
        "stations_lonlat": [(float(xs[ix]), float(ys[iy])) for iy, ix in stations],
        "station_gains": gains,
        "useful_stations": n_useful,
        "sources_covered": len(covered),
        "coverage_fraction": len(covered) / n_sources if n_sources else 0.0,
        "coverage_curve": curve.tolist(),
        "footprint_km2_mean": float(footprint[time_index].mean()),
        "footprint_km2_median": float(np.median(footprint[time_index])),
        "representative_source": representative_source(footprint),
        "_det": det,
        "_footprint": footprint,
        "_hot": detectability_map(det),
    }


def _public(results):
    return {k: v for k, v in results.items() if not k.startswith("_")}


def write_figures(ds, results, out_dir: Path, *, dpi=150) -> list[Path]:
    """Render the single-site figures. Returns what was written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    t = results["time_index"]
    xs, ys = ds["x"].values, ds["y"].values

    # 1. how many sources each cell can detect, at the chosen time
    fig, ax = plt.subplots(figsize=(6, 5))
    hot = results["_hot"][t]
    im = ax.pcolormesh(xs, ys, np.where(hot > 0, hot, np.nan), shading="nearest")
    fig.colorbar(im, ax=ax, label="sources detectable")
    # Only the stations that add coverage. The optimiser returns exactly k, and
    # any surplus beyond saturation sits on an arbitrary cell - drawing those
    # would put a marker where nothing is detectable.
    useful = [(lon, lat) for (lon, lat), gain
              in zip(results["stations_lonlat"], results["station_gains"]) if gain > 0]
    if useful:
        sx, sy = zip(*useful)
        n = len(useful)
        ax.plot(sx, sy, "r*", markersize=14, markeredgecolor="k",
                label=f"{n} station{'s' if n > 1 else ''}"
                      + (f" ({len(results['stations_lonlat']) - n} more add nothing)"
                         if n < len(results["stations_lonlat"]) else ""))
        ax.legend(loc="upper right")
    ax.set(xlabel="longitude", ylabel="latitude",
           title=f"{ds.attrs.get('run_name','')}  detectability at t={t}")
    p = out_dir / "detectability_map.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    # 2. footprint area over time: spread across sources, plus the median source
    fig, ax = plt.subplots(figsize=(6, 4))
    fp = results["_footprint"]
    hours = np.arange(fp.shape[0])
    ax.fill_between(hours, np.percentile(fp, 5, axis=1), np.percentile(fp, 95, axis=1),
                    alpha=0.25, label="5-95th percentile of sources")
    ax.plot(hours, np.median(fp, axis=1), lw=2, label="median source")
    ax.plot(hours, fp[:, results["representative_source"]], "--", lw=1,
            label=f"representative source #{results['representative_source']}")
    ax.axvline(t, color="k", ls=":", lw=1, label=f"chosen time t={t}")
    ax.set(xlabel="output time step", ylabel="detection footprint [km$^2$]",
           title="footprint area per source")
    ax.legend(fontsize=8)
    p = out_dir / "footprint_timeseries.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    # 3. diminishing returns from adding stations
    fig, ax = plt.subplots(figsize=(6, 4))
    curve = np.asarray(results["coverage_curve"])
    k = np.arange(1, len(curve) + 1)
    ax.plot(k, curve / results["n_sources"] * 100, "o-")
    ax.axvline(len(results["stations_grid"]), color="r", ls=":",
               label=f"{len(results['stations_grid'])} stations")
    ax.set(xlabel="number of stations", ylabel="sources detected [%]",
           title="coverage vs station count", ylim=(0, 100))
    ax.legend(fontsize=8)
    p = out_dir / "coverage_curve.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    summary = _public(results)
    summary["source"] = {k: (v.item() if hasattr(v, "item") else v)
                         for k, v in ds.attrs.items()}
    p = out_dir / "summary.json"
    p.write_text(json.dumps(summary, indent=2, default=str))
    written.append(p)
    return written
