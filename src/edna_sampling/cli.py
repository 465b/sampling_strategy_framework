"""Command line interface.

    edna info   <config> [--profile P]        summarise an experiment
    edna mesh   <config>  --profile P         derive/cache the hindcast's mesh
    edna points <config>  --profile P         generate/load the release points
    edna params <config>  --profile P --chunk N   print one chunk's parameters
    edna run    <config>  --profile P --chunk N   run one chunk
    edna submit <config>  --profile P             launch every chunk locally
    edna status <config>  --profile P             how far along a run is
    edna condense <config> --profile P            chunks -> one portable netCDF
    edna figures  <config> --profile P            figures + derived numbers
    edna all      <config> --profile P            submit + condense + figures

The stages are separate commands on purpose: `points` is seconds on a laptop and
wants eyeballing, `run` is one chunk. That boundary is where the desktop/HPC
round trip happens. `all` is for when it does not: one machine, one sitting,
everything downstream of the release points. It deliberately stops short of
`points`, which wants looking at before committing to a few hundred model runs.

`submit` only runs chunks locally. On a cluster, submit `edna run --chunk N`
yourself - one chunk is one independent process with no shared state, so it maps
onto a job array directly, and `edna status --format chunks --resume` gives you
the outstanding chunks as an array spec.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edna_sampling import __version__
from edna_sampling.config import ConfigError, MachineProfile, SiteConfig

__all__ = ["main"]


def _resolve_profile(value: str | None) -> MachineProfile:
    """Accept a profile name or a path."""
    if value is None:
        raise ConfigError(
            "no machine profile given. Pass --profile <name|path>.\n"
            "  A profile says where hindcasts and output live on this machine; "
            "start from profiles/template.yaml."
        )
    candidate = Path(value)
    if candidate.is_file():
        return MachineProfile.load(candidate)
    for root in (Path.cwd() / "profiles", Path(__file__).resolve().parents[1] / "profiles"):
        p = root / f"{value}.yaml"
        if p.is_file():
            return MachineProfile.load(p)
    known = sorted(p.stem for p in (Path.cwd() / "profiles").glob("*.yaml"))
    raise ConfigError(
        f"no machine profile {value!r}. Known here: {', '.join(known) or '(none)'}. "
        "Pass a path, or add profiles/<name>.yaml."
    )


def _cmd_info(args) -> int:
    site = SiteConfig.load(args.config)
    # resolve the profile before printing anything, so a bad --profile fails
    # fast instead of after a screenful of output
    profile = _resolve_profile(args.profile) if args.profile is not None else None

    print(f"experiment   {site.run_name}   (fingerprint {site.fingerprint()})")
    print(f"config       {site.source_path}")
    print(f"polygon      {site.domain}")
    hindcast_line = site.hindcast
    mesh_line = "(needs a profile to resolve)"
    if profile is not None:
        try:
            entry = profile.hindcast(site.hindcast)
            hindcast_line = f"{site.hindcast}  ->  {entry.input_dir}/{entry.file_mask}"
            mesh = profile.mesh_path(site.hindcast)
            state = "cached" if mesh.is_file() else "not derived yet - run `edna mesh`"
            mesh_line = f"{mesh}  [{'override' if entry.mesh else state}]"
        except ConfigError as exc:
            hindcast_line = f"{site.hindcast}  ->  {exc}"
            mesh_line = "-"
    print(f"hindcast     {hindcast_line}")
    print(f"mesh         {mesh_line}")
    print(f"sources      {site.release_points.n} points, depth "
          f"{site.release_points.depth_range[0]:g}-{site.release_points.depth_range[1]:g} m, "
          f"seed {site.release_points.seed}"
          + (f", min_flooded_fraction {site.release_points.min_flooded_fraction:g}"
             if site.release_points.min_flooded_fraction is not None else ""))
    pulse = site.source.particles_per_release(site.model.time_step)
    print(f"release      {pulse} particles every {site.model.time_step:g} s "
          f"({site.source.copies_per_particle:g} copies each)")
    when = (f"from {site.model.start}" if site.model.start
            else "from wherever the hindcast begins")
    print(f"duration     {site.model.duration_hours:g} h {when}, "
          f"eDNA half-life {site.model.edna_half_life_hours:g} h")
    derived = site.model.particle_buffer_per_release_group is None
    print(f"particles    ~{site.particles_alive_at_end:,.0f} alive per source at the end of the run; "
          f"buffer {site.particle_buffer_size:,}"
          f"{' (derived)' if derived else ' (set in the config)'}")
    print(f"chunks       {site.n_chunks} x {site.chunking.release_groups_per_chunk} release group(s)")
    for s in site.stats:
        detail = (f"{s.rows}x{s.cols}" + (f"x{s.layers}" if s.layers else "")
                  + f" @ {s.grid_center[0]:g},{s.grid_center[1]:g}")
        # Surface the vertical selection: for a 3D statistic the reference frame,
        # for a 2D one the tow window. The window decides what fraction of the
        # water column is sampled at all, so it should never be implicit.
        if s.vertical_range_measured_relative_to:
            mode = f" [{s.vertical_range_measured_relative_to}]"
        elif s.near_seasurface is not None:
            mode = f" [top {s.near_seasurface:g} m below surface]"
        elif s.near_seabed is not None:
            mode = f" [within {s.near_seabed:g} m of seabed]"
        else:
            mode = " [whole column]"
        chosen = " <- analysed" if (len(site.stats) > 1 and s.name == site.analysis.statistic) else ""
        print(f"stats        {s.name}  {s.kind}  {detail}{mode}{chosen}")

    a = site.analysis
    print(f"detection    >= {a.detection.threshold_copies:g} copies in "
          f"{site.filtered_volume:g} m3 filtered "
          f"({a.detection.individuals_per_source:g} individuals x "
          f"{site.source.copies_per_particle:g} copies x "
          f"{a.detection.sample_volume_m3:g} m3)")
    print(f"stations     {a.optimizer.n_stations} via {a.optimizer.kind} "
          f"(seed {a.optimizer.random_seed})")

    if profile is not None:
        from edna_sampling.params import root_output_dir
        from edna_sampling.pipeline import release_points_path
        cache = release_points_path(site)
        print(f"profile      {profile.name}")
        print(f"output       {root_output_dir(site, profile)}")
        print(f"points       {cache}  [{'cached' if cache.exists() else 'not generated yet'}]")
    return 0


def _cmd_mesh(args) -> int:
    from edna_sampling.mesh import ensure_mesh, load_mesh

    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    target = profile.mesh_path(site.hindcast)
    existed = target.is_file()
    path = ensure_mesh(profile, site.hindcast, force=args.force)
    if existed and not args.force:
        mesh = load_mesh(path)
        print(f"mesh for {site.hindcast!r} already cached: {path}")
        print(f"  {mesh.n_nodes:,} nodes, {mesh.n_triangles:,} triangles "
              f"(--force re-derives it)")
    else:
        print(f"mesh for {site.hindcast!r}: {path}")
    return 0


def _cmd_points(args) -> int:
    from edna_sampling.pipeline import (ensure_release_points, release_points_figure_path,
                                        release_points_path)
    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    points, generated = ensure_release_points(
        site, profile, force=args.force, make_figure=not args.no_figure)
    verb = "generated" if generated else "loaded cached"
    print(f"{verb} {len(points)} release points for {site.run_name}")
    print(f"  {release_points_path(site)}")
    if generated and not args.no_figure:
        print(f"  {release_points_figure_path(site, profile)}   <- check this before running")
    return 0


def _cmd_params(args) -> int:
    import numpy as np
    from edna_sampling.params import build_params
    from edna_sampling.pipeline import release_points_path

    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    cache = release_points_path(site)
    if not cache.exists():
        raise ConfigError(f"release points not generated yet. Run:\n  edna points {args.config}")
    points = np.loadtxt(cache, delimiter=",")
    print(json.dumps(build_params(site, profile, args.chunk, points),
                     indent=2, sort_keys=True, default=float))
    return 0


def _cmd_run(args) -> int:
    from edna_sampling.pipeline import run_chunk

    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    run_chunk(site, profile, args.chunk,
              ignore_manifest=args.ignore_manifest, dry_run=args.dry_run)
    return 0


def _parse_chunks(spec: str, n_chunks: int) -> list[int]:
    """Parse "1-10,15,20-30" into a sorted list of chunk numbers."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part.lstrip("-"):
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo), int(hi)
                if lo_i > hi_i:
                    raise ValueError
                out.update(range(lo_i, hi_i + 1))
            else:
                out.add(int(part))
        except ValueError:
            raise ConfigError(
                f"--chunks: cannot parse {part!r}. Expected forms: 5, 1-10, 1-10,15,20-30"
            ) from None
    bad = sorted(n for n in out if not 1 <= n <= n_chunks)
    if bad:
        raise ConfigError(
            f"--chunks: {', '.join(map(str, bad[:5]))}"
            f"{'...' if len(bad) > 5 else ''} outside 1-{n_chunks}"
        )
    if not out:
        raise ConfigError("--chunks: selected nothing")
    return sorted(out)


def _cmd_figures(args) -> int:
    from edna_sampling.condense import condensed_path, open_condensed
    from edna_sampling.figures import site_results, write_figures

    site = SiteConfig.load(args.config)
    if args.condensed:
        # The point of the condensed product is that it travels: rsync it to a
        # laptop that has no hindcast and no output_root and make figures there.
        # Demanding a machine profile to read a file whose path you just gave
        # would defeat that, so --condensed makes the profile optional.
        source = Path(args.condensed)
        if not source.is_file():
            raise ConfigError(f"no condensed product at {source}")
    else:
        profile = _resolve_profile(args.profile)
        source = condensed_path(site, profile)
        if not source.is_file():
            raise ConfigError(
                f"no condensed product at {source}\n"
                f"  Run first:  edna condense {args.config} --profile {profile.name}")

    a = site.analysis
    ds = open_condensed(source)
    try:
        print(f"{site.run_name}: reading {source.name}")
        results = site_results(
            ds, filtered_volume=site.filtered_volume,
            threshold_copies=a.detection.threshold_copies,
            n_stations=a.optimizer.n_stations, optimizer=a.optimizer.kind,
            random_seed=a.optimizer.random_seed, k_max=a.sweeps.n_stations_max,
            time_index=args.time_index)
        out = Path(args.out) if args.out else source.parent / "figures"
        written = write_figures(ds, results, out)
    finally:
        ds.close()

    n_useful = results["useful_stations"]
    asked = len(results["stations_grid"])
    print(f"  {results['sources_covered']}/{results['n_sources']} sources "
          f"({results['coverage_fraction']:.1%}) from "
          f"{n_useful} station(s) at t={results['time_index']}")
    for (lon, lat), gain in zip(results["stations_lonlat"], results["station_gains"]):
        note = f"+{gain} sources" if gain else "adds nothing - coverage already saturated"
        print(f"    station  {lon:.5f}, {lat:.5f}   {note}")
    if n_useful < asked:
        print(f"  note: {asked - n_useful} of the {asked} requested stations add no "
              f"coverage; lower analysis.optimizer.n_stations")
    for p in written:
        print(f"  wrote {p}")
    return 0


def _cmd_condense(args) -> int:
    from edna_sampling.condense import condense

    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    spec = site.statistic()          # fails early if the config is ambiguous
    print(f"{site.run_name}: condensing {spec.name} ({spec.kind})")
    out = condense(site, profile, out=args.out, subsample=args.subsample,
                   policy=args.partial_cells)
    size_mb = out.stat().st_size / 1e6
    print(f"  wrote {out}  ({size_mb:.1f} MB)")
    return 0


def _cmd_status(args) -> int:
    from edna_sampling.backends import COMPLETE, MISSING, PARTIAL, run_status
    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    from edna_sampling.params import root_output_dir

    status = run_status(site, profile)

    unfinished = sorted(status[PARTIAL] + status[MISSING])

    if args.format == "chunks":
        # Machine-readable array spec, for feeding to a scheduler. Without
        # --resume it is the whole run (what to submit); with it, only the
        # unfinished chunks (what to re-submit) - the same selection
        # `edna submit --resume` makes for a local run.
        wanted = unfinished if args.resume else list(range(1, site.n_chunks + 1))
        if wanted:
            print(_compact(wanted))
        return 0 if not unfinished else 1

    total = site.n_chunks
    done = len(status[COMPLETE])
    print(f"{site.run_name}  ->  {root_output_dir(site, profile)}")
    print(f"  complete {done}/{total}"
          f"   partial {len(status[PARTIAL])}   missing {len(status[MISSING])}")
    if status[PARTIAL]:
        print(f"  partial chunks started but produced no complete statistics; "
              f"rerunning them is safe")
    from edna_sampling.manifest import read_manifest
    recorded = read_manifest(site, profile)
    if recorded is not None:
        drift = "" if recorded.get("fingerprint") == site.fingerprint() else \
                "   <- CONFIG HAS CHANGED since this run started"
        print(f"  started {recorded.get('created')} "
              f"fingerprint {recorded.get('fingerprint')}{drift}")

    if args.resume:
        # The spec to hand a scheduler, alongside the human-readable summary.
        print(f"  resume: {_compact(unfinished) if unfinished else '(nothing outstanding)'}")
    if args.list:
        for state in (PARTIAL, MISSING):
            if status[state]:
                print(f"  {state}: {_compact(status[state])}")
    return 0 if done == total else 1


def _compact(nums: list[int]) -> str:
    """1,2,3,7 -> 1-3,7"""
    if not nums:
        return ""
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def _cmd_submit(args) -> int:
    from edna_sampling.backends import incomplete_chunks, submit_local
    from edna_sampling.pipeline import release_points_path

    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    site.validate_inputs(profile)

    cache = release_points_path(site)
    if not cache.exists():
        raise ConfigError(
            f"release points not generated yet: {cache}\n"
            f"  Run once before submitting:  edna points {args.config} --profile {profile.name}"
        )

    if args.chunks:
        chunks = _parse_chunks(args.chunks, site.n_chunks)
    elif args.resume:
        chunks = incomplete_chunks(site, profile)
        if not chunks:
            print(f"{site.run_name}: all {site.n_chunks} chunks already complete")
            return 0
    else:
        chunks = list(range(1, site.n_chunks + 1))

    if not args.dry_run:
        from edna_sampling.manifest import check_manifest, write_manifest
        if not args.ignore_manifest:
            check_manifest(site, profile, cache)
        manifest_file = write_manifest(site, profile, cache)
        print(f"  provenance: {manifest_file}")

    print(f"{site.run_name}: {len(chunks)} chunk(s) locally [{_compact(chunks)}]")
    result = submit_local(site, profile, chunks,
                          n_parallel=args.n_parallel, dry_run=args.dry_run,
                          verbose=args.verbose)
    if result.failed:
        print(f"failed chunks: {_compact(result.failed)}", file=sys.stderr)
        return 1
    return 0


def _cmd_all(args) -> int:
    from edna_sampling.pipeline import run_all

    site = SiteConfig.load(args.config)
    profile = _resolve_profile(args.profile)
    chunks = _parse_chunks(args.chunks, site.n_chunks) if args.chunks else None
    run_all(site, profile,
            chunks=chunks, resume=args.resume, n_parallel=args.n_parallel,
            verbose=args.verbose, ignore_manifest=args.ignore_manifest,
            subsample=args.subsample, policy=args.partial_cells,
            condensed_out=args.condensed_out, figures_out=args.figures_out,
            time_index=args.time_index)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edna", description="eDNA sampling-strategy pipeline")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("config", help="path to a site config YAML")
        p.add_argument("--profile", default=None,
                       help="machine profile name or path")
        return p

    p_info = add_common(sub.add_parser("info", help="summarise an experiment"))
    p_info.set_defaults(func=_cmd_info)

    p_mesh = add_common(sub.add_parser(
        "mesh", help="derive the hindcast's mesh and cache it on this machine"))
    p_mesh.add_argument("--force", action="store_true",
                        help="re-derive even if a cached mesh exists")
    p_mesh.set_defaults(func=_cmd_mesh)

    p_points = add_common(sub.add_parser("points", help="generate or load release points"))
    p_points.add_argument("--force", action="store_true",
                          help="regenerate even if a cache exists")
    p_points.add_argument("--no-figure", action="store_true",
                          help="skip the diagnostic figure")
    p_points.set_defaults(func=_cmd_points)

    p_params = add_common(sub.add_parser("params", help="print one chunk's OceanTracker params"))
    p_params.add_argument("--chunk", type=int, required=True, help="1-based chunk number")
    p_params.set_defaults(func=_cmd_params)

    p_run = add_common(sub.add_parser("run", help="run one chunk"))
    p_run.add_argument("--chunk", type=int, required=True, help="1-based chunk number")
    p_run.add_argument("--dry-run", action="store_true",
                       help="build parameters and create the output directory, but do not run")
    p_run.add_argument("--ignore-manifest", action="store_true",
                       help="run even though the config changed since the run started")
    p_run.set_defaults(func=_cmd_run)

    p_condense = add_common(sub.add_parser(
        "condense", help="stitch a run's chunks into one portable netCDF"))
    p_condense.add_argument("--out", type=Path, default=None,
                            help="write here instead of beside the run output")
    p_condense.add_argument("--subsample", type=int, default=5,
                            help="sub-points per stats cell for the effective-volume "
                                 "calculation (default 5)")
    p_condense.add_argument("--partial-cells", choices=("weight", "drop"), default="weight",
                            help="part-land cells: down-weight by wet fraction (default) "
                                 "or drop entirely, as a conservative sensitivity check")
    p_condense.set_defaults(func=_cmd_condense)

    p_figures = add_common(sub.add_parser(
        "figures", help="figures and derived numbers from a condensed product"))
    p_figures.add_argument("--condensed", type=Path, default=None,
                           help="read this file instead of the run's own")
    p_figures.add_argument("--out", type=Path, default=None,
                           help="write here instead of <run output>/figures/")
    p_figures.add_argument("--time-index", type=int, default=None,
                           help="snapshot to choose stations on (default: the time "
                                "with the largest mean footprint)")
    p_figures.set_defaults(func=_cmd_figures)

    p_status = add_common(sub.add_parser("status", help="how many chunks are finished"))
    p_status.add_argument("--format", choices=("text", "chunks"), default="text",
                          help="chunks: print an array spec (e.g. 1-400) and nothing "
                               "else, for feeding to your own scheduler")
    p_status.add_argument("--resume", action="store_true",
                          help="restrict to unfinished chunks - the same selection "
                               "`edna submit --resume` makes")
    p_status.add_argument("--list", action="store_true",
                          help="list the chunk numbers still to do")
    p_status.set_defaults(func=_cmd_status)

    p_submit = add_common(sub.add_parser("submit", help="launch a run's chunks"))
    p_submit.add_argument("--chunks", default=None,
                          help="explicit selection, e.g. 1-10,15,20-30 (default: all)")
    p_submit.add_argument("--resume", action="store_true",
                          help="only chunks that are missing or incomplete")
    p_submit.add_argument("--n-parallel", type=int, default=None,
                          help="local: override local.n_parallel_jobs")
    p_submit.add_argument("--dry-run", action="store_true",
                          help="print the commands; run nothing")
    p_submit.add_argument("--verbose", action="store_true",
                          help="stream each chunk's OceanTracker output to the "
                               "terminal. Off by default: it is written to "
                               "<output>/logs/chunk_NNN.log instead, since "
                               "n_parallel runs interleave into noise")
    p_submit.add_argument("--ignore-manifest", action="store_true",
                          help="submit even though the config changed since the run started")
    p_submit.set_defaults(func=_cmd_submit)

    p_all = add_common(sub.add_parser(
        "all", help="submit + condense + figures, in one go"))
    p_all.add_argument("--chunks", default=None,
                       help="explicit selection, e.g. 1-10,15,20-30 (default: all)")
    p_all.add_argument("--resume", action="store_true",
                       help="model only the chunks that are missing or incomplete")
    p_all.add_argument("--n-parallel", type=int, default=None,
                       help="override local.n_parallel_jobs")
    p_all.add_argument("--verbose", action="store_true",
                       help="stream each chunk's OceanTracker output to the terminal")
    p_all.add_argument("--ignore-manifest", action="store_true",
                       help="run even though the config changed since the run started")
    p_all.add_argument("--subsample", type=int, default=5,
                       help="sub-points per stats cell for the effective-volume "
                            "calculation (default 5)")
    p_all.add_argument("--partial-cells", choices=("weight", "drop"), default="weight",
                       help="part-land cells: down-weight by wet fraction (default) "
                            "or drop entirely")
    p_all.add_argument("--condensed-out", type=Path, default=None,
                       help="write the condensed product here instead of beside "
                            "the run output")
    p_all.add_argument("--figures-out", type=Path, default=None,
                       help="write figures here instead of <run output>/figures/")
    p_all.add_argument("--time-index", type=int, default=None,
                       help="snapshot to choose stations on (default: the time "
                            "with the largest mean footprint)")
    p_all.set_defaults(func=_cmd_all)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
