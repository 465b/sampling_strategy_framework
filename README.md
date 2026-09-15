# eDNA sampling-strategy framework

1) Where should you take environmental DNA to capture the largest ratio of the target area at a given time?
2) Which footprint i.e. area does a sample taken at a space and time represent?

This framework answers that by running [OceanTracker](https://oceantracker.github.io/oceantracker/)
particle-tracking experiments over a hydrodynamic hindcast: eDNA is released from
evenly-spaced source locations, transported and decayed, and counted onto a grid.
The resulting concentration fields drive a detectability analysis and optimization method that picks the best sampling stations.
Note, that this is not only valid to eDNA but any decaying tracer.

See [docs/architecture.md](docs/architecture.md) for the differen pieces are put together.

## Install

```bash
git clone <repo> && cd sampling_strategy_framework
conda env create -f environment.yml
conda activate edna-sampling
pip install -e .
```

## Two files describe a run

| | holds |
|---|---|
| **site config** `experiments/<site>/config.yml` | what the experiment *is*: geometry, sources, model settings, statistics |
| **machine profile** `profiles/<machine>.yaml` | where things live on *one computer*: hindcast locations, output root, parallelism |

A site config may not contain an absolute path.
It names a hindcast, and the profile resolves it. This is what lets
the same experiment run unchanged on different machines.

```yaml
# profiles/desktop.yaml
hindcasts:
  hauraki_gulf:
    input_dir: /hpcfreenas/hindcast/.../2018/01
    file_mask: "schout_*.nc"
```
```yaml
# experiments/mysite/config.yml
hindcast: hauraki_gulf              # name, not a path
domain: target_area.csv             # the polygon sources are spread over
```

## Examples, and where your own experiments go

`examples/` holds two maintained, deliberately minimal sites — a config and a
target-area polygon each, and nothing else. The release points are not committed:
they are a deterministic function of the config and the hindcast, so `edna points`
regenerates them byte for byte (5 s for cape_rodney, 43 s for auckland, which also
scans the hindcast for wet/dry state).

```
examples/
  cape_rodney/     2D statistic
  auckland/        2D statistic plus a 3D statistic
```

`experiments/` is git-ignored.

```bash
mkdir -p experiments/mysite
cp examples/cape_rodney/config.yml experiments/mysite/
$EDITOR experiments/mysite/config.yml     # name, version, polygon, statistics
```

## Setting up a new machine

```bash
cp profiles/template.yaml profiles/<profile>.yaml
$EDITOR profiles/<profile>.yaml     # a hindcast, an output root, a job count
edna info <config> --profile <profile>
```

`--profile` takes a name or a path.

## Running an experiment

```bash
# 1. look at what you are about to run
edna info <config> --profile <profile>

# 2. generate the source locations (seconds) and eyeball the diagnostic figure
#    derives the mesh first, if this machine has not got one yet
edna points <config> --profile <profile>

# 3. run the chunks. Chunk n always covers release points [(n-1)k, nk).
edna run <config> --profile <profile> --chunk 1
```

To inspect exactly what OceanTracker will be told, without running anything:

```bash
edna params <config> --profile <profile> --chunk 1
edna run    <config> --profile <profile> --chunk 1 --dry-run
```

### Many chunks at once

A production run is one job per chunk. The default is one source per chunk and typical source numbers are in  the hundreds.
`edna submit` runs them locally as a bounded pool of processes.

```bash
edna submit <config> --profile <profile>
edna status <config> --profile <profile> --list   # shows the status of chunks
edna submit <config> --profile <profile> --resume # only what is outstanding
```
or if you would like to run a individual chunk
```bash
edna run --chunk N
```

Always look before you leap — this prints the commands and runs nothing:

```bash
edna submit <config> --profile <profile> --dry-run
```

A chunk counts as complete only if it has `caseInfo.json` *and* a statistics
file per configured statistic, so a job that died partway is reported as
`partial` and picked up by `--resume`.

**Sizing.** Start with "coarse" prototype. Low number of sources (<100), coarse statistic grids (<100x100 cells) and use a high "copies_per_particle" values (>100)

## From model output to figures

A finished run is one directory per chunk, most of which the depth-averaged
analysis does not need. `condense` turns them into a single file smaller file that can between machines to e.g., run a large model on a cluster, continue the analysis on your local machine:

```bash
edna condense <config> --profile <profile>
```

It writes `<run>_condensed.nc` a vertical averaged (either full depth, or tow-window) concentration field, the effective volumes, the grid, and the run's provenance.

Then figures, from that file alone:

```bash
edna figures <config> --condensed <run>_condensed.nc
```

That produces a detectability map with the chosen stations marked, a footprint
time series, a coverage-vs-stations curve, and `summary.json` — the derived
numbers, which is what you diff to compare two runs.

Stations that add no coverage are reported as such rather than drawn on the map:
the optimiser always returns exactly the number you asked for, and past
saturation the surplus lands on an arbitrary cell.

There is an option to record 3D statistics as well to e.g. manually analyse horizontal tows.

## Driving it from Python

Every `edna` command is a thin wrapper over a function, 
so a python script can run the same pipeline  — useful for parameter sweeps, or for running from a notebook:

```python
from edna_sampling import (
    SiteConfig, MachineProfile,
    ensure_release_points, run_chunk,
    open_condensed, site_results, write_figures,
)
from edna_sampling.condense import condense   # not re-exported; see the note below

site = SiteConfig.load("examples/cape_rodney/config.yml")
profile = MachineProfile.load("profiles/desktop.yaml")

# edna points - derives the mesh first if this machine has not got one yet
points, generated = ensure_release_points(site, profile)

# edna run - one chunk per call. In production these are separate processes,
# one per job-array index; this loop is the single-machine equivalent.
for chunk in range(1, site.n_chunks + 1):
    run_chunk(site, profile, chunk)

# edna condense
product = condense(site, profile)

# edna figures
analysis = site.analysis
ds = open_condensed(product)
try:
    results = site_results(
        ds,
        filtered_volume=site.filtered_volume,
        threshold_copies=analysis.detection.threshold_copies,
        n_stations=analysis.optimizer.n_stations,
        optimizer=analysis.optimizer.kind,
        random_seed=analysis.optimizer.random_seed,
        k_max=analysis.sweeps.n_stations_max,
    )
    write_figures(ds, results, product.parent / "figures")
finally:
    ds.close()

print(f"{results['sources_covered']} of {results['n_sources']} sources covered")
```

## Analysis parameters

An optional `analysis:` block in the site config defines the generic output.

```yaml
analysis:
  statistic: around_cape_rodney_2D   # required only if the config defines several
  detection:
    threshold_copies: 10             # K0*: copies needed in the sample elute
    sample_volume_m3: 1.0            # physical volume one sample represents
    individuals_per_source: 100      # colony size per potential source
  optimizer:
    kind: greedy                     # greedy | greedy_reference | simulated_annealing | genetic
    n_stations: 3
    random_seed: 0
  sweeps:
    coverage_levels: [0.125, 0.25, 0.5, 0.75]
    n_stations_max: 10
    volume_exponents_m3: [-3, 2, 0.25]   # log10 m^3: 1e-3 .. 1e2 in quarter decades
    n_naive_draws: 100
```

## The mesh is derived, not supplied

To generate the mesh manually - it should already been
```bash
edna mesh <config> --profile <profile>
```
It reads the hindcast through OceanTracker and caches the result at
`<scratch or output_root>/meshes/<hindcast>.nc`.
`edna points` calls it for you when the cache is cold, so
in practice you never run it directly — it exists so you can check the step, and
force a rebuild with `--force`.

This is what makes hindcast formats interchangeable. Nothing in this package
parses a hindcast itself: OceanTracker's reader detects the format from the
files, so SCHISM, ROMS, FVCOM and DELFT3D-FM all work without a code change.

If you already have a mesh you would rather use, name it in the profile and it
is used verbatim — an OceanTracker `grid000.nc` or a raw SCHISM file are both
recognised:

```yaml
hindcasts:
  hauraki_gulf:
    input_dir: /path/to/hindcast
    mesh: /path/to/grid.nc          # optional override
```

**Two things are still SCHISM-specific**, and say so rather than failing
obscurely: `release_points.min_flooded_fraction` reads SCHISM's `wetdry_node`
flag, and hindcast files are ordered by a trailing number in their name.


## Data

Hindcasts for the Hauraki Gulf come from the
[SCHISM hindcast catalogue](https://hauraki-gulf-schism-hindcast.cloud.edu.au/thredds/catalog/outputs/outputs/catalog.html).


## A full run, start to finish

The shipped Cape Rodney example, on a machine whose profile is `desktop`. Paths
are shortened to `...`; the `edna info` block is real output, and the later
counts and coordinates are representative rather than from one particular run.

**1. Check what you are about to run.** Nothing is computed; every derived value
is shown so you can disagree with it before spending hours.

```console
$ edna info examples/cape_rodney/config.yml --profile desktop
experiment   v00_cape_rodney   (fingerprint a58c2956a34c)
polygon      .../examples/cape_rodney/target_area.csv
hindcast     hauraki_gulf  ->  /.../2018/01/schout_*.nc
mesh         /.../meshes/hauraki_gulf.nc  [not derived yet - run `edna mesh`]
sources      100 points, depth 0-30 m, seed 42, min_flooded_fraction 1
release      333 particles every 120 s (1000 copies each)
duration     48 h from 2018-01-01T00:00:00, eDNA half-life 6 h
particles    ~86,137 alive per source at the end of the run; buffer 103,365 (derived)
chunks       100 x 1 release group(s)
stats        around_cape_rodney_2D  gridded_2d  250x250 @ 174.822,-36.278 [whole column]
detection    >= 10 copies in 100000 m3 filtered (100 individuals x 1000 copies x 1 m3)
stations     3 via greedy (seed 0)
output       /.../v00_cape_rodney
points       .../v00_cape_rodney_lloyd_points.csv  [not generated yet]
```

**2. Generate the source locations**, then look at the figure before committing to
a few hundred model runs. This derives the mesh first if the machine has not got
one yet. Doing it once up front is deliberate: every chunk reads this same cache,
so they cannot disagree about where the sources are.

```console
$ edna points examples/cape_rodney/config.yml --profile desktop
generated 100 release points for v00_cape_rodney
  .../examples/cape_rodney/v00_cape_rodney_lloyd_points.csv
  /.../v00_cape_rodney_release_points.png   <- check this before running
```

**3. Look before you leap.** `--dry-run` prints the commands and runs nothing;
`edna params` prints exactly what OceanTracker will be handed.

```console
$ edna submit examples/cape_rodney/config.yml --profile desktop --dry-run
$ edna params examples/cape_rodney/config.yml --profile desktop --chunk 1 | head
```

**4. Run the chunks.** One source per chunk, a bounded pool locally.

```console
$ edna submit examples/cape_rodney/config.yml --profile desktop
  provenance: /.../v00_cape_rodney/manifest.json
v00_cape_rodney: 100 chunk(s) locally [1-100]
chunk 1: started (pid 930633), 1/12 running, 99 queued
...
chunk 100: done
```

Each chunk's OceanTracker output goes to `<output>/logs/chunk_NNN.log`, not to the
terminal — a dozen runs interleaved is unreadable, and every line is on disk
anyway, in that log and in the chunk's own `run_log.txt`. A chunk that fails
prints the tail of its log so you do not have to go looking. `--verbose` streams
everything instead, which is what you want when debugging one chunk.

Since it is no longer noisy, backgrounding it works as you would expect:

```console
$ nohup edna submit examples/cape_rodney/config.yml --profile desktop > submit.log 2>&1 &
```

Or use 
```
`edna run --chunk N`
```
to run an individual chunk.

**5. Check it finished**, and resubmit only what did not.

```console
$ edna status examples/cape_rodney/config.yml --profile desktop
v00_cape_rodney  ->  /.../v00_cape_rodney
  complete 100/100   partial 0   missing 0
  started 2026-09-15T09:12:44+12:00 fingerprint a58c2956a34c

$ edna submit examples/cape_rodney/config.yml --profile desktop --resume
v00_cape_rodney: all 100 chunks already complete
```

`--format chunks --resume` prints the outstanding chunks as an array spec
(`17,31,204-206`) and exits non-zero while any remain, which is what a resubmit
loop wants.

**6. Condense**, turning one directory per chunk into a single portable file.

```console
$ edna condense examples/cape_rodney/config.yml --profile desktop
v00_cape_rodney: condensing around_cape_rodney_2D (gridded_2d)
  tow window whole column, effective volume at 5x5 sub-points, policy weight
  wrote /.../v00_cape_rodney_condensed.nc  (33.1 MB)
```

**7. Figures.** From the condensed file alone — copy it to a laptop and run the
same command there with `--condensed`.

```console
$ edna figures examples/cape_rodney/config.yml --profile desktop
v00_cape_rodney: reading v00_cape_rodney_condensed.nc
  61/100 sources (61.0%) from 3 station(s) at t=17
    station  174.82421, -36.28272   +34 sources
    station  174.82021, -36.29812   +18 sources
    station  174.82261, -36.27632   +9 sources
  wrote /.../figures/detectability_map.png
  wrote /.../figures/footprint_timeseries.png
  wrote /.../figures/coverage_curve.png
  wrote /.../figures/summary.json
```

`summary.json` holds the derived numbers — chosen stations, sources covered, the
coverage curve, footprint areas. That is the file to diff when comparing two
runs, rather than eyeballing the maps.


### Everything after the release points, in one call

The three stages above are separate because that is where the desktop/HPC round
trip happens. When it does not — one machine, one sitting — `run_all` is the
whole of it:

```python
from edna_sampling import SiteConfig, MachineProfile, ensure_release_points, run_all

site = SiteConfig.load("examples/cape_rodney/config.yml")
profile = MachineProfile.load("profiles/desktop.yaml")

ensure_release_points(site, profile)        # look at the figure before going on
out = run_all(site, profile)                # submit -> condense -> figures

print(out["condensed"], out["results"]["coverage_fraction"])
```

or, the same thing from the shell:

```bash
edna points <config> --profile <profile>
edna all    <config> --profile <profile>
```

`run_all` stops at the first stage that fails rather than carrying on. A
condensed product built from a run with failed chunks opens perfectly happily —
it just holds fewer sources than the config declares — so the failure is worth
more than the artifact. Fix the cause, then `--resume` to model only what is
outstanding and rebuild the analysis.

It deliberately does not generate the release points: they want eyeballing
before you commit to a few hundred model runs, and doing it here would let a
typo in `depth_range` move every source unseen.
