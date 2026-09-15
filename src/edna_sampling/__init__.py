"""eDNA sampling-strategy framework.

Plan, run and analyse OceanTracker particle-tracking experiments that ask where
to sample for environmental DNA.

The pipeline has five stages; see docs/architecture.md for how they fit together.

  1. mesh             the hindcast's grid, derived once per machine
  2. release points   Lloyd-relaxed source locations for a site
  3. model run        OceanTracker, one chunk of release groups per job
  4. condense         chunked model output -> one portable analysis product
  5. figures          detectability analysis and paper figures

Stages 1-3 need OceanTracker and the hindcast and run where the data is; stages
4-5 read the condensed product and run anywhere. The analysis code never imports
OceanTracker, which is what keeps it quick to load.
"""

__version__ = "0.1.0"

# The stage functions, re-exported so a script can drive the pipeline without
# going through the CLI:
#
#     from edna_sampling import SiteConfig, MachineProfile, ensure_release_points
#
# `condense` is the one exception and is imported from its module directly - see
# the note in _LAZY below.
#
# Bound lazily (PEP 562) rather than imported here: `edna --help` also imports
# this package, and eager imports would make every CLI invocation pay for numpy,
# xarray and matplotlib. Nothing below pulls in OceanTracker at import time -
# only actually running a stage does - so stages 4-5 still work on a machine that
# has no modelling stack.
_LAZY = {
    "SiteConfig": "config",
    "MachineProfile": "config",
    "ConfigError": "config",
    "ensure_mesh": "mesh",
    "load_mesh": "mesh",
    "ensure_release_points": "pipeline",
    "run_chunk": "pipeline",
    "run_all": "pipeline",
    "release_points_path": "pipeline",
    "build_params": "params",
    "root_output_dir": "params",
    "run_status": "backends",
    "submit_local": "backends",
    # NB no bare "condense": the function shares its name with its module, and
    # any `from edna_sampling.condense import ...` elsewhere binds the submodule
    # onto this package first, after which the lazy hook never fires and the name
    # would hand back a module. Import it directly instead:
    #     from edna_sampling.condense import condense
    "open_condensed": "condense",
    "condensed_path": "condense",
    "site_results": "figures",
    "write_figures": "figures",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(f"{__name__}.{module}"), name)
    # Cache it, and do so *after* the import: importing `edna_sampling.condense`
    # binds the submodule as an attribute of this package, which would otherwise
    # shadow the function of the same name from the second access onward.
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
