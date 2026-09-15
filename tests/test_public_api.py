"""The package's public API: the stage functions a script uses instead of the CLI.

`edna <command>` is meant to be a thin wrapper, so anything the CLI can do must be
reachable from Python. These tests pin that, and pin the two properties that make
the re-exports safe to have at all.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import edna_sampling

REPO = Path(__file__).resolve().parents[1]


def test_every_advertised_export_resolves():
    """A name in __all__ that does not resolve is worse than no name at all: it
    only fails for the person who read the docs and tried it."""
    broken = [n for n in edna_sampling.__all__ if not hasattr(edna_sampling, n)]
    assert broken == []


def test_the_stage_functions_are_all_exported():
    """One per pipeline stage, so a script never has to reach into submodules."""
    for name in ("SiteConfig", "MachineProfile", "ensure_mesh",
                 "ensure_release_points", "run_chunk", "run_all", "build_params",
                 "open_condensed", "site_results", "write_figures"):
        assert name in edna_sampling.__all__, name
        assert callable(getattr(edna_sampling, name)) or isinstance(
            getattr(edna_sampling, name), type), name


def test_an_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        edna_sampling.definitely_not_a_stage


def test_importing_the_package_stays_cheap():
    """`edna --help` imports this package too. Eager re-exports would make every
    CLI invocation pay for numpy, xarray and matplotlib, so the bindings are lazy
    - and a clean interpreter is the only way to check that."""
    probe = textwrap.dedent("""
        import sys
        import edna_sampling                      # noqa: F401
        heavy = [m for m in ("numpy", "xarray", "matplotlib", "scipy", "pandas")
                 if m in sys.modules]
        print(",".join(heavy))
    """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"import pulled in: {out.stdout.strip()}"


def test_the_analysis_half_needs_no_modelling_stack():
    """Stages 4-5 must install on a laptop that will never run the model, so
    reaching the analysis exports must not import oceantracker."""
    probe = textwrap.dedent("""
        import sys
        from edna_sampling import open_condensed, site_results, write_figures  # noqa: F401
        print("oceantracker" if "oceantracker" in sys.modules else "")
    """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "the analysis half pulled in oceantracker"


def test_the_condense_function_is_not_shadowed_by_its_module():
    """`condense` names both a module and the function in it, so the package
    deliberately does not re-export the bare name - a submodule import elsewhere
    would bind the module onto the package and the export would silently become a
    module. Importing it directly always works, and is what the README shows."""
    from edna_sampling.condense import condense

    assert callable(condense) and not hasattr(condense, "__path__")
    assert "condense" not in edna_sampling.__all__
