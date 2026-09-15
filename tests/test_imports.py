"""The analysis path must not import the modelling stack.

OceanTracker is a core dependency, so this is no longer about whether it is
installed - it is about cost. Importing it takes seconds and prints a banner, and
`edna figures`, `edna condense` and `edna --help` have no use for it. Keeping it
off those import paths is what makes them quick.

It breaks invisibly - everything still works, just slower - so the check runs
from a clean subprocess where the absence of the import is observable.
"""

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# modules `edna figures` walks, none of which should pay for the modelling stack
ANALYSIS_MODULES = [
    "edna_sampling.config",
    "edna_sampling.detection",
    "edna_sampling.analysis",
    "edna_sampling.figures",
    "edna_sampling.condense",
    "edna_sampling.mesh",
]


@pytest.mark.parametrize("module", ANALYSIS_MODULES)
def test_analysis_modules_do_not_import_oceantracker(module):
    probe = textwrap.dedent(f"""
        import sys
        import {module}            # noqa: F401
        print("oceantracker" if "oceantracker" in sys.modules else "")
    """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        f"{module} imported oceantracker at import time; import it inside the "
        f"function that needs it, so the analysis path stays quick")


def test_detection_is_numpy_only():
    """detection.py is the innermost layer and stays lighter still: no xarray and
    no numba either, so it loads fast enough to import in a notebook cell."""
    probe = textwrap.dedent("""
        import sys
        import edna_sampling.detection            # noqa: F401
        heavy = [m for m in ("oceantracker", "xarray", "numba", "pandas", "IPython")
                 if m in sys.modules]
        print(",".join(heavy))
    """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"edna_sampling.detection pulled in: {out.stdout.strip()}"


def test_oceantracker_is_a_core_dependency():
    """It used to be the `[model]` extra so the analysis half could install
    without it. That split is gone - one `pip install -e .` gets everything - so
    the only thing left to protect is the import discipline above."""
    meta = tomllib.loads((REPO / "pyproject.toml").read_text())
    core = " ".join(meta["project"]["dependencies"])
    assert "oceantracker" in core
    assert "model" not in meta["project"]["optional-dependencies"]
