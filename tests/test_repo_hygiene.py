"""Repo-wide invariants.

The pipeline became unrunnable by anyone else because machine-specific paths
leaked into every layer. These tests keep that from happening again, and check
that the shipped configs actually load.
"""

import re
from pathlib import Path

import pytest

from edna_sampling.config import MachineProfile, SiteConfig

REPO = Path(__file__).resolve().parents[1]

# The shipped, maintained examples. Deliberately not `experiments/*`: that
# directory is git-ignored working space, so globbing it would make the suite
# test whatever the developer happens to have lying around, and a fresh clone
# would run a different set of tests from this machine.
EXAMPLE_CONFIGS = sorted(REPO.glob("examples/*/config.yml"))

# roots that belong to one particular machine
MACHINE_PATH = re.compile(r"(?<![\w.])/(?:hpcfreenas|nesi|data/share|home/[a-z]+)(?:/|\b)")

# where an absolute machine path is legitimate
ALLOWED = {
    "profiles",              # that is a profile's entire job
    "docs",                  # evidence and worked examples
    ".venv", "build", ".git", "__pycache__",
    "archiv",                # kept for reference, not maintained
    "experiments",           # git-ignored working space; not ours to police
}

# a line may opt out when the path is the point, e.g. a test asserting rejection
OPT_OUT = "machine-path-ok"

def _tracked_sources():
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
            continue
        rel = path.relative_to(REPO)
        if set(rel.parts) & ALLOWED:
            continue
        yield rel, path


def test_no_machine_specific_paths_outside_profiles():
    offenders = []
    for rel, path in _tracked_sources():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if MACHINE_PATH.search(line) and OPT_OUT not in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()[:90]}")
    assert not offenders, (
        "machine-specific absolute paths must live in profiles/ only "
        f"(add a trailing '# {OPT_OUT}' comment if the path is the point):\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("config", sorted(
    p.relative_to(REPO).as_posix() for p in EXAMPLE_CONFIGS))
def test_shipped_site_configs_load(config):
    site = SiteConfig.load(REPO / config)
    site.validate_inputs()
    assert site.n_chunks >= 1
    assert site.stats


@pytest.mark.parametrize("profile", sorted(
    p.relative_to(REPO).as_posix() for p in REPO.glob("profiles/*.yaml")))
def test_shipped_profiles_load(profile):
    MachineProfile.load(REPO / profile)
