"""The refactor must not change what OceanTracker is told to do.

`tests/data/legacy_params/*.json` was extracted by importing the original
experiment scripts with `oceantracker.main.run` stubbed out, capturing the
params dict they hand over for chunk 1. These tests assert that the YAML site
config plus `build_params` reproduce that dict exactly, so the move from
executable configs to declarative ones is provably behaviour-preserving.

Regenerating a fixture is a deliberate act: it means the experiment changed.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from edna_sampling.config import MachineProfile, SiteConfig
from edna_sampling.params import build_params

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "data" / "legacy_params"

# site key -> (yaml config, cached release points used by the original run)
#
# These live here rather than in examples/ because they are not examples: they
# are what the original scripts ran, kept to pin the equivalence. Their `v33` /
# `v2` labels are load-bearing and must not be tidied away the way the examples'
# were - `version:` feeds `run_name`, which the frozen JSON pins as
# `output_file_base` and `root_output_dir`. Renaming them would falsify the
# capture rather than modernise it.
CASES = {
    "cape_rodney": ("v33_cape_rodney.yaml", "v33_cape_rodney_lloyd_points.csv"),
    "auckland": ("v2_auckland_bay.yaml", "v2_auckland_bay_lloyd_points.csv"),
}

OUTPUT_ROOT = "<OUTPUT_ROOT>"
HINDCAST_DIR = "<HINDCAST_DIR>"

#: Keys that have deliberately diverged from the capture, with what they became.
#:
#: This is for changes to where output *lands*, never to what the model does.
#: The legacy scripts wrote to `<run_name>_chunked`; every run is chunked now,
#: so the suffix said nothing and the directory is just `<run_name>`. The
#: fixture still records what the original script asked for, which is the point
#: of keeping it - so the divergence is pinned here rather than edited into the
#: capture. Anything affecting the particles, the grid or the statistics belongs
#: in the fixture, not in this dict.
DELIBERATE_DIVERGENCE = {
    "root_output_dir": lambda legacy: legacy.removesuffix("_chunked"),
}


@pytest.fixture
def profile(tmp_path):
    """A profile whose paths are the fixtures' placeholders, so the generated
    params can be compared literally against the stored legacy dict.

    `file_mask` is supplied here rather than by the site config, which is where
    it moved to: the legacy scripts' `reader` dict is still reproduced exactly,
    now with the machine rather than the experiment naming the files.
    """
    p = tmp_path / "equiv.yaml"
    p.write_text(yaml.safe_dump({
        "hindcasts": {"hauraki_gulf": {"input_dir": HINDCAST_DIR,
                                               "file_mask": "schout_*.nc"}},
        "output_root": OUTPUT_ROOT,
    }))
    return MachineProfile.load(p)


def _build(site_key, profile):
    cfg_path, points_path = CASES[site_key]
    site = SiteConfig.load(FIXTURES / cfg_path)
    points = np.loadtxt(FIXTURES / points_path, delimiter=",")
    return build_params(site, profile, chunk_number=1, release_points=points)


def _legacy(site_key):
    return json.loads((FIXTURES / f"{site_key}.json").read_text())


@pytest.mark.parametrize("site_key", sorted(CASES))
def test_generated_params_match_legacy_script(site_key, profile):
    generated = _build(site_key, profile)
    legacy = _legacy(site_key)

    # compare via canonical JSON so numpy scalars and tuples normalise
    gen = json.loads(json.dumps(generated, sort_keys=True, default=float))
    assert set(gen) == set(legacy), (
        f"top-level keys differ: only generated={sorted(set(gen) - set(legacy))}, "
        f"only legacy={sorted(set(legacy) - set(gen))}"
    )
    for key in sorted(legacy):
        expected = legacy[key]
        if key in DELIBERATE_DIVERGENCE:
            expected = DELIBERATE_DIVERGENCE[key](expected)
            assert expected != legacy[key], (
                f"{key!r} is listed as diverging from the capture but matches it; "
                f"drop it from DELIBERATE_DIVERGENCE."
            )
        assert gen[key] == expected, f"{site_key}: params[{key!r}] differs"


@pytest.mark.parametrize("site_key", sorted(CASES))
def test_output_directory_is_the_run_name_with_no_suffix(site_key, profile):
    """Every run is chunked, so the directory does not say so."""
    site = SiteConfig.load(FIXTURES / CASES[site_key][0])
    assert _build(site_key, profile)["root_output_dir"] == f"{OUTPUT_ROOT}/{site.run_name}"


def test_chunking_covers_every_release_point_exactly_once(profile):
    """Chunk n always maps to points [(n-1)k, nk) - a property of the config,
    not of the order jobs happened to start."""
    from edna_sampling.params import chunk_slice

    site = SiteConfig.load(FIXTURES / CASES["cape_rodney"][0])
    seen = []
    for chunk in range(1, site.n_chunks + 1):
        sl = chunk_slice(site, chunk)
        seen.extend(range(sl.start, sl.stop))
    assert seen == list(range(site.release_points.n))


def test_chunk_number_is_one_based_and_bounded(profile):
    from edna_sampling.config import ConfigError
    from edna_sampling.params import chunk_slice, chunk_run_name

    site = SiteConfig.load(FIXTURES / CASES["cape_rodney"][0])
    assert chunk_run_name(site, 1) == f"{site.run_name}_chunk_001"
    assert chunk_slice(site, 1).start == 0
    for bad in (0, site.n_chunks + 1):
        with pytest.raises(ConfigError, match="out of range"):
            chunk_slice(site, bad)


def test_stale_release_point_cache_is_rejected(profile):
    site = SiteConfig.load(FIXTURES / CASES["cape_rodney"][0])
    with pytest.raises(Exception, match="stale"):
        build_params(site, profile, chunk_number=1, release_points=np.zeros((399, 2)))


