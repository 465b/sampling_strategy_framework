"""Shared fixtures. Deliberately free of hindcast data or model runs: everything
here must pass in seconds on a laptop with no access to an HPC filesystem."""

import pytest


@pytest.fixture
def site_dict():
    """A minimal but complete site config, as parsed YAML."""
    return {
        "name": "testsite",
        "version": "v1",
        "domain": "testsite/poly.csv",
        "hindcast": "some_hindcast",
        "release_points": {"n": 8, "depth_range": [5, 30], "seed": 42},
        "source": {"shedding_rate_per_hour": 1.0e7, "copies_per_particle": 1000},
        "model": {
            "time_step": 120,
            "duration_hours": 47,
            "edna_half_life_hours": 6,
            "critical_friction_velocity": 0.009,
        },
        "stats": [{
            "name": "test_3d",
            "kind": "gridded_3d",
            "grid_center": [174.8, -36.3],
            "rows": 10, "cols": 10, "span": [0.05, 0.05],
            "layers": 20, "z_min": 0, "z_max": 20,
            "vertical_range_measured_relative_to": "surface",
        }],
        "chunking": {"release_groups_per_chunk": 2},
    }


@pytest.fixture
def profile_dict(tmp_path):
    return {
        "hindcasts": {"some_hindcast": {"input_dir": str(tmp_path / "hindcast"),
                                        "file_mask": "schout_*.nc"}},
        "output_root": str(tmp_path / "out"),
        "local": {"n_parallel_jobs": 2},
    }
