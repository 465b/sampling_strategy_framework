"""Run provenance, and the guard against a config changing mid-run.

A run is hundreds of jobs over hours. Editing the site config while they are in
flight leaves early and late chunks belonging to different experiments, and the
stitched arrays are then quietly inconsistent.
"""


import pytest
import yaml

from edna_sampling.config import ConfigError, MachineProfile, SiteConfig
from edna_sampling.manifest import (check_manifest, manifest_path, read_manifest,
                                    write_manifest)


@pytest.fixture
def setup(tmp_path, site_dict, profile_dict):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "poly.csv").write_text("")
    cfg_path = site_dir / "v1.yaml"
    cfg_path.write_text(yaml.safe_dump(site_dict))

    prof_path = tmp_path / "p.yaml"
    prof_path.write_text(yaml.safe_dump(profile_dict))

    points = site_dir / "v1_testsite_lloyd_points.csv"
    points.write_text("\n".join(f"{i}.0,{i}.5" for i in range(8)) + "\n")
    return SiteConfig.load(cfg_path), MachineProfile.load(prof_path), points, cfg_path, site_dict


def test_manifest_records_provenance(setup):
    site, profile, points, *_ = setup
    path = write_manifest(site, profile, points)
    assert path == manifest_path(site, profile)

    m = read_manifest(site, profile)
    assert m["run_name"] == site.run_name
    assert m["fingerprint"] == site.fingerprint()
    assert m["n_chunks"] == site.n_chunks
    assert m["release_points"]["n"] == site.release_points.n
    assert len(m["release_points"]["sha256"]) == 64
    assert m["versions"]["edna_sampling"]
    assert m["config"]["name"] == "testsite"


def test_absent_manifest_is_not_an_error(setup):
    """Runs predating the manifest, and the first chunk of a new run, have none."""
    site, profile, points, *_ = setup
    assert read_manifest(site, profile) is None
    check_manifest(site, profile, points)


def test_unchanged_config_passes(setup):
    site, profile, points, *_ = setup
    write_manifest(site, profile, points)
    check_manifest(site, profile, points)


def test_changed_config_is_caught(setup):
    site, profile, points, cfg_path, site_dict = setup
    write_manifest(site, profile, points)

    site_dict["model"]["edna_half_life_hours"] = 12      # different experiment
    cfg_path.write_text(yaml.safe_dump(site_dict))
    changed = SiteConfig.load(cfg_path)

    with pytest.raises(ConfigError, match="config has changed"):
        check_manifest(changed, profile, points)


def test_changed_release_points_are_caught(setup):
    """Same config, different sources: chunks already written used other locations."""
    site, profile, points, *_ = setup
    write_manifest(site, profile, points)
    points.write_text("\n".join(f"{i}.0,{i}.9" for i in range(8)) + "\n")
    with pytest.raises(ConfigError, match="release points have changed"):
        check_manifest(site, profile, points)


def test_error_names_the_way_out(setup):
    site, profile, points, cfg_path, site_dict = setup
    write_manifest(site, profile, points)
    site_dict["model"]["edna_half_life_hours"] = 12
    cfg_path.write_text(yaml.safe_dump(site_dict))
    with pytest.raises(ConfigError) as exc:
        check_manifest(SiteConfig.load(cfg_path), profile, points)
    message = str(exc.value)
    assert "Bump `version`" in message
    assert "--ignore-manifest" in message


def test_corrupt_manifest_is_reported(setup):
    site, profile, points, *_ = setup
    write_manifest(site, profile, points)
    manifest_path(site, profile).write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        read_manifest(site, profile)
