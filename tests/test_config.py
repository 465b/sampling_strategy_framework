from pathlib import Path

import pytest
import yaml

from edna_sampling.config import ConfigError, MachineProfile, SiteConfig


def _load(site_dict, tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump(site_dict))
    return SiteConfig.load(p)


# --- happy path ------------------------------------------------------------

def test_loads_and_derives(site_dict, tmp_path):
    cfg = _load(site_dict, tmp_path)
    assert cfg.run_name == "v1_testsite"
    # 8 points at 2 per chunk
    assert cfg.n_chunks == 4
    # 1e7 copies/h / 1000 copies per particle * 120 s = 333.3 -> 333
    assert cfg.source.particles_per_release(cfg.model.time_step) == 333


def test_n_chunks_rounds_up(site_dict, tmp_path):
    site_dict["release_points"]["n"] = 7
    cfg = _load(site_dict, tmp_path)
    assert cfg.n_chunks == 4  # 2+2+2+1


def test_paths_resolve_relative_to_the_config(site_dict, tmp_path):
    """The polygon is a path beside the config; the hindcast is a logical name the
    machine profile resolves, so it is deliberately not a path at all."""
    cfg = _load(site_dict, tmp_path)
    assert cfg.domain == (tmp_path / "testsite" / "poly.csv").resolve()
    assert cfg.hindcast == "some_hindcast"


def test_domain_grid_is_rejected_with_the_migration(site_dict, tmp_path):
    """`domain.grid` named a mesh per experiment. The mesh belongs to the
    hindcast, so the key is gone - and saying so beats "unknown key"."""
    site_dict["domain"] = {"grid": "hauraki_gulf", "polygon": "testsite/poly.csv"}
    with pytest.raises(ConfigError, match="no longer part of a site config"):
        _load(site_dict, tmp_path)


def test_the_old_nested_domain_section_is_rejected_with_the_migration(site_dict, tmp_path):
    """`domain:` held only the polygon once the mesh moved out, so it collapsed
    to one line. Show the replacement rather than reporting an unknown key."""
    site_dict["domain"] = {"polygon": "testsite/poly.csv"}
    with pytest.raises(ConfigError, match=r"domain: testsite/poly.csv"):
        _load(site_dict, tmp_path)


def test_hindcast_as_a_mapping_is_rejected_with_the_migration(site_dict, tmp_path):
    """`file_mask` moved profile-side; the old two-key block must say where."""
    site_dict["hindcast"] = {"name": "some_hindcast", "file_mask": "schout_*.nc"}
    with pytest.raises(ConfigError, match="file_mask.*moved to the machine profile"):
        _load(site_dict, tmp_path)


def test_hindcast_given_as_a_path_is_rejected(site_dict, tmp_path):
    site_dict["hindcast"] = "/data/hindcast/2018"   # machine-path-ok
    with pytest.raises(ConfigError, match="logical name"):
        _load(site_dict, tmp_path)


def test_fingerprint_is_stable_and_sensitive(site_dict, tmp_path):
    a = _load(site_dict, tmp_path)
    b = _load(site_dict, tmp_path)
    assert a.fingerprint() == b.fingerprint()
    site_dict["release_points"]["seed"] = 43
    assert _load(site_dict, tmp_path).fingerprint() != a.fingerprint()


# --- the config must not be able to bind an experiment to one machine ------

def test_absolute_paths_are_rejected(site_dict, tmp_path):
    site_dict["domain"] = "/hpcfreenas/laurin/poly.csv"  # machine-path-ok
    with pytest.raises(ConfigError, match="absolute path"):
        _load(site_dict, tmp_path)


# --- validation ------------------------------------------------------------

def test_unknown_key_is_reported_with_alternatives(site_dict, tmp_path):
    site_dict["release_points"]["seeed"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        _load(site_dict, tmp_path)


def test_missing_key_is_named(site_dict, tmp_path):
    del site_dict["model"]["time_step"]
    with pytest.raises(ConfigError, match="time_step"):
        _load(site_dict, tmp_path)


def test_source_that_sheds_less_than_one_particle_is_rejected(site_dict, tmp_path):
    site_dict["source"]["copies_per_particle"] = 1e12
    with pytest.raises(ConfigError, match="particles_per_release"):
        _load(site_dict, tmp_path)


def test_depth_range_must_be_ordered(site_dict, tmp_path):
    site_dict["release_points"]["depth_range"] = [30, 5]
    with pytest.raises(ConfigError, match="min < max"):
        _load(site_dict, tmp_path)


def test_duplicate_stats_names_rejected(site_dict, tmp_path):
    site_dict["stats"].append(dict(site_dict["stats"][0]))
    with pytest.raises(ConfigError, match="duplicate name"):
        _load(site_dict, tmp_path)


# --- the vertical-semantics trap -------------------------------------------

def test_3d_stats_must_state_vertical_mode(site_dict, tmp_path):
    del site_dict["stats"][0]["vertical_range_measured_relative_to"]
    with pytest.raises(ConfigError, match="vertical_range_measured_relative_to"):
        _load(site_dict, tmp_path)


def test_3d_rejects_negative_z_min_for_surface_mode(site_dict, tmp_path):
    site_dict["stats"][0]["z_min"] = -5
    with pytest.raises(ConfigError, match="must be >= 0"):
        _load(site_dict, tmp_path)


def test_2d_only_keys_rejected_on_3d(site_dict, tmp_path):
    site_dict["stats"][0]["near_seasurface"] = 19
    with pytest.raises(ConfigError, match="only apply to gridded_2d"):
        _load(site_dict, tmp_path)


def test_3d_only_keys_rejected_on_2d(site_dict, tmp_path):
    site_dict["stats"] = [{
        "name": "tow", "kind": "gridded_2d", "grid_center": [1, 2],
        "rows": 5, "cols": 5, "span": [0.1, 0.1], "layers": 20,
    }]
    with pytest.raises(ConfigError, match="only apply to gridded_3d"):
        _load(site_dict, tmp_path)


def test_2d_rejects_two_tow_references(site_dict, tmp_path):
    site_dict["stats"] = [{
        "name": "tow", "kind": "gridded_2d", "grid_center": [1, 2],
        "rows": 5, "cols": 5, "span": [0.1, 0.1],
        "near_seasurface": 19, "near_seabed": 5,
    }]
    with pytest.raises(ConfigError, match="at most one"):
        _load(site_dict, tmp_path)


def test_2d_tow_window_accepted(site_dict, tmp_path):
    site_dict["stats"] = [{
        "name": "tow_full", "kind": "gridded_2d", "grid_center": [1, 2],
        "rows": 5, "cols": 5, "span": [0.1, 0.1], "near_seasurface": 19,
    }]
    cfg = _load(site_dict, tmp_path)
    assert cfg.stats[0].near_seasurface == 19


# --- machine profile -------------------------------------------------------

def _profile(profile_dict, tmp_path, name="prof.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(profile_dict))
    return MachineProfile.load(p)


def test_profile_resolves_hindcast(profile_dict, tmp_path):
    entry = _profile(profile_dict, tmp_path).hindcast("some_hindcast")
    assert entry.input_dir == (tmp_path / "hindcast")
    assert entry.file_mask == "schout_*.nc"


def test_profile_accepts_the_bare_directory_shorthand(profile_dict, tmp_path):
    """`name: /path` when the defaults will do, so a profile for a
    conventionally-named hindcast stays one line."""
    profile_dict["hindcasts"] = {"some_hindcast": str(tmp_path / "hindcast")}
    entry = _profile(profile_dict, tmp_path).hindcast("some_hindcast")
    assert entry.input_dir == (tmp_path / "hindcast")
    assert entry.file_mask == "*.nc"
    assert entry.mesh is None


def test_profile_unknown_hindcast_lists_known_ones(profile_dict, tmp_path):
    prof = _profile(profile_dict, tmp_path)
    with pytest.raises(ConfigError, match="some_hindcast"):
        prof.hindcast("nope")


def test_profile_rejects_an_unknown_key_inside_a_hindcast(profile_dict, tmp_path):
    profile_dict["hindcasts"]["some_hindcast"]["grid"] = "old.nc"
    with pytest.raises(ConfigError, match="unknown key"):
        _profile(profile_dict, tmp_path)


# --- where the mesh comes from ---------------------------------------------

def test_mesh_defaults_to_the_cache_under_scratch(profile_dict, tmp_path):
    """With no override the mesh is derived and cached per hindcast, so the path
    is a function of the profile alone - `edna points` and `edna mesh` must agree
    about it without being told."""
    profile_dict["scratch"] = str(tmp_path / "fast")
    prof = _profile(profile_dict, tmp_path)
    assert prof.mesh_path("some_hindcast") == tmp_path / "fast" / "meshes" / "some_hindcast.nc"


def test_mesh_cache_falls_back_to_output_root_without_scratch(profile_dict, tmp_path):
    prof = _profile(profile_dict, tmp_path)
    assert prof.mesh_path("some_hindcast") == tmp_path / "out" / "meshes" / "some_hindcast.nc"


def test_a_mesh_override_wins_and_resolves_against_the_profile(profile_dict, tmp_path):
    """A relative override lets a profile ship a mesh beside it, without
    hard-coding anyone's home directory."""
    profile_dict["hindcasts"]["some_hindcast"]["mesh"] = "meshes/mine.nc"
    prof = _profile(profile_dict, tmp_path)
    assert prof.mesh_path("some_hindcast") == (tmp_path / "meshes" / "mine.nc").resolve()


def test_profile_rejects_the_retired_grids_section(profile_dict, tmp_path):
    """`grids:` was a top-level mapping of its own; it moved inside `hindcasts:`
    because a mesh only means anything against the hindcast it came from."""
    profile_dict["grids"] = {"hauraki_gulf": "hydro_grid.nc"}
    with pytest.raises(ConfigError, match="moved under"):
        _profile(profile_dict, tmp_path)


def test_profile_expands_a_home_relative_path(profile_dict, tmp_path):
    """`output_root: ~/edna` should mean the home directory, not a directory
    literally called '~'."""
    profile_dict["output_root"] = "~/edna"
    assert _profile(profile_dict, tmp_path).output_root == Path.home() / "edna"


def test_profile_rejects_a_typo_in_the_local_section(profile_dict, tmp_path):
    """Every other section rejects unknown keys. Without this, `n_parallel_job`
    is silently ignored and the run is serial."""
    profile_dict["local"] = {"n_parallel_job": 8}
    with pytest.raises(ConfigError, match="unknown key"):
        _profile(profile_dict, tmp_path)


def test_profile_rejects_retired_scheduler_key(profile_dict, tmp_path):
    """`scheduler:` was removed with the SLURM backend; say so rather than
    reporting it as an unknown key."""
    profile_dict["scheduler"] = "slurm"
    p = tmp_path / "prof.yaml"
    p.write_text(yaml.safe_dump(profile_dict))
    with pytest.raises(ConfigError, match="no longer supported"):
        MachineProfile.load(p)


# --- input existence is checked separately from parsing --------------------

def test_validate_inputs_reports_missing_files(site_dict, tmp_path):
    cfg = _load(site_dict, tmp_path)
    with pytest.raises(ConfigError, match="not found"):
        cfg.validate_inputs()


def test_validate_inputs_reports_a_missing_hindcast_directory(site_dict, tmp_path,
                                                              profile_dict):
    """Checked here rather than at first read: a run is 400 jobs, and finding out
    from job 1 that the profile points nowhere is the cheapest moment."""
    (tmp_path / "testsite").mkdir()
    (tmp_path / "testsite" / "poly.csv").write_text("")
    prof = _profile(profile_dict, tmp_path)          # tmp_path/hindcast not created
    with pytest.raises(ConfigError, match="some_hindcast"):
        _load(site_dict, tmp_path).validate_inputs(prof)


def test_validate_inputs_passes_when_present(site_dict, tmp_path):
    (tmp_path / "testsite").mkdir()
    (tmp_path / "testsite" / "poly.csv").write_text("")
    _load(site_dict, tmp_path).validate_inputs()


# --- the analysis block -----------------------------------------------------

def test_analysis_defaults_are_the_published_values(site_dict, tmp_path):
    """The defaults are what 2026_07_22_paper_figures.ipynb used, so omitting the
    block reproduces the paper rather than something arbitrary."""
    cfg = _load(site_dict, tmp_path)
    d = cfg.analysis.detection
    assert (d.threshold_copies, d.sample_volume_m3, d.individuals_per_source) == (10.0, 1.0, 100.0)
    assert cfg.analysis.optimizer.kind == "greedy"
    assert cfg.analysis.optimizer.n_stations == 3
    assert cfg.analysis.sweeps.coverage_levels == (0.125, 0.25, 0.5, 0.75)


def test_filtered_volume_takes_copies_per_particle_from_the_source_block(site_dict, tmp_path):
    """The model's discretisation and the analysis must not be able to disagree
    about copies_per_particle, so it is not restated under analysis:."""
    site_dict["source"]["copies_per_particle"] = 500.0
    site_dict["analysis"] = {"detection": {"individuals_per_source": 2, "sample_volume_m3": 3.0}}
    cfg = _load(site_dict, tmp_path)
    assert cfg.filtered_volume == 2 * 500.0 * 3.0


def test_analysis_is_excluded_from_the_fingerprint(site_dict, tmp_path):
    """Re-deciding a detection threshold must not invalidate a finished
    400-chunk run or trip the mid-run drift check."""
    before = _load(site_dict, tmp_path).fingerprint()
    site_dict["analysis"] = {"detection": {"threshold_copies": 999},
                             "optimizer": {"kind": "genetic", "n_stations": 17}}
    assert _load(site_dict, tmp_path).fingerprint() == before


def test_a_model_change_still_moves_the_fingerprint(site_dict, tmp_path):
    before = _load(site_dict, tmp_path).fingerprint()
    site_dict["model"]["time_step"] = site_dict["model"]["time_step"] * 2
    assert _load(site_dict, tmp_path).fingerprint() != before


def test_statistic_is_implicit_when_only_one_is_defined(site_dict, tmp_path):
    cfg = _load(site_dict, tmp_path)
    assert cfg.statistic() is cfg.stats[0]


def test_several_statistics_require_saying_which(site_dict, tmp_path):
    """The auckland example carries a 2D and a 3D statistic at once; guessing would
    silently analyse the wrong field."""
    second = dict(site_dict["stats"][0]); second["name"] = "another"
    site_dict["stats"] = [site_dict["stats"][0], second]
    with pytest.raises(ConfigError, match="must say which one"):
        _load(site_dict, tmp_path).statistic()


def test_unknown_statistic_lists_the_known_ones(site_dict, tmp_path):
    site_dict["analysis"] = {"statistic": "nope"}
    with pytest.raises(ConfigError, match="Known statistics"):
        _load(site_dict, tmp_path).statistic()


@pytest.mark.parametrize("block,match", [
    ({"optimizer": {"kind": "annealing"}}, "must be one of"),
    ({"optimizer": {"n_stations": 0}}, "at least 1"),
    ({"detection": {"threshold_copies": -1}}, "threshold_copies"),
    ({"sweeps": {"coverage_levels": [0, 0.5]}}, "fractions in"),
    ({"sweeps": {"volume_exponents_m3": [-3, 2]}}, "start, stop, step"),
    ({"sweeps": {"volume_exponents_m3": [2, -3, 0.25]}}, "stop > start"),
    ({"detection": {"threshhold_copies": 10}}, "unknown"),
])
def test_analysis_rejects_bad_input(site_dict, tmp_path, block, match):
    site_dict["analysis"] = block
    with pytest.raises(ConfigError, match=match):
        _load(site_dict, tmp_path)


# --- the particle buffer, derived from the decay physics --------------------

def test_particles_alive_reproduces_a_real_run(site_dict, tmp_path):
    """The v33 cape_rodney configuration - 1e7 copies/h at 1000 copies/particle,
    120 s steps, 6 h half-life, 47 h run - logged 86,114 particles alive at its
    peak. N(t) = (R/alpha)(1 - exp(-alpha t)) should land on that."""
    site_dict["source"].update(shedding_rate_per_hour=1.0e7, copies_per_particle=1000)
    site_dict["model"].update(time_step=120, duration_hours=47, edna_half_life_hours=6)
    cfg = _load(site_dict, tmp_path)
    assert cfg.particles_alive_at_end == pytest.approx(86_114, rel=0.01)


def test_a_long_run_approaches_the_equilibrium(site_dict, tmp_path):
    """N -> R/alpha as t -> inf, which is the paper's equilibrium result."""
    import math
    site_dict["source"].update(shedding_rate_per_hour=1.0e7, copies_per_particle=1000)
    site_dict["model"].update(time_step=120, duration_hours=1000, edna_half_life_hours=6)
    cfg = _load(site_dict, tmp_path)
    rate = cfg.source.particles_per_release(120) / 120
    equilibrium = rate / (math.log(2) / (6 * 3600))
    assert cfg.particles_alive_at_end == pytest.approx(equilibrium, rel=1e-6)


def test_a_shorter_half_life_needs_a_smaller_buffer(site_dict, tmp_path):
    """Faster decay means fewer particles coexist, which is the whole point of
    deriving this rather than guessing it."""
    site_dict["model"]["edna_half_life_hours"] = 6
    slow = _load(site_dict, tmp_path).particle_buffer_size
    site_dict["model"]["edna_half_life_hours"] = 1
    fast = _load(site_dict, tmp_path).particle_buffer_size
    assert fast < slow


def test_the_buffer_is_derived_when_not_given(site_dict, tmp_path):
    site_dict["model"].pop("particle_buffer_per_release_group", None)
    cfg = _load(site_dict, tmp_path)
    assert cfg.model.particle_buffer_per_release_group is None
    assert cfg.particle_buffer_size >= cfg.particles_alive_at_end
    # ...and not wildly more: this is the over-allocation the default replaces
    assert cfg.particle_buffer_size < 2 * cfg.particles_alive_at_end


def test_an_explicit_buffer_still_wins(site_dict, tmp_path):
    site_dict["model"]["particle_buffer_per_release_group"] = 12_345
    assert _load(site_dict, tmp_path).particle_buffer_size == 12_345


def test_a_nonsense_explicit_buffer_is_rejected(site_dict, tmp_path):
    site_dict["model"]["particle_buffer_per_release_group"] = 0
    with pytest.raises(ConfigError, match="particle_buffer_per_release_group"):
        _load(site_dict, tmp_path)


def test_model_start_is_optional_and_validated(site_dict, tmp_path):
    """Unset means "wherever the hindcast begins", which is what the examples do."""
    site_dict["model"].pop("start", None)
    assert _load(site_dict, tmp_path).model.start is None
    site_dict["model"]["start"] = "2018-01-02T00:00:00"
    assert _load(site_dict, tmp_path).model.start == "2018-01-02T00:00:00"


def test_a_malformed_start_is_caught_at_parse_time(site_dict, tmp_path):
    """Not hours into a run, when OceanTracker would be the one to complain."""
    site_dict["model"]["start"] = "the second of January"
    with pytest.raises(ConfigError, match="ISO-8601"):
        _load(site_dict, tmp_path)


def test_start_changes_the_experiment_identity(site_dict, tmp_path):
    """Two runs over different periods are different experiments."""
    site_dict["model"]["start"] = "2018-01-02T00:00:00"
    a = _load(site_dict, tmp_path).fingerprint()
    site_dict["model"]["start"] = "2018-02-02T00:00:00"
    assert _load(site_dict, tmp_path).fingerprint() != a
