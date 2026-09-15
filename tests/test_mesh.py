"""Reading and writing the hydrodynamic mesh.

`derive_mesh` is not tested here: it needs oceantracker and a real hindcast, so
it is exercised by running `edna mesh`, not by the suite - which must stay
seconds-fast on a laptop. What *is* tested is everything that happens to a mesh
once it exists, including the three on-disk shapes one can arrive in.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from edna_sampling.config import ConfigError
from edna_sampling.mesh import Mesh, load_mesh, write_mesh

# a unit square split into two triangles, with a depth at each corner
NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
DEPTH = np.array([5.0, 10.0, 15.0, 20.0])


def _mesh():
    return Mesh(node_x=NODE_X, node_y=NODE_Y, triangles=TRIANGLES, water_depth=DEPTH)


def _assert_matches(mesh):
    np.testing.assert_allclose(mesh.node_x, NODE_X)
    np.testing.assert_allclose(mesh.node_y, NODE_Y)
    np.testing.assert_array_equal(mesh.triangles, TRIANGLES)
    np.testing.assert_allclose(mesh.water_depth, DEPTH)


def test_round_trips_through_its_own_format(tmp_path):
    path = write_mesh(_mesh(), tmp_path / "m.nc")
    _assert_matches(load_mesh(path))


def test_reads_an_oceantracker_grid_file(tmp_path):
    """`grid000.nc`, which every run drops in its output directory - so a profile
    can point `mesh:` at one someone already has."""
    path = tmp_path / "grid000.nc"
    xr.Dataset({
        "x": (("node", "vector2D"), np.column_stack([NODE_X, NODE_Y])),
        "triangles": (("tri", "vertex"), TRIANGLES),
        "water_depth": ("node", DEPTH),
    }).to_netcdf(path)
    _assert_matches(load_mesh(path))


def test_reads_a_raw_schism_file(tmp_path):
    """SCHISM face_nodes are 1-based, and a mixed mesh carries a fourth column
    that is not part of the triangle."""
    path = tmp_path / "schout_1.nc"
    quads = np.column_stack([TRIANGLES + 1, np.full(len(TRIANGLES), -99)])
    xr.Dataset({
        "SCHISM_hgrid_node_x": ("node", NODE_X),
        "SCHISM_hgrid_node_y": ("node", NODE_Y),
        "SCHISM_hgrid_face_nodes": (("face", "vertex"), quads),
        "depth": ("node", DEPTH),
    }).to_netcdf(path)
    _assert_matches(load_mesh(path))


def test_its_own_format_wins_over_the_others(tmp_path):
    """A file carrying both shapes must be read as the edna one, or a mesh that
    happens to keep an `x` variable would silently be read the wrong way."""
    path = tmp_path / "both.nc"
    ds = xr.Dataset({
        "node_x": ("node", NODE_X), "node_y": ("node", NODE_Y),
        "triangles": (("tri", "vertex"), TRIANGLES),
        "water_depth": ("node", DEPTH),
        "x": (("node", "vector2D"), np.column_stack([NODE_Y, NODE_X])),  # transposed
    })
    ds.to_netcdf(path)
    _assert_matches(load_mesh(path))


def test_a_missing_mesh_says_how_to_make_one(tmp_path):
    with pytest.raises(ConfigError, match="edna mesh"):
        load_mesh(tmp_path / "nope.nc")


def test_an_unrecognised_file_lists_what_was_tried(tmp_path):
    path = tmp_path / "other.nc"
    xr.Dataset({"temperature": ("node", DEPTH)}).to_netcdf(path)
    with pytest.raises(ConfigError, match="not a mesh this package recognises"):
        load_mesh(path)


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ConfigError, match="same length"):
        Mesh(node_x=NODE_X, node_y=NODE_Y[:2], triangles=TRIANGLES, water_depth=DEPTH)


def test_non_triangular_connectivity_is_rejected():
    with pytest.raises(ConfigError, match=r"\(n, 3\)"):
        Mesh(node_x=NODE_X, node_y=NODE_Y, water_depth=DEPTH,
             triangles=np.zeros((2, 4), dtype=np.int32))


def test_bounds_and_triangulation_describe_the_same_mesh():
    mesh = _mesh()
    assert mesh.bounds() == (0.0, 0.0, 1.0, 1.0)
    assert mesh.n_nodes == 4 and mesh.n_triangles == 2
    # the trifinder is what release_points and detection use to test membership
    finder = mesh.triangulation().get_trifinder()
    assert finder(np.array([0.5]), np.array([0.4]))[0] != -1     # inside
    assert finder(np.array([2.0]), np.array([2.0]))[0] == -1     # outside


def test_the_written_form_is_much_smaller_than_an_oceantracker_grid(tmp_path):
    """The reason for having a format of our own: the Hauraki Gulf mesh is 2.8 MB
    here against 27 MB as a grid000.nc, and none of the difference is read."""
    n = 20_000
    rng = np.random.default_rng(0)
    big = Mesh(node_x=rng.random(n), node_y=rng.random(n),
               triangles=rng.integers(0, n, (2 * n, 3)).astype(np.int32),
               water_depth=rng.random(n) * 50)
    ours = write_mesh(big, tmp_path / "ours.nc")

    fat = tmp_path / "grid000.nc"
    xr.Dataset({
        "x": (("node", "vector2D"), np.column_stack([big.node_x, big.node_y])),
        "triangles": (("tri", "vertex"), big.triangles),
        "water_depth": ("node", big.water_depth),
        # the parts a grid000.nc carries and nothing here ever reads
        "adjacency": (("tri", "vertex"), big.triangles),
        "bc_transform": (("tri", "r", "c"), rng.random((2 * n, 3, 2))),
        "node_to_tri_map": (("node", "m"), rng.integers(0, n, (n, 10))),
    }).to_netcdf(fat)

    assert Path(ours).stat().st_size < fat.stat().st_size / 3
