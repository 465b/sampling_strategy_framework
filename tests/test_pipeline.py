

# --- hindcast file ordering -------------------------------------------------

def test_hindcast_files_sort_numerically_within_a_directory():
    """`schout_10.nc` is later than `schout_2.nc`; lexical order says otherwise."""
    from edna_sampling.pipeline import _time_index

    files = ["/h/2018/01/schout_10.nc", "/h/2018/01/schout_2.nc", "/h/2018/01/schout_1.nc"]
    assert sorted(files, key=_time_index) == [
        "/h/2018/01/schout_1.nc", "/h/2018/01/schout_2.nc", "/h/2018/01/schout_10.nc"]


def test_hindcast_files_sort_by_directory_before_file_number():
    """SCHISM output is split into per-month directories whose numbering restarts
    at 1, so every month has a schout_1.nc. Sorting on the filename alone would
    interleave the months - and the flooded-fraction mask is computed over
    exactly this ordering, so it would be wrong without saying so."""
    from edna_sampling.pipeline import _time_index

    files = ["/h/2018/02/schout_1.nc", "/h/2018/01/schout_10.nc",
             "/h/2018/10/schout_1.nc", "/h/2018/01/schout_1.nc"]
    assert sorted(files, key=_time_index) == [
        "/h/2018/01/schout_1.nc", "/h/2018/01/schout_10.nc",
        "/h/2018/02/schout_1.nc", "/h/2018/10/schout_1.nc"]


def test_the_sort_key_is_totally_ordered_for_odd_paths():
    """Mixing numbers and text in a sort key raises TypeError if the shapes
    differ; the key pads each element so unlike paths still compare."""
    from edna_sampling.pipeline import _time_index

    odd = ["/h/schout.nc", "/h/2018/schout_3.nc", "/h/a/b/c.nc", "/h/9/9/9.nc"]
    sorted(odd, key=_time_index)          # must not raise
