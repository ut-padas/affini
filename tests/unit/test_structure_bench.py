"""The structure bench (src/structure/bench.py): where it caches what."""

from src.structure import bench


def test_slice_directions_are_cached_in_one_shared_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "CACHE", tmp_path)
    W = bench.directions("rosenbrock", 3)
    assert W.shape == (3, 2)
    assert [p.name for p in tmp_path.iterdir()] == ["directions"]
    assert [p.name for p in (tmp_path / "directions").iterdir()] == ["rosenbrock_d3.npz"]
