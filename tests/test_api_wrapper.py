"""Integration tests for the public `stream_lpa(...)` API wrapper.

These tests validate wrapper semantics (explicit vs temporary work dirs),
basic parameter validation, and deterministic behavior on a tiny graph.
"""

import numpy as np
import pytest

from lpkit import stream_lpa
from lpkit.label_propagation import label_propagation


def groups_from_labels(labels):
    """Return a partition representation independent of concrete label IDs."""
    buckets = {}
    for i, lab in enumerate(labels):
        buckets.setdefault(int(lab), set()).add(i)
    return {frozenset(s) for s in buckets.values()}


def test_stream_lpa_wrapper_runs_with_explicit_workdir(tmp_path):
    """Explicit `work_dir` should preserve intermediate artifacts and return their paths."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")

    out = tmp_path / "labels.npy"
    work = tmp_path / "work"

    info = stream_lpa(
        raw,
        out,
        work_dir=work,
        block_size=3,
        seed=1337,
        max_sweeps=10,
        min_sweeps=1,
        tie_break="min",
        workers=1,
    )

    assert out.exists()
    assert info["n"] == 6
    assert info["block_size"] == 3
    assert "sweeps" in info
    assert "block_count" in info and info["block_count"] >= 1
    assert info["artifacts_persistent"] is True

    assert work.exists()
    assert "sorted_sym_path" in info
    assert "blocks_dir" in info
    assert (work / "g.sorted.sym").exists()
    assert (work / "g.blocks").exists()
    assert info["sorted_sym_path"] == str(work / "g.sorted.sym")
    assert info["blocks_dir"] == str(work / "g.blocks")

    mm = np.lib.format.open_memmap(out, mode="r")
    assert mm.shape == (6,)


def test_stream_lpa_wrapper_matches_ram_partition_on_two_cliques(tmp_path):
    """Wrapper pipeline should match the in-memory baseline partition on a deterministic toy graph."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")

    out = tmp_path / "labels.npy"
    work = tmp_path / "work"

    # RAM baseline
    adj = [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
    labels_ram, _ = label_propagation(adj, seed=1337)

    info = stream_lpa(
        raw,
        out,
        work_dir=work,
        block_size=6,
        seed=1337,
        max_sweeps=100,
        min_sweeps=1,
        tie_break="min",
        workers=1,
    )

    assert info["sweeps"] >= 1

    labels_stream = np.lib.format.open_memmap(out, mode="r")
    assert groups_from_labels(labels_ram) == groups_from_labels(labels_stream)


def test_stream_lpa_invalid_input_path(tmp_path):
    """Missing input file should raise a clear `FileNotFoundError`."""
    out = tmp_path / "labels.npy"
    out = tmp_path / "labels.npy"
    with pytest.raises(FileNotFoundError):
        stream_lpa(tmp_path / "does_not_exist.edgelist", out)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0},
        {"block_size": -1},
        {"max_sweeps": 0},
        {"max_sweeps": -1},
        {"min_sweeps": 0},
        {"min_sweeps": -1},
    ],
)
def test_stream_lpa_rejects_invalid_numeric_params(tmp_path, kwargs):
    """Basic non-positive numeric arguments should be rejected by the wrapper."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n")
    out = tmp_path / "labels.npy"

    with pytest.raises(ValueError):
        stream_lpa(raw, out, **kwargs)


def test_stream_lpa_rejects_min_sweeps_greater_than_max(tmp_path):
    """`min_sweeps` must not exceed `max_sweeps`."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n")
    out = tmp_path / "labels.npy"

    with pytest.raises(ValueError):
        stream_lpa(raw, out, min_sweeps=10, max_sweeps=5)


def test_stream_lpa_temp_workdir_still_produces_output_and_is_honest_about_artifacts(tmp_path):
    """Temp mode should produce labels but must not claim intermediate artifacts persist."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    out = tmp_path / "labels.npy"

    info = stream_lpa(
        raw,
        out,
        work_dir=None,
        block_size=3,
        seed=1337,
        max_sweeps=10,
        workers=1,
        tie_break="min",
    )

    assert out.exists()
    mm = np.lib.format.open_memmap(out, mode="r")
    assert mm.shape == (6,)

    # temp artifacts are deleted after return; API should not claim persistent paths
    assert info["artifacts_persistent"] is False
    assert "sorted_sym_path" not in info
    assert "blocks_dir" not in info


def test_stream_lpa_deterministic_on_toy_graph(tmp_path):
    """With fixed seed/tie-break/workers, wrapper output should be reproducible on a toy graph."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")

    out1 = tmp_path / "labels1.npy"
    out2 = tmp_path / "labels2.npy"

    stream_lpa(
        raw,
        out1,
        work_dir=tmp_path / "work1",
        block_size=3,
        seed=1337,
        max_sweeps=50,
        workers=1,
        tie_break="min",
    )
    stream_lpa(
        raw,
        out2,
        work_dir=tmp_path / "work2",
        block_size=3,
        seed=1337,
        max_sweeps=50,
        workers=1,
        tie_break="min",
    )

    a = np.lib.format.open_memmap(out1, mode="r")
    b = np.lib.format.open_memmap(out2, mode="r")

    assert groups_from_labels(a) == groups_from_labels(b)
