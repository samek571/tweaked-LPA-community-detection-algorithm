from __future__ import annotations

"""High-level programmatic API for LPKit.

This module exposes a single convenience wrapper (`stream_lpa`) runnig full pipeline:
edgelist -> symmetrize/sort -> split blocks -> init labels -> streaming sweeps

it exists so callers do not need to coordinate paths unless control is desired
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .stream import (
    symmetrize_and_sort,
    split_sorted_sym_to_blocks,
    init_labels_memmap,
    stream_multi_sweep_parallel_blocks,
)


def stream_lpa(
    in_path: str | Path,
    out_path: str | Path,
    *,
    work_dir: str | Path | None = None,
    block_size: int = 100_000,
    seed: int = 1337,
    max_sweeps: int = 50,
    min_sweeps: int = 1,
    tie_break: str = "min",
    workers: int = 1,
) -> dict[str, Any]:
    """Run the full streaming LPA pipeline in one call.

    Parameters
    ----------
    in_path:
        Input edgelist path. Expected format: two whitespace-separated integer
        vertex ids per line (`u v`). Comment handling is delegated to the
        lower-level streaming parser.
    out_path:
        Destination `.npy` labels memmap path.
    work_dir:
        Directory for intermediate artifacts (`.sorted.sym`, `.blocks/`). If
        omitted, a temporary directory is used and removed before returning.
    block_size:
        Approximate number of source-vertex-range edges processed per block.
        This is the primary RAM/throughput tradeoff knob.
    seed, max_sweeps, min_sweeps, tie_break, workers:
        Forwarded to the streaming propagation routine.

    Returns
    -------
    dict
        The result info emitted by the streaming sweep function plus pipeline
        metadata (`n`, `block_count`, `labels_path`, etc.).

        If `work_dir` is None, temporary artifacts are deleted before return and
        the result reports `artifacts_persistent=False` without transient paths.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input edgelist not found: {in_path}")

    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be > 0")
    if min_sweeps <= 0:
        raise ValueError("min_sweeps must be > 0")
    if min_sweeps > max_sweeps:
        raise ValueError("min_sweeps must be <= max_sweeps")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _run(base: Path) -> dict[str, Any]:
        """Execute the streaming pipeline using `base` for intermediates."""
        base.mkdir(parents=True, exist_ok=True)

        stem = in_path.stem
        sorted_sym = base / f"{stem}.sorted.sym"
        blocks_dir = base / f"{stem}.blocks"

        meta = symmetrize_and_sort(str(in_path), str(sorted_sym))
        n = int(meta["n"])

        block_paths = split_sorted_sym_to_blocks(
            str(sorted_sym),
            n=n,
            block_size=int(block_size),
            out_dir=str(blocks_dir),
        )

        init_labels_memmap(str(out_path), n=n)

        info = stream_multi_sweep_parallel_blocks(
            block_paths,
            str(out_path),
            n=n,
            block_size=int(block_size),
            seed=int(seed),
            max_sweeps=int(max_sweeps),
            min_sweeps=int(min_sweeps),
            tie_break=tie_break,
            workers=int(workers),
        )

        result = dict(info)
        result.update(
            {
                "input_path": str(in_path),
                "labels_path": str(out_path),
                "sorted_sym_path": str(sorted_sym),
                "blocks_dir": str(blocks_dir),
                "n": n,
                "block_count": len(block_paths),
                "block_size": int(block_size),
            }
        )
        return result

    if work_dir is not None:
        result = _run(Path(work_dir))
        result["artifacts_persistent"] = True
        return result

    #tmp mode for scripts that care only about labels.npy
    with TemporaryDirectory(prefix="lpkit_stream_") as tmp:
        result = _run(Path(tmp))
        #paths become invalid after the context exits, no exposure
        result["artifacts_persistent"] = False
        result.pop("sorted_sym_path", None)
        result.pop("blocks_dir", None)
        return result
