#!/usr/bin/env python3
"""Precompute and cache the RLBench in-context imitation learning dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.datasets.in_context import (
    InContextCachedDataset,
    prepare_in_context_point_cloud_cache,
)


def _parse_variations(args: argparse.Namespace) -> Optional[List[int] | tuple[int, int]]:
    if args.variation_range is not None:
        start, end = args.variation_range
        return (start, end)
    if args.variations:
        return [int(v) for v in args.variations]
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output_data"), help="Root directory containing RLBench exports.")
    parser.add_argument("--task", help="Single task name to preprocess (e.g. 'open_wine_bottle').")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="List of task names to preprocess. Overrides --task if provided.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--variations", nargs="*", type=int, help="Explicit variation ids to include.")
    group.add_argument("--variation-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive variation range to include.")

    parser.add_argument("--support-size-range", nargs=2, type=int, default=[1, 4], metavar=("MIN", "MAX"), help="Inclusive support set size.")
    parser.add_argument("--support-frames", type=int, default=7, help="Number of subsampled frames for each support episode.")
    parser.add_argument("--query-future-steps", type=int, default=7, help="Number of future actions for the query episode.")
    parser.add_argument("--query-future-stride", type=int, default=10, help="Stride between future actions in raw timesteps.")
    parser.add_argument("--proprio-keys", nargs="*", default=["gripper_pose", "gripper_open"], help="Proprioceptive keys to include.")
    parser.add_argument("--action-keys", nargs="*", default=["gripper_pose", "gripper_open"], help="Action keys to include in the query future sequence.")
    parser.add_argument("--match-variation", action="store_true", help="Restrict support sampling to the query variation.")
    parser.add_argument("--cache-size", type=int, default=16, help="In-memory demo cache size during preprocessing.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic support sampling.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory to place the generated cache (defaults to <root>/.rlbench_cache).")
    parser.add_argument("--rebuild", action="store_true", help="Force regeneration even if the cache already exists.")
    parser.add_argument("--no-progress", action="store_true", help="Suppress periodic progress logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variations = _parse_variations(args)

    task_list: List[str]
    if args.tasks:
        task_list = args.tasks
    elif args.task:
        task_list = [args.task]
    else:
        raise SystemExit("Please specify at least one task via --task or --tasks.")

    for task_name in task_list:
        cache_path = prepare_in_context_point_cloud_cache(
            root=args.root,
            task=task_name,
            variations=variations,
            support_size_range=tuple(args.support_size_range),
            support_frames=args.support_frames,
            query_future_steps=args.query_future_steps,
            query_future_stride=args.query_future_stride,
            proprio_keys=tuple(args.proprio_keys),
            action_keys=tuple(args.action_keys),
            match_variation=args.match_variation,
            cache_size=args.cache_size,
            seed=args.seed,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild,
            progress=not args.no_progress,
        )

        dataset = InContextCachedDataset(cache_path)
        print(f"[{task_name}] cached dataset ready at {cache_path} with {len(dataset)} query episodes.")


if __name__ == "__main__":
    main()
