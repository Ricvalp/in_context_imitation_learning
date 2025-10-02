#!/usr/bin/env python3
"""Precompute and cache the RLBench temporal point-cloud dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.datasets.imitation_learning import (
    TemporalPointCloudCachedDataset,
    prepare_temporal_point_cloud_cache,
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

    parser.add_argument("--point-cloud-history", type=int, default=1, help="Number of previous point-cloud frames to include.")
    parser.add_argument("--proprio-history", type=int, default=1, help="Number of previous proprio frames to include.")
    parser.add_argument("--future-action-window", type=int, default=4, help="Number of future actions to expose.")
    parser.add_argument(
        "--future-action-stride",
        type=int,
        default=1,
        help="Stride between future actions in raw timesteps.",
    )
    parser.add_argument("--proprio-keys", nargs="*", default=["gripper_pose", "gripper_open"], help="Proprioceptive keys to include.")
    parser.add_argument("--action-keys", nargs="*", default=["gripper_pose", "gripper_open"], help="Action keys to include.")
    parser.add_argument("--cache-size", type=int, default=16, help="In-memory demo cache size during preprocessing.")
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
        cache_path = prepare_temporal_point_cloud_cache(
            root=args.root,
            task=task_name,
            variations=variations,
            point_cloud_history=args.point_cloud_history,
            proprio_history=args.proprio_history,
            future_action_window=args.future_action_window,
            future_action_stride=args.future_action_stride,
            proprio_keys=tuple(args.proprio_keys),
            action_keys=tuple(args.action_keys),
            cache_size=args.cache_size,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild,
            progress=not args.no_progress,
        )

        dataset = TemporalPointCloudCachedDataset(cache_path)
        print(f"[{task_name}] cached dataset ready at {cache_path} with {len(dataset)} samples.")


if __name__ == "__main__":
    main()
