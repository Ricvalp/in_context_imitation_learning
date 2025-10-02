#!/usr/bin/env python3
"""Visualize samples from a cached RLBench temporal dataset using viser."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

try:
    import viser
except ImportError as exc:  # pragma: no cover - CLI guard
    raise ImportError(
        "This demo requires the `viser` package. Install it with `pip install viser`."
    ) from exc

from tools.datasets.imitation_learning import TemporalPointCloudCachedDataset


def _tensor_to_uint8_colors(tensor: torch.Tensor) -> np.ndarray:
    colors = tensor.detach().cpu().numpy()
    colors = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    return colors


def _visualize_sample(sample: dict, *, point_size: float, pose_axes: float, pose_radius: float) -> None:
    cloud_sequence = sample["observation"]["point_cloud_sequence"]
    current_cloud = cloud_sequence[-1]
    points = current_cloud["points"].detach().cpu().numpy()
    colors = _tensor_to_uint8_colors(current_cloud["colors"])
    mask = current_cloud["masks"].detach().cpu().numpy()

    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    points = points[mask]
    colors = colors[mask]

    server = viser.ViserServer()
    cloud = server.add_point_cloud(
        "/observation",
        points=points,
        colors=colors,
        point_size=point_size,
    )
    cloud.point_size = point_size

    future_actions: torch.Tensor = sample["action"]
    for idx, pose_vec in enumerate(future_actions.detach().cpu().numpy(), start=1):
        pose = np.asarray(pose_vec, dtype=np.float32).ravel()
        if pose.size < 7:
            continue
        position = pose[:3]
        qx, qy, qz, qw = pose[3:7]
        wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float32)
        if np.linalg.norm(wxyz) == 0:
            continue
        frame = server.scene.add_frame(
            f"/future_pose_{idx}",
            axes_length=pose_axes,
            axes_radius=pose_radius,
            origin_radius=pose_radius * 2.5,
        )
        frame.position = position
        frame.wxyz = wxyz / np.linalg.norm(wxyz)

    meta = sample["meta"]
    print(
        "Loaded cache sample — task: {task}, variation: {variation}, episode: {episode}, step: {step}".format(
            task=meta["task"],
            variation=meta["variation"],
            episode=meta["episode"],
            step=meta["step"],
        )
    )
    print("Viser server running at:", server.get_host())
    print("Press Ctrl+C to exit the viewer.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viewer...")
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/davidknigge/hard-disk2/storage/robotics/rlbench_20tasks_100episodes_preprocessed/temporal"),
        help="Root directory containing cached temporal datasets (defaults to preprocessed repo).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="reach_target",
        help="Task subdirectory to load from the cache root.",
    )
    parser.add_argument(
        "--cache-key",
        type=str,
        default=None,
        help="Optional cache filename (without path). Defaults to the first .h5 file in the task directory.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to visualize.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.01,
        help="Point size for the point cloud visualization.",
    )
    parser.add_argument(
        "--pose-axes",
        type=float,
        default=0.05,
        help="Axes length for future end-effector poses.",
    )
    parser.add_argument(
        "--pose-radius",
        type=float,
        default=0.008,
        help="Axes radius for future end-effector poses.",
    )
    args = parser.parse_args()

    task_dir = args.root / args.task
    if not task_dir.is_dir():
        available_tasks = ", ".join(sorted(p.name for p in args.root.iterdir() if p.is_dir()))
        raise FileNotFoundError(
            f"Task directory '{task_dir}' not found. Available tasks: {available_tasks}"
        )

    if args.cache_key:
        cache_path = task_dir / args.cache_key
    else:
        h5_files = sorted(task_dir.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found under {task_dir}")
        cache_path = h5_files[0]

    if cache_path.is_dir():
        h5_files = sorted(cache_path.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found under {cache_path}")
        cache_path = h5_files[0]
    print(f"Using cache file: {cache_path}")

    dataset = TemporalPointCloudCachedDataset(cache_path)
    if not (0 <= args.index < len(dataset)):
        raise IndexError(f"Index {args.index} out of bounds for dataset of length {len(dataset)}")

    sample = dataset[args.index]
    _visualize_sample(
        sample,
        point_size=args.point_size,
        pose_axes=args.pose_axes,
        pose_radius=args.pose_radius,
    )


if __name__ == "__main__":
    main()
