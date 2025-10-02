"""CLI utility to inspect RLBenchInContextDataset samples and visualize them with viser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tools.datasets.in_context import (
    RLBenchInContextDataset,
)


def _tensor_colors_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return array


def _resolve_variations(
    *,
    variations: Optional[Sequence[int]],
    variation_range: Optional[Sequence[int]],
) -> Optional[Sequence[int] | Tuple[int, int]]:
    if variation_range is not None:
        if len(variation_range) != 2:
            raise ValueError("variation_range expects exactly two integers")
        return (int(variation_range[0]), int(variation_range[1]))
    if variations:
        return [int(v) for v in variations]
    return None


def _combine_support_clouds(
    support_clouds: List[Dict[str, torch.Tensor]],
) -> Tuple[np.ndarray, np.ndarray]:
    points_list: List[np.ndarray] = []
    colors_list: List[np.ndarray] = []
    for cloud in support_clouds:
        points = cloud["points"].detach().cpu().numpy()
        colors = _tensor_colors_to_uint8(cloud["colors"])
        # Apply mask to filter out invalid points.
        masks = cloud["masks"].detach().cpu().numpy()
        points = points[masks]
        colors = colors[masks]
        points_list.append(points)
        colors_list.append(colors)
    if not points_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    points = np.concatenate(points_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)
    return points, colors


def _visualize_sample_with_viser(
    sample: Dict[str, object],
    *,
    point_size: float,
    support_point_size: float,
    future_pose_axes_length: float,
    future_pose_axes_radius: float,
) -> None:
    try:
        import viser
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "viser is required for visualization. Install with `pip install viser`."
        ) from exc

    query_cloud = sample["query"]["observation"]["point_cloud"]
    query_points = query_cloud["points"].detach().cpu().numpy()
    query_colors = _tensor_colors_to_uint8(query_cloud["colors"])
    
    # Apply mask to filter out invalid points.
    masks = query_cloud["masks"].detach().cpu().numpy()
    query_points = query_points[masks]
    query_colors = query_colors[masks]

    server = viser.ViserServer()
    query_node = server.add_point_cloud(
        "/query_point_cloud",
        points=query_points,
        colors=query_colors,
        point_size=point_size,
    )
    query_node.point_size = point_size

    # Visualize future gripper poses as frames.
    future_actions: torch.Tensor = sample["query"]["future_actions"]
    future_np = future_actions.detach().cpu().numpy()
    for idx, pose_vec in enumerate(future_np, start=1):
        pose = np.asarray(pose_vec, dtype=np.float32).ravel()
        if pose.size < 7:
            continue
        position = pose[:3]
        qx, qy, qz, qw = pose[3:7]
        wxyz = np.array([qw, qx, qy, qz], dtype=np.float32)
        norm = np.linalg.norm(wxyz)
        if norm == 0:
            continue
        frame = server.scene.add_frame(
            f"/future_pose_{idx}",
            axes_length=future_pose_axes_length,
            axes_radius=future_pose_axes_radius,
            origin_radius=future_pose_axes_radius * 2.5,
        )
        frame.position = position
        frame.wxyz = wxyz / norm

    # Add aggregated support point clouds and corresponding gripper pose frames.
    support_set = sample["support"]
    for support_idx, support in enumerate(support_set, start=1):
        support_points, support_colors = _combine_support_clouds(support["point_cloud_sequence"])
        node = server.add_point_cloud(
            f"/support_{support_idx}",
            points=support_points,
            colors=support_colors,
            point_size=support_point_size,
        )
        node.point_size = support_point_size

        pose_tensor: torch.Tensor = support["gripper_pose_sequence"]
        pose_np = pose_tensor.detach().cpu().numpy()
        for frame_idx, pose_vec in enumerate(pose_np):
            pose = np.asarray(pose_vec, dtype=np.float32).ravel()
            if pose.size < 7:
                continue
            position = pose[:3]
            qx, qy, qz, qw = pose[3:7]
            wxyz = np.array([qw, qx, qy, qz], dtype=np.float32)
            norm = np.linalg.norm(wxyz)
            if norm == 0:
                continue
            support_frame = server.scene.add_frame(
                f"/support_{support_idx}_pose_{frame_idx}",
                axes_length=future_pose_axes_length * 0.75,
                axes_radius=future_pose_axes_radius * 0.75,
                origin_radius=future_pose_axes_radius * 1.5,
            )
            support_frame.position = position
            support_frame.wxyz = wxyz / norm

    query_meta = sample["query"]["meta"]
    print(
        "Loaded query — task: {task}, variation: {variation}, episode: {episode}".format(
            task=query_meta["task"],
            variation=query_meta["variation"],
            episode=query_meta["episode"],
        )
    )
    print(f"Future poses visualized: {len(future_np)}")
    print("Support episodes:")
    for support_idx, support in enumerate(support_set, start=1):
        meta = support["meta"]
        indices = meta["indices"]
        print(
            f"  Support #{support_idx}: variation={meta['variation']}, episode={meta['episode']}, "
            f"frames={len(indices)}"
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


def _parse_support_range(arg: Optional[Sequence[int]]) -> Tuple[int, int]:
    if arg is None:
        return (1, 4)
    if len(arg) != 2:
        raise ValueError("support-size-range expects exactly two integers")
    start, end = int(arg[0]), int(arg[1])
    if start < 1 or start > end:
        raise ValueError("support-size-range must satisfy 1 <= start <= end")
    return start, end


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an RLBench in-context imitation sample: visualize the query "
            "point cloud, future gripper poses, and supporting few-shot episodes."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output_data"),
        help="Root directory containing RLBench exports.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="reach_target",
        help="Task name inside the root directory.",
    )
    parser.add_argument(
        "--variations",
        type=int,
        nargs="*",
        help="Explicit list of variation ids to include (default: all).",
    )
    parser.add_argument(
        "--variation-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive variation id range to include.",
    )
    parser.add_argument(
        "--support-size-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Inclusive range for the number of support episodes to draw.",
    )
    parser.add_argument(
        "--support-frames",
        type=int,
        default=7,
        help="Number of timesteps retained per support episode after subsampling.",
    )
    parser.add_argument(
        "--query-future-steps",
        type=int,
        default=7,
        help="Number of future gripper poses to visualise for the query episode.",
    )
    parser.add_argument(
        "--query-future-stride",
        type=int,
        default=10,
        help="Stride (in raw RLBench steps) between future gripper poses (default samples at ~1/10 speed).",
    )
    parser.add_argument(
        "--match-variation",
        action="store_true",
        help="Restrict support episodes to share the query's variation id.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index into the dataset after filtering by variations.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.002,
        help="Point size for the query point cloud.",
    )
    parser.add_argument(
        "--support-point-size",
        type=float,
        default=0.0015,
        help="Point size used when rendering aggregated support clouds.",
    )
    parser.add_argument(
        "--future-frame-axes-length",
        type=float,
        default=0.1,
        help="Axes length for future pose frames in viser.",
    )
    parser.add_argument(
        "--future-frame-axes-radius",
        type=float,
        default=0.004,
        help="Axes radius for future pose frames in viser.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=8,
        help="Number of demos to memoise in the in-memory cache.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed controlling support episode sampling.",
    )

    args = parser.parse_args()

    variation_arg = _resolve_variations(
        variations=args.variations, variation_range=args.variation_range
    )
    support_range = _parse_support_range(args.support_size_range)

    try:
        dataset = RLBenchInContextDataset(
            root=args.root,
            task=args.task,
            variations=variation_arg,
            support_size_range=support_range,
            support_frames=args.support_frames,
            query_future_steps=args.query_future_steps,
            query_future_stride=args.query_future_stride,
            match_variation=args.match_variation,
            cache_size=args.cache_size,
            seed=args.seed,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading RLBench pickle files requires RLBench's dependencies. "
            "Ensure packages such as `gymnasium` and `pyrep` are installed."
        ) from exc

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Ensure the configuration leaves enough data after subsampling."
        )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {args.sample_index} is out of bounds for dataset of size {len(dataset)}"
        )

    sample = dataset[args.sample_index]

    # Attach task string explicitly in meta for clarity.
    sample["query"]["meta"]["task"] = args.task

    _visualize_sample_with_viser(
        sample,
        point_size=args.point_size,
        support_point_size=args.support_point_size,
        future_pose_axes_length=args.future_frame_axes_length,
        future_pose_axes_radius=args.future_frame_axes_radius,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    _main()
