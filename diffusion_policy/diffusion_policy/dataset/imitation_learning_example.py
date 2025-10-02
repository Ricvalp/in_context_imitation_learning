from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import time
import argparse
from pathlib import Path

from tools.datasets.imitation_learning import RLBenchTemporalPointCloudDataset


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
    

def _visualize_sample_with_viser(
    sample: Dict[str, object],
    *,
    point_size: float,
    future_pose_axes_length: float,
    future_pose_axes_radius: float,
) -> None:
    try:
        import viser
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "viser is required for visualization. Install with `pip install viser`."
        ) from exc

    cloud_entry = sample["observation"]["point_cloud_sequence"][-1]
    points = cloud_entry["points"].detach().cpu().numpy()
    colors = _tensor_colors_to_uint8(cloud_entry["colors"])
    
    # Apply mask to filter out invalid points.
    masks = cloud_entry["masks"].detach().cpu().numpy()
    points = points[masks]
    colors = colors[masks]

    server = viser.ViserServer()
    cloud = server.add_point_cloud(
        "/observation",
        points=points,
        colors=colors,
        point_size=point_size,
    )
    cloud.point_size = point_size

    action_tensor: torch.Tensor = sample["action"]
    future_actions = action_tensor.detach().cpu().numpy()
    frames = []
    for idx, pose_vec in enumerate(future_actions, start=1):
        pose = np.asarray(pose_vec, dtype=np.float32).ravel()
        if pose.size < 7:
            raise ValueError(
                f"Expected pose vector with at least 7 elements, got {pose.size}"
            )
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
        frames.append(frame)

    meta = sample["meta"]
    print(
        "Loaded sample — task: {task}, variation: {variation}, episode: {episode}, step: {step}".format(
            task=meta["task"],
            variation=meta["variation"],
            episode=meta["episode"],
            step=meta["step"],
        )
    )
    print(f"Visualizing {len(future_actions)} future end-effector poses.")
    print("Viser server running at:", server.get_host())
    print("Press Ctrl+C to exit the viewer.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viewer...")
    finally:
        server.stop()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a single RLBench temporal sample and visualize its point "
            "cloud plus future end-effector poses using viser."
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
        "--point-cloud-history",
        type=int,
        default=0,
        help="Number of previous point-cloud frames to load (0 for current only).",
    )
    parser.add_argument(
        "--proprio-history",
        type=int,
        default=0,
        help="Number of previous proprioceptive frames to load.",
    )
    parser.add_argument(
        "--future-steps",
        type=int,
        default=16,
        help="Number of future end-effector poses to visualise.",
    )
    parser.add_argument(
        "--future-step-stride",
        type=int,
        default=1,
        help="Stride between future action steps when sampling actions.",
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
        help="Point size for the rendered point cloud.",
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
        help="Number of demos to keep in the in-memory cache.",
    )
    args = parser.parse_args()

    variation_arg = _resolve_variations(
        variations=args.variations, variation_range=args.variation_range
    )

    try:
        dataset = RLBenchTemporalPointCloudDataset(
            root=args.root,
            task=args.task,
            variations=variation_arg,
            point_cloud_history=args.point_cloud_history,
            proprio_history=args.proprio_history,
            future_action_window=args.future_steps,
            future_action_stride=args.future_step_stride,
            cache_size=args.cache_size,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading RLBench pickle files requires RLBench's dependencies. "
            "Ensure packages such as `gymnasium` and `pyrep` are installed."
        ) from exc

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Ensure the specified variations contain enough steps "
            "to provide the requested histories and future window."
        )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {args.sample_index} is out of bounds for dataset of size {len(dataset)}"
        )

    sample = dataset[args.sample_index]

    # Ensure the task label reflects the CLI argument when variations were filtered.
    sample["meta"]["task"] = args.task

    _visualize_sample_with_viser(
        sample,
        point_size=args.point_size,
        future_pose_axes_length=args.future_frame_axes_length,
        future_pose_axes_radius=args.future_frame_axes_radius,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    _main()
