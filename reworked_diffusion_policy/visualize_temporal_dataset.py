#!/usr/bin/env python3
"""Interactive viser viewer for batches from :class:`RLBenchTemporalH5Dataset`."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import viser
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "This script requires `viser`. Install it with `pip install viser`."
    ) from exc


def _prepare_imports() -> None:
    """Ensure package imports work when the script is launched as a file."""
    if __package__:
        return
    package_root = pathlib.Path(__file__).resolve().parents[1]
    root_str = str(package_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_prepare_imports()

from reworked_diffusion_policy.dataset import (  # noqa: E402  (import after path tweak)
    DatasetConfig,
    RLBenchTemporalH5Dataset,
    collate_temporal_batch,
)


def _split_point_cloud(tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points, colors) with colors encoded as uint8."""
    array = tensor.detach().cpu().numpy().astype(np.float32)
    if array.shape[1] >= 6:
        points = array[:, :3]
        colors = array[:, 3:6]
    else:
        points = array[:, :3]
        colors = None
    if colors is None:
        color_array = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    else:
        color_array = np.clip(colors, 0.0, 1.0)
        color_array = (color_array * 255.0).astype(np.uint8)
    return points.astype(np.float32), color_array


def _pose_from_tensor(tensor: torch.Tensor) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[float]]]:
    """Extract (position, quaternion wxyz, gripper) from a tensor, if possible."""
    array = tensor.detach().cpu().numpy().astype(np.float32)
    if array.shape[0] < 3:
        return None
    position = array[:3]
    if array.shape[0] >= 7:
        quat_xyzw = array[3:7]
        wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        norm = np.linalg.norm(wxyz)
        if norm > 1e-6:
            wxyz = wxyz / norm
        else:
            wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gripper = float(array[7]) if array.shape[0] >= 8 else None
    return position, wxyz, gripper


def _update_action_frames(
    frames: Sequence[viser.FrameHandle],
    actions: torch.Tensor,
) -> List[Optional[float]]:
    """Update viser frames with future gripper poses and return gripper values."""
    grip_vals: List[Optional[float]] = []
    horizon = actions.shape[0]
    for idx, frame in enumerate(frames):
        if idx < horizon:
            pose = _pose_from_tensor(actions[idx])
            if pose is None:
                frame.visible = False
                grip_vals.append(None)
                continue
            position, wxyz, gripper = pose
            frame.visible = True
            frame.position = position
            frame.wxyz = wxyz
            grip_vals.append(gripper)
        else:
            frame.visible = False
    return grip_vals


def _format_grippers(tag: str, values: Iterable[Optional[float]]) -> str:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return f"- {tag}: n/a"
    sample = ", ".join(f"{v:.3f}" for v in filtered[:6])
    suffix = "" if len(filtered) <= 6 else " …"
    return f"- {tag}: {sample}{suffix}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an RLBenchTemporalH5Dataset data loader and explore mini-batches "
            "interactively using viser."
        )
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to an .h5 file or root directory of cached tasks.")
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=[],
        help="Task name to include when dataset-path is a directory. Repeat for multiple tasks.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the data loader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--sample-points", type=int, default=4096, help="Number of points sampled per observation point cloud.")
    parser.add_argument("--n-obs-steps", type=int, default=2, help="Number of observation steps included per sample.")
    parser.add_argument("--action-horizon", type=int, default=16, help="Number of future actions per sample.")
    parser.add_argument("--no-point-colors", action="store_true", help="Disable point colors in the dataset sampler.")
    parser.add_argument("--point-size", type=float, default=0.003, help="Rendered point size inside viser.")
    parser.add_argument("--axes-length", type=float, default=0.08, help="Axis length for pose frames.")
    parser.add_argument("--axes-radius", type=float, default=0.003, help="Axis radius for pose frames.")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling when drawing batches from the loader.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin memory in the PyTorch data loader.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    dataset_cfg = DatasetConfig(
        path=args.dataset_path,
        sample_points=args.sample_points,
        n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon,
        use_point_colors=not args.no_point_colors,
        task_names=tuple(args.tasks) if args.tasks else None,
    )

    print("Loading dataset …")
    dataset = RLBenchTemporalH5Dataset(dataset_cfg)
    print(f"Loaded {len(dataset)} samples from {len(dataset.source_files)} cache file(s).")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=collate_temporal_batch,
    )

    loader_iter = iter(dataloader)
    batch_counter = 0
    current_batch: Optional[Dict[str, torch.Tensor]] = None

    server = viser.ViserServer()
    point_cloud = server.add_point_cloud(
        "/observation/point_cloud",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.full((1, 3), 200, dtype=np.uint8),
        point_size=args.point_size,
    )
    point_cloud.point_size = args.point_size

    agent_frame = server.scene.add_frame(
        "/observation/agent_pose",
        axes_length=args.axes_length,
        axes_radius=args.axes_radius,
        origin_radius=args.axes_radius * 2.5,
    )
    agent_frame.visible = False

    action_frames = [
        server.scene.add_frame(
            f"/future/action_{idx}",
            axes_length=args.axes_length,
            axes_radius=args.axes_radius,
            origin_radius=args.axes_radius * 2.0,
        )
        for idx in range(args.action_horizon)
    ]
    for frame in action_frames:
        frame.visible = False

    info_panel = server.gui.add_markdown("Waiting for batch …")
    batch_slider = server.gui.add_slider(
        "Sample index",
        min=0,
        max=args.batch_size - 1,
        step=1,
        initial_value=0,
        hint="Pick which element of the current batch to display.",
    )
    obs_slider = server.gui.add_slider(
        "Observation timestep",
        min=0,
        max=max(0, args.n_obs_steps - 1),
        step=1,
        initial_value=max(0, args.n_obs_steps - 1),
        hint="Toggle between temporal observation inputs.",
    )
    next_batch_btn = server.gui.add_button(
        "Next batch",
        hint="Fetch the next random batch from the data loader.",
    )

    def _slider_index(handle: viser.GuiSliderHandle[int | float]) -> Optional[int]:
        try:
            return int(handle.value)
        except (TypeError, ValueError):
            return None

    def update_visualization() -> None:
        nonlocal current_batch
        if current_batch is None:
            return
        batch = current_batch
        batch_size = batch["point_clouds"].shape[0]
        if batch_size == 0:
            return
        obs_steps = batch["point_clouds"].shape[1]
        max_obs_index = max(0, obs_steps - 1)
        slider_sample_idx = _slider_index(batch_slider)
        if slider_sample_idx is None:
            batch_slider.value = int(getattr(batch_slider, "min", 0))
            return
        slider_obs_idx = _slider_index(obs_slider)
        if slider_obs_idx is None:
            obs_slider.value = int(getattr(obs_slider, "min", 0))
            return
        sample_idx = min(slider_sample_idx, batch_size - 1)
        obs_idx = min(slider_obs_idx, max_obs_index)

        obs_slider.max = max_obs_index
        if obs_slider.value != obs_idx:
            obs_slider.value = int(obs_idx)
            return

        pc_tensor = batch["point_clouds"][sample_idx, obs_idx]
        points, colors = _split_point_cloud(pc_tensor)
        point_cloud.points = points
        point_cloud.colors = colors

        agent_tensor = batch["agent_pos"][sample_idx, obs_idx]
        agent_pose = _pose_from_tensor(agent_tensor)
        if agent_pose is None:
            agent_frame.visible = False
            agent_gripper = None
        else:
            position, wxyz, agent_gripper = agent_pose
            agent_frame.visible = True
            agent_frame.position = position
            agent_frame.wxyz = wxyz

        action_tensor = batch["action"][sample_idx]
        gripper_values = _update_action_frames(action_frames, action_tensor)

        info_lines = [
            f"**Batch {batch_counter}**",
            f"- Samples: {batch_size}",
            f"- Observation: {obs_idx + 1} / {obs_steps}",
            f"- Points: {points.shape[0]}",
        ]
        if agent_gripper is not None:
            info_lines.append(f"- Gripper (obs): {agent_gripper:.3f}")
        info_lines.append(_format_grippers("Gripper (future)", gripper_values))
        info_panel.content = "\n".join(info_lines)

    def load_next_batch() -> None:
        nonlocal loader_iter, current_batch, batch_counter
        try:
            raw_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            raw_batch = next(loader_iter)
        current_batch = {key: tensor.detach().cpu() for key, tensor in raw_batch.items()}
        batch_counter += 1
        batch_size = current_batch["point_clouds"].shape[0]
        batch_slider.max = max(0, batch_size - 1)
        target_idx = _slider_index(batch_slider)
        if target_idx is None:
            target_idx = 0
        target_idx = min(target_idx, batch_slider.max)
        if batch_slider.value != target_idx:
            batch_slider.value = int(target_idx)
        update_visualization()

    @batch_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        update_visualization()

    @obs_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        update_visualization()

    @next_batch_btn.on_click
    def _(event: viser.GuiEvent[viser.GuiButtonHandle]):
        if not next_batch_btn.value:
            return
        load_next_batch()
        next_batch_btn.value = False

    load_next_batch()
    print("Viser server running at:", server.get_host())

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping viewer …")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
