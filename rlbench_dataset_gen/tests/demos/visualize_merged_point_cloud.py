"""Utility to play back merged point clouds from an RLBench demo using viser."""

import argparse
import glob
import os
import pickle
import time
from typing import List, Tuple

import numpy as np

try:
    import viser
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "viser is required for visualization. Install with `pip install viser`."
    ) from exc

from rlbench.backend.const import MERGED_POINT_CLOUD_FOLDER, LOW_DIM_PICKLE


def _collect_frame_paths(episode_path: str) -> List[Tuple[int, str]]:
    merged_dir = os.path.join(episode_path, MERGED_POINT_CLOUD_FOLDER)
    if not os.path.isdir(merged_dir):
        raise FileNotFoundError(
            f"Merged point cloud directory not found: {merged_dir}. "
            "Make sure dataset_generator_pc.py has been run with merged dumps enabled."
        )

    files = sorted(
        glob.glob(os.path.join(merged_dir, '*.npz')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )
    if not files:
        raise FileNotFoundError(
            f"No merged point cloud frames found in {merged_dir}."
        )

    return [
        (int(os.path.splitext(os.path.basename(path))[0]), path)
        for path in files
    ]


def _load_gripper_poses(episode_path: str) -> dict[int, Tuple[np.ndarray, np.ndarray]]:
    low_dim_path = os.path.join(episode_path, LOW_DIM_PICKLE)
    if not os.path.isfile(low_dim_path):
        print('Warning: low-dim observation file not found. Skipping gripper visualization.')
        return {}

    with open(low_dim_path, 'rb') as f:
        demo = pickle.load(f)

    poses: dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, obs in enumerate(demo):
        pose = getattr(obs, 'gripper_pose', None)
        if pose is None:
            continue
        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_arr.size != 7:
            raise ValueError(
                f'Expected gripper_pose of length 7 at step {idx}, got {pose_arr.size}.'
            )
        position = pose_arr[:3]
        qx, qy, qz, qw = pose_arr[3:]
        wxyz = np.array([qw, qx, qy, qz], dtype=np.float32)
        norm = np.linalg.norm(wxyz)
        if norm == 0:
            continue
        poses[idx] = (position, wxyz / norm)
    if not poses:
        print('Warning: No gripper poses found in low-dim data.')
    return poses


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualize RLBench merged point clouds over time with viser.'
    )
    parser.add_argument(
        'episode_path',
        type=str,
        help='Path to a single RLBench episode folder (e.g. .../episodes/episode0).'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=10.0,
        help='Playback speed in frames per second. Set <=0 to step on enter.',
    )
    parser.add_argument(
        '--point_size',
        type=float,
        default=0.002,
        help='Size of rendered points inside viser.',
    )
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Replay the sequence continuously until interrupted.',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.episode_path):
        raise FileNotFoundError(f'Episode path does not exist: {args.episode_path}')

    frame_paths = _collect_frame_paths(args.episode_path)
    gripper_poses = _load_gripper_poses(args.episode_path)

    server = viser.ViserServer()
    cloud = server.add_point_cloud(
        '/merged_point_cloud',
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        point_size=args.point_size,
    )

    gripper_frame = None
    if gripper_poses:
        gripper_frame = server.scene.add_frame(
            '/gripper_pose',
            axes_length=0.1,
            axes_radius=0.004,
            origin_radius=0.01,
        )

    delay = 0.0 if args.fps <= 0 else 1.0 / args.fps

    print('Viser server running at:', server.get_host())
    print('Press Ctrl+C to stop playback.')

    total_frames = len(frame_paths)
    try:
        while True:
            for step_idx, path in frame_paths:
                with np.load(path) as data:
                    points = np.asarray(data['points'], dtype=np.float32)
                    colors = np.asarray(data['colors'], dtype=np.uint8)
                    masks = (
                        np.asarray(data['masks'], dtype=np.int32)
                        if 'masks' in data else None
                    )
                if points.ndim != 2 or points.shape[1] != 3:
                    raise ValueError(f"Unexpected point cloud shape in {path}: {points.shape}")
                if colors.shape != points.shape:
                    raise ValueError(
                        f"Color array shape {colors.shape} does not match point shape {points.shape} in {path}."
                    )
                if masks is not None and masks.shape[0] != points.shape[0]:
                    raise ValueError(
                        f"Mask array length {masks.shape[0]} does not match point count {points.shape[0]} in {path}."
                    )
                cloud.points = points
                cloud.colors = colors
                cloud.point_size = args.point_size
                if gripper_frame is not None:
                    pose = gripper_poses.get(step_idx)
                    if pose is None:
                        gripper_frame.visible = False
                    else:
                        position, wxyz = pose
                        gripper_frame.position = position
                        gripper_frame.wxyz = wxyz
                        gripper_frame.visible = True
                if masks is not None:
                    unique_ids = np.unique(masks)
                    print(
                        f'Showing frame {step_idx} / {frame_paths[-1][0]} '
                        f'({total_frames} frames total) — {len(unique_ids)} unique mask ids'
                    )
                else:
                    print(
                        f'Showing frame {step_idx} / {frame_paths[-1][0]} '
                        f'({total_frames} frames total)'
                    )
                if delay == 0.0:
                    input('Press Enter for next frame...')
                else:
                    time.sleep(delay)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass

    print('Stopping viewer...')
    server.stop()


if __name__ == '__main__':
    main()
