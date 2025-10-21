#!/usr/bin/env python3
"""Generate a custom RLBench dataset for in-context imitation experiments.

This script collects demonstrations for the customised PutItemInDrawer task
where the target object is one of a curated set (banana, carrot, frying pan,
small container) and additional distractors appear according to predefined
scenarios. It writes the results using the same folder structure as the
standard RLBench dataset generator, but the variation directories are named
after the scenario labels (e.g. 'train-banana').
"""

from __future__ import annotations

import argparse
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_RLBENCH = REPO_ROOT / "rlbench_dataset_gen"
if str(BUNDLED_RLBENCH) not in sys.path:
    sys.path.insert(0, str(BUNDLED_RLBENCH))

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend import const as _const
from rlbench.environment import Environment

try:  # RLBench >= 1.2.0 (bundled segmentation utils)
    from rlbench.segmentation_utils import (
        DEFAULT_MAP_FILENAME,
        build_handle_label_map,
        collect_mask_handles,
        write_label_map,
    )
except ModuleNotFoundError:  # Older releases ship these under tools
    from rlbench.tools.segmentation_utils import (  # type: ignore
        DEFAULT_MAP_FILENAME,
        build_handle_label_map,
        collect_mask_handles,
        write_label_map,
    )

from rlbench.tasks.put_item_in_drawer import PutItemInDrawer

# ---------------------------------------------------------------------------
# Demo saving utilities (lifted from rlbench.dataset_generator_pc) ----------


DEPTH_SCALE = _const.DEPTH_SCALE
EPISODE_FOLDER = _const.EPISODE_FOLDER
EPISODES_FOLDER = _const.EPISODES_FOLDER
FRONT_DEPTH_FOLDER = _const.FRONT_DEPTH_FOLDER
FRONT_MASK_FOLDER = _const.FRONT_MASK_FOLDER
FRONT_RGB_FOLDER = _const.FRONT_RGB_FOLDER
IMAGE_FORMAT = _const.IMAGE_FORMAT
LEFT_SHOULDER_DEPTH_FOLDER = _const.LEFT_SHOULDER_DEPTH_FOLDER
LEFT_SHOULDER_MASK_FOLDER = _const.LEFT_SHOULDER_MASK_FOLDER
LEFT_SHOULDER_RGB_FOLDER = _const.LEFT_SHOULDER_RGB_FOLDER
LOW_DIM_PICKLE = _const.LOW_DIM_PICKLE
MERGED_POINT_CLOUD_FOLDER = getattr(_const, "MERGED_POINT_CLOUD_FOLDER", "merged_point_cloud")
OVERHEAD_DEPTH_FOLDER = _const.OVERHEAD_DEPTH_FOLDER
OVERHEAD_MASK_FOLDER = _const.OVERHEAD_MASK_FOLDER
OVERHEAD_RGB_FOLDER = _const.OVERHEAD_RGB_FOLDER
RIGHT_SHOULDER_DEPTH_FOLDER = _const.RIGHT_SHOULDER_DEPTH_FOLDER
RIGHT_SHOULDER_MASK_FOLDER = _const.RIGHT_SHOULDER_MASK_FOLDER
RIGHT_SHOULDER_RGB_FOLDER = _const.RIGHT_SHOULDER_RGB_FOLDER
VARIATION_DESCRIPTIONS = _const.VARIATION_DESCRIPTIONS
WRIST_DEPTH_FOLDER = _const.WRIST_DEPTH_FOLDER
WRIST_MASK_FOLDER = _const.WRIST_MASK_FOLDER
WRIST_RGB_FOLDER = _const.WRIST_RGB_FOLDER


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_demo(demo, example_path: Path) -> None:
    from PIL import Image

    left_shoulder_rgb_path = example_path / LEFT_SHOULDER_RGB_FOLDER
    left_shoulder_depth_path = example_path / LEFT_SHOULDER_DEPTH_FOLDER
    left_shoulder_mask_path = example_path / LEFT_SHOULDER_MASK_FOLDER
    right_shoulder_rgb_path = example_path / RIGHT_SHOULDER_RGB_FOLDER
    right_shoulder_depth_path = example_path / RIGHT_SHOULDER_DEPTH_FOLDER
    right_shoulder_mask_path = example_path / RIGHT_SHOULDER_MASK_FOLDER
    overhead_rgb_path = example_path / OVERHEAD_RGB_FOLDER
    overhead_depth_path = example_path / OVERHEAD_DEPTH_FOLDER
    overhead_mask_path = example_path / OVERHEAD_MASK_FOLDER
    wrist_rgb_path = example_path / WRIST_RGB_FOLDER
    wrist_depth_path = example_path / WRIST_DEPTH_FOLDER
    wrist_mask_path = example_path / WRIST_MASK_FOLDER
    front_rgb_path = example_path / FRONT_RGB_FOLDER
    front_depth_path = example_path / FRONT_DEPTH_FOLDER
    front_mask_path = example_path / FRONT_MASK_FOLDER
    merged_point_cloud_path = example_path / MERGED_POINT_CLOUD_FOLDER

    for directory in (
        left_shoulder_rgb_path,
        left_shoulder_depth_path,
        left_shoulder_mask_path,
        right_shoulder_rgb_path,
        right_shoulder_depth_path,
        right_shoulder_mask_path,
        overhead_rgb_path,
        overhead_depth_path,
        overhead_mask_path,
        wrist_rgb_path,
        wrist_depth_path,
        wrist_mask_path,
        front_rgb_path,
        front_depth_path,
        front_mask_path,
        merged_point_cloud_path,
    ):
        _ensure_dir(directory)

    camera_names = ["left_shoulder", "right_shoulder", "overhead", "wrist", "front"]

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE
        )
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8)
        )
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE
        )
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8)
        )
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE
        )
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8)
        )
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE
        )
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE
        )
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(left_shoulder_rgb_path / (IMAGE_FORMAT % i))
        left_shoulder_depth.save(left_shoulder_depth_path / (IMAGE_FORMAT % i))
        left_shoulder_mask.save(left_shoulder_mask_path / (IMAGE_FORMAT % i))
        right_shoulder_rgb.save(right_shoulder_rgb_path / (IMAGE_FORMAT % i))
        right_shoulder_depth.save(right_shoulder_depth_path / (IMAGE_FORMAT % i))
        right_shoulder_mask.save(right_shoulder_mask_path / (IMAGE_FORMAT % i))
        overhead_rgb.save(overhead_rgb_path / (IMAGE_FORMAT % i))
        overhead_depth.save(overhead_depth_path / (IMAGE_FORMAT % i))
        overhead_mask.save(overhead_mask_path / (IMAGE_FORMAT % i))
        wrist_rgb.save(wrist_rgb_path / (IMAGE_FORMAT % i))
        wrist_depth.save(wrist_depth_path / (IMAGE_FORMAT % i))
        wrist_mask.save(wrist_mask_path / (IMAGE_FORMAT % i))
        front_rgb.save(front_rgb_path / (IMAGE_FORMAT % i))
        front_depth.save(front_depth_path / (IMAGE_FORMAT % i))
        front_mask.save(front_mask_path / (IMAGE_FORMAT % i))

        merged_points = []
        merged_colors = []
        merged_masks = []
        for name in camera_names:
            pc = getattr(obs, f"{name}_point_cloud", None)
            rgb = getattr(obs, f"{name}_rgb", None)
            mask_arr = getattr(obs, f"{name}_mask", None)
            if pc is None or rgb is None or mask_arr is None:
                continue
            pc = np.asarray(pc).reshape(-1, 3)
            rgb = np.asarray(rgb).reshape(-1, 3)
            mask_arr = np.asarray(mask_arr).reshape(-1)
            valid = np.isfinite(pc).all(axis=1)
            if not np.all(valid):
                pc = pc[valid]
                rgb = rgb[valid]
                mask_arr = mask_arr[valid]
            if pc.size == 0:
                continue
            merged_points.append(pc.astype(np.float32))
            merged_colors.append(rgb.astype(np.uint8))
            merged_masks.append(mask_arr.astype(np.int32))

        if merged_points:
            merged_points = np.concatenate(merged_points, axis=0)
            merged_colors = np.concatenate(merged_colors, axis=0)
            merged_masks = np.concatenate(merged_masks, axis=0)
        else:
            merged_points = np.empty((0, 3), dtype=np.float32)
            merged_colors = np.empty((0, 3), dtype=np.uint8)
            merged_masks = np.empty((0,), dtype=np.int32)

        np.savez_compressed(
            merged_point_cloud_path / f"{i}.npz",
            points=merged_points,
            colors=merged_colors,
            masks=merged_masks,
        )

        # Null out heavy image fields before pickling.
        for name in camera_names:
            setattr(obs, f"{name}_rgb", None)
            setattr(obs, f"{name}_depth", None)
            setattr(obs, f"{name}_point_cloud", None)
            setattr(obs, f"{name}_mask", None)

    with (example_path / LOW_DIM_PICKLE).open("wb") as handle:
        pickle.dump(demo, handle)


# ---------------------------------------------------------------------------
# Scenario definitions ------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    label: str
    variation: int
    episodes: int
    description_hint: str


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    return [
        Scenario(
            "train-banana",
            variation=0,
            episodes=args.train_banana_episodes,
            description_hint="Train split: banana target with carrot & pan distractors.",
        ),
        Scenario(
            "train-carrot",
            variation=1,
            episodes=args.train_carrot_episodes,
            description_hint="Train split: carrot target with banana & pan distractors.",
        ),
        Scenario(
            "train-pan",
            variation=2,
            episodes=args.train_pan_episodes,
            description_hint="Train split: pan target with banana & carrot distractors.",
        ),
        Scenario(
            "test-container",
            variation=3,
            episodes=args.test_container_episodes,
            description_hint="Test split: only the small container present.",
        ),
        Scenario(
            "test-container-banana",
            variation=4,
            episodes=args.test_container_banana_episodes,
            description_hint="Test split: small container target with banana distractor.",
        ),
        Scenario(
            "test-container-carrot-pan",
            variation=5,
            episodes=args.test_container_carrot_pan_episodes,
            description_hint="Test split: small container target with carrot & pan distractors.",
        ),
    ]


# ---------------------------------------------------------------------------
# Collection helpers --------------------------------------------------------


def _configure_observation(args: argparse.Namespace) -> ObservationConfig:
    img_size = list(map(int, args.image_size))
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.masks_as_one_channel = True
    obs_config.right_shoulder_camera.masks_as_one_channel = True
    obs_config.overhead_camera.masks_as_one_channel = True
    obs_config.wrist_camera.masks_as_one_channel = True
    obs_config.front_camera.masks_as_one_channel = True
    if args.renderer == "opengl":
        from pyrep.const import RenderMode

        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL
    elif args.renderer == "opengl3":
        from pyrep.const import RenderMode

        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL3
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL3
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL3
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL3
        obs_config.front_camera.render_mode = RenderMode.OPENGL3
    return obs_config


def _write_variation_descriptions(path: Path, descriptions: Iterable[str], hint: str) -> None:
    desc_list = list(descriptions)
    if hint and hint not in desc_list:
        desc_list.append(hint)
    with (path / VARIATION_DESCRIPTIONS).open("wb") as handle:
        pickle.dump(desc_list, handle)


def _build_mask_label_map(task_env, variation_root: Path, variation_index: int) -> None:
    handles = collect_mask_handles(variation_root)
    if not handles:
        print(f"[mask-labels] No mask handles for {variation_root.name}; skipping.")
        return
    try:
        task_env.set_variation(variation_index)
        task_env.reset()
    except Exception as exc:
        print(f"[mask-labels] Warning: reset failed before label map ({exc}).")
    mapping, outstanding = build_handle_label_map(task_env, handles)
    if mapping:
        write_label_map(variation_root / DEFAULT_MAP_FILENAME, mapping, overwrite=True)
        print(f"[mask-labels] wrote {len(mapping)} labels for {variation_root.name}")
    if outstanding:
        preview = ", ".join(str(h) for h in sorted(outstanding)[:10])
        print(
            f"[mask-labels] {len(outstanding)} handles missing names for {variation_root.name}: {preview}"
        )


def collect_scenario(
    task_env,
    scenario: Scenario,
    root: Path,
    *,
    max_attempts: int = 10,
) -> None:
    if scenario.episodes <= 0:
        print(f"[skip] Scenario {scenario.label} has zero requested episodes.")
        return

    variation_dir = root / scenario.label
    episodes_dir = variation_dir / EPISODES_FOLDER
    _ensure_dir(episodes_dir)

    descriptions: Optional[List[str]] = None

    for episode_idx in range(scenario.episodes):
        task_env.set_variation(scenario.variation)
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                descriptions, _ = task_env.reset()
                (demo,) = task_env.get_demos(amount=1, live_demos=True)
                break
            except Exception as exc:  # pragma: no cover - requires simulator
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Failed to collect demo for {scenario.label} episode {episode_idx}"
                    ) from exc
                print(
                    f"[retry] Scenario {scenario.label} episode {episode_idx} "
                    f"retry {attempt}/{max_attempts} due to: {exc}"
                )
        episode_path = episodes_dir / (EPISODE_FOLDER % episode_idx)
        save_demo(demo, episode_path)
        print(
            f"[collect] {scenario.label} (variation {scenario.variation}) episode {episode_idx} saved."
        )

    if descriptions is None:
        descriptions = []
    _write_variation_descriptions(variation_dir, descriptions, scenario.description_hint)
    _build_mask_label_map(task_env, variation_dir, scenario.variation)


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/put_item_in_drawer_custom"),
        help="Directory to store the generated dataset.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=[128, 128],
        metavar=("W", "H"),
        help="Image size for rendered observations.",
    )
    parser.add_argument(
        "--renderer",
        choices=["opengl", "opengl3"],
        default="opengl3",
        help="Rendering backend for RGB/depth captures.",
    )
    parser.add_argument(
        "--arm-max-velocity",
        type=float,
        default=1.0,
        help="Maximum arm joint velocity for motion planning.",
    )
    parser.add_argument(
        "--arm-max-acceleration",
        type=float,
        default=4.0,
        help="Maximum arm joint acceleration for motion planning.",
    )
    parser.add_argument(
        "--train-banana-episodes",
        type=int,
        default=100,
        help="Number of training episodes with banana as the target.",
    )
    parser.add_argument(
        "--train-carrot-episodes",
        type=int,
        default=100,
        help="Number of training episodes with carrot as the target.",
    )
    parser.add_argument(
        "--train-pan-episodes",
        type=int,
        default=100,
        help="Number of training episodes with frying pan as the target.",
    )
    parser.add_argument(
        "--test-container-episodes",
        type=int,
        default=10,
        help="Number of test episodes with only the small container present.",
    )
    parser.add_argument(
        "--test-container-banana-episodes",
        type=int,
        default=10,
        help="Number of test episodes with small container + banana.",
    )
    parser.add_argument(
        "--test-container-carrot-pan-episodes",
        type=int,
        default=10,
        help="Number of test episodes with small container + carrot + pan.",
    )
    parser.add_argument(
        "--debug-small",
        action="store_true",
        help="Quick toggle to produce a tiny dataset (10/10/10 train, 5/5/5 test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - requires simulator
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.debug_small:
        args.train_banana_episodes = 10
        args.train_carrot_episodes = 10
        args.train_pan_episodes = 10
        args.test_container_episodes = 5
        args.test_container_banana_episodes = 5
        args.test_container_carrot_pan_episodes = 5
        print("[debug] Using reduced episode counts: 10/10/10 train, 5/5/5 test.")

    scenarios = build_scenarios(args)
    active_scenarios = [s for s in scenarios if s.episodes > 0]
    if not active_scenarios:
        raise SystemExit("No episodes requested; nothing to do.")

    obs_config = _configure_observation(args)
    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=False,
    )
    env.launch()

    try:
        task_env = env.get_task(PutItemInDrawer)
        _ensure_dir(args.output_root)
        for scenario in active_scenarios:
            collect_scenario(task_env, scenario, args.output_root)
    finally:
        env.shutdown()

    print("Dataset generation complete. Output stored in", args.output_root)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
