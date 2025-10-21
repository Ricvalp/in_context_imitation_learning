#!/usr/bin/env python3
"""Visualize RLBench named assets (banana.ttm, carrot.ttm, etc.) in a grid.

The script imports each asset into a headless PyRep session, extracts its mesh,
and renders a labelled subplot using matplotlib. This avoids camera placement
issues and lets you inspect multiple objects at once.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from pyrep import PyRep
    from pyrep.const import ObjectType
    from pyrep.objects.shape import Shape
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "PyRep (and a working CoppeliaSim install) is required for this script."
    ) from exc

from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "rlbench_dataset_gen" / "rlbench" / "task_design.ttt"
ASSETS_DIR = REPO_ROOT / "rlbench_dataset_gen" / "rlbench" / "assets"
DEFAULT_ASSETS = [
    "banana",
    "carrot",
    "chopping_board",
    "plate",
    "frying_pan",
    "knife_block",
    "large_container",
    "small_container",
    "dish_rack",
    "door",
]


def _triangulate(indices: Iterable[int]) -> np.ndarray:
    """Convert a flat index list into Nx3 triangles."""
    arr = np.asarray(indices, dtype=np.int32)
    if arr.size % 3 != 0:
        raise ValueError("Mesh indices are not a multiple of 3.")
    return arr.reshape((-1, 3))


def _transform_vertices(vertices: np.ndarray, shape: Shape) -> np.ndarray:
    """Move vertices from local coordinates into the world frame."""
    matrix = np.asarray(shape.get_matrix(), dtype=np.float32)
    if matrix.size == 12:  # legacy PyRep (3x4 matrix)
        matrix = matrix.reshape(3, 4)
        rotation = matrix[:, :3]
        translation = matrix[:, 3]
    elif matrix.size == 16:  # newer PyRep returns full 4x4 transform
        matrix = matrix.reshape(4, 4)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
    else:
        raise ValueError(f"Unexpected matrix size from PyRep: {matrix.size}")
    return vertices @ rotation.T + translation


def _extract_mesh(model: Shape) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate mesh data from all renderable child shapes in the model."""
    shapes: List[Shape] = [model]
    try:
        children = model.get_objects_in_tree(object_type=ObjectType.SHAPE, include_model=False)
    except TypeError:
        children = model.get_objects_in_tree(object_type=ObjectType.SHAPE)
    shapes.extend(children)
    all_vertices: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    offset = 0
    for shape in shapes:
        if hasattr(shape, "get_renderable") and not shape.get_renderable():
            continue
        verts, indices, _ = shape.get_mesh_data()
        verts = np.asarray(verts, dtype=np.float32).reshape((-1, 3))
        if verts.size == 0 or len(indices) == 0:
            continue
        verts_world = _transform_vertices(verts, shape)
        tris = _triangulate(indices) + offset
        all_vertices.append(verts_world)
        all_faces.append(tris)
        offset += verts_world.shape[0]
    if not all_vertices:
        raise RuntimeError("No mesh data extracted from model.")
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return vertices, faces


def _centre_vertices(vertices: np.ndarray) -> np.ndarray:
    centre = vertices.mean(axis=0, keepdims=True)
    return vertices - centre


def _plot_mesh(ax, vertices: np.ndarray, faces: np.ndarray) -> None:
    poly3d = [vertices[tri] for tri in faces]
    collection = Poly3DCollection(
        poly3d,
        facecolors=(0.4, 0.7, 0.9, 1.0),
        edgecolor="k",
        linewidths=0.3,
        alpha=0.95,
    )
    ax.add_collection3d(collection)
    max_extent = float(np.ptp(vertices, axis=0).max())
    if max_extent <= 0.0:
        max_extent = 0.1
    half = max_extent * 0.6
    centre = vertices.mean(axis=0)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=25, azim=40)


def capture_assets(
    asset_names: Sequence[str],
    *,
    scene_path: Path,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Import each asset, extract mesh data, and return world-space vertices."""
    if not scene_path.is_file():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    pr = PyRep()
    pr.launch(str(scene_path), headless=True)
    pr.start()
    results: List[Tuple[str, np.ndarray, np.ndarray]] = []
    try:
        for asset in asset_names:
            asset_path = ASSETS_DIR / f"{asset}.ttm"
            if not asset_path.is_file():
                raise FileNotFoundError(f"Asset TTM missing: {asset_path}")
            model = pr.import_model(str(asset_path))
            vertices, faces = _extract_mesh(model)
            model.remove()
            results.append((asset, vertices, faces))
    finally:
        pr.stop()
        pr.shutdown()
    return results


def render_figure(
    meshes: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    *,
    cols: int,
    figsize: Tuple[int, int],
) -> plt.Figure:
    rows = int(np.ceil(len(meshes) / cols))
    fig = plt.figure(figsize=figsize)
    for idx, (name, vertices, faces) in enumerate(meshes, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        verts = _centre_vertices(vertices)
        _plot_mesh(ax, verts, faces)
        ax.set_title(name, fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assets",
        nargs="*",
        default=DEFAULT_ASSETS,
        help="Asset names (without .ttm). Defaults to a representative set.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="CoppeliaSim scene to launch before importing assets.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the subplot grid.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 8.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Matplotlib figure size in inches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to save the figure (e.g. figures/named_assets.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening a matplotlib window (useful when saving only).",
    )
    args = parser.parse_args()

    meshes = capture_assets(args.assets, scene_path=args.scene)
    fig = render_figure(
        meshes,
        cols=args.cols,
        figsize=tuple(args.figsize),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved preview to {args.output}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
