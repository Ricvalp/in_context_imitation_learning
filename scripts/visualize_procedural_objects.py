#!/usr/bin/env python3
"""Quick viewer for RLBench procedural object meshes.

Render a grid of objects (by ID) using matplotlib, with each subplot labeled so
you can eyeball potential distractors for custom datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

DEFAULT_SCALE = 0.005  # matches rlbench.backend.task_utils.sample_procedural_objects


def _parse_obj(path: Path) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Load vertices and face indices from a simple Wavefront OBJ file."""
    vertices: List[List[float]] = []
    faces: List[Tuple[int, ...]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                indices = []
                for item in line[2:].split():
                    token = item.split("/")[0]
                    if token:
                        indices.append(int(token) - 1)  # OBJ is 1-indexed
                if len(indices) >= 3:
                    faces.append(tuple(indices))
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    return np.asarray(vertices, dtype=np.float32), faces


def _triangulate(face: Sequence[int]) -> Iterable[Tuple[int, int, int]]:
    """Yield triangles from an arbitrary polygon face."""
    if len(face) == 3:
        yield face[0], face[1], face[2]
        return
    root = face[0]
    for idx in range(1, len(face) - 1):
        yield root, face[idx], face[idx + 1]


def _load_mesh(path: Path, scale: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return scaled vertices and triangular faces for plotting."""
    verts, faces = _parse_obj(path)
    verts = verts * scale
    tris: List[np.ndarray] = []
    for face in faces:
        for tri in _triangulate(face):
            tris.append(np.asarray(tri, dtype=np.int32))
    return verts, tris


def _centred_vertices(vertices: np.ndarray) -> np.ndarray:
    """Centre vertices around their mean to keep plots tidy."""
    centre = vertices.mean(axis=0, keepdims=True)
    return vertices - centre


def _plot_mesh(ax: Axes3D, vertices: np.ndarray, tris: List[np.ndarray]) -> None:
    poly3d = [
        vertices[tri]
        for tri in tris
    ]
    collection = Poly3DCollection(
        poly3d,
        facecolors=cm.viridis(0.65),
        edgecolor="k",
        linewidths=0.2,
        alpha=0.85,
    )
    ax.add_collection3d(collection)
    # Autoscale axes based on vertex extents.
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    max_extent = float(extents.max())
    if max_extent == 0.0:
        max_extent = 0.05
    half_extent = max_extent * 0.6
    for axis, centre in zip("xyz", vertices.mean(axis=0)):
        getattr(ax, f"set_{axis}lim")(centre - half_extent, centre + half_extent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=35)


def visualize_objects(
    ids: Sequence[str],
    *,
    assets_root: Path,
    cols: int,
    scale: float,
) -> plt.Figure:
    ids = [item.zfill(3) for item in ids]
    rows = int(np.ceil(len(ids) / cols))
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for idx, obj_id in enumerate(ids, start=1):
        obj_dir = assets_root / obj_id
        mesh_path = obj_dir / f"{obj_id}.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing OBJ for id {obj_id}: {mesh_path}")
        verts, tris = _load_mesh(mesh_path, scale=scale)
        verts = _centred_vertices(verts)
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        _plot_mesh(ax, verts, tris)
        ax.set_title(f"ID {obj_id}", fontsize=12, pad=10)
    fig.suptitle("RLBench Procedural Objects", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ids",
        nargs="*",
        help="Optional list of object IDs (e.g. 010 023 045). If omitted, a random sample is drawn.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "rlbench_dataset_gen"
        / "rlbench"
        / "assets"
        / "procedural_objects",
        help="Root folder containing RLBench procedural_objects subdirectories.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="How many columns to use in the subplot grid.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=12,
        help="Number of random objects to visualize when no explicit IDs are provided.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Scale factor applied to vertices (match RLBench defaults).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the rendered figure (e.g. figures/sample.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI to use when saving the figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip launching an interactive viewer window.",
    )
    args = parser.parse_args()

    ids = args.ids
    if not ids:
        all_ids = sorted(
            entry.name for entry in args.assets_root.iterdir()
            if entry.is_dir() and entry.name.isdigit()
        )
        if args.sample > len(all_ids):
            raise ValueError(
                f"Requested sample size {args.sample} exceeds available objects ({len(all_ids)})."
            )
        rng = np.random.default_rng()
        ids = rng.choice(all_ids, size=args.sample, replace=False)
    fig = visualize_objects(ids, assets_root=args.assets_root, cols=args.cols, scale=args.scale)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Saved figure to {args.output}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
