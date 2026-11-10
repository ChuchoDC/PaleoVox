#!/usr/bin/env python3
"""PaleoVox damage atlas — storytelling presentation figures.

Generates, for one base specimen (test_data/skull.ply voxelized at 256^3),
a three-view panel set per taphonomic damage:

    {damage}_voxel.png   occupied voxels as a blocky point cloud (offscreen)
    {damage}_mesh.png    watertight reconstruction, PBR "bone" (offscreen)
    {damage}_2d.png      "X-ray" projection, original (blue) vs damage (red)

Damages: deformacion, erosion, rotacion, fractura, combinado.
All figures come from the real paleovoxpy algorithms. Deterministic seeds and
cached intermediates in /tmp/opencode/paleovox_figures/historia.

The 2D view plane and the 3D camera are chosen per damage so the effect is
always visible:
    deformacion  -> compress Z: side camera, XZ projection
    erosion      -> truncate X: top camera, XY projection
    rotacion     -> rigid turn:  side camera, YZ projection
    fractura     -> crack network: pattern rendered (voxel panel) and
                    overlaid on the original (2D panel, XY)
    combinado    -> accumulated damage: side camera, XZ projection

Usage:  python presentation/scripts/make_damage_figures.py [--only NAME]
Output: presentation/assets/figures/damage/*.png + stats.json
"""
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi

import paleovoxpy as pv
import paleovox_viewer as pvw

OUT = os.path.join(REPO, "presentation", "assets", "figures", "damage")
os.makedirs(OUT, exist_ok=True)
WORK = "/tmp/opencode/paleovox_figures/historia"
os.makedirs(WORK, exist_ok=True)

UNAM_BLUE = [0.122, 0.435, 0.659]   # #1F6FA8
UNAM_RED = [0.910, 0.267, 0.165]    # #E8442A
UNAM_BLUE_HEX = "#1F6FA8"
UNAM_RED_HEX = "#E8442A"
UNAM_DARK_HEX = "#003D64"

SIZE = 1200
BG = [0.965, 0.973, 0.976, 1.0]
UP = np.array([0.0, 1.0, 0.0])
FOV = 50.0
DIST_FACTOR = 0.97

# Specimen axes in voxel space: X = lateral (width), Y = dorsal (height),
# Z = longitudinal (length). The side profile is the most recognizable view.
SIDE_EYE = np.array([1.0, 0.24, 0.34])
FRONT_EYE = np.array([0.30, 0.25, 1.0])

DAMAGE_ORDER = ["deformacion", "erosion", "rotacion", "fractura", "combinado"]
SEEDS = {
    "deformacion": 21,
    "erosion": 42,
    "rotacion": 7,
    "fractura": 5,
    "combinado": 11,
}
LABELS = {
    "deformacion": "Deformación",
    "erosion": "Erosión",
    "rotacion": "Rotación",
    "fractura": "Fractura",
    "combinado": "Daño combinado",
}
EYES = {
    "deformacion": SIDE_EYE,
    "erosion": FRONT_EYE,
    "rotacion": SIDE_EYE,
    "fractura": SIDE_EYE,
    "combinado": SIDE_EYE,
}
VIEWS = {
    "deformacion": "xz",
    "erosion": "xy",
    "rotacion": "yz",
    "fractura": "yz",
    "combinado": "xz",
}


def log(msg):
    print(f"[atlas] {msg}", flush=True)


def load_reference():
    """Dense base grid: raw voxelization + one "Dilate Voxels" pass (GUI step).

    Thickening the surface (the GUI's dilation analogy at 256^3) makes damage
    visible in the voxel renders and produces cleaner watertight reconstructions.
    """
    cache = os.path.join(WORK, "base256_dil2.npy")
    if os.path.exists(cache):
        base = np.load(cache)
        log(f"base grid loaded from cache: occupied={int((base > 0).sum()):,}")
        return base
    raw_cache = os.path.join(WORK, "skull256_100k.npy")
    if os.path.exists(raw_cache):
        raw = np.load(raw_cache)
    else:
        mesh = pv.load_mesh("test_data/skull.ply")
        raw = pv.mesh_to_voxel(mesh, npoints=100000, dimensions=256)
        np.save(raw_cache, raw)
    base = ndi.binary_dilation(raw > 0, iterations=2).astype(np.uint8)
    np.save(cache, base)
    log(f"base grid built: raw={int((raw > 0).sum()):,} -> dilated={int((base > 0).sum()):,}")
    return base


def get_damaged(ref, name):
    cache = os.path.join(WORK, f"{name}.npy")
    if os.path.exists(cache):
        aug = np.load(cache)
        removed = None
        if name == "fractura":
            removed = ((ref > 0) & (aug == 0)).astype(np.uint8)
        log(f"{name}: loaded from cache: occupied={int((aug > 0).sum()):,}")
        return aug, removed
    t0 = time.time()
    np.random.seed(SEEDS[name])
    random.seed(SEEDS[name])
    aug = ref.astype(np.uint8)
    removed = None
    if name == "deformacion":
        aug = pv.deformation(aug, compaction_factor=0.65, compaction_axis=2)
    elif name == "erosion":
        aug = pv.erotion_general(aug, axis_idx=0, increment_min=0.40)
    elif name == "rotacion":
        aug = pv.rotate_voxel(aug, np.deg2rad(15.0), np.deg2rad(25.0), np.deg2rad(10.0))
    elif name == "fractura":
        aug = pv.propagator_fracture(aug, max_position=14, return_both=False)
        aug = (aug == 1).astype(np.uint8)
        removed = ((ref > 0) & (aug == 0)).astype(np.uint8)
        log(f"fractura: crack voxels removed={int(removed.sum()):,}")
    elif name == "combinado":
        aug = pv.deformation(aug, compaction_factor=0.85, compaction_axis=2)
        aug = pv.erotion_general(aug, axis_idx=0, increment_min=0.45)
        aug = pv.propagator_fracture(aug, max_position=14, return_both=False)
        aug = (aug == 1).astype(np.uint8)
    else:
        raise ValueError(name)
    np.save(cache, aug.astype(np.uint8))
    log(f"{name}: computed in {time.time() - t0:.1f}s, occupied={int((aug > 0).sum()):,}")
    return aug, removed


def ref_camera(ref, eye_dir):
    pts = np.argwhere(ref > 0).astype(float)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    center = (mn + mx) / 2.0
    diag = float(np.linalg.norm(mx - mn))
    eye_dir = np.asarray(eye_dir, dtype=float)
    eye_dir = eye_dir / np.linalg.norm(eye_dir)
    eye = center + eye_dir * diag * DIST_FACTOR
    return center, eye, diag


def surface_mask(voxel):
    """Outer shell of the occupied volume (renders as a stippled cloud)."""
    occ = voxel > 0
    return occ & ~ndi.binary_erosion(occ, iterations=1)


def render_voxels(voxel, path, cam, color=None, point_size=2.8):
    center, eye, diag = cam
    pts = np.argwhere(surface_mask(voxel)).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(color if color is not None else UNAM_RED)

    renderer = o3d.visualization.rendering.OffscreenRenderer(SIZE, SIZE)
    scene = renderer.scene
    scene.set_background(BG)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    scene.add_geometry("voxels", pcd, mat)
    renderer.setup_camera(FOV, center, eye, UP)
    img = renderer.render_to_image()
    o3d.io.write_image(path, img)
    log(f"wrote {os.path.relpath(path, REPO)} ({len(pts):,} points)")


def render_voxels_pair(base, removed, path, cam):
    center, eye, diag = cam
    pts_base = np.argwhere(surface_mask(base)).astype(np.float64)
    pts_rm = np.argwhere(removed > 0).astype(np.float64)
    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(pts_base)
    pcd_base.paint_uniform_color([0.72, 0.78, 0.83])
    pcd_rm = o3d.geometry.PointCloud()
    pcd_rm.points = o3d.utility.Vector3dVector(pts_rm)
    pcd_rm.paint_uniform_color(UNAM_RED)

    renderer = o3d.visualization.rendering.OffscreenRenderer(SIZE, SIZE)
    scene = renderer.scene
    scene.set_background(BG)
    mat_base = o3d.visualization.rendering.MaterialRecord()
    mat_base.shader = "defaultUnlit"
    mat_base.point_size = 2.8
    mat_rm = o3d.visualization.rendering.MaterialRecord()
    mat_rm.shader = "defaultUnlit"
    mat_rm.point_size = 4.5
    scene.add_geometry("base", pcd_base, mat_base)
    scene.add_geometry("cracks", pcd_rm, mat_rm)
    renderer.setup_camera(FOV, center, eye, UP)
    img = renderer.render_to_image()
    o3d.io.write_image(path, img)
    log(f"wrote {os.path.relpath(path, REPO)} (base {len(pts_base):,} + grietas {len(pts_rm):,})")


def render_mesh_pbr(mesh, path, cam):
    center, eye, diag = cam
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector([])

    renderer = o3d.visualization.rendering.OffscreenRenderer(SIZE, SIZE)
    scene = renderer.scene
    scene.set_background(pvw._PBR_BACKGROUNDS["meshlab"])
    scene.show_skybox(True)
    scene.scene.enable_indirect_light(True)
    scene.scene.set_indirect_light_intensity(35000)
    scene.scene.set_sun_light(direction=[0.45, -0.9, -0.6],
                              color=[1.0, 0.97, 0.88], intensity=65000)
    scene.scene.enable_sun_light(True)
    mat = pvw._pbr_make_material("bone")
    scene.add_geometry("mesh", mesh, mat)
    renderer.setup_camera(FOV, center, eye, UP)
    img = renderer.render_to_image()
    o3d.io.write_image(path, img)
    log(f"wrote {os.path.relpath(path, REPO)} (tris={len(mesh.triangles):,})")


def reconstruct(grid, name, cam):
    cache = os.path.join(WORK, f"recon_{name}.ply")
    if os.path.exists(cache):
        mesh = o3d.io.read_triangle_mesh(cache)
        if len(mesh.triangles) > 0:
            log(f"{name}: reconstruction loaded from cache (tris={len(mesh.triangles):,})")
            return mesh
    t0 = time.time()
    mesh = pv.high_quality_voxel_to_mesh(grid, voxel_size=1.0)
    o3d.io.write_triangle_mesh(cache, mesh)
    log(f"{name}: reconstructed in {time.time() - t0:.1f}s (tris={len(mesh.triangles):,})")
    return mesh


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def render_2d(ref, damaged, path, view="xy", overlay=None, overlay_label="Dañado"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.edgecolor": "#C9D6DD",
        "axes.linewidth": 2.0,
    })
    a, b = view[0], view[1]
    ia, ib = AXIS_INDEX[a], AXIS_INDEX[b]
    on1 = np.argwhere(ref == 1)
    on2 = np.argwhere(overlay > 0) if overlay is not None else np.argwhere(damaged > 0)

    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=150)
    ax.scatter(on1[:, ia], on1[:, ib], s=2.4, marker="o",
               c=UNAM_BLUE_HEX, alpha=0.20, linewidths=0, label="Original")
    ax.scatter(on2[:, ia], on2[:, ib], s=1.6, marker="o",
               c=UNAM_RED_HEX, alpha=0.80, linewidths=0, label=overlay_label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                    markerscale=8, frameon=False,
                    fontsize=30, handletextpad=0.4, columnspacing=1.4)
    ax.set_xlabel(f"Eje {a.upper()}", fontsize=42, color=UNAM_DARK_HEX, labelpad=10)
    ax.set_ylabel(f"Eje {b.upper()}", fontsize=42, color=UNAM_DARK_HEX, labelpad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    log(f"wrote {os.path.relpath(path, REPO)} ({len(on2):,} marked points, vista {view.upper()})")


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    ref = load_reference()
    cam_side = ref_camera(ref, SIDE_EYE)

    stats = {}
    stats_path = os.path.join(OUT, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            stats = json.load(fh)

    names = [n for n in DAMAGE_ORDER if only is None or n == only]

    if only is None or only == "original":
        render_voxels(ref, os.path.join(OUT, "original_voxel.png"), cam_side, color=UNAM_BLUE)

    for name in names:
        aug, removed = get_damaged(ref, name)
        cam = ref_camera(ref, EYES[name])
        if name == "fractura":
            render_voxels_pair(ref, removed, os.path.join(OUT, "fractura_voxel.png"), cam)
            render_2d(ref, aug, os.path.join(OUT, "fractura_2d.png"),
                      view=VIEWS[name], overlay=removed, overlay_label="Grietas")
        else:
            render_voxels(aug, os.path.join(OUT, f"{name}_voxel.png"), cam)
            render_2d(ref, aug, os.path.join(OUT, f"{name}_2d.png"), view=VIEWS[name])
        mesh = reconstruct(aug, name, cam)
        render_mesh_pbr(mesh, os.path.join(OUT, f"{name}_mesh.png"), cam)
        stats[name] = {
            "label": LABELS[name],
            "occupied": int((aug > 0).sum()),
            "occupied_original": int((ref > 0).sum()),
            "view_2d": VIEWS[name].upper(),
        }
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)

    log("done")


if __name__ == "__main__":
    main()
