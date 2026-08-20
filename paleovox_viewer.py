# -*- coding: utf-8 -*-
"""Standalone PBR mesh viewer for PaleoVox.

This module is intentionally lightweight (only ``open3d`` and ``numpy``) so it
can be run in a dedicated subprocess. Each viewer must run in a fresh process
because Open3D's ``gui.Application`` is single-use per process: calling
``run()`` a second time (after a viewer window has been closed) crashes.

The parent process serializes meshes to temporary ``.ply`` files and invokes
this module with a single JSON argument:

    python paleovox_viewer.py '<json>'

where ``<json>`` is::

    {
        "paths": ["/tmp/.../mesh_0.ply", ...],
        "colors": ["blue", [0.8, 0.2, 0.2], ...],
        "names": ["Original", "Current", ...],
        "bg": "meshlab",
        "wireframe": false,
        "title": "PaleoVox — Viewer"
    }
"""

import json
import sys

import numpy as np
import open3d as o3d


_PBR_PRESETS = {
    "bone"   : (0.93, 0.87, 0.78, 0.55, 0.00, 0.35),
    "white"  : (0.95, 0.95, 0.95, 0.45, 0.00, 0.30),
    "gray"   : (0.65, 0.65, 0.65, 0.50, 0.00, 0.30),
    "teal"   : (0.20, 0.65, 0.70, 0.50, 0.05, 0.40),
    "gold"   : (0.90, 0.75, 0.25, 0.35, 0.60, 0.60),
    "red"    : (0.80, 0.20, 0.20, 0.55, 0.00, 0.30),
    "blue"   : (0.25, 0.45, 0.80, 0.50, 0.00, 0.35),
    "silver" : (0.80, 0.80, 0.85, 0.25, 0.85, 0.80),
    "green"  : (0.20, 0.80, 0.20, 0.50, 0.00, 0.30),
    "orange" : (1.00, 0.60, 0.00, 0.45, 0.05, 0.35),
    "purple" : (0.60, 0.20, 0.80, 0.50, 0.00, 0.30),
    "cyan"   : (0.00, 0.80, 0.80, 0.50, 0.05, 0.35),
    "yellow" : (1.00, 1.00, 0.00, 0.40, 0.10, 0.35),
}

_PBR_BACKGROUNDS = {
    "meshlab": [0.36, 0.36, 0.40, 1.0],
    "dark"   : [0.10, 0.10, 0.12, 1.0],
    "black"  : [0.00, 0.00, 0.00, 1.0],
    "light"  : [0.90, 0.90, 0.90, 1.0],
    "white"  : [1.00, 1.00, 1.00, 1.0],
}


def _pbr_make_material(color, roughness=None, metallic=None):
    if isinstance(color, str):
        preset = _PBR_PRESETS.get(color.lower(), _PBR_PRESETS["bone"])
        r, g, b, r_rough, r_metal, r_refl = preset
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        r, g, b = color
        r_rough = 0.50
        r_metal = 0.00
        r_refl = 0.35
    else:
        r, g, b, r_rough, r_metal, r_refl = _PBR_PRESETS["bone"]

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [r, g, b, 1.0]
    mat.base_roughness = roughness if roughness is not None else r_rough
    mat.base_metallic = metallic if metallic is not None else r_metal
    mat.base_reflectance = r_refl
    mat.base_clearcoat = 0.0
    return mat


def _pbr_compute_camera(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.array(bbox.get_center(), dtype=np.float32)
    extent = np.array(bbox.get_extent(), dtype=np.float32)
    dist = float(np.linalg.norm(extent)) * 1.2
    eye = center + np.array([0.0, -dist * 0.3, dist], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return center, eye, up


def visualize_mesh(meshes, colors=None, names=None, bg="meshlab",
                   wireframe=False, title="PaleoVox — Viewer"):
    """Visualize one or more 3D meshes using Open3D's PBR-shaded viewer.

    Intended to run in a fresh process (see module docstring); therefore it
    initializes the application unconditionally and never re-runs it.
    """
    if colors is None:
        colors = ["bone"] * len(meshes)
    if names is None:
        names = [f"mesh_{i}" for i in range(len(meshes))]

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = o3d.visualization.O3DVisualizer(title, 1280, 900)
    win.show_settings = True

    for i, (mesh, color) in enumerate(zip(meshes, colors)):
        # Smooth normals: required for defaultLit to light correctly.
        mesh.compute_vertex_normals()
        # Clear vertex colors so the PBR material controls the color.
        mesh.vertex_colors = o3d.utility.Vector3dVector([])
        mat = _pbr_make_material(color)
        win.add_geometry(names[i], mesh, mat)

    if wireframe and len(meshes) > 0:
        wf_mat = o3d.visualization.rendering.MaterialRecord()
        wf_mat.shader = "unlitLine"
        wf_mat.line_width = 0.5
        wf_mat.base_color = [0.1, 0.1, 0.1, 0.3]
        wireframe_ls = o3d.geometry.LineSet.create_from_triangle_mesh(meshes[0])
        win.add_geometry("wireframe", wireframe_ls, wf_mat)

    bg_rgba = np.array(_PBR_BACKGROUNDS.get(bg, _PBR_BACKGROUNDS["meshlab"]), dtype=np.float32)
    win.set_background(bg_rgba, None)

    win.show_skybox(True)
    win.scene.scene.enable_indirect_light(True)
    win.scene.scene.set_indirect_light_intensity(35000)

    win.scene.scene.set_sun_light(
        direction=[0.45, -0.9, -0.6],
        color=[1.0, 0.97, 0.88],
        intensity=65000
    )
    win.scene.scene.enable_sun_light(True)

    if len(meshes) > 0:
        center, eye, up = _pbr_compute_camera(meshes[0])
        win.setup_camera(60.0, center, eye, up)

    win.show_axes = False
    app.add_window(win)
    app.run()


def main(argv):
    if len(argv) < 2:
        print("usage: python paleovox_viewer.py '<json>'", file=sys.stderr)
        return 2
    cfg = json.loads(argv[1])
    meshes = [o3d.io.read_triangle_mesh(p) for p in cfg["paths"]]
    visualize_mesh(
        meshes,
        colors=cfg.get("colors"),
        names=cfg.get("names"),
        bg=cfg.get("bg", "meshlab"),
        wireframe=cfg.get("wireframe", False),
        title=cfg.get("title", "PaleoVox — Viewer"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
