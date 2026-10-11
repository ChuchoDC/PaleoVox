# PaleoVox

*Turn one fossil into thousands — realistic 3D damage augmentation for paleontology AI.*

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.0.6-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

**PaleoVox** is an open-source Python toolkit (library + desktop GUI) for **3D data
augmentation of fossil and paleontological specimens**. It converts 3D meshes into binary
voxel grids, simulates several kinds of taphonomic damage (compaction/deformation, erosion,
rotation, fracturing), reconstructs watertight meshes from the augmented voxels, and provides
visual comparison and analysis tools.

Deep learning applied to paleontology is bottlenecked by data: fossil specimens are rare,
fragile, incomplete and expensive to scan. PaleoVox multiplies a single 3D model into many
realistic damaged variants, directly attacking that data-scarcity bottleneck — entirely
locally, with no cloud service involved.

## Installation (Miniconda)

PaleoVox runs in a conda environment. The project is developed and tested on **Python 3.9.12**.

```bash
# 1. Create and activate the environment
conda create -n paleovox python=3.9.12 -y
conda activate paleovox

# 2. Install the dependencies
pip install -r requirements.txt
```

That's it — no further setup required.

## Repository structure

```
PaleoVox/
├── paleovoxpy.py          # Backend library: conversion, augmentation, reconstruction, I/O
├── paleovox_gui.py        # PyQt5 desktop application (drag-and-drop GUI)
├── paleovox_viewer.py     # Standalone PBR 3D mesh viewer (runs as a subprocess)
├── requirements.txt       # Pinned runtime dependencies
├── logo/                  # PaleoVox mark (window icon + GUI logo)
├── branding/              # UNAM brandbook and logo assets
├── docs/
│   ├── SSOT.md            # Single source of truth — full technical documentation
│   └── QA/                # Unified QA test document (.tex / .pdf)
└── test_data/             # Sample meshes and voxel grids for demos/testing
```

## Quick start

### Desktop GUI (no code)

```bash
python paleovox_gui.py
```

1. Drag a `.ply` / `.obj` mesh (or a `.npy` voxel grid) into the drop zone.
2. Click **Mesh → Voxels** to voxelize the loaded mesh.
3. Apply any augmentations: **Deformation**, **Erosion**, **Rotation**, **Fracture**.
4. Click **Voxels → Mesh** to rebuild a mesh, or **Save Deformed Voxels** to export `.npy`.

### As a Python library

```python
import paleovoxpy as pv

mesh = pv.load_mesh("fossil.ply")
voxels = pv.mesh_to_voxel(mesh, npoints=100000, dimensions=256)
damaged = pv.deformation(voxels, compaction_factor=0.85, compaction_axis=2)
pv.save_voxel(damaged, "fossil_deformed.npy")
```

## Documentation

The full technical documentation, backend API catalog, GUI reference and known-issues register
live in [`docs/SSOT.md`](docs/SSOT.md).

## Authors

| Name | Email | Affiliation |
|---|---|---|
| Alan Gabriel Amaro Colin | alan_amaro@ciencias.unam.mx | Facultad de Ciencias, UNAM |
| Ángel Ángeles Cortés | angel_10@ciencias.unam.mx | Facultad de Ciencias, UNAM |
| Jair Israel Barrientos Lara | jair@geologia.unam.mx | Instituto de Geología, UNAM |
| Jesús Alberto Díaz Cruz | vertebrata.j@geologia.unam.mx | Facultad de Ciencias e Instituto de Geología, UNAM |

## License

PaleoVox is open source, distributed under the MIT License.
