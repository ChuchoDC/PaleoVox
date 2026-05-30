# PaleoVoxPy

**Version 1.0.8** — *Data Augmentation for 3D Fossils*

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open3D](https://img.shields.io/badge/Open3D-0.18+-orange.svg)](http://www.open3d.org/)

**PaleoVoxPy** is a Python library designed for **3D fossil data augmentation**. It provides a comprehensive pipeline to convert fossil meshes into voxel grids, apply realistic geological deformations (compaction, erosion, rotation, fracturing), and reconstruct high‑quality meshes. Perfect for training deep learning models on paleontological and morphological datasets.

---

## ✨ Features

- **Mesh ↔ Voxel conversion** — load/save meshes, convert to binary voxel grids
- **Morphological enhancement** — close holes, fill interiors, thicken surfaces for robust reconstruction
- **Damage simulation** — random erosion, axial compaction, synthetic fractures
- **Rigid transformations** — 3D rotation (ZYX and XYZ order) with nearest‑neighbor interpolation
- **High‑quality mesh reconstruction** — Poisson surface reconstruction with adaptive depth, density filtering, and Taubin smoothing
- **Visualisation tools** — interactive 3D plots (Plotly), 2D projections (Matplotlib), and t‑SNE embeddings
- **Save / load** — save voxel grids (`.npy`) and meshes (`.ply`, `.obj`, `.stl`, …)

---

## 📦 Installation

```bash
pip install open3d numpy scipy matplotlib seaborn plotly scikit-learn
```

To use the GUI, also install PyQt5:

```bash
pip install PyQt5
```

Alternatively, install everything at once:

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/AlanAmaro13/PaleoVox
cd PaleoVox
```

---

## 🚀 Quick Start

```python
from paleovoxpy import *

# 0. Set the path to your 3D fossil file (.obj, .ply, .stl, …)
path = './my_fossil.obj'

# 1. Load mesh & get bounding info
_mesh, min_bound, max_bound, dimensions = load_mesh(path, return_bounds=True)
_mesh  # displays triangle mesh info
```

**Step 1 — Mesh to voxel conversion**

```python
# 2. Convert the mesh into a binary voxel grid (128³)
_voxel, scale_factor, orig_min, orig_max, orig_center = mesh_to_voxel(
    mesh=_mesh,
    npoints=10000,
    dimensions=128,
    pr=True,
    return_scale_info=True
)

# 3. Visualise the voxel grid
plot_voxels(_voxel, names=['Voxel'], colors=['blue'], title='There is a fossil!')
```

**Step 2 — Fill holes with binary dilation**  

```python
# 4. Apply binary dilation to fill gaps and thicken the surface
_dilation = binary_dilation(_voxel, iterations=3)
plot_voxels(_dilation)
```

**Step 3 — Voxel to mesh reconstruction**

```python
# 5. Reconstruct a high‑quality 3D mesh
_mesh3d = high_quality_voxel_to_mesh(
    _dilation,
    voxel_size=1.0,
    target_scale=dimensions,          # original dimensions
    original_bounds=(orig_min, orig_max)  # original position
)
_mesh3d  # displays reconstructed mesh info
```

**Step 4 — Compare original vs. reconstructed**

```python
# 6. Side‑by‑side visualisation
plot_meshes(_mesh3d, _mesh, names=['Reconstructed', 'Original'])
```

**Step 5 — Apply damage augmentations (optional)**

```python
# Compaction (simulate burial)
compacted = deformation(_voxel, compaction_factor=0.7, compaction_axis=0)
plot_voxels(_voxel, compacted, names=['Original', 'Compacted'])

# Erosion
eroded = erotion_general(_voxel, axis_idx=0, increment_min=0.5, pr=True)

# Rotation
import math
rotated = rotate_voxel(_voxel, math.radians(15), math.radians(25), math.radians(10))

# Fractures
fractured, fractures_only = propagator_fracture(
    _voxel, max_position=10, return_both=True, pr=True
)
plot_voxels(fractured, fractures_only)
```

**Step 6 — Save your results**

```python
# Save mesh and voxel to disk
save_mesh(_mesh3d, './output.ply')
save_voxel(_voxel, './output.npy')
```

---

## 🖥️ GUI Application

PaleoVoxPy ships with a **drag-and-drop desktop GUI** built with `PyQt5` (`paleovox_gui.py`). No coding required — drop a `.ply`, `.obj`, or `.npy` file, visualise, apply augmentations, and export results.

### Launch

```bash
python3 paleovox_gui.py
```

### Supported File Formats

| Format | Type | Description |
|--------|------|-------------|
| `.ply` | Mesh | Polygon File Format |
| `.obj` | Mesh | Wavefront OBJ |
| `.npy` | Voxels | NumPy binary array (load pre-computed voxel grids) |

### Layout

| Panel | Contents |
|-------|----------|
| **Left** | Drop zone (drag `.ply`/`.obj`/`.npy` or click Browse), file info (path, vertices, triangles, voxel shape/occupancy), Reset, About |
| **Right — Pipeline & Augmentation** | Load Mesh → Mesh→Voxels (configurable `npoints`/`dim`) → Dilate Voxels → Voxels→Mesh; Deformation (axis + factor), Erosion (axis + increment), Rotation (X°/Y°/Z°), Fracture (max\_pos + return both), Save Deformed Voxels |
| **Right — Reconstruction & Comparison** | Reconstruct from Voxels, Compare Original vs Reconstructed mesh/voxels (Both / Original Only / Reconstructed / Current Only), Save Reconstructed Mesh |
| **Right — View & Save** | Color picker, Open3D 3D viewers for mesh and voxels, export mesh as `.ply`/`.obj` or voxels as `.npy` |
| **Right — t-SNE Analysis** | Side-by-side shared t-SNE embedding of original vs current voxels (customizable perplexity, seed, sampling ratio, point size) |
| **Right — 2D Perspectives** | Single-view 2D projection (XY/XZ/YZ) with customizable color, marker, size; original vs current comparison overlay |

### Workflow

1. **Drop** a `.ply`, `.obj`, or `.npy` file onto the left panel
2. For meshes: click **Mesh → Voxels** to convert (adjust `npoints` and `dim` as needed)
3. For `.npy` files: voxels are loaded directly — skip to augmentations or reconstruction
4. Apply **augmentations** on the voxels: deformation, erosion, rotation, fracture
5. **Reconstruct** a mesh from voxels (Reconstruction tab or pipeline)
6. **Compare** original vs processed data (mesh and voxel comparison viewers)
7. **View** the result in an interactive Open3D window
8. **Analyze** with t‑SNE embeddings or 2D perspective projections
9. **Save** the modified mesh or voxel array

All operations are wrapped in try/catch blocks with error dialogs and status-bar feedback.

### Dynamic Button States

Buttons are automatically enabled/disabled based on the current state (no file, mesh only, voxels only, reconstructed, `.npy` loaded) to prevent invalid operations.

---

## 📚 Function Overview

### Core conversion
| Function | Description |
|----------|-------------|
| `load_mesh(path)` | Load 3D mesh (OBJ, PLY, STL, …) using Open3D |
| `load_voxel(path)` | Load voxel array from `.npy` file |
| `mesh_to_voxel(mesh, dimensions=128)` | Sample mesh surface (Poisson disk) → binary voxel grid |
| `high_quality_voxel_to_mesh(voxel_array, voxel_size=1.0)` | Poisson reconstruction + cleaning + smoothing |
| `save_mesh(mesh, path)`, `save_voxel(voxel, path)` | Save results |

### Augmentation & damage
| Function | Description |
|----------|-------------|
| `binary_dilation(array_3d, iterations=2)` | Morphological closing to fill gaps and interiors |
| `deformation(voxel_array, compaction_factor, compaction_axis)` | Axial compaction (e.g., sedimentary burial) |
| `erotion_general(voxel, axis_idx, increment_min)` | Random erosion along a chosen axis |
| `rotate_voxel(voxel, angle_x, angle_y, angle_z)` | Rigid rotation (ZYX order) |
| `rotate_voxel_inv(voxel, angle_x, angle_y, angle_z)` | Rigid rotation (XYZ order) |
| `propagator_fracture(voxel_grid, max_position=10, return_both=False)` | Stochastic fracture propagation |

### Visualisation
| Function | Description |
|----------|-------------|
| `plot_voxels(voxel_array, voxel_array2=None, ...)` | Interactive 3D scatter (Plotly) |
| `plot_meshes(mesh1, mesh2=None, ...)` | Interactive 3D mesh viewer |
| `plot_2d_perspective(voxel_array, axis=['x','y'], save_path=None, ...)` | 2D projection (Matplotlib); supports headless save via `save_path` |
| `plot_2d_perspective_2samples(v1, v2, ..., save_path=None)` | Overlay two voxel grids with headless save support |
| `tsne_visualization(voxel_array, percentage=0.5, perplexity=100)` | t‑SNE embedding of voxel coordinates |
| `tsne_compare(vox_orig, vox_def, ..., save_path=None)` | Shared-space t‑SNE comparison with headless save support |

### Utilities
| Function | Description |
|----------|-------------|
| `create_voxel_grid(size)` | Empty binary grid |
| `add_line_to_voxel(voxel_grid, start, end)` | 3D Bresenham line drawing |
| `null_planes(voxel_curve, axis)` | Project occupied voxels onto a plane |

---

## 🧪 Example Workflows

### 1. Create augmented training dataset
```python
original = pv.mesh_to_voxel("trilobite.obj", dimensions=128)
augmented = []

# Compacted versions
for axis in [0,1,2]:
    for factor in [0.7, 0.85, 0.95]:
        aug = pv.deformation(original, factor, axis)
        augmented.append(aug)

# Eroded versions
for axis in [0,1,2]:
    for inc in [0.3, 0.6]:
        aug = pv.erotion_general(original, axis, inc)
        augmented.append(aug)

# Rotated versions
angles = [np.radians(a) for a in [-30, -15, 15, 30]]
for ax in angles:
    aug = pv.rotate_voxel(original, ax, 0, 0)
    augmented.append(aug)
```

### 2. Compare original and fractured fossil
```python
fractured, fractures_only = pv.propagator_fracture(voxels, max_position=8, return_both=True)
pv.plot_2d_perspective_2samples(voxels, fractured, axis=['x','z'],
                                 colors=['blue','red'], labels=['Original','Fractured'])
```

### 3. Load and process pre-computed voxels
```python
# Load a previously saved voxel grid
vox = pv.load_voxel("fossil_voxels.npy")

# Apply augmentations directly
deformed = pv.deformation(vox, compaction_factor=0.8, compaction_axis=0)

# Reconstruct a mesh
mesh = pv.high_quality_voxel_to_mesh(deformed, voxel_size=1.0)
pv.save_mesh(mesh, "output.ply")
```

### 4. Batch processing
```python
import glob

for obj_file in glob.glob("meshes/*.obj"):
    mesh = pv.load_mesh(obj_file)
    vox = pv.mesh_to_voxel(mesh, dimensions=64)
    pv.save_voxel(vox, f"voxels/{obj_file[:-4]}_voxel")
```

---

## 📖 Dependencies

- [Open3D](http://www.open3d.org/) — mesh I/O, sampling, Poisson reconstruction
- [NumPy](https://numpy.org/) — array handling
- [SciPy](https://scipy.org/) — morphological operations, affine transforms
- [Matplotlib](https://matplotlib.org/) — 2D projections
- [Seaborn](https://seaborn.pydata.org/) — style settings
- [Plotly](https://plotly.com/python/) — interactive 3D plots
- [scikit‑learn](https://scikit-learn.org/) — t‑SNE visualisation
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) — **GUI** application (`paleovox_gui.py`)

---

## 📦 Portable Executables

Standalone executables can be built using PyInstaller. No Python installation required by end users.

```
portable_versions/
├── PaleoVox.spec              ← Cross-platform PyInstaller spec
├── linux/
│   ├── run_paleovox.sh        ← Double-click launcher
│   └── PaleoVox/              ← Self-contained distribution (~850 MB)
│       ├── PaleoVox.bin       ← 64-bit ELF executable
│       └── _internal/         ← Bundled Python + all libraries
├── windows/
│   ├── build_windows.bat      ← Run on Windows to build
│   └── run_paleovox.bat       ← Launcher
└── mac/
    ├── build_mac.sh           ← Run on macOS to build
    └── run_paleovox.command   ← Finder launcher
```

### Build Requirements

- **Linux:** Python 3.8+, `pip install pyinstaller`
- **Windows:** Python 3.8+, `pip install pyinstaller` (build on Windows only)
- **macOS:** Python 3.8+, `pip3 install pyinstaller` (build on macOS only)

### Building

```bash
# From the project root directory
pyinstaller --distpath portable_versions/<platform> \
    --workpath /tmp/paleovox_build --clean \
    portable_versions/PaleoVox.spec
```

Or use the provided build scripts:
- **Linux:** already built in `portable_versions/linux/`
- **Windows:** run `build_windows.bat` on a Windows machine
- **macOS:** run `build_mac.sh` on a Mac

### Size
The standalone bundle is ~850 MB due to bundled compiled libraries (Open3D ~215 MB, PyQt5 ~100 MB, SciPy ~80 MB). This is the minimum realistic size for a scientific Python application with these dependencies. CUDA libraries (765 MB) are excluded — only CPU mode is bundled.

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| `README.md` | This file — project overview |
| `gui_documentation.tex` | Full GUI user manual (LaTeX) |
| `backend_technical_documentation.tex` | Technical backend reference (LaTeX) |
| `qa_test_document.tex` | 67‑case QA test plan (LaTeX) |
| `portable_versions/` | PyInstaller spec + build scripts |

---

## 🤝 Contributing

Contributions are welcome! If you find a bug or have an idea for a new augmentation (e.g., shearing, noise, localised damage), please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👥 Creators

| Name | Institution | Email |
|------|------------|-------|
| Alan Gabriel Amaro Colin | Universidad Nacional Autónoma de México | alan\_amaro@ciencias.unam.mx |
| Angel Angeles Córtes | Universidad Nacional Autónoma de México | pendiente |
| Dr. Jair | Universidad Nacional Autónoma de México | pendiente |
| Dr. Jesus | Universidad Nacional Autónoma de México | pendiente |

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub or reach out to any of the creators above.

---

**If you use PaleoVoxPy in your research, please cite:**  
> *PaleoVoxPy: A Python Library for 3D Fossil Data Augmentation* (2026).  
> [Provide DOI or repository link when available]
