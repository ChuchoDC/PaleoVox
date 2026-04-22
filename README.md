# PaleoVoxPy

**Version 1.0.7** – *Data Augmentation for 3D Fossils*

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open3D](https://img.shields.io/badge/Open3D-0.18+-orange.svg)](http://www.open3d.org/)

**PaleoVoxPy** is a Python library designed for **3D fossil data augmentation**. It provides a comprehensive pipeline to convert fossil meshes into voxel grids, apply realistic geological deformations (compaction, erosion, rotation, fracturing), and reconstruct high‑quality meshes. Perfect for training deep learning models on paleontological and morphological datasets.

---

## ✨ Features

- **Mesh ↔ Voxel conversion** – load/save meshes, convert to binary voxel grids
- **Morphological enhancement** – close holes, fill interiors, thicken surfaces for robust reconstruction
- **Damage simulation** – random erosion, axial compaction, synthetic fractures
- **Rigid transformations** – 3D rotation (ZYX and XYZ order) with nearest‑neighbor interpolation
- **High‑quality mesh reconstruction** – Poisson surface reconstruction with adaptive depth, density filtering, and Taubin smoothing
- **Visualisation tools** – interactive 3D plots (Plotly), 2D projections (Matplotlib), and t‑SNE embeddings
- **Save / load** – save voxel grids (`.npy`) and meshes (`.ply`, `.obj`, `.stl`, …)

---

## 📦 Installation

```bash
pip install open3d numpy scipy matplotlib seaborn plotly scikit-learn
```

Clone the repository:

```bash
git clone https://github.com/AlanAmaro13/PaleoVox
cd PaleoVox
```

Place `paleovoxpy.py` in your project or import directly.

---

## 🚀 Quick Start

```python
import paleovoxpy as pv

# 1. Load a mesh
_mesh, min_bound, max_bound, dimensions = pv.load_mesh(path, return_bounds= True)

# 2. Convert to voxel grid (128³)
voxels, scale_factor, orig_min, orig_max, orig_center = pv.mesh_to_voxel(
    mesh = _mesh,
    npoints = 10000,
    dimensions = 128,
    pr = True, 
    return_scale_info= True
)

# 2.1 [Optional but highly recommened]: Apply binary dilation to increase the point density in the voxel: 
voxels = pv.binary_dilation(voxels, iterations = 5) # 5 often gives better results

# 3. Apply compaction (simulate burial)
compacted = pv.deformation(voxels, compaction_factor=0.85, compaction_axis=0)

# 4. Add random fractures
fractured = pv.propagator_fracture(compacted, max_position=10)

# 5. Reconstruct a mesh
reconstructed = high_quality_voxel_to_mesh(
                fractured, 
                voxel_size=1.0,
                target_scale=dimensions,  # Original dimensions
                original_bounds=(orig_min, orig_max)  # Original position
            )

# 6. Visualise
pv.plot_meshes(_mesh, reconstructed, names=['Original', 'Augmented'])
```

---

## 📚 Function Overview

### Core conversion
| Function | Description |
|----------|-------------|
| `load_mesh(path)` | Load 3D mesh (OBJ, PLY, STL, …) using Open3D |
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
| `plot_2d_perspective(voxel_array, axis=['x','y'], ...)` | 2D projection (Matplotlib) |
| `plot_2d_perspective_2samples(...)` | Overlay two voxel grids |
| `tsne_visualization(voxel_array, percentage=0.5, perplexity=100)` | t‑SNE embedding of voxel coordinates |

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

### 3. Batch processing
```python
import glob

for obj_file in glob.glob("meshes/*.obj"):
    mesh = pv.load_mesh(obj_file)
    vox = pv.mesh_to_voxel(mesh, dimensions=64)
    pv.save_voxel(vox, f"voxels/{obj_file[:-4]}_voxel")
```

---

## 📖 Dependencies

- [Open3D](http://www.open3d.org/) – mesh I/O, sampling, Poisson reconstruction
- [NumPy](https://numpy.org/) – array handling
- [SciPy](https://scipy.org/) – morphological operations, affine transforms
- [Matplotlib](https://matplotlib.org/) – 2D projections
- [Seaborn](https://seaborn.pydata.org/) – style settings
- [Plotly](https://plotly.com/python/) – interactive 3D plots
- [scikit‑learn](https://scikit-learn.org/) – t‑SNE visualisation

---

## 🤝 Contributing

Contributions are welcome! If you find a bug or have an idea for a new augmentation (e.g., shearing, noise, localised damage), please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or collaborations, please contact [alan_amaro@ciencias.unam.mx] or open an issue on GitHub.

---

**If you use PaleoVoxPy in your research, please cite:**  
> *PaleoVoxPy: A Python Library for 3D Fossil Data Augmentation* (2026).  
> [Provide DOI or repository link when available]

