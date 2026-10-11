# PaleoVox — Single Source of Truth (SSOT)

> **Document status:** Living document. Every statement here is derived from the current source tree
> (`paleovoxpy.py`, `paleovox_gui.py`, `paleovox_viewer.py`, `branding/`, `test_data/`).
> **Last updated:** 2026-10-10
> **Maintainers:** Alan Gabriel Amaro Colin and the PaleoVox team (see [Appendix D](#appendix-d-authors-contacts--project-identity)).
> **How to use this document:** pick the section that matches your need —
> [non-technical overview](#2-non-technical-guide) for plain-language explanation,
> [product & market context](#3-product--market-context) for marketing,
> [technical documentation](#4-technical-documentation) for engineering.
>
> **Maintenance rule:** when code changes, update the matching SSOT section in the same commit.
> Section 4.11 lists exactly what to touch for each kind of change.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Non-Technical Guide](#2-non-technical-guide)
3. [Product & Market Context](#3-product--market-context)
4. [Technical Documentation](#4-technical-documentation)
   - 4.1 [Repository Map](#41-repository-map)
   - 4.2 [Runtime & Dependencies](#42-runtime--dependencies)
   - 4.3 [Data Model & Conventions](#43-data-model--conventions)
   - 4.4 [Backend API Catalog](#44-backend-api-catalog)
   - 4.5 [GUI Architecture](#45-gui-architecture)
   - 4.6 [PBR Viewer Subprocess](#46-pbr-viewer-subprocess)
   - 4.7 [Pipelines & Data Flows](#47-pipelines--data-flows)
   - 4.8 [Algorithm Reference](#48-algorithm-reference)
   - 4.9 [Known Issues & Technical Debt](#49-known-issues--technical-debt)
   - 4.10 [Testing & QA](#410-testing--qa)
   - 4.11 [Contributor How-Tos](#411-contributor-how-tos)
5. [Roadmap & Backlog](#5-roadmap--backlog)
6. Appendix A: [API Index A–Z](#appendix-a-api-index-az)
7. Appendix B: [GUI Widget Defaults](#appendix-b-gui-widget-defaults)
8. Appendix C: [Viewer Controls](#appendix-c-3d-viewer-controls)
9. Appendix D: [Authors, Contacts & Project Identity](#appendix-d-authors-contacts--project-identity)
10. Appendix E: [Glossary](#appendix-e-glossary)

---

## 1. Executive Summary

**PaleoVox** is an open-source Python toolkit (library + desktop GUI) for **3D data augmentation of
fossil and paleontological specimens**. It converts 3D meshes into binary voxel grids, simulates
several kinds of taphonomic damage (compaction/deformation, erosion, rotation, fracturing),
reconstructs watertight meshes from the augmented voxels, and provides visual comparison and
analysis tools.

**The problem it solves.** Deep learning applied to paleontology is bottlenecked by data:
fossil specimens are rare, fragile, incomplete, and expensive to scan. Datasets are small and
imbalanced, so models overfit and generalize poorly to damaged or partial specimens. PaleoVox
multiplies a single 3D model into many realistic damaged variants, directly attacking that
data-scarcity bottleneck without handling physical specimens.

**What ships today.**

| Component | File | Role |
|---|---|---|
| Backend library | `paleovoxpy.py` (3,401 lines, 25 public functions) | Conversion, augmentation, reconstruction, visualization, I/O |
| Desktop GUI | `paleovox_gui.py` (1,659 lines, PyQt5) | No-code drag-and-drop workflow with 5 tabs |
| PBR viewer | `paleovox_viewer.py` (165 lines) | Standalone Open3D physically-based-rendering mesh viewer run as a subprocess |
| Branding | `branding/` + `logo/` | UNAM brandbook and logo assets |
| Sample data | `test_data/` | Meshes and voxel grids for testing/demos |

**Status.** Functional research software, version string `1.0.6` in the backend
([paleovoxpy.py:9](paleovoxpy.py)). Ready for research use; the known-issues register
([section 4.9](#49-known-issues--technical-debt)) lists gaps to resolve before a public release.

---

## 2. Non-Technical Guide

*This section is written for paleontologists, museum staff, educators, students, and anyone who
wants to understand PaleoVox without reading code.*

### 2.1 The problem, in plain words

Training an artificial-intelligence (AI) model to recognize, complete, or classify fossils
requires **many examples**. But real fossils are:

- **Rare** — there is only one of each specimen.
- **Fragile** — handling them damages them.
- **Incomplete** — erosion, compaction, and fracture destroy parts of the original shape.
- **Expensive to digitize** — CT scanning and photogrammetry take time, equipment, and money.

As a result, AI models trained on fossil data often see only a handful of clean, complete examples.
When they meet a real, damaged fossil in the field, their accuracy drops.

### 2.2 What PaleoVox does

PaleoVox is a **virtual damage laboratory for 3D fossils**. Starting from a single digital model
(a 3D mesh such as `.ply` or `.obj`), it can generate many *new*, different-looking versions of
that model by simulating the kinds of damage fossils actually suffer:

| Damage type | What it simulates | Plain-language analogy |
|---|---|---|
| **Deformation (compaction)** | The specimen is squeezed along one axis | A fossil buried under layers of sediment for millions of years |
| **Erosion** | A random portion is removed along one axis | Weathering, water, or excavation removing part of the fossil |
| **Rotation** | The specimen is turned in 3D space | The same fossil found lying at a different angle |
| **Fracture** | Crack lines are propagated through the volume | The broken, cracked fossils commonly found in the field |

These damaged versions can be **combined** (for example: compact, then erode, then fracture) and
exported as new voxel data (`.npy` files) for training AI models, or reconstructed back into 3D
meshes (`.ply`/`.obj`) for visualization and sharing.

### 2.3 What a "voxel" is

A **voxel** is a 3D pixel — a small cube in a 3D grid. PaleoVox converts a fossil surface into a
grid of cubes that are either "occupied" (part of the fossil) or "empty" (air). Working with cubes
makes it easy to cut, squeeze, crack, and count the shape, which is exactly what damage simulation
needs. The voxel grid is saved as a `.npy` file.

Typical grid sizes are **128×128×128** or **256×256×256** cubes. The GUI uses **256³** by default.

### 2.4 The desktop application, tab by tab

Launch with `python3 paleovox_gui.py`. The window has a **left panel** and **five tabs** on the right.

**Left panel (file management)**

- **Drop zone** — drag and drop a `.ply`, `.obj`, or `.npy` file, or click to browse.
- **File Info** — shows the file path, number of vertices/triangles (meshes), and voxel grid
  shape and occupied-voxel count.
- **Reset All** — clears everything and restores all settings to their defaults.
- **More about PaleoVox...** — credits and project description.

**Tab 1 — Pipeline & Augmentation** (the main workflow)

1. Click **Mesh → Voxels** to convert a loaded mesh into a voxel grid (adjustable resolution).
2. Optionally click **Dilate Voxels** to fill small holes and thicken the surface.
3. Apply any number of **augmentations**: Deformation, Erosion, Rotation, Fracture.
4. Click **Voxels → Mesh** to rebuild a 3D mesh from the (possibly damaged) voxels.
5. **Save Deformed Voxels** exports the augmented grid as `.npy`.

**Tab 2 — Reconstruction & Comparison**

- **Reconstruct from Voxels** and **Save Reconstructed Mesh**.
- **Compare Original vs Reconstructed** to see both 3D meshes (blue = original, red = reconstructed).
- **Compare Voxels** for a voxel-level comparison (original vs current), with an optional
  Marching Cubes mode that shows voxels as smooth surfaces.

**Tab 3 — View & Save**

- Pick a display color; open the mesh viewer or voxel viewer; save the current mesh or voxels.
- A **Marching Cubes** checkbox switches voxel viewing between blocks and smooth surfaces.

**Tab 4 — t-SNE Analysis**

Generates a shared 2D "map" of the original and damaged voxels side by side. Voxels that are
similar cluster together; differences between the two maps reveal how the damage changed the shape.
Adjustable: perplexity, random seed, sampling percentage, point size. The resulting image can be saved.

**Tab 5 — 2D Perspectives**

Shows "X-ray" projections of the voxel grid onto a plane (XY, XZ, or YZ) — useful for quickly
checking shape, density, and orientation. You can compare the original vs the damaged voxels
overlaid in two colors, or load **two external `.npy` files** for a standalone comparison.
Images can be saved as PNG/JPG.

### 2.5 Typical user journeys

**A. From a scanned mesh (most common)**

1. Drag `my_fossil.ply` into the drop zone.
2. Set resolution (`dim`) and sampling (`npoints`), click **Mesh → Voxels**.
3. Apply damage: e.g., Deformation Z 0.85 → Erosion X 0.5 → Fracture max 10.
4. Save the deformed voxels (`.npy`) for training, and/or reconstruct and save a mesh for display.

**B. From existing voxel data**

1. Drag `specimen.npy` into the drop zone — voxels load directly, no mesh conversion needed.
2. Apply augmentations, analyze (t-SNE / 2D), and save.

**C. Comparing two datasets**

1. Open the 2D Perspectives tab, tick **Use External Voxels for Comparison**.
2. Drop Voxel A and Voxel B `.npy` files and click **Generate Comparison**.

### 2.6 Frequently asked questions

| Question | Answer |
|---|---|
| Do I need to program? | No. The GUI covers the full workflow. Programmers can use `paleovoxpy.py` as a library. |
| Which file types can I load? | Meshes: `.ply`, `.obj` (GUI filters); voxels: `.npy`. The backend can read any format Open3D supports. |
| Is PaleoVox generating real fossils? | No. It generates **synthetic damaged variants** of a digital model for data augmentation. |
| Can I combine damages? | Yes. Each augmentation is applied to the current voxel grid, so effects stack. |
| Does my data leave my computer? | No. Everything runs locally; no cloud service is involved. |
| Why did "Dilate Voxels" fail? | Known limitation: dilation requires a 128³ grid while the GUI defaults to 256³. See [issue K-01](#49-known-issues--technical-debt); set `dim` to 128 before converting. |
| Is there an undo? | No. Save intermediate voxel files if you want to experiment safely. **Reset All** only clears the session. |
| Is it free? | The project is open source (MIT per historical project metadata; LICENSE file pending — see [section 4.9](#49-known-issues--technical-debt)). |

---

## 3. Product & Market Context

*Fact base for marketing campaigns, grant applications, talks, and outreach. Derived from the code
and the branding assets in `branding/`. Nothing here overrides the technical truth in section 4.*

### 3.1 Positioning statement

PaleoVox is an **open-source, domain-specific 3D data-augmentation toolkit for paleontology**:
a Python library plus a no-code desktop app that turns a single digitized fossil into a rich,
damage-varied dataset — simulating the taphonomic processes (compaction, erosion, fracture,
rotation) that real specimens experience.

### 3.2 Elevator pitches

- **One line:** "Turn one fossil into thousands — realistic 3D damage augmentation for
  paleontology AI."
- **Short (30 s):** "AI for paleontology starves for data: specimens are rare, fragile, and
  incomplete. PaleoVox converts a 3D fossil model into voxels, simulates compaction, erosion,
  fracture, and rotation to generate many realistic damaged variants, then reconstructs and
  visualizes the results — all locally, with a drag-and-drop GUI and an open Python backend."
- **Long (conference/paragraph):** see [section 1](#1-executive-summary) and
  [section 2.2](#22-what-paleovox-does).

### 3.3 Target audiences / personas

| Persona | Description | What PaleoVox gives them |
|---|---|---|
| **Researcher in DL + paleontology** | Trains 3D CNNs on scans; limited specimens | Scriptable augmentation pipeline; thousands of labeled variants; reproducibility controls |
| **Paleontologist / curator** | Digitizes collection specimens; limited coding | Drag-and-drop GUI; visual before/after comparison; exports `.npy`/`.ply`/`.obj` |
| **Museum / outreach team** | Wants engaging 3D visuals | PBR-rendered viewers; 2D X-ray projections; shareable images and meshes |
| **Educator / student** | Teaches fossil morphology and taphonomy | Interactive, free tool; damage simulation as a teaching metaphor |
| **ML engineer / data scientist** | Integrates augmentation into training | Pure-Python functions with documented signatures; NumPy in/out |

### 3.4 Value propositions and differentiators

1. **Damage models are paleontology-specific** — compaction, erosional truncation, fracture
   propagation, and rigid rotation, not generic flips/noise.
2. **End-to-end pipeline** — mesh → voxels → augmentation → reconstruction → export → comparison
   in one tool.
3. **Two ways to work** — a no-code GUI for domain experts and a plain-Python API for ML pipelines.
4. **Runs locally and free** — no cloud, no account, no data leaving the machine.
5. **Built-in scientific QA visuals** — side-by-side mesh/voxel comparison, 2D projections,
   shared t-SNE embedding, PBR rendering.
6. **Born at UNAM** — developed within the Faculty of Sciences and the Institute of Geology,
   UNAM (see [Appendix D](#appendix-d-authors-contacts--project-identity)).

### 3.5 Proof points (verifiable from the codebase)

- 25 public backend functions covering conversion, augmentation, morphology, reconstruction,
  visualization, and I/O.
- 4 damage simulators: deformation, erosion, rotation (two conventions), fracture.
- 5-tab desktop GUI with 34 user-adjustable widgets, all with documented defaults.
- Watertight mesh reconstruction via a 7-step Marching Cubes + pymeshfix pipeline.
- PBR-shaded 3D viewers with 13 material presets and 5 backgrounds.
- Bundled sample data (meshes + 256³/512³ voxel grids) for demos.
- Deterministic options: seeded rotations/t-SNE embeddings; every operation reversible by
  re-loading saved `.npy` files.

### 3.6 Brand identity & assets

The project's own mark lives at `logo/logo.png` (used as window icon and GUI logo).
Institutional identity follows the **UNAM brandbook** at `branding/brandbook.md`.

**Colors (UNAM core palette):**

| Color | HEX | RGB | CMYK | Use |
|---|---|---|---|---|
| UNAM Blue (primary) | `#003D64` | 11/61/98 | 100/75/36/25 | Primary brand color |
| Light Grey (neutral) | `#EAEDEF` | 234/237/239 | 9/5/6/0 | Backgrounds, support |
| Bright Blue (accent) | `#0897D5` | 6/151/213 | 77/25/0/0 | Highlights |
| Red (accent) | `#E8442A` | 232/68/42 | 0/84/86/0 | Emphasis, warnings |
| Grey-Green (accent) | `#7F9297` | 127/146/151 | 53/31/33/11 | Secondary support |

**Typography:** Gotham (headlines), Booster Next FY (body), Myriad Pro Black (specific
applications), Arial/Helvetica (general use).

**Logo usage rules (summary):** minimum height 1.5 cm; keep clear space around the logo; never
distort, crop, recolor, or combine with decorative elements; prefer the two-color version on white;
provide light- and dark-background variants.

**Available logo files:** `branding/logos/unam-azul.png`, `branding/logos/unam-dorado.png`,
`branding/logos/geologia-color.png`, `branding/logos/geologia-mono.png`.
`branding/brandbook.md` is the authoritative reference — this SSOT only summarizes it.

### 3.7 Keywords, hashtags & channels

- **Keywords:** paleontology, virtual paleontology, fossil, data augmentation, deep learning,
  3D voxel, mesh reconstruction, taphonomy, compaction, erosion, fracture, open science.
- **Hashtags:** `#PaleoVox` `#Paleontology` `#DataAugmentation` `#DeepLearning` `#VirtualPaleontology`
  `#OpenSource` `#UNAM` `#3D`
- **Channels:** GitHub repository, academic conferences and seminars, UNAM faculty/institute
  networks, museum partnerships, LinkedIn/X technical threads, educational workshops.

### 3.8 Messaging guardrails (do not overclaim)

- PaleoVox produces **synthetic augmentation**, not reconstructions of real biological truth.
- Damage models are **simplified approximations** of taphonomic processes, not simulations of
  specific geological histories.
- It does **not** replace physical curation, CT scanning, or phylogenetic analysis.
- Downstream model accuracy must be **validated**; augmentation is a hypothesis, not a guarantee.

---

## 4. Technical Documentation

### 4.1 Repository Map

```
PaleoVox/
├── paleovoxpy.py          # Backend library: 25 public functions (3,401 lines)
├── paleovox_gui.py        # PyQt5 desktop application (1,659 lines)
├── paleovox_viewer.py     # Standalone PBR mesh viewer, run as subprocess (165 lines)
├── requirements.txt       # Pinned runtime dependencies (pip install -r requirements.txt)
├── logo/
│   └── logo.png           # PaleoVox mark (window icon + GUI panel logo)
├── branding/
│   ├── brandbook.md       # UNAM brand identity guidelines (colors, type, logo rules)
│   └── logos/             # unam-azul.png, unam-dorado.png, geologia-color.png, geologia-mono.png
├── docs/
│   └── SSOT.md            # This document
└── test_data/
    ├── skull.ply                        # 57 MB, decimated skull mesh
    ├── test_object.obj                  # 21 MB, test mesh
    ├── skull_deformed.npy               # (256,256,256) uint8, deformed skull voxels
    ├── skull_deformed_voxel.npy         # (256,256,256) bool, voxelized deformed skull
    ├── test_object_deformed.npy         # (256,256,256) uint8, deformed test object
    └── skull_512.npy                    # (512,512,512) uint8, high-res skull (gitignored, local only)
```

**Module responsibilities**

| Module | Responsibility | Must not contain |
|---|---|---|
| `paleovoxpy.py` | Algorithms, file I/O, plotting, viewer launcher. Pure functions + matplotlib/Open3D side effects. | PyQt imports |
| `paleovox_gui.py` | UI, state, threading, user feedback. Calls backend only through `paleovoxpy as pv`. | Algorithm duplication |
| `paleovox_viewer.py` | One-shot PBR rendering in a fresh process. Only `open3d` + `numpy` imports. | App state, Qt, file dialogs |

### 4.2 Runtime & Dependencies

**Python:** 3.7+ historically claimed; 3.8–3.11 is the practical target (Open3D/PyQt5 wheels).
The code uses f-strings and type hints.

**Third-party packages**

| Package | Import | Required by | Notes |
|---|---|---|---|
| `open3d` | `o3d` | backend, GUI, viewer | Mesh I/O, voxel grids, viewers |
| `numpy` | `np` | all modules | Arrays, math |
| `scipy` | `ndimage`, `affine_transform`, `zoom` | backend | Morphology, geometric transforms |
| `matplotlib` | `plt` | backend, GUI | 2D plots (GUI forces `Agg`) |
| `seaborn` | `sns` | backend import-time styling | Plot style only |
| `scikit-learn` | `sklearn.manifold.TSNE` | t-SNE functions | Lazy import inside functions |
| `scikit-image` | `skimage` | reconstruction functions | Lazy import inside functions |
| `pymeshfix` | `pymeshfix` | `high_quality_voxel_to_mesh` | Lazy import; watertight repair |
| `plotly` | `plotly.graph_objects` | `plot_voxels`, `plot_meshes` only | Lazy import; not needed by GUI workflow |
| `PyQt5` | `PyQt5.QtWidgets/QtCore/QtGui` | GUI only | Desktop application |

**Install:**

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for development and testing (from the `paleovox`
conda environment, Python 3.9).

**Platform notes**

- `paleovox_gui.py` forces `matplotlib.use('Agg')` (headless-safe) and sets
  `XDG_SESSION_TYPE=x11` on Linux before importing matplotlib
  ([paleovox_gui.py:3-10](paleovox_gui.py)).
- The GUI never calls `plt.show()`; every plot is rendered to a temporary PNG via `save_path`.
- The mesh viewer must run in a **fresh process** per view (Open3D `gui.Application` is
  single-use). See [section 4.6](#46-pbr-viewer-subprocess).

### 4.3 Data Model & Conventions

#### 4.3.1 Voxel arrays

- NumPy arrays, conventionally **cubic** `(N, N, N)`, where `N` ∈ {128, 256, 512}.
- Binary semantics: occupied = `1`/`True`, empty = `0`/`False`. Most plotting functions test
  `== 1`; augmentation functions treat `> 0` as occupied.
- **dtype drift is real**: `mesh_to_voxel` returns `uint8`; `binary_dilation` returns `int8`;
  rotation/erosion preserve input dtype; loaded `.npy` files may be `uint8` or `bool`.
- Stored as `.npy` (NumPy native binary). Backend defaults: `dimensions=128`, `npoints=10000`.
  GUI defaults: `dim=256`, `npoints=100000`.

#### 4.3.2 Mesh objects

- `open3d.geometry.TriangleMesh` with vertices, triangles, and (after processing) vertex/triangle
  normals. GUI supports `.ply` and `.obj`; the backend can also read `.stl`, `.off`, `.gltf/glb`.
- `save_mesh` exports **both** `.ply` and `.obj` by default for `.ply`/`.obj`/no-extension paths
  (see 4.4.7); other extensions are written as-is.

#### 4.3.3 Axis conventions (critical)

Two conventions coexist and must not be confused:

| Context | Mapping |
|---|---|
| GUI axis labels ([paleovox_gui.py:28](paleovox_gui.py)) | `X → numpy axis 0`, `Y → 1`, `Z → 2` |
| NumPy arrays | axis 0 = first dimension, axis 1 = second, axis 2 = third |
| `deformation`/`erotion_general` **docstrings** | claim `0 = Z, 1 = Y, 2 = X` — **contradicts the code and GUI** (documentation bug, see K-02) |

**The code is the truth:** `compaction_axis=0` compresses `array[:, :, :]` along axis 0,
and the GUI maps "X" to 0, "Y" to 1, "Z" to 2.

#### 4.3.4 Scale & position bookkeeping

`mesh_to_voxel(..., return_scale_info=True)` returns
`(voxel_array, scale_factor, original_min_bound, original_max_bound, original_center)`.

The GUI stores `scale_info = result[1:]` and later reconstructs with
`target_scale = orig_max - orig_min` and `original_bounds = (orig_min, orig_max)`
([paleovox_gui.py:1024-1027](paleovox_gui.py)). Note `scale_factor` is informational —
reconstruction uses the bounds directly. For directly loaded `.npy` files there is no scale
info, so reconstructions stay in voxel units.

#### 4.3.5 GUI in-memory state

| Attribute | Type | Meaning |
|---|---|---|
| `file_path` | `str/None` | Last loaded file |
| `mesh` | `TriangleMesh/None` | Current working mesh (becomes reconstructed mesh after `Voxels→Mesh`) |
| `original_mesh` | `TriangleMesh/None` | Untouched mesh loaded from disk |
| `reconstructed_mesh` | `TriangleMesh/None` | Latest reconstruction from voxels |
| `voxel` | `ndarray/None` | Current (possibly augmented) voxel grid |
| `original_voxel` | `ndarray/None` | Reference voxel grid (copy at load/conversion time) |
| `bounds` | `(min, max, dims)/None` | Mesh bounds from `load_mesh` |
| `scale_info` | tuple/None | `(scale_factor, orig_min, orig_max, orig_center)` |
| `fracture_pattern` | `ndarray/None` | Last fracture lines when "Return both" is checked |
| `voxel_display_color` | RGB tuple | Display color for non-MC voxel viewer |
| `external_voxel1/2`, `external_path1/2` | ndarray/str | External `.npy` comparison pair |

### 4.4 Backend API Catalog

All signatures are exact; parameter tables list defaults and accepted ranges. "Gotchas" are
code-verified behaviors worth knowing before relying on a function.

Dependency codes: **NP** = numpy, **SC** = scipy, **O3D** = open3d, **MPL** = matplotlib,
**SKL** = scikit-learn, **SKI** = scikit-image, **PLT** = plotly, **PMF** = pymeshfix.

#### 4.4.1 Load functions

##### `load_mesh(path, return_bounds=False)` — `paleovoxpy.py:124`

Loads a triangle mesh with `o3d.io.read_triangle_mesh`.
Returns the mesh, or `(mesh, min_bound, max_bound, dimensions)` when `return_bounds=True`
(also prints the bounds to stdout). Deps: O3D.

##### `load_voxel(path)` — `paleovoxpy.py:165`

Thin `np.load` wrapper returning the voxel array. No shape/dtype validation. Deps: NP.

#### 4.4.2 Conversion functions

##### `mesh_to_voxel(mesh, npoints=10000, dimensions=128, pr=False, return_scale_info=False)` — `paleovoxpy.py:218`

Samples the mesh surface with Poisson-disk sampling (`npoints` points), rasterizes into a
`dimensions³` grid (`uint8`, 1 = occupied), and optionally reports scale info. Deps: O3D, NP.

| Return mode | Value |
|---|---|
| `return_scale_info=False` | `voxel_array` `(j, j, j)` uint8 |
| `return_scale_info=True` | `(voxel_array, scale_factor, original_min_bound, original_max_bound, original_center)` |

**Gotchas:** `npoints` and `dimensions` are independent — low `npoints` can leave gaps; `scale_factor`
is computed on the sampled point cloud, not the voxel grid; the actual occupied cells are derived
from `VoxelGrid.get_voxels()` and clipped into the array.

#### 4.4.3 Morphology functions

##### `binary_dilation(array_3d, iterations=2)` — `paleovoxpy.py:456`

Pipeline: 6-connected dilation → erosion (closing) → `binary_fill_holes`. Returns `int8`, same
shape. Deps: NP, SC.

**Gotchas:**

- **Requires exactly `(128,128,128)`**; raises `ValueError("Array must be 128x128x128")` otherwise
  ([paleovoxpy.py:645-646](paleovoxpy.py)). This clashes with the GUI default of 256³ (issue K-01).
- `iterations` > 3 can merge distinct structures and close legitimate holes.

#### 4.4.4 Reconstruction functions

##### `high_quality_voxel_to_mesh(voxel_array, voxel_size=1.0, target_scale=None, original_bounds=None, closing_radius=3, gaussian_sigma=1.2, level=None, fix_holes=True, max_triangles=None)` — `paleovoxpy.py:662`

The production reconstruction pipeline:

1. **Morphological closing** (`skimage.morphology.closing`, ball radius `closing_radius`) to join fragments.
2. **Largest connected component** kept (`skimage.measure.label`), discarding debris.
3. **Gaussian smoothing** (`gaussian_sigma`) to soften voxel stair-stepping.
4. **Marching Cubes** (`skimage.measure.marching_cubes`) at isovalue `level` (auto = midpoint).
5. **pymeshfix repair** when `fix_holes=True` for a watertight surface.
6. **Scale/translate** to `target_scale` and `original_bounds` if provided.
7. **Optional decimation** to `max_triangles` (quadric edge collapse).

Returns an `o3d.geometry.TriangleMesh`. Deps: SKI, O3D, NP, PMF.

**Gotchas:** prints progress to stdout; `closing_radius` too high fuses structures; decimation may
drop fine detail; scale/translate operate on the post-repair mesh.

##### `voxel_to_mesh_mc(voxel_array)` — `paleovoxpy.py:806`

Fast visualization-only conversion: closing (ball 2) → Gaussian (σ = 1.0) → Marching Cubes at
level 0.3. **No** repair, scaling, or translation. Returns an `o3d.geometry.TriangleMesh`.
Deps: SKI, O3D, NP.

#### 4.4.5 Damage functions

##### `deformation(voxel_array, compaction_factor=0.85, compaction_axis=2, preserve_binary=True, threshold=0.5)` — `paleovoxpy.py:1237`

Compresses along `compaction_axis` using `scipy.ndimage.zoom(order=0)` and pads the result back to
the original shape (centered). `compaction_factor` ∈ (0, 1]; 1.0 = identity. Returns array with the
same shape (binary when `preserve_binary=True`). Deps: NP, SC.

**Gotchas:** docstring axis labels are wrong (K-02); compression loses information below ~0.7;
the "non-binary input" warning is printed but no conversion is enforced
([paleovoxpy.py:1425-1427](paleovoxpy.py)).

##### `erotion_general(voxel, axis_idx=0, increment_min=0.5, pr=False)` — `paleovoxpy.py:1474`

Random truncation along `axis_idx`. Computes the occupied extent along the axis, then cuts at
`np.random.randint(restriction, max_val)` where `restriction = int(min_val * coefficient * increment_min)`
and `coefficient = max_val / min_val`. Everything beyond the cut is zeroed. Returns a copy.
Deps: NP.

**Gotchas:** uses the global NumPy RNG (seed externally for reproducibility); docstring axis
labels are wrong (K-02). The `increment_min=1.0` degenerate-extent `ValueError` was fixed
(K-13).

##### `rotate_voxel(voxel, angle_x, angle_y, angle_z)` — `paleovoxpy.py:1705`

Rigid rotation about the volume center using `R = Rz @ Ry @ Rx` and
`scipy.ndimage.affine_transform(order=0, mode='constant', cval=0)`. Angles in **radians**.
Preserves binary values; regions rotated outside the volume are lost (filled 0). Deps: NP, SC.

##### `rotate_voxel_inv(voxel, angle_x, angle_y, angle_z)` — `paleovoxpy.py:1966`

Same as `rotate_voxel` but with `R = Rx @ Ry @ Rz` (XYZ convention).
**Not** the mathematical inverse of `rotate_voxel` despite the name
([paleovoxpy.py:2167-2168](paleovoxpy.py)). Angles in radians. Deps: NP, SC.

##### `propagator_fracture(voxel_grid, max_position=10, return_both=False, pr=False)` — `paleovoxpy.py:2230`

Stochastic fracture propagation:

1. Sort occupied voxels lexicographically (`np.lexsort`).
2. Start at the voxel closest to the origin.
3. Repeatedly select a random one among the nearest `max_position + 1` candidates that are
   strictly ahead in all coordinates (`np.all(a < on_sorted, axis=1)`).
4. Draw a Bresenham line from the current point to the selected point.
5. On termination, compute the fractured object as `original − null_planes(lines, axis)` with a
   255 → 0 fix (uint8 underflow guard).

Returns the fractured object, or `[fractured_object, fracture_pattern]` when `return_both=True`.
Deps: NP, `random`. Helper: `null_planes`, `add_line_to_voxel`, `get_closest_to_origin`,
`create_voxel_grid`.

**Gotchas:** termination relies on a bare `except Exception` catching the `IndexError` raised when
fewer candidates exist than the random index requires ([paleovoxpy.py:2484,2494](paleovoxpy.py)) —
it works, but is fragile and silent; uses Python's `random` module (unseeded in the GUI);
directionally biased by design.

#### 4.4.6 Utility functions

##### `create_voxel_grid(size=128)` — `paleovoxpy.py:954`

Empty boolean grid `(size, size, size)`.

##### `add_line_to_voxel(voxel_grid, start_point, end_point)` — `paleovoxpy.py:958`

Draws a 1-voxel-thick 3D Bresenham line **in place**; returns the same grid. Bounds-safe.
Deps: NP.

##### `find_closest_to_origin(vectors)` — `paleovoxpy.py:1198`

Returns `(closest_vector, index)`. Currently unused by the GUI/library workflows.

##### `get_closest_to_origin(vectors)` — `paleovoxpy.py:1222`

Returns only the closest vector. Used by `propagator_fracture`. Deps: NP.

##### `null_planes(voxel_curve, axis)` — `paleovoxpy.py:2212`

Projects every occupied coordinate across the full extent along `axis`, producing full "cut planes"
(used to remove material around fracture lines). Uses `create_voxel_grid(shape[0])` — cubic grids
only. Deps: NP.

#### 4.4.7 Save functions

##### `save_mesh(mesh, path, export_both=True)` — `paleovoxpy.py:3078`

Robust mesh export:

1. **Orientation fix:** computes the signed volume of the mesh; if negative (inward-facing
   triangles from Marching Cubes), flips triangle winding
   ([paleovoxpy.py:3202-3208](paleovoxpy.py)).
2. Recomputes and normalizes vertex/triangle normals (prevents the "flat, unlit" look in
   MeshLab/Blender).
3. When `export_both=True` and the extension is `.ply`, `.obj`, or empty: writes **both**
   `<base>.ply` and `<base>.obj`. Otherwise writes the given path only.
4. Writes with normals, vertex colors, and triangle UVs enabled.

Returns `None`; **catches all exceptions and prints them** (fail-safe).
Deps: O3D, NP.

**Gotcha:** because errors are swallowed, callers cannot tell success from failure; the GUI always
prints a success status (K-10).

##### `save_voxel(voxel, path)` — `paleovoxpy.py:3236`

`np.save` wrapper; prints `Correctly saved: <path>`. NumPy appends `.npy` when missing. Catches all
exceptions and prints. Deps: NP.

#### 4.4.8 Visualization functions

##### `plot_voxels(voxel_array, voxel_array2=None, names=['Vox 1','Vox 2'], colors=['blue','red'], title='Comparison')` — `paleovoxpy.py:292`

Interactive Plotly 3D scatter of occupied voxels (marker size 1, opacity 0.8). Displays via
`fig.show()`. Not used by the GUI. Deps: NP, PLT (lazy import).

##### `plot_meshes(mesh1, mesh2=None, opacity1=0.5, opacity2=0.5, colors=None, names=['M1','M2'], title='Mesh!')` — `paleovoxpy.py:846`

Overlaid Plotly Mesh3d rendering of one or two meshes. Not used by the GUI. Deps: NP, PLT (lazy).

##### `plot_2d_perspective(voxel_array, axis=['x','y'], color='blue', marker='o', size=1.0, save_path=None)` — `paleovoxpy.py:2514`

Matplotlib scatter "X-ray" projection of occupied voxels onto two axes (axis strings `x/y/z`
mapped `0/1/2`). Enforces equal aspect. If `save_path` is given: saves PNG at 150 dpi and closes
the figure, returning the path; otherwise `plt.show()`. Deps: NP, MPL.

##### `plot_2d_perspective_2samples(voxel_array_1, voxel_array_2, axis=['x','y'], colors=['blue','red'], markers=['o','o'], sizes=[1.0,1.0], labels=['V1','V2'], alphas=[1.0,1.0], save_path=None)` — `paleovoxpy.py:2696`

Overlaid two-sample comparison with per-sample color, marker, size, label, and alpha. Legend uses
`markerscale=5` and forces legend alpha to 1.0 so transparency does not hide legend markers.
Same save/show semantics as above. Deps: NP, MPL.

##### `tsne_visualization(voxel_array, seed=42, percentage=0.5, pp=100, size=1.0, color='blue', pr=False)` — `paleovoxpy.py:2930`

Samples `percentage` of occupied voxels, runs 2D t-SNE (perplexity `pp`), displays a scatter.
Returns `None`. Deps: NP, MPL, SKL (lazy).

**Gotcha:** the sampling uses unseeded `np.random.choice` — only the embedding is seeded, so
sampling differs between runs (K-04).

##### `tsne_compare(voxel_original, voxel_deformed, seed=42, percentage=0.5, pp=100, size=1.0, colors=('blue','red'), labels=('Original','Deformed'), pr=False, save_path=None)` — `paleovoxpy.py:2979`

Samples both grids with `np.random.default_rng(seed)`, concatenates occupied coordinates, fits a
**single shared** t-SNE embedding, splits it, and renders two side-by-side subplots. With
`save_path`: saves at 150 dpi and returns the path; otherwise `plt.show()`. This is the function
the GUI uses. Deps: NP, MPL, SKL (lazy).

##### `visualize_mesh(meshes, colors=None, names=None, bg='meshlab', wireframe=False, title='PaleoVox — Viewer')` — `paleovoxpy.py:3340`

Convenience launcher: serializes each mesh to a temp `.ply`, invokes
`paleovox_viewer.py` in a **subprocess** with a JSON payload, and waits (blocking). Cleans the
temp directory in a `finally` block. Colors may be preset names or RGB tuples. Deps: O3D,
os/sys/subprocess/json/tempfile.

#### 4.4.9 Viewer module public surface (`paleovox_viewer.py`)

| Symbol | Line | Purpose |
|---|---|---|
| `_PBR_PRESETS` | 33 | 13 named materials: `(r, g, b, roughness, metallic, reflectance)` — bone, white, gray, teal, gold, red, blue, silver, green, orange, purple, cyan, yellow |
| `_PBR_BACKGROUNDS` | 49 | `meshlab`, `dark`, `black`, `light`, `white` |
| `_pbr_make_material(color, roughness=None, metallic=None)` | 58 | Builds an Open3D `MaterialRecord` (`defaultLit`); tuple colors get roughness 0.50 / metallic 0.00 / reflectance 0.35 |
| `_pbr_compute_camera(mesh)` | 80 | Auto-frames the camera: `dist = 1.2 × ‖bbox extent‖`, eye offset `(0, −0.3·dist, dist)` |
| `visualize_mesh(meshes, colors=None, names=None, bg='meshlab', wireframe=False, title=...)` | 90 | Creates a 1280×900 `O3DVisualizer`, applies materials, skybox, indirect light (35000), sun light (65000), optional wireframe `LineSet`, then `app.run()` |
| `main(argv)` | 147 | Parses the JSON argument, loads `.ply` meshes, calls `visualize_mesh`; usage error exits with code 2 |

**JSON payload schema** (produced by `paleovoxpy.visualize_mesh`):

```json
{
  "paths": ["/tmp/paleovox_view_xxx/mesh_0.ply", "..."],
  "colors": ["blue", [0.8, 0.2, 0.2]],
  "names": ["Original", "Current"],
  "bg": "meshlab",
  "wireframe": false,
  "title": "PaleoVox — Viewer"
}
```

### 4.5 GUI Architecture

#### 4.5.1 Class structure

- **`DropZone(QLabel)`** ([paleovox_gui.py:31](paleovox_gui.py)) — drag-and-drop/file-picker widget.
  Accepts `.ply`, `.obj`, `.npy`; emits `file_dropped(path)`; visual states `idle`/`hover`/`loaded`.
  Reused three times: main file drop and external Voxel A/B drops (the external zones restrict to
  `.npy` on load).
- **`PaleoVoxGUI(QMainWindow)`** ([paleovox_gui.py:92](paleovox_gui.py)) — application window:
  left panel + `QTabWidget` with five tabs; owns all state (see 4.3.5), all handlers, and
  `_update_button_states`.
- **`main()`** ([paleovox_gui.py:1647](paleovox_gui.py)) — creates `QApplication`, sets the
  application name and 10 pt font, shows the window.

#### 4.5.2 UI construction map

| Method | Line | Builds |
|---|---|---|
| `_init_ui` | 112 | Window title/icon/style, splitter (350/700), status bar |
| `_build_left_panel` | 173 | Logo, DropZone, Browse, File Info, Reset All, About |
| `_on_show_about` | 227 | About dialog with description and creators |
| `_build_right_panel` | 265 | The five tabs |
| `_build_pipeline_group` | 307 | Load/npoints/dim/Mesh→Voxels/Dilate/Voxels→Mesh |
| `_build_augment_group` | 356 | Deformation, Erosion, Rotation, Fracture, Save Deformed Voxels |
| `_build_view_save_group` | 445 | Color, View Mesh/Voxels, Save Mesh/Voxel, MC toggle |
| `_build_reconstruction_group` | 490 | Reconstruct, Compare meshes, Show filter, Save reconstructed |
| `_build_voxel_comparison_group` | 528 | Show filter, Compare Voxels, MC toggle |
| `_build_tsne_tab` | 560 | t-SNE parameters and Generate |
| `_build_2d_perspectives_group` | 614 | Axis, single view, comparison, external voxels |

#### 4.5.3 Event handlers

| Handler | Line | Action |
|---|---|---|
| `on_load_mesh` | 934 | Loads mesh (stores `original_mesh`) or `.npy` (stores `original_voxel`); raises `Load Error` when a mesh file loads with 0 vertices/faces (QA EH-001) |
| `on_convert_to_voxels` | 981 | `pv.mesh_to_voxel(..., return_scale_info=True)` |
| `on_dilate` | 1003 | `pv.binary_dilation` (see K-01) |
| `on_voxels_to_mesh` | 1017 | `pv.high_quality_voxel_to_mesh`; sets `mesh` **and** `reconstructed_mesh` |
| `on_deform` | 1046 | `pv.deformation` |
| `on_erode` | 1065 | `pv.erotion_general` |
| `on_rotate` | 1084 | Degrees → radians → `pv.rotate_voxel` |
| `on_fracture` | 1102 | `pv.propagator_fracture`; stores `fracture_pattern` when "Return both" |
| `_on_reconstruct` | 1192 | Same reconstruction as `on_voxels_to_mesh` |
| `_on_compare_meshes` | 1221 | PBR viewer with Original (blue) vs Reconstructed (red) |
| `_on_compare_voxels` | 1254 | Voxel-grid or Marching-Cubes comparison |
| `_on_generate_tsne` | 1333 | `pv.tsne_compare` → temp PNG → dialog |
| `_on_generate_2d_single` | 1412 | `pv.plot_2d_perspective` → temp PNG → dialog |
| `_on_generate_2d_comparison` | 1466 | `pv.plot_2d_perspective_2samples` → temp PNG → dialog |
| `on_view_mesh` | 1145 | Prefers `reconstructed_mesh`, else `mesh`; subprocess PBR viewer |
| `on_view_voxels` | 1156 | MC mesh subprocess viewer, or blocky `VoxelGrid` via `draw_geometries` |
| `on_save_mesh` / `on_save_voxel` | 1612 / 1629 | File dialogs + `pv.save_mesh` / `pv.save_voxel` |
| `_on_save_deformed_voxels` | 1127 | Saves current voxels with `_deformed.npy` default name |
| `_on_save_reconstructed` | 1595 | Saves reconstructed mesh |
| `_on_load_external_voxel1/2` | 1543 / 1560 | Loads external `.npy` for 2D comparison |
| `_on_reset` | 870 | Clears state and restores all 34 widget defaults |

#### 4.5.4 Dynamic button-state matrix

Computed in `_update_button_states` ([paleovox_gui.py:827-856](paleovox_gui.py)) from
`has_mesh`, `has_voxel`, `has_path`, `has_reconstructed`, `has_original`, `has_original_voxel`,
`has_external_voxels`:

| Button | Enabled when |
|---|---|
| Load Mesh | a file path exists |
| Mesh → Voxels | mesh loaded |
| Dilate Voxels / Voxels → Mesh | voxels exist |
| Deformation / Erosion / Rotation / Fracture Apply | voxels exist |
| View Mesh / Save Mesh | mesh or reconstructed mesh exists |
| View Voxels / Save Voxel | voxels exist |
| Reconstruct from Voxels | voxels exist |
| Compare Original vs Reconstructed | original **and** reconstructed mesh exist |
| Save Reconstructed Mesh | reconstructed mesh exists |
| Compare Voxels | original voxels **and** current voxels exist |
| Generate t-SNE | original voxels **and** current voxels exist |
| Save Deformed Voxels / Generate 2D Perspective | voxels exist |
| Generate 2D Comparison | (original voxels **and** current voxels) **or** both external voxels |

**Note:** augmentation handlers update the info panel but not button states; because `has_voxel`
does not change during augmentation, the matrix remains valid.

#### 4.5.5 Threading & viewer rules

| Operation | Threading |
|---|---|
| Mesh view (`on_view_mesh`), mesh comparison | `QTimer.singleShot(0, ...)` then backend subprocess viewer (blocking, lets Qt repaint first) |
| MC voxel view / MC voxel comparison | `threading.Thread(daemon=True)` computing the mesh then launching the subprocess viewer |
| Blocky voxel view / comparison | `threading.Thread(daemon=True)` calling `o3d.visualization.draw_geometries` |
| t-SNE / 2D plots | Synchronous on the Qt thread; results rendered to a temp PNG and shown in a dialog |
| Backend-only operations (voxelization, augment, reconstruction) | Synchronous on the Qt thread |

**Hard rule:** never call Open3D `gui.Application.run()` twice in one process. The backend's
`visualize_mesh` therefore spawns `paleovox_viewer.py` as a fresh process for every view.

#### 4.5.6 Error handling & status

- `_show_error(title, msg)` → `QMessageBox.critical` ([paleovox_gui.py:931](paleovox_gui.py)).
- Every handler wraps its logic in `try/except` and reports failures through `_show_error`.
- `_status(msg)` writes to the `QStatusBar`.
- Temporary images are cleaned with `_safe_unlink` and `finally` blocks (bug TA-BON fix), so a
  backend failure never leaks temp files.

### 4.6 PBR Viewer Subprocess

**Why a subprocess?** Open3D's `gui.Application` is single-use per process: calling `run()` a
second time (after a window closes) crashes. The design splits responsibilities:

```
GUI / library (parent process)
  └── paleovoxpy.visualize_mesh(meshes, colors, names, bg, wireframe, title)
        ├── writes each mesh to /tmp/paleovox_view_<rand>/mesh_i.ply
        ├── spawns: python paleovox_viewer.py '<json payload>'   (blocking)
        └── finally: removes the temp directory
paleovox_viewer.py (fresh process each time)
  ├── json.loads(argv[1])
  ├── o3d.io.read_triangle_mesh(...) for each path
  ├── app.initialize()
  ├── O3DVisualizer(title, 1280, 900)
  │     ├── per mesh: compute_vertex_normals(), clear vertex colors, apply PBR material
  │     ├── optional wireframe LineSet (unlitLine, 0.5 line width)
  │     ├── background preset, skybox, indirect light 35000, sun light 65000
  │     └── auto-framed camera (60° FOV)
  └── app.add_window(win); app.run()
```

**Rendering details verified in code:**

- Shader `defaultLit`; vertex colors are cleared so the material controls color.
- Presets: 13 named materials (4.4.9). Unknown names fall back to `bone`.
- Backgrounds: `meshlab` (default), `dark`, `black`, `light`, `white`.
- Sun: direction `[0.45, −0.9, −0.6]`, warm white `[1.0, 0.97, 0.88]`, intensity 65000.
- The wireframe overlay is built from the **first** mesh only.
- Multiple meshes share one window (used for Original vs Current comparisons).

### 4.7 Pipelines & Data Flows

#### 4.7.1 Mesh → Voxel → Augment → Reconstruct (primary)

```
.ply/.obj ──load_mesh──► mesh ──mesh_to_voxel──► voxel (uint8)
    │                                              │
    │ original_mesh (reference)                    ├─ binary_dilation (128³ only)
    │                                              ├─ deformation
    │                                              ├─ erotion_general
    │                                              ├─ rotate_voxel
    │                                              └─ propagator_fracture
    │                                              │
    │          high_quality_voxel_to_mesh ◄────────┘
    │          (closing → largest component → gaussian → MC
    │           → pymeshfix → scale/translate → decimate)
    ▼                                              ▼
save_mesh (both .ply+.obj)                 save_voxel (.npy)
```

The GUI passes `target_scale`/`original_bounds` from `scale_info`, so reconstructed meshes return
to the original world scale and position. The `.npy` path has no scale info, so reconstruction
stays in voxel units.

#### 4.7.2 Direct voxel pipeline (`.npy`)

```
.npy ──load_voxel──► voxel ──(augmentations)──► modified voxel
                       │                              │
                       └─ original_voxel (reference)  ├─ high_quality_voxel_to_mesh → save_mesh
                                                      ├─ save_voxel
                                                      ├─ tsne_compare (vs original)
                                                      └─ plot_2d_perspective_2samples (vs original)
```

#### 4.7.3 External comparison pipeline

```
voxel_A.npy ─┐
             ├─► plot_2d_perspective_2samples → dialog → optional PNG/JPG save
voxel_B.npy ─┘
```

#### 4.7.4 Visualization flows

| Need | Path |
|---|---|
| Mesh, PBR shaded | `pv.visualize_mesh` → subprocess `paleovox_viewer.py` |
| Voxels as blocks | GUI thread → `VoxelGrid.create_from_point_cloud` → `draw_geometries` |
| Voxels as smooth surface | `pv.voxel_to_mesh_mc` → `pv.visualize_mesh` |
| 2D projection | `pv.plot_2d_perspective(_2samples)(save_path=tmp.png)` → Qt dialog |
| Shape analysis | `pv.tsne_compare(save_path=tmp.png)` → Qt dialog |
| Interactive 3D scatter | `pv.plot_voxels` / `pv.plot_meshes` (Plotly; library users only) |

### 4.8 Algorithm Reference

#### 4.8.1 Bresenham 3D line (`add_line_to_voxel`)

Selects the driving axis (largest delta) and steps along it, maintaining two error terms for the
other axes. Pure integer arithmetic; bounds-checked per voxel; sets the endpoint at the end.
Produces 6-connected continuous lines.

#### 4.8.2 Reconstruction pipeline (`high_quality_voxel_to_mesh`)

1. `closing(data, ball(closing_radius))` — binary closing with a spherical structuring element.
2. `label` + `bincount` — keep the largest connected component (noise removal).
3. `gaussian(mask, sigma=gaussian_sigma)` — continuous scalar field.
4. `marching_cubes(smoothed, level)` at `level = (min+max)/2` when `level=None`.
5. `pymeshfix.MeshFix(...).repair()` — guaranteed watertight when enabled.
6. Uniform scaling by `target_scale / current_scale` and translation of the centroid to the
   original center.
7. `simplify_quadric_decimation(max_triangles)` — optional triangle budget.

#### 4.8.3 Deformation

`new_shape[axis] = max(1, int(shape[axis] × factor))`; `scipy.ndimage.zoom(order=0)`; then
centered zero-padding back to the original shape; thresholded to binary if requested. Bounding box
is preserved; volume is not conserved.

#### 4.8.4 Erosion cut point

```
coords        = occupied coordinates along axis
max_val       = max(coords)
min_val       = min(coords) + 1          # +1 offset
coefficient   = max_val / min_val
restriction   = int(min_val × coefficient × increment_min)
cut_point     = randint(restriction, max_val)
result[cut_point:, ...] = 0              # (axis-dependent slicing)
```

Because `min_val × coefficient ≈ max_val`, the cut is effectively bounded below by
`max_val × increment_min`. Determinism requires seeding NumPy's global RNG.

#### 4.8.5 Rotation matrices

```
Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]

rotate_voxel:      R = Rz @ Ry @ Rx   (ZYX matrix order)
rotate_voxel_inv:  R = Rx @ Ry @ Rz   (XYZ matrix order)
```

Both apply `R_inv` through `affine_transform` with `offset = center − R_inv @ center`,
nearest-neighbor interpolation, and zero fill. Rotation is about the geometric center `shape/2`.

#### 4.8.6 Fracture propagation

Forward-only stochastic walk over lexicographically sorted occupied voxels; at each step the line
to a random near candidate is carved with Bresenham; final fractured geometry is
`original − null_planes(lines)` (planes extruded from the crack path, i.e., material removal).
Termination is implicit (candidate exhaustion → `IndexError` → `except` path).

#### 4.8.7 t-SNE comparison

Both grids are sampled independently (`percentage` of occupied voxels each), coordinates are
concatenated, one TSNE model (`n_components=2`, `perplexity=pp`, `random_state=seed`) is fit on the
union, then the embedding is split back into two panels. Sampling uses `default_rng(seed)` in
`tsne_compare` (reproducible) but unseeded `np.random.choice` in `tsne_visualization` (not fully
reproducible).

### 4.9 Known Issues & Technical Debt

Severity: **High** = blocks a default workflow; **Medium** = misleading or fragile;
**Low** = cleanup. All are code-verified with line references.

| ID | Severity | Issue | Evidence / Suggested fix |
|---|---|---|---|
| K-01 | **High** | `binary_dilation` requires exactly 128³, but the GUI defaults to `dim=256`; clicking "Dilate Voxels" on a default conversion raises `ValueError` | `paleovoxpy.py:645-646` vs `paleovox_gui.py:327-328,1009`. Parameterize the shape check (accept any cubic grid) |
| K-02 | Medium | `deformation` and `erotion_general` docstrings claim axis `0=Z, 1=Y, 2=X`, contradicting the code and the GUI mapping `X→0, Y→1, Z→2` | `paleovoxpy.py:1270-1275,1498-1503` vs `paleovox_gui.py:28`. Fix the docstrings to the NumPy convention |
| K-03 | Medium | `propagator_fracture` terminates via a bare `except Exception` catching an `IndexError` from an out-of-range random index; also uses unseeded `random` | `paleovoxpy.py:2484,2494`. Replace with an explicit candidate-count check and optional seed parameter |
| K-04 | Medium | `tsne_visualization` samples with unseeded `np.random.choice` (only the embedding is seeded) while `tsne_compare` uses `default_rng(seed)` | `paleovoxpy.py:2965` vs `3034`. Pass an RNG through both |
| K-05 | Low | Dead/duplicated code: `find_closest_to_origin` unused; `plot_voxels`/`plot_meshes` unused by the GUI and require optional `plotly`; `plot_voxels` docstring references a nonexistent `plot_fossile` | `paleovoxpy.py:292,846,1198,2658`. Mark plotly functions optional or remove/refactor |
| K-06 | Low | `deformation` prints "may not be binary" then does nothing (`pass`) | `paleovoxpy.py:1425-1427`. Either binarize or raise |
| K-07 | Low | Two independent `visualize_mesh` implementations (subprocess launcher vs in-process viewer) can drift | `paleovoxpy.py:3340` vs `paleovox_viewer.py:90`. Keep the JSON protocol documented (4.6) and treat the viewer as the contract |
| K-08 | Low | `rotate_voxel_inv` is not the mathematical inverse of `rotate_voxel`; misleading name | `paleovoxpy.py:1966,2167-2168`. Rename to `rotate_voxel_xyz`/`rotate_voxel_zyx` |
| K-09 | Low | Version string mismatch: backend banner says 1.0.6; historical README claimed 1.0.8; no single version source | `paleovoxpy.py:9`; prior `README.md`. Introduce one `__version__` constant |
| K-10 | Low | `save_mesh`/`save_voxel` swallow all exceptions and print; the GUI reports success regardless | `paleovoxpy.py:3233-3234,3335-3336`; GUI handlers. Return a bool or raise |
| K-11 | Medium | Plotting functions compare `== 1` strictly, so dtype drift (`int8`, `bool`, values > 1) can silently produce empty plots | e.g. `paleovoxpy.py:396,2669,2881`. Normalize via `> 0` |
| K-12 | Low | Directly loaded `.npy` voxels carry no scale metadata, so reconstructed meshes from them are in voxel units | `paleovox_gui.py:939-957,1192-1207`. Consider embedding scale info in the `.npy` (e.g. sidecar or `np.savez`) |
| K-13 | ~~Low~~ Resolved | `erotion_general` raised `ValueError` for `increment_min=1.0` (`restriction >= max_val`) | Fixed 2026-10-11: `restriction` clamped to `max_val - 1` before `np.random.randint` (`paleovoxpy.py:1680-1686`; QA AU-015) |
| K-14 | Low | `QPalette`, `QColor` imported but unused in the GUI; augmentation handlers skip `_update_button_states` (harmless today) | `paleovox_gui.py:25`; handlers 1046-1125. Cleanup |
| K-15 | Low | Repository docs and packaging are being regenerated: `README.md`, `docs/*.tex`, `portable_versions/` are deleted in the working tree (present in git history, pending rewrite for this version) | Working tree status. Track with the doc regeneration work |

### 4.10 Testing & QA

**There is no automated test suite in the repository.** The `bugs/` folder (QA reports and a local
test harness) is gitignored. Historically, a 67-case manual QA plan and per-widget default checks
existed (latent QA IDs such as `QA-23`, `TA-BON`, `WD-006` appear in commit history); those
documents are being regenerated.

**Sample data available for manual testing:**

| File | Shape / Size | dtype | Suggested use |
|---|---|---|---|
| `test_data/skull.ply` | 57 MB mesh | — | Full mesh→voxel pipeline |
| `test_data/test_object.obj` | 21 MB mesh | — | Lightweight pipeline testing |
| `test_data/skull_deformed.npy` | (256,256,256) | uint8 | Deformed voxels for comparison/t-SNE |
| `test_data/skull_deformed_voxel.npy` | (256,256,256) | bool | Direct voxel load; tests bool dtype path |
| `test_data/test_object_deformed.npy` | (256,256,256) | uint8 | External comparison pair with `skull_deformed.npy` |
| `test_data/skull_512.npy` | (512,512,512) | uint8 | High-res/performance testing (gitignored, local) |

**Smoke-test checklist (manual):**

1. Load `test_data/skull.ply`, convert to voxels at `dim=128` (dilation requires 128³ until K-01 is fixed).
2. Dilate, deform, erode, rotate, fracture; verify occupied counts update and no crash.
3. Reconstruct; compare original vs reconstructed in the viewer.
4. Generate t-SNE and both 2D projections; save the images.
5. Load `skull_deformed_voxel.npy` directly; verify the direct-voxel flow.
6. Toggle "Use External Voxels" and compare the two `*_deformed.npy` files.
7. Save mesh, reconstructed mesh, voxels, and deformed voxels; confirm files exist on disk
   (remember `save_mesh` writes both `.ply` and `.obj` by default).
8. Reset All; confirm all 34 widgets return to defaults (guarded historically by WD-006, which
   counted the 33 pre-existing checks plus `chk_use_external`).

### 4.11 Contributor How-Tos

**Adding a backend function**

1. Implement in `paleovoxpy.py` with a full NumPy-style docstring (params, returns, warnings).
2. Update: [4.4](#44-backend-api-catalog), [Appendix A](#appendix-a-api-index-az),
   [section 3.5](#35-proof-points-verifiable-from-the-codebase) if it is a user-visible capability.

**Adding a GUI control or tab**

1. Add widgets in the matching `_build_*` method; wire handlers with `try/except` +
   `_show_error` + `_status`.
2. Add enable/disable logic to `_update_button_states` and the matrix in
   [4.5.4](#454-dynamic-button-state-matrix).
3. Add the default value to `_on_reset` and [Appendix B](#appendix-b-gui-widget-defaults).
4. Update [4.5.3](#453-event-handlers).

**Adding a damage type**

1. Implement the algorithm in `paleovoxpy.py`; keep binary in → binary out where possible.
2. Add GUI widgets + handler in `_build_augment_group` / `on_<damage>`.
3. Update: [2.2](#22-what-paleovox-does) (both damage tables), [3.5](#35-proof-points-verifiable-from-the-codebase),
   [4.4.5](#445-damage-functions), [4.7.1](#471-mesh--voxel--augment--reconstruct-primary).

**Changing the viewer protocol or presets**

1. Edit `paleovox_viewer.py` presets/backgrounds; keep the JSON schema compatible with
   `paleovoxpy.visualize_mesh`.
2. Update [4.4.9](#449-viewer-module-public-surface-paleovox_viewerpy) and [4.6](#46-pbr-viewer-subprocess).

**Versioning and release**

1. Keep a single version string (resolve K-09), update [section 1](#1-executive-summary) metadata.
2. Regenerate README/packaging (K-15) in the same change as any version bump.

---

## 5. Roadmap & Backlog

*Suggested priorities derived from the known-issues register and observed gaps. Not commitments.*

**Near term (correctness)**

- Fix K-01 (parameterize dilation shape) — removes the most likely first-user failure.
- Fix K-02 (axis documentation).
- Replace the bare-except termination in `propagator_fracture` (K-03) with explicit logic.
- Make plotting occupancy checks robust to dtype drift (K-11).

**Near term (packaging & docs)**

- Regenerate `README.md` and the LaTeX documentation set (K-15).
- Publish this SSOT as the canonical entry point for all future docs.
- Add a LICENSE file and resolve the MIT/open-source status.

**Medium term (capability)**

- Batch augmentation API/CLI: generate N damaged samples per input from a config file.
- Seeded, reproducible augmentation runs across all damage functions (RNG plumbing).
- `.npy` metadata (shape/scale/provenance) so reconstructions from loaded voxels keep world scale (K-12).
- PyTorch `Dataset`/DataLoader adapter for direct training integration.

**Long term (scope)**

- Additional damage models (abrasion, encrustation, post-burial distortion).
- Import/export of common volumetric formats (DICOM/NIfTI) for CT pipelines.
- Portable builds for Linux/Windows/macOS (previously existed; regenerate).
- Automated regression suite (pytest) covering backend functions and GUI state logic.

---

## Appendix A: API Index A–Z

| Function | File:Line | Group |
|---|---|---|
| `add_line_to_voxel` | `paleovoxpy.py:958` | Utility |
| `binary_dilation` | `paleovoxpy.py:456` | Morphology |
| `create_voxel_grid` | `paleovoxpy.py:954` | Utility |
| `deformation` | `paleovoxpy.py:1237` | Damage |
| `erotion_general` | `paleovoxpy.py:1474` | Damage |
| `find_closest_to_origin` | `paleovoxpy.py:1198` | Utility |
| `get_closest_to_origin` | `paleovoxpy.py:1222` | Utility |
| `high_quality_voxel_to_mesh` | `paleovoxpy.py:662` | Reconstruction |
| `load_mesh` | `paleovoxpy.py:124` | Load |
| `load_voxel` | `paleovoxpy.py:165` | Load |
| `mesh_to_voxel` | `paleovoxpy.py:218` | Conversion |
| `null_planes` | `paleovoxpy.py:2212` | Damage (helper) |
| `plot_2d_perspective` | `paleovoxpy.py:2514` | Visualization |
| `plot_2d_perspective_2samples` | `paleovoxpy.py:2696` | Visualization |
| `plot_meshes` | `paleovoxpy.py:846` | Visualization |
| `plot_voxels` | `paleovoxpy.py:292` | Visualization |
| `propagator_fracture` | `paleovoxpy.py:2230` | Damage |
| `rotate_voxel` | `paleovoxpy.py:1705` | Damage |
| `rotate_voxel_inv` | `paleovoxpy.py:1966` | Damage |
| `save_mesh` | `paleovoxpy.py:3078` | Save |
| `save_voxel` | `paleovoxpy.py:3236` | Save |
| `tsne_compare` | `paleovoxpy.py:2979` | Visualization |
| `tsne_visualization` | `paleovoxpy.py:2930` | Visualization |
| `visualize_mesh` (launcher) | `paleovoxpy.py:3340` | Viewer |
| `visualize_mesh` (in-process) | `paleovox_viewer.py:90` | Viewer |
| `DropZone` (class) | `paleovox_gui.py:31` | GUI |
| `PaleoVoxGUI` (class) | `paleovox_gui.py:92` | GUI |
| `main` (GUI entry) | `paleovox_gui.py:1647` | GUI |
| `main` (viewer entry) | `paleovox_viewer.py:147` | Viewer |

## Appendix B: GUI Widget Defaults

Restored by `_on_reset` ([paleovox_gui.py:870-929](paleovox_gui.py)). These are the canonical
defaults for manual QA (historical WD-001…WD-006 checks).

| Widget | Default | Range / Options |
|---|---|---|
| `spin_npoints` | 100000 | 100–200000, step 1000 |
| `spin_dims` | 256 | 32–512 |
| `spin_dilate_iter` | 2 | 1–10 |
| `combo_def_axis` | Z (index 2) | X, Y, Z |
| `spin_def_factor` | 0.85 | 0.10–0.99, step 0.05 |
| `combo_ero_axis` | X (index 0) | X, Y, Z |
| `spin_ero_inc` | 0.5 | 0.01–1.0, step 0.05 |
| `spin_rot_x` | 0.0 | −180–180, 1 decimal |
| `spin_rot_y` | 0.0 | −180–180, 1 decimal |
| `spin_rot_z` | 0.0 | −180–180, 1 decimal |
| `spin_frac_max` | 10 | 1–50 |
| `chk_frac_both` | unchecked | — |
| `combo_color` | Blue (index 0) | Blue, Red, Green, Orange, Purple, Cyan, Yellow, White, Gray |
| `chk_voxel_mc` | unchecked | — |
| `combo_compare_vis` | Both (index 0) | Both, Original Only, Reconstructed Only |
| `combo_voxel_vis` | Both (index 0) | Both, Original Only, Current Only |
| `chk_voxel_comp_mc` | unchecked | — |
| `spin_tsne_pp` | 100 | 5–500 |
| `spin_tsne_seed` | 42 | 0–9999 |
| `spin_tsne_pct` | 0.50 | 0.01–1.00, step 0.05 |
| `spin_tsne_size` | 1.0 | 0.1–10.0, step 0.5 |
| `combo_2d_axis` | XY (index 0) | XY, XZ, YZ |
| `combo_2d_color` | Blue (index 0) | color list |
| `combo_2d_marker` | o (index 0) | o, ., ^, s, *, +, x |
| `spin_2d_size` | 1.0 | 0.1–10.0 |
| `combo_2d_cmp_color1` | Blue (index 0) | color list |
| `combo_2d_cmp_marker1` | o (index 0) | marker list |
| `spin_2d_cmp_size1` | 1.0 | 0.1–10.0 |
| `spin_2d_cmp_alpha1` | 1.0 | 0.0–1.0, step 0.05 |
| `combo_2d_cmp_color2` | Red (index 1) | color list |
| `combo_2d_cmp_marker2` | ^ (index 2) | marker list |
| `spin_2d_cmp_size2` | 1.0 | 0.1–10.0 |
| `spin_2d_cmp_alpha2` | 1.0 | 0.0–1.0, step 0.05 |
| `chk_use_external` | unchecked | — |

## Appendix C: 3D Viewer Controls

Standard Open3D viewer interactions (PBR viewer and `draw_geometries` windows):

| Action | Control |
|---|---|
| Rotate | Left mouse drag |
| Pan | Middle mouse drag (or Shift + left drag) |
| Zoom | Mouse wheel |
| Reset camera | `R` / toolbar |
| Change lighting/background | Viewer settings panel (PBR viewer, `show_settings=True`) |
| Close window | Window close button; the process exits afterwards |

Note: each mesh view launches a fresh process, so closing the window ends that viewer session.

## Appendix D: Authors, Contacts & Project Identity

From the in-app About dialog ([paleovox_gui.py:247-252](paleovox_gui.py)):

| Name | Contact | Affiliation |
|---|---|---|
| Alan Gabriel Amaro Colin | alan_amaro@ciencias.unam.mx | Facultad de Ciencias, UNAM |
| Ángel Ángeles Cortés | angel_10@ciencias.unam.mx | Facultad de Ciencias, UNAM |
| Jair Israel Barrientos Lara | jair@geologia.unam.mx | Instituto de Geología, UNAM |
| Jesús Alberto Díaz Cruz | vertebrata.j@geologia.unam.mx | Facultad de Ciencias e Instituto de Geología, UNAM |

**Project identity**

- **Name:** PaleoVox (backend banner: "PaleoVoxPy", engine "PaleoVox V. 1.0.6").
- **Repository (git remote):** `git@github.com:ChuchoDC/PaleoVox.git` (branch `main`).
  Historical documentation referenced `github.com/AlanAmaro13/PaleoVox` — canonical URL to be
  confirmed.
- **License:** MIT per historical project metadata; no `LICENSE` file in the current tree (K-15).
- **GUI window title:** "PaleoVox — Fossil Voxel Augmentation".

## Appendix E: Glossary

| Term | Meaning |
|---|---|
| **Augmentation** | Generating new, varied training samples from existing data. |
| **Binary voxel grid** | 3D array of 0/1 cubes: occupied vs empty. |
| **Bresenham (3D)** | Integer algorithm that draws a straight continuous line between two grid cells. |
| **Compaction** | Geological squeezing of a fossil under sediment; simulated by `deformation`. |
| **Erosion** | Removal of material by weathering/water; simulated by `erotion_general`. |
| **Fracture** | Cracks propagated through the volume; simulated by `propagator_fracture`. |
| **Marching Cubes** | Algorithm converting a scalar field into a triangle surface (used for smooth reconstruction). |
| **Mesh** | Surface representation made of vertices and triangles (`.ply`, `.obj`). |
| **`.npy`** | NumPy's native binary array format; how PaleoVox stores voxel grids. |
| **PBR** | Physically Based Rendering; realistic material shading used by the 3D viewer. |
| **pymeshfix** | Library that repairs holes to make meshes watertight. |
| **Scale info** | Tuple returned by `mesh_to_voxel` that allows reconstruction at original size/position. |
| **SSOT** | Single Source of Truth — this document. |
| **t-SNE** | Dimensionality-reduction technique mapping high-dimensional points to a 2D map for visual inspection. |
| **Voxel grid (blocky) view** | Rendering each occupied voxel as a cube. |
| **Watertight** | A mesh with no holes, enclosing a well-defined volume. |
