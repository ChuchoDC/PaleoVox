# Portable Build Notes

## PBR viewer runs in a subprocess (important for frozen builds)

The PBR mesh/voxel viewers (View Mesh, View Voxels with *Marching Cubes*,
Compare Meshes, Compare Voxels with *Marching Cubes*) now render in a dedicated
subprocess via `paleovox_viewer.py`.

**Why:** Open3D's `gui.Application` (Filament-based `O3DVisualizer`) is
single-use per process. Running `app.run()` a second time — after a viewer
window has been closed — segfaults the process. To support opening multiple
viewers in sequence, `paleovoxpy.visualize_mesh()` serializes the meshes to
temporary `.ply` files and spawns:

```
python paleovox_viewer.py '<json>'
```

Each viewer therefore gets a fresh `Application` instance.

## Impact on PyInstaller builds

In a frozen onefile build (`PaleoVox.bin`, see `PaleoVox.spec`):

- `sys.executable` is the `PaleoVox.bin` launcher, **not** a Python
  interpreter, so `[sys.executable, "paleovox_viewer.py", json]` will **not**
  work as-is.
- `paleovox_viewer.py` is **not** bundled; only `paleovoxpy.py` is listed in
  the spec's `datas`.

So the current subprocess approach must be adapted before shipping a portable
build. Options:

1. **Frozen `--view` mode.** Bundle `paleovox_viewer.py` (add it to `datas`) and
   spawn `PaleoVox.bin --view '<json>'` instead of `sys.executable`. Add an
   early branch in the app entry point that detects `--view` and runs the
   viewer logic, then exits.
2. **`multiprocessing` + `freeze_support()`.** Run the viewer in a child process
   via `multiprocessing` with `freeze_support()` at entry. Keep the viewer
   function in a module that is bundled.
3. **In-process fallback.** For frozen builds only, fall back to
   `o3d.visualization.draw_geometries` (legacy GLFW viewer) in-process, which
   can be re-opened repeatedly but loses PBR shading.

When building, also confirm Open3D's GUI/rendering plugins are collected by
PyInstaller (they are not in the spec's `hiddenimports` today).
