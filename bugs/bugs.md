### 18 - WD-006 — Defaults restored after Reset All — RESOLVED
All 33 widget-value checks fail; only `chk use external` is restored.

**Defect:** `_on_reset()` in `paleovox_gui.py:870-896` clears internal state
(mesh/voxel/paths/labels) and resets `chk_use_external`, but it does **not**
restore any spinbox, combobox, or checkbox widget values. After changing widget
values and clicking "Reset All", the widgets retain their modified values
instead of returning to the documented defaults.

**Impact:** The "Reset All" control does not fully reset the GUI. Users who
reload a different file after Reset All inherit stale pipeline/augmentation/
view settings from the previous session, which contradicts the documented
behavior ("All return to documented defaults").

**Solution (applied):** in `_on_reset()`, re-apply the default values/indices
for all widgets listed in WD-001…WD-005. Added `setValue`/`setCurrentIndex`/
`setChecked` calls to restore the 33 widgets (all spinboxes, comboboxes, and
checkboxes) to their documented defaults: npoints=100000, dims=256,
dilate iter=2, def axis=Z, def factor=0.85, ero axis=X, ero inc=0.5,
rot x/y/z=0.0, frac max=10, frac both=unchecked, color=Blue, voxel mc=unchecked,
compare vis=Both, voxel vis=Both, voxel comp mc=unchecked, tsne pp=100,
tsne seed=42, tsne pct=0.5, tsne size=1.0, 2d axis=XY, 2d color=Blue,
2d marker=o, 2d size=1.0, orig color=Blue, orig marker=o, orig size=1.0,
orig alpha=1.0, curr color=Red, curr marker=^, curr size=1.0, curr alpha=1.0
(`chk_use_external` was already reset). Resetting `combo_color` to index 0 also
re-triggers `_on_color_changed`, restoring the display color to Blue.

**Verification:** `bugs/test_widget_defaults.py` — all 68 checks pass
(WD-001…WD-006).
