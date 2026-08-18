#!/usr/bin/env python3
"""QA Section 18 — Widget Default Value Tests (WD-001 .. WD-006).

Headless harness: instantiates PaleoVoxGUI offscreen and asserts every
documented widget default, plus Reset All behaviour (WD-006).
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PyQt5.QtWidgets import QApplication

import paleovox_gui as gui_mod


def get_str(w):
    try:
        return w.currentText()
    except Exception:
        return None


def get_val(w):
    try:
        return w.value()
    except Exception:
        return None


def get_checked(w):
    try:
        return w.isChecked()
    except Exception:
        return None


RESULTS = []


def record(case, check, expected, actual):
    ok = actual == expected
    RESULTS.append((case, check, expected, actual, ok))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {case} :: {check}  (expected={expected!r}, actual={actual!r})")
    return ok


def main():
    app = QApplication(sys.argv)
    win = gui_mod.PaleoVoxGUI()

    # ---------------------------------------------------------------- WD-001
    record("WD-001", "npoints", 100000, get_val(win.spin_npoints))
    record("WD-001", "dims", 256, get_val(win.spin_dims))
    record("WD-001", "dilate iter", 2, get_val(win.spin_dilate_iter))
    record("WD-001", "def axis", "Z", get_str(win.combo_def_axis))
    record("WD-001", "def factor", 0.85, get_val(win.spin_def_factor))
    record("WD-001", "ero axis", "X", get_str(win.combo_ero_axis))
    record("WD-001", "ero inc", 0.5, get_val(win.spin_ero_inc))
    record("WD-001", "rot x", 0.0, get_val(win.spin_rot_x))
    record("WD-001", "rot y", 0.0, get_val(win.spin_rot_y))
    record("WD-001", "rot z", 0.0, get_val(win.spin_rot_z))
    record("WD-001", "frac max", 10, get_val(win.spin_frac_max))
    record("WD-001", "frac both", False, get_checked(win.chk_frac_both))

    # ---------------------------------------------------------------- WD-002
    record("WD-002", "color", "Blue", get_str(win.combo_color))
    record("WD-002", "voxel mc", False, get_checked(win.chk_voxel_mc))

    # ---------------------------------------------------------------- WD-003
    record("WD-003", "compare vis", "Both", get_str(win.combo_compare_vis))
    record("WD-003", "voxel vis", "Both", get_str(win.combo_voxel_vis))
    record("WD-003", "voxel comp mc", False, get_checked(win.chk_voxel_comp_mc))

    # ---------------------------------------------------------------- WD-004
    record("WD-004", "tsne pp", 100, get_val(win.spin_tsne_pp))
    record("WD-004", "tsne seed", 42, get_val(win.spin_tsne_seed))
    record("WD-004", "tsne pct", 0.5, get_val(win.spin_tsne_pct))
    record("WD-004", "tsne size", 1.0, get_val(win.spin_tsne_size))

    # ---------------------------------------------------------------- WD-005
    record("WD-005", "2d axis", "XY", get_str(win.combo_2d_axis))
    record("WD-005", "2d color", "Blue", get_str(win.combo_2d_color))
    record("WD-005", "2d marker", "o", get_str(win.combo_2d_marker))
    record("WD-005", "2d size", 1.0, get_val(win.spin_2d_size))
    record("WD-005", "orig color", "Blue", get_str(win.combo_2d_cmp_color1))
    record("WD-005", "orig marker", "o", get_str(win.combo_2d_cmp_marker1))
    record("WD-005", "orig size", 1.0, get_val(win.spin_2d_cmp_size1))
    record("WD-005", "orig alpha", 1.0, get_val(win.spin_2d_cmp_alpha1))
    record("WD-005", "curr color", "Red", get_str(win.combo_2d_cmp_color2))
    record("WD-005", "curr marker", "^", get_str(win.combo_2d_cmp_marker2))
    record("WD-005", "curr size", 1.0, get_val(win.spin_2d_cmp_size2))
    record("WD-005", "curr alpha", 1.0, get_val(win.spin_2d_cmp_alpha2))
    record("WD-005", "use external", False, get_checked(win.chk_use_external))

    # ---------------------------------------------------------------- WD-006
    # Mutate widget values away from defaults, then Reset All, then re-read.
    win.spin_npoints.setValue(50000)
    win.spin_dims.setValue(128)
    win.spin_dilate_iter.setValue(5)
    win.combo_def_axis.setCurrentIndex(0)
    win.spin_def_factor.setValue(0.25)
    win.combo_ero_axis.setCurrentIndex(2)
    win.spin_ero_inc.setValue(0.99)
    win.spin_rot_x.setValue(45.0)
    win.spin_rot_y.setValue(-30.0)
    win.spin_rot_z.setValue(90.0)
    win.spin_frac_max.setValue(50)
    win.chk_frac_both.setChecked(True)
    win.combo_color.setCurrentIndex(3)
    win.chk_voxel_mc.setChecked(True)
    win.combo_compare_vis.setCurrentIndex(2)
    win.combo_voxel_vis.setCurrentIndex(1)
    win.chk_voxel_comp_mc.setChecked(True)
    win.spin_tsne_pp.setValue(50)
    win.spin_tsne_seed.setValue(7)
    win.spin_tsne_pct.setValue(0.05)
    win.spin_tsne_size.setValue(5.0)
    win.combo_2d_axis.setCurrentIndex(2)
    win.combo_2d_color.setCurrentIndex(5)
    win.combo_2d_marker.setCurrentIndex(3)
    win.spin_2d_size.setValue(8.0)
    win.combo_2d_cmp_color1.setCurrentIndex(4)
    win.combo_2d_cmp_marker1.setCurrentIndex(6)
    win.spin_2d_cmp_size1.setValue(3.0)
    win.spin_2d_cmp_alpha1.setValue(0.2)
    win.combo_2d_cmp_color2.setCurrentIndex(7)
    win.combo_2d_cmp_marker2.setCurrentIndex(4)
    win.spin_2d_cmp_size2.setValue(4.0)
    win.spin_2d_cmp_alpha2.setValue(0.3)
    win.chk_use_external.setChecked(True)

    win._on_reset()

    record("WD-006", "npoints restored", 100000, get_val(win.spin_npoints))
    record("WD-006", "dims restored", 256, get_val(win.spin_dims))
    record("WD-006", "dilate iter restored", 2, get_val(win.spin_dilate_iter))
    record("WD-006", "def axis restored", "Z", get_str(win.combo_def_axis))
    record("WD-006", "def factor restored", 0.85, get_val(win.spin_def_factor))
    record("WD-006", "ero axis restored", "X", get_str(win.combo_ero_axis))
    record("WD-006", "ero inc restored", 0.5, get_val(win.spin_ero_inc))
    record("WD-006", "rot x restored", 0.0, get_val(win.spin_rot_x))
    record("WD-006", "rot y restored", 0.0, get_val(win.spin_rot_y))
    record("WD-006", "rot z restored", 0.0, get_val(win.spin_rot_z))
    record("WD-006", "frac max restored", 10, get_val(win.spin_frac_max))
    record("WD-006", "frac both restored", False, get_checked(win.chk_frac_both))
    record("WD-006", "color restored", "Blue", get_str(win.combo_color))
    record("WD-006", "voxel mc restored", False, get_checked(win.chk_voxel_mc))
    record("WD-006", "compare vis restored", "Both", get_str(win.combo_compare_vis))
    record("WD-006", "voxel vis restored", "Both", get_str(win.combo_voxel_vis))
    record("WD-006", "voxel comp mc restored", False, get_checked(win.chk_voxel_comp_mc))
    record("WD-006", "tsne pp restored", 100, get_val(win.spin_tsne_pp))
    record("WD-006", "tsne seed restored", 42, get_val(win.spin_tsne_seed))
    record("WD-006", "tsne pct restored", 0.5, get_val(win.spin_tsne_pct))
    record("WD-006", "tsne size restored", 1.0, get_val(win.spin_tsne_size))
    record("WD-006", "2d axis restored", "XY", get_str(win.combo_2d_axis))
    record("WD-006", "2d color restored", "Blue", get_str(win.combo_2d_color))
    record("WD-006", "2d marker restored", "o", get_str(win.combo_2d_marker))
    record("WD-006", "2d size restored", 1.0, get_val(win.spin_2d_size))
    record("WD-006", "orig color restored", "Blue", get_str(win.combo_2d_cmp_color1))
    record("WD-006", "orig marker restored", "o", get_str(win.combo_2d_cmp_marker1))
    record("WD-006", "orig size restored", 1.0, get_val(win.spin_2d_cmp_size1))
    record("WD-006", "orig alpha restored", 1.0, get_val(win.spin_2d_cmp_alpha1))
    record("WD-006", "curr color restored", "Red", get_str(win.combo_2d_cmp_color2))
    record("WD-006", "curr marker restored", "^", get_str(win.combo_2d_cmp_marker2))
    record("WD-006", "curr size restored", 1.0, get_val(win.spin_2d_cmp_size2))
    record("WD-006", "curr alpha restored", 1.0, get_val(win.spin_2d_cmp_alpha2))
    record("WD-006", "use external restored", False, get_checked(win.chk_use_external))

    # ---------------------------------------------------------------- summary
    total = len(RESULTS)
    passed = sum(1 for _, _, _, _, ok in RESULTS if ok)
    failed = total - passed
    print("\n================ SUMMARY ================")
    print(f"Total checks: {total}  Passed: {passed}  Failed: {failed}")
    cases = {}
    for case, _, _, _, ok in RESULTS:
        cases.setdefault(case, True)
        cases[case] = cases[case] and ok
    for case in ["WD-001", "WD-002", "WD-003", "WD-004", "WD-005", "WD-006"]:
        print(f"  {case}: {'PASS' if cases[case] else 'FAIL'}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
