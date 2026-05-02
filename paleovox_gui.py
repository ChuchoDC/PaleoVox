#!/usr/bin/env python3

import sys
import os

os.environ['XDG_SESSION_TYPE'] = 'x11'

import threading
import numpy as np
import open3d as o3d
import paleovoxpy as pv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFileDialog, QStatusBar, QMessageBox,
    QSplitter, QFrame, QSizePolicy, QDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap


AXIS_MAP = {"X": 0, "Y": 1, "Z": 2}


class DropZone(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop .ply or .obj file here\n\n— or click to browse —")
        self.setWordWrap(True)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._apply_style("idle")

    def _apply_style(self, state):
        colors = {
            "idle": ("#aaa", "#f5f5f5", "#666"),
            "hover": ("#4a9eff", "#e8f1ff", "#4a9eff"),
            "loaded": ("#4caf50", "#e8f5e9", "#2e7d32"),
        }
        border, bg, fg = colors.get(state, colors["idle"])
        self.setStyleSheet(f"""
            QLabel {{
                border: 3px dashed {border};
                border-radius: 12px;
                padding: 16px;
                font-size: 14px;
                color: {fg};
                background: {bg};
            }}
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.ply', '.obj')):
                    event.acceptProposedAction()
                    self._apply_style("hover")
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._apply_style("idle")

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.ply', '.obj')):
                self._apply_style("loaded")
                self.file_dropped.emit(path)
                return

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Mesh", "",
            "3D Mesh Files (*.ply *.obj);;PLY Files (*.ply);;OBJ Files (*.obj)"
        )
        if path:
            self._apply_style("loaded")
            self.file_dropped.emit(path)


class PaleoVoxGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.mesh = None
        self.original_mesh = None
        self.reconstructed_mesh = None
        self.voxel = None
        self.bounds = None
        self.scale_info = None
        self.fracture_pattern = None
        self.voxel_display_color = (0.2, 0.2, 0.8)
        self._init_ui()
        self._update_button_states()

    def _init_ui(self):
        self.setWindowTitle("PaleoVox — Fossil Voxel Augmentation")
        self.setMinimumSize(1100, 700)
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo", "logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 18px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QPushButton {
                padding: 6px 14px;
                border: 1px solid #bbb;
                border-radius: 4px;
                background: #f8f8f8;
            }
            QPushButton:hover {
                background: #e0e8f0;
            }
            QPushButton:pressed {
                background: #c0d0e0;
            }
            QPushButton:disabled {
                color: #aaa;
                background: #f0f0f0;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 3px 6px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([350, 700])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._status("Ready — Drag & drop a .ply or .obj file to begin")

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo", "logo.png")
        if os.path.exists(logo_path):
            logo_lbl = QLabel()
            logo_pixmap = QPixmap(logo_path).scaledToWidth(200, Qt.SmoothTransformation)
            logo_lbl.setPixmap(logo_pixmap)
            logo_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_lbl)

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.on_load_mesh)
        layout.addWidget(self.drop_zone)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse)
        layout.addWidget(self.btn_browse)

        info_group = QGroupBox("File Info")
        info_layout = QVBoxLayout(info_group)

        self.lbl_path = QLabel("Path: —")
        self.lbl_path.setWordWrap(True)
        self.lbl_vertices = QLabel("Vertices: —")
        self.lbl_triangles = QLabel("Triangles: —")
        self.lbl_voxel_shape = QLabel("Voxel shape: —")
        self.lbl_voxel_occupied = QLabel("Occupied voxels: —")

        for lbl in [self.lbl_path, self.lbl_vertices, self.lbl_triangles,
                     self.lbl_voxel_shape, self.lbl_voxel_occupied]:
            info_layout.addWidget(lbl)

        layout.addWidget(info_group)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(reset_btn)

        layout.addStretch()

        btn_about = QPushButton("More about PaleoVox...")
        btn_about.clicked.connect(self._on_show_about)
        layout.addWidget(btn_about)

        return panel

    def _on_show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About PaleoVox")
        dialog.setMinimumSize(500, 350)
        layout = QVBoxLayout(dialog)

        about_text = (
            "PaleoVox is a Python library for 3D fossil data augmentation "
            "using voxel representations. It supports mesh-to-voxel conversion, "
            "morphological operations, damage simulations (erosion, deformation, "
            "fracture, rotation), and high-quality voxel-to-mesh reconstruction."
        )
        about_lbl = QLabel(about_text)
        about_lbl.setWordWrap(True)
        layout.addWidget(about_lbl)

        creators_lbl = QLabel("Creators:")
        creators_lbl.setStyleSheet("font-weight: bold; margin-top: 12px;")
        layout.addWidget(creators_lbl)

        creators_text = (
            "Author 1 — email@institution.edu — Institution\n"
            "Author 2 — email@institution.edu — Institution\n"
            "Author 3 — email@institution.edu — Institution\n"
            "Author 4 — email@institution.edu — Institution"
        )
        creators_info = QLabel(creators_text)
        creators_info.setWordWrap(True)
        creators_info.setStyleSheet("font-size: 10px; color: #555;")
        layout.addWidget(creators_info)

        layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.show()

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(self._build_pipeline_group())
        layout.addWidget(self._build_augment_group())
        layout.addWidget(self._build_reconstruction_group())
        layout.addWidget(self._build_view_save_group())
        layout.addStretch()
        return panel

    def _build_pipeline_group(self):
        group = QGroupBox("Pipeline")
        gl = QVBoxLayout(group)

        row1 = QHBoxLayout()
        self.btn_load = QPushButton("Load Mesh")
        self.btn_load.clicked.connect(self._on_load_clicked)
        row1.addWidget(self.btn_load)
        gl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("npoints:"))
        self.spin_npoints = QSpinBox()
        self.spin_npoints.setRange(100, 200000)
        self.spin_npoints.setSingleStep(1000)
        self.spin_npoints.setValue(10000)
        self.spin_npoints.setToolTip("Number of points sampled from mesh surface")
        row2.addWidget(self.spin_npoints)
        row2.addWidget(QLabel("dim:"))
        self.spin_dims = QSpinBox()
        self.spin_dims.setRange(32, 512)
        self.spin_dims.setValue(128)
        self.spin_dims.setToolTip("Voxel grid resolution (N×N×N)")
        row2.addWidget(self.spin_dims)
        self.btn_to_voxel = QPushButton("Mesh → Voxels")
        self.btn_to_voxel.clicked.connect(self.on_convert_to_voxels)
        row2.addWidget(self.btn_to_voxel)
        gl.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Iterations:"))
        self.spin_dilate_iter = QSpinBox()
        self.spin_dilate_iter.setRange(1, 10)
        self.spin_dilate_iter.setValue(2)
        self.spin_dilate_iter.setToolTip("Morphological dilation iterations")
        row3.addWidget(self.spin_dilate_iter)
        self.btn_dilate = QPushButton("Dilate Voxels")
        self.btn_dilate.clicked.connect(self.on_dilate)
        row3.addWidget(self.btn_dilate)
        gl.addLayout(row3)

        row4 = QHBoxLayout()
        self.btn_to_mesh = QPushButton("Voxels → Mesh")
        self.btn_to_mesh.clicked.connect(self.on_voxels_to_mesh)
        row4.addWidget(self.btn_to_mesh)
        gl.addLayout(row4)

        return group

    def _build_augment_group(self):
        group = QGroupBox("Augmentations (operate on voxels)")
        al = QVBoxLayout(group)

        def_row = QHBoxLayout()
        def_row.addWidget(QLabel("Deformation"))
        self.combo_def_axis = QComboBox()
        self.combo_def_axis.addItems(["X", "Y", "Z"])
        self.combo_def_axis.setCurrentIndex(2)
        def_row.addWidget(self.combo_def_axis)
        def_row.addWidget(QLabel("Factor:"))
        self.spin_def_factor = QDoubleSpinBox()
        self.spin_def_factor.setRange(0.10, 0.99)
        self.spin_def_factor.setSingleStep(0.05)
        self.spin_def_factor.setValue(0.85)
        self.spin_def_factor.setToolTip("Compaction factor (0.1–0.99)")
        def_row.addWidget(self.spin_def_factor)
        self.btn_deform = QPushButton("Apply")
        self.btn_deform.clicked.connect(self.on_deform)
        def_row.addWidget(self.btn_deform)
        al.addLayout(def_row)

        ero_row = QHBoxLayout()
        ero_row.addWidget(QLabel("Erosion   "))
        self.combo_ero_axis = QComboBox()
        self.combo_ero_axis.addItems(["X", "Y", "Z"])
        ero_row.addWidget(self.combo_ero_axis)
        ero_row.addWidget(QLabel("Increment:"))
        self.spin_ero_inc = QDoubleSpinBox()
        self.spin_ero_inc.setRange(0.01, 1.0)
        self.spin_ero_inc.setSingleStep(0.05)
        self.spin_ero_inc.setValue(0.5)
        self.spin_ero_inc.setToolTip("Minimum erosion increment (0.01–1.0)")
        ero_row.addWidget(self.spin_ero_inc)
        self.btn_erode = QPushButton("Apply")
        self.btn_erode.clicked.connect(self.on_erode)
        ero_row.addWidget(self.btn_erode)
        al.addLayout(ero_row)

        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation  "))
        rot_row.addWidget(QLabel("X\u00b0:"))
        self.spin_rot_x = QDoubleSpinBox()
        self.spin_rot_x.setRange(-180, 180)
        self.spin_rot_x.setDecimals(1)
        self.spin_rot_x.setToolTip("Rotation around X axis (degrees)")
        rot_row.addWidget(self.spin_rot_x)
        rot_row.addWidget(QLabel("Y\u00b0:"))
        self.spin_rot_y = QDoubleSpinBox()
        self.spin_rot_y.setRange(-180, 180)
        self.spin_rot_y.setDecimals(1)
        self.spin_rot_y.setToolTip("Rotation around Y axis (degrees)")
        rot_row.addWidget(self.spin_rot_y)
        rot_row.addWidget(QLabel("Z\u00b0:"))
        self.spin_rot_z = QDoubleSpinBox()
        self.spin_rot_z.setRange(-180, 180)
        self.spin_rot_z.setDecimals(1)
        self.spin_rot_z.setToolTip("Rotation around Z axis (degrees)")
        rot_row.addWidget(self.spin_rot_z)
        self.btn_rotate = QPushButton("Apply")
        self.btn_rotate.clicked.connect(self.on_rotate)
        rot_row.addWidget(self.btn_rotate)
        al.addLayout(rot_row)

        frac_row = QHBoxLayout()
        frac_row.addWidget(QLabel("Fracture "))
        frac_row.addWidget(QLabel("Max pos:"))
        self.spin_frac_max = QSpinBox()
        self.spin_frac_max.setRange(1, 50)
        self.spin_frac_max.setValue(10)
        self.spin_frac_max.setToolTip("Max propagation candidates per step")
        frac_row.addWidget(self.spin_frac_max)
        self.chk_frac_both = QCheckBox("Return both")
        self.chk_frac_both.setToolTip("Also return fracture pattern separately")
        frac_row.addWidget(self.chk_frac_both)
        self.btn_fracture = QPushButton("Apply")
        self.btn_fracture.clicked.connect(self.on_fracture)
        frac_row.addWidget(self.btn_fracture)
        al.addLayout(frac_row)

        return group

    def _build_view_save_group(self):
        group = QGroupBox("View & Save")
        vsl = QVBoxLayout(group)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Color:"))
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Blue", "Red", "Green", "Orange", "Purple", "Cyan", "Yellow", "White", "Gray"])
        self.combo_color.setCurrentIndex(0)
        self.combo_color.setToolTip("Display color for mesh and voxel viewers")
        self.combo_color.currentIndexChanged.connect(self._on_color_changed)
        top_row.addWidget(self.combo_color)
        top_row.addStretch()
        vsl.addLayout(top_row)

        row = QHBoxLayout()
        self.btn_view_mesh = QPushButton("View Mesh")
        self.btn_view_mesh.clicked.connect(self.on_view_mesh)
        row.addWidget(self.btn_view_mesh)

        self.btn_view_voxel = QPushButton("View Voxels")
        self.btn_view_voxel.clicked.connect(self.on_view_voxels)
        row.addWidget(self.btn_view_voxel)

        self.btn_save_mesh = QPushButton("Save Mesh")
        self.btn_save_mesh.clicked.connect(self.on_save_mesh)
        row.addWidget(self.btn_save_mesh)

        self.btn_save_voxel = QPushButton("Save Voxel")
        self.btn_save_voxel.clicked.connect(self.on_save_voxel)
        row.addWidget(self.btn_save_voxel)
        vsl.addLayout(row)

        return group

    def _build_reconstruction_group(self):
        group = QGroupBox("Reconstruction & Comparison")
        rl = QVBoxLayout(group)

        info_row = QHBoxLayout()
        self.lbl_recon_info = QLabel("Reconstructed: —")
        info_row.addWidget(self.lbl_recon_info)
        info_row.addStretch()
        rl.addLayout(info_row)

        btn_row1 = QHBoxLayout()
        self.btn_reconstruct = QPushButton("Reconstruct from Voxels")
        self.btn_reconstruct.clicked.connect(self._on_reconstruct)
        btn_row1.addWidget(self.btn_reconstruct)

        self.btn_compare = QPushButton("Compare Original vs Reconstructed")
        self.btn_compare.clicked.connect(self._on_compare_meshes)
        btn_row1.addWidget(self.btn_compare)
        rl.addLayout(btn_row1)

        compare_opts = QHBoxLayout()
        compare_opts.addWidget(QLabel("Show:"))
        self.combo_compare_vis = QComboBox()
        self.combo_compare_vis.addItems(["Both", "Original Only", "Reconstructed Only"])
        self.combo_compare_vis.setCurrentIndex(0)
        compare_opts.addWidget(self.combo_compare_vis)
        compare_opts.addStretch()
        rl.addLayout(compare_opts)

        btn_row2 = QHBoxLayout()
        self.btn_save_reconstructed = QPushButton("Save Reconstructed Mesh")
        self.btn_save_reconstructed.clicked.connect(self._on_save_reconstructed)
        btn_row2.addWidget(self.btn_save_reconstructed)
        btn_row2.addStretch()
        rl.addLayout(btn_row2)

        return group

    def _on_color_changed(self):
        color_map = {
            "Blue": (0.2, 0.2, 0.8),
            "Red": (0.8, 0.2, 0.2),
            "Green": (0.2, 0.8, 0.2),
            "Orange": (1.0, 0.6, 0.0),
            "Purple": (0.6, 0.2, 0.8),
            "Cyan": (0.0, 0.8, 0.8),
            "Yellow": (1.0, 1.0, 0.0),
            "White": (0.9, 0.9, 0.9),
            "Gray": (0.5, 0.5, 0.5),
        }
        self.voxel_display_color = color_map.get(
            self.combo_color.currentText(), (0.2, 0.2, 0.8)
        )

    def _status(self, msg):
        self.status_bar.showMessage(msg)

    def _update_info_panel(self):
        if self.file_path:
            home = os.path.expanduser("~")
            display = self.file_path.replace(home, "~")
            self.lbl_path.setText(f"Path: {display}")
        else:
            self.lbl_path.setText("Path: —")

        if self.mesh is not None:
            verts = len(self.mesh.vertices)
            tris = len(self.mesh.triangles)
            self.lbl_vertices.setText(f"Vertices: {verts:,}")
            self.lbl_triangles.setText(f"Triangles: {tris:,}")
        else:
            self.lbl_vertices.setText("Vertices: —")
            self.lbl_triangles.setText("Triangles: —")

        if self.voxel is not None:
            shape = self.voxel.shape
            occupied = int(np.sum(self.voxel > 0))
            self.lbl_voxel_shape.setText(f"Voxel shape: {shape}")
            self.lbl_voxel_occupied.setText(f"Occupied voxels: {occupied:,}")
        else:
            self.lbl_voxel_shape.setText("Voxel shape: —")
            self.lbl_voxel_occupied.setText("Occupied voxels: —")

    def _update_button_states(self):
        has_mesh = self.mesh is not None
        has_voxel = self.voxel is not None
        has_path = self.file_path is not None
        has_reconstructed = self.reconstructed_mesh is not None
        has_original = self.original_mesh is not None

        self.btn_load.setEnabled(has_path)
        self.btn_to_voxel.setEnabled(has_mesh)
        self.btn_dilate.setEnabled(has_voxel)
        self.btn_to_mesh.setEnabled(has_voxel)
        self.btn_deform.setEnabled(has_voxel)
        self.btn_erode.setEnabled(has_voxel)
        self.btn_rotate.setEnabled(has_voxel)
        self.btn_fracture.setEnabled(has_voxel)
        self.btn_view_mesh.setEnabled(has_mesh or has_reconstructed)
        self.btn_view_voxel.setEnabled(has_voxel)
        self.btn_save_mesh.setEnabled(has_mesh or has_reconstructed)
        self.btn_save_voxel.setEnabled(has_voxel)
        self.btn_reconstruct.setEnabled(has_voxel)
        self.btn_compare.setEnabled(has_original and has_reconstructed)
        self.btn_save_reconstructed.setEnabled(has_reconstructed)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Mesh", "",
            "3D Mesh Files (*.ply *.obj);;PLY Files (*.ply);;OBJ Files (*.obj)"
        )
        if path:
            self.on_load_mesh(path)

    def _on_load_clicked(self):
        if self.file_path:
            self.on_load_mesh(self.file_path)

    def _on_reset(self):
        self.file_path = None
        self.mesh = None
        self.original_mesh = None
        self.reconstructed_mesh = None
        self.voxel = None
        self.bounds = None
        self.scale_info = None
        self.fracture_pattern = None
        self.lbl_recon_info.setText("Reconstructed: —")
        self.drop_zone._apply_style("idle")
        self.drop_zone.setText("Drop .ply or .obj file here\n\n— or click to browse —")
        self._update_info_panel()
        self._update_button_states()
        self._status("Reset — ready for new file")

    def _show_error(self, title, msg):
        QMessageBox.critical(self, title, str(msg))

    def on_load_mesh(self, path):
        try:
            self.file_path = path
            self.mesh, min_bound, max_bound, dimensions = pv.load_mesh(
                path, return_bounds=True
            )
            self.original_mesh = o3d.geometry.TriangleMesh(self.mesh)
            self.reconstructed_mesh = None
            self.bounds = (min_bound, max_bound, dimensions)
            self.voxel = None
            self.scale_info = None
            self.fracture_pattern = None

            fname = os.path.basename(path)
            self.drop_zone.setText(f"Loaded:\n{fname}")
            self.drop_zone._apply_style("loaded")

            self._update_info_panel()
            self._update_button_states()
            verts = len(self.mesh.vertices)
            tris = len(self.mesh.triangles)
            self._status(f"Loaded {fname} — {verts:,} vertices, {tris:,} triangles")
        except Exception as e:
            self._show_error("Load Error", f"Failed to load mesh:\n{e}")

    def on_convert_to_voxels(self):
        if self.mesh is None:
            return
        try:
            npoints = self.spin_npoints.value()
            dims = self.spin_dims.value()
            self._status(f"Converting mesh to voxels (npoints={npoints}, dim={dims})...")
            result = pv.mesh_to_voxel(
                self.mesh, npoints=npoints, dimensions=dims,
                return_scale_info=True
            )
            self.voxel = result[0]
            self.scale_info = result[1:]
            self.fracture_pattern = None
            self._update_info_panel()
            self._update_button_states()
            occupied = int(np.sum(self.voxel > 0))
            self._status(f"Voxel grid {self.voxel.shape} — {occupied:,} occupied voxels")
        except Exception as e:
            self._show_error("Conversion Error", f"Failed to convert mesh to voxels:\n{e}")

    def on_dilate(self):
        if self.voxel is None:
            return
        try:
            iterations = self.spin_dilate_iter.value()
            self._status(f"Dilating voxels ({iterations} iterations)...")
            self.voxel = pv.binary_dilation(self.voxel, iterations=iterations)
            self.fracture_pattern = None
            self._update_info_panel()
            occupied = int(np.sum(self.voxel > 0))
            self._status(f"Dilation complete — {occupied:,} occupied voxels")
        except Exception as e:
            self._show_error("Dilation Error", f"Failed to dilate voxels:\n{e}")

    def on_voxels_to_mesh(self):
        if self.voxel is None:
            return
        try:
            self._status("Reconstructing mesh from voxels...")
            target_scale = None
            original_bounds = None
            if self.scale_info is not None:
                _, orig_min, orig_max, _ = self.scale_info
                target_scale = orig_max - orig_min
                original_bounds = (orig_min, orig_max)
            self.mesh = pv.high_quality_voxel_to_mesh(
                self.voxel, voxel_size=1.0,
                target_scale=target_scale,
                original_bounds=original_bounds
            )
            self.reconstructed_mesh = self.mesh
            self._update_info_panel()
            self._update_button_states()
            verts = len(self.mesh.vertices)
            tris = len(self.mesh.triangles)
            self.lbl_recon_info.setText(
                f"Reconstructed: {verts:,} verts, {tris:,} tris"
            )
            self._status(f"Mesh reconstructed — {verts:,} vertices, {tris:,} triangles")
        except Exception as e:
            self._show_error("Reconstruction Error",
                           f"Failed to reconstruct mesh:\n{e}")

    def on_deform(self):
        if self.voxel is None:
            return
        try:
            axis_label = self.combo_def_axis.currentText()
            axis_idx = AXIS_MAP[axis_label]
            factor = self.spin_def_factor.value()
            self._status(f"Applying deformation (axis={axis_label}, factor={factor:.2f})...")
            self.voxel = pv.deformation(
                self.voxel, compaction_factor=factor,
                compaction_axis=axis_idx
            )
            self.fracture_pattern = None
            self._update_info_panel()
            occupied = int(np.sum(self.voxel > 0))
            self._status(f"Deformation applied — {occupied:,} occupied voxels")
        except Exception as e:
            self._show_error("Deformation Error", f"Failed to apply deformation:\n{e}")

    def on_erode(self):
        if self.voxel is None:
            return
        try:
            axis_label = self.combo_ero_axis.currentText()
            axis_idx = AXIS_MAP[axis_label]
            increment = self.spin_ero_inc.value()
            self._status(f"Applying erosion (axis={axis_label}, increment={increment:.2f})...")
            self.voxel = pv.erotion_general(
                self.voxel, axis_idx=axis_idx,
                increment_min=increment
            )
            self.fracture_pattern = None
            self._update_info_panel()
            occupied = int(np.sum(self.voxel > 0))
            self._status(f"Erosion applied — {occupied:,} occupied voxels")
        except Exception as e:
            self._show_error("Erosion Error", f"Failed to apply erosion:\n{e}")

    def on_rotate(self):
        if self.voxel is None:
            return
        try:
            rx = float(np.radians(self.spin_rot_x.value()))
            ry = float(np.radians(self.spin_rot_y.value()))
            rz = float(np.radians(self.spin_rot_z.value()))
            self._status(f"Applying rotation (X={self.spin_rot_x.value():.1f}\u00b0, "
                        f"Y={self.spin_rot_y.value():.1f}\u00b0, "
                        f"Z={self.spin_rot_z.value():.1f}\u00b0)...")
            self.voxel = pv.rotate_voxel(self.voxel, rx, ry, rz)
            self.fracture_pattern = None
            self._update_info_panel()
            occupied = int(np.sum(self.voxel > 0))
            self._status(f"Rotation applied — {occupied:,} occupied voxels")
        except Exception as e:
            self._show_error("Rotation Error", f"Failed to apply rotation:\n{e}")

    def on_fracture(self):
        if self.voxel is None:
            return
        try:
            max_pos = self.spin_frac_max.value()
            return_both = self.chk_frac_both.isChecked()
            self._status(f"Applying fracture (max_pos={max_pos})...")
            result = pv.propagator_fracture(
                self.voxel, max_position=max_pos,
                return_both=return_both
            )
            if return_both:
                self.voxel = result[0]
                self.fracture_pattern = result[1]
                pat_occ = int(np.sum(self.fracture_pattern > 0))
                self._status(f"Fracture applied — pattern has {pat_occ:,} voxels")
            else:
                self.voxel = result
                self.fracture_pattern = None
                occupied = int(np.sum(self.voxel > 0))
                self._status(f"Fracture applied — {occupied:,} occupied voxels")
            self._update_info_panel()
        except Exception as e:
            self._show_error("Fracture Error", f"Failed to apply fracture:\n{e}")

    def on_view_mesh(self):
        mesh = self.reconstructed_mesh if self.reconstructed_mesh is not None else self.mesh
        if mesh is not None:
            mesh_copy = o3d.geometry.TriangleMesh(mesh)
            mesh_copy.paint_uniform_color(self.voxel_display_color)
            threading.Thread(
                target=lambda: o3d.visualization.draw_geometries(
                    [mesh_copy], window_name="PaleoVox — Mesh View",
                    width=1024, height=768
                ),
                daemon=True
            ).start()

    def on_view_voxels(self):
        if self.voxel is not None:
            occupied = np.argwhere(self.voxel > 0)
            if len(occupied) == 0:
                self._status("No occupied voxels to display")
                return
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(occupied.astype(np.float64))
            pcd.paint_uniform_color(self.voxel_display_color)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                pcd, voxel_size=1.0
            )
            threading.Thread(
                target=lambda: o3d.visualization.draw_geometries(
                    [voxel_grid], window_name="PaleoVox — Voxel View",
                    width=1024, height=768
                ),
                daemon=True
            ).start()

    def _on_reconstruct(self):
        if self.voxel is None:
            return
        try:
            self._status("Reconstructing mesh from voxels...")
            target_scale = None
            original_bounds = None
            if self.scale_info is not None:
                _, orig_min, orig_max, _ = self.scale_info
                target_scale = orig_max - orig_min
                original_bounds = (orig_min, orig_max)
            self.reconstructed_mesh = pv.high_quality_voxel_to_mesh(
                self.voxel, voxel_size=1.0,
                target_scale=target_scale,
                original_bounds=original_bounds
            )
            self.mesh = self.reconstructed_mesh
            verts = len(self.reconstructed_mesh.vertices)
            tris = len(self.reconstructed_mesh.triangles)
            self.lbl_recon_info.setText(
                f"Reconstructed: {verts:,} verts, {tris:,} tris"
            )
            self._update_info_panel()
            self._update_button_states()
            self._status(f"Mesh reconstructed — {verts:,} vertices, {tris:,} triangles")
        except Exception as e:
            self._show_error("Reconstruction Error",
                           f"Failed to reconstruct mesh:\n{e}")

    def _on_compare_meshes(self):
        if self.original_mesh is None or self.reconstructed_mesh is None:
            return
        try:
            vis_mode = self.combo_compare_vis.currentText()
            geoms = []
            title = "PaleoVox — "
            if vis_mode in ("Both", "Original Only"):
                orig = o3d.geometry.TriangleMesh(self.original_mesh)
                orig.paint_uniform_color((0.2, 0.2, 0.8))
                geoms.append(orig)
                title += "Original (Blue)"
            if vis_mode in ("Both", "Reconstructed Only"):
                recon = o3d.geometry.TriangleMesh(self.reconstructed_mesh)
                recon.paint_uniform_color((0.8, 0.2, 0.2))
                geoms.append(recon)
                if vis_mode == "Both":
                    title += " vs "
                title += "Reconstructed (Red)"
            if not geoms:
                return
            self._status(f"Displaying comparison: {vis_mode}")
            threading.Thread(
                target=lambda: o3d.visualization.draw_geometries(
                    geoms, window_name=title,
                    width=1024, height=768
                ),
                daemon=True
            ).start()
        except Exception as e:
            self._show_error("Comparison Error", f"Failed to compare meshes:\n{e}")

    def _on_save_reconstructed(self):
        if self.reconstructed_mesh is not None:
            default_name = ""
            if self.file_path:
                base = os.path.splitext(os.path.basename(self.file_path))[0]
                default_name = f"{base}_reconstructed.ply"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Reconstructed Mesh", default_name,
                "PLY Files (*.ply);;OBJ Files (*.obj)"
            )
            if path:
                try:
                    pv.save_mesh(self.reconstructed_mesh, path)
                    self._status(f"Reconstructed mesh saved to {path}")
                except Exception as e:
                    self._show_error("Save Error", f"Failed to save mesh:\n{e}")

    def on_save_mesh(self):
        if self.mesh is not None:
            default_name = ""
            if self.file_path:
                base = os.path.splitext(os.path.basename(self.file_path))[0]
                default_name = f"{base}_processed.ply"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Mesh", default_name,
                "PLY Files (*.ply);;OBJ Files (*.obj)"
            )
            if path:
                try:
                    pv.save_mesh(self.mesh, path)
                    self._status(f"Mesh saved to {path}")
                except Exception as e:
                    self._show_error("Save Error", f"Failed to save mesh:\n{e}")

    def on_save_voxel(self):
        if self.voxel is not None:
            default_name = ""
            if self.file_path:
                base = os.path.splitext(os.path.basename(self.file_path))[0]
                default_name = f"{base}_voxel.npy"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Voxel", default_name,
                "NumPy Files (*.npy)"
            )
            if path:
                try:
                    pv.save_voxel(self.voxel, path)
                    self._status(f"Voxel saved to {path}")
                except Exception as e:
                    self._show_error("Save Error", f"Failed to save voxel:\n{e}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PaleoVox")
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    window = PaleoVoxGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
