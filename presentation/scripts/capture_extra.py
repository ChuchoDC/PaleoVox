#!/usr/bin/env python3
"""Extra PaleoVox GUI captures for the congress presentation.

Captures the GUI in states not covered by the poster asset set:
  - gui_initial.png   window at startup (idle drop zone)
  - gui_loaded.png    skull.ply loaded (file info populated, actions enabled)
  - gui_about.png     "More about PaleoVox..." dialog

Usage:  python presentation/scripts/capture_extra.py
Output: presentation/assets/screenshots/*.png
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_SCALE_FACTOR", "2")

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QDialog

import paleovox_gui as gui

OUT = os.path.join(REPO, "presentation", "assets", "screenshots")
os.makedirs(OUT, exist_ok=True)


def snap(widget, name):
    path = os.path.join(OUT, name)
    pix = widget.grab()
    pix.save(path)
    print(f"[snap] {name}: {pix.width()}x{pix.height()}", flush=True)


def run():
    app = QApplication.instance()
    window = gui.PaleoVoxGUI()

    errors = []
    window._show_error = lambda title, msg: errors.append(f"{title}: {msg}")

    def pulse(ms=300):
        end = time.time() + ms / 1000.0
        while time.time() < end:
            app.processEvents()

    window.resize(1600, 1020)
    window.show()
    pulse(500)
    snap(window, "gui_initial.png")

    t0 = time.time()
    window.on_load_mesh(os.path.join(REPO, "test_data", "skull.ply"))
    pulse(600)
    print(f"[run] load skull.ply: {time.time() - t0:.1f}s", flush=True)
    snap(window, "gui_loaded.png")

    window._on_show_about()
    pulse(400)
    captured = False
    for widget in app.topLevelWidgets():
        if isinstance(widget, QDialog) and widget.isVisible():
            pulse(200)
            snap(widget, "gui_about.png")
            widget.accept()
            captured = True
            break
    print(f"[run] about dialog captured: {captured}", flush=True)

    if errors:
        print("[run] GUI reported errors:")
        for e in errors:
            print("   ", e)
    print("[run] done", flush=True)
    app.quit()


app = QApplication(sys.argv)
app.setApplicationName("PaleoVox")

font = QFont()
font.setPointSize(10)
app.setFont(font)

QTimer.singleShot(0, run)
app.exec_()
