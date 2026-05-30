# -*- mode: python ; coding: utf-8 -*-
import sys
import os

# Only the hidden imports actually needed by paleovoxpy.py and paleovox_gui.py
hiddenimports = [
    # SciPy (morphology, transforms)
    'scipy.ndimage',
    'scipy.ndimage._ni_support',
    'scipy.ndimage._nd_image',
    'scipy.ndimage._filters',
    'scipy.ndimage._morphology',
    'scipy.ndimage._interpolation',
    'scipy.ndimage._measurements',
    # NumPy internals
    'numpy.core._methods',
    'numpy.lib.format',
    # scikit-learn (only TSNE is used)
    'sklearn.manifold._t_sne',
    'sklearn.manifold._utils',
    'sklearn.neighbors._ball_tree',
    'sklearn.neighbors._kd_tree',
    'sklearn.neighbors._distance_metrics',
    'sklearn.utils._cython_blas',
    'sklearn.utils._openmp_helpers',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    # Matplotlib (Agg backend only)
    'matplotlib.backends.backend_agg',
    # Seaborn (style config)
    'seaborn',
    # Plotly + Dash (required by open3d.visualization.draw_plotly)
    'plotly.graph_objects',
    'plotly.express',
    'dash',
    'dash.dcc',
    'dash.html',
    # PyQt5 (GUI framework)
    'PyQt5.sip',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
]

# Data files to bundle
datas = [
    (os.path.abspath('logo/logo.png'), 'logo'),
    (os.path.abspath('paleovoxpy.py'), '.'),
]

a = Analysis(
    [os.path.abspath('paleovox_gui.py')],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'sqlalchemy',
        'PIL.ImageShow',
        'matplotlib.tests',
        'scipy.tests',
        'sklearn.tests',
        'open3d.examples',
        'open3d.cuda',
        'PyQt5.QtWebEngine',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.QtWebChannel',
        'PyQt5.QtQuick',
        'PyQt5.QtQuickWidgets',
        'PyQt5.QtQuick3D',
        'PyQt5.QtMultimedia',
        'PyQt5.QtMultimediaWidgets',
        'PyQt5.QtBluetooth',
        'PyQt5.QtSql',
        'PyQt5.QtTest',
        'PyQt5.QtXml',
        'PyQt5.QtXmlPatterns',
        'PyQt5.QtSensors',
        'PyQt5.QtSerialPort',
        'PyQt5.QtSvg',
        'PyQt5.QtHelp',
        'PyQt5.QtLocation',
        'PyQt5.QtNfc',
        'PyQt5.QtPositioning',
        'PyQt5.QtRemoteObjects',
        'PyQt5.QtTextToSpeech',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PaleoVox.bin',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PaleoVox',
)
