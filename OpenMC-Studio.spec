# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build for OpenMC Studio.
#
#   pyinstaller --clean --noconfirm OpenMC-Studio.spec
#
# Produces a one-FOLDER build in dist/OpenMC-Studio/. That is deliberate:
# a one-file build of a PySide6 + matplotlib + openmc application unpacks
# several hundred megabytes into a temporary directory on every launch,
# which makes startup slow and trips antivirus scanners. The folder build
# starts quickly and updates cleanly.
#
# What this bundles: the graphical application and the openmc PYTHON
# package. It does NOT bundle the compiled `openmc` EXECUTABLE or any
# cross-section data -- those are a separate installation, and the app
# shells out to `openmc` on PATH for plotting, tracking and running. The
# Particle Tracks page checks for it and says so when it is missing.

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# Files the app opens by path at runtime rather than importing.
datas += [
    ('app_icon.ico', '.'),
    ('ui/assets', 'ui/assets'),
    ('material_library.json', '.'),
]

# collect_all pulls the package's data files and dynamic libraries too,
# which these four need: matplotlib its fonts and styles, scipy and h5py
# their compiled extensions, openmc its XML schemas.
for package in ('matplotlib', 'scipy', 'openmc', 'h5py'):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# The 3D voxel viewer is optional -- the page degrades gracefully without
# it -- so a missing pyvista must not fail the build.
for package in ('pyvista', 'pyvistaqt', 'vtk'):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        pass

# Imported indirectly and easy for the dependency scan to miss:
# pandas' Excel writers, matplotlib's Qt backend, and the app's own ui
# subpackages, several of which are imported by name.
hiddenimports += [
    'pandas', 'openpyxl', 'PIL', 'PIL.Image',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_agg',
    'scipy.signal', 'scipy.optimize',
]
hiddenimports += collect_submodules('ui')
hiddenimports += collect_submodules('core')

# Qt ships several large frameworks this app never opens. Excluding them
# takes a sizeable bite out of the build without changing behaviour.
excludes = [
    'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
    'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.QtQuick',
    'PySide6.QtQml', 'PySide6.QtMultimedia', 'PySide6.QtBluetooth',
    'PyQt5', 'PyQt6', 'tkinter',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OpenMC-Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is off: it compresses the DLLs but is a frequent cause of false
    # positives in antivirus software, and it slows the first launch.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='OpenMC-Studio',
)
