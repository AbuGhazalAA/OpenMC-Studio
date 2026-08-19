# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build description for OpenMC Studio.

Build it on the machine that already RUNS the application, from the same
Python environment (PyInstaller freezes the interpreter and packages it
finds, so a Windows .exe can only be produced on Windows):

    pip install pyinstaller
    pyinstaller OpenMC-Studio.spec --noconfirm

The result is the folder  dist\\OpenMC-Studio\\  with OpenMC-Studio.exe
inside it. The whole folder ships together -- the .exe alone will not run.

Note that this packages the INTERFACE, not OpenMC's transport solver: the
solver still runs from the OpenMC/WSL installation configured in the app,
and the cross-section libraries stay where they are. Nothing about the
data paths changes when the interface is frozen.
"""
from PyInstaller.utils.hooks import collect_all

datas = [('app_icon.ico', '.')]
binaries = []
# main.py imports six directly to work around its meta-path importer, and
# nothing else imports it, so PyInstaller cannot see it from the graph.
hiddenimports = ['six']


def bundle(package, required=True):
    """Pull in one package with its data files, extensions and submodules.

    Wrapped so that an optional component that is not installed -- the 3D
    voxel viewer's VTK stack, typically -- produces one clear line in the
    build log and a build that still finishes, instead of a traceback out
    of collect_all that stops everything.
    """
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception as exc:
        if required:
            raise SystemExit(
                f"[OpenMC-Studio] required package '{package}' could not be "
                f"collected: {exc}\\n"
                f"Install it in this environment and build again.")
        print(f"[OpenMC-Studio] optional package '{package}' not found "
              f"({exc}) -- building without it. The parts of the interface "
              f"that use it will not work in this build.")
        return
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)


# Required: the application cannot start without these.
bundle('openmc')
bundle('scipy')
bundle('matplotlib')
bundle('h5py')
bundle('pandas')
bundle('uncertainties')       # openmc returns k as ufloat objects

# Optional: only the 3D voxel viewer needs them, and they are the largest
# part of the build by far.
bundle('pyvista', required=False)
bundle('pyvistaqt', required=False)
bundle('vtk', required=False)

# Never needed at runtime, and they drag in hundreds of megabytes through
# openmc's own optional imports.
excludes = [
    'tkinter', 'notebook', 'jupyterlab', 'jupyter_server', 'nbconvert',
    'nbformat', 'ipykernel', 'ipywidgets', 'IPython', 'pytest', 'sphinx',
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
    upx=True,
    # No console window. main.py therefore writes any startup failure to
    # startup_error.log beside the executable, since a windowed build has
    # no stderr for the traceback to reach.
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
    upx=True,
    upx_exclude=[],
    name='OpenMC-Studio',
)
