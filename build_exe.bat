@echo off
REM ============================================================
REM  OpenMC Studio - build the Windows executable
REM
REM  Run this from the project folder, in the SAME Python
REM  environment that already runs the application (the one with
REM  openmc, PySide6, pyvista installed). In PyCharm: open the
REM  Terminal tab and type  build_exe.bat
REM
REM  Result:  dist\OpenMC-Studio\OpenMC-Studio.exe
REM  Ship the whole dist\OpenMC-Studio folder, not the .exe alone.
REM ============================================================

echo.
echo === Checking PyInstaller ===
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller is not installed in this environment. Installing it now...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo.
        echo Could not install PyInstaller. Build stopped.
        exit /b 1
    )
)

echo.
echo === Checking that the application's own libraries are visible ===
python -c "import openmc, PySide6, h5py, pandas, matplotlib, scipy; print('  all core libraries found')"
if errorlevel 1 (
    echo.
    echo This Python environment cannot import the application's libraries,
    echo so the build would produce an executable that cannot start.
    echo Activate the environment that runs OpenMC Studio and try again.
    exit /b 1
)

echo.
echo === Building (this takes several minutes) ===
python -m PyInstaller OpenMC-Studio.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo Build failed. The log above names the module that could not be packaged.
    exit /b 1
)

echo.
echo === Done ===
echo Executable: %CD%\dist\OpenMC-Studio\OpenMC-Studio.exe
echo Ship the whole folder: %CD%\dist\OpenMC-Studio\
echo.
echo If the application fails to start, read startup_error.log written
echo next to the executable - it names the missing piece.
