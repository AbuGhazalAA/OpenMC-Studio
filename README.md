# OpenMC Studio

> **A modern Integrated Development Environment (IDE) for OpenMC Monte Carlo simulations.**

**OpenMC Studio** is a graphical Integrated Development Environment (IDE) built to simplify reactor physics modeling, radiation transport simulations, shielding analysis, criticality calculations, and result visualization using the **OpenMC** Monte Carlo code.

Developed by **Dr. Ayman Abu Ghazal**, OpenMC Studio combines an intuitive graphical interface with the flexibility of the OpenMC Python API, allowing users to create, edit, visualize, execute, and analyze OpenMC models within a single application.

---

## 🚧 Project Status

> **Beta Release**

OpenMC Studio is currently under active development.
Core functionality is operational; however, some features are still being refined and expanded. Minor bugs or interface changes may occur between releases.

Feedback, bug reports, and feature suggestions are highly appreciated.

---

## ✨ Features

- Modern graphical user interface (GUI)
- Integrated Python script editor with syntax highlighting
- Automatic synchronization between GUI and Python scripts
- Interactive material editor
- Geometry builder and visualization tools
- 2D geometry plotting
- 3D voxel visualization
- **Real particle-track visualization** — source birth points, interaction/collision events, and trajectories rendered from OpenMC's native tracking output, with per-layer display toggles and animated playback
- Support for importing and editing legacy MCNP-based models
- One-click execution of OpenMC simulations
- Live console output and progress monitoring
- Automatic statepoint loading
- Tally extraction and visualization
- Gaussian Energy Broadening (GEB) calibration for HPGe pulse-height tallies

---

## 📋 Requirements

Before running OpenMC Studio, install:

- Python 3.10 or newer
- OpenMC Python API 0.15.x (tested with 0.15.3)
- HDF5 nuclear data libraries
- Correctly configured `OPENMC_CROSS_SECTIONS` environment variable

For a known-working set of exact dependency versions, see `requirements.txt`.

---

## 🚀 Installation

Clone the repository:

```
git clone https://github.com/AymanAbuGhazal/OpenMC-Studio.git
```

Go to the project directory:

```
cd OpenMC-Studio
```

Install the required packages:

```
pip install -r requirements.txt
```

Run the application:

```
python main.py
```

---

## 📦 Building a Standalone Executable

OpenMC Studio can be packaged into a standalone Windows executable using PyInstaller:

```
pip install pyinstaller pyinstaller-hooks-contrib
pyinstaller --clean OpenMC-Studio.spec
```

The build results appear in `dist/OpenMC-Studio/`.

> ⚠️ **Known issue — build with Python 3.11, not 3.12.** scipy has a confirmed
> incompatibility with PyInstaller specifically under Python 3.12
> (see [pyinstaller/pyinstaller#7992](https://github.com/pyinstaller/pyinstaller/issues/7992),
> closed as a scipy-side issue, not PyInstaller's). It surfaces as:
> ```
> NameError: name 'obj' is not defined
>   File "scipy/stats/_distn_infrastructure.py", line 370, in <module>
>     del obj
> ```
> This has not yet been fixed in scipy (confirmed present through at least
> scipy 1.18.0). Build from a Python 3.11 virtual environment to avoid it —
> Python 3.11 remains fully supported for running the application itself.

---

## 📖 Citation

If you use OpenMC Studio in research or academic publications, please cite:

```
Abu Ghazal, A. (2026).
OpenMC Studio: A Modern Integrated Development Environment for OpenMC Monte Carlo Simulations.
Computer Software.
https://github.com/AymanAbuGhazal/OpenMC-Studio
```

---

## 🤖 Development Note

The core logic, physics models, and architecture of OpenMC Studio were designed and directed by the author. AI assistance (Google Gemini and Claude) was used as a coding tool — to generate, structure, and debug the underlying Python and PySide6 code — under the author's direct oversight throughout development. This is disclosed in the interest of transparency, consistent with growing norms around AI-assisted development in scientific software.

---

## 📄 License

This project is released under the **MIT License**.

---

## 👨‍💻 Developer

**Dr. Ayman Abu Ghazal**
GitHub: https://github.com/AymanAbuGhazal

---

## 🤝 Contributing

Contributions, bug reports, feature requests, and suggestions are welcome.
If you encounter an issue or have an idea for improving OpenMC Studio, please open an Issue or submit a Pull Request.

---

## ⚠️ Disclaimer

OpenMC Studio is an independent open-source project developed by **Dr. Ayman Abu Ghazal**.
It is **not** an official product of the Jordan Atomic Energy Commission (JAEC) and does **not** represent the official policies, positions, or endorsements of the JAEC.
