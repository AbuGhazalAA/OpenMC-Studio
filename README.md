# OpenMC Studio

> **A modern Integrated Development Environment (IDE) for OpenMC Monte Carlo simulations.**

**OpenMC Studio** is a graphical Integrated Development Environment (IDE) built to simplify reactor physics modeling, radiation transport simulations, shielding analysis, and criticality calculations using the **OpenMC** Monte Carlo code.

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
- OpenMC Python API 0.14 or newer
- HDF5 nuclear data libraries
- Correctly configured `OPENMC_CROSS_SECTIONS` environment variable

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

## 📖 Citation

If you use OpenMC Studio in research or academic publications, please cite:

```
Abu Ghazal, A. (2026).

OpenMC Studio: A Modern Integrated Development Environment for OpenMC Monte Carlo Simulations.

Computer Software.

https://github.com/AymanAbuGhazal/OpenMC-Studio
```

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
