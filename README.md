# OpenMC Studio

**OpenMC Studio** is a dedicated graphical user interface (GUI) and integrated development environment (IDE) designed to streamline reactor physics modeling, radiation shielding simulations, and criticality calculations using the **OpenMC** Monte Carlo engine.

Developed by **Dr. Ayman Abu Ghazal**, OpenMC Studio bridges the gap between Python-based OpenMC scripting and an intuitive visual workflow. It is designed for researchers, students, and nuclear engineers, providing powerful tools for geometry creation, material definition, simulation management, and results analysis.

---

## 🌟 Key Features

- **Python Script Engine**
  - Full-featured script editor with syntax highlighting.
  - Real-time synchronization between the GUI and Python scripts.
  - Safe execution environment.

- **Geometry & Material Builders**
  - Interactive creation of materials and compositions.
  - Visual construction of reactor and shielding geometries.
  - Support for models translated from legacy MCNP input files.

- **Visualization**
  - Built-in 2D geometry plotting.
  - Advanced voxel visualization.
  - Interactive model inspection.

- **Simulation Management**
  - Run eigenvalue and fixed-source simulations.
  - Live output console.
  - Progress monitoring and estimated completion time.

- **Results & Post-Processing**
  - Automatic statepoint loading.
  - Tally extraction and visualization.
  - Gaussian Energy Broadening (GEB) calibration for HPGe pulse-height simulations.

---

## 📋 Requirements

Before running OpenMC Studio, install:

- Python 3.10 or newer
- OpenMC Python API 0.14.0 or newer
- HDF5 nuclear data libraries with the `OPENMC_CROSS_SECTIONS` environment variable configured correctly

---

## 🚀 Getting Started

### Clone the repository

```
git clone https://github.com/AymanAbuGhazal/OpenMC-Studio.git
```

### Navigate to the project folder

```
cd OpenMC-Studio
```

### Install the required packages

```
pip install -r requirements.txt
```

### Launch the application

```
python main.py
```

---

## 📖 Citation

If you use OpenMC Studio in academic research or technical projects, please cite:

```
Abu Ghazal, A. (2026).
OpenMC Studio: A Graphical Environment for Monte Carlo Particle Transport Simulations.
Computer software.
https://github.com/AymanAbuGhazal/OpenMC-Studio
```

---

## 📄 License

This project is released under the MIT License.

---

## 📬 Contact

**Developer:** Dr. Ayman Abu Ghazal

**GitHub:** https://github.com/AymanAbuGhazal

---

## ⚠️ Disclaimer

OpenMC Studio is an independent open-source project developed by **Dr. Ayman Abu Ghazal**.

It is **not** an official product of the **Jordan Atomic Energy Commission (JAEC)** and does **not** represent the official policies, positions, or endorsements of the JAEC.
