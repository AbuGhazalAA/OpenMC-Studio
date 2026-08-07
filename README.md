# OpenMC Studio

**OpenMC Studio** is a dedicated graphical user interface (GUI) and integrated development environment (IDE) designed to streamline reactor physics modeling, radiation shielding simulations, and criticality calculations using the **OpenMC** Monte Carlo engine.

Developed by **Ayman Abu Ghazal**, this tool bridges the gap between complex Python-based OpenMC scripts and an intuitive visual workflow. It is highly optimized for researchers and nuclear engineers, featuring 2D/3D geometry viewers, live output consoles, material builders, and automated validation tools.

---

## 🌟 Key Features

* **Python Script Engine:** Full script editor with real-time synchronization, safe execution sandboxing, and syntax support.
* **Geometry & Material Builders:** Interactive tools to define materials, compositions, and complex nuclear reactor geometries. Easily handles models translated from legacy MCNP decks.
* **Visualization Tools:** Built-in 2D geometry plotting and advanced voxel viewing capabilities.
* **Simulation Management:** Automated execution of eigenvalue and fixed-source calculations with real-time progress tracking, ETA estimation, and live output logging.
* **Results & Post-Processing:** Automated statepoint loading, tally extraction, and Gaussian Energy Broadening (GEB) calibration for radiation detectors (e.g., HPGe pulse-height tallies).

---

## 📋 Prerequisites

To run OpenMC Studio and its exported simulation scripts, ensure you have the following installed on your system:

1. **Python** (>= 3.10)
2. **OpenMC Python API** (>= 0.14.0)
3. **Nuclear Data Libraries:** Configured HDF5 cross-section libraries with the `OPENMC_CROSS_SECTIONS` environment variable set correctly.

---

## 🚀 Getting Started

1. **Clone the repository:**

   git clone [https://github.com/AymanAbuGhazal/OpenMC-Studio.git](https://github.com/AymanAbuGhazal/OpenMC-Studio.git)


2. **Navigate to the project directory:**
cd OpenMC-Studio




3. **Install the required dependencies:**
pip install -r requirements.txt

```


4. **Launch the application:**
python main.py




---

## 📖 How to Cite

If you use OpenMC Studio in your research, academic papers, or technical projects, please cite it as follows:

> Abu Ghazal, A. (2026). *OpenMC Studio: A Graphical Environment for Monte Carlo Particle Transport Simulations* [Computer software]. https://github.com/AymanAbuGhazal/OpenMC-Studio

---

## 📄 License

This project is open-source and released under the [MIT License](https://www.google.com/search?q=LICENSE).

---

## 📬 Contact

* **Developer:** Dr. Ayman Abu Ghazal
* **GitHub:** [@AymanAbuGhazal](https://www.google.com/search?q=https://github.com/AymanAbuGhazal)

---

## ⚠️ Disclaimer

This project is an independent personal initiative developed by Dr. Ayman Abu Ghazal. It is not an official product of the Jordan Atomic Energy Commission (JAEC) and does not reflect the official views, policies, or endorsements of the JAEC.

```

```
