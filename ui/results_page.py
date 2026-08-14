import os
import glob
import re
import h5py
import openmc
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QLineEdit,
    QApplication, QDialog, QListWidget, QAbstractItemView
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


def _gaussian(x, amplitude, mu, sigma, baseline):
    return baseline + amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _expected_sigma_keV(energy_keV, a, b, c):
    E_MeV = energy_keV / 1000.0
    fwhm_MeV = a + b * np.sqrt(np.clip(E_MeV + c * E_MeV ** 2, 0.0, None))
    return max(fwhm_MeV * 1000.0 / 2.3548, 1e-6)


NUCLIDE_GAMMA_LINES = {
    'Am-241': [(59.541, 0.359)],
    'Cd-109': [(88.034, 0.0366)],
    'Ce-139': [(165.857, 0.799)],
    'Co-57': [(122.061, 0.8560), (136.474, 0.1068)],
    'Co-60': [(1173.228, 0.9985), (1332.492, 0.999826)],
    'Cs-137': [(661.657, 0.851)],
    'Ba-133': [(53.163, 0.00141), (79.614, 0.0265), (81.0, 0.3406), (160.6, 0.00638),
               (223.2, 0.00450), (276.4, 0.0716), (302.85, 0.1833), (356.01, 0.6205), (383.85, 0.0894)],
    'Sr-85': [(514.0, 0.984)],
    'Y-88': [(898.042, 0.937), (1836.063, 0.992)],
    'Cr-51': [(320.082, 0.0991)],
    'Mn-54': [(834.848, 0.9998)],
    'Zn-65': [(1115.539, 0.506)],
}


# Atomic numbers of the actinides. "Heavy metal" in the burnup unit MWd/kgHM
# conventionally means Z >= 90 (thorium and above) -- structural elements and
# the oxygen in an oxide fuel are deliberately NOT part of that mass.
_HEAVY_METAL_Z = {
    'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
}
_AVOGADRO = 6.02214076e23


def _split_nuclide_name(name):
    """'U235' -> ('U', 235); 'Am242m' -> ('Am', 242); returns (None, None)
    for anything that is not an element symbol followed by a mass number."""
    m = re.match(r'^([A-Za-z]+)(\d+)', str(name))
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _heavy_metal_mass_kg(atoms_row, nuc_index):
    """Initial heavy-metal mass in kg, computed straight from the atom
    inventory the depletion file already stores.

        m_HM = sum over actinides of  N_i * A_i / N_A

    Burnup is by definition referred to the INITIAL heavy metal mass, so
    this is meant to be called with the FIRST depletion step's row.

    The mass number A is used in place of the exact atomic mass: for the
    actinides that approximation is accurate to about 0.02% (U235 is
    235.0439, U238 is 238.0508), which is far below the statistical
    uncertainty of any Monte Carlo depletion run.
    """
    if atoms_row is None:
        return 0.0
    grams = 0.0
    for name, idx in nuc_index.items():
        element, mass_number = _split_nuclide_name(name)
        if element in _HEAVY_METAL_Z and mass_number and idx < len(atoms_row):
            grams += float(atoms_row[idx]) * mass_number / _AVOGADRO
    return grams / 1000.0


def _reactivity_pcm(k_val, k_err):
    """Reactivity in pcm, the unit reactor physicists actually quote:

        rho = (k - 1) / k   x 1e5

    The uncertainty follows from d(rho)/dk = 1/k^2, so sigma_rho =
    sigma_k / k^2 x 1e5. Steps where k is non-positive (never physical,
    but possible in a corrupt file) are returned as NaN rather than
    raising, so one bad step cannot destroy the whole curve.
    """
    k = np.asarray(k_val, dtype=float)
    e = np.asarray(k_err, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where(k > 0, (k - 1.0) / k * 1.0e5, np.nan)
        rho_err = np.where(k > 0, e / (k ** 2) * 1.0e5, np.nan)
    return rho, rho_err


def _cycle_end(x_vals, k_val):
    """Where the reactor goes subcritical: the x-value at which k first
    crosses 1.0, found by linear interpolation between the two bracketing
    depletion steps.

    Returns (x_at_k1, index_after) or (None, None) when k never crosses --
    either it stays above 1 for the whole run (cycle not finished) or it
    started below 1 (already subcritical at t=0).
    """
    k = np.asarray(k_val, dtype=float)
    x = np.asarray(x_vals, dtype=float)
    n = min(len(k), len(x))
    if n < 2 or k[0] < 1.0:
        return None, None
    for i in range(1, n):
        if k[i - 1] >= 1.0 > k[i]:
            dk = k[i - 1] - k[i]
            if dk == 0:
                return float(x[i]), i
            frac = (k[i - 1] - 1.0) / dk
            return float(x[i - 1] + frac * (x[i] - x[i - 1])), i
    return None, None


def _identify_nuclide(energy_keV, fwhm_keV, tolerance_factor=3.0, min_tolerance_keV=1.0):
    tolerance = max(tolerance_factor * fwhm_keV, min_tolerance_keV)
    best = None
    best_diff = None
    for nuclide, lines in NUCLIDE_GAMMA_LINES.items():
        for line_energy, intensity in lines:
            diff = abs(line_energy - energy_keV)
            if diff <= tolerance and (best_diff is None or diff < best_diff):
                best = (nuclide, line_energy, intensity)
                best_diff = diff
    return best


class ResultsPageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sp = None
        self._last_spectrum_keV = None
        self._last_spectrum_counts = None
        self._last_geb_abc = None
        self._current_fig = None
        self._current_ax = None
        self._last_peaks = []
        self._depletion_data = {}

        self._init_ui()
        self._setup_tally_plot_dialog()
        self._setup_hpge_dialog()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 1. File Load Section ---
        load_layout = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load StatePoint File")
        self.btn_load.setStyleSheet(
            "background-color: #0e639c; color: white; font-weight: bold; padding: 10px; font-size: 14px;")
        self.btn_load.clicked.connect(self.load_statepoint)

        self.lbl_file = QLabel("No file loaded.")
        self.lbl_file.setStyleSheet("color: #555; font-style: italic;")

        load_layout.addWidget(self.btn_load)
        load_layout.addWidget(self.lbl_file)
        load_layout.addStretch()
        layout.addLayout(load_layout)

        # --- 2. K-Effective Section ---
        keff_group = QGroupBox("⚛️ K-Effective (Multiplication Factor)")
        keff_group.setStyleSheet("font-weight: bold; font-size: 14px;")
        keff_layout = QVBoxLayout(keff_group)

        self.lbl_keff = QLabel("k-effective = N/A")
        self.lbl_keff.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_keff.setStyleSheet("font-size: 28px; font-weight: bold; color: #217346; padding: 20px;")
        keff_layout.addWidget(self.lbl_keff)
        layout.addWidget(keff_group)

        # --- 3. General Section: Tallies & Results ---
        tally_group = QGroupBox("📊 General Tallies & Results")
        tally_group.setStyleSheet("font-weight: bold; font-size: 14px;")
        tally_layout = QVBoxLayout(tally_group)

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Select Tally to View:"))
        self.combo_tallies = QComboBox()
        self.combo_tallies.currentIndexChanged.connect(self.display_tally)
        combo_layout.addWidget(self.combo_tallies)

        self.btn_export = QPushButton("💾 Export to CSV")
        self.btn_export.setStyleSheet("background-color: #217346; color: white; padding: 5px; font-weight: bold;")
        self.btn_export.clicked.connect(self.export_to_csv)
        combo_layout.addWidget(self.btn_export)

        combo_layout.addWidget(QLabel("Mesh Score:"))
        self.combo_mesh_score = QComboBox()
        combo_layout.addWidget(self.combo_mesh_score)

        self.btn_plot_mesh = QPushButton("🗺️ Plot Mesh (Heatmap)")
        self.btn_plot_mesh.setStyleSheet("background-color: #d97706; color: white; padding: 5px; font-weight: bold;")
        self.btn_plot_mesh.clicked.connect(self.plot_mesh_tally)
        combo_layout.addWidget(self.btn_plot_mesh)

        # --- Button to open independent plot window ---
        self.btn_open_plot_dialog = QPushButton("📈 Open Plot Window")
        self.btn_open_plot_dialog.setStyleSheet(
            "background-color: #0e639c; color: white; padding: 5px; font-weight: bold;")
        self.btn_open_plot_dialog.clicked.connect(self.open_tally_plot_window)
        combo_layout.addWidget(self.btn_open_plot_dialog)

        combo_layout.addStretch()
        tally_layout.addLayout(combo_layout)

        # Raw data table
        self.table_tally = QTableWidget()
        self.table_tally.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_tally.setStyleSheet("font-weight: normal; font-size: 12px;")
        tally_layout.addWidget(self.table_tally)

        layout.addWidget(tally_group)

        # --- 4. Spectral analysis window button (hidden by default) ---
        self.btn_open_hpge = QPushButton("🚀 Open Advanced HPGe Spectral Analysis")
        self.btn_open_hpge.setStyleSheet(
            "background-color: #8e44ad; color: white; font-weight: bold; font-size: 15px; padding: 12px;")
        self.btn_open_hpge.setVisible(False)
        self.btn_open_hpge.clicked.connect(self.show_hpge_dialog)
        layout.addWidget(self.btn_open_hpge)

    def _setup_tally_plot_dialog(self):
        """Setup independent plot window"""
        self.tally_plot_dialog = QDialog(self)
        self.tally_plot_dialog.setWindowTitle("📈 Advanced Tally Visualization")
        self.tally_plot_dialog.setMinimumSize(900, 650)
        self.tally_plot_dialog.setWindowFlags(
            self.tally_plot_dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
        self.tally_plot_dialog.setStyleSheet("QDialog { background-color: #FFFFFF; }")

        layout = QVBoxLayout(self.tally_plot_dialog)

        plot_controls = QHBoxLayout()

        # 1. Plot settings for normal tallies
        plot_controls.addWidget(QLabel("<b>Plot Type:</b>"))
        self.tally_plot_dialog.combo_plot_type = QComboBox()
        self.tally_plot_dialog.combo_plot_type.addItems(["Auto", "Line", "Bar", "Mesh Heatmap"])
        self.tally_plot_dialog.combo_plot_type.currentIndexChanged.connect(lambda _: self._plot_selected_tally())
        plot_controls.addWidget(self.tally_plot_dialog.combo_plot_type)

        # 2. Plot settings for Depletion/Burnup data
        self.tally_plot_dialog.depletion_controls = QWidget()
        dep_layout = QHBoxLayout(self.tally_plot_dialog.depletion_controls)
        dep_layout.setContentsMargins(0, 0, 0, 0)

        dep_layout.addWidget(QLabel("<b>X-Axis:</b>"))
        self.tally_plot_dialog.combo_dep_x = QComboBox()
        self.tally_plot_dialog.combo_dep_x.addItems(["Time (Days)", "Burnup (MWd/kgHM)"])
        self.tally_plot_dialog.combo_dep_x.currentIndexChanged.connect(self._plot_depletion_data)
        dep_layout.addWidget(self.tally_plot_dialog.combo_dep_x)

        # Quantity: what physically goes on the Y axis. Reactivity gets its own
        # entry because reactor physicists work in pcm, not in raw k.
        dep_layout.addWidget(QLabel("<b>Quantity:</b>"))
        self.tally_plot_dialog.combo_dep_y = QComboBox()
        self.tally_plot_dialog.combo_dep_y.addItems(
            ["k-effective", "Reactivity (pcm)", "Nuclide inventory"])
        self.tally_plot_dialog.combo_dep_y.setToolTip(
            "k-effective      -- the multiplication factor as computed\n"
            "Reactivity (pcm) -- rho = (k-1)/k x 1e5, plus the cycle end where k crosses 1\n"
            "Nuclide inventory-- how the selected nuclides evolve with burnup")
        self.tally_plot_dialog.combo_dep_y.currentIndexChanged.connect(self._on_dep_quantity_changed)
        dep_layout.addWidget(self.tally_plot_dialog.combo_dep_y)

        # Material selector: summing every depletable material together hides
        # the behaviour of each individual region (different fuel zones burn at
        # very different rates), so the material is selectable and the total is
        # only the default.
        dep_layout.addWidget(QLabel("<b>Material:</b>"))
        self.tally_plot_dialog.combo_dep_mat = QComboBox()
        self.tally_plot_dialog.combo_dep_mat.setMinimumWidth(150)
        self.tally_plot_dialog.combo_dep_mat.setToolTip(
            "Which depletable material to trace.\n"
            "'All materials (sum)' adds every region together, which can hide\n"
            "how an individual fuel zone actually behaves.")
        self.tally_plot_dialog.combo_dep_mat.currentIndexChanged.connect(self._plot_depletion_data)
        dep_layout.addWidget(self.tally_plot_dialog.combo_dep_mat)

        plot_controls.addWidget(self.tally_plot_dialog.depletion_controls)
        self.tally_plot_dialog.depletion_controls.setVisible(False)

        self.tally_plot_dialog.btn_refresh = QPushButton("🔄 Refresh Plot")
        self.tally_plot_dialog.btn_refresh.setStyleSheet(
            "background-color: #0e639c; color: white; padding: 6px; font-weight: bold;")
        self.tally_plot_dialog.btn_refresh.clicked.connect(self._refresh_active_plot)
        plot_controls.addWidget(self.tally_plot_dialog.btn_refresh)

        self.tally_plot_dialog.btn_save = QPushButton("💾 Save Plot")
        self.tally_plot_dialog.btn_save.clicked.connect(self._save_tally_plot)
        plot_controls.addWidget(self.tally_plot_dialog.btn_save)

        self.tally_plot_dialog.btn_export = QPushButton("📊 Export Plot Data to Excel/CSV")
        self.tally_plot_dialog.btn_export.setStyleSheet(
            "background-color: #d97706; color: white; padding: 6px; font-weight: bold;")
        self.tally_plot_dialog.btn_export.clicked.connect(self.export_plot_to_excel)
        plot_controls.addWidget(self.tally_plot_dialog.btn_export)

        plot_controls.addStretch()
        layout.addLayout(plot_controls)

        # Second row, for the inventory view only. Nuclide list + units + scale
        # were originally squeezed into the single row above, which left the
        # Material dropdown truncated and the list overlapping its neighbours.
        self.tally_plot_dialog.nuclide_box = QWidget()
        nuc_box_layout = QHBoxLayout(self.tally_plot_dialog.nuclide_box)
        nuc_box_layout.setContentsMargins(0, 0, 0, 0)

        # Multi-select nuclide list: comparing U235 against Pu239 against Pu241
        # on one set of axes is the single most common thing asked of a burnup
        # plot, which a one-at-a-time dropdown cannot do.
        nuc_box_layout.addWidget(QLabel("<b>Nuclides:</b>"))
        self.tally_plot_dialog.list_dep_nuclides = QListWidget()
        self.tally_plot_dialog.list_dep_nuclides.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tally_plot_dialog.list_dep_nuclides.setFixedHeight(96)
        self.tally_plot_dialog.list_dep_nuclides.setMinimumWidth(200)
        self.tally_plot_dialog.list_dep_nuclides.setToolTip(
            "Hold Ctrl to pick several nuclides and overlay them on one plot.")
        self.tally_plot_dialog.list_dep_nuclides.itemSelectionChanged.connect(
            self._plot_depletion_data)
        nuc_box_layout.addWidget(self.tally_plot_dialog.list_dep_nuclides)

        nuc_box_layout.addSpacing(12)
        nuc_box_layout.addWidget(QLabel("<b>Units:</b>"))
        self.tally_plot_dialog.combo_dep_units = QComboBox()
        self.tally_plot_dialog.combo_dep_units.setMinimumWidth(190)
        self.tally_plot_dialog.combo_dep_units.addItems([
            "Atoms", "Mass (g)", "Weight %", "Atom density (a/b-cm)", "Normalized N/N0"])
        self.tally_plot_dialog.combo_dep_units.setToolTip(
            "Atoms                 -- raw inventory as stored in the file\n"
            "Mass (g)              -- atoms x A / N_A\n"
            "Weight %              -- share of that material's total mass\n"
            "Atom density (a/b-cm) -- the unit OpenMC inputs use (needs the material volume)\n"
            "Normalized N/N0       -- relative to the first step, the clearest view of burn rate")
        self.tally_plot_dialog.combo_dep_units.currentIndexChanged.connect(self._plot_depletion_data)
        nuc_box_layout.addWidget(self.tally_plot_dialog.combo_dep_units)

        nuc_box_layout.addSpacing(12)
        nuc_box_layout.addWidget(QLabel("<b>Y Scale:</b>"))
        self.tally_plot_dialog.combo_dep_scale = QComboBox()
        self.tally_plot_dialog.combo_dep_scale.setMinimumWidth(110)
        self.tally_plot_dialog.combo_dep_scale.addItems(["Auto", "Linear", "Log"])
        self.tally_plot_dialog.combo_dep_scale.setToolTip(
            "Auto   -- logarithmic when the values span more than two decades\n"
            "          and none of them is zero, otherwise linear\n"
            "Linear -- always linear\n"
            "Log    -- always logarithmic. A logarithm of zero is undefined, so a\n"
            "          nuclide that is bred from nothing (Pu239, fission products)\n"
            "          starts at exactly 0 and cannot be shown on a log axis.")
        self.tally_plot_dialog.combo_dep_scale.currentIndexChanged.connect(
            self._on_dep_scale_changed)
        nuc_box_layout.addWidget(self.tally_plot_dialog.combo_dep_scale)

        nuc_box_layout.addStretch()
        layout.addWidget(self.tally_plot_dialog.nuclide_box)
        self.tally_plot_dialog.nuclide_box.setVisible(False)

        self.tally_plot_dialog.status_label = QLabel("Select a Tally from the main window to display its result.")
        self.tally_plot_dialog.status_label.setStyleSheet("color: #666666; font-style: italic; font-weight: normal;")
        layout.addWidget(self.tally_plot_dialog.status_label)

        # Plot area
        self.tally_plot_dialog.figure = Figure(tight_layout=True)
        self.tally_plot_dialog.canvas = FigureCanvas(self.tally_plot_dialog.figure)
        self.tally_plot_dialog.ax = self.tally_plot_dialog.figure.add_subplot(111)
        self.tally_plot_dialog.toolbar = NavigationToolbar(self.tally_plot_dialog.canvas, self.tally_plot_dialog)

        layout.addWidget(self.tally_plot_dialog.toolbar)
        layout.addWidget(self.tally_plot_dialog.canvas)

    def _setup_hpge_dialog(self):
        """Setup HPGe tools popup window"""
        self.hpge_dialog = QDialog(self)
        self.hpge_dialog.setWindowTitle("☢️ Advanced HPGe Spectral Analysis")
        self.hpge_dialog.setMinimumSize(900, 600)
        self.hpge_dialog.setWindowFlags(
            self.hpge_dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)

        hpge_layout = QVBoxLayout(self.hpge_dialog)

        # --- Group 1: Plotting & GEB ---
        plot_group = QGroupBox("1. Spectrum Plotting & GEB Calibration")
        plot_group.setStyleSheet("font-weight: bold; font-size: 13px; color: #0e639c;")
        plot_layout = QHBoxLayout(plot_group)

        self.btn_plot = QPushButton("📈 Plot Spectrum + GEB")
        self.btn_plot.setStyleSheet("background-color: #0e639c; color: white; padding: 8px; font-weight: bold;")
        self.btn_plot.clicked.connect(self.plot_spectrum_graph)
        plot_layout.addWidget(self.btn_plot)

        plot_layout.addSpacing(20)
        plot_layout.addWidget(QLabel("<b>GEB (a, b, c):</b>"))
        self.input_a = QLineEdit("0.000496")
        self.input_a.setFixedWidth(80)
        plot_layout.addWidget(self.input_a)

        self.input_b = QLineEdit("0.001176")
        self.input_b.setFixedWidth(80)
        plot_layout.addWidget(self.input_b)

        self.input_c = QLineEdit("0.0")
        self.input_c.setFixedWidth(80)
        plot_layout.addWidget(self.input_c)
        plot_layout.addStretch()
        hpge_layout.addWidget(plot_group)

        # --- Group 2: Peak Analysis & Isotopes ---
        analysis_group = QGroupBox("2. Peak Fitting & Efficiency (FEPE)")
        analysis_group.setStyleSheet("font-weight: bold; font-size: 13px; color: #8e44ad;")
        analysis_layout = QHBoxLayout(analysis_group)

        analysis_layout.addWidget(QLabel("Min Prominence (%):"))
        self.input_prominence = QLineEdit("50.0")
        self.input_prominence.setFixedWidth(60)
        analysis_layout.addWidget(self.input_prominence)

        self.btn_find_peaks = QPushButton("🔎 Find & Fit Peaks")
        self.btn_find_peaks.setStyleSheet("background-color: #8e44ad; color: white; padding: 8px; font-weight: bold;")
        self.btn_find_peaks.clicked.connect(self.find_and_fit_peaks)
        analysis_layout.addWidget(self.btn_find_peaks)

        analysis_layout.addSpacing(20)

        self.btn_identify = QPushButton("🎯 Identify Nuclides")
        self.btn_identify.setStyleSheet("background-color: #2c3e50; color: white; padding: 8px; font-weight: bold;")
        self.btn_identify.clicked.connect(self.identify_nuclides)
        analysis_layout.addWidget(self.btn_identify)

        self.btn_fepe = QPushButton("📊 Calc FEPE & Plot Curve")
        self.btn_fepe.setStyleSheet("background-color: #16a085; color: white; padding: 8px; font-weight: bold;")
        self.btn_fepe.clicked.connect(self.calculate_fepe_and_plot)
        analysis_layout.addWidget(self.btn_fepe)
        analysis_layout.addStretch()
        hpge_layout.addWidget(analysis_group)

        # --- Results Table ---
        self.table_peaks = QTableWidget()
        self.table_peaks.setColumnCount(6)
        self.table_peaks.setHorizontalHeaderLabels(
            ["Peak #", "Centroid [keV]", "FWHM [keV]", "Area [counts]", "Nuclide", "FEPE"])
        self.table_peaks.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_peaks.setStyleSheet("font-weight: normal; font-size: 13px; color: black;")
        hpge_layout.addWidget(self.table_peaks)

    def open_tally_plot_window(self):
        self.tally_plot_dialog.show()
        self.tally_plot_dialog.raise_()
        self.tally_plot_dialog.activateWindow()
        self._refresh_active_plot()

    def _refresh_active_plot(self):
        if self.tally_plot_dialog.depletion_controls.isVisible():
            self._plot_depletion_data()
        else:
            self._plot_selected_tally()

    def show_hpge_dialog(self):
        self.hpge_dialog.show()
        self.hpge_dialog.raise_()
        self.hpge_dialog.activateWindow()

    def _log(self, msg):
        try:
            for widget in QApplication.topLevelWidgets():
                if hasattr(widget, 'console_widget'):
                    widget.console_widget.append_log(msg)
                    return
        except Exception:
            pass
        print(msg)

    def auto_load_latest_statepoint(self):
        try:
            search_path_main = os.path.join(os.getcwd(), "*.h5")
            search_path_export = os.path.join(os.getcwd(), "export", "*.h5")
            h5_files = glob.glob(search_path_main) + glob.glob(search_path_export)

            if not h5_files:
                return

            latest_file = max(h5_files, key=os.path.getctime)
            self.lbl_file.setText(f"Auto-Loaded: {os.path.basename(latest_file)}")
            self.process_statepoint(latest_file)
        except Exception as e:
            self._log(f"⚠️ Auto-load failed: {e}")

    def load_statepoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open StatePoint File", os.getcwd(), "HDF5 Files (*.h5);;All Files (*)"
        )
        if file_path:
            self.lbl_file.setText(f"Loaded: {os.path.basename(file_path)}")
            self.process_statepoint(file_path)

    def _process_depletion_results(self, filepath):
        """Analyze depletion file securely using 100% native h5py (Bypassing Windows GUI crash)."""
        try:
            with h5py.File(filepath, 'r') as f:
                if 'eigenvalues' not in f or 'time' not in f:
                    raise ValueError(
                        "Not a valid OpenMC depletion results file (missing 'eigenvalues' or 'time' datasets).")

                # 1. Extract K-effective
                k_data = np.array(f['eigenvalues'])
                if k_data.ndim == 1:
                    if k_data.dtype.names is not None:
                        k_val = k_data[k_data.dtype.names[0]]
                        k_err = k_data[k_data.dtype.names[1]] if len(k_data.dtype.names) > 1 else np.zeros_like(k_val)
                    else:
                        k_val = k_data
                        k_err = np.zeros_like(k_val)
                else:
                    if k_data.shape[-1] >= 2:
                        k_val = k_data[..., 0].flatten()
                        k_err = k_data[..., 1].flatten()
                    else:
                        k_val = k_data[..., 0].flatten()
                        k_err = np.zeros_like(k_val)

                # 2. Extract Time
                time_data = np.array(f['time'])
                if time_data.ndim == 2 and time_data.shape[1] == 2:
                    time_s = np.append(time_data[:, 0], time_data[-1, 1])
                else:
                    time_s = time_data.flatten()
                time_days = time_s / (24 * 60 * 60)

                min_len = min(len(time_days), len(k_val))
                time_days = time_days[:min_len]
                k_val = k_val[:min_len]
                k_err = k_err[:min_len]

                # 3. Extract Burnup OR calculate Energy (MWd)
                burnup_data = None
                self._x_is_energy = False
                if 'burnup' in f:
                    b_data = np.array(f['burnup'])
                    if b_data.ndim > 1:
                        burnup_data = b_data.sum(axis=1)
                    else:
                        burnup_data = b_data
                    burnup_data = burnup_data[:min_len]
                elif 'source_rate' in f:
                    try:
                        power_W = np.array(f['source_rate']).flatten()
                        power_MW = power_W / 1e6
                        if len(power_MW) > 0:
                            dt_days = np.diff(time_days)
                            energy_MWd = np.zeros(len(time_days))
                            energy_MWd[1:] = np.cumsum(power_MW[:len(dt_days)] * dt_days)
                            burnup_data = energy_MWd
                            self._x_is_energy = True
                    except Exception:
                        pass

                # 4. Extract nuclide / material indices and the atom inventory.
                #
                # The layout openmc.deplete actually writes is:
                #     /nuclides/<name>     Group -> attrs['atom number index']
                #     /materials/<mat_id>  Group -> attrs['index'], attrs['volume']
                #     /number              Dataset (steps, stages, materials, nuclides)
                #
                # Two things worth spelling out, because both silently break the
                # isotope curves if assumed otherwise:
                #   * /nuclides is a GROUP whose SUBGROUP NAMES are the nuclide
                #     names -- it is not a string Dataset. Testing it with
                #     isinstance(..., h5py.Dataset) therefore never matches and
                #     the isotope list comes back empty, leaving the Y-axis
                #     dropdown with nothing to choose from.
                #   * /number has FOUR dimensions, not three. Used as-is,
                #     atoms[:, idx] returns a 3-D slab instead of a single curve.
                nuc_index = {}
                mat_index = {}
                mat_volume = {}
                atoms_by_mat = None

                try:
                    if 'nuclides' in f and isinstance(f['nuclides'], h5py.Group):
                        for _name, _node in f['nuclides'].items():
                            _idx = _node.attrs.get('atom number index',
                                                   _node.attrs.get('index'))
                            if _idx is not None:
                                nuc_index[str(_name)] = int(_idx)

                    if 'materials' in f and isinstance(f['materials'], h5py.Group):
                        for _name, _node in f['materials'].items():
                            _idx = _node.attrs.get('index')
                            if _idx is not None:
                                mat_index[str(_name)] = int(_idx)
                            # Volume is stored alongside the index and is what
                            # makes atom DENSITY (atom/b-cm) -- the unit OpenMC
                            # inputs actually use -- computable at all.
                            _vol = _node.attrs.get('volume')
                            if _vol is not None:
                                mat_volume[str(_name)] = float(_vol)

                    if 'number' in f and isinstance(f['number'], h5py.Dataset):
                        _num = np.array(f['number'], dtype=float)
                        if _num.ndim == 4:
                            atoms_by_mat = _num[:, 0, :, :]      # first stage
                        elif _num.ndim == 3:
                            atoms_by_mat = _num
                        elif _num.ndim == 2:
                            atoms_by_mat = _num[:, None, :]
                        if atoms_by_mat is not None:
                            atoms_by_mat = atoms_by_mat[:min_len]

                except Exception as ex:
                    self._log(f"⚠️ Notice: minor issue while extracting isotope counts: {ex}")

                # Ordered by the index stored in the file, so a name's position
                # in this list always matches its column in atoms_by_mat.
                nuclides = sorted(nuc_index, key=lambda _n: nuc_index[_n])
                materials = sorted(mat_index, key=lambda _m: mat_index[_m])

                # 5. Real burnup (MWd/kgHM).
                #
                # The energy computed above is MWd -- it only becomes burnup
                # after dividing by the initial heavy-metal mass. That mass is
                # already implicit in the file (the step-0 actinide inventory),
                # so it is derived from the results themselves rather than asked
                # of the user or re-read from the script: a mass taken from the
                # script could belong to a model that has since been edited,
                # whereas this one is guaranteed to match the run that produced
                # these numbers.
                hm_mass_kg = 0.0
                hm_mass_by_mat = {}
                if atoms_by_mat is not None and atoms_by_mat.ndim == 3 and len(atoms_by_mat):
                    initial = atoms_by_mat[0]                       # (materials, nuclides)
                    for _m, _mi in mat_index.items():
                        if _mi < initial.shape[0]:
                            hm_mass_by_mat[_m] = _heavy_metal_mass_kg(initial[_mi], nuc_index)
                    hm_mass_kg = float(sum(hm_mass_by_mat.values()))

                if burnup_data is not None and hm_mass_kg > 0 and self._x_is_energy:
                    burnup_data = np.asarray(burnup_data, dtype=float) / hm_mass_kg
                    self._x_is_energy = False                        # now genuine MWd/kgHM
                    self._log(f"✅ Burnup axis available: initial heavy-metal mass "
                              f"= {hm_mass_kg:.4g} kg (from the step-0 actinide inventory).")
                elif self._x_is_energy:
                    self._log("ℹ️ No heavy metal found in the inventory -- the burnup axis will "
                              "show total energy produced (MWd) instead of MWd/kgHM.")

                # Store data for plotting
                self._depletion_data = {
                    'time_days': time_days,
                    'burnup': burnup_data,
                    'k_val': k_val,
                    'k_err': k_err,
                    'nuclides': nuclides,
                    'materials': materials,
                    'nuc_index': nuc_index,
                    'mat_index': mat_index,
                    'atoms_by_mat': atoms_by_mat,
                    'mat_volume': mat_volume,
                    'hm_mass_kg': hm_mass_kg,
                    'hm_mass_by_mat': hm_mass_by_mat
                }

                # --- Update UI ---
                final_k = float(k_val[-1])
                final_err = float(k_err[-1])

                self.lbl_keff.setText(f"Final k-effective = {final_k:.5f} ± {final_err:.5f}")
                self.lbl_keff.setStyleSheet("font-size: 26px; font-weight: bold; color: #d35400; padding: 20px;")

                self.combo_tallies.blockSignals(True)
                self.combo_tallies.clear()
                self.combo_tallies.addItem("Depletion Results", -1)
                self.combo_tallies.blockSignals(False)
                self.btn_open_hpge.setVisible(False)

                # Populate K-eff and Time table
                self.table_tally.setRowCount(len(time_days))
                self.table_tally.setColumnCount(3)
                self.table_tally.setHorizontalHeaderLabels(["Time [Days]", "k-effective", "Std Dev"])
                self.table_tally.setUpdatesEnabled(False)

                for row in range(len(time_days)):
                    item_time = QTableWidgetItem(f"{time_days[row]:.4f}")
                    item_k = QTableWidgetItem(f"{k_val[row]:.5f}")
                    item_err = QTableWidgetItem(f"{k_err[row]:.5f}")

                    item_time.setFlags(item_time.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item_k.setFlags(item_k.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item_err.setFlags(item_err.flags() & ~Qt.ItemFlag.ItemIsEditable)

                    self.table_tally.setItem(row, 0, item_time)
                    self.table_tally.setItem(row, 1, item_k)
                    self.table_tally.setItem(row, 2, item_err)
                self.table_tally.setUpdatesEnabled(True)

                # Update independent plot window settings
                self.tally_plot_dialog.combo_plot_type.setVisible(False)
                self.tally_plot_dialog.depletion_controls.setVisible(True)

                # Fill the nuclide list, most-studied ones first so the common
                # case needs no scrolling.
                popular_isotopes = ['U235', 'U238', 'Pu239', 'Pu240', 'Pu241',
                                    'Xe135', 'Sm149', 'Cs137', 'Sr90']
                lw = self.tally_plot_dialog.list_dep_nuclides
                lw.blockSignals(True)
                lw.clear()
                ordered = [n for n in popular_isotopes if n in nuc_index]
                ordered += [n for n in nuclides if n not in ordered]
                lw.addItems(ordered)
                if lw.count():
                    lw.item(0).setSelected(True)
                lw.blockSignals(False)

                # Populate the material selector ("All" first, then each one)
                self.tally_plot_dialog.combo_dep_mat.blockSignals(True)
                self.tally_plot_dialog.combo_dep_mat.clear()
                self.tally_plot_dialog.combo_dep_mat.addItem("All materials (sum)", None)
                for _m in materials:
                    self.tally_plot_dialog.combo_dep_mat.addItem(f"Material {_m}", _m)
                self.tally_plot_dialog.combo_dep_mat.setEnabled(bool(materials))
                self.tally_plot_dialog.combo_dep_mat.blockSignals(False)

                if not nuclides:
                    self._log("ℹ️ No nuclide inventory found in this depletion file -- "
                              "only k-effective can be plotted.")
                else:
                    self._log(f"✅ Depletion inventory loaded: {len(nuclides)} nuclide(s) "
                              f"across {len(materials)} material(s).")

                # Plot data immediately and open window
                self._plot_depletion_data()
                self.open_tally_plot_window()

        except Exception as e:
            QMessageBox.critical(self, "Depletion Read Error", f"Could not read depletion file:\n{str(e)}")
            self._log(f"⚠️ Depletion analysis error: {e}")

    def _on_dep_quantity_changed(self):
        """Nuclide picker and units only make sense for the inventory view."""
        is_inventory = self.tally_plot_dialog.combo_dep_y.currentText() == "Nuclide inventory"
        self.tally_plot_dialog.nuclide_box.setVisible(is_inventory)
        self._plot_depletion_data()

    def _on_dep_scale_changed(self):
        """Re-arms the one-shot explanation so that deliberately choosing
        'Log' again tells the user why it was refused, instead of silently
        falling back to a linear axis a second time."""
        self._log_scale_notice_shown = False
        self._plot_depletion_data()

    def _apply_y_scale(self, ax, curves):
        """Applies the requested Y scale and returns a note for the status bar.

        `curves` is the list of arrays actually drawn. A logarithmic axis is
        undefined at zero, and in a burnup calculation a bred nuclide (Pu239,
        every fission product) legitimately starts at exactly zero -- so the
        request has to be refused rather than silently producing an empty or
        clipped plot. When the user asked for Log explicitly, say so once in a
        dialog; 'Auto' just decides quietly.
        """
        mode = self.tally_plot_dialog.combo_dep_scale.currentText()
        if not curves:
            return ""

        all_values = np.concatenate([np.asarray(c, dtype=float).ravel() for c in curves])
        finite = all_values[np.isfinite(all_values)]
        if finite.size == 0:
            return ""

        has_non_positive = bool(np.any(finite <= 0))
        positive = finite[finite > 0]
        span = (finite.max() / positive.min()) if positive.size else 0.0

        if mode == "Linear":
            ax.set_yscale('linear')
            return ""

        if mode == "Log":
            if has_non_positive:
                ax.set_yscale('linear')
                message = (
                    "A logarithmic Y axis cannot be used for this selection.\n\n"
                    "At least one of the plotted curves contains a zero (or negative) "
                    "value, and log(0) is undefined.\n\n"
                    "This is normal and not an error: a nuclide that is bred during "
                    "irradiation — Pu239, Am241, or any fission product — starts at "
                    "exactly zero atoms before the first depletion step.\n\n"
                    "The plot has been drawn on a linear axis instead. To use a "
                    "logarithmic axis, either deselect the nuclides that start at "
                    "zero, or switch Units to 'Weight %' or 'Normalized N/N0'."
                )
                if not getattr(self, '_log_scale_notice_shown', False):
                    self._log_scale_notice_shown = True
                    QMessageBox.information(
                        self.tally_plot_dialog, "Logarithmic Scale Not Applicable", message)
                self._log("ℹ️ Log Y axis refused: a plotted curve contains zero "
                          "(a bred nuclide starts at 0 atoms). Using a linear axis.")
                return "log refused (a curve starts at zero) — linear used"
            ax.set_yscale('log')
            return "log scale"

        # Auto
        if not has_non_positive and span > 100:
            ax.set_yscale('log')
            return "auto: log"
        ax.set_yscale('linear')
        return "auto: linear"

    def _dep_series(self, nuc_name, unit, mat_sel):
        """One nuclide's curve in the requested unit, or (None, reason).

        Units other than raw atoms need extra context that is all already in
        the file: the mass number (from the nuclide's own name), the material
        volume (for atom density), or the material's total mass (for weight %).
        """
        d = self._depletion_data
        atoms_by_mat = d.get('atoms_by_mat')
        nuc_index = d.get('nuc_index', {})
        mat_index = d.get('mat_index', {})
        mat_volume = d.get('mat_volume', {})

        if atoms_by_mat is None or atoms_by_mat.ndim != 3 or nuc_name not in nuc_index:
            return None, "no inventory data"
        ni = nuc_index[nuc_name]
        if ni >= atoms_by_mat.shape[2]:
            return None, "index out of range"

        if mat_sel is None:
            atoms = atoms_by_mat[:, :, ni].sum(axis=1)
            volume = sum(mat_volume.get(m, 0.0) for m in mat_index)
            block = atoms_by_mat.sum(axis=1)                  # (steps, nuclides)
        else:
            if mat_sel not in mat_index:
                return None, "material not found"
            mi = mat_index[mat_sel]
            atoms = atoms_by_mat[:, mi, ni]
            volume = mat_volume.get(mat_sel, 0.0)
            block = atoms_by_mat[:, mi, :]                    # (steps, nuclides)

        atoms = np.asarray(atoms, dtype=float)

        if unit == "Atoms":
            return atoms, None

        _el, mass_number = _split_nuclide_name(nuc_name)
        if unit == "Mass (g)":
            if not mass_number:
                return None, "mass number unknown"
            return atoms * mass_number / _AVOGADRO, None

        if unit == "Weight %":
            if not mass_number:
                return None, "mass number unknown"
            # Total mass of the material at each step, from every nuclide in it.
            total = np.zeros(block.shape[0], dtype=float)
            for name, idx in nuc_index.items():
                _e, a = _split_nuclide_name(name)
                if a and idx < block.shape[1]:
                    total += block[:, idx] * a
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = np.where(total > 0, atoms * mass_number / total * 100.0, np.nan)
            return pct, None

        if unit == "Atom density (a/b-cm)":
            if volume <= 0:
                return None, "material volume not stored in this file"
            # atoms/cm^3 x 1e-24 cm^2/barn
            return atoms / volume * 1.0e-24, None

        if unit == "Normalized N/N0":
            if atoms.size == 0 or atoms[0] == 0:
                return None, "starts at zero (cannot normalise)"
            return atoms / atoms[0], None

        return atoms, None

    def _plot_depletion_data(self):
        """Plot depletion, burnup, and isotope data"""
        if not hasattr(self, '_depletion_data') or not self._depletion_data:
            return

        time_days = self._depletion_data['time_days']
        burnup_data = self._depletion_data.get('burnup')
        k_val = self._depletion_data['k_val']
        k_err = self._depletion_data['k_err']
        nuc_index = self._depletion_data.get('nuc_index', {})
        mat_index = self._depletion_data.get('mat_index', {})
        atoms_by_mat = self._depletion_data.get('atoms_by_mat', None)

        x_mode = self.tally_plot_dialog.combo_dep_x.currentText()
        y_text = self.tally_plot_dialog.combo_dep_y.currentText()
        y_mode = y_text

        # Setup X-Axis
        if "Burnup" in x_mode:
            if burnup_data is not None:
                x_data = burnup_data
                if getattr(self, '_x_is_energy', False):
                    # Energy only -- no heavy metal was found to divide by, so
                    # say so rather than mislabel megawatt-days as burnup.
                    x_label = 'Energy Produced (MWd) - [no heavy metal found]'
                else:
                    hm = self._depletion_data.get('hm_mass_kg', 0.0)
                    x_label = (f'Burnup (MWd/kgHM)  [m_HM = {hm:.4g} kg]'
                               if hm else 'Burnup (MWd/kgHM)')
            else:
                x_data = time_days
                x_label = 'Time (Days) - [Burnup data missing in file]'
        else:
            x_data = time_days
            x_label = 'Time (Days)'

        self._clear_tally_plot()
        ax = self.tally_plot_dialog.ax

        status = ""

        # Setup Y-Axis
        if y_mode == "k-effective":
            ax.errorbar(x_data, k_val, yerr=k_err, marker='o', linestyle='-',
                        color='#d35400', ecolor='red', capsize=5, markersize=6, linewidth=2, label='k-effective')
            ax.axhline(1.0, color='green', linestyle='--', linewidth=1.2, alpha=0.8, label='critical (k = 1)')
            ax.set_ylabel('k-effective', fontsize=10, fontweight='bold')
            ax.set_title('Reactor k-effective vs. Depletion', fontsize=12, fontweight='bold')
            x_end, _ = _cycle_end(x_data, k_val)
            if x_end is not None:
                ax.axvline(x_end, color='purple', linestyle=':', linewidth=1.5,
                           label=f'cycle end @ {x_end:.4g}')
                status = f"Cycle end (k = 1) at {x_end:.4g} — "
            status += f"k: {k_val[0]:.5f} → {k_val[-1]:.5f}"

        elif y_mode == "Reactivity (pcm)":
            rho, rho_err = _reactivity_pcm(k_val, k_err)
            ax.errorbar(x_data, rho, yerr=rho_err, marker='o', linestyle='-',
                        color='#c0392b', ecolor='#e59866', capsize=5, markersize=6,
                        linewidth=2, label='reactivity')
            ax.axhline(0.0, color='green', linestyle='--', linewidth=1.2, alpha=0.8,
                       label='critical (rho = 0)')
            ax.set_ylabel('Reactivity  ρ = (k−1)/k  [pcm]', fontsize=10, fontweight='bold')
            ax.set_title('Reactivity vs. Depletion', fontsize=12, fontweight='bold')
            x_end, _ = _cycle_end(x_data, k_val)
            if x_end is not None:
                ax.axvline(x_end, color='purple', linestyle=':', linewidth=1.5,
                           label=f'cycle end @ {x_end:.4g}')
            finite = rho[np.isfinite(rho)]
            if finite.size:
                swing = finite[0] - finite[-1]
                status = (f"ρ: {finite[0]:.0f} → {finite[-1]:.0f} pcm "
                          f"(swing {swing:.0f} pcm)")
                if x_end is not None:
                    status += f" — cycle end at {x_end:.4g}"

        else:  # Nuclide inventory
            unit = self.tally_plot_dialog.combo_dep_units.currentText()
            mat_sel = self.tally_plot_dialog.combo_dep_mat.currentData()
            scope = "all materials" if mat_sel is None else f"material {mat_sel}"
            selected = [i.text() for i in
                        self.tally_plot_dialog.list_dep_nuclides.selectedItems()]

            plotted, skipped, curves = [], [], []
            for nuc_name in selected:
                y_data, reason = self._dep_series(nuc_name, unit, mat_sel)
                if y_data is None:
                    skipped.append(f"{nuc_name} ({reason})")
                    continue
                ax.plot(x_data[:len(y_data)], y_data, marker='s', linestyle='-',
                        markersize=5, linewidth=1.8, label=nuc_name)
                plotted.append(nuc_name)
                curves.append(y_data)

            if plotted:
                ax.set_ylabel(f'{unit} — {scope}', fontsize=10, fontweight='bold')
                title_nucs = ", ".join(plotted[:4]) + ("…" if len(plotted) > 4 else "")
                ax.set_title(f'Inventory: {title_nucs} — {scope}',
                             fontsize=12, fontweight='bold')
                scale_note = self._apply_y_scale(ax, curves)
                status = f"{len(plotted)} nuclide(s) in {unit}, {scope}"
                if scale_note:
                    status += f" | {scale_note}"
                if skipped:
                    status += f" | skipped: {', '.join(skipped)}"
            else:
                msg = "Select one or more nuclides from the list."
                if skipped:
                    msg = "Could not plot: " + ", ".join(skipped)
                ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes)
                status = msg

        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        if ax.lines or ax.containers:
            ax.legend(loc='best', fontsize=8)

        self.tally_plot_dialog.figure.tight_layout()
        self.tally_plot_dialog.canvas.draw_idle()
        self.tally_plot_dialog.status_label.setText(f"✅ {y_text} vs {x_label}  |  {status}")

    def process_statepoint(self, filepath):
        try:
            if self.sp is not None:
                try:
                    self.sp.close()
                except Exception:
                    pass
            self.sp = None

            if "depletion_results" in os.path.basename(filepath).lower():
                self._process_depletion_results(filepath)
                return

            self.sp = openmc.StatePoint(filepath)
            try:
                if hasattr(self.sp, 'k_combined') and self.sp.k_combined is not None:
                    keff = self.sp.k_combined
                    self.lbl_keff.setText(f"k-effective = {keff.nominal_value:.5f} ± {keff.std_dev:.5f}")
                    self.lbl_keff.setStyleSheet("font-size: 28px; font-weight: bold; color: #217346; padding: 20px;")
                else:
                    self.lbl_keff.setText("k-effective = N/A (Fixed Source Simulation)")
                    self.lbl_keff.setStyleSheet("font-size: 20px; font-weight: bold; color: #555; padding: 20px;")
            except Exception:
                self.lbl_keff.setText("k-effective = N/A (Fixed Source Simulation)")
                self.lbl_keff.setStyleSheet("font-size: 20px; font-weight: bold; color: #555; padding: 20px;")

            self.combo_tallies.clear()
            self._last_spectrum_keV = None
            self._last_spectrum_counts = None
            self._last_geb_abc = None
            self._current_fig = None
            self._current_ax = None
            self.table_peaks.setRowCount(0)
            if hasattr(self.sp, 'tallies') and self.sp.tallies:
                tallies_source = self.sp.tallies.items() if isinstance(self.sp.tallies, dict) else enumerate(
                    self.sp.tallies)
                for tally_id, tally in tallies_source:
                    name = tally.name if tally.name else f"Tally {tally_id}"
                    self.combo_tallies.addItem(f"ID {tally_id}: {name}", tally_id)
            else:
                self.combo_tallies.addItem("No tallies found.")
                self.table_tally.clear()
                self.table_tally.setRowCount(0)
                self.table_tally.setColumnCount(0)
                self.btn_open_hpge.setVisible(False)
        except Exception as e:
            self.lbl_keff.setText("⚠️ Error loading file")
            self._log(f"⚠️ Error loading StatePoint: {e}")

    def display_tally(self, index):
        tally_id = self.combo_tallies.currentData()
        if tally_id is None:
            self.btn_open_hpge.setVisible(False)
            return

        if tally_id == -1:
            self.btn_open_hpge.setVisible(False)
            if hasattr(self.tally_plot_dialog, 'depletion_controls'):
                self.tally_plot_dialog.combo_plot_type.setVisible(False)
                self.tally_plot_dialog.depletion_controls.setVisible(True)
                if self.tally_plot_dialog.isVisible():
                    self._plot_depletion_data()
            return

        # Restore normal tally settings
        if hasattr(self.tally_plot_dialog, 'depletion_controls'):
            self.tally_plot_dialog.depletion_controls.setVisible(False)
            self.tally_plot_dialog.combo_plot_type.setVisible(True)

        if not self.sp or index < 0 or not hasattr(self.sp, 'tallies') or not self.sp.tallies:
            self.btn_open_hpge.setVisible(False)
            return

        try:
            tally = self.sp.tallies[tally_id]
            self._refresh_mesh_score_combo(tally)

            if 'pulse-height' in tally.scores:
                self.btn_open_hpge.setVisible(True)
            else:
                self.btn_open_hpge.setVisible(False)

            try:
                df = tally.get_pandas_dataframe()
            except Exception as e_df:
                self._log(f"⚠️ OpenMC failed to parse Pandas DataFrame ({e_df}). Using generic raw data display.")
                df = pd.DataFrame()
                if hasattr(tally, 'mean'):
                    try:
                        df['mean'] = np.asarray(tally.mean, dtype=float).flatten()
                    except Exception:
                        pass
                if hasattr(tally, 'std_dev'):
                    try:
                        df['std_dev'] = np.asarray(tally.std_dev, dtype=float).flatten()
                    except Exception:
                        pass
                if df.empty:
                    df['Status'] = ["Complex Tally Data: Please Use Export to CSV."]

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(filter(None, map(str, col))) for col in df.columns]
            else:
                df.columns = df.columns.astype(str)

            max_preview_rows = 2000
            truncated = df.shape[0] > max_preview_rows
            df_preview = df.iloc[:max_preview_rows] if truncated else df

            self.table_tally.setRowCount(df_preview.shape[0])
            self.table_tally.setColumnCount(df_preview.shape[1])
            self.table_tally.setHorizontalHeaderLabels(list(df_preview.columns))

            self.table_tally.setUpdatesEnabled(False)
            try:
                for row in range(df_preview.shape[0]):
                    for col in range(df_preview.shape[1]):
                        val = df_preview.iloc[row, col]

                        if isinstance(val, (list, tuple, np.ndarray)):
                            val_str = str(val)
                        elif isinstance(val, (float, np.floating)) and pd.isna(val):
                            val_str = "NaN"
                        elif isinstance(val, (float, np.floating)):
                            val_str = f"{val:.5e}"
                        else:
                            val_str = str(val)

                        item = QTableWidgetItem(val_str)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.table_tally.setItem(row, col, item)
            finally:
                self.table_tally.setUpdatesEnabled(True)

            if truncated:
                self._log(f"ℹ️ Tally preview showing the first {max_preview_rows} of {df.shape[0]} rows "
                          f"-- use 'Export to CSV' for the complete data.")

            if self.tally_plot_dialog.isVisible():
                self._plot_selected_tally()

        except Exception as e:
            self._log(f"⚠️ Could not display tally data: {e}")

    # ==========================================================
    # --- Advanced Window Plot Logic ---
    # ==========================================================
    def _clear_tally_plot(self):
        self.tally_plot_dialog.figure.clear()
        self.tally_plot_dialog.ax = self.tally_plot_dialog.figure.add_subplot(111)

    def _get_tally_filter_names(self, tally):
        names = []
        try:
            for filt in getattr(tally, "filters", []):
                names.append(filt.__class__.__name__)
        except Exception:
            pass
        return names

    def _find_energy_filter_for_plot(self, tally):
        try:
            return tally.find_filter(openmc.EnergyFilter)
        except Exception:
            return None

    def _plot_selected_tally(self):
        if not getattr(self, "combo_tallies", None):
            return

        tally_id = self.combo_tallies.currentData()
        if tally_id is None or tally_id == -1:
            return

        if not self.sp or not hasattr(self.sp, "tallies") or not self.sp.tallies:
            return

        try:
            tally = self.sp.tallies[tally_id]
            requested = self.tally_plot_dialog.combo_plot_type.currentText()
            filters = self._get_tally_filter_names(tally)

            mesh_filter = self._find_mesh_filter(tally)
            if requested == "Mesh Heatmap" or (
                    requested == "Auto" and mesh_filter is not None
            ):
                if mesh_filter is not None:
                    if self._plot_tally_mesh_embedded(tally, tally_id, mesh_filter):
                        return

            scores = list(getattr(tally, "scores", []) or [])
            score = scores[0] if scores else "mean"

            try:
                plot_tally = tally.get_slice(scores=[score]) if scores else tally
                raw_mean = np.asarray(plot_tally.mean)
            except Exception:
                raw_mean = np.asarray(tally.mean)

            raw_mean = np.squeeze(raw_mean)
            if raw_mean.size == 0:
                self.tally_plot_dialog.status_label.setText(
                    "No numerical data available for the selected Tally."
                )
                return

            try:
                y = np.asarray(raw_mean, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                values = []
                for item in np.asarray(raw_mean, dtype=object).reshape(-1):
                    try:
                        values.append(float(np.asarray(item).reshape(-1)[0]))
                    except (TypeError, ValueError, IndexError):
                        values.append(0.0)
                y = np.asarray(values, dtype=float)

            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            if np.all(y == 0.0):
                self.tally_plot_dialog.status_label.setText(
                    "The selected Tally has no numerical values to plot."
                )
                return

            y_col = score
            self._clear_tally_plot()
            ax = self.tally_plot_dialog.ax

            energy_filter = self._find_energy_filter_for_plot(tally)
            auto_energy = requested == "Auto" and energy_filter is not None

            if requested == "Line" or auto_energy:
                x = np.arange(1, len(y) + 1, dtype=float)
                xlabel = "Bin / Filter Index"
                if energy_filter is not None:
                    try:
                        bins = np.asarray(energy_filter.bins, dtype=float)
                        if bins.ndim == 2 and bins.shape[0] == len(y):
                            x = ((bins[:, 0] + bins[:, 1]) / 2.0) / 1000.0
                            xlabel = "Energy [keV]"
                    except (TypeError, ValueError, IndexError):
                        pass

                ax.plot(x, y, linewidth=1.6)
                ax.fill_between(x, y, 0, alpha=0.10)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(str(y_col))
                ax.set_title(f"Tally {tally_id} — {tally.name or 'Result'}")
                ax.grid(True, alpha=0.25)

            else:
                x = np.arange(len(y))
                ax.bar(x, y, alpha=0.8)
                ax.set_xlabel("Filter / Result Index")
                ax.set_ylabel(str(y_col))
                ax.set_title(f"Tally {tally_id} — {tally.name or 'Result'}")
                ax.grid(True, axis="y", alpha=0.25)

            self.tally_plot_dialog.figure.tight_layout()
            self.tally_plot_dialog.canvas.draw_idle()

            self.tally_plot_dialog.status_label.setText(
                f"Showing: Tally {tally_id} | Score: {y_col} | "
                f"Filters: {', '.join(filters) if filters else 'Global'}"
            )

        except Exception as e:
            self.tally_plot_dialog.status_label.setText(f"Plot unavailable: {e}")
            self._log(f"⚠️ Automatic tally plot failed: {e}")

    def _plot_tally_mesh_embedded(self, tally, tally_id, mesh_filter):
        try:
            mesh = mesh_filter.mesh
            dims = tuple(int(d) for d in mesh.dimension)
            lower_left = tuple(float(v) for v in mesh.lower_left)
            upper_right = tuple(float(v) for v in mesh.upper_right)

            score = None
            for s in getattr(tally, "scores", []):
                if s != "pulse-height":
                    score = s
                    break
            if not score:
                return False

            tslice = tally.get_slice(scores=[score])
            try:
                arr = np.asarray(
                    tslice.get_reshaped_data(value="mean")
                )
                arr = np.squeeze(arr)
            except (AttributeError, TypeError, ValueError):
                try:
                    arr = np.asarray(tslice.mean, dtype=float).reshape(-1)
                except Exception:
                    return False
                if arr.size != np.prod(dims):
                    return False
                arr = arr.reshape(dims, order="F")

            arr = np.squeeze(arr)
            non_unit_axes = [i for i, d in enumerate(dims) if d > 1]
            self._clear_tally_plot()
            ax = self.tally_plot_dialog.ax

            if arr.ndim <= 1 or len(non_unit_axes) <= 1:
                axis = non_unit_axes[0] if non_unit_axes else 0
                n = dims[axis]
                lo, hi = lower_left[axis], upper_right[axis]
                centers = lo + (np.arange(n) + 0.5) * (hi - lo) / n
                values = arr.flatten()[:n]
                ax.bar(centers, values, width=(hi - lo) / n * 0.9)
                ax.set_xlabel(["X", "Y", "Z"][axis] + " [cm]")
                ax.set_ylabel(f"{score} [per source particle]")
                ax.set_title(f"Mesh Tally {tally_id} — {score}")
            else:
                if arr.ndim == 3:
                    plot_data = arr.sum(axis=2)
                    h_axis, v_axis = 0, 1
                    suffix = " (summed over Z)"
                else:
                    plot_data = arr
                    h_axis, v_axis = non_unit_axes[0], non_unit_axes[1]
                    suffix = ""

                extent = (
                    lower_left[h_axis], upper_right[h_axis],
                    lower_left[v_axis], upper_right[v_axis]
                )
                im = ax.imshow(
                    plot_data.T,
                    origin="lower",
                    extent=extent,
                    aspect="auto"
                )
                self.tally_plot_dialog.figure.colorbar(
                    im, ax=ax, label=f"{score} [per source particle]"
                )
                labels = ["X", "Y", "Z"]
                ax.set_xlabel(f"{labels[h_axis]} [cm]")
                ax.set_ylabel(f"{labels[v_axis]} [cm]")
                ax.set_title(f"Mesh Tally {tally_id} — {score}{suffix}")

            self.tally_plot_dialog.figure.tight_layout()
            self.tally_plot_dialog.canvas.draw_idle()
            self.tally_plot_dialog.status_label.setText(
                f"Showing Mesh Tally {tally_id} | Score: {score} | "
                f"Dimensions: {dims}"
            )
            return True

        except Exception as e:
            self._log(f"⚠️ Embedded mesh tally plot failed: {e}")
            return False

    def _save_tally_plot(self):
        if self.tally_plot_dialog.figure is None:
            QMessageBox.information(
                self, "Save Plot", "Generate a Tally plot first."
            )
            return

        tally_id = self.combo_tallies.currentData()
        default_name = (
            f"Tally_{tally_id}_Plot.png"
            if tally_id is not None else "Tally_Plot.png"
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Tally Plot",
            os.path.join(os.getcwd(), default_name),
            "PNG Image (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not save_path:
            return

        try:
            self.tally_plot_dialog.figure.savefig(
                save_path, dpi=200, bbox_inches="tight"
            )
            self.tally_plot_dialog.status_label.setText(
                f"✅ Plot saved: {os.path.basename(save_path)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Save Plot Error", f"Could not save plot: {e}"
            )

    def export_plot_to_excel(self):
        ax = None
        if hasattr(self, 'hpge_dialog') and self.hpge_dialog.isVisible() and self._current_ax is not None:
            ax = self._current_ax
        elif hasattr(self,
                     'tally_plot_dialog') and self.tally_plot_dialog.isVisible() and self.tally_plot_dialog.ax is not None:
            ax = self.tally_plot_dialog.ax
        else:
            QMessageBox.warning(self, "Warning", "No active plot window is open to export data from.")
            return

        if ax is None or (not ax.lines and not ax.patches):
            QMessageBox.warning(self, "Warning", "No line or bar data found in the current plot.")
            return

        data_dict = {}
        for i, line in enumerate(ax.lines):
            label = line.get_label() if line.get_label() and not line.get_label().startswith('_') else f"Line_{i + 1}"
            data_dict[f"{label}_X"] = line.get_xdata()
            data_dict[f"{label}_Y"] = line.get_ydata()

        if ax.patches:
            x_vals = [p.get_x() + p.get_width() / 2 for p in ax.patches]
            y_vals = [p.get_height() for p in ax.patches]
            data_dict["Bars_X"] = x_vals
            data_dict["Bars_Y"] = y_vals

        if not data_dict:
            QMessageBox.warning(self, "Warning", "Could not extract numerical 1D data from this plot type.")
            return

        try:
            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot Data to Excel/CSV",
                                                       os.path.join(os.getcwd(), "Plot_Raw_Data.csv"),
                                                       "CSV Files (*.csv)")
            if save_path:
                df.to_csv(save_path, index=False)
                QMessageBox.information(self, "Success",
                                        f"Plot data successfully exported to:\n{save_path}\n\nYou can now open this in Excel or OriginLab.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_to_csv(self):
        tally_id = self.combo_tallies.currentData()

        if tally_id == -1:
            try:
                rows = self.table_tally.rowCount()
                cols = self.table_tally.columnCount()
                data = []
                for r in range(rows):
                    row_data = []
                    for c in range(cols):
                        item = self.table_tally.item(r, c)
                        row_data.append(item.text() if item else "")
                    data.append(row_data)
                df = pd.DataFrame(data, columns=["Time [Days]", "k-effective", "Std Dev"])
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Excel CSV", os.path.join(os.getcwd(), "Depletion_Results.csv"), "CSV Files (*.csv)"
                )
                if save_path:
                    df.to_csv(save_path, index=False)
                    self.lbl_file.setText(f"✅ Exported successfully to: {os.path.basename(save_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export Error: {e}")
            return

        if not self.sp or not hasattr(self.sp, 'tallies') or not self.sp.tallies:
            QMessageBox.warning(self, "Warning", "Load a StatePoint file with valid tallies first.")
            return

        if tally_id is None:
            QMessageBox.warning(self, "Warning", "Select a valid Tally from the dropdown first.")
            return

        try:
            tally = self.sp.tallies[tally_id]
            df = tally.get_pandas_dataframe()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read tally data: {e}")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel CSV", os.path.join(os.getcwd(), f"Tally_{tally_id}_Results.csv"), "CSV Files (*.csv)"
        )
        if save_path:
            try:
                df.to_csv(save_path, index=False)
                self.lbl_file.setText(f"✅ Exported successfully to: {os.path.basename(save_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export Error: {e}")

    def apply_geb_broadening(self, bin_centers_eV, counts, a, b, c, n_sigma=6.0):
        E_MeV = np.asarray(bin_centers_eV, dtype=float) * 1.0e-6
        counts = np.asarray(counts, dtype=float)

        fwhm_MeV = a + b * np.sqrt(np.clip(E_MeV + c * E_MeV ** 2, 0.0, None))
        sigma_MeV = np.maximum(fwhm_MeV / 2.3548, 1e-12)

        n = len(counts)
        d_MeV = (E_MeV[1] - E_MeV[0]) if n > 1 else 1.0
        broadened = np.zeros(n, dtype=float)

        for i in range(n):
            N0 = counts[i]
            if N0 == 0.0:
                continue
            half_width_bins = int(np.ceil(n_sigma * sigma_MeV[i] / d_MeV)) + 1
            lo = max(0, i - half_width_bins)
            hi = min(n, i + half_width_bins + 1)
            w = np.exp(-0.5 * ((E_MeV[lo:hi] - E_MeV[i]) / sigma_MeV[i]) ** 2)
            w /= w.sum()
            broadened[lo:hi] += N0 * w

        return broadened

    def plot_spectrum_graph(self):
        if not self.sp or not hasattr(self.sp, 'tallies') or not self.sp.tallies:
            QMessageBox.warning(self.hpge_dialog, "Warning", "Please load a StatePoint file with valid tallies first.")
            return

        tally_id = self.combo_tallies.currentData()
        if tally_id is None:
            QMessageBox.warning(self.hpge_dialog, "Warning",
                                "Please select a valid Tally from the dropdown menu first.")
            return

        try:
            a_val = float(self.input_a.text())
            b_val = float(self.input_b.text())
            c_val = float(self.input_c.text())
        except ValueError:
            QMessageBox.critical(self.hpge_dialog, "Input Error",
                                 "Please enter valid numeric values for GEB parameters (a, b, c).")
            return

        try:
            tally = self.sp.tallies[tally_id]
            energy_filter = tally.find_filter(openmc.EnergyFilter)
            if not energy_filter:
                QMessageBox.warning(self.hpge_dialog, "Warning",
                                    "Selected Tally does not contain an Energy Filter for plotting spectrum.")
                return

            midpoints_eV = [(b[0] + b[1]) / 2 for b in energy_filter.bins]
            midpoints_keV = [e / 1000.0 for e in midpoints_eV]

            fig, ax1 = plt.subplots(figsize=(9, 5))
            self._last_spectrum_keV = None
            self._last_spectrum_counts = None
            self.table_peaks.setRowCount(0)

            if 'pulse-height' in tally.scores:
                raw_values = tally.get_slice(scores=['pulse-height']).mean.flatten()

                broadened_values = self.apply_geb_broadening(
                    midpoints_eV, raw_values, a=a_val, b=b_val, c=c_val
                )

                broadened_values = np.where(broadened_values <= 0, 1e-20, broadened_values)

                ax1.plot(midpoints_keV, broadened_values, color='purple', linewidth=1.2, label='Pulse-Height + GEB')
                ax1.set_yscale('log')
                ax1.set_ylabel('Broadened Counts / Bin [per source particle]', color='purple', fontweight='bold')
                ax1.set_title(f'HPGe Detector Spectrum - Tally {tally_id} (GEB Applied)')
                ax1.grid(True, which="both", linestyle='--', alpha=0.6)

                self._last_spectrum_keV = np.asarray(midpoints_keV, dtype=float)
                self._last_spectrum_counts = np.asarray(broadened_values, dtype=float)
                self._last_geb_abc = (a_val, b_val, c_val)
            else:
                heating = tally.get_slice(
                    scores=['heating']).mean.flatten() if 'heating' in tally.scores else np.zeros_like(midpoints_eV)
                flux = tally.get_slice(scores=['flux']).mean.flatten() if 'flux' in tally.scores else np.zeros_like(
                    midpoints_eV)

                ax1.plot(midpoints_keV, heating, color='red', marker='o', linewidth=1.5, label='Heating')
                ax1.set_xlabel('Energy [keV]')
                ax1.set_ylabel('Heating [eV/source]', color='red')
                ax1.tick_params(axis='y', labelcolor='red')
                ax1.grid(True, linestyle='--', alpha=0.6)

                ax2 = ax1.twinx()
                ax2.plot(midpoints_keV, flux, color='blue', marker='s', linewidth=1.5, label='Flux')
                ax2.set_ylabel('Flux [particles/cm^2/source]', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
                ax1.set_title(f'Detector Spectrum: Tally {tally_id} Analysis')

            ax1.set_xlabel('Energy [keV]', fontsize=12, fontweight='bold')
            fig.tight_layout()
            fig.show()

            self._current_fig = fig
            self._current_ax = ax1

        except Exception as e:
            QMessageBox.critical(self.hpge_dialog, "Plot Error", f"Failed to generate graph: {str(e)}")

    def _find_mesh_filter(self, tally):
        for filter_cls_name in ('MeshFilter', 'MeshSurfaceFilter'):
            filter_cls = getattr(openmc, filter_cls_name, None)
            if filter_cls is None:
                continue
            try:
                mesh_filter = tally.find_filter(filter_cls)
            except Exception:
                mesh_filter = None
            if mesh_filter is not None:
                return mesh_filter
        return None

    def _refresh_mesh_score_combo(self, tally):
        self.combo_mesh_score.clear()
        mesh_filter = self._find_mesh_filter(tally)
        if mesh_filter is None:
            return
        for score in tally.scores:
            if score == 'pulse-height':
                continue
            self.combo_mesh_score.addItem(score)

    def plot_mesh_tally(self):
        if not self.sp or not hasattr(self.sp, 'tallies') or not self.sp.tallies:
            QMessageBox.warning(self, "Warning", "Please load a StatePoint file with valid tallies first.")
            return

        tally_id = self.combo_tallies.currentData()
        if tally_id is None:
            QMessageBox.warning(self, "Warning", "Please select a valid Tally from the dropdown menu first.")
            return

        try:
            tally = self.sp.tallies[tally_id]
            mesh_filter = self._find_mesh_filter(tally)
            if mesh_filter is None:
                QMessageBox.warning(self, "Warning",
                                    "Selected Tally does not contain a Mesh or MeshSurface Filter.")
                return

            score = self.combo_mesh_score.currentText()
            if not score:
                QMessageBox.warning(self, "Warning", "Selected Tally has no plottable scores.")
                return

            mesh = mesh_filter.mesh
            dims = tuple(int(d) for d in mesh.dimension)
            lower_left = tuple(float(v) for v in mesh.lower_left)
            upper_right = tuple(float(v) for v in mesh.upper_right)

            tslice = tally.get_slice(scores=[score])

            arr = None
            reshape_is_verified = True
            try:
                arr = np.asarray(tslice.get_reshaped_data(value='mean'))
                arr = np.squeeze(arr)
            except AttributeError:
                arr = None

            if arr is None:
                reshape_is_verified = False
                arr = np.asarray(tslice.mean, dtype=float).flatten()
                if arr.size != np.prod(dims):
                    QMessageBox.critical(
                        self, "Plot Error",
                        f"Could not reshape tally data ({arr.size} values) to the mesh dimensions "
                        f"{dims} ({np.prod(dims)} cells) -- the tally may combine the Mesh Filter "
                        f"with another filter, which this simplified fallback can't separate. Use "
                        f"'Export to CSV' and OpenMC's own get_pandas_dataframe() for exact cell mapping.")
                    return
                arr = arr.reshape(dims, order='F')
                self._log("⚠️ Mesh reshape used a best-effort cell ordering (this OpenMC version has no "
                          "get_reshaped_data) -- verify against 'Export to CSV' if exact spatial mapping matters.")

            arr = np.squeeze(arr)
            non_unit_axes = [i for i, d in enumerate(dims) if d > 1]

            fig, ax = plt.subplots(figsize=(8, 6))

            if arr.ndim <= 1 or len(non_unit_axes) <= 1:
                axis = non_unit_axes[0] if non_unit_axes else 0
                n = dims[axis]
                lo, hi = lower_left[axis], upper_right[axis]
                centers = lo + (np.arange(n) + 0.5) * (hi - lo) / n
                values_1d = arr.flatten()[:n]
                axis_label = ['X', 'Y', 'Z'][axis]
                ax.bar(centers, values_1d, width=(hi - lo) / n * 0.9, color='#d97706')
                ax.set_xlabel(f'{axis_label} [cm]', fontweight='bold')
                ax.set_ylabel(f'{score} [per source particle]')
                ax.set_title(f'Mesh Tally {tally_id} -- {score} (1D, {axis_label}-axis)')
            else:
                if arr.ndim == 3:
                    plot_data = arr.sum(axis=2)
                    title_suffix = ' (summed over Z -- 3D mesh projection)'
                    h_axis, v_axis = 0, 1
                else:
                    plot_data = arr
                    title_suffix = ''
                    h_axis, v_axis = non_unit_axes[0], non_unit_axes[1]

                extent = (lower_left[h_axis], upper_right[h_axis],
                          lower_left[v_axis], upper_right[v_axis])
                im = ax.imshow(plot_data.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
                fig.colorbar(im, ax=ax, label=f'{score} [per source particle]')
                axis_labels = ['X', 'Y', 'Z']
                ax.set_xlabel(f'{axis_labels[h_axis]} [cm]', fontweight='bold')
                ax.set_ylabel(f'{axis_labels[v_axis]} [cm]', fontweight='bold')
                ax.set_title(f'Mesh Tally {tally_id} -- {score}{title_suffix}')

            fig.tight_layout()
            fig.show()
            self._log(f"✅ Mesh tally plotted ({score}, dims={dims})."
                      + ("" if reshape_is_verified else " Used best-effort reshape ordering."))

        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot mesh tally: {str(e)}")

    def _fit_peaks(self, energies_keV, counts, geb_abc, prominence_frac=0.5, window_factor=5.0):
        a, b, c = geb_abc
        energies = np.asarray(energies_keV, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if counts.size < 5 or counts.max() <= 0:
            return []

        log_counts = np.log10(np.clip(counts, 1e-300, None))
        prominence_db = np.log10(1.0 + max(prominence_frac, 1e-6))

        peak_idx, _ = find_peaks(log_counts, prominence=prominence_db, distance=5)

        d = float(np.median(np.diff(energies))) if energies.size > 1 else 1.0
        n = counts.size
        results = []
        for idx in peak_idx:
            sigma0 = max(_expected_sigma_keV(energies[idx], a, b, c), d * 0.25)
            win = max(6, int(window_factor * sigma0 / d))
            lo = max(0, idx - win)
            hi = min(n, idx + win + 1)
            if hi - lo < 9:
                continue
            xw = energies[lo:hi]
            yw = counts[lo:hi]
            baseline0 = float(np.min(yw))
            net0 = counts[idx] - baseline0

            p0 = [max(net0, 1e-30), energies[idx], sigma0, baseline0]
            bounds = ([0, xw.min(), d * 0.1, 0],
                      [np.inf, xw.max(), sigma0 * 5.0, max(float(yw.max()), 1e-300)])
            try:
                popt, _ = curve_fit(_gaussian, xw, yw, p0=p0, bounds=bounds, maxfev=5000)
                amplitude, mu, sigma, baseline = popt
                if not (xw.min() <= mu <= xw.max()):
                    continue
                if amplitude <= 0 or not np.isfinite(amplitude):
                    continue
                fwhm = 2.3548 * sigma
                net_local = np.maximum(yw - baseline, 0.0)
                area = float(np.sum(net_local))
                results.append(dict(energy=mu, fwhm=fwhm, area=area,
                                    amplitude=amplitude, sigma=sigma, baseline=baseline))
            except Exception:
                continue

        results.sort(key=lambda p: p['energy'])
        return results

    def find_and_fit_peaks(self):
        if self._last_spectrum_keV is None or self._last_spectrum_counts is None:
            QMessageBox.warning(
                self.hpge_dialog, "Warning",
                "Plot a Pulse-Height spectrum first (click 'Plot Spectrum + GEB' on a tally with a "
                "pulse-height score) -- peak fitting needs a detector spectrum to search."
            )
            return

        if self._current_fig is None or not plt.fignum_exists(self._current_fig.number):
            QMessageBox.warning(
                self.hpge_dialog, "Warning",
                "The spectrum window was closed -- click 'Plot Spectrum + GEB' again first."
            )
            return

        try:
            prominence_frac = float(self.input_prominence.text()) / 100.0
        except ValueError:
            QMessageBox.critical(self.hpge_dialog, "Input Error", "Min Prominence must be a number (percent).")
            return

        peaks = self._fit_peaks(self._last_spectrum_keV, self._last_spectrum_counts,
                                self._last_geb_abc, prominence_frac=prominence_frac)
        for pk in peaks:
            pk['nuclide_match'] = None
            pk['fepe'] = None
        self._last_peaks = peaks

        self.table_peaks.setRowCount(len(peaks))
        for row, pk in enumerate(peaks):
            for col, val in enumerate((row + 1, pk['energy'], pk['fwhm'], pk['area'])):
                text = str(val) if col == 0 else f"{val:.5g}"
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table_peaks.setItem(row, col, item)
            for col in (4, 5):
                item = QTableWidgetItem("-")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table_peaks.setItem(row, col, item)

        ax1 = self._current_ax
        for artist in list(ax1.lines) + list(ax1.texts):
            if artist.get_gid() == '_peak_fit':
                artist.remove()

        for pk in peaks:
            half_span = max(6.0 * pk['sigma'], 1e-6)
            local_x = np.linspace(pk['energy'] - half_span, pk['energy'] + half_span, 200)
            ax1.axvline(pk['energy'], color='black', linestyle=':', linewidth=0.6, alpha=0.5, gid='_peak_fit')
            fit_curve = _gaussian(local_x, pk['amplitude'], pk['energy'], pk['sigma'], pk['baseline'])
            ax1.plot(local_x, np.where(fit_curve <= 0, 1e-20, fit_curve),
                     color='black', linewidth=1.0, alpha=0.8, gid='_peak_fit')

        self._current_fig.canvas.draw()
        self._current_fig.show()

        if not peaks:
            QMessageBox.information(self.hpge_dialog, "No Peaks Found",
                                    "No peaks met the prominence threshold. Try lowering 'Min Prominence (%)'.")

    def _extract_source_energy_probs(self):
        main_win = self.window()
        if not hasattr(main_win, 'script_editor'):
            raise ValueError("No script editor found.")
        code = main_win.script_editor.editor.toPlainText()
        if not code.strip():
            raise ValueError("Script is empty.")

        code = re.sub(r'\bopenmc\.run\s*\(\s*\)', 'pass', code)
        code = re.sub(r'\bopenmc\.plot_geometry\s*\(\s*\)', 'pass', code)

        namespace = {'openmc': openmc, 'os': os, '__name__': '__main__'}
        try:
            openmc.reset_auto_ids()
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Could not execute the script safely to extract sources: {e}")

        settings_obj = next((v for v in namespace.values() if isinstance(v, openmc.Settings)), None)
        if settings_obj is None or getattr(settings_obj, 'source', None) is None:
            raise ValueError("No 'openmc.Settings' with a 'source' was found in the script.")

        sources = settings_obj.source
        if not isinstance(sources, (list, tuple)):
            sources = [sources]

        total_strength = sum((getattr(s, 'strength', None) or 1.0) for s in sources)
        if total_strength <= 0:
            total_strength = 1.0

        energy_probs = {}
        for s in sources:
            energy_dist = getattr(s, 'energy', None)
            xs = getattr(energy_dist, 'x', None)
            ps = getattr(energy_dist, 'p', None)
            if xs is None or ps is None:
                continue
            xs = np.asarray(xs, dtype=float)
            ps = np.asarray(ps, dtype=float)
            p_sum = ps.sum()
            if p_sum <= 0:
                continue
            ps = ps / p_sum
            strength_frac = (getattr(s, 'strength', None) or 1.0) / total_strength
            for x_eV, p in zip(xs, ps):
                e_keV = x_eV / 1000.0
                energy_probs[e_keV] = energy_probs.get(e_keV, 0.0) + p * strength_frac

        if not energy_probs:
            raise ValueError(
                "Could not find a discrete source energy distribution (openmc.stats.Discrete) "
                "in settings.source -- FEPE needs to know the per-line sampling probability, "
                "which only a discrete multi-line source has."
            )
        return energy_probs

    def identify_nuclides(self):
        if not self._last_peaks:
            QMessageBox.warning(self.hpge_dialog, "Warning", "Run 'Find & Fit Peaks' first.")
            return

        n_matched = 0
        for row, pk in enumerate(self._last_peaks):
            match = _identify_nuclide(pk['energy'], pk['fwhm'])
            pk['nuclide_match'] = match
            if match:
                n_matched += 1
                nuclide, expected_e, _intensity = match
                text = f"{nuclide} ({expected_e:.2f} keV)"
            else:
                text = "unidentified"
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_peaks.setItem(row, 4, item)

        self._log(f"✅ Identified {n_matched}/{len(self._last_peaks)} peaks against the reference nuclide library.")

    def calculate_fepe_and_plot(self):
        if not self._last_peaks:
            QMessageBox.warning(self.hpge_dialog, "Warning", "Run 'Find & Fit Peaks' first.")
            return
        if not any(pk.get('nuclide_match') for pk in self._last_peaks):
            QMessageBox.warning(self.hpge_dialog, "Warning",
                                "Run 'Identify Nuclides' first -- FEPE needs identified peaks.")
            return

        try:
            energy_probs = self._extract_source_energy_probs()
        except ValueError as e:
            QMessageBox.critical(self.hpge_dialog, "Source Error", str(e))
            return

        prob_energies = np.array(sorted(energy_probs.keys()))
        match_tolerance_keV = 1.0

        fepe_points = []
        for row, pk in enumerate(self._last_peaks):
            match = pk.get('nuclide_match')
            fepe_item = QTableWidgetItem("-")
            fepe_item.setFlags(fepe_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if match:
                nuclide, expected_e, _intensity = match
                nearest_idx = int(np.argmin(np.abs(prob_energies - expected_e)))
                nearest_e = prob_energies[nearest_idx]
                if abs(nearest_e - expected_e) <= match_tolerance_keV:
                    prob = energy_probs[nearest_e]
                    if prob > 0:
                        fepe = pk['area'] / prob
                        pk['fepe'] = fepe
                        fepe_item.setText(f"{fepe:.4g}")
                        fepe_points.append((expected_e, fepe, nuclide))
                else:
                    self._log(f"⚠️ Peak at {pk['energy']:.1f} keV matched to '{nuclide}' "
                              f"({expected_e:.2f} keV), but no source energy line was found "
                              f"within {match_tolerance_keV} keV of it -- skipped.")
            self.table_peaks.setItem(row, 5, fepe_item)

        if not fepe_points:
            QMessageBox.warning(
                self.hpge_dialog, "No FEPE Computed",
                "None of the identified peaks' energies matched a line in the script's source "
                "distribution closely enough."
            )
            return

        best_per_line = {}
        n_duplicates = 0
        for e, fepe, nuclide in fepe_points:
            key = (nuclide, round(e, 3))
            if key not in best_per_line or fepe > best_per_line[key][1]:
                if key in best_per_line:
                    n_duplicates += 1
                best_per_line[key] = (e, fepe, nuclide)
            else:
                n_duplicates += 1
        fepe_points = list(best_per_line.values())
        if n_duplicates:
            self._log(f"ℹ️ {n_duplicates} duplicate peak match(es) for the same line were dropped from "
                      f"the efficiency curve (kept the strongest one each time) -- likely statistical "
                      f"noise peaks landing near a real line. Consider raising 'Min Prominence (%)' if "
                      f"this happens a lot.")

        fepe_points.sort(key=lambda p: p[0])
        energies = [p[0] for p in fepe_points]
        efficiencies = [p[1] for p in fepe_points]
        labels = [p[2] for p in fepe_points]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(energies, efficiencies, 'o-', color='#16a085', markersize=6)
        for e, eff, nuclide in fepe_points:
            ax.annotate(nuclide, (e, eff), xytext=(0, 7), textcoords='offset points',
                        fontsize=7, ha='center')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy [keV]', fontsize=11, fontweight='bold')
        ax.set_ylabel('FEPE (Full-Energy Peak Efficiency)', fontsize=11, fontweight='bold')
        ax.set_title('Detector Efficiency Curve')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.show()

        self._log(f"✅ Computed FEPE for {len(fepe_points)} identified peak(s): "
                  f"{', '.join(labels)}. Efficiency curve plotted.")
