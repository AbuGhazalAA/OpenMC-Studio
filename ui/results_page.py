import os
import glob
import openmc
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QLineEdit,
    QDateEdit, QApplication
)
from PySide6.QtCore import Qt, QDate


def _gaussian(x, amplitude, mu, sigma, baseline):
    return baseline + amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _expected_sigma_keV(energy_keV, a, b, c):
    """The Gaussian sigma a peak at this energy SHOULD have, straight from
    the same GEB formula apply_geb_broadening used to smear it -- not an
    empirical guess. Used to size each peak's fit window."""
    E_MeV = energy_keV / 1000.0
    fwhm_MeV = a + b * np.sqrt(np.clip(E_MeV + c * E_MeV ** 2, 0.0, None))
    return max(fwhm_MeV * 1000.0 / 2.3548, 1e-6)


# Gamma-ray energy (keV) and emission probability (intensity per decay of
# THAT nuclide, not per source particle) for the principal lines of the
# 12-nuclide mixed reference source this page's Nuclide ID / FEPE tools
# target (Am-241, Cd-109, Ce-139, Co-57, Co-60, Cs-137, Ba-133, Sr-85,
# Y-88, Cr-51, Mn-54, Zn-65) -- standard nuclear decay-scheme data, so
# these are safe to ship as fixed constants (unlike source ACTIVITY, which
# is specific to one certified source/batch and decays with time -- that
# lives in the editable Reference Source table in the UI instead).
NUCLIDE_GAMMA_LINES = {
    'Am-241': [(59.541, 0.359)],
    'Cd-109': [(88.034, 0.0366)],
    'Ce-139': [(165.857, 0.799)],
    'Co-57':  [(122.061, 0.8560), (136.474, 0.1068)],
    'Co-60':  [(1173.228, 0.9985), (1332.492, 0.999826)],
    'Cs-137': [(661.657, 0.851)],
    'Ba-133': [(53.163, 0.00141), (79.614, 0.0265), (81.0, 0.3406), (160.6, 0.00638),
               (223.2, 0.00450), (276.4, 0.0716), (302.85, 0.1833), (356.01, 0.6205), (383.85, 0.0894)],
    'Sr-85':  [(514.0, 0.984)],
    'Y-88':   [(898.042, 0.937), (1836.063, 0.992)],
    'Cr-51':  [(320.082, 0.0991)],
    'Mn-54':  [(834.848, 0.9998)],
    'Zn-65':  [(1115.539, 0.506)],
}

# Editable defaults for the Reference Source table: (nuclide, half-life
# value, half-life unit, reference activity [kBq], activity uncertainty
# [kBq]) -- pre-filled from the user's certified mixed reference source
# certificate; the UI table lets these be edited/replaced for a different
# source without touching code.
DEFAULT_REFERENCE_SOURCE = [
    ('Am-241', 432.6, 'y', 2412, 31),
    ('Cd-109', 461.9, 'd', 10.02, 0.16),
    ('Ce-139', 137.641, 'd', 1304, 18),
    ('Co-57', 271.81, 'd', 1182, 15),
    ('Co-60', 5.2711, 'y', 1496, 27),
    ('Cs-137', 30.018, 'y', 1180, 17),
    ('Ba-133', 10.539, 'y', 1092, 19),
    ('Sr-85', 64.850, 'd', 2790, 50),
    ('Y-88', 106.63, 'd', 3434, 48),
    ('Cr-51', 27.704, 'd', 9.89, 0.13),
    ('Mn-54', 312.19, 'd', 3626, 47),
    ('Zn-65', 244.01, 'd', 4331, 56),
]

_HALF_LIFE_UNIT_SECONDS = {'s': 1.0, 'd': 86400.0, 'y': 365.25 * 86400.0}


def _half_life_seconds(value, unit):
    return value * _HALF_LIFE_UNIT_SECONDS.get(unit.strip().lower(), 1.0)


def _decayed_activity(a0, half_life_s, dt_seconds):
    """Radioactive decay law A(t) = A0 * 2^(-t/T_half). dt_seconds may be
    negative (reference date after the analysis date) -- decay works the
    same either direction, just growing backwards toward a higher A0."""
    if half_life_s <= 0:
        return a0
    return a0 * 2.0 ** (-dt_seconds / half_life_s)


def _identify_nuclide(energy_keV, fwhm_keV, tolerance_factor=3.0, min_tolerance_keV=1.0):
    """Match a fitted peak centroid to the nearest known gamma line across
    ALL nuclides in NUCLIDE_GAMMA_LINES, within a tolerance that scales
    with the peak's own fitted width (a sharper peak -> a tighter, more
    confident match window; `min_tolerance_keV` keeps that window sane
    for very narrow low-energy peaks). Returns (nuclide, expected_energy,
    intensity) for the closest match within tolerance, or None."""
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
        self.sp = None  # لتخزين كائن الـ StatePoint
        # Cached from the last "Plot Spectrum + GEB" call, ONLY when it drew
        # a pulse-height spectrum (peak fitting isn't physically meaningful
        # for the heating/flux branch) -- consumed by find_and_fit_peaks().
        self._last_spectrum_keV = None
        self._last_spectrum_counts = None
        # The GEB (a, b, c) that PRODUCED _last_spectrum_counts -- used to
        # size each candidate peak's fit window from the actual expected
        # resolution at that energy (see _expected_sigma_keV), not a
        # fragile empirical width guess. Cached at plot time rather than
        # re-read from the input fields at fit time, since the user may
        # have edited them in between without re-plotting.
        self._last_geb_abc = None
        # The popped-up matplotlib figure/axes from the last spectrum plot
        # -- kept so "Find & Fit Peaks" can overlay results onto that SAME
        # window instead of the plot living embedded in the app (users
        # asked for the standalone window back: it gets more screen space
        # than any panel inside the app, and matplotlib's own toolbar
        # already gives free pan/zoom there).
        self._current_fig = None
        self._current_ax = None
        # The peak dicts from the last "Find & Fit Peaks" call (each may
        # gain a 'nuclide_match' key after "Identify Nuclides" and a
        # 'fepe' key after "Calculate FEPE") -- kept so those two later
        # steps don't need to re-fit anything, just annotate this list.
        self._last_peaks = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 1. قسم تحميل الملف ---
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

        # --- 2. قسم معامل التكاثر النيوتروني (K-Effective) ---
        keff_group = QGroupBox("⚛️ K-Effective (Multiplication Factor)")
        keff_group.setStyleSheet("font-weight: bold; font-size: 14px;")
        keff_layout = QVBoxLayout(keff_group)

        self.lbl_keff = QLabel("k-effective = N/A")
        self.lbl_keff.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_keff.setStyleSheet("font-size: 28px; font-weight: bold; color: #217346; padding: 20px;")
        keff_layout.addWidget(self.lbl_keff)
        layout.addWidget(keff_group)

        # --- 3. قسم العدادات والنتائج (Tallies & GEB Configuration) ---
        tally_group = QGroupBox("📊 Tallies, Spectrum & GEB Calibration")
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

        self.btn_plot = QPushButton("📈 Plot Spectrum + GEB")
        self.btn_plot.setStyleSheet("background-color: #0e639c; color: white; padding: 5px; font-weight: bold;")
        self.btn_plot.clicked.connect(self.plot_spectrum_graph)
        combo_layout.addWidget(self.btn_plot)

        combo_layout.addStretch()
        tally_layout.addLayout(combo_layout)

        # --- قسم معاملات الـ GEB المرنة (قابل للتعديل في أي وقت) ---
        geb_form = QHBoxLayout()
        geb_form.addWidget(QLabel("<b>GEB Parameters:</b>"))

        geb_form.addWidget(QLabel("a:"))
        self.input_a = QLineEdit("0.000496")
        self.input_a.setFixedWidth(80)
        geb_form.addWidget(self.input_a)

        geb_form.addWidget(QLabel("b:"))
        self.input_b = QLineEdit("0.001176")
        self.input_b.setFixedWidth(80)
        geb_form.addWidget(self.input_b)

        geb_form.addWidget(QLabel("c:"))
        self.input_c = QLineEdit("0.0")
        self.input_c.setFixedWidth(80)
        geb_form.addWidget(self.input_c)

        geb_form.addStretch()
        tally_layout.addLayout(geb_form)

        # --- قسم مطابقة القمم (Peak Fitting), يعمل على آخر طيف Pulse-Height مرسوم في النافذة المنبثقة ---
        peak_form = QHBoxLayout()
        peak_form.addWidget(QLabel("<b>Peak Fitting:</b>"))

        peak_form.addWidget(QLabel("Min Prominence (%):"))
        self.input_prominence = QLineEdit("50.0")
        self.input_prominence.setFixedWidth(60)
        self.input_prominence.setToolTip(
            "A peak must rise at least this % ABOVE ITS OWN LOCAL BASELINE\n"
            "(multiplicative -- e.g. 50% means the peak must reach at least\n"
            "1.5x its surrounding continuum), NOT relative to the spectrum's\n"
            "global maximum -- real gamma spectra span orders of magnitude,\n"
            "so a global-max-relative threshold misses real peaks sitting on\n"
            "a lower part of the continuum.\n"
            "Seeing noise-driven false peaks? Raise this. Missing a real\n"
            "peak? Lower it -- real photopeaks are usually far more than\n"
            "50% above their local continuum, so this rarely needs to go\n"
            "below ~10-20% except on a very noisy (low-statistics) run."
        )
        peak_form.addWidget(self.input_prominence)

        self.btn_find_peaks = QPushButton("🔎 Find && Fit Peaks")
        self.btn_find_peaks.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px; font-weight: bold;")
        self.btn_find_peaks.clicked.connect(self.find_and_fit_peaks)
        peak_form.addWidget(self.btn_find_peaks)

        peak_form.addStretch()
        tally_layout.addLayout(peak_form)

        # --- قسم تحديد النظير + حساب FEPE ورسم منحنى الكفاءة ---
        ref_group = QGroupBox("🎯 Reference Source -- Nuclide ID + FEPE / Efficiency Curve")
        ref_layout = QVBoxLayout(ref_group)

        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Reference (certificate) Date:"))
        self.ref_date_edit = QDateEdit(calendarPopup=True)
        self.ref_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.ref_date_edit.setDate(QDate.currentDate())
        self.ref_date_edit.setToolTip(
            "The date the reference source's activities below were certified\n"
            "at (its calibration certificate date). Leave equal to the\n"
            "Simulation Date if you don't need decay correction."
        )
        date_row.addWidget(self.ref_date_edit)

        date_row.addWidget(QLabel("Simulation/Analysis Date:"))
        self.sim_date_edit = QDateEdit(calendarPopup=True)
        self.sim_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.sim_date_edit.setDate(QDate.currentDate())
        date_row.addWidget(self.sim_date_edit)
        date_row.addStretch()
        ref_layout.addLayout(date_row)

        self.table_ref_source = QTableWidget()
        self.table_ref_source.setColumnCount(5)
        self.table_ref_source.setHorizontalHeaderLabels(
            ["Nuclide", "Half-life", "Unit (s/d/y)", "Ref. Activity [kBq]", "Uncertainty [kBq]"])
        self.table_ref_source.setRowCount(len(DEFAULT_REFERENCE_SOURCE))
        for row, (nuclide, half_life, unit, activity, uncertainty) in enumerate(DEFAULT_REFERENCE_SOURCE):
            for col, val in enumerate((nuclide, half_life, unit, activity, uncertainty)):
                self.table_ref_source.setItem(row, col, QTableWidgetItem(str(val)))
        self.table_ref_source.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_ref_source.setMaximumHeight(160)
        self.table_ref_source.setToolTip(
            "Editable: replace with your own certified reference source's\n"
            "nuclides/half-lives/activities if different from the default."
        )
        ref_layout.addWidget(self.table_ref_source)

        ref_btn_row = QHBoxLayout()
        self.btn_identify = QPushButton("🎯 Identify Nuclides")
        self.btn_identify.setStyleSheet("background-color: #2c3e50; color: white; padding: 5px; font-weight: bold;")
        self.btn_identify.clicked.connect(self.identify_nuclides)
        ref_btn_row.addWidget(self.btn_identify)

        self.btn_fepe = QPushButton("📊 Calculate FEPE + Plot Efficiency Curve")
        self.btn_fepe.setStyleSheet("background-color: #16a085; color: white; padding: 5px; font-weight: bold;")
        self.btn_fepe.clicked.connect(self.calculate_fepe_and_plot)
        ref_btn_row.addWidget(self.btn_fepe)
        ref_btn_row.addStretch()
        ref_layout.addLayout(ref_btn_row)

        tally_layout.addWidget(ref_group)

        # جدول عرض البيانات الخام
        self.table_tally = QTableWidget()
        self.table_tally.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_tally.setStyleSheet("font-weight: normal; font-size: 12px;")
        tally_layout.addWidget(self.table_tally)

        # جدول نتائج مطابقة القمم -- صغير أسفل الجدول الخام، الرسم نفسه في نافذة منفصلة
        self.table_peaks = QTableWidget()
        self.table_peaks.setColumnCount(6)
        self.table_peaks.setHorizontalHeaderLabels(
            ["Peak #", "Centroid [keV]", "FWHM [keV]", "Area [counts]", "Nuclide", "FEPE"])
        self.table_peaks.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_peaks.setMaximumHeight(160)
        tally_layout.addWidget(self.table_peaks)

        layout.addWidget(tally_group)

    def _log(self, msg):
        """إرسال ذكي للرسائل إلى نافذة الكونسول أينما كانت -- نفس النمط
        المستخدم في plots_page.py و tracks_page.py."""
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
            print(f"Auto-load failed: {e}")

    def load_statepoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open StatePoint File", os.getcwd(), "HDF5 Files (*.h5);;All Files (*)"
        )
        if file_path:
            self.lbl_file.setText(f"Loaded: {os.path.basename(file_path)}")
            self.process_statepoint(file_path)

    def process_statepoint(self, filepath):
        try:
            # يُغلق أي StatePoint سابق قبل فتح واحد جديد -- بدون هذا يبقى
            # مقبض الملف القديم مفتوحاً حتى تُجمعه Python كنفاية (garbage
            # collection)، وعلى ويندوز هذا قد يمنع لاحقاً إعادة تسمية أو
            # حذف ملف statepoint جديد يحمل نفس المسار (بالضبط ما يحاول
            # main.py._on_simulation_finished تفاديه خارجياً عبر البحث
            # بـ hasattr(val, 'close') قبل بدء أي محاكاة جديدة).
            if self.sp is not None:
                try:
                    self.sp.close()
                except Exception:
                    pass

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
        except Exception as e:
            self.lbl_keff.setText("⚠️ Error loading file")
            print(f"Error loading StatePoint: {e}")

    def display_tally(self, index):
        if not self.sp or index < 0 or not hasattr(self.sp, 'tallies') or not self.sp.tallies:
            return
        tally_id = self.combo_tallies.currentData()
        if tally_id is None:
            return
        try:
            tally = self.sp.tallies[tally_id]
            df = tally.get_pandas_dataframe()

            # معالجة الـ MultiIndex لتجنب أخطاء التسلسل والمصفوفات
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(filter(None, map(str, col))) for col in df.columns]
            else:
                df.columns = df.columns.astype(str)

            self.table_tally.setRowCount(df.shape[0])
            self.table_tally.setColumnCount(df.shape[1])
            self.table_tally.setHorizontalHeaderLabels(list(df.columns))

            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    val = df.iloc[row, col]
                    if isinstance(val, (np.ndarray, list, tuple)):
                        val_str = ", ".join(
                            [f"{v:.5e}" if isinstance(v, (float, np.floating)) else str(v) for v in val])
                    elif isinstance(val, (float, np.floating)):
                        val_str = f"{val:.5e}"
                    else:
                        val_str = str(val)

                    item = QTableWidgetItem(val_str)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table_tally.setItem(row, col, item)
        except Exception as e:
            print(f"Could not display tally data: {e}")

    def export_to_csv(self):
        if not self.sp or self.table_tally.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No tally data available to export.")
            return
        tally_id = self.combo_tallies.currentData()
        tally = self.sp.tallies[tally_id]
        df = tally.get_pandas_dataframe()

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
            QMessageBox.warning(self, "Warning", "Please load a StatePoint file with valid tallies first.")
            return

        tally_id = self.combo_tallies.currentData()
        if tally_id is None:
            QMessageBox.warning(self, "Warning", "Please select a valid Tally from the dropdown menu first.")
            return

        try:
            a_val = float(self.input_a.text())
            b_val = float(self.input_b.text())
            c_val = float(self.input_c.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid numeric values for GEB parameters (a, b, c).")
            return

        try:
            tally = self.sp.tallies[tally_id]
            energy_filter = tally.find_filter(openmc.EnergyFilter)
            if not energy_filter:
                QMessageBox.warning(self, "Warning",
                                    "Selected Tally does not contain an Energy Filter for plotting spectrum.")
                return

            # حساب منتصف الحزم وتحويلها إلى وحدة كيلو إلكترون فولت (keV) لتكون مقروءة وواضحة
            midpoints_eV = [(b[0] + b[1]) / 2 for b in energy_filter.bins]
            midpoints_keV = [e / 1000.0 for e in midpoints_eV]  # تحويل من eV إلى keV

            fig, ax1 = plt.subplots(figsize=(9, 5))
            self._last_spectrum_keV = None
            self._last_spectrum_counts = None
            self.table_peaks.setRowCount(0)

            if 'pulse-height' in tally.scores:
                raw_values = tally.get_slice(scores=['pulse-height']).mean.flatten()

                # تطبيق GEB
                broadened_values = self.apply_geb_broadening(
                    midpoints_eV, raw_values, a=a_val, b=b_val, c=c_val
                )

                # حماية القيم الصفرية لتجنب تحذيرات المحور اللوغاريتمي
                broadened_values = np.where(broadened_values <= 0, 1e-20, broadened_values)

                ax1.plot(midpoints_keV, broadened_values, color='purple', linewidth=1.2, label='Pulse-Height + GEB')
                ax1.set_yscale('log')  # تفعيل المقياس اللوغاريتمي بأمان
                ax1.set_ylabel('Broadened Counts / Bin [per source particle]', color='purple', fontweight='bold')
                ax1.set_title(f'HPGe Detector Spectrum - Tally {tally_id} (GEB Applied)')
                ax1.grid(True, which="both", linestyle='--', alpha=0.6)

                # يُتاح لاستخدام "Find && Fit Peaks" مباشرة على هذا الطيف --
                # غير متاح لفرع heating/flux لأنه ليس طيف قمم فيزيائيًا.
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
            QMessageBox.critical(self, "Plot Error", f"Failed to generate graph: {str(e)}")

    def _fit_peaks(self, energies_keV, counts, geb_abc, prominence_frac=0.5, window_factor=5.0):
        """Find local peaks in `counts` and fit each with an independent
        Gaussian-plus-baseline model in a window sized from the DETECTOR'S
        OWN expected resolution at that energy. Returns a list of dicts
        with centroid/fwhm/area, in ascending energy order. Fitting
        failures are simply skipped (one bad candidate shouldn't discard
        every other peak).

        Two things that look like small tweaks below are both fixes for
        real failure modes hit while testing this against actual detector
        spectra, not stylistic choices:

        1. Detection runs in LOG10 space with a MULTIPLICATIVE prominence
           threshold, not linear counts relative to the spectrum's global
           maximum. Real gamma spectra span many orders of magnitude
           (that's why they're always plotted log-scale), so a threshold
           expressed as "% of the tallest bin anywhere in the spectrum"
           can completely miss a real, visually obvious photopeak sitting
           on a lower part of the Compton continuum. In log10 space,
           `find_peaks`'s prominence is local by construction, so
           `prominence_frac=0.5` uniformly means "this peak is at least
           1.5x its own local baseline" everywhere in the spectrum,
           regardless of where that baseline sits or what units the tally
           values are in (raw counts, or a per-source-particle mean as
           small as 1e-6 -- both occur in practice).

        2. Each candidate's fit window is sized from `_expected_sigma_keV`
           -- the SAME GEB formula that broadened the spectrum in the
           first place -- rather than an empirical "walk outward from the
           peak until the count rate drops below half-max" heuristic. On
           a monotonically-decaying continuum (present in every real
           spectrum, and especially right at the low-energy end) that
           walk can travel arbitrarily far before ever dropping below
           half of the starting value, producing a wildly oversized
           window and a degenerate fit across a huge span. Since the true
           post-broadening width at any energy is already known exactly
           from (a, b, c), there's no need to guess it. curve_fit's
           `bounds` (sigma constrained to roughly that expected width,
           mu constrained to the fit window) keep individual fits from
           diverging without needing a separate statistical model of
           what counts as "noise" -- which, tried here, turned out to
           silently reject real narrow peaks whenever a bin's own energy
           width happened to be coarser than the detector's actual
           resolution at that energy (an easy thing for a user to hit
           just from how many energy bins they picked).

        `prominence_frac` is the one knob exposed in the UI -- raise it if
        real peaks come with too many noise-driven false positives, lower
        it if real (usually far more prominent) peaks are still missing.
        """
        a, b, c = geb_abc
        energies = np.asarray(energies_keV, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if counts.size < 5 or counts.max() <= 0:
            return []

        log_counts = np.log10(np.clip(counts, 1e-300, None))
        prominence_db = np.log10(1.0 + max(prominence_frac, 1e-6))
        # Minimum bin separation between accepted peaks -- cheap guard
        # against counting several adjacent noisy bins around one real
        # feature as separate peaks.
        peak_idx, _ = find_peaks(log_counts, prominence=prominence_db, distance=5)

        d = float(np.median(np.diff(energies))) if energies.size > 1 else 1.0
        n = counts.size
        results = []
        for idx in peak_idx:
            # Deliberately NOT floored at `d` -- a real peak's physical
            # resolution can legitimately be narrower than one energy
            # bin (e.g. fine GEB parameters with a coarse bin count),
            # and forcing sigma0 up to a full bin width in that case
            # oversizes the window and biases the fit.
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
                area = amplitude * sigma * np.sqrt(2.0 * np.pi)
                results.append(dict(energy=mu, fwhm=fwhm, area=area,
                                     amplitude=amplitude, sigma=sigma, baseline=baseline))
            except Exception:
                continue

        results.sort(key=lambda p: p['energy'])
        return results

    def find_and_fit_peaks(self):
        if self._last_spectrum_keV is None or self._last_spectrum_counts is None:
            QMessageBox.warning(
                self, "Warning",
                "Plot a Pulse-Height spectrum first (click 'Plot Spectrum + GEB' on a tally with a "
                "pulse-height score) -- peak fitting needs a detector spectrum to search."
            )
            return

        if self._current_fig is None or not plt.fignum_exists(self._current_fig.number):
            QMessageBox.warning(
                self, "Warning",
                "The spectrum window was closed -- click 'Plot Spectrum + GEB' again first."
            )
            return

        try:
            prominence_frac = float(self.input_prominence.text()) / 100.0
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Min Prominence must be a number (percent).")
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
        # Clear any peak markers from a previous fit (identified by the
        # '_peak_fit' gid tag) before drawing the current set, so repeated
        # "Find && Fit Peaks" clicks don't stack duplicate overlays.
        for artist in list(ax1.lines) + list(ax1.texts):
            if artist.get_gid() == '_peak_fit':
                artist.remove()

        for pk in peaks:
            # Each peak's fitted curve is drawn ONLY across its own local
            # window (+/- 6 sigma), not the full spectrum -- evaluating
            # _gaussian across the entire x-range instead (the previous
            # behavior) draws a near-flat line at that peak's own
            # baseline level all the way across the plot, and with many
            # peaks those overlapping near-flat lines are exactly what
            # turned the whole spectrum into visual noise.
            half_span = max(6.0 * pk['sigma'], 1e-6)
            local_x = np.linspace(pk['energy'] - half_span, pk['energy'] + half_span, 200)
            ax1.axvline(pk['energy'], color='black', linestyle=':', linewidth=0.8, alpha=0.7, gid='_peak_fit')
            fit_curve = _gaussian(local_x, pk['amplitude'], pk['energy'], pk['sigma'], pk['baseline'])
            ax1.plot(local_x, np.where(fit_curve <= 0, 1e-20, fit_curve),
                      color='black', linewidth=1.0, alpha=0.8, gid='_peak_fit')
            ax1.annotate(f"{pk['energy']:.1f} keV\nFWHM {pk['fwhm']:.2f}",
                         xy=(pk['energy'], pk['amplitude'] + pk['baseline']),
                         xytext=(0, 8), textcoords='offset points',
                         fontsize=7, ha='center', color='black', gid='_peak_fit')

        self._current_fig.canvas.draw()
        self._current_fig.show()

        if not peaks:
            QMessageBox.information(self, "No Peaks Found",
                                     "No peaks met the prominence threshold. Try lowering 'Min Prominence (%)'.")

    def _read_reference_source_table(self):
        """Parse the editable Reference Source table into
        {nuclide: (half_life_seconds, reference_activity_kBq)}.
        Raises ValueError with a user-facing message on bad input."""
        ref = {}
        for row in range(self.table_ref_source.rowCount()):
            nuclide_item = self.table_ref_source.item(row, 0)
            if nuclide_item is None or not nuclide_item.text().strip():
                continue
            nuclide = nuclide_item.text().strip()
            try:
                half_life_val = float(self.table_ref_source.item(row, 1).text())
                unit = self.table_ref_source.item(row, 2).text().strip()
                activity = float(self.table_ref_source.item(row, 3).text())
            except (AttributeError, ValueError):
                raise ValueError(f"Row {row + 1} ({nuclide}): half-life/activity must be numbers.")
            ref[nuclide] = (_half_life_seconds(half_life_val, unit), activity)
        return ref

    def identify_nuclides(self):
        """Match each already-fitted peak's centroid to the nearest known
        gamma line (see NUCLIDE_GAMMA_LINES) and fill the Nuclide column.
        Does not require the Reference Source table -- that's only needed
        for the FEPE step, since line energies/intensities are fixed
        nuclear data independent of any specific source's activity."""
        if not self._last_peaks:
            QMessageBox.warning(self, "Warning", "Run 'Find & Fit Peaks' first.")
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
        """For every peak already matched to a nuclide (run 'Identify
        Nuclides' first), compute the Full-Energy Peak Efficiency:

            FEPE = peak_net_area / (source_weight_of_that_nuclide * emission_probability_of_that_line)

        `peak_net_area` is already in units of "counts per source
        particle" (straight from the OpenMC tally mean the spectrum was
        built from). `source_weight_of_that_nuclide` is that nuclide's
        share of the TOTAL source activity -- i.e. what fraction of all
        simulated source particles represent a decay of THIS nuclide --
        computed from the Reference Source table's activities, decay-
        corrected from the Reference Date to the Simulation/Analysis
        Date. This is the standard formula for back-computing per-line,
        per-nuclide efficiency out of a single combined mixed-source run
        (rather than one run per nuclide): dividing out both the
        emission probability AND the nuclide's own share of the mixed
        source recovers "efficiency per photon actually emitted at that
        energy", independent of how the source mixture was simulated.
        """
        if not self._last_peaks:
            QMessageBox.warning(self, "Warning", "Run 'Find & Fit Peaks' first.")
            return
        if not any(pk.get('nuclide_match') for pk in self._last_peaks):
            QMessageBox.warning(self, "Warning", "Run 'Identify Nuclides' first -- FEPE needs identified peaks.")
            return

        try:
            ref = self._read_reference_source_table()
        except ValueError as e:
            QMessageBox.critical(self, "Reference Source Error", str(e))
            return
        if not ref:
            QMessageBox.critical(self, "Reference Source Error", "Reference Source table is empty.")
            return

        ref_date = self.ref_date_edit.date().toPython()
        sim_date = self.sim_date_edit.date().toPython()
        dt_seconds = (sim_date - ref_date).days * 86400.0

        decayed_activity = {nuclide: _decayed_activity(activity, half_life_s, dt_seconds)
                             for nuclide, (half_life_s, activity) in ref.items()}
        total_activity = sum(decayed_activity.values())
        if total_activity <= 0:
            QMessageBox.critical(self, "Reference Source Error", "Total decayed activity is zero or negative.")
            return

        fepe_points = []  # (energy_keV, fepe, nuclide) for the plot
        for row, pk in enumerate(self._last_peaks):
            match = pk.get('nuclide_match')
            fepe_item = QTableWidgetItem("-")
            fepe_item.setFlags(fepe_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if match:
                nuclide, expected_e, intensity = match
                activity_i = decayed_activity.get(nuclide)
                if activity_i is not None and intensity > 0:
                    weight = activity_i / total_activity
                    fepe = pk['area'] / (weight * intensity)
                    pk['fepe'] = fepe
                    fepe_item.setText(f"{fepe:.4g}")
                    fepe_points.append((expected_e, fepe, nuclide))
                else:
                    self._log(f"⚠️ Peak at {pk['energy']:.1f} keV matched to '{nuclide}', "
                               f"but that nuclide isn't in the Reference Source table -- skipped.")
            self.table_peaks.setItem(row, 5, fepe_item)

        if not fepe_points:
            QMessageBox.warning(
                self, "No FEPE Computed",
                "None of the identified peaks' nuclides are in the Reference Source table."
            )
            return

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
