import os
import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QFormLayout, QMessageBox, QFileDialog, QStackedWidget, QScrollArea, QCheckBox
)
from PySide6.QtCore import Signal


class SettingsPageWidget(QWidget):
    """
    OpenMC Settings, Cross Sections & Advanced Source Manager.
    Includes automated UI for Particle Tracking and full Particle Transport (Neutrons, Photons, Electrons, Positrons).
    """
    script_generated = Signal(str)

    def __init__(self, project_manager=None, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # ==========================================
        # --- 1. مدير المكتبات (Cross Sections) ---
        # ==========================================
        xs_group = QGroupBox("1. Nuclear Data (Cross Sections Manager)")
        xs_layout = QHBoxLayout(xs_group)

        self.xs_path_field = QLineEdit()
        default_xs = os.environ.get('OPENMC_CROSS_SECTIONS', '')
        self.xs_path_field.setText(default_xs)
        self.xs_path_field.setPlaceholderText("Select cross_sections.xml path...")

        btn_browse_xs = QPushButton("📂 Browse...")
        btn_browse_xs.setStyleSheet("background-color: #4a4a4a; color: white;")
        btn_browse_xs.clicked.connect(self.browse_cross_sections)

        xs_layout.addWidget(QLabel("Library Path:"))
        xs_layout.addWidget(self.xs_path_field)
        xs_layout.addWidget(btn_browse_xs)
        layout.addWidget(xs_group)

        # ==========================================
        # --- 2. إعدادات المحاكاة والتتبع (Tracking) ---
        # ==========================================
        param_group = QGroupBox("2. Simulation Parameters & Tracking")
        param_layout = QFormLayout(param_group)

        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItems(["fixed source", "eigenvalue"])
        self.run_mode_combo.currentTextChanged.connect(self._toggle_source_panel)
        param_layout.addRow("Run Mode:", self.run_mode_combo)

        self.batches_field = QLineEdit("100")
        param_layout.addRow("Total Batches:", self.batches_field)

        self.particles_field = QLineEdit("10000")
        param_layout.addRow("Particles per Batch:", self.particles_field)

        self.inactive_field = QLineEdit("20")
        param_layout.addRow("Inactive Batches:", self.inactive_field)

        # --- واجهة تتبع الجسيمات (Tracking) ---
        track_layout = QHBoxLayout()
        self.chk_track = QCheckBox("Enable Particle Tracking")
        self.chk_track.setStyleSheet("font-weight: bold; color: #d97706;")
        self.chk_track.stateChanged.connect(self._toggle_track_field)

        self.track_count_field = QLineEdit("50")
        self.track_count_field.setEnabled(False)  # معطل حتى يتم تفعيل المربع
        self.track_count_field.setToolTip("Number of particles to track (Warning: High numbers create many files!)")

        track_layout.addWidget(self.chk_track)
        track_layout.addWidget(QLabel("How many particles?"))
        track_layout.addWidget(self.track_count_field)
        param_layout.addRow("Visual Tracking:", track_layout)

        # --- واجهة الجسيمات الإضافية (Transport Modes) ---
        particles_layout = QHBoxLayout()
        self.chk_photon = QCheckBox("Photons Transport")
        self.chk_electron = QCheckBox("Electrons/Positrons Transport")
        self.chk_photon.setStyleSheet("font-weight: bold; color: #8e44ad;")
        self.chk_electron.setStyleSheet("font-weight: bold; color: #2980b9;")
        particles_layout.addWidget(self.chk_photon)
        particles_layout.addWidget(self.chk_electron)
        param_layout.addRow("Active Particles:", particles_layout)

        layout.addWidget(param_group)

        # ==========================================
        # --- 3. تصميم المصدر (Advanced Source) ---
        # ==========================================
        self.source_group = QGroupBox("3. Advanced Source Definition")
        source_layout = QFormLayout(self.source_group)

        self.particle_combo = QComboBox()
        # إضافة كافة الجسيمات المدعومة
        self.particle_combo.addItems(["neutron", "photon", "electron", "positron"])
        source_layout.addRow("Particle Type:", self.particle_combo)

        self.spatial_combo = QComboBox()
        self.spatial_combo.addItems(["Point Source", "Box (Volumetric)"])
        self.spatial_combo.currentTextChanged.connect(self._toggle_spatial_stack)
        source_layout.addRow("Spatial Distribution:", self.spatial_combo)

        self.spatial_stack = QStackedWidget()

        point_widget = QWidget()
        pt_layout = QHBoxLayout(point_widget)
        pt_layout.setContentsMargins(0, 0, 0, 0)
        self.pt_coords = QLineEdit("0.0, 0.0, 0.0")
        pt_layout.addWidget(QLabel("Coordinates (x, y, z):"))
        pt_layout.addWidget(self.pt_coords)
        self.spatial_stack.addWidget(point_widget)

        box_widget = QWidget()
        box_layout = QHBoxLayout(box_widget)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self.box_ll = QLineEdit("-5.0, -5.0, -5.0")
        self.box_ur = QLineEdit("5.0, 5.0, 5.0")
        box_layout.addWidget(QLabel("Lower Left:"))
        box_layout.addWidget(self.box_ll)
        box_layout.addWidget(QLabel("Upper Right:"))
        box_layout.addWidget(self.box_ur)
        self.spatial_stack.addWidget(box_widget)

        source_layout.addRow(self.spatial_stack)

        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["Discrete (Monoenergetic)", "Watt Spectrum (Fission)"])
        self.energy_combo.currentTextChanged.connect(self._toggle_energy_stack)
        source_layout.addRow("Energy Distribution:", self.energy_combo)

        self.energy_stack = QStackedWidget()

        disc_widget = QWidget()
        disc_layout = QHBoxLayout(disc_widget)
        disc_layout.setContentsMargins(0, 0, 0, 0)
        self.disc_energy = QLineEdit("2000000.0")
        disc_layout.addWidget(QLabel("Energy (eV):"))
        disc_layout.addWidget(self.disc_energy)
        self.energy_stack.addWidget(disc_widget)

        watt_widget = QWidget()
        watt_layout = QHBoxLayout(watt_widget)
        watt_layout.setContentsMargins(0, 0, 0, 0)
        self.watt_a = QLineEdit("0.988e6")
        self.watt_b = QLineEdit("2.249e-6")
        watt_layout.addWidget(QLabel("a (eV):"))
        watt_layout.addWidget(self.watt_a)
        watt_layout.addWidget(QLabel("b (1/eV):"))
        watt_layout.addWidget(self.watt_b)
        self.energy_stack.addWidget(watt_widget)

        source_layout.addRow(self.energy_stack)
        layout.addWidget(self.source_group)

        # ==========================================
        # --- زر التوليد (Generate Button) ---
        # ==========================================
        btn_apply = QPushButton("Apply Settings & Generate Script")
        btn_apply.setStyleSheet(
            "background-color: #0e639c; color: white; padding: 10px; font-weight: bold; font-size: 14px;")
        btn_apply.clicked.connect(self.generate_settings_script)
        layout.addWidget(btn_apply)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def _toggle_source_panel(self, mode):
        self.source_group.setVisible(mode == "fixed source")

    def _toggle_spatial_stack(self, text):
        if text == "Point Source":
            self.spatial_stack.setCurrentIndex(0)
        elif text == "Box (Volumetric)":
            self.spatial_stack.setCurrentIndex(1)

    def _toggle_energy_stack(self, text):
        if text == "Discrete (Monoenergetic)":
            self.energy_stack.setCurrentIndex(0)
        elif text == "Watt Spectrum (Fission)":
            self.energy_stack.setCurrentIndex(1)

    def _toggle_track_field(self, state):
        self.track_count_field.setEnabled(state == 2)

    def browse_cross_sections(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select cross_sections.xml", "",
                                                  "XML Files (*.xml);;All Files (*)")
        if filepath:
            self.xs_path_field.setText(filepath)

    def generate_settings_script(self):
        try:
            batches = int(self.batches_field.text())
            particles = int(self.particles_field.text())
            inactive = int(self.inactive_field.text())
            track_count = int(self.track_count_field.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Batches, Particles, and Tracking count must be valid integers.")
            return

        run_mode = self.run_mode_combo.currentText()
        xs_path = self.xs_path_field.text().strip()

        py_code = "\n# ==========================================\n"
        py_code += "# --- Settings & Cross Sections ---\n"
        py_code += "# ==========================================\n"

        if xs_path:
            xs_path_clean = xs_path.replace('\\', '/')
            py_code += f"openmc.config['cross_sections'] = r'{xs_path_clean}'\n\n"

        py_code += "settings = openmc.Settings()\n"
        py_code += f"settings.run_mode = '{run_mode}'\n"
        py_code += f"settings.batches = {batches}\n"
        py_code += f"settings.particles = {particles}\n"

        if run_mode == "eigenvalue":
            py_code += f"settings.inactive = {inactive}\n"

        # إضافة كود التتبع
        if self.chk_track.isChecked():
            py_code += f"\n# --- Particle Tracking Setup ---\n"
            py_code += f"settings.track = list(range(1, {track_count} + 1))\n"

        # المتغيرات الخاصة بأنواع الجسيمات
        enable_photon = self.chk_photon.isChecked()
        enable_electron = self.chk_electron.isChecked()

        if run_mode == "fixed source":
            py_code += "\n# --- Advanced Source Definition ---\n"
            py_code += "try:\n"
            py_code += "    source = openmc.IndependentSource()\n"
            py_code += "except AttributeError:\n"
            py_code += "    source = openmc.Source()\n\n"

            particle = self.particle_combo.currentText()
            py_code += f"source.particle = '{particle}'\n"

            # تفعيل النقل ضمناً بناءً على المصدر
            if particle == "photon":
                enable_photon = True
            elif particle in ["electron", "positron"]:
                enable_photon = True
                enable_electron = True

            spatial = self.spatial_combo.currentText()
            if spatial == "Point Source":
                coords = self.pt_coords.text()
                py_code += f"source.space = openmc.stats.Point(({coords}))\n"
            elif spatial == "Box (Volumetric)":
                ll = self.box_ll.text()
                ur = self.box_ur.text()
                py_code += f"spatial_dist = openmc.stats.Box(({ll}), ({ur}), only_fissionable=False)\n"
                py_code += "source.space = spatial_dist\n"

            energy = self.energy_combo.currentText()
            if energy == "Discrete (Monoenergetic)":
                e_val = self.disc_energy.text()
                py_code += f"source.energy = openmc.stats.Discrete([{e_val}], [1.0])\n"
            elif energy == "Watt Spectrum (Fission)":
                a_val = self.watt_a.text()
                b_val = self.watt_b.text()
                py_code += f"source.energy = openmc.stats.Watt(a={a_val}, b={b_val})\n"

            py_code += "settings.source = source\n"

        # كتابة إعدادات نقل الجسيمات
        if enable_photon or enable_electron:
            py_code += "\n# --- Transport Modes ---\n"
            py_code += "settings.photon_transport = True\n"
        if enable_electron:
            py_code += "settings.electron_treatment = 'ttb'  # Enables Electron/Positron transport\n"

        self.script_generated.emit(py_code)
        QMessageBox.information(self, "Settings Applied",
                                "Settings applied successfully! Check the Python Script Editor.")