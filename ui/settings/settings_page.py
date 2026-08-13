import os
import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QFormLayout, QMessageBox, QFileDialog, QStackedWidget, QScrollArea, QCheckBox,
    QListWidget
)
from PySide6.QtCore import Signal


def _parse_floats(text, n=None):
    """Parses a comma-separated float list; raises ValueError on bad input
    or a wrong count when n is given. Used to validate every free-typed
    numeric field before it gets baked into generated Python code -- these
    fields used to go straight into the script unvalidated, so a typo only
    surfaced as a cryptic error once the script actually ran."""
    vals = [float(x.strip()) for x in text.split(",")]
    if n is not None and len(vals) != n:
        raise ValueError(f"expected {n} comma-separated numbers, got {len(vals)}")
    return vals


class SettingsPageWidget(QWidget):
    """
    OpenMC Settings, Cross Sections & Advanced Source Manager.
    Supports multiple weighted sources, angular distributions, energy
    cutoffs, convergence triggers, reproducible seeding, and full
    Particle Transport (Neutrons, Photons, Electrons, Positrons).
    """
    script_generated = Signal(str)

    def __init__(self, project_manager=None, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.pending_sources = []  # list of dicts: {code, var, label, strength}
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
        param_layout.addRow("Inactive Batches (eigenvalue only):", self.inactive_field)

        self.gen_per_batch_field = QLineEdit("1")
        self.gen_per_batch_field.setToolTip("eigenvalue mode only: settings.generations_per_batch")
        param_layout.addRow("Generations per Batch (eigenvalue only):", self.gen_per_batch_field)

        self.seed_field = QLineEdit("")
        self.seed_field.setPlaceholderText("(optional) integer RNG seed for reproducible runs")
        param_layout.addRow("Random Seed (optional):", self.seed_field)

        # --- واجهة تتبع الجسيمات (Tracking) ---
        # ملاحظة: تتبع الجسيمات (settings.track) له صفحة مخصصة (Tracks) تبني
        # وتعرض المسارات فعلياً؛ لا يوجد هنا تحكم مكرر لتفادي تعارض قيمتين
        # مختلفتين لـ settings.track تُكتبان في نفس السكريبت (الثانية كانت
        # تكتب فوق الأولى بصمت دون تفسير للمستخدم لماذا لا يعمل تغييره).
        track_note = QLabel(
            "Particle tracking (settings.track) is configured on the dedicated 'Tracks' page, "
            "which also renders the tracked paths visually.")
        track_note.setWordWrap(True)
        track_note.setStyleSheet("color: #666666; font-style: italic;")
        param_layout.addRow("Visual Tracking:", track_note)

        # --- واجهة الجسيمات الإضافية (Transport Modes) ---
        particles_layout = QHBoxLayout()
        self.chk_photon = QCheckBox("Photons Transport")
        self.chk_electron = QCheckBox("Detailed Electron/Positron Treatment (TTB)")
        self.chk_electron.setToolTip(
            "OpenMC has no standalone electron-transport switch: electrons/positrons are already "
            "handled whenever Photon Transport is on and the data library includes electron data. "
            "This checkbox only selects the 'ttb' (thick-target bremsstrahlung) treatment instead of "
            "the default; it also turns on Photon Transport since TTB requires it.")
        self.chk_photon.setStyleSheet("font-weight: bold; color: #8e44ad;")
        self.chk_electron.setStyleSheet("font-weight: bold; color: #2980b9;")
        particles_layout.addWidget(self.chk_photon)
        particles_layout.addWidget(self.chk_electron)
        param_layout.addRow("Active Particles:", particles_layout)

        layout.addWidget(param_group)

        # ==========================================
        # --- 3. تصميم المصدر (Advanced Source, يدعم مصادر متعددة موزونة) ---
        # ==========================================
        self.source_group = QGroupBox(
            "3. Source Definition -- configure below, click 'Add Source' (repeat for multiple weighted sources)")
        source_layout = QFormLayout(self.source_group)

        self.particle_combo = QComboBox()
        self.particle_combo.addItems(["neutron", "photon", "electron", "positron"])
        source_layout.addRow("Particle Type:", self.particle_combo)

        self.strength_field = QLineEdit("1.0")
        self.strength_field.setToolTip("Relative statistical weight of this source among all added sources.")
        source_layout.addRow("Strength (relative weight):", self.strength_field)

        self.spatial_combo = QComboBox()
        self.spatial_combo.addItems(["Point Source", "Box (Volumetric)", "Mesh (uniform over a RegularMesh)"])
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

        mesh_src_widget = QWidget()
        mesh_src_form = QFormLayout(mesh_src_widget)
        mesh_src_form.setContentsMargins(0, 0, 0, 0)
        self.src_mesh_dim = QLineEdit("5, 5, 5")
        self.src_mesh_ll = QLineEdit("-5.0, -5.0, -5.0")
        self.src_mesh_ur = QLineEdit("5.0, 5.0, 5.0")
        mesh_src_form.addRow("Dimension (Nx, Ny, Nz):", self.src_mesh_dim)
        mesh_src_form.addRow("Lower Left (x, y, z):", self.src_mesh_ll)
        mesh_src_form.addRow("Upper Right (x, y, z):", self.src_mesh_ur)
        self.spatial_stack.addWidget(mesh_src_widget)

        source_layout.addRow(self.spatial_stack)

        self.angle_combo = QComboBox()
        self.angle_combo.addItems(["Isotropic (default)", "Monodirectional"])
        self.angle_combo.currentTextChanged.connect(self._toggle_angle_stack)
        source_layout.addRow("Angular Distribution:", self.angle_combo)

        self.angle_stack = QStackedWidget()
        self.angle_stack.addWidget(QLabel("Isotropic emission (no source.angle set)."))
        mono_widget = QWidget()
        mono_layout = QHBoxLayout(mono_widget)
        mono_layout.setContentsMargins(0, 0, 0, 0)
        self.mono_uvw = QLineEdit("0.0, 0.0, 1.0")
        mono_layout.addWidget(QLabel("Reference direction (u, v, w):"))
        mono_layout.addWidget(self.mono_uvw)
        self.angle_stack.addWidget(mono_widget)
        source_layout.addRow(self.angle_stack)

        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["Discrete (Monoenergetic)", "Watt Spectrum (Fission)", "Maxwell (Fission-like)"])
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

        maxwell_widget = QWidget()
        maxwell_layout = QHBoxLayout(maxwell_widget)
        maxwell_layout.setContentsMargins(0, 0, 0, 0)
        self.maxwell_theta = QLineEdit("1.29e6")
        maxwell_layout.addWidget(QLabel("theta (eV):"))
        maxwell_layout.addWidget(self.maxwell_theta)
        self.energy_stack.addWidget(maxwell_widget)

        source_layout.addRow(self.energy_stack)

        src_btn_layout = QHBoxLayout()
        btn_add_source = QPushButton("➕ Add Source")
        btn_add_source.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        btn_add_source.clicked.connect(self.add_source_action)
        btn_remove_source = QPushButton("🗑️ Remove Selected Source")
        btn_remove_source.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        btn_remove_source.clicked.connect(self.remove_source_action)
        src_btn_layout.addWidget(btn_add_source)
        src_btn_layout.addWidget(btn_remove_source)
        source_layout.addRow(src_btn_layout)

        source_layout.addRow(QLabel("Sources queued (settings.source will be this list):"))
        self.pending_sources_list = QListWidget()
        self.pending_sources_list.setFixedHeight(80)
        source_layout.addRow(self.pending_sources_list)

        layout.addWidget(self.source_group)

        # ==========================================
        # --- 4. التحكم المتقدم بالتشغيل (Cutoff / Trigger) ---
        # ==========================================
        adv_group = QGroupBox("4. Advanced Run Control (Cutoffs & Convergence Trigger)")
        adv_layout = QFormLayout(adv_group)

        self.chk_cutoff = QCheckBox("Enable Cutoffs")
        adv_layout.addRow(self.chk_cutoff)

        self.cutoff_particle_combo = QComboBox()
        self.cutoff_particle_combo.addItems(["neutron", "photon", "electron", "positron"])
        adv_layout.addRow("Energy Cutoff -- Particle:", self.cutoff_particle_combo)
        self.cutoff_energy_field = QLineEdit("")
        self.cutoff_energy_field.setPlaceholderText("(optional) minimum tracked energy in eV")
        adv_layout.addRow("Energy Cutoff (eV):", self.cutoff_energy_field)
        self.cutoff_weight_field = QLineEdit("")
        self.cutoff_weight_field.setPlaceholderText("(optional) weight cutoff for Russian roulette")
        adv_layout.addRow("Weight Cutoff:", self.cutoff_weight_field)
        self.cutoff_weight_avg_field = QLineEdit("")
        self.cutoff_weight_avg_field.setPlaceholderText("(optional) survival weight after Russian roulette")
        adv_layout.addRow("Weight-Avg Cutoff:", self.cutoff_weight_avg_field)

        self.chk_trigger = QCheckBox("Enable Convergence Trigger (extend batches until tallies converge)")
        adv_layout.addRow(self.chk_trigger)
        self.trigger_max_batches_field = QLineEdit("10000")
        adv_layout.addRow("Trigger Max Batches:", self.trigger_max_batches_field)
        self.trigger_batch_interval_field = QLineEdit("10")
        adv_layout.addRow("Trigger Batch Interval:", self.trigger_batch_interval_field)

        layout.addWidget(adv_group)

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
        elif text.startswith("Mesh"):
            self.spatial_stack.setCurrentIndex(2)

    def _toggle_angle_stack(self, text):
        if text == "Isotropic (default)":
            self.angle_stack.setCurrentIndex(0)
        elif text == "Monodirectional":
            self.angle_stack.setCurrentIndex(1)

    def _toggle_energy_stack(self, text):
        if text == "Discrete (Monoenergetic)":
            self.energy_stack.setCurrentIndex(0)
        elif text == "Watt Spectrum (Fission)":
            self.energy_stack.setCurrentIndex(1)
        elif text == "Maxwell (Fission-like)":
            self.energy_stack.setCurrentIndex(2)

    def browse_cross_sections(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select cross_sections.xml", "",
                                                  "XML Files (*.xml);;All Files (*)")
        if filepath:
            self.xs_path_field.setText(filepath)

    # ==========================================
    # --- Source queue (multiple weighted sources) ---
    # ==========================================
    def add_source_action(self):
        try:
            strength = float(self.strength_field.text().strip())
            if strength <= 0:
                raise ValueError("Strength must be a positive number.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid Strength: {e}")
            return

        particle = self.particle_combo.currentText()
        spatial = self.spatial_combo.currentText()
        angle = self.angle_combo.currentText()
        energy = self.energy_combo.currentText()

        is_mesh_source = spatial.startswith("Mesh")

        idx = len(self.pending_sources) + 1
        var = f"src{idx}"
        # A Mesh Source needs the actual settings.source entry to be an
        # openmc.MeshSource wrapping (mesh, template_source) -- the
        # particle/angle/energy attributes below are set on the TEMPLATE
        # source object, then that template is wrapped, so `var` itself
        # only becomes the final openmc.MeshSource at the end.
        base_var = f"{var}_tpl" if is_mesh_source else var
        code = "try:\n"
        code += f"    {base_var} = openmc.IndependentSource()\n"
        code += "except AttributeError:\n"
        code += f"    {base_var} = openmc.Source()\n"
        code += f"{base_var}.particle = '{particle}'\n"
        code += f"{base_var}.strength = {strength}\n"

        try:
            if spatial == "Point Source":
                coords = _parse_floats(self.pt_coords.text(), 3)
                code += f"{base_var}.space = openmc.stats.Point(({', '.join(str(v) for v in coords)}))\n"
            elif spatial == "Box (Volumetric)":
                ll = _parse_floats(self.box_ll.text(), 3)
                ur = _parse_floats(self.box_ur.text(), 3)
                code += (f"{base_var}.space = openmc.stats.Box(({', '.join(str(v) for v in ll)}), "
                          f"({', '.join(str(v) for v in ur)}), only_fissionable=False)\n")
            else:
                dim = _parse_floats(self.src_mesh_dim.text(), 3)
                mll = _parse_floats(self.src_mesh_ll.text(), 3)
                mur = _parse_floats(self.src_mesh_ur.text(), 3)
                mesh_var = f"{var}_mesh"
                code += f"{mesh_var} = openmc.RegularMesh()\n"
                code += f"{mesh_var}.dimension = ({', '.join(str(int(v)) for v in dim)})\n"
                code += f"{mesh_var}.lower_left = ({', '.join(str(v) for v in mll)})\n"
                code += f"{mesh_var}.upper_right = ({', '.join(str(v) for v in mur)})\n"

            if angle == "Monodirectional":
                uvw = _parse_floats(self.mono_uvw.text(), 3)
                if all(v == 0.0 for v in uvw):
                    raise ValueError("Reference direction (u, v, w) cannot be (0, 0, 0).")
                code += f"{base_var}.angle = openmc.stats.Monodirectional(reference_uvw=({', '.join(str(v) for v in uvw)}))\n"

            if energy == "Discrete (Monoenergetic)":
                e_val = _parse_floats(self.disc_energy.text(), 1)[0]
                if e_val <= 0:
                    raise ValueError("Energy must be a positive number.")
                code += f"{base_var}.energy = openmc.stats.Discrete([{e_val}], [1.0])\n"
            elif energy == "Watt Spectrum (Fission)":
                a_val, b_val = _parse_floats(self.watt_a.text(), 1)[0], _parse_floats(self.watt_b.text(), 1)[0]
                if a_val <= 0 or b_val <= 0:
                    raise ValueError("Watt 'a' and 'b' must both be positive numbers.")
                code += f"{base_var}.energy = openmc.stats.Watt(a={a_val}, b={b_val})\n"
            else:
                theta = _parse_floats(self.maxwell_theta.text(), 1)[0]
                if theta <= 0:
                    raise ValueError("Maxwell 'theta' must be a positive number.")
                code += f"{base_var}.energy = openmc.stats.Maxwell(theta={theta})\n"

            if is_mesh_source:
                code += f"{var} = openmc.MeshSource({var}_mesh, {base_var})\n"
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid source parameter: {e}")
            return

        label = f"{particle} | strength={strength} | {spatial} | {angle} | {energy}"
        self.pending_sources.append(
            {"code": code, "var": var, "label": label, "strength": strength, "particle": particle})
        self.pending_sources_list.addItem(label)

    def remove_source_action(self):
        row = self.pending_sources_list.currentRow()
        if row < 0:
            return
        self.pending_sources_list.takeItem(row)
        self.pending_sources.pop(row)

    def _clear_pending_sources(self):
        self.pending_sources = []
        self.pending_sources_list.clear()

    def generate_settings_script(self):
        try:
            batches = int(self.batches_field.text())
            particles = int(self.particles_field.text())
            inactive = int(self.inactive_field.text())
            gen_per_batch = int(self.gen_per_batch_field.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error",
                                 "Batches, Particles, Inactive Batches, and Generations per Batch "
                                 "must all be valid integers.")
            return

        if batches <= 0 or particles <= 0:
            QMessageBox.warning(self, "Input Error", "Total Batches and Particles per Batch must be > 0.")
            return
        if inactive < 0:
            QMessageBox.warning(self, "Input Error", "Inactive Batches cannot be negative.")
            return
        if gen_per_batch <= 0:
            QMessageBox.warning(self, "Input Error", "Generations per Batch must be > 0.")
            return

        run_mode = self.run_mode_combo.currentText()

        if run_mode == "eigenvalue" and inactive >= batches:
            QMessageBox.warning(self, "Input Error",
                                 "Inactive Batches must be smaller than Total Batches "
                                 "(there must be at least one active batch).")
            return

        seed_txt = self.seed_field.text().strip()
        seed = None
        if seed_txt:
            try:
                seed = int(seed_txt)
                if seed <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Random Seed must be a positive integer.")
                return

        if run_mode == "fixed source" and not self.pending_sources:
            QMessageBox.warning(self, "Input Error",
                                 "A fixed source run requires at least one Source -- configure one above and "
                                 "click 'Add Source' first.")
            return

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
            if gen_per_batch != 1:
                py_code += f"settings.generations_per_batch = {gen_per_batch}\n"

        if seed is not None:
            py_code += f"settings.seed = {seed}\n"

        # المتغيرات الخاصة بأنواع الجسيمات
        enable_photon = self.chk_photon.isChecked()
        enable_electron = self.chk_electron.isChecked()

        if run_mode == "fixed source":
            py_code += "\n# --- Source Definition (may combine multiple weighted sources) ---\n"
            for src in self.pending_sources:
                py_code += src["code"]
                if src["particle"] == "photon":
                    enable_photon = True
                elif src["particle"] in ("electron", "positron"):
                    enable_photon = True
                    enable_electron = True

            src_vars = ", ".join(s["var"] for s in self.pending_sources)
            py_code += f"settings.source = [{src_vars}]\n"

        # كتابة إعدادات نقل الجسيمات
        if enable_photon or enable_electron:
            py_code += "\n# --- Transport Modes ---\n"
            py_code += "settings.photon_transport = True\n"
        if enable_electron:
            py_code += "settings.electron_treatment = 'ttb'  # Detailed (TTB) electron/positron treatment\n"

        # --- Cutoffs ---
        if self.chk_cutoff.isChecked():
            cutoff_entries = []
            e_txt = self.cutoff_energy_field.text().strip()
            if e_txt:
                try:
                    e_val = float(e_txt)
                except ValueError:
                    QMessageBox.warning(self, "Input Error", f"Invalid Energy Cutoff value '{e_txt}'.")
                    return
                cutoff_entries.append(f"'energy_{self.cutoff_particle_combo.currentText()}': {e_val}")
            w_txt = self.cutoff_weight_field.text().strip()
            if w_txt:
                try:
                    w_val = float(w_txt)
                except ValueError:
                    QMessageBox.warning(self, "Input Error", f"Invalid Weight Cutoff value '{w_txt}'.")
                    return
                cutoff_entries.append(f"'weight': {w_val}")
            wa_txt = self.cutoff_weight_avg_field.text().strip()
            if wa_txt:
                try:
                    wa_val = float(wa_txt)
                except ValueError:
                    QMessageBox.warning(self, "Input Error", f"Invalid Weight-Avg Cutoff value '{wa_txt}'.")
                    return
                cutoff_entries.append(f"'weight_avg': {wa_val}")

            if cutoff_entries:
                py_code += "\n# --- Cutoffs ---\n"
                py_code += "settings.cutoff = {" + ", ".join(cutoff_entries) + "}\n"

        # --- Trigger ---
        if self.chk_trigger.isChecked():
            try:
                trigger_max = int(self.trigger_max_batches_field.text())
                trigger_interval = int(self.trigger_batch_interval_field.text())
                if trigger_max <= 0 or trigger_interval <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Input Error",
                                     "Trigger Max Batches and Trigger Batch Interval must be positive integers.")
                return
            py_code += "\n# --- Convergence Trigger ---\n"
            py_code += "settings.trigger_active = True\n"
            py_code += f"settings.trigger_max_batches = {trigger_max}\n"
            py_code += f"settings.trigger_batch_interval = {trigger_interval}\n"

        self.script_generated.emit(py_code)
        QMessageBox.information(self, "Settings Applied",
                                "Settings applied successfully! Check the Python Script Editor.")
        self._clear_pending_sources()
