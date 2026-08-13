import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QComboBox, QFormLayout, QListWidget, QMessageBox, QCheckBox
)
from PySide6.QtCore import Signal


def _normalize_nuclide_symbol(raw):
    """Normalize element/nuclide symbol casing the way OpenMC expects it
    (element part capitalized, e.g. 'fe'->'Fe', 'u235'->'U235',
    'am242m'->'Am242m') -- the mass-number/metastable-state suffix after
    the letters is left untouched. Without this, a lowercase element
    symbol (a very easy typo) is passed straight to add_element() and
    only fails once the generated script actually runs in OpenMC."""
    raw = raw.strip()
    i = 0
    while i < len(raw) and raw[i].isalpha():
        i += 1
    element_part, rest = raw[:i], raw[i:]
    return element_part.capitalize() + rest


class MaterialsPageWidget(QWidget):
    """
    OpenMC Materials Builder Page.
    Includes manual building (with enrichment, S(alpha,beta), temperature/
    volume/depletable), a built-in standard materials library, and
    openmc.Material.mix_materials() support.
    """
    script_generated = Signal(str)

    def __init__(self, project_manager=None, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.current_material_nuclides = []
        self.current_material_sab = []
        # Tracks every Python variable name already emitted into the
        # script this session, so adding the same library material (or a
        # custom material with a colliding name) twice doesn't silently
        # generate two `X = openmc.Material(...)` statements that shadow
        # each other and leave an orphaned Material instance registered.
        self._used_material_names = set()
        self._init_ui()

    def _dedupe_material_name(self, base_name):
        """Return base_name, or base_name with a numeric suffix appended
        if it's already been used this session (used for library
        materials, where the name is auto-derived, not user-typed)."""
        name = base_name
        n = 2
        while name in self._used_material_names:
            name = f"{base_name}_{n}"
            n += 1
        self._used_material_names.add(name)
        return name

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # ==================================================
        # --- الجانب الأيسر: مكتبة المواد الجاهزة + بناء مادة جديدة ---
        # ==================================================
        left_layout = QVBoxLayout()

        # 1. مكتبة المواد الجاهزة (Built-in Library)
        lib_group = QGroupBox("1. Built-in Materials Library")
        lib_layout = QFormLayout(lib_group)

        self.lib_combo = QComboBox()
        # إضافة قائمة بالمواد الأكثر استخداماً في المفاعلات والتدريع
        self.lib_combo.addItems([
            "Select a material...",
            "Water (H2O)",
            "Heavy Water (D2O)",
            "B4C (Boron Carbide)",
            "Lead (Pb)",
            "Stainless Steel 304",
            "Stainless Steel 316",
            "Concrete (Standard)",
            "Polyethylene (PE)",
            "Borated Polyethylene (5%)",
            "Uranium Dioxide (UO2 - 3% Enriched)",
            "HPGe (High-Purity Germanium)",
            "NaI (Sodium Iodide)"
        ])
        lib_layout.addRow("Material:", self.lib_combo)

        btn_add_lib = QPushButton("Add Library Material to Script")
        btn_add_lib.setStyleSheet("background-color: #217346; color: white; font-weight: bold;")
        btn_add_lib.clicked.connect(self.add_library_material)
        lib_layout.addRow(btn_add_lib)

        left_layout.addWidget(lib_group)

        # 2. بناء مادة مخصصة (Custom Material Builder)
        build_group = QGroupBox("2. Custom Material Builder")
        build_layout = QFormLayout(build_group)

        self.mat_name_field = QLineEdit()
        build_layout.addRow("Material Name:", self.mat_name_field)

        self.density_val_field = QLineEdit()
        self.density_unit_combo = QComboBox()
        self.density_unit_combo.addItems(['g/cm3', 'g/cc', 'kg/m3', 'atom/b-cm'])
        density_layout = QHBoxLayout()
        density_layout.addWidget(self.density_val_field)
        density_layout.addWidget(self.density_unit_combo)
        build_layout.addRow("Density:", density_layout)

        self.temperature_field = QLineEdit()
        self.temperature_field.setPlaceholderText("(optional) Kelvin")
        build_layout.addRow("Temperature:", self.temperature_field)

        self.volume_field = QLineEdit()
        self.volume_field.setPlaceholderText("(optional) cm^3 -- needed for depletion")
        build_layout.addRow("Volume:", self.volume_field)

        self.chk_depletable = QCheckBox("Depletable (for burnup/depletion calculations)")
        build_layout.addRow(self.chk_depletable)

        self.nuclide_field = QLineEdit()
        self.nuclide_field.setPlaceholderText("e.g., U235, or an element symbol like U")
        self.fraction_field = QLineEdit()
        self.fraction_type_combo = QComboBox()
        self.fraction_type_combo.addItems(['wo', 'ao'])

        nuclide_layout = QHBoxLayout()
        nuclide_layout.addWidget(self.nuclide_field)
        nuclide_layout.addWidget(QLabel("Fraction:"))
        nuclide_layout.addWidget(self.fraction_field)
        nuclide_layout.addWidget(self.fraction_type_combo)
        build_layout.addRow("Nuclide/Element:", nuclide_layout)

        self.enrichment_field = QLineEdit()
        self.enrichment_field.setPlaceholderText("(optional, elements only) enrichment %, e.g. 3.5")
        self.enrichment_target_field = QLineEdit()
        self.enrichment_target_field.setPlaceholderText("target nuclide, default U235")
        self.enrichment_type_combo = QComboBox()
        self.enrichment_type_combo.addItems(['wo', 'ao'])
        enrich_layout = QHBoxLayout()
        enrich_layout.addWidget(self.enrichment_field)
        enrich_layout.addWidget(self.enrichment_target_field)
        enrich_layout.addWidget(self.enrichment_type_combo)
        build_layout.addRow("Enrichment:", enrich_layout)

        nuclide_btn_layout = QHBoxLayout()
        btn_add_nuclide = QPushButton("➕ Add Nuclide/Element")
        btn_add_nuclide.clicked.connect(self.add_nuclide)
        btn_remove_nuclide = QPushButton("🗑️ Remove Selected")
        btn_remove_nuclide.clicked.connect(self.remove_nuclide)
        nuclide_btn_layout.addWidget(btn_add_nuclide)
        nuclide_btn_layout.addWidget(btn_remove_nuclide)
        build_layout.addRow(nuclide_btn_layout)

        self.nuclide_list_widget = QListWidget()
        build_layout.addRow(self.nuclide_list_widget)

        self.sab_field = QLineEdit()
        self.sab_field.setPlaceholderText("e.g. c_H_in_H2O")
        sab_layout = QHBoxLayout()
        btn_add_sab = QPushButton("➕ Add S(α,β) Table")
        btn_add_sab.clicked.connect(self.add_sab)
        btn_remove_sab = QPushButton("🗑️ Remove Selected")
        btn_remove_sab.clicked.connect(self.remove_sab)
        sab_layout.addWidget(self.sab_field)
        build_layout.addRow("S(α,β) Table Name:", sab_layout)
        sab_btn_layout = QHBoxLayout()
        sab_btn_layout.addWidget(btn_add_sab)
        sab_btn_layout.addWidget(btn_remove_sab)
        build_layout.addRow(sab_btn_layout)

        self.sab_list_widget = QListWidget()
        self.sab_list_widget.setFixedHeight(60)
        build_layout.addRow(self.sab_list_widget)

        btn_generate_custom = QPushButton("Generate Custom Material Script")
        btn_generate_custom.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold;")
        btn_generate_custom.clicked.connect(self.generate_custom_material_script)
        build_layout.addRow(btn_generate_custom)

        left_layout.addWidget(build_group)

        # 3. مزج المواد (Mix Materials)
        mix_group = QGroupBox("3. Mix Materials (openmc.Material.mix_materials)")
        mix_layout = QFormLayout(mix_group)

        self.mix_name_field = QLineEdit()
        self.mix_name_field.setPlaceholderText("e.g. borated_water")
        mix_layout.addRow("Mixed Material Name:", self.mix_name_field)

        self.mix_vars_field = QLineEdit()
        self.mix_vars_field.setPlaceholderText("comma-separated existing material variables, e.g. WATER, B4C")
        mix_layout.addRow("Materials to Mix:", self.mix_vars_field)

        self.mix_fracs_field = QLineEdit()
        self.mix_fracs_field.setPlaceholderText("comma-separated fractions matching order above, e.g. 0.95, 0.05")
        mix_layout.addRow("Fractions:", self.mix_fracs_field)

        self.mix_type_combo = QComboBox()
        self.mix_type_combo.addItems(['vo', 'wo', 'ao'])
        mix_layout.addRow("Fraction Type:", self.mix_type_combo)

        btn_mix = QPushButton("Generate Mix Materials Script")
        btn_mix.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold;")
        btn_mix.clicked.connect(self.generate_mix_materials_script)
        mix_layout.addRow(btn_mix)

        left_layout.addWidget(mix_group)

        layout.addLayout(left_layout)

    # ==================================================
    # --- دوال مكتبة المواد الجاهزة ---
    # ==================================================
    def add_library_material(self):
        selection = self.lib_combo.currentText()
        if selection == "Select a material...":
            QMessageBox.warning(self, "Warning", "Please select a material from the library.")
            return

        mat_name = self._dedupe_material_name(selection.split(" (")[0].replace(" ", "_").replace("-", "_"))
        py_code = f"\n# --- Library Material: {selection} ---\n"
        py_code += f"{mat_name} = openmc.Material(name='{mat_name}')\n"

        if selection == "Water (H2O)":
            py_code += f"{mat_name}.set_density('g/cm3', 1.0)\n"
            py_code += f"{mat_name}.add_element('H', 2.0)\n"
            py_code += f"{mat_name}.add_element('O', 1.0)\n"
            py_code += f"{mat_name}.add_s_alpha_beta('c_H_in_H2O')\n"

        elif selection == "Heavy Water (D2O)":
            py_code += f"{mat_name}.set_density('g/cm3', 1.1056)\n"
            py_code += f"{mat_name}.add_nuclide('H2', 2.0)\n"
            py_code += f"{mat_name}.add_element('O', 1.0)\n"
            py_code += f"{mat_name}.add_s_alpha_beta('c_D_in_D2O')\n"

        elif selection == "B4C (Boron Carbide)":
            py_code += f"{mat_name}.set_density('g/cm3', 2.52)\n"
            py_code += f"{mat_name}.add_element('B', 4.0)\n"
            py_code += f"{mat_name}.add_element('C', 1.0)\n"

        elif selection == "Lead (Pb)":
            py_code += f"{mat_name}.set_density('g/cm3', 11.34)\n"
            py_code += f"{mat_name}.add_element('Pb', 1.0)\n"

        elif selection == "Stainless Steel 304":
            # Fe adjusted from 0.6805 to 0.6842 -- the previous weight
            # fractions summed to 0.9963, not 1.0 (Fe is the balance/
            # remainder element; Cr/Ni/Mn/Si/C keep their standard values).
            py_code += f"{mat_name}.set_density('g/cm3', 8.0)\n"
            py_code += f"{mat_name}.add_element('Fe', 0.6842, 'wo')\n"
            py_code += f"{mat_name}.add_element('Cr', 0.1900, 'wo')\n"
            py_code += f"{mat_name}.add_element('Ni', 0.0950, 'wo')\n"
            py_code += f"{mat_name}.add_element('Mn', 0.0200, 'wo')\n"
            py_code += f"{mat_name}.add_element('Si', 0.0100, 'wo')\n"
            py_code += f"{mat_name}.add_element('C', 0.0008, 'wo')\n"

        elif selection == "Stainless Steel 316":
            py_code += f"{mat_name}.set_density('g/cm3', 8.0)\n"
            py_code += f"{mat_name}.add_element('Fe', 0.655, 'wo')\n"
            py_code += f"{mat_name}.add_element('Cr', 0.170, 'wo')\n"
            py_code += f"{mat_name}.add_element('Ni', 0.120, 'wo')\n"
            py_code += f"{mat_name}.add_element('Mo', 0.025, 'wo')\n"
            py_code += f"{mat_name}.add_element('Mn', 0.020, 'wo')\n"
            py_code += f"{mat_name}.add_element('Si', 0.010, 'wo')\n"

        elif selection == "Concrete (Standard)":
            py_code += f"{mat_name}.set_density('g/cm3', 2.3)\n"
            py_code += f"{mat_name}.add_element('O', 0.529, 'wo')\n"
            py_code += f"{mat_name}.add_element('Si', 0.337, 'wo')\n"
            py_code += f"{mat_name}.add_element('Ca', 0.044, 'wo')\n"
            py_code += f"{mat_name}.add_element('Al', 0.034, 'wo')\n"
            py_code += f"{mat_name}.add_element('Na', 0.016, 'wo')\n"
            py_code += f"{mat_name}.add_element('Fe', 0.014, 'wo')\n"
            py_code += f"{mat_name}.add_element('K', 0.013, 'wo')\n"
            py_code += f"{mat_name}.add_element('H', 0.010, 'wo')\n"

        elif selection == "Polyethylene (PE)":
            py_code += f"{mat_name}.set_density('g/cm3', 0.93)\n"
            py_code += f"{mat_name}.add_element('C', 1.0)\n"
            py_code += f"{mat_name}.add_element('H', 2.0)\n"
            py_code += f"{mat_name}.add_s_alpha_beta('c_H_in_CH2')\n"

        elif selection == "Borated Polyethylene (5%)":
            py_code += f"{mat_name}.set_density('g/cm3', 0.95)\n"
            py_code += f"{mat_name}.add_element('C', 0.814, 'wo')\n"
            py_code += f"{mat_name}.add_element('H', 0.136, 'wo')\n"
            py_code += f"{mat_name}.add_element('B', 0.050, 'wo')\n"
            py_code += f"{mat_name}.add_s_alpha_beta('c_H_in_CH2')\n"

        elif selection == "Uranium Dioxide (UO2 - 3% Enriched)":
            py_code += f"{mat_name}.set_density('g/cm3', 10.5)\n"
            py_code += f"{mat_name}.add_nuclide('U235', 0.03, 'wo')\n"
            py_code += f"{mat_name}.add_nuclide('U238', 0.8515, 'wo')\n"
            py_code += f"{mat_name}.add_element('O', 0.1185, 'wo')\n"

        elif selection == "HPGe (High-Purity Germanium)":
            py_code += f"{mat_name}.set_density('g/cm3', 5.323)\n"
            py_code += f"{mat_name}.add_element('Ge', 1.0)\n"

        elif selection == "NaI (Sodium Iodide)":
            py_code += f"{mat_name}.set_density('g/cm3', 3.67)\n"
            py_code += f"{mat_name}.add_element('Na', 1.0)\n"
            py_code += f"{mat_name}.add_element('I', 1.0)\n"

        self.script_generated.emit(py_code)
        QMessageBox.information(self, "Success", f"{selection} added to the script!")

    # ==================================================
    # --- دوال بناء مادة مخصصة يدوياً ---
    # ==================================================
    def add_nuclide(self):
        nuc_raw = self.nuclide_field.text().strip()
        frac = self.fraction_field.text().strip()
        f_type = self.fraction_type_combo.currentText()
        if not nuc_raw or not frac:
            QMessageBox.warning(self, "Warning", "Please enter both nuclide and fraction.")
            return
        try:
            float(frac)
        except ValueError:
            QMessageBox.warning(self, "Warning", f"Fraction must be a number, got '{frac}'.")
            return

        # Normalized here (not just at generation time) so the list widget
        # itself shows the corrected form immediately, matching what the
        # generated script will actually contain.
        nuc = _normalize_nuclide_symbol(nuc_raw)

        enrichment_txt = self.enrichment_field.text().strip()
        enrichment = None
        enrichment_target = None
        enrichment_type = self.enrichment_type_combo.currentText()
        if enrichment_txt:
            if not nuc.isalpha():
                QMessageBox.warning(
                    self, "Warning",
                    "Enrichment only applies to an element symbol (e.g. 'U'), not a specific nuclide like 'U235'.")
                return
            try:
                enrichment = float(enrichment_txt)
            except ValueError:
                QMessageBox.warning(self, "Warning", f"Enrichment must be a number, got '{enrichment_txt}'.")
                return
            if not (0 < enrichment < 100):
                QMessageBox.warning(self, "Warning", "Enrichment must be a percentage between 0 and 100.")
                return
            enrichment_target = self.enrichment_target_field.text().strip() or None

        self.current_material_nuclides.append(
            (nuc, frac, f_type, enrichment, enrichment_target, enrichment_type))
        display = f"{nuc} : {frac} {f_type}"
        if enrichment is not None:
            display += f" | enrichment={enrichment}% ({enrichment_target or 'U235'}, {enrichment_type})"
        self.nuclide_list_widget.addItem(display)

        self.nuclide_field.clear()
        self.fraction_field.clear()
        self.enrichment_field.clear()
        self.enrichment_target_field.clear()

    def remove_nuclide(self):
        row = self.nuclide_list_widget.currentRow()
        if row >= 0:
            self.nuclide_list_widget.takeItem(row)
            self.current_material_nuclides.pop(row)

    def add_sab(self):
        name = self.sab_field.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter an S(α,β) table name (e.g. c_H_in_H2O).")
            return
        self.current_material_sab.append(name)
        self.sab_list_widget.addItem(name)
        self.sab_field.clear()

    def remove_sab(self):
        row = self.sab_list_widget.currentRow()
        if row >= 0:
            self.sab_list_widget.takeItem(row)
            self.current_material_sab.pop(row)

    def generate_custom_material_script(self):
        mat_name = self.mat_name_field.text().strip()
        dens_val = self.density_val_field.text().strip()
        dens_unit = self.density_unit_combo.currentText()

        if not mat_name or not dens_val:
            QMessageBox.warning(self, "Warning", "Please provide a Material Name and Density.")
            return

        if not self.current_material_nuclides:
            QMessageBox.warning(self, "Warning", "Please add at least one nuclide/element.")
            return

        if mat_name in self._used_material_names:
            QMessageBox.warning(
                self, "Warning",
                f"A material named '{mat_name}' was already added this session. "
                f"Choose a different Material Name to avoid overwriting it.")
            return

        temperature_txt = self.temperature_field.text().strip()
        if temperature_txt:
            try:
                float(temperature_txt)
            except ValueError:
                QMessageBox.warning(self, "Warning", f"Temperature must be a number, got '{temperature_txt}'.")
                return

        volume_txt = self.volume_field.text().strip()
        if volume_txt:
            try:
                float(volume_txt)
            except ValueError:
                QMessageBox.warning(self, "Warning", f"Volume must be a number, got '{volume_txt}'.")
                return

        self._used_material_names.add(mat_name)

        py_code = f"\n# --- Custom Material: {mat_name} ---\n"
        py_code += f"{mat_name} = openmc.Material(name='{mat_name}')\n"
        py_code += f"{mat_name}.set_density('{dens_unit}', {dens_val})\n"

        for (nuc, frac, f_type, enrichment, enrichment_target, enrichment_type) in self.current_material_nuclides:
            if nuc.isalpha():
                extra = ""
                if enrichment is not None:
                    extra = f", enrichment={enrichment}, enrichment_type='{enrichment_type}'"
                    if enrichment_target:
                        extra += f", enrichment_target='{enrichment_target}'"
                py_code += f"{mat_name}.add_element('{nuc}', {frac}, percent_type='{f_type}'{extra})\n"
            else:
                py_code += f"{mat_name}.add_nuclide('{nuc}', {frac}, percent_type='{f_type}')\n"

        for sab in self.current_material_sab:
            py_code += f"{mat_name}.add_s_alpha_beta('{sab}')\n"

        if temperature_txt:
            py_code += f"{mat_name}.temperature = {temperature_txt}\n"
        if volume_txt:
            py_code += f"{mat_name}.volume = {volume_txt}\n"
        if self.chk_depletable.isChecked():
            py_code += f"{mat_name}.depletable = True\n"

        self.script_generated.emit(py_code)
        QMessageBox.information(self, "Success", f"Material {mat_name} added to script!")

        # تصفير الحقول للبدء بمادة جديدة
        self.current_material_nuclides.clear()
        self.current_material_sab.clear()
        self.nuclide_list_widget.clear()
        self.sab_list_widget.clear()
        self.mat_name_field.clear()
        self.density_val_field.clear()
        self.temperature_field.clear()
        self.volume_field.clear()
        self.chk_depletable.setChecked(False)

    # ==================================================
    # --- مزج المواد (Mix Materials) ---
    # ==================================================
    def generate_mix_materials_script(self):
        mix_name = self.mix_name_field.text().strip()
        vars_txt = self.mix_vars_field.text().strip()
        fracs_txt = self.mix_fracs_field.text().strip()
        mix_type = self.mix_type_combo.currentText()

        if not mix_name or not vars_txt or not fracs_txt:
            QMessageBox.warning(self, "Warning",
                                 "Please provide a Mixed Material Name, the Materials to Mix, and their Fractions.")
            return

        if mix_name in self._used_material_names:
            QMessageBox.warning(
                self, "Warning",
                f"A material named '{mix_name}' was already added this session. Choose a different name.")
            return

        mat_vars = [v.strip() for v in vars_txt.split(",") if v.strip()]
        try:
            fracs = [float(f.strip()) for f in fracs_txt.split(",") if f.strip()]
        except ValueError:
            QMessageBox.warning(self, "Warning", f"Fractions must all be numbers, got '{fracs_txt}'.")
            return

        if len(mat_vars) < 2:
            QMessageBox.warning(self, "Warning", "Provide at least two materials to mix.")
            return
        if len(mat_vars) != len(fracs):
            QMessageBox.warning(
                self, "Warning",
                f"The number of Materials ({len(mat_vars)}) and Fractions ({len(fracs)}) must match.")
            return

        self._used_material_names.add(mix_name)

        mats_str = ", ".join(mat_vars)
        fracs_str = ", ".join(str(f) for f in fracs)
        py_code = f"\n# --- Mixed Material: {mix_name} ---\n"
        py_code += f"{mix_name} = openmc.Material.mix_materials([{mats_str}], [{fracs_str}], '{mix_type}')\n"
        py_code += f"{mix_name}.name = '{mix_name}'\n"

        self.script_generated.emit(py_code)
        QMessageBox.information(self, "Success", f"Mixed material {mix_name} added to script!")

        self.mix_name_field.clear()
        self.mix_vars_field.clear()
        self.mix_fracs_field.clear()
