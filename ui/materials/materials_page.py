import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QComboBox, QFormLayout, QListWidget, QMessageBox
)
from PySide6.QtCore import Signal


class MaterialsPageWidget(QWidget):
    """
    OpenMC Materials Builder Page.
    Includes manual building and a built-in standard materials library.
    """
    script_generated = Signal(str)

    def __init__(self, project_manager=None, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.current_material_nuclides = []
        self._init_ui()

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

        self.nuclide_field = QLineEdit()
        self.nuclide_field.setPlaceholderText("e.g., U235")
        self.fraction_field = QLineEdit()
        self.fraction_type_combo = QComboBox()
        self.fraction_type_combo.addItems(['wo', 'ao'])

        nuclide_layout = QHBoxLayout()
        nuclide_layout.addWidget(self.nuclide_field)
        nuclide_layout.addWidget(QLabel("Fraction:"))
        nuclide_layout.addWidget(self.fraction_field)
        nuclide_layout.addWidget(self.fraction_type_combo)
        build_layout.addRow("Nuclide:", nuclide_layout)

        btn_add_nuclide = QPushButton("Add Nuclide")
        btn_add_nuclide.clicked.connect(self.add_nuclide)
        build_layout.addRow(btn_add_nuclide)

        self.nuclide_list_widget = QListWidget()
        build_layout.addRow(self.nuclide_list_widget)

        btn_generate_custom = QPushButton("Generate Custom Material Script")
        btn_generate_custom.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold;")
        btn_generate_custom.clicked.connect(self.generate_custom_material_script)
        build_layout.addRow(btn_generate_custom)

        left_layout.addWidget(build_group)
        layout.addLayout(left_layout)

    # ==================================================
    # --- دوال مكتبة المواد الجاهزة ---
    # ==================================================
    def add_library_material(self):
        selection = self.lib_combo.currentText()
        if selection == "Select a material...":
            QMessageBox.warning(self, "Warning", "Please select a material from the library.")
            return

        mat_name = selection.split(" (")[0].replace(" ", "_").replace("-", "_")
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
            py_code += f"{mat_name}.set_density('g/cm3', 8.0)\n"
            py_code += f"{mat_name}.add_element('Fe', 0.6805, 'wo')\n"
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
        nuc = self.nuclide_field.text().strip()
        frac = self.fraction_field.text().strip()
        f_type = self.fraction_type_combo.currentText()
        if not nuc or not frac:
            QMessageBox.warning(self, "Warning", "Please enter both nuclide and fraction.")
            return

        self.current_material_nuclides.append((nuc, frac, f_type))
        self.nuclide_list_widget.addItem(f"{nuc} : {frac} {f_type}")

        self.nuclide_field.clear()
        self.fraction_field.clear()

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

        py_code = f"\n# --- Custom Material: {mat_name} ---\n"
        py_code += f"{mat_name} = openmc.Material(name='{mat_name}')\n"
        py_code += f"{mat_name}.set_density('{dens_unit}', {dens_val})\n"

        for (nuc, frac, f_type) in self.current_material_nuclides:
            if nuc.isalpha():
                py_code += f"{mat_name}.add_element('{nuc}', {frac}, percent_type='{f_type}')\n"
            else:
                py_code += f"{mat_name}.add_nuclide('{nuc}', {frac}, percent_type='{f_type}')\n"

        self.script_generated.emit(py_code)
        QMessageBox.information(self, "Success", f"Material {mat_name} added to script!")

        # تصفير الحقول للبدء بمادة جديدة
        self.current_material_nuclides.clear()
        self.nuclide_list_widget.clear()
        self.mat_name_field.clear()
        self.density_val_field.clear()