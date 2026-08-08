import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QListWidget, QAbstractItemView, QMessageBox, QFormLayout, QStackedWidget
)
from PySide6.QtCore import Signal


class TalliesPageWidget(QWidget):
    """
    Advanced OpenMC Tallies Configuration Page.
    Supports Cell, Material, Energy, Mesh, and Pulse Height (Detector Simulation) filters.
    """
    script_generated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # ==========================================
        # --- قسم تعريف العداد (Tally Definition) ---
        # ==========================================
        tally_group = QGroupBox("Define Tally")
        tally_layout = QFormLayout(tally_group)

        self.tally_name_field = QLineEdit("tally1")
        tally_layout.addRow("Tally Name (Variable):", self.tally_name_field)

        self.scores_list = QListWidget()
        self.scores_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # أضفنا pulse-height إلى قائمة القياسات
        scores = ["flux", "absorption", "fission", "total", "scatter", "nu-fission", "capture", "pulse-height"]
        self.scores_list.addItems(scores)
        self.scores_list.setFixedHeight(120)
        self.scores_list.setToolTip(
            "Hold Ctrl to select multiple scores. Select 'pulse-height' for HPGe/NaI detector simulations.")
        tally_layout.addRow("Scores to Measure:", self.scores_list)

        # ==========================================
        # --- قسم الفلتر الديناميكي (Dynamic Filters) ---
        # ==========================================
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems([
            "No Filter (Global)",
            "CellFilter",
            "MaterialFilter",
            "EnergyFilter",
            "MeshFilter",
            "Pulse Height (Detector F8)"  # إضافة خيار محاكاة الكاشف
        ])
        self.filter_type_combo.currentTextChanged.connect(self._on_filter_changed)
        tally_layout.addRow("Filter Type:", self.filter_type_combo)

        # حاوية ديناميكية تتغير حسب اختيار الفلتر
        self.filter_stack = QStackedWidget()

        # 0: بدون فلتر
        self.filter_stack.addWidget(QLabel("Global tally: Measures across the entire geometry."))

        # 1: فلتر خلية أو مادة
        self.cell_mat_widget = QWidget()
        cm_layout = QFormLayout(self.cell_mat_widget)
        cm_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_target_field = QLineEdit("c1")
        cm_layout.addRow("Target Variable (e.g., c1 or WATER):", self.filter_target_field)
        self.filter_stack.addWidget(self.cell_mat_widget)

        # 2: فلتر طاقة (Energy)
        self.energy_widget = QWidget()
        e_layout = QFormLayout(self.energy_widget)
        e_layout.setContentsMargins(0, 0, 0, 0)
        self.e_bins_field = QLineEdit("0.0, 1.0e6, 2.0e7")
        self.e_bins_field.setToolTip("Enter energy bin boundaries in eV separated by commas.")
        e_layout.addRow("Energy Bins (eV):", self.e_bins_field)
        self.filter_stack.addWidget(self.energy_widget)

        # 3: فلتر شبكي (Mesh)
        self.mesh_widget = QWidget()
        m_layout = QFormLayout(self.mesh_widget)
        m_layout.setContentsMargins(0, 0, 0, 0)
        self.m_dim_field = QLineEdit("10, 10, 1")
        self.m_ll_field = QLineEdit("-10.0, -10.0, -10.0")
        self.m_ur_field = QLineEdit("10.0, 10.0, 10.0")
        m_layout.addRow("Dimension (Nx, Ny, Nz):", self.m_dim_field)
        m_layout.addRow("Lower Left (x, y, z):", self.m_ll_field)
        m_layout.addRow("Upper Right (x, y, z):", self.m_ur_field)
        self.filter_stack.addWidget(self.mesh_widget)

        # 4: محاكاة الكاشف (Pulse Height)
        self.detector_widget = QWidget()
        d_layout = QFormLayout(self.detector_widget)
        d_layout.setContentsMargins(0, 0, 0, 0)
        self.d_cell_field = QLineEdit("c1")
        self.d_bins_field = QLineEdit("np.linspace(0, 3e6, 3000)")  # 3000 channel up to 3 MeV
        self.d_bins_field.setToolTip("Use numpy for bins, e.g., np.linspace(min, max, num_channels)")
        self.d_geb_a = QLineEdit("0.0")
        self.d_geb_b = QLineEdit("0.0001")
        self.d_geb_c = QLineEdit("0.0")

        d_layout.addRow("Detector Cell Variable:", self.d_cell_field)
        d_layout.addRow("Energy Channels (eV):", self.d_bins_field)
        d_layout.addRow(QLabel("Gaussian Energy Broadening (GEB) Parameters: FWHM = a + b*sqrt(E + c*E^2)"))
        d_layout.addRow("GEB 'a':", self.d_geb_a)
        d_layout.addRow("GEB 'b':", self.d_geb_b)
        d_layout.addRow("GEB 'c':", self.d_geb_c)
        self.filter_stack.addWidget(self.detector_widget)

        tally_layout.addRow("Filter Settings:", self.filter_stack)
        main_layout.addWidget(tally_group)

        # ==========================================
        # --- الأزرار والقائمة السفلية ---
        # ==========================================
        btn_add = QPushButton("Add Tally & Generate Script")
        btn_add.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold; padding: 10px;")
        btn_add.clicked.connect(self.add_tally_action)
        main_layout.addWidget(btn_add)

        main_layout.addWidget(QLabel("<b>Created Tallies:</b>"))
        self.created_tallies_list = QListWidget()
        main_layout.addWidget(self.created_tallies_list)

    def _on_filter_changed(self, text):
        """تغيير الواجهة بناءً على نوع الفلتر المختار"""
        if text == "No Filter (Global)":
            self.filter_stack.setCurrentIndex(0)
        elif text in ["CellFilter", "MaterialFilter"]:
            self.filter_stack.setCurrentIndex(1)
        elif text == "EnergyFilter":
            self.filter_stack.setCurrentIndex(2)
        elif text == "MeshFilter":
            self.filter_stack.setCurrentIndex(3)
        elif text == "Pulse Height (Detector F8)":
            self.filter_stack.setCurrentIndex(4)
            # تحديد pulse-height تلقائياً
            items = self.scores_list.findItems("pulse-height", openmc.Qt.MatchFlag.MatchExactly) if hasattr(openmc,
                                                                                                            'Qt') else None  # Fallback for pyqt/pyside
            if not items:  # Simple loop fallback
                for i in range(self.scores_list.count()):
                    if self.scores_list.item(i).text() == "pulse-height":
                        self.scores_list.item(i).setSelected(True)
                        break

    def add_tally_action(self):
        t_name = self.tally_name_field.text().strip()
        selected_items = self.scores_list.selectedItems()
        f_type = self.filter_type_combo.currentText()

        if not t_name:
            QMessageBox.warning(self, "Warning", "Please provide a valid Tally Name.")
            return

        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one Score (e.g., flux).")
            return

        selected_scores = [item.text() for item in selected_items]

        # --- توليد كود البايثون (Script Generation) ---
        py_code = f"\n# --- Tally Definition: {t_name} ---\n"
        py_code += f"{t_name} = openmc.Tally(name='{t_name}')\n"
        scores_str = ", ".join([f"'{s}'" for s in selected_scores])
        py_code += f"{t_name}.scores = [{scores_str}]\n"

        filter_desc = "Global"

        if f_type == "CellFilter":
            target = self.filter_target_field.text().strip()
            py_code += f"{t_name}_filter = openmc.CellFilter([{target}])\n"
            py_code += f"{t_name}.filters = [{t_name}_filter]\n"
            filter_desc = f"Cell({target})"

        elif f_type == "MaterialFilter":
            target = self.filter_target_field.text().strip()
            py_code += f"{t_name}_filter = openmc.MaterialFilter([{target}])\n"
            py_code += f"{t_name}.filters = [{t_name}_filter]\n"
            filter_desc = f"Material({target})"

        elif f_type == "EnergyFilter":
            bins = self.e_bins_field.text().strip()
            py_code += f"{t_name}_filter = openmc.EnergyFilter([{bins}])\n"
            py_code += f"{t_name}.filters = [{t_name}_filter]\n"
            filter_desc = f"Energy Bins"

        elif f_type == "MeshFilter":
            dim = self.m_dim_field.text().strip()
            ll = self.m_ll_field.text().strip()
            ur = self.m_ur_field.text().strip()
            mesh_var = f"mesh_{t_name}"
            py_code += f"{mesh_var} = openmc.RegularMesh()\n"
            py_code += f"{mesh_var}.dimension = ({dim})\n"
            py_code += f"{mesh_var}.lower_left = ({ll})\n"
            py_code += f"{mesh_var}.upper_right = ({ur})\n"
            py_code += f"{t_name}_filter = openmc.MeshFilter({mesh_var})\n"
            py_code += f"{t_name}.filters = [{t_name}_filter]\n"
            filter_desc = f"Mesh({dim})"

        elif f_type == "Pulse Height (Detector F8)":
            if "pulse-height" not in selected_scores:
                QMessageBox.warning(self, "Warning",
                                    "You must select 'pulse-height' in scores for a detector simulation.")
                return

            target_cell = self.d_cell_field.text().strip()
            energy_bins = self.d_bins_field.text().strip()

            py_code += f"import numpy as np\n"
            py_code += f"{t_name}_cell_filter = openmc.CellFilter([{target_cell}])\n"
            py_code += f"{t_name}_energy_bins = {energy_bins}\n"
            py_code += f"{t_name}_energy_filter = openmc.EnergyFilter({t_name}_energy_bins)\n"
            py_code += f"{t_name}.filters = [{t_name}_cell_filter, {t_name}_energy_filter]\n"
            filter_desc = f"Detector({target_cell})"

            # ملاحظة: في النسخ الحديثة جداً من OpenMC هناك نقاش حول إضافة GEB مباشرة للفلتر
            # لكن الطريقة الأضمن حاليا (كما في MCNP) هي تطبيق التوسيع على النتائج
            py_code += f"# GEB Parameters: a={self.d_geb_a.text()}, b={self.d_geb_b.text()}, c={self.d_geb_c.text()}\n"

        py_code += (
            f"try:\n"
            f"    tallies.append({t_name})\n"
            f"except NameError:\n"
            f"    tallies = openmc.Tallies([{t_name}])\n"
        )

        self.script_generated.emit(py_code)
        self.created_tallies_list.addItem(
            f"Tally: {t_name} | Filter: {filter_desc} | Scores: {', '.join(selected_scores)}")
        QMessageBox.information(self, "Success", f"Tally '{t_name}' added to script successfully!")