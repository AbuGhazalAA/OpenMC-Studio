import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout,
                               QComboBox, QLineEdit, QLabel, QHBoxLayout, QPushButton, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt, Signal


class DepletionBurnupWidget(QWidget):
    # إشارة لإرسال الكود المولد إلى محرر النصوص الرئيسي
    script_generated = Signal(str)

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- صندوق الإعدادات الشامل ---
        group_box = QGroupBox("Depletion & Burnup Settings")
        form_layout = QFormLayout()

        # 0. اختيار ملف الاستنزاف (Chain File) - الجديد!
        chain_layout = QHBoxLayout()
        self.chain_input = QLineEdit()
        self.chain_input.setPlaceholderText("Select chain_endfb71_pwr.xml ...")
        self.btn_browse_chain = QPushButton("📂 Browse")
        self.btn_browse_chain.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        self.btn_browse_chain.clicked.connect(self.browse_chain_file)
        chain_layout.addWidget(self.chain_input)
        chain_layout.addWidget(self.btn_browse_chain)
        form_layout.addRow("Depletion Chain (XML):", chain_layout)

        # 1. إعدادات الطاقة
        power_layout = QHBoxLayout()
        self.power_input = QLineEdit("1.0")
        self.power_unit = QComboBox()
        self.power_unit.addItems(["MW", "W", "W/gHM", "kW/gHM"])
        power_layout.addWidget(self.power_input)
        power_layout.addWidget(self.power_unit)
        form_layout.addRow("System Power:", power_layout)

        # 2. نوع المحاكاة (زمن أم احتراق)
        self.step_type_combo = QComboBox()
        self.step_type_combo.addItems(["Time Steps (Days)", "Burnup Steps (MWd/kgHM)"])
        self.step_type_combo.currentIndexChanged.connect(self.update_step_label)
        form_layout.addRow("Simulation Mode:", self.step_type_combo)

        # 3. إدخال الخطوات
        self.steps_input = QLineEdit("0.1, 0.5, 1.0")
        self.step_label = QLabel("Time Steps (Days):")
        form_layout.addRow(self.step_label, self.steps_input)

        # 4. خوارزمية التكامل
        self.integrator_combo = QComboBox()
        self.integrator_combo.addItems([
            "PredictorIntegrator",
            "CECMIntegrator",
            "EPCIntegrator",
            "LEQIIntegrator"
        ])
        form_layout.addRow("Algorithm (Integrator):", self.integrator_combo)

        group_box.setLayout(form_layout)
        main_layout.addWidget(group_box)

        # --- زر توليد الكود ---
        self.btn_generate = QPushButton("🔥 Generate Depletion Script")
        self.btn_generate.setStyleSheet(
            "background-color: #FF8C00; color: white; font-weight: bold; padding: 10px; font-size: 14px; border-radius: 5px;")
        self.btn_generate.clicked.connect(self.generate_script)
        main_layout.addWidget(self.btn_generate)

        main_layout.addStretch()
        # استدعاء لتحديث النصوص عند الفتح
        self.update_step_label()

    def update_step_label(self):
        if self.step_type_combo.currentIndex() == 0:
            self.step_label.setText("Time Steps (Days):")
            self.steps_input.setPlaceholderText("e.g., 10, 20, 30")
        else:
            self.step_label.setText("Burnup Steps (MWd/kgHM):")
            self.steps_input.setPlaceholderText("e.g., 0.1, 0.5, 1.0")

    def browse_chain_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Depletion Chain File", "",
                                                  "XML Files (*.xml);;All Files (*)")
        if filepath:
            self.chain_input.setText(filepath)

    # --- دالة صياغة كود البايثون الخاص بـ OpenMC ---
    def generate_script(self):
        chain_path = self.chain_input.text().strip()
        if not chain_path:
            QMessageBox.warning(self, "Warning", "Please select a Depletion Chain File (XML) first.")
            return

        # --- التحويل السحري من مسار Windows إلى مسار WSL ---
        wsl_chain_path = chain_path.replace('\\', '/')
        if len(wsl_chain_path) > 2 and wsl_chain_path[1] == ':':
            drive_letter = wsl_chain_path[0].lower()
            wsl_chain_path = f"/mnt/{drive_letter}" + wsl_chain_path[2:]
        # ----------------------------------------------------

        power = self.power_input.text().strip()
        unit = self.power_unit.currentText()
        steps_text = self.steps_input.text().strip()
        mode = self.step_type_combo.currentIndex()
        integrator = self.integrator_combo.currentText()

        # التحقق من صحة الأرقام المدخلة
        try:
            steps_list = [float(x.strip()) for x in steps_text.split(',')]
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers separated by commas for steps.")
            return

        steps_str = ", ".join(str(x) for x in steps_list)

        power_var = "power"
        if unit == "W/gHM" or unit == "kW/gHM":
            power_var = "power_density"

        code = f"\n# --- Depletion & Burnup Setup ---\n"
        code += "import openmc.deplete\n\n"
        code += "# 1. Set Chain File (WSL Compatible Path)\n"
        code += f"chain_file = '{wsl_chain_path}'\n\n"
        code += "# 2. Setup Depletion Operator\n"
        code += "model = openmc.Model(geometry=geometry, materials=materials, settings=settings)\n"
        code += "operator = openmc.deplete.CoupledOperator(model, chain_file)\n\n"

        code += f"# 3. Power and Steps\n"
        if unit == "MW":
            code += f"{power_var} = {power}e6  # Watts\n"
        elif unit == "kW/gHM":
            code += f"{power_var} = {power}e3  # W/gHM\n"
        else:
            code += f"{power_var} = {power}  # {unit}\n"

        if mode == 0:  # Time Steps
            code += f"time_steps = [{steps_str}]  # Days\n"
            code += f"time_steps_sec = [t * 24 * 60 * 60 for t in time_steps] # Convert to seconds\n\n"
            integrator_args = f"operator, time_steps_sec, {power_var}={power_var}"
        else:  # Burnup Steps
            code += f"burnup_steps = [{steps_str}]  # MWd/kgHM\n\n"
            integrator_args = f"operator, burnup_steps, {power_var}={power_var}, timestep_units='MWd/kg'"

        code += "# 4. Run Integrator\n"
        code += f"integrator = openmc.deplete.{integrator}({integrator_args})\n"
        code += "integrator.integrate()\n"

        self.script_generated.emit(code)