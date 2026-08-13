import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QFormLayout, QMessageBox, QFileDialog
)
from PySide6.QtCore import Signal


class DepletionPageWidget(QWidget):
    script_generated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Chain File
        chain_group = QGroupBox("1. Nuclear Data (Decay/Yield Chain)")
        chain_group.setStyleSheet("font-weight: bold; color: #008000;")
        chain_layout = QHBoxLayout(chain_group)

        self.chain_path_field = QLineEdit("chain_endfb71_pwr.xml")
        self.chain_path_field.setStyleSheet("font-weight: normal; color: black;")

        btn_browse_chain = QPushButton("Browse...")
        btn_browse_chain.setStyleSheet("font-weight: normal; color: black;")
        btn_browse_chain.clicked.connect(self._browse_chain)

        chain_layout.addWidget(QLabel("Chain XML File:"))
        chain_layout.addWidget(self.chain_path_field)
        chain_layout.addWidget(btn_browse_chain)
        layout.addWidget(chain_group)

        # 2. Power and Timesteps
        burnup_group = QGroupBox("2. Power & Timesteps")
        burnup_group.setStyleSheet("font-weight: bold; color: #008000;")
        burnup_layout = QFormLayout(burnup_group)

        self.power_field = QLineEdit("100.0e6")
        self.power_field.setStyleSheet("font-weight: normal; color: black;")
        self.power_field.setToolTip("Total thermal power in Watts")
        burnup_layout.addRow("System Power (Watts):", self.power_field)

        self.timesteps_field = QLineEdit("30.0, 30.0, 30.0, 30.0")
        self.timesteps_field.setStyleSheet("font-weight: normal; color: black;")
        self.timesteps_field.setToolTip("Comma-separated list of time steps")
        burnup_layout.addRow("Time Steps:", self.timesteps_field)

        self.time_units_combo = QComboBox()
        self.time_units_combo.setStyleSheet("font-weight: normal; color: black;")
        self.time_units_combo.addItems(["d (Days)", "h (Hours)", "s (Seconds)", "MWd/kg (Burnup)"])
        burnup_layout.addRow("Units:", self.time_units_combo)

        layout.addWidget(burnup_group)

        # 3. Integrator
        int_group = QGroupBox("3. Integrator Algorithm")
        int_group.setStyleSheet("font-weight: bold; color: #008000;")
        int_layout = QFormLayout(int_group)
        self.integrator_combo = QComboBox()
        self.integrator_combo.setStyleSheet("font-weight: normal; color: black;")
        self.integrator_combo.addItems(["PredictorIntegrator", "CECMIntegrator", "EPCRK4Integrator", "LEQIIntegrator"])
        int_layout.addRow("Integrator:", self.integrator_combo)
        layout.addWidget(int_group)

        # 4. Generate Button
        btn_generate = QPushButton("🔥 Append Depletion Script")
        btn_generate.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold; padding: 10px;")
        btn_generate.clicked.connect(self.generate_script)
        layout.addWidget(btn_generate)
        layout.addStretch()

    def _browse_chain(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Chain File", "", "XML Files (*.xml);;All Files (*)")
        if path:
            self.chain_path_field.setText(path)

    def generate_script(self):
        chain_file = self.chain_path_field.text().strip()
        power = self.power_field.text().strip()
        timesteps = self.timesteps_field.text().strip()
        units_text = self.time_units_combo.currentText()
        unit = units_text.split(" ")[0]
        if unit == "MWd/kg":
            unit = "MWd/kg"

        integrator = self.integrator_combo.currentText()

        if not chain_file or not power or not timesteps:
            QMessageBox.warning(self, "Missing Data", "Please fill in all fields.")
            return

        code = f"\n# ==========================================\n"
        code += f"# Depletion / Burnup Setup\n"
        code += f"# ==========================================\n"
        code += f"import openmc.deplete\n\n"
        code += f"chain_file = '{chain_file}'\n"
        code += f"power = {power}  # Watts\n"
        code += f"timesteps = [{timesteps}]\n\n"

        code += f"# Initialize the coupled operator\n"
        code += f"operator = openmc.deplete.CoupledOperator(geometry, settings, chain_file)\n\n"

        code += f"integrator = openmc.deplete.{integrator}(operator, timesteps, power, timestep_units='{unit}')\n"

        code += f"\n# Run the depletion simulation\n"
        code += f"# (This replaces the standard openmc.run() command)\n"
        code += f"integrator.integrate()\n"

        self.script_generated.emit(code)
        QMessageBox.information(self, "Success",
                                "Depletion script appended! Note: Depletion replaces the standard openmc.run() command.")