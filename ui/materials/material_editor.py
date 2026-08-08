import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QComboBox, QDialogButtonBox, QDoubleSpinBox
)

logger = logging.getLogger("OpenMC Studio")

class MaterialEditorDialog(QDialog):
    """
    Dialog to define basic OpenMC material properties like density and temperature.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Material Editor")
        self.setMinimumWidth(350)
        self._setup_ui()
        logger.debug("MaterialEditorDialog initialized.")

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Material Name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., UO2, Light Water, Steel")

        # Density Input (allows precise decimals)
        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(0.0001, 100.0)
        self.density_input.setValue(1.0)
        self.density_input.setDecimals(4)

        # Density Units
        self.density_units = QComboBox()
        self.density_units.addItems(["g/cm3", "g/cc", "kg/m3", "atom/b-cm"])

        # Temperature (Kelvin)
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 5000.0)
        self.temperature_input.setValue(293.6)
        self.temperature_input.setSuffix(" K")
        self.temperature_input.setDecimals(1)

        form_layout.addRow("Material Name:", self.name_input)
        form_layout.addRow("Density:", self.density_input)
        form_layout.addRow("Density Units:", self.density_units)
        form_layout.addRow("Default Temperature:", self.temperature_input)

        layout.addLayout(form_layout)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_material_data(self) -> dict:
        """Returns the configured material properties."""
        return {
            "name": self.name_input.text(),
            "density": self.density_input.value(),
            "units": self.density_units.currentText(),
            "temperature": self.temperature_input.value()
        }