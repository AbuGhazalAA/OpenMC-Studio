import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QComboBox, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QRadioButton
)

logger = logging.getLogger("OpenMC Studio")

class NuclideEditorDialog(QDialog):
    """
    Dialog to add a nuclide or element to a specific material.
    Defines the composition fraction (atomic or weight).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Nuclide / Element")
        self.setMinimumWidth(350)
        self._setup_ui()
        logger.debug("NuclideEditorDialog initialized.")

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Type (Nuclide vs Element)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Nuclide", "Element", "Macroscopic"])

        # Name/Symbol
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., U235, O, H1")

        # Fraction Value
        self.fraction_input = QDoubleSpinBox()
        self.fraction_input.setRange(0.000001, 100.0)
        self.fraction_input.setValue(1.0)
        self.fraction_input.setDecimals(6)

        # Fraction Type (Atomic vs Weight)
        fraction_type_layout = QHBoxLayout()
        self.radio_ao = QRadioButton("Atomic (ao)")
        self.radio_wo = QRadioButton("Weight (wo)")
        self.radio_ao.setChecked(True)
        fraction_type_layout.addWidget(self.radio_ao)
        fraction_type_layout.addWidget(self.radio_wo)

        form_layout.addRow("Component Type:", self.type_combo)
        form_layout.addRow("Name / Symbol:", self.name_input)
        form_layout.addRow("Fraction Value:", self.fraction_input)
        form_layout.addRow("Fraction Type:", fraction_type_layout)

        layout.addLayout(form_layout)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_nuclide_data(self) -> dict:
        """Returns the configured nuclide/element properties."""
        frac_type = "ao" if self.radio_ao.isChecked() else "wo"
        return {
            "type": self.type_combo.currentText(),
            "name": self.name_input.text(),
            "fraction": self.fraction_input.value(),
            "fraction_type": frac_type
        }