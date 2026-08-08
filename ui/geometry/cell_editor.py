import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QComboBox, QDialogButtonBox, QLabel
)

logger = logging.getLogger("OpenMC Studio")


class CellEditorDialog(QDialog):
    """
    Dialog to create and edit OpenMC cells.
    Allows users to define cell regions and assign materials.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Editor")
        self.setMinimumWidth(400)
        self._setup_ui()
        logger.debug("CellEditorDialog initialized.")

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Fuel Pin, Moderator")

        # Placeholder for materials (will be linked to the Materials module later)
        self.material_combo = QComboBox()
        self.material_combo.addItems(["Void", "Water (Placeholder)", "UO2 (Placeholder)"])

        # Region definition (Intersection/Union of surfaces)
        self.region_input = QLineEdit()
        self.region_input.setPlaceholderText("e.g., -1 2 -3 (using surface IDs)")

        form_layout.addRow("Cell Name:", self.name_input)
        form_layout.addRow("Fill Material:", self.material_combo)
        form_layout.addRow("Region Definition:", self.region_input)

        layout.addLayout(form_layout)

        # Help text for region syntax
        help_text = QLabel(
            "<small><i>Tip: Use negative numbers for 'inside' and positive for 'outside' a surface.</i></small>")
        help_text.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(help_text)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_cell_data(self) -> dict:
        """Returns a dictionary of the configured cell properties."""
        return {
            "name": self.name_input.text(),
            "material": self.material_combo.currentText(),
            "region": self.region_input.text()
        }