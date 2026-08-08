import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QDialogButtonBox, QLabel
)

logger = logging.getLogger("OpenMC Studio")


class UniverseEditorDialog(QDialog):
    """
    Dialog to create and edit OpenMC universes.
    Used to group cells together for complex geometries or lattices.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Universe Editor")
        self.setMinimumWidth(350)
        self._setup_ui()
        logger.debug("UniverseEditorDialog initialized.")

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Fuel Assembly, Core")

        form_layout.addRow("Universe Name:", self.name_input)
        layout.addLayout(form_layout)

        # Help text
        help_text = QLabel(
            "<small><i>Universes are used to group multiple cells and can be filled into other cells or lattices.</i></small>")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(help_text)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_universe_data(self) -> dict:
        """Returns a dictionary of the configured universe properties."""
        return {
            "name": self.name_input.text()
        }