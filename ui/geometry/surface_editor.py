import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QComboBox, QLineEdit,
    QFormLayout, QStackedWidget, QWidget, QDialogButtonBox, QLabel
)
from PySide6.QtCore import Qt

logger = logging.getLogger("OpenMC Studio")


class SurfaceEditorDialog(QDialog):
    """
    Dialog to create and edit OpenMC surfaces.
    Features dynamic input fields based on the selected surface type.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Surface Editor")
        self.setMinimumWidth(400)
        self._setup_ui()
        logger.debug("SurfaceEditorDialog initialized.")

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 1. Common Properties
        form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Reactor Vessel Outer")

        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Sphere", "X-Plane", "Y-Plane", "Z-Plane",
            "X-Cylinder", "Y-Cylinder", "Z-Cylinder"
        ])
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)

        self.bc_combo = QComboBox()
        self.bc_combo.addItems(["transmission", "vacuum", "reflective", "periodic"])

        form_layout.addRow("Surface Name:", self.name_input)
        form_layout.addRow("Surface Type:", self.type_combo)
        form_layout.addRow("Boundary Condition:", self.bc_combo)
        layout.addLayout(form_layout)

        # 2. Dynamic Parameters using QStackedWidget
        self.stacked_params = QStackedWidget()

        # --- Sphere Parameters ---
        sphere_widget = QWidget()
        sphere_layout = QFormLayout(sphere_widget)
        self.sphere_r = QLineEdit("10.0")
        self.sphere_x0 = QLineEdit("0.0")
        self.sphere_y0 = QLineEdit("0.0")
        self.sphere_z0 = QLineEdit("0.0")
        sphere_layout.addRow("Radius (R):", self.sphere_r)
        sphere_layout.addRow("Center X (x0):", self.sphere_x0)
        sphere_layout.addRow("Center Y (y0):", self.sphere_y0)
        sphere_layout.addRow("Center Z (z0):", self.sphere_z0)
        self.stacked_params.addWidget(sphere_widget)

        # --- Plane Parameters ---
        plane_widget = QWidget()
        plane_layout = QFormLayout(plane_widget)
        self.plane_d = QLineEdit("0.0")
        plane_layout.addRow("Intercept (Location):", self.plane_d)
        self.stacked_params.addWidget(plane_widget)

        # --- Cylinder Parameters ---
        cylinder_widget = QWidget()
        cylinder_layout = QFormLayout(cylinder_widget)
        self.cyl_r = QLineEdit("10.0")
        self.cyl_x0 = QLineEdit("0.0")
        self.cyl_y0 = QLineEdit("0.0")
        cylinder_layout.addRow("Radius (R):", self.cyl_r)
        cylinder_layout.addRow("Center Coords 1:", self.cyl_x0)
        cylinder_layout.addRow("Center Coords 2:", self.cyl_y0)
        self.stacked_params.addWidget(cylinder_widget)

        layout.addWidget(QLabel("<b>Surface Parameters</b>"))
        layout.addWidget(self.stacked_params)

        # 3. Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Trigger initial view
        self._on_type_changed(0)

    def _on_type_changed(self, index):
        """Switches the visible parameters based on surface type."""
        text = self.type_combo.currentText()
        if text == "Sphere":
            self.stacked_params.setCurrentIndex(0)
        elif "Plane" in text:
            self.stacked_params.setCurrentIndex(1)
        elif "Cylinder" in text:
            self.stacked_params.setCurrentIndex(2)

    def get_surface_data(self) -> dict:
        """Returns a dictionary of the configured surface properties."""
        return {
            "name": self.name_input.text(),
            "type": self.type_combo.currentText(),
            "boundary": self.bc_combo.currentText()
            # Advanced parameter extraction will be linked to core models later
        }