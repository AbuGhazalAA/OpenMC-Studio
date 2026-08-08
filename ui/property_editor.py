import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QHeaderView
)
from PySide6.QtCore import Qt

logger = logging.getLogger("OpenMC Studio")


class PropertyEditorWidget(QWidget):
    """
    A dynamic property editor that updates based on the selected object
    in the Project Tree. Mimics professional CAD property grids.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_default_view()

        logger.debug("PropertyEditorWidget initialized.")

    def _setup_ui(self):
        """Sets up the property grid layout and styling."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Header label indicating current selection
        self.header_label = QLabel("No Object Selected")
        self.header_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #ffffff;
                font-weight: bold;
                padding: 6px;
                border-bottom: 1px solid #1e1e1e;
            }
        """)
        self.layout.addWidget(self.header_label)

        # Property Grid (TreeWidget)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Property", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(15)

        # Configure headers for professional look
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.layout.addWidget(self.tree)

    def _setup_default_view(self):
        """Shows a placeholder view when nothing is selected."""
        self.clear_properties()
        item = QTreeWidgetItem(["Status", "Waiting for selection..."])
        item.setDisabled(True)
        self.tree.addTopLevelItem(item)

    def clear_properties(self):
        """Clears all current properties from the grid."""
        self.tree.clear()
        self.header_label.setText("No Object Selected")

    def load_properties(self, object_name: str, properties: dict):
        """
        Dynamically populates the property grid based on a dictionary of properties.
        Later, this will be connected to the OpenMC core data models.
        """
        self.clear_properties()
        self.header_label.setText(f"Properties: {object_name}")

        for key, value in properties.items():
            item = QTreeWidgetItem(self.tree)
            item.setText(0, key)
            item.setText(1, str(value))

            # Make the value column editable
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)

        self.tree.expandAll()