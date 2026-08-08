import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QMenu,
    QToolBar, QApplication, QStyle
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QAction, QIcon
from PySide6.QtCore import Qt, QSize

logger = logging.getLogger("OpenMC Studio")


class ProjectTreeWidget(QWidget):
    """
    An interactive Project Tree widget representing the OpenMC simulation structure.
    Uses QTreeView and QStandardItemModel to strictly separate data from presentation (MVC).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_model()
        self._populate_default_structure()

        logger.debug("ProjectTreeWidget initialized.")

    def _setup_ui(self):
        """Sets up the layout, toolbar, and tree view."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Local Toolbar for Tree Actions
        self.toolbar = QToolBar("Tree Toolbar")
        self.toolbar.setIconSize(QSize(16, 16))

        style = QApplication.style()

        self.action_expand = QAction(style.standardIcon(QStyle.StandardPixmap.SP_ToolBarVerticalExtensionButton),
                                     "Expand All", self)
        self.action_expand.triggered.connect(self.expand_all)

        self.action_collapse = QAction(style.standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton), "Collapse All",
                                       self)
        self.action_collapse.triggered.connect(self.collapse_all)

        self.toolbar.addAction(self.action_expand)
        self.toolbar.addAction(self.action_collapse)

        self.layout.addWidget(self.toolbar)

        # Tree View setup
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setSelectionMode(QTreeView.SelectionMode.SingleSelection)
        self.tree_view.setEditTriggers(QTreeView.EditTrigger.NoEditTriggers)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._show_context_menu)

        self.layout.addWidget(self.tree_view)

    def _setup_model(self):
        """Initializes the standard item model for the tree."""
        self.model = QStandardItemModel()
        self.tree_view.setModel(self.model)

    def _create_item(self, text: str,
                     icon_type: QStyle.StandardPixmap = QStyle.StandardPixmap.SP_DirIcon) -> QStandardItem:
        """Helper method to create a standardized tree item."""
        item = QStandardItem(text)
        item.setIcon(QApplication.style().standardIcon(icon_type))
        item.setEditable(False)
        return item

    def _populate_default_structure(self):
        """Populates the tree with the standard OpenMC XML module structure."""
        # Root Node
        self.root_node = self._create_item("Simulation Project", QStyle.StandardPixmap.SP_ComputerIcon)
        self.model.appendRow(self.root_node)

        # Primary OpenMC Modules
        self.geom_node = self._create_item("Geometry", QStyle.StandardPixmap.SP_DirIcon)
        self.mat_node = self._create_item("Materials", QStyle.StandardPixmap.SP_DirIcon)
        self.settings_node = self._create_item("Settings", QStyle.StandardPixmap.SP_DirIcon)
        self.tallies_node = self._create_item("Tallies", QStyle.StandardPixmap.SP_DirIcon)
        self.plots_node = self._create_item("Plots", QStyle.StandardPixmap.SP_DirIcon)

        self.root_node.appendRow(self.geom_node)
        self.root_node.appendRow(self.mat_node)
        self.root_node.appendRow(self.settings_node)
        self.root_node.appendRow(self.tallies_node)
        self.root_node.appendRow(self.plots_node)

        # Expand the root by default
        self.tree_view.expand(self.root_node.index())

    def _show_context_menu(self, position):
        """Displays a context menu based on the selected item."""
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return

        item = self.model.itemFromIndex(index)
        menu = QMenu()

        if item.text() == "Geometry":
            menu.addAction("Add Cell")
            menu.addAction("Add Surface")
        elif item.text() == "Materials":
            menu.addAction("Add Material")
        else:
            menu.addAction("Properties")

        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def expand_all(self):
        self.tree_view.expandAll()

    def collapse_all(self):
        self.tree_view.collapseAll()