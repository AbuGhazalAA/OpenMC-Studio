import os
import re
import time
import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QListWidget, QGroupBox, QSplitter, QFormLayout,
    QTabWidget, QStackedWidget, QPlainTextEdit
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QPixmap


class GeometryPageWidget(QWidget):
    script_generated = Signal(str)
    log_signal = Signal(str)

    # كل أنواع الأسطح المدعومة في OpenMC مع أسماء ووسائط الدوال البانية
    # لكل نوع (key, label, default). الترتيب هنا هو نفس ترتيب القائمة
    # المنسدلة وترتيب صفحات QStackedWidget (نفس الفهرس).
    SURFACE_TYPES = {
        "Sphere": {"ctor": "Sphere", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"),
            ("z0", "z0 (cm)", "0.0"), ("r", "Radius r (cm)", "10.0")]},
        "X-Cylinder": {"ctor": "XCylinder", "params": [
            ("y0", "y0 (cm)", "0.0"), ("z0", "z0 (cm)", "0.0"),
            ("r", "Radius r (cm)", "10.0")]},
        "Y-Cylinder": {"ctor": "YCylinder", "params": [
            ("x0", "x0 (cm)", "0.0"), ("z0", "z0 (cm)", "0.0"),
            ("r", "Radius r (cm)", "10.0")]},
        "Z-Cylinder": {"ctor": "ZCylinder", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"),
            ("r", "Radius r (cm)", "10.0")]},
        "X-Plane": {"ctor": "XPlane", "params": [("x0", "x0 (cm)", "0.0")]},
        "Y-Plane": {"ctor": "YPlane", "params": [("y0", "y0 (cm)", "0.0")]},
        "Z-Plane": {"ctor": "ZPlane", "params": [("z0", "z0 (cm)", "0.0")]},
        "Plane (general)": {"ctor": "Plane", "params": [
            ("a", "A", "1.0"), ("b", "B", "0.0"), ("c", "C", "0.0"), ("d", "D", "0.0")]},
        "X-Cone": {"ctor": "XCone", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"),
            ("z0", "z0 (cm)", "0.0"), ("r2", "r² (slope²)", "1.0")]},
        "Y-Cone": {"ctor": "YCone", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"),
            ("z0", "z0 (cm)", "0.0"), ("r2", "r² (slope²)", "1.0")]},
        "Z-Cone": {"ctor": "ZCone", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"),
            ("z0", "z0 (cm)", "0.0"), ("r2", "r² (slope²)", "1.0")]},
        "X-Torus": {"ctor": "XTorus", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"), ("z0", "z0 (cm)", "0.0"),
            ("a", "Major radius A (cm)", "10.0"),
            ("b", "Minor radius B (cm)", "2.0"), ("c", "Minor radius C (cm)", "2.0")]},
        "Y-Torus": {"ctor": "YTorus", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"), ("z0", "z0 (cm)", "0.0"),
            ("a", "Major radius A (cm)", "10.0"),
            ("b", "Minor radius B (cm)", "2.0"), ("c", "Minor radius C (cm)", "2.0")]},
        "Z-Torus": {"ctor": "ZTorus", "params": [
            ("x0", "x0 (cm)", "0.0"), ("y0", "y0 (cm)", "0.0"), ("z0", "z0 (cm)", "0.0"),
            ("a", "Major radius A (cm)", "10.0"),
            ("b", "Minor radius B (cm)", "2.0"), ("c", "Minor radius C (cm)", "2.0")]},
        "Quadric (general 2nd-order)": {"ctor": "Quadric", "params": [
            ("a", "A", "1.0"), ("b", "B", "1.0"), ("c", "C", "1.0"), ("d", "D", "0.0"),
            ("e", "E", "0.0"), ("f", "F", "0.0"), ("g", "G", "0.0"), ("h", "H", "0.0"),
            ("j", "J", "0.0"), ("k", "K", "-100.0")]},
    }

    # أنواع تعبئة الخلايا (Cell.fill): فراغ / مادة / كون متداخل / شبكة متداخلة
    FILL_MODES = ["Void", "Material", "Universe (nested)", "Lattice (nested)"]
    FILL_MODE_KEYS = {
        "Void": "void", "Material": "material",
        "Universe (nested)": "universe", "Lattice (nested)": "lattice",
    }

    def __init__(self, project_manager=None, materials_page=None, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.materials_page = materials_page

        self.surfaces_data = []
        self.cells_data = []
        self.lattices_data = []

        self.history_surf = []
        self.history_cell = []
        self.history_lat = []

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.tab_surfaces = QWidget()
        self._setup_surfaces_tab()
        self.tabs.addTab(self.tab_surfaces, "Surfaces")

        self.tab_cells = QWidget()
        self._setup_cells_tab()
        self.tabs.addTab(self.tab_cells, "Cells")

        self.tab_lattices = QWidget()
        self._setup_lattices_tab()
        self.tabs.addTab(self.tab_lattices, "Lattices (Advanced)")

        left_layout.addWidget(self.tabs)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.lbl_viewer = QLabel("🎨 Add at least one Cell to see the live preview.")
        self.lbl_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_viewer.setStyleSheet("background-color: #ffffff; border: 2px dashed #aaaaaa; color: #555555;")
        self.lbl_viewer.setMinimumSize(400, 400)
        right_layout.addWidget(self.lbl_viewer)
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter)

        # تعبئة أولية لقائمة المواد إن كان محرر السكريبت موجوداً فعلاً
        self.refresh_materials()

    # ==========================================
    # --- Surfaces Tab ---
    # ==========================================
    def _setup_surfaces_tab(self):
        layout = QVBoxLayout(self.tab_surfaces)

        surf_group = QGroupBox("Define CSG Surface")
        form = QFormLayout(surf_group)

        self.surf_name = QLineEdit("s1")
        self.surf_type = QComboBox()
        self.surf_type.addItems(list(self.SURFACE_TYPES.keys()))
        self.surf_type.currentTextChanged.connect(self._on_surf_type_changed)
        self.surf_boundary = QComboBox()
        self.surf_boundary.addItems(["transmission", "vacuum", "reflective"])

        form.addRow("Name:", self.surf_name)
        form.addRow("Type:", self.surf_type)

        self.surf_param_stack = QStackedWidget()
        self.surf_param_widgets = {}
        for type_name, spec in self.SURFACE_TYPES.items():
            page = QWidget()
            page_form = QFormLayout(page)
            page_form.setContentsMargins(0, 0, 0, 0)
            field_map = {}
            for key, label, default in spec["params"]:
                edit = QLineEdit(default)
                page_form.addRow(f"{label}:", edit)
                field_map[key] = edit
            self.surf_param_widgets[type_name] = field_map
            self.surf_param_stack.addWidget(page)
        form.addRow("Parameters:", self.surf_param_stack)

        form.addRow("Boundary Type:", self.surf_boundary)
        layout.addWidget(surf_group)

        btn_layout = QHBoxLayout()
        self.btn_add_surf = QPushButton("➕ Add Surface")
        self.btn_add_surf.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold;")
        self.btn_add_surf.clicked.connect(self.add_surface)

        self.btn_undo_surf = QPushButton("↩️ Undo")
        self.btn_undo_surf.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold;")
        self.btn_undo_surf.clicked.connect(self.undo_surface)

        btn_layout.addWidget(self.btn_add_surf)
        btn_layout.addWidget(self.btn_undo_surf)
        layout.addLayout(btn_layout)

        layout.addWidget(QLabel("Created Surfaces:"))
        self.surf_list = QListWidget()
        layout.addWidget(self.surf_list)

        self.btn_del_surf = QPushButton("🗑️ Delete Selected")
        self.btn_del_surf.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_del_surf.clicked.connect(self.delete_surface)
        layout.addWidget(self.btn_del_surf)

    def _on_surf_type_changed(self, text):
        names = list(self.SURFACE_TYPES.keys())
        if text in names:
            self.surf_param_stack.setCurrentIndex(names.index(text))

    # ==========================================
    # --- Cells Tab ---
    # ==========================================
    def _setup_cells_tab(self):
        layout = QVBoxLayout(self.tab_cells)

        cell_group = QGroupBox("Define Cell")
        form = QFormLayout(cell_group)

        self.cell_name = QLineEdit("c1")

        self.fill_mode_combo = QComboBox()
        self.fill_mode_combo.addItems(self.FILL_MODES)
        self.fill_mode_combo.currentTextChanged.connect(self._on_fill_mode_changed)
        form.addRow("Fill Type:", self.fill_mode_combo)

        self.fill_stack = QStackedWidget()

        # 0: Void -- لا تعبئة
        self.fill_stack.addWidget(QLabel("Void region (no material)."))

        # 1: Material -- تعبئة بمادة من صفحة Materials / السكريبت
        mat_widget = QWidget()
        mat_layout = QHBoxLayout(mat_widget)
        mat_layout.setContentsMargins(0, 0, 0, 0)
        self.cell_material = QComboBox()
        self.btn_refresh_mats = QPushButton("🔄")
        self.btn_refresh_mats.setToolTip("Refresh Materials List")
        self.btn_refresh_mats.setFixedWidth(35)
        self.btn_refresh_mats.clicked.connect(self.refresh_materials)
        mat_layout.addWidget(self.cell_material, stretch=1)
        mat_layout.addWidget(self.btn_refresh_mats)
        self.fill_stack.addWidget(mat_widget)

        # 2: Universe (nested) -- تعبئة بكون متداخل (اسم كون تم استخدامه في حقل "Belongs to Universe")
        self.cell_universe_fill_combo = QComboBox()
        self.cell_universe_fill_combo.setEditable(True)
        self.cell_universe_fill_combo.setToolTip(
            "Name of an existing Universe (see other cells' 'Belongs to Universe' field) to nest inside this cell.")
        self.fill_stack.addWidget(self.cell_universe_fill_combo)

        # 3: Lattice (nested) -- تعبئة بشبكة (Lattice) من تبويب Lattices
        self.cell_lattice_fill_combo = QComboBox()
        self.cell_lattice_fill_combo.setEditable(True)
        self.cell_lattice_fill_combo.setToolTip(
            "Name of a Lattice defined in the 'Lattices (Advanced)' tab to nest inside this cell.")
        self.fill_stack.addWidget(self.cell_lattice_fill_combo)

        form.addRow("Fill Value:", self.fill_stack)

        # ==========================================
        # --- أداة بناء الـ CSG الذكية ---
        # ==========================================
        csg_builder_layout = QVBoxLayout()
        csg_builder_layout.setSpacing(5)

        surf_layout = QHBoxLayout()
        self.combo_available_surfaces = QComboBox()
        self.combo_available_surfaces.setToolTip("Select a surface")

        btn_insert_surf = QPushButton("↙️ Insert Surface")
        btn_insert_surf.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        btn_insert_surf.clicked.connect(self._insert_surface_to_region)

        surf_layout.addWidget(self.combo_available_surfaces, stretch=1)
        surf_layout.addWidget(btn_insert_surf)
        csg_builder_layout.addLayout(surf_layout)

        op_layout = QHBoxLayout()
        self.combo_operators = QComboBox()
        self.combo_operators.addItems([
            "- (Inside)",
            "+ (Outside)",
            "& (AND / Intersection)",
            "| (OR / Union)",
            "~ (NOT / Complement)",
            "( (Open Bracket)",
            ") (Close Bracket)"
        ])

        btn_insert_op = QPushButton("↙️ Insert Operator")
        btn_insert_op.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold;")
        btn_insert_op.clicked.connect(self._insert_operator_to_region)

        op_layout.addWidget(self.combo_operators, stretch=1)
        op_layout.addWidget(btn_insert_op)
        csg_builder_layout.addLayout(op_layout)
        # ==========================================

        self.cell_region = QLineEdit()
        self.cell_region.setPlaceholderText("e.g. -s1 & +s2 | ~s3")
        self.cell_region.setText("")

        self.cell_universe = QLineEdit("root")

        form.addRow("Cell Name:", self.cell_name)
        form.addRow("CSG Builder:", csg_builder_layout)
        form.addRow("Region (CSG):", self.cell_region)
        form.addRow("Belongs to Universe (container):", self.cell_universe)
        layout.addWidget(cell_group)

        btn_layout = QHBoxLayout()
        self.btn_add_cell = QPushButton("➕ Add Cell")
        self.btn_add_cell.setStyleSheet("background-color: #217346; color: white; font-weight: bold;")
        self.btn_add_cell.clicked.connect(self.add_cell)

        self.btn_undo_cell = QPushButton("↩️ Undo")
        self.btn_undo_cell.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold;")
        self.btn_undo_cell.clicked.connect(self.undo_cell)

        btn_layout.addWidget(self.btn_add_cell)
        btn_layout.addWidget(self.btn_undo_cell)
        layout.addLayout(btn_layout)

        layout.addWidget(QLabel("Created Cells:"))
        self.cell_list = QListWidget()
        layout.addWidget(self.cell_list)

        self.btn_del_cell = QPushButton("🗑️ Delete Selected")
        self.btn_del_cell.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_del_cell.clicked.connect(self.delete_cell)
        layout.addWidget(self.btn_del_cell)

    def _on_fill_mode_changed(self, text):
        idx = self.FILL_MODES.index(text) if text in self.FILL_MODES else 0
        self.fill_stack.setCurrentIndex(idx)
        if text == "Material":
            self.refresh_materials()
        elif text == "Universe (nested)":
            self.refresh_universes_combo()
        elif text == "Lattice (nested)":
            self.refresh_lattices_combo()

    def _insert_surface_to_region(self):
        surface_name = self.combo_available_surfaces.currentText()
        if surface_name:
            self.cell_region.insert(surface_name)
            self.cell_region.setFocus()

    def _insert_operator_to_region(self):
        op_text = self.combo_operators.currentText()
        symbol = op_text.split(" ")[0]

        if symbol in ["&", "|", "~"]:
            symbol = f" {symbol} "

        self.cell_region.insert(symbol)
        self.cell_region.setFocus()

    # ==========================================
    # --- Lattices Tab ---
    # ==========================================
    def _setup_lattices_tab(self):
        layout = QVBoxLayout(self.tab_lattices)

        lat_group = QGroupBox("Define Lattice (RectLattice / HexLattice) of existing Universes")
        form = QFormLayout(lat_group)

        self.lat_name = QLineEdit("lat1")
        self.lat_kind = QComboBox()
        self.lat_kind.addItems(["Rectangular (RectLattice)", "Hexagonal (HexLattice)"])
        self.lat_kind.currentTextChanged.connect(self._on_lat_kind_changed)

        self.lat_pitch = QLineEdit("2.0, 2.0")
        self.lat_pitch.setToolTip(
            "Comma-separated pitch per axis, e.g. '2.0, 2.0' for a 2D RectLattice, or '2.0' for a 2D HexLattice.")
        self.lat_dimension = QLineEdit("3, 3")
        self.lat_dimension.setToolTip("RectLattice only (informational): number of elements per axis, e.g. '3, 3'.")
        self.lat_lower_left = QLineEdit("-3.0, -3.0")
        self.lat_lower_left.setToolTip("RectLattice only: coordinates of the lower-left corner of the lattice.")
        self.lat_center = QLineEdit("0.0, 0.0")
        self.lat_center.setToolTip("HexLattice only: coordinates of the center of the lattice.")
        self.lat_orientation = QComboBox()
        self.lat_orientation.addItems(["y", "x"])
        self.lat_num_rings = QLineEdit("3")
        self.lat_num_rings.setToolTip(
            "HexLattice only: number of hexagonal rings, center ring included (used only by "
            "'Generate Ring Template' below -- OpenMC itself infers ring count from the universes list).")
        self.lat_outer = QLineEdit("")
        self.lat_outer.setPlaceholderText("(optional) name of an outer Universe to fill space beyond the lattice")

        self.lat_universes = QPlainTextEdit()
        self.lat_universes.setPlaceholderText(
            "Python nested-list of Universe names, row-major (must match Belongs-to-Universe names used in Cells).\n"
            "RectLattice 2D example (3x3):\n"
            "[[u1, u2, u3],\n [u4, u5, u6],\n [u7, u8, u9]]\n\n"
            "HexLattice example (orientation='y', 2 rings):\n"
            "[[u2, u3, u4, u5, u6, u7], [u1]]"
        )
        self.lat_universes.setFixedHeight(120)

        btn_template = QPushButton("🧩 Generate Ring/Grid Template")
        btn_template.setToolTip(
            "Fills the Universes Grid above with correctly-sized placeholder names (u1, u2, ...) for "
            "the current Dimension (RectLattice) or Number of Rings (HexLattice) -- computing the "
            "right hex-ring sizes (1, 6, 12, 18, ... elements per ring) by hand is a common source of "
            "'universes list has wrong length' errors. Just replace the placeholder names with your "
            "real Universe names afterward.")
        btn_template.clicked.connect(self.generate_lattice_template)

        form.addRow("Lattice Name:", self.lat_name)
        form.addRow("Kind:", self.lat_kind)
        form.addRow("Pitch (cm):", self.lat_pitch)
        form.addRow("Dimension (Nx,Ny[,Nz]):", self.lat_dimension)
        form.addRow("Lower-Left (RectLattice):", self.lat_lower_left)
        form.addRow("Center (HexLattice):", self.lat_center)
        form.addRow("Orientation (HexLattice):", self.lat_orientation)
        form.addRow("Number of Rings (HexLattice):", self.lat_num_rings)
        form.addRow("Outer Universe (optional):", self.lat_outer)
        form.addRow(btn_template)
        form.addRow("Universes Grid (Python list):", self.lat_universes)
        layout.addWidget(lat_group)

        btn_layout = QHBoxLayout()
        self.btn_add_lat = QPushButton("➕ Add Lattice")
        self.btn_add_lat.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold;")
        self.btn_add_lat.clicked.connect(self.add_lattice)

        self.btn_undo_lat = QPushButton("↩️ Undo")
        self.btn_undo_lat.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold;")
        self.btn_undo_lat.clicked.connect(self.undo_lattice)

        btn_layout.addWidget(self.btn_add_lat)
        btn_layout.addWidget(self.btn_undo_lat)
        layout.addLayout(btn_layout)

        layout.addWidget(QLabel("Created Lattices:"))
        self.lat_list = QListWidget()
        layout.addWidget(self.lat_list)

        self.btn_del_lat = QPushButton("🗑️ Delete Selected")
        self.btn_del_lat.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_del_lat.clicked.connect(self.delete_lattice)
        layout.addWidget(self.btn_del_lat)

        self._on_lat_kind_changed(self.lat_kind.currentText())

    def _on_lat_kind_changed(self, text):
        is_rect = text.startswith("Rectangular")
        self.lat_dimension.setEnabled(is_rect)
        self.lat_lower_left.setEnabled(is_rect)
        self.lat_center.setEnabled(not is_rect)
        self.lat_orientation.setEnabled(not is_rect)
        self.lat_num_rings.setEnabled(not is_rect)

    def generate_lattice_template(self):
        """Fills the Universes Grid textbox with correctly-sized
        placeholder Universe names, sparing the user from hand-computing
        hex-ring sizes (1, 6, 12, 18, ... per ring, outermost ring first)
        -- a routine source of 'universes list has the wrong length'
        errors that only surface once the script actually runs."""
        kind = self.lat_kind.currentText()
        counter = [0]

        def next_name():
            counter[0] += 1
            return f"u{counter[0]}"

        if kind.startswith("Rectangular"):
            dim_txt = self.lat_dimension.text().strip()
            try:
                dims = [int(float(x)) for x in dim_txt.split(",")]
            except ValueError:
                self.log_signal.emit(f"❌ Lattice Error: Invalid Dimension '{dim_txt}' for template generation.")
                return
            nx = dims[0] if len(dims) > 0 else 0
            ny = dims[1] if len(dims) > 1 else 1
            if nx <= 0 or ny <= 0:
                self.log_signal.emit("❌ Lattice Error: Dimension values must be positive integers.")
                return
            rows = [[next_name() for _ in range(nx)] for _ in range(ny)]
        else:
            try:
                num_rings = int(self.lat_num_rings.text().strip())
            except ValueError:
                self.log_signal.emit(
                    f"❌ Lattice Error: Invalid Number of Rings '{self.lat_num_rings.text()}'.")
                return
            if num_rings <= 0:
                self.log_signal.emit("❌ Lattice Error: Number of Rings must be a positive integer.")
                return
            # OpenMC's HexLattice.universes is ordered OUTERMOST ring
            # first, innermost (center, always exactly 1 element) last.
            # Ring sizes going outward from center are 1, 6, 12, 18, ...
            # i.e. 6*k for the k-th ring out (k=0 at center) -- so listed
            # outermost-first for N rings: 6*(N-1), 6*(N-2), ..., 6, 1.
            rows = []
            for ring_from_outside in range(num_rings):
                ring_from_center = num_rings - 1 - ring_from_outside
                size = 6 * ring_from_center if ring_from_center > 0 else 1
                rows.append([next_name() for _ in range(size)])

        text = "[\n" + ",\n".join("  [" + ", ".join(row) + "]" for row in rows) + "\n]"
        self.lat_universes.setPlainText(text)
        total = counter[0]
        self.log_signal.emit(
            f"✅ Generated a {kind.split(' ')[0].lower()} template with {len(rows)} row(s), "
            f"{total} placeholder Universe name(s) (u1..u{total}) -- replace them with real Universe names.")

    def add_lattice(self):
        name = self.lat_name.text().strip()
        kind = "rect" if self.lat_kind.currentText().startswith("Rectangular") else "hex"
        pitch_txt = self.lat_pitch.text().strip()
        universes_txt = self.lat_universes.toPlainText().strip()
        outer_txt = self.lat_outer.text().strip()

        if not name:
            self.log_signal.emit("❌ Lattice Error: Lattice name cannot be empty!")
            return
        if name in [l['name'] for l in self.lattices_data]:
            self.log_signal.emit(f"❌ Lattice Error: Lattice '{name}' already exists!")
            return
        if not pitch_txt:
            self.log_signal.emit("❌ Lattice Error: Pitch cannot be empty!")
            return
        try:
            [float(x) for x in pitch_txt.split(",")]
        except ValueError:
            self.log_signal.emit(f"❌ Lattice Error: Invalid Pitch values '{pitch_txt}'!")
            return
        if not universes_txt:
            self.log_signal.emit("❌ Lattice Error: Universes Grid cannot be empty!")
            return

        lat_info = {
            'name': name, 'kind': kind, 'pitch': pitch_txt,
            'universes': universes_txt, 'outer': outer_txt,
        }

        if kind == 'rect':
            dim_txt = self.lat_dimension.text().strip()
            ll_txt = self.lat_lower_left.text().strip()
            if not ll_txt:
                self.log_signal.emit("❌ Lattice Error: Lower-Left cannot be empty for a RectLattice!")
                return
            try:
                [float(x) for x in ll_txt.split(",")]
            except ValueError:
                self.log_signal.emit(f"❌ Lattice Error: Invalid Lower-Left values '{ll_txt}'!")
                return
            lat_info['dimension'] = dim_txt
            lat_info['lower_left'] = ll_txt
        else:
            center_txt = self.lat_center.text().strip()
            if not center_txt:
                self.log_signal.emit("❌ Lattice Error: Center cannot be empty for a HexLattice!")
                return
            try:
                [float(x) for x in center_txt.split(",")]
            except ValueError:
                self.log_signal.emit(f"❌ Lattice Error: Invalid Center values '{center_txt}'!")
                return
            lat_info['center'] = center_txt
            lat_info['orientation'] = self.lat_orientation.currentText()

        self.lattices_data.append(lat_info)
        self.history_lat.append(("add", lat_info))
        self._refresh_lat_list()
        self.log_signal.emit(f"✅ Success: Added Lattice '{name}'.")
        self.refresh_lattices_combo()
        self.generate_script()

    def undo_lattice(self):
        if not self.history_lat:
            return
        entry = self.history_lat.pop()
        action = entry[0]
        if action == "add":
            data = entry[1]
            self.lattices_data.remove(data)
            self._refresh_lat_list()
            self.log_signal.emit("↩️ Undo: Reverted last lattice.")
        elif action == "delete":
            _, data, row = entry
            insert_row = min(row, len(self.lattices_data))
            self.lattices_data.insert(insert_row, data)
            self._refresh_lat_list()
            self.log_signal.emit(f"↩️ Undo: Restored lattice '{data['name']}'.")
        self.refresh_lattices_combo()
        self.generate_script()

    def delete_lattice(self):
        row = self.lat_list.currentRow()
        if row >= 0:
            data = self.lattices_data.pop(row)
            self._refresh_lat_list()
            self.history_lat.append(("delete", data, row))
            self.log_signal.emit(f"🗑️ Deleted: Lattice '{data['name']}'.")
            self.refresh_lattices_combo()
            self.generate_script()

    def _refresh_lat_list(self):
        self.lat_list.clear()
        for l in self.lattices_data:
            self.lat_list.addItem(
                f"{l['name']} | {l['kind']} | pitch=({l['pitch']}) | outer={l['outer'] or '-'}")

    def _lattice_code(self, lat):
        code = ""
        if lat['kind'] == 'rect':
            code += f"{lat['name']} = openmc.RectLattice(name='{lat['name']}')\n"
            code += f"{lat['name']}.lower_left = ({lat['lower_left']})\n"
            code += f"{lat['name']}.pitch = ({lat['pitch']})\n"
        else:
            code += f"{lat['name']} = openmc.HexLattice(name='{lat['name']}')\n"
            code += f"{lat['name']}.center = ({lat['center']})\n"
            code += f"{lat['name']}.pitch = ({lat['pitch']})\n"
            code += f"{lat['name']}.orientation = '{lat['orientation']}'\n"
        code += f"{lat['name']}.universes = {lat['universes']}\n"
        if lat['outer']:
            code += f"{lat['name']}.outer = {lat['outer']}\n"
        return code

    # ==========================================
    # --- Shared helpers ---
    # ==========================================
    def _on_tab_changed(self, index):
        if index == 1:
            self.refresh_materials()
            self.refresh_universes_combo()
            self.refresh_lattices_combo()

    def refresh_materials(self):
        current_mat = self.cell_material.currentText()
        self.cell_material.clear()

        try:
            from PySide6.QtWidgets import QApplication
            full_code = ""

            for widget in QApplication.topLevelWidgets():
                if hasattr(widget, 'script_editor'):
                    full_code = widget.script_editor.editor.toPlainText()
                    break

            if full_code:
                mat_vars = re.findall(r"([a-zA-Z0-9_]+)\s*=\s*openmc\.Material", full_code)
                for mat in sorted(set(mat_vars)):
                    self.cell_material.addItem(mat)

        except Exception as e:
            self.log_signal.emit(f"⚠️ Warning: Could not parse materials from script. ({e})")

        index = self.cell_material.findText(current_mat)
        if index >= 0:
            self.cell_material.setCurrentIndex(index)
        elif current_mat:
            self.cell_material.setEditText(current_mat)

    def refresh_universes_combo(self):
        current = self.cell_universe_fill_combo.currentText()
        self.cell_universe_fill_combo.clear()
        names = sorted(set(c['universe'] for c in self.cells_data if c.get('universe')))
        self.cell_universe_fill_combo.addItems(names)
        idx = self.cell_universe_fill_combo.findText(current)
        if idx >= 0:
            self.cell_universe_fill_combo.setCurrentIndex(idx)
        elif current:
            self.cell_universe_fill_combo.setEditText(current)

    def refresh_lattices_combo(self):
        current = self.cell_lattice_fill_combo.currentText()
        self.cell_lattice_fill_combo.clear()
        self.cell_lattice_fill_combo.addItems([l['name'] for l in self.lattices_data])
        idx = self.cell_lattice_fill_combo.findText(current)
        if idx >= 0:
            self.cell_lattice_fill_combo.setCurrentIndex(idx)
        elif current:
            self.cell_lattice_fill_combo.setEditText(current)

    # ==========================================
    # --- Surfaces logic ---
    # ==========================================
    def _validate_surface_params(self, stype, values):
        """Surface types whose geometric meaning breaks for a non-positive
        value (radii, torus semi-axes) need an explicit check -- OpenMC will
        happily build the object with e.g. r=-5 and only fail confusingly
        much later, during plotting or the actual run."""
        positive_keys = []
        if stype in ("Sphere", "X-Cylinder", "Y-Cylinder", "Z-Cylinder"):
            positive_keys = ["r"]
        elif stype in ("X-Cone", "Y-Cone", "Z-Cone"):
            positive_keys = ["r2"]
        elif stype in ("X-Torus", "Y-Torus", "Z-Torus"):
            positive_keys = ["a", "b", "c"]
        for k in positive_keys:
            if values.get(k, 0.0) <= 0:
                return f"'{k}' must be a positive number (> 0)!"
        return None

    def add_surface(self):
        name = self.surf_name.text().strip()
        stype = self.surf_type.currentText()
        btype = self.surf_boundary.currentText()

        if not name:
            self.log_signal.emit("❌ Geometry Error: Surface name cannot be empty!")
            return
        if name in [s['name'] for s in self.surfaces_data]:
            self.log_signal.emit(f"❌ Geometry Error: Surface '{name}' already exists!")
            return

        field_map = self.surf_param_widgets[stype]
        params = {}
        for key, edit in field_map.items():
            txt = edit.text().strip()
            try:
                params[key] = float(txt)
            except ValueError:
                self.log_signal.emit(f"❌ Geometry Error: Invalid number '{txt}' for parameter '{key}'!")
                return

        err = self._validate_surface_params(stype, params)
        if err:
            self.log_signal.emit(f"❌ Geometry Error: {err}")
            return

        surface_info = {'name': name, 'type': stype, 'params': params, 'boundary': btype}
        self.surfaces_data.append(surface_info)
        self.history_surf.append(("add", surface_info))

        self.surf_list.addItem(self._surface_list_line(surface_info))
        self.combo_available_surfaces.addItem(name)

        self.log_signal.emit(f"✅ Success: Added Surface '{name}'.")
        self.generate_script()

    def _surface_list_line(self, s):
        params_str = ", ".join(f"{k}={v:g}" for k, v in s['params'].items())
        return f"{s['name']} | {s['type']} | {params_str} | {s['boundary']}"

    def undo_surface(self):
        if not self.history_surf:
            return
        entry = self.history_surf.pop()
        action = entry[0]
        if action == "add":
            data = entry[1]
            self.surfaces_data.remove(data)
            self._refresh_surf_list()
            self.log_signal.emit("↩️ Undo: Reverted last surface.")
            self.generate_script()
        elif action == "delete":
            _, data, row = entry
            insert_row = min(row, len(self.surfaces_data))
            self.surfaces_data.insert(insert_row, data)
            self._refresh_surf_list()
            self.log_signal.emit(f"↩️ Undo: Restored surface '{data['name']}'.")
            self.generate_script()

    def delete_surface(self):
        row = self.surf_list.currentRow()
        if row >= 0:
            data = self.surfaces_data.pop(row)
            self._refresh_surf_list()
            self.history_surf.append(("delete", data, row))
            self.log_signal.emit(f"🗑️ Deleted: Surface '{data['name']}'.")
            self.generate_script()

    def _refresh_surf_list(self):
        self.surf_list.clear()
        self.combo_available_surfaces.clear()
        for s in self.surfaces_data:
            self.surf_list.addItem(self._surface_list_line(s))
            self.combo_available_surfaces.addItem(s['name'])

    # ==========================================
    # --- Cells logic ---
    # ==========================================
    def add_cell(self):
        name = self.cell_name.text().strip()
        region = self.cell_region.text().strip()
        univ = self.cell_universe.text().strip()
        fill_mode_text = self.fill_mode_combo.currentText()
        fill_mode = self.FILL_MODE_KEYS.get(fill_mode_text, "void")

        if fill_mode == "material":
            fill_value = self.cell_material.currentText().strip()
        elif fill_mode == "universe":
            fill_value = self.cell_universe_fill_combo.currentText().strip()
        elif fill_mode == "lattice":
            fill_value = self.cell_lattice_fill_combo.currentText().strip()
        else:
            fill_value = ""

        if not name or not region:
            self.log_signal.emit("❌ Cell Error: Name and Region are required!")
            return
        if name in [c['name'] for c in self.cells_data]:
            self.log_signal.emit(f"❌ Cell Error: Cell '{name}' already exists!")
            return
        if fill_mode != "void" and not fill_value:
            self.log_signal.emit(f"❌ Cell Error: Fill Type '{fill_mode_text}' requires a Fill Value!")
            return
        if not univ:
            univ = "root"

        cell_info = {
            'name': name, 'fill_mode': fill_mode, 'fill_value': fill_value,
            'region': region, 'universe': univ,
        }
        self.cells_data.append(cell_info)
        self.history_cell.append(("add", cell_info))
        self.cell_list.addItem(self._cell_list_line(cell_info, fill_mode_text))
        self.log_signal.emit(f"✅ Success: Added Cell '{name}'.")
        self.refresh_universes_combo()
        self.generate_script()

    def _cell_list_line(self, c, fill_mode_text=None):
        if fill_mode_text is None:
            reverse = {v: k for k, v in self.FILL_MODE_KEYS.items()}
            fill_mode_text = reverse.get(c['fill_mode'], "Void")
        fill_str = f"{fill_mode_text}={c['fill_value']}" if c['fill_value'] else fill_mode_text
        return f"{c['name']} | Fill: {fill_str} | Reg: {c['region']} | Univ: {c['universe']}"

    def undo_cell(self):
        if not self.history_cell:
            return
        entry = self.history_cell.pop()
        action = entry[0]
        if action == "add":
            data = entry[1]
            self.cells_data.remove(data)
            self._refresh_cell_list()
            self.log_signal.emit("↩️ Undo: Reverted last cell.")
            self.generate_script()
        elif action == "delete":
            _, data, row = entry
            insert_row = min(row, len(self.cells_data))
            self.cells_data.insert(insert_row, data)
            self._refresh_cell_list()
            self.log_signal.emit(f"↩️ Undo: Restored cell '{data['name']}'.")
            self.generate_script()
        self.refresh_universes_combo()

    def delete_cell(self):
        row = self.cell_list.currentRow()
        if row >= 0:
            data = self.cells_data.pop(row)
            self._refresh_cell_list()
            self.history_cell.append(("delete", data, row))
            self.log_signal.emit(f"🗑️ Deleted: Cell '{data['name']}'.")
            self.refresh_universes_combo()
            self.generate_script()

    def _refresh_cell_list(self):
        self.cell_list.clear()
        for c in self.cells_data:
            self.cell_list.addItem(self._cell_list_line(c))

    # ==========================================
    # --- Script generation ---
    # ==========================================
    def generate_script(self):
        code = "\n# ==========================================\n"
        code += "# --- Geometry Setup (Surfaces, Cells & Lattices) ---\n"
        code += "# ==========================================\n"

        for s in self.surfaces_data:
            b_arg = f", boundary_type='{s['boundary']}'" if s['boundary'] != 'transmission' else ""
            ctor = self.SURFACE_TYPES[s['type']]['ctor']
            kwargs_str = ", ".join(f"{k}={v}" for k, v in s['params'].items())
            code += f"{s['name']} = openmc.{ctor}({kwargs_str}{b_arg})\n"

        code += "\n# --- Cells ---\n"
        univ_dict = {}
        deferred_fill = []
        for c in self.cells_data:
            if c['fill_mode'] == 'material' and c['fill_value']:
                mat_str = f"fill={c['fill_value']}, "
            else:
                mat_str = ""
            code += f"{c['name']} = openmc.Cell(name='{c['name']}', {mat_str}region=({c['region']}))\n"

            if c['fill_mode'] in ('universe', 'lattice') and c['fill_value']:
                deferred_fill.append((c['name'], c['fill_value']))

            u_name = c['universe']
            univ_dict.setdefault(u_name, []).append(c['name'])

        code += "\n# --- Universes ---\n"
        for u, c_list in univ_dict.items():
            code += f"{u} = openmc.Universe(name='{u}', cells=[{', '.join(c_list)}])\n"

        if self.lattices_data:
            code += "\n# --- Lattices ---\n"
            for lat in self.lattices_data:
                code += self._lattice_code(lat)

        if deferred_fill:
            # يُسنَد fill الكون/الشبكة المتداخلة بعد بناء كل الأكوان والشبكات
            # -- هذا يتجنّب مشكلة الترتيب الدائري (الكون الأب يحتاج كائن
            # الشبكة/الكون الابن قبل إنشائه، بينما الشبكة/الكون الابن يُبنى
            # من خلايا أخرى قد تُعرَّف لاحقاً في القائمة).
            code += "\n# --- Nested Fills (Universe/Lattice) ---\n"
            for cname, fval in deferred_fill:
                code += f"{cname}.fill = {fval}\n"

        code += "\n# --- Root Geometry ---\n"
        if 'root' in univ_dict:
            code += "geometry = openmc.Geometry(root)\n"
        elif univ_dict:
            first_u = list(univ_dict.keys())[0]
            code += f"geometry = openmc.Geometry({first_u})\n"

        self.script_generated.emit(code)

        QTimer.singleShot(200, self.render_preview)

    def render_preview(self):
        """نظام رسم حي معزول وذكي"""
        if not self.cells_data:
            self.lbl_viewer.clear()
            self.lbl_viewer.setText("🎨 Add at least one Cell to see the live preview.")
            return

        export_path = os.path.join(os.getcwd(), "export", "preview")
        os.makedirs(export_path, exist_ok=True)

        try:
            openmc.reset_auto_ids()
            objects_dict = {}

            # 1. بناء الأسطح (Surfaces)
            for s in self.surfaces_data:
                kw = dict(s['params'])
                if s['boundary'] != 'transmission':
                    kw['boundary_type'] = s['boundary']
                ctor_cls = getattr(openmc, self.SURFACE_TYPES[s['type']]['ctor'])
                objects_dict[s['name']] = ctor_cls(**kw)

            # 2. بناء الخلايا والأكوان (Cells & Universes)
            univ_cells = {}
            deferred_fill = []
            for c in self.cells_data:
                region = eval(c['region'], {"__builtins__": {}}, objects_dict)
                cell = openmc.Cell(name=c['name'], region=region)
                objects_dict[c['name']] = cell

                if c['fill_mode'] in ('universe', 'lattice') and c['fill_value']:
                    deferred_fill.append((cell, c['fill_value']))

                u_name = c['universe']
                univ_cells.setdefault(u_name, []).append(cell)

            for u_name, c_list in univ_cells.items():
                objects_dict[u_name] = openmc.Universe(name=u_name, cells=c_list)

            # 3. بناء الشبكات (Lattices)
            for lat in self.lattices_data:
                pitch_vals = tuple(float(x) for x in lat['pitch'].split(","))
                universes_grid = eval(lat['universes'], {"__builtins__": {}}, objects_dict)
                if lat['kind'] == 'rect':
                    lat_obj = openmc.RectLattice(name=lat['name'])
                    lat_obj.lower_left = tuple(float(x) for x in lat['lower_left'].split(","))
                    lat_obj.pitch = pitch_vals
                else:
                    lat_obj = openmc.HexLattice(name=lat['name'])
                    lat_obj.center = tuple(float(x) for x in lat['center'].split(","))
                    lat_obj.pitch = pitch_vals
                    lat_obj.orientation = lat['orientation']
                lat_obj.universes = universes_grid
                if lat['outer'] and lat['outer'] in objects_dict:
                    lat_obj.outer = objects_dict[lat['outer']]
                objects_dict[lat['name']] = lat_obj

            for cell, fval in deferred_fill:
                if fval in objects_dict:
                    cell.fill = objects_dict[fval]

            # 4. تحديد الجذر وتصدير الملفات (Root & Export)
            if 'root' in univ_cells:
                root_univ = objects_dict['root']
            else:
                first_u = list(univ_cells.keys())[0]
                root_univ = objects_dict[first_u]

            geom = openmc.Geometry(root_univ)
            geom.export_to_xml(os.path.join(export_path, 'geometry.xml'))

            openmc.Materials().export_to_xml(os.path.join(export_path, 'materials.xml'))

            # 5. إعداد الرسم الهندسي (Plotting)
            plot = openmc.Plot()
            plot.filename = 'geometry_plot'
            plot.basis = 'xy'
            plot.width = (100.0, 100.0)
            plot.pixels = (400, 400)
            plot.color_by = 'cell'

            plots = openmc.Plots([plot])
            plots.export_to_xml(os.path.join(export_path, 'plots.xml'))

            ppm_path = os.path.join(export_path, "geometry_plot.ppm")
            img_path = os.path.join(export_path, "geometry_plot.png")

            original_dir = os.getcwd()
            os.chdir(export_path)
            try:
                max_attempts = 3
                last_err = None
                for attempt in range(1, max_attempts + 1):
                    if os.path.exists(ppm_path):
                        try:
                            os.remove(ppm_path)
                        except OSError:
                            pass
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except OSError:
                            pass

                    try:
                        openmc.plot_geometry(output=False)

                        # تحويل الصورة إلى PNG لتعرضها الواجهة
                        from PIL import Image
                        if os.path.exists("geometry_plot.ppm"):
                            im = Image.open("geometry_plot.ppm")
                            im.save("geometry_plot.png", "PNG")

                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < max_attempts:
                            time.sleep(0.4 * attempt)
                if last_err is not None:
                    raise last_err
            finally:
                os.chdir(original_dir)

            # 6. عرض الصورة النهائية على الواجهة
            if os.path.exists(img_path):
                self.lbl_viewer.setPixmap(QPixmap(img_path))
                self.lbl_viewer.setScaledContents(True)
                self.lbl_viewer.setStyleSheet("border: none; background-color: white;")

        except Exception as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            self.lbl_viewer.clear()
            self.lbl_viewer.setText(f"⚠️ Geometry Error:\n{str(e)}")
            self.log_signal.emit(f"⚠️ Live Preview Error: {str(e)}")
