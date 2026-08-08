import os
import re
import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QListWidget, QGroupBox, QSplitter, QFormLayout, QTabWidget
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QPixmap


class GeometryPageWidget(QWidget):
    script_generated = Signal(str)
    log_signal = Signal(str)

    def __init__(self, project_manager=None, materials_page=None, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.materials_page = materials_page

        self.surfaces_data = []
        self.cells_data = []

        self.history_surf = []
        self.history_cell = []

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

    def _setup_surfaces_tab(self):
        layout = QVBoxLayout(self.tab_surfaces)

        surf_group = QGroupBox("Define CSG Surface")
        form = QFormLayout(surf_group)

        self.surf_name = QLineEdit("s1")
        self.surf_type = QComboBox()
        self.surf_type.addItems(["Sphere", "X-Cylinder", "Y-Cylinder", "Z-Cylinder", "X-Plane", "Y-Plane", "Z-Plane"])
        self.surf_param = QLineEdit("10.0")
        self.surf_boundary = QComboBox()
        self.surf_boundary.addItems(["transmission", "vacuum", "reflective"])

        form.addRow("Name:", self.surf_name)
        form.addRow("Type:", self.surf_type)
        form.addRow("Radius/Coord (cm):", self.surf_param)
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

    def _setup_cells_tab(self):
        layout = QVBoxLayout(self.tab_cells)

        cell_group = QGroupBox("Define Cell")
        form = QFormLayout(cell_group)

        self.cell_name = QLineEdit("c1")

        mat_layout = QHBoxLayout()
        self.cell_material = QComboBox()
        self.cell_material.addItem("void")

        self.btn_refresh_mats = QPushButton("🔄")
        self.btn_refresh_mats.setToolTip("Refresh Materials List")
        self.btn_refresh_mats.setFixedWidth(35)
        self.btn_refresh_mats.clicked.connect(self.refresh_materials)

        mat_layout.addWidget(self.cell_material)
        mat_layout.addWidget(self.btn_refresh_mats)

        # ==========================================
        # --- أداة بناء الـ CSG الذكية (الشكل المرتب الجديد) ---
        # ==========================================
        csg_builder_layout = QVBoxLayout()
        csg_builder_layout.setSpacing(5)

        # 1. قائمة الأسطح
        surf_layout = QHBoxLayout()
        self.combo_available_surfaces = QComboBox()
        self.combo_available_surfaces.setToolTip("Select a surface")

        btn_insert_surf = QPushButton("↙️ Insert Surface")
        btn_insert_surf.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        btn_insert_surf.clicked.connect(self._insert_surface_to_region)

        surf_layout.addWidget(self.combo_available_surfaces, stretch=1)
        surf_layout.addWidget(btn_insert_surf)
        csg_builder_layout.addLayout(surf_layout)

        # 2. قائمة العمليات المنطقية (القائمة المنسدلة الجديدة)
        op_layout = QHBoxLayout()
        self.combo_operators = QComboBox()
        self.combo_operators.addItems([
            "+ (Inside)",
            "- (Outside)",
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
        form.addRow("Material:", mat_layout)
        form.addRow("CSG Builder:", csg_builder_layout)
        form.addRow("Region (CSG):", self.cell_region)
        form.addRow("Universe:", self.cell_universe)
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

    def _insert_surface_to_region(self):
        """إدراج اسم السطح"""
        surface_name = self.combo_available_surfaces.currentText()
        if surface_name:
            self.cell_region.insert(surface_name)
            self.cell_region.setFocus()

    def _insert_operator_to_region(self):
        """إدراج العملية المنطقية من القائمة"""
        op_text = self.combo_operators.currentText()
        symbol = op_text.split(" ")[0]  # أخذ الرمز الأول فقط (+, -, &, |, ~, (, ))

        # إضافة مسافات جمالية حول الرموز المعينة لتسهيل القراءة
        if symbol in ["&", "|", "~"]:
            symbol = f" {symbol} "

        self.cell_region.insert(symbol)
        self.cell_region.setFocus()

    def _on_tab_changed(self, index):
        if index == 1:
            self.refresh_materials()

    def refresh_materials(self):
        current_mat = self.cell_material.currentText()
        self.cell_material.clear()
        self.cell_material.addItem("void")

        try:
            from PySide6.QtWidgets import QApplication
            full_code = ""

            for widget in QApplication.topLevelWidgets():
                if hasattr(widget, 'script_editor'):
                    full_code = widget.script_editor.editor.toPlainText()
                    break

            if full_code:
                import re
                mat_vars = re.findall(r"([a-zA-Z0-9_]+)\s*=\s*openmc\.Material", full_code)
                for mat in sorted(set(mat_vars)):
                    self.cell_material.addItem(mat)

        except Exception as e:
            self.log_signal.emit(f"⚠️ Warning: Could not parse materials from script. ({e})")

        index = self.cell_material.findText(current_mat)
        if index >= 0:
            self.cell_material.setCurrentIndex(index)

    def add_surface(self):
        name = self.surf_name.text().strip()
        stype = self.surf_type.currentText()
        param_str = self.surf_param.text().strip()
        btype = self.surf_boundary.currentText()

        if not name:
            self.log_signal.emit("❌ Geometry Error: Surface name cannot be empty!")
            return
        if name in [s['name'] for s in self.surfaces_data]:
            self.log_signal.emit(f"❌ Geometry Error: Surface '{name}' already exists!")
            return

        try:
            param_val = float(param_str)
            if stype in ["Sphere", "X-Cylinder", "Y-Cylinder", "Z-Cylinder"] and param_val <= 0:
                self.log_signal.emit("❌ Geometry Error: Radius must be a positive number (> 0)!")
                return
        except ValueError:
            self.log_signal.emit(f"❌ Geometry Error: Invalid number '{param_str}'!")
            return

        surface_info = {'name': name, 'type': stype, 'param': param_val, 'boundary': btype}
        self.surfaces_data.append(surface_info)
        self.history_surf.append(("add", surface_info))

        self.surf_list.addItem(f"{name} | {stype} | {param_val} | {btype}")
        self.combo_available_surfaces.addItem(name)

        self.log_signal.emit(f"✅ Success: Added Surface '{name}'.")
        self.generate_script()

    def undo_surface(self):
        if not self.history_surf:
            return
        action, data = self.history_surf.pop()
        if action == "add":
            self.surfaces_data.remove(data)
            self._refresh_surf_list()
            self.log_signal.emit("↩️ Undo: Reverted last surface.")
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
            self.surf_list.addItem(f"{s['name']} | {s['type']} | {s['param']} | {s['boundary']}")
            self.combo_available_surfaces.addItem(s['name'])

    def add_cell(self):
        name = self.cell_name.text().strip()
        mat = self.cell_material.currentText()
        region = self.cell_region.text().strip()
        univ = self.cell_universe.text().strip()

        if not name or not region:
            self.log_signal.emit("❌ Cell Error: Name and Region are required!")
            return
        if name in [c['name'] for c in self.cells_data]:
            self.log_signal.emit(f"❌ Cell Error: Cell '{name}' already exists!")
            return
        if not univ:
            univ = "root"

        cell_info = {'name': name, 'material': mat, 'region': region, 'universe': univ}
        self.cells_data.append(cell_info)
        self.history_cell.append(("add", cell_info))
        self.cell_list.addItem(f"{name} | Mat: {mat} | Reg: {region} | Univ: {univ}")
        self.log_signal.emit(f"✅ Success: Added Cell '{name}'.")
        self.generate_script()

    def undo_cell(self):
        if not self.history_cell:
            return
        action, data = self.history_cell.pop()
        if action == "add":
            self.cells_data.remove(data)
            self._refresh_cell_list()
            self.log_signal.emit("↩️ Undo: Reverted last cell.")
            self.generate_script()

    def delete_cell(self):
        row = self.cell_list.currentRow()
        if row >= 0:
            data = self.cells_data.pop(row)
            self._refresh_cell_list()
            self.history_cell.append(("delete", data, row))
            self.log_signal.emit(f"🗑️ Deleted: Cell '{data['name']}'.")
            self.generate_script()

    def _refresh_cell_list(self):
        self.cell_list.clear()
        for c in self.cells_data:
            self.cell_list.addItem(f"{c['name']} | Mat: {c['material']} | Reg: {c['region']} | Univ: {c['universe']}")

    def generate_script(self):
        code = "\n# ==========================================\n"
        code += "# --- Geometry Setup (Surfaces & Cells) ---\n"
        code += "# ==========================================\n"

        for s in self.surfaces_data:
            b_arg = f", boundary_type='{s['boundary']}'" if s['boundary'] != 'transmission' else ""
            if s['type'] == "Sphere":
                code += f"{s['name']} = openmc.Sphere(r={s['param']}{b_arg})\n"
            elif s['type'] == "X-Cylinder":
                code += f"{s['name']} = openmc.XCylinder(r={s['param']}{b_arg})\n"
            elif s['type'] == "Y-Cylinder":
                code += f"{s['name']} = openmc.YCylinder(r={s['param']}{b_arg})\n"
            elif s['type'] == "Z-Cylinder":
                code += f"{s['name']} = openmc.ZCylinder(r={s['param']}{b_arg})\n"
            elif s['type'] == "X-Plane":
                code += f"{s['name']} = openmc.XPlane(x0={s['param']}{b_arg})\n"
            elif s['type'] == "Y-Plane":
                code += f"{s['name']} = openmc.YPlane(y0={s['param']}{b_arg})\n"
            elif s['type'] == "Z-Plane":
                code += f"{s['name']} = openmc.ZPlane(z0={s['param']}{b_arg})\n"

        code += "\n# --- Cells ---\n"
        univ_dict = {}
        for c in self.cells_data:
            mat_str = "" if c['material'] == 'void' else f"fill={c['material']}, "
            code += f"{c['name']} = openmc.Cell(name='{c['name']}', {mat_str}region=({c['region']}))\n"

            u_name = c['universe']
            if u_name not in univ_dict:
                univ_dict[u_name] = []
            univ_dict[u_name].append(c['name'])

        code += "\n# --- Universes & Geometry ---\n"
        for u, c_list in univ_dict.items():
            code += f"{u} = openmc.Universe(name='{u}', cells=[{', '.join(c_list)}])\n"

        if 'root' in univ_dict:
            code += "\ngeometry = openmc.Geometry(root)\n"
        elif univ_dict:
            first_u = list(univ_dict.keys())[0]
            code += f"\ngeometry = openmc.Geometry({first_u})\n"

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

            for s in self.surfaces_data:
                kw = {}
                if s['boundary'] != 'transmission': kw['boundary_type'] = s['boundary']
                if s['type'] == "Sphere":
                    obj = openmc.Sphere(r=s['param'], **kw)
                elif s['type'] == "X-Cylinder":
                    obj = openmc.XCylinder(r=s['param'], **kw)
                elif s['type'] == "Y-Cylinder":
                    obj = openmc.YCylinder(r=s['param'], **kw)
                elif s['type'] == "Z-Cylinder":
                    obj = openmc.ZCylinder(r=s['param'], **kw)
                elif s['type'] == "X-Plane":
                    obj = openmc.XPlane(x0=s['param'], **kw)
                elif s['type'] == "Y-Plane":
                    obj = openmc.YPlane(y0=s['param'], **kw)
                elif s['type'] == "Z-Plane":
                    obj = openmc.ZPlane(z0=s['param'], **kw)
                objects_dict[s['name']] = obj

            univ_cells = {}
            for c in self.cells_data:
                region = eval(c['region'], {}, objects_dict)
                cell = openmc.Cell(name=c['name'], region=region)
                objects_dict[c['name']] = cell

                u_name = c['universe']
                if u_name not in univ_cells: univ_cells[u_name] = []
                univ_cells[u_name].append(cell)

            if 'root' in univ_cells:
                root_univ = openmc.Universe(cells=univ_cells['root'])
            else:
                first_u = list(univ_cells.keys())[0]
                root_univ = openmc.Universe(cells=univ_cells[first_u])

            geom = openmc.Geometry(root_univ)
            geom.export_to_xml(os.path.join(export_path, 'geometry.xml'))

            openmc.Materials().export_to_xml(os.path.join(export_path, 'materials.xml'))

            plot = openmc.Plot()
            plot.filename = 'preview_plot'
            plot.basis = 'xy'
            plot.width = (100.0, 100.0)
            plot.pixels = (400, 400)
            plot.color_by = 'cell'

            plots = openmc.Plots([plot])
            plots.export_to_xml(os.path.join(export_path, 'plots.xml'))

            original_dir = os.getcwd()
            os.chdir(export_path)
            openmc.plot_geometry(output=False)
            os.chdir(original_dir)

            img_path = os.path.join(export_path, "preview_plot.png")
            if os.path.exists(img_path):
                self.lbl_viewer.setPixmap(QPixmap(img_path))
                self.lbl_viewer.setScaledContents(True)
                self.lbl_viewer.setStyleSheet("border: none; background-color: white;")

        except Exception as e:
            if 'original_dir' in locals(): os.chdir(original_dir)
            self.lbl_viewer.clear()
            self.lbl_viewer.setText(f"⚠️ Geometry Syntax Error:\n{str(e)}")
            self.log_signal.emit(
                f"⚠️ Live Preview Error: Syntax issue in one of the regions. Please check your signs (- & +).")