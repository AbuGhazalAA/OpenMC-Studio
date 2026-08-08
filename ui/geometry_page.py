import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QListWidget, QGroupBox, QSplitter, QFormLayout
)
from PySide6.QtCore import Signal, Qt


class GeometryPageWidget(QWidget):
    script_generated = Signal(str)
    log_signal = Signal(str)  # إشارة لإرسال الأخطاء للكونسول

    def __init__(self, project_manager=None, materials_page=None, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.materials_page = materials_page
        self.history = []  # لتخزين خطوات الرسم واسترجاعها (Undo)
        self.surfaces_data = []
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # إعدادات السطح
        surf_group = QGroupBox("Define CSG Surface")
        form = QFormLayout(surf_group)

        self.surf_name = QLineEdit("s1")
        self.surf_type = QComboBox()
        self.surf_type.addItems(["Sphere", "Z-Cylinder", "X-Plane", "Y-Plane", "Z-Plane"])
        self.surf_param = QLineEdit("10.0")
        self.surf_boundary = QComboBox()
        self.surf_boundary.addItems(["transmission", "vacuum", "reflective"])

        form.addRow("Name:", self.surf_name)
        form.addRow("Type:", self.surf_type)
        form.addRow("Radius/Coord (cm):", self.surf_param)
        form.addRow("Boundary Type:", self.surf_boundary)
        left_layout.addWidget(surf_group)

        # أزرار الإضافة والتراجع
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("➕ Add Surface")
        self.btn_add.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold;")
        self.btn_add.clicked.connect(self.add_surface)

        self.btn_undo = QPushButton("↩️ Undo")
        self.btn_undo.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold;")
        self.btn_undo.clicked.connect(self.undo_last)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_undo)
        left_layout.addLayout(btn_layout)

        left_layout.addWidget(QLabel("Created Surfaces:"))
        self.surf_list = QListWidget()
        left_layout.addWidget(self.surf_list)

        self.btn_delete = QPushButton("🗑️ Delete Selected")
        self.btn_delete.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_delete.clicked.connect(self.delete_surface)
        left_layout.addWidget(self.btn_delete)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.lbl_viewer = QLabel("Build geometry and generate script to render.")
        self.lbl_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.lbl_viewer)
        splitter.addWidget(right_panel)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def add_surface(self):
        name = self.surf_name.text().strip()
        stype = self.surf_type.currentText()
        param_str = self.surf_param.text().strip()
        btype = self.surf_boundary.currentText()

        # --- الفحص الذكي للأخطاء (Validation) ---
        if not name:
            self.log_signal.emit("❌ Geometry Error: Surface name cannot be empty! [Tip: Enter a valid name like 's1']")
            return

        if name in [s['name'] for s in self.surfaces_data]:
            self.log_signal.emit(
                f"❌ Geometry Error: Surface '{name}' already exists! [Tip: Change the name to avoid conflicts]")
            return

        try:
            param_val = float(param_str)
            if stype in ["Sphere", "Z-Cylinder"] and param_val <= 0:
                self.log_signal.emit(
                    "❌ Geometry Error: Radius must be a positive number (> 0)! [Tip: Check your parameter value]")
                return
        except ValueError:
            self.log_signal.emit(
                f"❌ Geometry Error: Invalid number '{param_str}'! [Tip: Use standard numeric values like 5.0]")
            return

        # في حال نجاح الفحص
        surface_info = {'name': name, 'type': stype, 'param': param_val, 'boundary': btype}
        self.surfaces_data.append(surface_info)
        self.history.append(("add_surface", surface_info))

        self.surf_list.addItem(f"{name} | {stype} | {param_val} | {btype}")
        self.log_signal.emit(f"✅ Success: Added Surface '{name}' successfully.")
        self.generate_script()

    def undo_last(self):
        if not self.history:
            self.log_signal.emit("⚠️ Warning: Nothing to undo.")
            return

        action, data = self.history.pop()
        if action == "add_surface":
            self.surfaces_data.remove(data)
            self.surf_list.clear()
            for s in self.surfaces_data:
                self.surf_list.addItem(f"{s['name']} | {s['type']} | {s['param']} | {s['boundary']}")
            self.log_signal.emit("↩️ Undo: Reverted last surface addition.")
            self.generate_script()

    def delete_surface(self):
        row = self.surf_list.currentRow()
        if row >= 0:
            data = self.surfaces_data.pop(row)
            self.surf_list.takeItem(row)
            self.history.append(("delete_surface", data, row))
            self.log_signal.emit(f"🗑️ Deleted: Surface '{data['name']}' removed.")
            self.generate_script()
        else:
            self.log_signal.emit("⚠️ Warning: Please select a surface to delete.")

    def generate_script(self):
        code = "\n# --- Geometry Setup ---\n"
        for s in self.surfaces_data:
            b_arg = f", boundary_type='{s['boundary']}'" if s['boundary'] != 'transmission' else ""
            if s['type'] == "Sphere":
                code += f"{s['name']} = openmc.Sphere(r={s['param']}{b_arg})\n"
            elif s['type'] == "Z-Cylinder":
                code += f"{s['name']} = openmc.ZCylinder(r={s['param']}{b_arg})\n"
            elif s['type'] == "X-Plane":
                code += f"{s['name']} = openmc.XPlane(x0={s['param']}{b_arg})\n"
            elif s['type'] == "Y-Plane":
                code += f"{s['name']} = openmc.YPlane(y0={s['param']}{b_arg})\n"
            elif s['type'] == "Z-Plane":
                code += f"{s['name']} = openmc.ZPlane(z0={s['param']}{b_arg})\n"
        self.script_generated.emit(code)