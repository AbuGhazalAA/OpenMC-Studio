import os
import sys
import subprocess
import h5py
import numpy as np
import openmc
from matplotlib.colors import ListedColormap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFormLayout, QGroupBox, QMessageBox, QSplitter,
    QSlider, QCheckBox, QListWidget, QListWidgetItem, QProgressBar
)
from PySide6.QtCore import Signal, Qt, QThread, QSize
from PySide6.QtGui import QColor, QPixmap, QIcon
import pyvista as pv
from pyvistaqt import QtInteractor


# Fixed palette (matplotlib tab20 + tab20b, as 0-1 RGB), so a given material
# ID is assigned the same color every time this session -- keeps the 3D view
# and the materials legend visually consistent across re-renders.
def _hex_to_rgb01(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


_PALETTE_HEX = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5', '#393b79', '#5254a3', '#6b6ecf', '#9c9ede',
    '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
    '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c',
    '#7b4173', '#a55194', '#ce6dbd', '#de9ed6',
]
_PALETTE = [_hex_to_rgb01(h) for h in _PALETTE_HEX]


def _color_for_material_id(mat_id):
    return _PALETTE[mat_id % len(_PALETTE)]


class VoxelGenerationWorker(QThread):
    """
    يشغّل أداة OpenMC لتوليد ملف الـ Voxel (plot_3d.h5) في خيط منفصل عن
    واجهة المستخدم، حتى لا تتجمّد النافذة أثناء الحسابات على الشبكات
    عالية الدقة.
    """
    finished_signal = Signal(bool, str)

    def __init__(self, export_dir):
        super().__init__()
        self.export_dir = export_dir

    def run(self):
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(
                ["openmc", "--plot"], cwd=self.export_dir,
                capture_output=True, text=True, creationflags=creationflags
            )
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "(no output captured)").strip()
                self.finished_signal.emit(
                    False, f"Voxel generation failed (exit code {result.returncode}):\n{details}")
                return
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class VoxelPageWidget(QWidget):
    """
    OpenMC 3D Voxel Plotter powered by PyVista (GPU Hardware Acceleration).
    Provides real-time interactive 3D rendering reflecting the actual script geometry,
    with real per-material colors, a materials legend, a clipping plane,
    an opacity control, and camera view presets.
    """
    script_generated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gen_worker = None
        self._export_path = None
        self._materials_by_id = {}
        self._current_mesh = None
        self._mesh_actor = None
        self._cmap = None
        self._clim = (0, 1)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter()

        # ==========================================
        # --- الجزء الأيسر: إعدادات الـ 3D Voxel ---
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        voxel_group = QGroupBox("3D Voxel Configuration")
        form_layout = QFormLayout(voxel_group)

        self.width_field = QLineEdit("50.0, 50.0, 50.0")
        form_layout.addRow("Width (x,y,z) cm:", self.width_field)

        self.pixels_field = QLineEdit("80, 80, 80")
        self.pixels_field.setToolTip("GPU-accelerated grid resolution.")
        form_layout.addRow("Resolution:", self.pixels_field)

        left_layout.addWidget(voxel_group)

        self.btn_gen_script = QPushButton("1. Generate Voxel Script")
        self.btn_gen_script.setStyleSheet("background-color: #0e639c; color: white; padding: 8px; font-weight: bold;")
        self.btn_gen_script.clicked.connect(self.generate_voxel_script)
        left_layout.addWidget(self.btn_gen_script)

        self.btn_load_h5 = QPushButton("2. Render Real-Time 3D (GPU)")
        self.btn_load_h5.setStyleSheet("background-color: #217346; color: white; padding: 8px; font-weight: bold;")
        self.btn_load_h5.clicked.connect(self.render_3d_model)
        left_layout.addWidget(self.btn_load_h5)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate -- generation time isn't predictable
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # --- أدوات العرض التفاعلية ---
        display_group = QGroupBox("Display")
        display_layout = QFormLayout(display_group)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(80)
        self.opacity_label = QLabel("80%")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_label)
        display_layout.addRow("Opacity:", opacity_row)

        self.clip_checkbox = QCheckBox("Enable Clipping Plane (drag to slice)")
        self.clip_checkbox.toggled.connect(self._refresh_display)
        display_layout.addRow(self.clip_checkbox)

        left_layout.addWidget(display_group)

        # --- محاور الكاميرا الجاهزة ---
        view_group = QGroupBox("Camera View")
        view_layout = QHBoxLayout(view_group)
        for label, handler in (
            ("Top", self._view_top),
            ("Front", self._view_front),
            ("Side", self._view_side),
            ("Iso", self._view_iso),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            view_layout.addWidget(btn)
        left_layout.addWidget(view_group)

        # --- مفتاح الألوان (Materials Legend) ---
        legend_group = QGroupBox("Materials Legend")
        legend_layout = QVBoxLayout(legend_group)
        self.legend_list = QListWidget()
        self.legend_list.setIconSize(QSize(14, 14))
        legend_layout.addWidget(self.legend_list)
        left_layout.addWidget(legend_group)

        # ==========================================
        # --- الجزء الأيمن: شاشة العرض PyVista 3D ---
        # ==========================================
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")  # خلفية داكنة احترافية

        splitter.addWidget(left_widget)
        splitter.addWidget(self.plotter)
        splitter.setSizes([300, 800])

        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Camera view presets
    # ------------------------------------------------------------------
    def _view_top(self):
        self.plotter.view_xy()
        self.plotter.render()

    def _view_front(self):
        self.plotter.view_xz()
        self.plotter.render()

    def _view_side(self):
        self.plotter.view_yz()
        self.plotter.render()

    def _view_iso(self):
        self.plotter.view_isometric()
        self.plotter.render()

    # ------------------------------------------------------------------
    def _on_opacity_changed(self, value):
        self.opacity_label.setText(f"{value}%")
        self._refresh_display()

    def _refresh_display(self):
        """يعيد رسم المجسّم الحالي بإعدادات الشفافية وقص المستوى الحالية،
        دون التأثير على موضع الكاميرا التي يتحكم بها المستخدم."""
        if self._current_mesh is None:
            return

        self.plotter.clear_plane_widgets()
        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor)
            except Exception:
                pass
            self._mesh_actor = None

        common_kwargs = dict(
            scalars="MaterialIdx",
            cmap=self._cmap,
            clim=self._clim,
            show_edges=False,
            opacity=self.opacity_slider.value() / 100.0,
            show_scalar_bar=False,
        )
        if self.clip_checkbox.isChecked():
            self._mesh_actor = self.plotter.add_mesh_clip_plane(self._current_mesh, **common_kwargs)
        else:
            self._mesh_actor = self.plotter.add_mesh(self._current_mesh, **common_kwargs)
        self.plotter.render()

    def generate_voxel_script(self):
        w = self.width_field.text().strip()
        p = self.pixels_field.text().strip()

        py_code = (
            f"\n# ==========================================\n"
            f"# 3D Voxel Plot Definition\n"
            f"# ==========================================\n"
            f"voxel = openmc.Plot()\n"
            f"voxel.type = 'voxel'\n"
            f"voxel.filename = 'plot_3d'\n"
            f"voxel.width = ({w})\n"
            f"voxel.pixels = ({p})\n"
            f"voxel.color_by = 'material'\n\n"
            f"try:\n"
            f"    plots.append(voxel)\n"
            f"except NameError:\n"
            f"    plots = openmc.Plots([voxel])\n"
        )
        self.script_generated.emit(py_code)
        QMessageBox.information(self, "Success", "Voxel script appended! Click 'Render Real-Time 3D' to view.")

    def render_3d_model(self):
        """
        يقوم بتصدير كود السكريبت الحالي مؤقتاً، وتوليد ملف voxel حقيقي للهندسة الحالية
        في خيط منفصل (حتى لا تتجمّد الواجهة)، ثم عرضه عبر الـ GPU باستخدام PyVista
        بالألوان الفعلية لكل مادة لتطابق الرسمة 100% مع الكود.
        """
        if self._gen_worker is not None and self._gen_worker.isRunning():
            return  # توليد سابق ما زال قيد التنفيذ

        try:
            # 1. التأكد من وجود مجلد التصدير
            export_path = os.path.join(os.getcwd(), "export")
            if not os.path.exists(export_path):
                os.makedirs(export_path)

            # 2. الحصول على كود السكريبت من النافذة الرئيسية عبر الأب
            main_win = self.window()
            if not hasattr(main_win, 'script_editor'):
                QMessageBox.warning(self, "No Script", "Could not find the script editor.")
                return

            code = main_win.script_editor.editor.toPlainText()

            # إلحاق تعريف الـ Voxel إذا لم يكن موجوداً لضمان توليد الملف
            if "voxel.type = 'voxel'" not in code:
                w = self.width_field.text().strip()
                p = self.pixels_field.text().strip()
                code += f"\n\nvoxel = openmc.Plot()\nvoxel.type = 'voxel'\nvoxel.filename = 'plot_3d'\nvoxel.width = ({w})\nvoxel.pixels = ({p})\nvoxel.color_by = 'material'\nplots = openmc.Plots([voxel])\n"

            # تنفيذ الكود محلياً لتوليد ملفات OpenMC
            openmc.reset_auto_ids()
            namespace = {'openmc': openmc, 'os': os}
            exec(code, namespace)

            cells = [obj for obj in namespace.values() if isinstance(obj, openmc.Cell)]
            geometry = openmc.Geometry(openmc.Universe(cells=cells))

            # استخدام الهندسة الفعلية إن وجدت في الـ namespace
            if 'geom' in namespace:
                geometry = namespace['geom']

            mats_list = [obj for obj in namespace.values() if isinstance(obj, openmc.Material)]
            materials = openmc.Materials(mats_list)
            self._materials_by_id = {m.id: m for m in mats_list}

            settings = next((obj for obj in namespace.values() if isinstance(obj, openmc.Settings)), None)
            if not settings:
                settings = openmc.Settings()
                settings.batches = 2
                settings.particles = 100
                settings.source = openmc.Source(space=openmc.stats.Point((0, 0, 0)))

            plots_obj = namespace.get('plots')

            # تصدير ملفات الـ XML والـ Voxel Plot
            materials.export_to_xml(os.path.join(export_path, "materials.xml"))
            geometry.export_to_xml(os.path.join(export_path, "geometry.xml"))
            settings.export_to_xml(os.path.join(export_path, "settings.xml"))
            if plots_obj:
                plots_obj.export_to_xml(os.path.join(export_path, "plots.xml"))

        except Exception as e:
            QMessageBox.critical(self, "Render Error", f"Failed to prepare 3D model:\n{str(e)}")
            return

        # 3. تشغيل أداة الرسم في خيط منفصل حتى لا تتجمّد الواجهة أثناء التوليد
        self._export_path = export_path
        self.progress_bar.setVisible(True)
        self.btn_load_h5.setEnabled(False)
        self.btn_gen_script.setEnabled(False)

        self._gen_worker = VoxelGenerationWorker(export_path)
        self._gen_worker.finished_signal.connect(self._on_voxel_generated)
        self._gen_worker.start()

    def _on_voxel_generated(self, success, message):
        self.progress_bar.setVisible(False)
        self.btn_load_h5.setEnabled(True)
        self.btn_gen_script.setEnabled(True)

        if not success:
            QMessageBox.critical(self, "Render Error", f"Failed to render 3D model:\n{message}")
            return

        self._load_and_display_voxel(self._export_path)

    def _load_and_display_voxel(self, export_path):
        """يقرأ ملف الـ Voxel الناتج ويعرضه بالـ GPU مع ألوان حقيقية لكل مادة ومفتاح ألوان."""
        voxel_file = os.path.join(export_path, "plot_3d.h5")
        if not os.path.exists(voxel_file):
            QMessageBox.warning(self, "File Not Found", f"Could not generate {voxel_file}. Check script geometry.")
            return

        try:
            with h5py.File(voxel_file, 'r') as f:
                data = f['data'][()]

            grid = pv.ImageData()
            grid.dimensions = np.array(data.shape) + 1
            flat_data = data.flatten(order='F')
            grid.cell_data["Material"] = flat_data

            # تصفية الفراغات لعرض العناصر الصلبة فقط
            thresholded_mesh = grid.threshold(0.5, scalars="Material")

            if thresholded_mesh.n_cells == 0:
                QMessageBox.warning(self, "Empty Data", "No solid materials found in the Voxel grid.")
                return

            # ربط كل معرّف مادة فعلي بلون ثابت، وبناء فهرس متراص (0..N-1)
            # يطابق خريطة الألوان المميزة (ListedColormap) بدقة.
            material_ids = np.unique(thresholded_mesh.cell_data["Material"]).astype(int)
            material_ids = material_ids[material_ids > 0]
            if material_ids.size == 0:
                QMessageBox.warning(self, "Empty Data", "No solid materials found in the Voxel grid.")
                return

            colors = [_color_for_material_id(mid) for mid in material_ids]
            idx_map = {mid: i for i, mid in enumerate(material_ids)}
            idx_array = np.vectorize(idx_map.get)(
                thresholded_mesh.cell_data["Material"].astype(int)
            )
            thresholded_mesh.cell_data["MaterialIdx"] = idx_array

            self._cmap = ListedColormap(colors)
            self._clim = (0, max(1, len(material_ids) - 1))
            self._current_mesh = thresholded_mesh

            self._populate_legend(material_ids, colors)

            self.plotter.clear()
            self._mesh_actor = None
            self._refresh_display()

            self.plotter.add_axes()
            self.plotter.reset_camera()

        except Exception as e:
            QMessageBox.critical(self, "Render Error", f"Failed to render 3D model:\n{str(e)}")

    def _populate_legend(self, material_ids, colors):
        self.legend_list.clear()
        for mid, color in zip(material_ids, colors):
            mat = self._materials_by_id.get(int(mid))
            name = mat.name if (mat is not None and mat.name) else f"Material {mid}"
            item = QListWidgetItem(f"{name}  (ID {mid})")

            r, g, b = (int(c * 255) for c in color)
            swatch = QPixmap(14, 14)
            swatch.fill(QColor(r, g, b))
            item.setIcon(QIcon(swatch))

            self.legend_list.addItem(item)
