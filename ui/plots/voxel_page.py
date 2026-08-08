import os
import h5py
import numpy as np
import openmc
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFormLayout, QGroupBox, QMessageBox, QSplitter
)
from PySide6.QtCore import Signal
import pyvista as pv
from pyvistaqt import QtInteractor


class VoxelPageWidget(QWidget):
    """
    OpenMC 3D Voxel Plotter powered by PyVista (GPU Hardware Acceleration).
    Provides real-time interactive 3D rendering reflecting the actual script geometry.
    """
    script_generated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
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

        left_layout.addStretch()

        # ==========================================
        # --- الجزء الأيمن: شاشة العرض PyVista 3D ---
        # ==========================================
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")  # خلفية داكنة احترافية

        splitter.addWidget(left_widget)
        splitter.addWidget(self.plotter)
        splitter.setSizes([300, 800])

        layout.addWidget(splitter)

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
        يقوم بتصدير كود السكريبت الحالي مؤقتاً، وتوليد ملف voxel حقيقي للهندسة الحالية،
        ثم عرضه عبر الـ GPU باستخدام PyVista لتطابق الرسمة 100% مع الكود.
        """
        try:
            # 1. التأكد من وجود مجلد التصدير
            export_path = os.path.join(os.getcwd(), "export")
            if not os.path.exists(export_path):
                os.makedirs(export_path)

            # 2. الحصول على كود السكريبت من النافذة الرئيسية عبر الأب
            main_win = self.window()
            if hasattr(main_win, 'script_editor'):
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

                # تشغيل أداة الرسم للبناء الفعلي لملف plot_3d.h5
                original_dir = os.getcwd()
                os.chdir(export_path)
                openmc.plot_geometry()
                os.chdir(original_dir)

            # 3. قراءة الملف الناتج وعرضه بالـ GPU
            voxel_file = os.path.join(export_path, "plot_3d.h5")
            if not os.path.exists(voxel_file):
                QMessageBox.warning(self, "File Not Found", f"Could not generate {voxel_file}. Check script geometry.")
                return

            self.plotter.clear()

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

                # رسم المجسم الثلاثي الأبعاد الحقيقي
                self.plotter.add_mesh(
                    thresholded_mesh,
                    scalars="Material",
                    cmap="tab20",
                    show_edges=False,
                    opacity=0.8,
                    show_scalar_bar=False
                )

                self.plotter.add_axes()
                self.plotter.reset_camera()

        except Exception as e:
            QMessageBox.critical(self, "Render Error", f"Failed to render 3D model:\n{str(e)}")