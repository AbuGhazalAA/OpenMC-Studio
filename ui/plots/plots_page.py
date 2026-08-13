import os
import re
import sys
import subprocess
import time
import numpy as np
import openmc
import multiprocessing

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QGroupBox,
    QComboBox, QPushButton, QFormLayout, QSplitter, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication
)
from PySide6.QtCore import Signal, QTimer, Qt
from PySide6.QtGui import QPixmap, QWheelEvent, QPainter


def _neutralize_interactive_plot_methods():
    def _safe_plot_stub(self, *args, **kwargs):
        print('[OpenMC Studio] Skipped interactive .plot() call '
              '(needs openmc.lib) -- use "Force Render Now" instead.')
        return None

    for cls_name in ('Universe', 'Model', 'Geometry', 'Cell', 'Region'):
        cls = getattr(openmc, cls_name, None)
        if cls is not None and hasattr(cls, 'plot'):
            cls.plot = _safe_plot_stub


class ZoomableImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #ffffff; border: 2px dashed #cccccc;")
        self.zoom_factor = 1.15

    def set_image(self, pixmap: QPixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.setStyleSheet("border: none; background-color: #e8e8e8;")

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            zoom = self.zoom_factor
        else:
            zoom = 1 / self.zoom_factor
        self.scale(zoom, zoom)


class PlotsPageWidget(QWidget):
    script_generated = Signal(str)
    ZOOM_MARGIN = 1.3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._applying_preset = False
        self._geometry_cells = {}
        self._init_ui()

        self.live_timer = QTimer(self)
        self.live_timer.setSingleShot(True)
        self.live_timer.timeout.connect(self.render_live_plot)

        self.origin_field.textChanged.connect(self._on_manual_geometry_edit)
        self.width_field.textChanged.connect(self._on_manual_geometry_edit)
        self.basis_combo.currentTextChanged.connect(self._on_manual_geometry_edit)
        self.pixels_field.textChanged.connect(self._trigger_live_update)
        self.color_combo.currentTextChanged.connect(self._trigger_live_update)
        self.view_combo.currentTextChanged.connect(self._on_view_combo_changed)
        self.cell_combo.currentIndexChanged.connect(self._on_cell_combo_changed)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter()

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        config_group = QGroupBox("2D Plot Configuration")
        form = QFormLayout(config_group)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Custom", "Fit Full Geometry"])
        form.addRow("View:", self.view_combo)

        self.cell_combo = QComboBox()
        self.cell_combo.addItem("-- Zoom to cell --")
        self.cell_combo.setToolTip("Populated automatically from whatever cells exist in the script.")
        form.addRow("Zoom to Cell:", self.cell_combo)

        self.basis_combo = QComboBox()
        self.basis_combo.addItems(["xy", "xz", "yz"])
        form.addRow("Basis (Plane):", self.basis_combo)

        self.origin_field = QLineEdit("0.0, 0.0, 0.0")
        self.origin_field.setToolTip("Center of the view, 'x, y, z' in cm.")
        form.addRow("Origin (cm):", self.origin_field)

        self.width_field = QLineEdit("40.0, 40.0")
        form.addRow("Width (cm):", self.width_field)

        self.pixels_field = QLineEdit("1500, 1500")
        form.addRow("Resolution:", self.pixels_field)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["material", "cell"])
        form.addRow("Color By:", self.color_combo)

        left_layout.addWidget(config_group)

        self.btn_render = QPushButton("🔄 Force Render Now")
        self.btn_render.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold; padding: 8px;")
        self.btn_render.clicked.connect(self.render_live_plot)
        left_layout.addWidget(self.btn_render)

        self.btn_append = QPushButton("💾 Append Plot Code to Script")
        self.btn_append.setStyleSheet("background-color: #4a4a4a; color: white; padding: 5px; font-weight: bold;")
        self.btn_append.clicked.connect(self.append_plot_code)
        left_layout.addWidget(self.btn_append)

        left_layout.addStretch()

        self.viewer = ZoomableImageViewer()
        self.viewer.setMinimumSize(400, 400)

        splitter.addWidget(left_widget)
        splitter.addWidget(self.viewer)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

    def _log(self, msg):
        try:
            for widget in QApplication.topLevelWidgets():
                if hasattr(widget, 'console_widget'):
                    widget.console_widget.append_log(msg)
                    return
        except Exception:
            pass
        print(msg)

    def _trigger_live_update(self):
        self.live_timer.start(300)

    def _on_manual_geometry_edit(self):
        if not self._applying_preset:
            for combo in (self.view_combo, self.cell_combo):
                if combo.currentIndex() != 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)
        self._trigger_live_update()

    def _exec_script_and_get_geometry(self):
        main_win = self.window()
        if not hasattr(main_win, 'script_editor'):
            return None, [], "No script editor found."

        full_code = main_win.script_editor.editor.toPlainText()
        if not full_code.strip():
            return None, [], "Script is empty."

        safe_code = re.sub(r'openmc\.run\([^)]*\)', '', full_code)
        _neutralize_interactive_plot_methods()
        namespace = {'openmc': openmc, 'os': os, '__name__': '__main__'}

        export_path = os.path.abspath(os.path.join(os.getcwd(), "export", "main_plot"))
        os.makedirs(export_path, exist_ok=True)
        original_dir = os.getcwd()
        try:
            os.chdir(export_path)
            openmc.reset_auto_ids()
            exec(safe_code, namespace)
        except Exception as e:
            return None, [], str(e)
        finally:
            os.chdir(original_dir)

        geom = None
        mats = []
        for val in namespace.values():
            if isinstance(val, openmc.Geometry):
                geom = val
            elif isinstance(val, openmc.Material):
                mats.append(val)

        if not geom:
            return None, mats, "No 'openmc.Geometry' object was found in the code."
        return geom, mats, None

    def _refresh_cell_list(self, geom):
        try:
            cells = geom.get_all_cells()
        except Exception:
            cells = {}
        self._geometry_cells = cells

        self.cell_combo.blockSignals(True)
        self.cell_combo.clear()
        self.cell_combo.addItem("-- Zoom to cell --")
        for cid in sorted(cells.keys()):
            cell = cells[cid]
            label = f"{cid}: {cell.name}" if cell.name else f"Cell {cid}"
            self.cell_combo.addItem(label, userData=cid)
        self.cell_combo.blockSignals(False)

    @staticmethod
    def _safe_extent_and_center(bbox, basis, fallback_width=(20.0, 20.0)):
        center = np.array(bbox.center, dtype=float)
        if not np.all(np.isfinite(center)):
            center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            left, right, bottom, top = bbox.extent[basis]
            width = (right - left, top - bottom)
        except Exception:
            width = fallback_width

        if not all(np.isfinite(v) and v > 0 for v in width):
            width = fallback_width

        return center, width

    def _fill_fields_and_render(self, basis, center, width):
        self._applying_preset = True
        try:
            self.basis_combo.setCurrentText(basis)
            ox, oy, oz = center
            self.origin_field.setText(f"{ox:.6g}, {oy:.6g}, {oz:.6g}")
            w0 = width[0] * self.ZOOM_MARGIN
            w1 = width[1] * self.ZOOM_MARGIN
            self.width_field.setText(f"{w0:.6g}, {w1:.6g}")
        finally:
            self._applying_preset = False
        self.render_live_plot()

    def _on_view_combo_changed(self, name):
        if name != "Fit Full Geometry":
            return
        geom, mats, err = self._exec_script_and_get_geometry()
        if err:
            self._log(f"⚠️ Error scanning geometry: {err}")
            return
        self._refresh_cell_list(geom)
        basis = self.basis_combo.currentText()
        center, width = self._safe_extent_and_center(geom.bounding_box, basis)
        self._fill_fields_and_render(basis, center, width)

    def _on_cell_combo_changed(self, index):
        cid = self.cell_combo.itemData(index)
        if cid is None:
            return
        cell = self._geometry_cells.get(cid)
        if cell is None:
            return
        basis = self.basis_combo.currentText()
        center, width = self._safe_extent_and_center(cell.bounding_box, basis)
        self._fill_fields_and_render(basis, center, width)
        self.view_combo.blockSignals(True)
        self.view_combo.setCurrentIndex(0)
        self.view_combo.blockSignals(False)

    def render_live_plot(self):
        try:
            o_str = self.origin_field.text().strip()
            w_str = self.width_field.text().strip()
            p_str = self.pixels_field.text().strip()

            try:
                origin_tuple = tuple(map(float, o_str.split(',')))
                width_tuple = tuple(map(float, w_str.split(',')))
                pixels_tuple = tuple(map(int, p_str.split(',')))
            except ValueError:
                self._log("⚠️ Error: Origin, Width, or Resolution format is incorrect.")
                return

            if len(origin_tuple) != 3 or len(width_tuple) != 2 or len(pixels_tuple) != 2:
                self._log(f"⚠️ Error: Origin must be (x,y,z). Width & Resolution must be (x,y).")
                return

            b = self.basis_combo.currentText()
            c = self.color_combo.currentText()

            geom, mats, err = self._exec_script_and_get_geometry()
            if err:
                if "No 'openmc.Geometry'" in err:
                    self._log(f"⚠️ Geometry Error: {err}")
                else:
                    self._log(f"⚠️ Live Preview Error: {err}")
                return

            self._refresh_cell_list(geom)

            export_path = os.path.abspath(os.path.join(os.getcwd(), "export", "main_plot"))
            os.makedirs(export_path, exist_ok=True)
            original_dir = os.getcwd()
            try:
                os.chdir(export_path)
                geom.export_to_xml()
                openmc.Materials(mats).export_to_xml()

                plot = openmc.Plot()
                plot.filename = 'live_view'
                plot.basis = b
                plot.origin = origin_tuple
                plot.width = width_tuple
                plot.pixels = pixels_tuple
                plot.color_by = c

                plots = openmc.Plots([plot])
                plots.export_to_xml()

                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                ppm_path = os.path.join(export_path, "live_view.ppm")
                img_path = os.path.join(export_path, "live_view.png")
                max_attempts = 3
                plot_result = None

                threads = str(multiprocessing.cpu_count())

                for attempt in range(1, max_attempts + 1):
                    if os.path.exists(ppm_path):
                        try: os.remove(ppm_path)
                        except OSError: pass
                    if os.path.exists(img_path):
                        try: os.remove(img_path)
                        except OSError: pass

                    plot_result = subprocess.run(
                        ["openmc", "-s", threads, "--plot"], capture_output=True, text=True,
                        creationflags=creationflags
                    )

                    from PIL import Image
                    if os.path.exists(ppm_path):
                        try:
                            im = Image.open(ppm_path)
                            im.save(img_path, "PNG")
                        except Exception as e:
                            self._log(f"⚠️ Image conversion failed: {e}")

                    if plot_result.returncode == 0 and os.path.exists(img_path):
                        break

                    if attempt < max_attempts:
                        time.sleep(0.4 * attempt)

                if plot_result.returncode != 0 or not os.path.exists(img_path):
                    details = (plot_result.stderr or plot_result.stdout or "(no output captured)").strip()
                    self._log(f"⚠️ Live Preview Error: Plot generation failed after {max_attempts} attempts "
                               f"(exit code {plot_result.returncode}):\n{details}")
                    return
            finally:
                os.chdir(original_dir)

            if os.path.exists(img_path):
                pixmap = QPixmap()
                pixmap.load(img_path)
                self.viewer.set_image(pixmap)
                self._log("✅ Plot Rendered Successfully.")
            else:
                self._log("⚠️ Error: Image was not generated by OpenMC plotter.")

        except Exception as e:
            self._log(f"⚠️ Live Preview Error: {str(e)}")

    def append_plot_code(self):
        o = self.origin_field.text().strip()
        w = self.width_field.text().strip()
        p = self.pixels_field.text().strip()
        b = self.basis_combo.currentText()
        c = self.color_combo.currentText()

        py_code = (
            f"\n# ==========================================\n"
            f"# --- 2D Plot Definition ---\n"
            f"# ==========================================\n"
            f"plot = openmc.Plot()\n"
            f"plot.filename = 'geometry_plot'\n"
            f"plot.basis = '{b}'\n"
            f"plot.origin = ({o})\n"
            f"plot.width = ({w})\n"
            f"plot.pixels = ({p})\n"
            f"plot.color_by = '{c}'\n"
            f"plots = openmc.Plots([plot])\n"
            f"plots.export_to_xml()\n"
        )
        self.script_generated.emit(py_code)
        self._log("✅ Plot code appended to script editor.")

    def display_image(self, path):
        if os.path.exists(path):
            pixmap = QPixmap()
            pixmap.load(path)
            self.viewer.set_image(pixmap)
            self._log("✅ Plot Rendered via Main Window.")
        else:
            self._log(f"⚠️ Error: Cannot find the generated plot image at {path}.")