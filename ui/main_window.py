import logging
import os
import glob
import sys
import subprocess
import time
import re
import gc
import json
import numpy as np
import multiprocessing
from datetime import datetime

# Import the new combined Depletion and Burnup interface
from ui.depletion_page import DepletionPageWidget
from ui.burnup_page import DepletionBurnupWidget

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
    QTabWidget, QLabel, QStatusBar, QToolBar, QStyle, QApplication, QMessageBox, QFileDialog, QSplitter, QDialog,
    QPushButton, QGroupBox, QProgressBar, QInputDialog, QLineEdit, QComboBox, QListWidget, QFormLayout, QTextEdit,
    QFrame,
    QPlainTextEdit, QSlider, QCheckBox, QListWidgetItem
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer, QRegularExpression
from PySide6.QtGui import QAction, QIcon, QTextDocument, QFont, QSyntaxHighlighter, QTextCharFormat, QColor, QPixmap

from ui.property_editor import PropertyEditorWidget
from ui.geometry.geometry_page import GeometryPageWidget
from ui.materials.materials_page import MaterialsPageWidget
from ui.settings.settings_page import SettingsPageWidget
from ui.project_tree import ProjectTreeWidget
from ui.console_widget import ConsoleWidget
from ui.tallies.tallies_page import TalliesPageWidget
from core.project_manager import ProjectManager
from ui.script_editor import ScriptEditorWidget
from ui.libraries_page import LibrariesPageWidget

# Load the 2D page only at startup to speed up launch
from ui.plots.plots_page import PlotsPageWidget

import openmc

logger = logging.getLogger("OpenMC Studio")


# =================================================================
# --- Syntax Highlighter ---
# =================================================================
class OpenMCSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.highlighting_rules = []

        # Python keywords (orange)
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#FF7700"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "import", "from", "def", "class", "if", "elif", "else", "for", "while",
            "return", "True", "False", "None", "and", "or", "not", "in", "is",
            "try", "except", "with", "as", "pass", "break", "continue"
        ]
        for word in keywords:
            pattern = QRegularExpression(rf"\b{word}\b")
            self.highlighting_rules.append((pattern, keyword_format))

        # OpenMC keywords (light blue / cyan)
        openmc_format = QTextCharFormat()
        openmc_format.setForeground(QColor("#00A0FF"))
        openmc_format.setFontWeight(QFont.Bold)
        openmc_words = [
            "openmc", "Material", "Materials", "Cell", "Geometry", "Universe",
            "Settings", "Tallies", "Plot", "Plots", "Source", "IndependentSource",
            "Sphere", "ZPlane", "XPlane", "YPlane", "ZCylinder", "XCylinder", "YCylinder",
            "RightCircularCylinder", "RectangularParallelepiped", "RectLattice"
        ]
        for word in openmc_words:
            pattern = QRegularExpression(rf"\b{word}\b")
            self.highlighting_rules.append((pattern, openmc_format))

        # Numbers (pink)
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF00FF"))
        self.highlighting_rules.append((QRegularExpression(r"\b[0-9]+[lL]?\b"), number_format))
        self.highlighting_rules.append(
            (QRegularExpression(r"\b[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b"), number_format))

        # Strings (light green)
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#00AA00"))
        self.highlighting_rules.append((QRegularExpression("\".*?\""), string_format))
        self.highlighting_rules.append((QRegularExpression("'.*?'"), string_format))

        # Comments (italic gray)
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#888888"))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression("#[^\n]*"), comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


# =================================================================
# --- Cross Section Plotter Dialog ---
# =================================================================
class CrossSectionPlotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📉 Cross Section Plotting")
        self.resize(500, 480)
        self.setStyleSheet("QDialog { background-color: #FFFFFF; } QLabel { font-weight: bold; }")
        layout = QVBoxLayout(self)

        lib_path = os.environ.get('OPENMC_CROSS_SECTIONS', 'Not Set (Using Default/System Library)')
        info_label = QLabel(f"Current Directory = {lib_path}")
        info_label.setStyleSheet(
            "color: #555555; font-size: 11px; font-style: italic; border: 1px solid #ccc; padding: 4px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QFormLayout()

        self.nuclide_input = QLineEdit("U235")
        self.nuclide_input.setToolTip("Enter Nuclide Name (e.g., U235, O16, H1)")
        form.addRow("Nuclide =", self.nuclide_input)

        self.particle_combo = QComboBox()
        self.particle_combo.addItems(["neutron", "photon"])
        form.addRow("Particle Type:", self.particle_combo)

        layout.addLayout(form)

        layout.addWidget(QLabel("Select Reactions (MT):"))
        self.reaction_list = QListWidget()
        self.reaction_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        reactions = [
            'total', 'elastic', 'inelastic', 'nonelastic', 'fission', 'capture', 'absorption',
            '(n,2n)', '(n,3n)', '(n,p)', '(n,d)', '(n,t)', '(n,3He)', '(n,a)', '(n,g)',
            'heating', 'damage', 'nubar', 'prompt_nubar', 'delayed_nubar'
        ]
        self.reaction_list.addItems(reactions)
        self.reaction_list.setCurrentRow(0)
        layout.addWidget(self.reaction_list)

        btn_layout = QHBoxLayout()
        self.btn_plot = QPushButton("📉 Plot Cross Sections")
        self.btn_plot.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold; padding: 10px;")
        self.btn_plot.clicked.connect(self.plot_xs)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_plot)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def plot_xs(self):
        nuclide = self.nuclide_input.text().strip()
        rxns = [item.text() for item in self.reaction_list.selectedItems()]

        if not nuclide or not rxns:
            QMessageBox.warning(self, "Warning", "Please select a nuclide and at least one reaction.")
            return

        try:
            import matplotlib.pyplot as plt
            import openmc

            plt.close('all')
            reactions_dict = {nuclide: rxns}
            fig = openmc.plot_xs(reactions_dict)
            fig.set_size_inches(9, 6)

            if fig.axes:
                ax = fig.axes[0]
                ax.set_title(f"Cross Sections for {nuclide}", fontweight='bold')
                ax.grid(True, which='both', linestyle='--', alpha=0.6)

            fig.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Plotting Error",
                                 f"Failed to plot cross sections:\n{str(e)}\n\nMake sure the nuclide exists in your cross_sections.xml database.")


# =================================================================
# --- Worker Threads ---
# =================================================================
class ExportWorker(QThread):
    finished_signal = Signal(bool, str, str, int)

    def __init__(self, script_text, project_dir):
        super().__init__()
        self.script_text = script_text
        self.project_dir = project_dir

    def run(self):
        try:
            compatibility_shim = (
                "# --- OpenMC Studio: auto-injected compatibility shim ---\n"
                "import openmc\n"
                "openmc.reset_auto_ids()\n"
                "import openmc as _omcs_shim_openmc\n"
                "def _omcs_safe_plot_stub(self, *args, **kwargs):\n"
                "    print('[OpenMC Studio] Skipped interactive .plot() call -- use GUI viewer instead.')\n"
                "    return None\n"
                "for _omcs_cls_name in ('Universe', 'Model', 'Geometry', 'Cell', 'Region'):\n"
                "    _omcs_cls = getattr(_omcs_shim_openmc, _omcs_cls_name, None)\n"
                "    if _omcs_cls is not None and hasattr(_omcs_cls, 'plot'):\n"
                "        _omcs_cls.plot = _omcs_safe_plot_stub\n"
                "# --- end compatibility shim ---\n\n"
            )

            auto_export_footer = (
                "\n# --- OpenMC Studio: auto-injected export footer ---\n"
                "_omcs_geom = None\n"
                "_omcs_mats_list = []\n"
                "_omcs_settings_obj = None\n"
                "_omcs_tallies_obj = None\n"
                "_omcs_plots_obj = None\n"
                "for _omcs_val in list(globals().values()):\n"
                "    if isinstance(_omcs_val, openmc.Geometry):\n"
                "        _omcs_geom = _omcs_val\n"
                "    elif isinstance(_omcs_val, openmc.Material):\n"
                "        _omcs_mats_list.append(_omcs_val)\n"
                "    elif isinstance(_omcs_val, openmc.Settings):\n"
                "        _omcs_settings_obj = _omcs_val\n"
                "    elif isinstance(_omcs_val, openmc.Settings):\n"
                "        _omcs_settings_obj = _omcs_val\n"
                "    elif isinstance(_omcs_val, openmc.Tallies):\n"
                "        _omcs_tallies_obj = _omcs_val\n"
                "    elif isinstance(_omcs_val, openmc.Plots):\n"
                "        _omcs_plots_obj = _omcs_val\n"
                "if _omcs_geom is not None:\n"
                "    _omcs_geom.export_to_xml()\n"
                "if _omcs_mats_list:\n"
                "    openmc.Materials(_omcs_mats_list).export_to_xml()\n"
                "if _omcs_settings_obj is not None:\n"
                "    _omcs_settings_obj.export_to_xml()\n"
                "if _omcs_tallies_obj is not None:\n"
                "    _omcs_tallies_obj.export_to_xml()\n"
                "if _omcs_plots_obj is None and _omcs_geom is not None:\n"
                "    _omcs_default_plot = openmc.Plot()\n"
                "    _omcs_default_plot.filename = 'geometry_plot'\n"
                "    _omcs_default_plot.color_by = 'material'\n"
                "    _omcs_default_plot.origin = (0.0, 0.0, 0.0)\n"
                "    _omcs_default_plot.width = (50.0, 50.0)\n"
                "    _omcs_default_plot.pixels = (1000, 1000)\n"
                "    try:\n"
                "        _omcs_bb = _omcs_geom.bounding_box\n"
                "        _omcs_ll, _omcs_ur = _omcs_bb[0], _omcs_bb[1]\n"
                "        _omcs_cx = float((_omcs_ll[0] + _omcs_ur[0]) / 2.0)\n"
                "        _omcs_cy = float((_omcs_ll[1] + _omcs_ur[1]) / 2.0)\n"
                "        _omcs_cz = float((_omcs_ll[2] + _omcs_ur[2]) / 2.0)\n"
                "        _omcs_finite = lambda _v: _v == _v and abs(_v) != float('inf')\n"
                "        if _omcs_finite(_omcs_cx) and _omcs_finite(_omcs_cy):\n"
                "            _omcs_default_plot.origin = (_omcs_cx, _omcs_cy, _omcs_cz if _omcs_finite(_omcs_cz) else 0.0)\n"
                "        _omcs_w = float((_omcs_ur[0] - _omcs_ll[0]) * 1.1)\n"
                "        _omcs_h = float((_omcs_ur[1] - _omcs_ll[1]) * 1.1)\n"
                "        if _omcs_finite(_omcs_w) and _omcs_w > 0 and _omcs_finite(_omcs_h) and _omcs_h > 0:\n"
                "            _omcs_default_plot.width = (_omcs_w, _omcs_h)\n"
                "    except Exception:\n"
                "        pass\n"
                "    _omcs_plots_obj = openmc.Plots([_omcs_default_plot])\n"
                "if _omcs_plots_obj is not None:\n"
                "    for _omcs_plot in _omcs_plots_obj:\n"
                "        try:\n"
                "            _omcs_origin = tuple(_omcs_plot.origin)\n"
                "            if len(_omcs_origin) != 3 or any((v != v or abs(v) == float('inf')) for v in _omcs_origin):\n"
                "                _omcs_plot.origin = (0.0, 0.0, 0.0)\n"
                "        except Exception:\n"
                "            _omcs_plot.origin = (0.0, 0.0, 0.0)\n"
                "    _omcs_plots_obj.export_to_xml()\n"
                "# --- end auto-injected export footer ---\n"
            )

            script_path = os.path.join(self.project_dir, "current_script.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(compatibility_shim)
                f.write(self.script_text)
                f.write(auto_export_footer)

            export_path = os.path.join(self.project_dir, "export")
            if not os.path.exists(export_path):
                os.makedirs(export_path)

            is_depletion_script = "openmc.deplete" in self.script_text

            if not is_depletion_script:
                result = subprocess.run(
                    [sys.executable, script_path], cwd=export_path,
                    capture_output=True, text=True, encoding='utf-8', errors='replace'
                )
                if result.returncode != 0:
                    details = (result.stderr or result.stdout or "(no output captured)").strip()
                    self.finished_signal.emit(
                        False, f"Script execution failed (exit code {result.returncode}):\n{details}", "", 0)
                    return
            else:
                import shutil
                shutil.copy(script_path, os.path.join(export_path, "current_script.py"))

            total_batches = 100
            settings_xml_path = os.path.join(export_path, "settings.xml")
            if os.path.exists(settings_xml_path):
                try:
                    import xml.etree.ElementTree as ET
                    batches_elem = ET.parse(settings_xml_path).getroot().find('batches')
                    if batches_elem is not None and batches_elem.text:
                        total_batches = int(batches_elem.text.strip())
                except Exception:
                    pass

            self.finished_signal.emit(True, "", export_path, total_batches)

        except Exception as e:
            self.finished_signal.emit(False, f"Script execution failed: {e}", "", 0)


class SimulationWorker(QThread):
    finished_signal = Signal(bool, str)
    log_signal = Signal(str)
    progress_signal = Signal(int, str)

    def __init__(self, export_dir, total_batches, is_depletion=False):
        super().__init__()
        self.export_dir = export_dir
        self.total_batches = total_batches
        self.is_depletion = is_depletion

    def run(self):
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

            if self.is_depletion:
                wsl_path = self.export_dir.replace('\\', '/').replace('C:', '/mnt/c').replace('D:', '/mnt/d')

                cmd = ["wsl", "bash", "-lic", f"cd '{wsl_path}' && conda run -n openmc_env python current_script.py"]
            else:
                cmd = ["openmc"]

            process = subprocess.Popen(
                cmd,
                cwd=self.export_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags,
                encoding='utf-8',
                errors='replace'
            )

            start_time = time.time()
            batch_pattern = re.compile(r'^\s*(\d+)\s*/\s*(\d+)')

            for line in process.stdout:
                clean_line = line.strip()
                self.log_signal.emit(clean_line)

                current_batch = 0
                match = batch_pattern.search(clean_line)
                if match:
                    current_batch = int(match.group(1))
                elif "Simulating batch" in clean_line:
                    try:
                        current_batch = int(clean_line.split()[-1])
                    except:
                        pass

                if current_batch > 0 and self.total_batches > 0:
                    elapsed = time.time() - start_time
                    percent = int((current_batch / self.total_batches) * 100)
                    percent = min(percent, 100)
                    time_per_batch = elapsed / current_batch
                    remaining_batches = self.total_batches - current_batch
                    eta_seconds = remaining_batches * time_per_batch
                    m, s = divmod(int(eta_seconds), 60)
                    eta_text = f"{m:02d}:{s:02d}"
                    self.progress_signal.emit(percent, eta_text)

            process.wait()
            if process.returncode == 0:
                self.finished_signal.emit(True, "Simulation/Depletion completed successfully!")
            else:
                self.finished_signal.emit(False, f"Simulation failed with return code {process.returncode}")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class PlotWorker(QThread):
    finished_signal = Signal(bool, str, str)

    def __init__(self, export_dir):
        super().__init__()
        self.export_dir = export_dir

    def run(self):
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

            ppm_path = os.path.join(self.export_dir, "geometry_plot.ppm")
            png_path = os.path.join(self.export_dir, "geometry_plot.png")

            if os.path.exists(ppm_path):
                try:
                    os.remove(ppm_path)
                except OSError:
                    pass
            if os.path.exists(png_path):
                try:
                    os.remove(png_path)
                except OSError:
                    pass

            threads = str(multiprocessing.cpu_count())
            result = subprocess.run(
                ["openmc", "-s", threads, "--plot"], cwd=self.export_dir,
                capture_output=True, text=True, creationflags=creationflags,
                encoding='utf-8', errors='replace'
            )

            from PIL import Image
            if os.path.exists(ppm_path):
                try:
                    im = Image.open(ppm_path)
                    im.save(png_path, "PNG")
                except Exception as e:
                    pass

            if result.returncode != 0 and not os.path.exists(png_path):
                details = (result.stderr or result.stdout or "(no output captured)").strip()
                self.finished_signal.emit(
                    False, f"Plot generation failed (exit code {result.returncode}):\n{details}", "")
                return

            if not os.path.exists(png_path):
                self.finished_signal.emit(False, "geometry_plot.png was not generated.", "")
                return

            self.finished_signal.emit(True, "Plot generated successfully!", png_path)
        except Exception as e:
            self.finished_signal.emit(False, str(e), "")


class SimulationProgressDialog(QDialog):
    def __init__(self, parent=None, theme="colored"):
        super().__init__(parent)
        self.setWindowTitle("Running Simulation")
        self.setFixedSize(450, 160)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.setModal(True)

        main_color = "#008000" if theme == "colored" else "#555555"
        sec_color = "#FF8C00" if theme == "colored" else "#777777"

        self.setStyleSheet(f"QDialog {{ background-color: #FFFFFF; border: 2px solid {main_color}; }}")

        layout = QVBoxLayout(self)
        self.lbl_title = QLabel("🔥 Simulation is Running... Please Wait")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet(f"color: {sec_color}; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(self.lbl_title)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: 2px solid {main_color}; border-radius: 5px; text-align: center; color: black; font-weight: bold; font-size: 14px; background-color: #F8F9FA;}}
            QProgressBar::chunk {{ background-color: {sec_color}; }}
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(25)
        layout.addWidget(self.progress_bar)

        self.lbl_time = QLabel("⏱️ Elapsed: 00:00 | ⏳ ETA: Calculating...")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_time.setStyleSheet(f"color: {main_color}; font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(self.lbl_time)

        self.start_time = time.time()
        self.eta_text = "Calculating..."
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time_display)
        self.timer.start(1000)

    def update_time_display(self):
        elapsed = int(time.time() - self.start_time)
        e_m, e_s = divmod(elapsed, 60)
        self.lbl_time.setText(f"⏱️ Elapsed Time: {e_m:02d}:{e_s:02d} Min | ⏳ Est. Time Left: {self.eta_text} Min")

    def update_progress(self, percent, eta_str):
        self.progress_bar.setValue(percent)
        self.eta_text = eta_str
        self.update_time_display()


# =================================================================
# --- MainWindow ---
# =================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenMC Studio v1.0")
        from PySide6.QtGui import QIcon
        import sys
        _icon_base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
        self.setWindowIcon(QIcon(os.path.join(_icon_base, 'app_icon.ico')))
        self.resize(1440, 900)
        self.setMinimumSize(1024, 768)

        self.current_theme = "colored"
        self.project_manager = ProjectManager()
        self._active_workers = []

        self.script_editor = ScriptEditorWidget()

        self._setup_tool_windows()
        self._setup_central_widget()
        self._setup_dock_widgets()
        self._setup_actions()
        self._setup_menus()
        self._setup_toolbars()
        self._setup_statusbar()
        self._apply_theme()

        # Ensure the reset command is written in the default page
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        initial_text = f"# c      Created on: {now_str}\n\nimport openmc\nopenmc.reset_auto_ids()\n\n"
        if hasattr(self.script_editor, 'editor'):
            self.script_editor.editor.setPlainText(initial_text)
        else:
            self.script_editor.setPlainText(initial_text)

        QTimer.singleShot(500, self._check_openmc_status)

    def _check_openmc_status(self):
        try:
            py_version = openmc.__version__

            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(['openmc', '--version'], capture_output=True, text=True,
                                    creationflags=creationflags, encoding='utf-8', errors='replace')

            if result.returncode == 0:
                match = re.search(r'version\s+([\d\.]+)', result.stdout)
                v_str = match.group(1) if match else py_version

                msg = f"✅ OpenMC v{v_str} Ready"
                self.openmc_status_label.setText(msg)
                self.openmc_status_label.setStyleSheet(
                    "color: #008000; font-weight: bold; font-size: 13px; padding: 0 15px;")
                self.console_widget.append_log(f"✅ OpenMC (v{v_str}) is installed and ready.")
            else:
                msg = f"⚠️ OpenMC CLI Error (API v{py_version})"
                self.openmc_status_label.setText(msg)
                self.openmc_status_label.setStyleSheet(
                    "color: #FF0000; font-weight: bold; font-size: 13px; padding: 0 15px;")
                self.console_widget.append_log("⚠️ OpenMC executable not found or failed.")

        except FileNotFoundError:
            self.openmc_status_label.setText("❌ OpenMC Not Found")
            self.openmc_status_label.setStyleSheet(
                "color: #FF0000; font-weight: bold; font-size: 13px; padding: 0 15px;")
            self.console_widget.append_log("❌ OpenMC executable not found in PATH.")
        except Exception as e:
            self.openmc_status_label.setText("⚠️ OpenMC Check Failed")
            self.openmc_status_label.setStyleSheet(
                "color: #FF8C00; font-weight: bold; font-size: 13px; padding: 0 15px;")
            self.console_widget.append_log(f"⚠️ Error checking OpenMC status: {str(e)}")

    def change_script_bg(self, bg_color, text_color):
        style = f"background-color: {bg_color}; color: {text_color}; font-family: Consolas, 'Courier New', monospace; font-size: 14px; border: none; padding: 5px;"
        if hasattr(self.script_editor, 'editor'):
            self.script_editor.editor.setStyleSheet(style)
        else:
            self.script_editor.setStyleSheet(style)
        self.statusbar.showMessage("🎨 Script background color updated.", 4000)

    def toggle_theme(self):
        self.current_theme = "grayscale" if self.current_theme == "colored" else "colored"
        self._apply_theme()
        for win in [self.win_materials, self.win_geometry, self.win_settings, self.win_tallies, self.win_libraries]:
            win.setStyleSheet(self.styleSheet())

    def _apply_theme(self):
        primary = "#008000" if self.current_theme == "colored" else "#555555"
        primary_hover = "#005500" if self.current_theme == "colored" else "#333333"
        secondary = "#FF8C00" if self.current_theme == "colored" else "#777777"

        self.setStyleSheet(f"""
            QMainWindow, QDialog {{ background-color: #FFFFFF; color: #000000; }}
            QMenuBar {{ background-color: {primary}; color: #FFFFFF; font-size: 14px; font-weight: bold; }}
            QMenuBar::item:selected {{ background-color: {primary_hover}; }}
            QMenu {{ background-color: #FFFFFF; color: {primary}; border: 1px solid {primary}; }}
            QMenu::item:selected {{ background-color: {primary}; color: #FFFFFF; }}
            QToolBar {{ background-color: #F8F9FA; border-bottom: 2px solid {primary}; }}
            QToolButton {{ color: #000000; font-weight: bold; }}
            QStatusBar {{ background-color: {primary}; color: white; font-weight: bold; }}

            /* Hide tabs completely */
            QTabWidget::pane {{ border: 2px solid {primary}; border-top-left-radius: 5px; border-top-right-radius: 5px; }}

            QGroupBox {{ border: 2px solid {primary}; border-radius: 5px; margin-top: 1ex; font-weight: bold; color: {primary}; }}
            QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }}
            QLabel, QCheckBox, QRadioButton {{ color: #000000; font-weight: bold; }}
            QLineEdit, QComboBox, QSpinBox {{ background-color: #FFFFFF; color: #000000; border: 1px solid {primary}; padding: 2px; }}
        """)

        if hasattr(self, 'script_header'):
            self.script_header.setStyleSheet(
                f"background-color: {secondary}; color: white; padding: 10px; font-weight: bold; font-size: 15px; border-radius: 4px;")

        if hasattr(self, 'script_editor'):
            self.script_editor.set_theme(self.current_theme)

    # --- New function to intelligently insert cross-section config at the top ---
    def insert_library_code_at_top(self, code):
        text = self._get_script_text()

        # Clean up any existing cross_sections config to avoid duplicates
        text = re.sub(r"\n# --- Cross Sections Library Setup.*?openmc\.config\['cross_sections'\].*?\n", "", text,
                      flags=re.DOTALL)
        text = re.sub(r"openmc\.config\['cross_sections'\].*\n", "", text)

        # Insert intelligently after openmc.reset_auto_ids() or import openmc
        target = "openmc.reset_auto_ids()"
        if target in text:
            new_text = text.replace(target, f"{target}\n{code}")
        elif "import openmc" in text:
            new_text = text.replace("import openmc", f"import openmc\n{code}")
        else:
            new_text = f"import openmc\n{code}\n{text}"

        # Update the editor
        if hasattr(self.script_editor, 'editor'):
            self.script_editor.editor.setPlainText(new_text)
        else:
            self.script_editor.setPlainText(new_text)

        if hasattr(self, 'console_widget'):
            self.console_widget.append_log("✅ Cross-sections path automatically inserted at the TOP of the script.")

    def _setup_tool_windows(self):
        for attr_name in ('win_materials', 'win_geometry', 'win_settings', 'win_tallies', 'win_libraries'):
            old = getattr(self, attr_name, None)
            if old is not None:
                old.close()
                old.deleteLater()

        self.materials_page = MaterialsPageWidget(project_manager=self.project_manager)
        self.geometry_page = GeometryPageWidget(project_manager=self.project_manager,
                                                materials_page=self.materials_page)
        self.settings_page = SettingsPageWidget(project_manager=self.project_manager)
        self.tallies_page = TalliesPageWidget()
        self.libraries_page = LibrariesPageWidget()

        self.materials_page.script_generated.connect(self.script_editor.append_code)
        self.geometry_page.script_generated.connect(
            lambda code: self.script_editor.replace_tagged_block("GEOMETRY", code))
        self.settings_page.script_generated.connect(self.script_editor.append_code)
        self.tallies_page.script_generated.connect(self.script_editor.append_code)

        # --- Use the new smart function for the libraries page ---
        self.libraries_page.script_generated.connect(self.insert_library_code_at_top)

        self.win_materials = self._create_popup_window("Materials Editor", self.materials_page)
        self.win_settings = self._create_popup_window("Simulation Settings", self.settings_page)
        self.win_tallies = self._create_popup_window("Tallies & Detectors", self.tallies_page)
        self.win_libraries = self._create_popup_window("Nuclear Data Libraries Manager", self.libraries_page)

        self.win_geometry = QDialog(self)
        self.win_geometry.setWindowTitle("Geometry Builder")
        self.win_geometry.setMinimumSize(800, 600)
        self.win_geometry.setWindowFlags(
            self.win_geometry.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint)
        geom_layout = QVBoxLayout(self.win_geometry)

        macro_layout = QHBoxLayout()
        macro_layout.addWidget(QLabel("<b>Insert Macrobody to Script:</b>"))
        self.macro_combo = QComboBox()
        self.macro_combo.addItems(["RCC (Cylinder)", "RPP (Box)", "SPH (Sphere)", "TRC (Truncated Cone)"])
        macro_layout.addWidget(self.macro_combo)

        self.btn_insert_macro = QPushButton("➕ Insert Code")
        self.btn_insert_macro.setStyleSheet("background-color: #555555; color: white; font-weight: bold; padding: 4px;")
        self.btn_insert_macro.clicked.connect(self.insert_macrobody)
        macro_layout.addWidget(self.btn_insert_macro)
        macro_layout.addStretch()

        geom_layout.addLayout(macro_layout)
        geom_layout.addWidget(self.geometry_page)

        self.depletion_page = DepletionBurnupWidget()
        if hasattr(self.depletion_page, 'script_generated'):
            self.depletion_page.script_generated.connect(self.script_editor.append_code)

        self.win_depletion = self._create_popup_window("Burnup & Depletion", self.depletion_page)

    def _create_popup_window(self, title, widget):
        win = QDialog(self)
        win.setWindowTitle(title)
        win.setMinimumSize(800, 600)
        win.setWindowFlags(win.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint
                           | Qt.WindowType.WindowMinimizeButtonHint)
        layout = QVBoxLayout(win)
        layout.addWidget(widget)
        return win

    def _setup_central_widget(self):
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.visual_tabs = QTabWidget()
        self.visual_tabs.tabBar().hide()

        self.welcome_page = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_page)

        welcome_logo = QLabel("OpenMC Studio")
        welcome_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_logo.setStyleSheet("font-size: 32px; color: #008000; font-weight: bold; border: none;")

        welcome_desc = QLabel("Select a Workspace from the menu above to begin")
        welcome_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_desc.setStyleSheet("font-size: 18px; color: #555555; border: none;")

        welcome_layout.addStretch()
        welcome_layout.addWidget(welcome_logo)
        welcome_layout.addWidget(welcome_desc)
        welcome_layout.addStretch()

        self.plots_page = PlotsPageWidget()
        self.plots_page.script_generated.connect(self.script_editor.append_code)

        self.visual_tabs.addTab(self.welcome_page, "🏠 Home")
        self.visual_tabs.addTab(self.plots_page, "🎨 2D Geometry Viewer")
        self.visual_tabs.addTab(QWidget(), "🧊 3D Voxel Viewer")
        self.visual_tabs.addTab(QWidget(), "🎯 Particle Tracks")
        self.visual_tabs.addTab(QWidget(), "📊 Simulation Results")

        self.visual_tabs.currentChanged.connect(self._lazy_load_tabs)
        self.main_splitter.addWidget(self.visual_tabs)

        script_container = QWidget()
        script_layout = QVBoxLayout(script_container)
        script_layout.setContentsMargins(0, 0, 0, 0)

        script_header_layout = QHBoxLayout()
        self.script_header = QLabel("⚛️ Python Script Engine")
        script_header_layout.addWidget(self.script_header)

        self.btn_check_syntax = QPushButton("🔍 Check Syntax")
        self.btn_check_syntax.setStyleSheet(
            "background-color: #555555; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_check_syntax.clicked.connect(self.check_script_syntax)
        script_header_layout.addWidget(self.btn_check_syntax)

        script_header_layout.addStretch()

        script_header_container = QWidget()
        script_header_container.setLayout(script_header_layout)

        script_layout.addWidget(script_header_container)
        script_layout.addWidget(self.script_editor)

        self.main_splitter.addWidget(script_container)
        self.main_splitter.setSizes([850, 550])
        self.setCentralWidget(self.main_splitter)

        self.building_overlay = QFrame(self.plots_page)
        self.building_overlay.setObjectName("buildingOverlay")
        self.building_overlay.setFixedSize(320, 104)
        self.building_overlay.setStyleSheet("""
            QFrame#buildingOverlay {
                background-color: rgba(255, 255, 255, 245);
                border: 1px solid #008000;
                border-radius: 8px;
            }
            QLabel {
                border: none;
            }
            QProgressBar {
                border: 1px solid #a0a0a0;
                border-radius: 5px;
                background-color: #f2f2f2;
                text-align: center;
                color: #222222;
                font-weight: bold;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #16a34a;
                border-radius: 4px;
            }
        """)

        building_layout = QVBoxLayout(self.building_overlay)
        building_layout.setContentsMargins(18, 12, 18, 12)
        building_layout.setSpacing(7)

        building_title = QLabel("Building Model...")
        building_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        building_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #008000;"
        )

        self.building_progress = QProgressBar()
        self.building_progress.setRange(0, 100)
        self.building_progress.setValue(0)
        self.building_progress.setFormat("%p%")

        self.building_stage = QLabel("Preparing model...")
        self.building_stage.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.building_stage.setStyleSheet(
            "font-size: 11px; color: #555555;"
        )

        building_layout.addWidget(building_title)
        building_layout.addWidget(self.building_progress)
        building_layout.addWidget(self.building_stage)

        self.building_timer = QTimer(self)
        self.building_timer.setInterval(120)
        self.building_timer.timeout.connect(self._advance_building_progress)
        self.building_overlay.hide()

        self.visual_tabs.setCurrentIndex(0)

        if hasattr(self.script_editor, 'editor'):
            self.highlighter = OpenMCSyntaxHighlighter(self.script_editor.editor.document())
        else:
            self.highlighter = OpenMCSyntaxHighlighter(self.script_editor.document())

    def check_script_syntax(self):
        code = self._get_script_text()
        try:
            compile(code, '<string>', 'exec')
            self.console_widget.append_log("✅ Syntax Check: No Python syntax errors found.")
            QMessageBox.information(self, "Syntax Check", "✅ Good job! Code is syntactically correct.")
        except SyntaxError as e:
            err_msg = f"❌ Syntax Error on line {e.lineno}:\n{e.msg}\n\nCode snippet:\n{e.text}"
            self.console_widget.append_log(err_msg)
            QMessageBox.critical(self, "Syntax Error", err_msg)
        except Exception as e:
            self.console_widget.append_log(f"⚠️ Syntax Check Error: {str(e)}")

    def _lazy_load_tabs(self, index):
        title = self.visual_tabs.tabText(index)

        if index == 0:
            self.statusbar.showMessage("Ready", 3000)
            return

        try:
            if "3D Voxel Viewer" in title and not hasattr(self, 'voxel_page_loaded'):
                self.statusbar.showMessage("⏳ Loading 3D Engine (PyVista/VTK)... Please wait", 0)
                QApplication.processEvents()

                from ui.plots.voxel_page import VoxelPageWidget

                self.voxel_page = VoxelPageWidget()
                self.voxel_page.script_generated.connect(self.script_editor.append_code)

                self.visual_tabs.blockSignals(True)
                self.visual_tabs.removeTab(index)
                self.visual_tabs.insertTab(index, self.voxel_page, "🧊 3D Voxel Viewer")
                self.visual_tabs.setCurrentIndex(index)
                self.visual_tabs.blockSignals(False)

                self.voxel_page_loaded = True
                self.statusbar.showMessage("Ready - 3D Voxel Viewer Activated", 3000)

            elif "Particle Tracks" in title and not hasattr(self, 'tracks_page_loaded'):
                self.statusbar.showMessage("⏳ Loading Tracks Engine...", 0)
                QApplication.processEvents()

                from ui.tracks.tracks_page import TracksPageWidget
                self.tracks_page = TracksPageWidget()
                self.tracks_page.script_generated.connect(self.script_editor.append_code)

                self.visual_tabs.blockSignals(True)
                self.visual_tabs.removeTab(index)
                self.visual_tabs.insertTab(index, self.tracks_page, "🎯 Particle Tracks")
                self.visual_tabs.setCurrentIndex(index)
                self.visual_tabs.blockSignals(False)

                self.tracks_page_loaded = True
                self.statusbar.showMessage("Ready - Particle Tracks Activated", 3000)

            elif "Simulation Results" in title and not hasattr(self, 'results_page_loaded'):
                self.statusbar.showMessage("⏳ Loading Data Analysis Engine (SciPy/Pandas)... Please wait", 0)
                QApplication.processEvents()

                from ui.results_page import ResultsPageWidget
                self.results_page = ResultsPageWidget()

                self.visual_tabs.blockSignals(True)
                self.visual_tabs.removeTab(index)
                self.visual_tabs.insertTab(index, self.results_page, "📊 Simulation Results")
                self.visual_tabs.setCurrentIndex(index)
                self.visual_tabs.blockSignals(False)

                self.results_page_loaded = True
                self.statusbar.showMessage("Ready - Simulation Results Activated", 3000)
            else:
                self.statusbar.showMessage(f"Ready - {title} Activated", 3000)

        except Exception as e:
            self.statusbar.showMessage("❌ Error loading module", 5000)
            if hasattr(self, 'console_widget'):
                self.console_widget.append_log(f"❌ Error loading tab '{title}': {str(e)}")
            QMessageBox.critical(self, "Module Load Error",
                                 f"Failed to load the module for '{title}'.\n\nError details:\n{str(e)}\n\nPlease check if required libraries (like pandas, scipy) are installed.")

    def _setup_dock_widgets(self):
        self.dock_project = QDockWidget("Project Explorer", self)
        self.project_tree = ProjectTreeWidget()
        self.dock_project.setWidget(self.project_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_project)
        self.dock_project.hide()

        self.dock_console = QDockWidget("Live Output Console", self)
        self.console_widget = ConsoleWidget()

        navy_blue_style = "background-color: #0e305a !important; color: #ffffff !important; font-family: Consolas, monospace;"
        self.console_widget.setStyleSheet(
            f"QWidget {{ {navy_blue_style} }} QTextEdit {{ {navy_blue_style} }} QPlainTextEdit {{ {navy_blue_style} }}")

        for child in self.console_widget.findChildren(QTextEdit):
            child.setStyleSheet(navy_blue_style)
        for child in self.console_widget.findChildren(QPlainTextEdit):
            child.setStyleSheet(navy_blue_style)

        self.dock_console.setWidget(self.console_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_console)

        self.geometry_page.log_signal.connect(self.console_widget.append_log)

    def _setup_actions(self):
        style = QApplication.style()
        self.action_new = QAction(style.standardIcon(QStyle.StandardPixmap.SP_FileIcon), "New Project", self)
        self.action_new.triggered.connect(lambda checked=False: self.new_project())
        self.action_open = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open Project...",
                                   self)
        self.action_open.triggered.connect(lambda checked=False: self.open_project())

        self.action_save = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
        self.action_save.setShortcut("Ctrl+S")
        self.action_save.triggered.connect(lambda checked=False: self.save_project(None))

        self.action_save_as = QAction("💾 Save Project As...", self)
        self.action_save_as.setShortcut("Ctrl+Shift+S")
        self.action_save_as.triggered.connect(lambda checked=False: self.save_project_as())

        self.action_exit = QAction("❌ Exit", self)
        self.action_exit.setShortcut("Alt+F4")
        self.action_exit.triggered.connect(lambda checked=False: self.close())

        self.action_validate_project = QAction("🛡️ Validate Project", self)
        self.action_validate_project.setShortcut("Ctrl+V")
        self.action_validate_project.triggered.connect(lambda checked=False: self.run_project_validation())

        self.action_run = QAction("▶️ Run OpenMC", self)
        self.action_run.setShortcut("F5")
        self.action_run.triggered.connect(lambda checked=False: self._run_openmc_simulation())
        self.action_validate = QAction("✔️ Validate Geometry & Plot", self)
        self.action_validate.triggered.connect(lambda checked=False: self._generate_and_show_plot())
        self.action_about = QAction("ℹ️ About OpenMC Studio", self)
        self.action_about.triggered.connect(lambda checked=False: self.show_about_dialog())
        self.action_theme = QAction("🌗 Toggle Grayscale Theme", self)
        self.action_theme.triggered.connect(lambda checked=False: self.toggle_theme())

        self.action_xs_plot = QAction("📉 Cross Section Plotter", self)
        self.action_xs_plot.triggered.connect(lambda checked=False: self.open_xs_plotter())

        # --- Script background color commands ---
        self.action_bg_default = QAction("🟠 Default Orange", self)
        self.action_bg_default.triggered.connect(lambda checked=False: self.change_script_bg("#FF8C00", "#000000"))

        self.action_bg_dark = QAction("🌑 Dark Space", self)
        self.action_bg_dark.triggered.connect(lambda checked=False: self.change_script_bg("#1E1E1E", "#E0E0E0"))

        self.action_bg_navy = QAction("🌌 Deep Navy", self)
        self.action_bg_navy.triggered.connect(lambda checked=False: self.change_script_bg("#0A192F", "#E0E0E0"))

        self.action_bg_white = QAction("📝 Classic White", self)
        self.action_bg_white.triggered.connect(lambda checked=False: self.change_script_bg("#FFFFFF", "#000000"))

        self.action_bg_hacker = QAction("🌲 Terminal Green", self)
        self.action_bg_hacker.triggered.connect(lambda checked=False: self.change_script_bg("#051505", "#00FF41"))

        # --- Built-in template commands ---
        self.act_ex_pwr = QAction("PWR Pin Cell", self)
        self.act_ex_pwr.triggered.connect(lambda checked=False: self.load_example("PWR Pin Cell"))

        self.act_ex_godiva = QAction("Godiva Fast Reactor", self)
        self.act_ex_godiva.triggered.connect(lambda checked=False: self.load_example("Godiva Sphere"))

        self.act_ex_1 = QAction("Infinite Medium", self)
        self.act_ex_1.triggered.connect(lambda checked=False, _name="Infinite Medium": self.load_example(_name))

        self.act_ex_2 = QAction("Slab Geometry", self)
        self.act_ex_2.triggered.connect(lambda checked=False, _name="Slab Geometry": self.load_example(_name))

        self.act_ex_3 = QAction("Bare Sphere", self)
        self.act_ex_3.triggered.connect(lambda checked=False, _name="Bare Sphere": self.load_example(_name))

        self.act_ex_4 = QAction("BWR Pin Cell", self)
        self.act_ex_4.triggered.connect(lambda checked=False, _name="BWR Pin Cell": self.load_example(_name))

        self.act_ex_5 = QAction("PWR Fuel Assembly", self)
        self.act_ex_5.triggered.connect(lambda checked=False, _name="PWR Fuel Assembly": self.load_example(_name))

        self.act_ex_6 = QAction("C5G7 Pin Cell", self)
        self.act_ex_6.triggered.connect(lambda checked=False, _name="C5G7 Pin Cell": self.load_example(_name))

        self.act_ex_7 = QAction("TRISO Fuel Particle", self)
        self.act_ex_7.triggered.connect(lambda checked=False, _name="TRISO Fuel Particle": self.load_example(_name))

        self.act_ex_8 = QAction("HTGR Fuel Compact", self)
        self.act_ex_8.triggered.connect(lambda checked=False, _name="HTGR Fuel Compact": self.load_example(_name))

        # --- Workspace switching commands ---
        self.act_ws_2d = QAction("🎨 2D Geometry Viewer", self)
        self.act_ws_2d.triggered.connect(lambda checked=False: self.visual_tabs.setCurrentIndex(1))

        self.act_ws_3d = QAction("🧊 3D Voxel Viewer", self)
        self.act_ws_3d.triggered.connect(lambda checked=False: self.visual_tabs.setCurrentIndex(2))

        self.act_ws_tracks = QAction("🎯 Particle Tracks", self)
        self.act_ws_tracks.triggered.connect(lambda checked=False: self.visual_tabs.setCurrentIndex(3))

        self.act_ws_results = QAction("📊 Simulation Results", self)
        self.act_ws_results.triggered.connect(lambda checked=False: self.visual_tabs.setCurrentIndex(4))

    def _setup_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_new)
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_save)
        file_menu.addAction(self.action_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        build_menu = menubar.addMenu("&Build (Setup)")
        act_mat = QAction("🧪 Materials Editor", self)
        act_mat.triggered.connect(lambda checked=False: self.win_materials.show())
        build_menu.addAction(act_mat)
        act_geo = QAction("📐 Geometry Builder", self)
        act_geo.triggered.connect(lambda checked=False: self.win_geometry.show())
        build_menu.addAction(act_geo)
        act_set = QAction("⚙️ Simulation Settings", self)
        act_set.triggered.connect(lambda checked=False: self.win_settings.show())
        build_menu.addAction(act_set)
        act_tal = QAction("🎯 Tallies & Detectors", self)
        act_tal.triggered.connect(lambda checked=False: self.win_tallies.show())
        build_menu.addAction(act_tal)

        build_menu.addSeparator()
        act_dep = QAction("🔥 Burnup & Depletion", self)
        act_dep.triggered.connect(lambda checked=False: self.win_depletion.show())
        build_menu.addAction(act_dep)

        build_menu.addSeparator()
        act_lib = QAction("📚 Data Libraries Manager", self)
        act_lib.triggered.connect(lambda checked=False: self.win_libraries.show())
        build_menu.addAction(act_lib)

        build_menu.addSeparator()
        build_menu.addAction(self.action_validate_project)

        data_menu = menubar.addMenu("&Data")
        data_menu.addAction(self.action_xs_plot)

        examples_menu = menubar.addMenu("&Examples")
        examples_menu.addAction(self.act_ex_pwr)
        examples_menu.addAction(self.act_ex_godiva)
        examples_menu.addSeparator()

        basic_menu = examples_menu.addMenu("Basic")
        basic_menu.addAction(self.act_ex_1)
        basic_menu.addAction(self.act_ex_2)
        basic_menu.addAction(self.act_ex_3)

        reactor_menu = examples_menu.addMenu("Reactor Physics")
        reactor_menu.addAction(self.act_ex_4)
        reactor_menu.addAction(self.act_ex_5)
        reactor_menu.addAction(self.act_ex_6)

        advanced_menu = examples_menu.addMenu("Advanced Reactors")
        advanced_menu.addAction(self.act_ex_7)
        advanced_menu.addAction(self.act_ex_8)

        ws_menu = menubar.addMenu("&Workspaces")
        ws_menu.addAction(self.act_ws_2d)
        ws_menu.addAction(self.act_ws_3d)
        ws_menu.addAction(self.act_ws_tracks)
        ws_menu.addSeparator()
        ws_menu.addAction(self.act_ws_results)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.dock_project.toggleViewAction())
        view_menu.addAction(self.dock_console.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self.action_theme)

        view_menu.addSeparator()
        bg_menu = view_menu.addMenu("🎨 Script Background Color")
        bg_menu.addAction(self.action_bg_default)
        bg_menu.addAction(self.action_bg_dark)
        bg_menu.addAction(self.action_bg_navy)
        bg_menu.addAction(self.action_bg_white)
        bg_menu.addAction(self.action_bg_hacker)

        run_menu = menubar.addMenu("&Run")
        run_menu.addAction(self.action_validate)
        run_menu.addSeparator()
        run_menu.addAction(self.action_run)

        help_menu = menubar.addMenu("&About")
        help_menu.addAction(self.action_about)

    def _setup_toolbars(self):
        self.main_toolbar = QToolBar("Main Toolbar")
        self.main_toolbar.setIconSize(QSize(24, 24))
        self.main_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.main_toolbar)
        self.main_toolbar.addAction(self.action_new)
        self.main_toolbar.addAction(self.action_open)
        self.main_toolbar.addAction(self.action_save)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.action_validate)
        self.main_toolbar.addAction(self.action_validate_project)
        self.main_toolbar.addSeparator()

        self.openmc_status_label = QLabel(" ⏳ Checking OpenMC... ")
        self.openmc_status_label.setStyleSheet("color: #FF8C00; font-weight: bold; font-size: 13px; padding: 0 10px;")
        self.main_toolbar.addWidget(self.openmc_status_label)

        self.main_toolbar.addAction(self.action_run)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready", 5000)

    def run_project_validation(self):
        self.statusbar.showMessage("🛡️ Running project validation...", 0)

        problems = []
        void_cells_count = 0

        try:
            subprocess.run(['openmc', '--version'], capture_output=True, check=True, text=True, encoding='utf-8',
                           errors='replace')
        except Exception:
            problems.append(("OpenMC", "OpenMC executable not found in PATH."))

        code = self._get_script_text()
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            problems.append(("Python Script", f"Syntax error on line {e.lineno}: {e.msg}"))

        namespace = {'openmc': openmc}
        try:
            code_safe = re.sub(r'\bopenmc\.run\s*\(\s*\)', 'pass', code)
            code_safe = re.sub(r'\bopenmc\.plot_geometry\s*\(\s*\)', 'pass', code_safe)

            exec(code_safe, namespace)

            has_material_filled_cells = False
            if 'geometry' in namespace:
                geom = namespace['geometry']
                for cell_id, cell in geom.get_all_cells().items():
                    if cell.fill is None:
                        void_cells_count += 1
                    elif isinstance(cell.fill, openmc.Material):
                        has_material_filled_cells = True
            else:
                problems.append(("Geometry", "Geometry object not found. Ensure it is named 'geometry'."))

            if has_material_filled_cells and ('materials' not in namespace or len(namespace['materials']) == 0):
                problems.append(("Materials", "Cells have materials, but 'materials' collection is missing or empty."))

        except Exception as e:
            problems.append(("XML Generation / Logic", f"Runtime error during script execution: {str(e)}"))

        if not problems:
            success_msg = "✅ Project validation successful!\nAll checks passed."
            if void_cells_count > 0:
                success_msg += f"\n\nℹ️ Note: {void_cells_count} cell(s) are recognized as Vacuum/Void."

            QMessageBox.information(self, "Validation", success_msg)
            self.console_widget.append_log("✅ Project validation successful.")
            if void_cells_count > 0:
                self.console_widget.append_log(f"ℹ️ {void_cells_count} cell(s) safely identified as Vacuum/Void.")
        else:
            msg = f"❌ {len(problems)} problems found:\n\n"
            for source, desc in problems:
                msg += f"• [{source}]: {desc}\n"

            QMessageBox.critical(self, "Project Validation", msg)
            self.console_widget.append_log(msg.replace("❌", "⚠️"))

        self.statusbar.showMessage("Validation complete.", 3000)

    def open_xs_plotter(self):
        if not hasattr(self, 'xs_plot_dialog'):
            self.xs_plot_dialog = CrossSectionPlotDialog(self)
        self.xs_plot_dialog.show()
        self.xs_plot_dialog.raise_()
        self.xs_plot_dialog.activateWindow()

    def insert_macrobody(self):
        shape = self.macro_combo.currentText()
        code = ""
        if "RCC" in shape:
            code = "\n# MCNP RCC Equivalent (Right Circular Cylinder)\nmy_rcc = openmc.model.RightCircularCylinder(center_base=(0.0, 0.0, 0.0), height=10.0, radius=5.0)\n"
        elif "RPP" in shape:
            code = "\n# MCNP RPP Equivalent (Rectangular Parallelepiped)\nmy_rpp = openmc.model.RectangularParallelepiped(xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0, zmin=-5.0, zmax=5.0)\n"
        elif "SPH" in shape:
            code = "\n# MCNP SPH Equivalent (Sphere)\nmy_sph = openmc.Sphere(x0=0.0, y0=0.0, z0=0.0, r=5.0)\n"
        elif "TRC" in shape:
            code = "\n# MCNP TRC Equivalent (Truncated Right Circular Cone)\nmy_trc = openmc.model.TruncatedRightCircularCone(center_base=(0.0, 0.0, 0.0), height=10.0, radius_base=5.0, radius_top=2.0)\n"

        self.script_editor.editor.insertPlainText(code)
        self.script_editor.editor.setFocus()
        self.console_widget.append_log(f"✅ Inserted {shape} Macrobody into script.")

    def new_project(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        initial_text = f"# c      Created on: {now_str}\n\nimport openmc\nopenmc.reset_auto_ids()\n\n"

        if hasattr(self.script_editor, 'editor'):
            self.script_editor.editor.setPlainText(initial_text)
        else:
            self.script_editor.setPlainText(initial_text)

        if hasattr(self, 'console_widget'):
            try:
                self.console_widget.clear_log()
            except Exception:
                pass

        if hasattr(self, 'plots_page'):
            self.plots_page.deleteLater()

        if hasattr(self, 'voxel_page_loaded'):
            if hasattr(self.voxel_page, 'plotter'):
                try:
                    self.voxel_page.plotter.close()
                except Exception:
                    pass
            self.voxel_page.deleteLater()
            del self.voxel_page_loaded

        if hasattr(self, 'tracks_page_loaded'):
            self.tracks_page.deleteLater()
            del self.tracks_page_loaded

        if hasattr(self, 'results_page_loaded'):
            self.results_page.deleteLater()
            del self.results_page_loaded

        self.visual_tabs.blockSignals(True)
        self.visual_tabs.clear()

        self.welcome_page = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_page)
        welcome_logo = QLabel("OpenMC Studio")
        welcome_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_logo.setStyleSheet("font-size: 32px; color: #008000; font-weight: bold; border: none;")
        welcome_desc = QLabel("Select a Workspace from the menu above to begin")
        welcome_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_desc.setStyleSheet("font-size: 18px; color: #555555; border: none;")
        welcome_layout.addStretch()
        welcome_layout.addWidget(welcome_logo)
        welcome_layout.addWidget(welcome_desc)
        welcome_layout.addStretch()

        self.plots_page = PlotsPageWidget()
        self.plots_page.script_generated.connect(self.script_editor.append_code)

        self.visual_tabs.addTab(self.welcome_page, "🏠 Home")
        self.visual_tabs.addTab(self.plots_page, "🎨 2D Geometry Viewer")
        self.visual_tabs.addTab(QWidget(), "🧊 3D Voxel Viewer")
        self.visual_tabs.addTab(QWidget(), "🎯 Particle Tracks")
        self.visual_tabs.addTab(QWidget(), "📊 Simulation Results")

        self.visual_tabs.blockSignals(False)
        self.visual_tabs.setCurrentIndex(0)

        self._setup_tool_windows()
        self.geometry_page.log_signal.connect(self.console_widget.append_log)
        self.statusbar.showMessage("New project started -- all views reset.", 5000)

    def load_example(self, example_name):
        reply = QMessageBox.question(self, "Load Example",
                                     f"Are you sure you want to load the '{example_name}' example?\nThis will overwrite your current script.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            code = ""
            if example_name == "PWR Pin Cell":
                code = self._get_pwr_pin_cell_code()
            elif example_name == "Godiva Sphere":
                code = self._get_godiva_code()
            elif example_name == "Infinite Medium":
                code = self._get_example_1_code()
            elif example_name == "Slab Geometry":
                code = self._get_example_2_code()
            elif example_name == "Bare Sphere":
                code = self._get_example_3_code()
            elif example_name == "BWR Pin Cell":
                code = self._get_example_4_code()
            elif example_name == "PWR Fuel Assembly":
                code = self._get_example_5_code()
            elif example_name == "C5G7 Pin Cell":
                code = self._get_example_6_code()
            elif example_name == "TRISO Fuel Particle":
                code = self._get_example_7_code()
            elif example_name == "HTGR Fuel Compact":
                code = self._get_example_8_code()

            if code:
                if "import openmc" in code and "openmc.reset_auto_ids()" not in code:
                    code = code.replace("import openmc", "import openmc\nopenmc.reset_auto_ids()")

                if hasattr(self.script_editor, 'editor'):
                    self.script_editor.editor.setPlainText(code)
                else:
                    self.script_editor.setPlainText(code)

                self.console_widget.append_log(f"✅ Loaded Example: {example_name}")
                self.statusbar.showMessage(f"Loaded {example_name}", 5000)

    def _get_pwr_pin_cell_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
import openmc
openmc.reset_auto_ids()

# --- Materials ---
uo2 = openmc.Material(name='UO2 fuel')
uo2.add_element('U', 1.0, enrichment=3.0)
uo2.add_element('O', 2.0)
uo2.set_density('g/cm3', 10.0)

water = openmc.Material(name='Water')
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 1.0)
water.add_s_alpha_beta('c_H_in_H2O')

zircaloy = openmc.Material(name='Zircaloy')
zircaloy.add_element('Zr', 1.0)
zircaloy.set_density('g/cm3', 6.6)

materials = openmc.Materials([uo2, zircaloy, water])

# --- Geometry ---
fuel_or = openmc.ZCylinder(r=0.39)
clad_ir = openmc.ZCylinder(r=0.40)
clad_or = openmc.ZCylinder(r=0.46)

box = openmc.model.RectangularParallelepiped(
    -0.63, 0.63, -0.63, 0.63, -10.0, 10.0,
    boundary_type='reflective'
)

fuel_cell = openmc.Cell(fill=uo2, region=-fuel_or & -box)
gap_cell = openmc.Cell(region=+fuel_or & -clad_ir & -box)
clad_cell = openmc.Cell(fill=zircaloy, region=+clad_ir & -clad_or & -box)
water_cell = openmc.Cell(fill=water, region=+clad_or & -box)

root_univ = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell])
geometry = openmc.Geometry(root_univ)

# --- Settings ---
settings = openmc.Settings()
settings.batches = 100
settings.inactive = 20
settings.particles = 1000

bounds = [-0.63, -0.63, -10, 0.63, 0.63, 10]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
settings.source = openmc.IndependentSource(space=uniform_dist)
"""

    def _get_godiva_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
import openmc
openmc.reset_auto_ids()

# --- Materials ---
pu = openmc.Material(name='Godiva enriched Uranium')
pu.add_nuclide('U234', 0.01024, 'wo')
pu.add_nuclide('U235', 0.93737, 'wo')
pu.add_nuclide('U238', 0.05239, 'wo')
pu.set_density('g/cm3', 18.74)
materials = openmc.Materials([pu])

# --- Geometry ---
sph = openmc.Sphere(r=8.7407, boundary_type='vacuum')
sphere_cell = openmc.Cell(fill=pu, region=-sph)

root_univ = openmc.Universe(cells=[sphere_cell])
geometry = openmc.Geometry(root_univ)

# --- Settings ---
settings = openmc.Settings()
settings.batches = 100
settings.inactive = 20
settings.particles = 5000

settings.source = openmc.IndependentSource(space=openmc.stats.Point())
"""

    def _get_example_1_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: Infinite Medium
import openmc
openmc.reset_auto_ids()

material = openmc.Material(name='Water')
material.add_nuclide('H1', 2.0)
material.add_nuclide('O16', 1.0)
material.set_density('g/cm3', 1.0)
materials = openmc.Materials([material])

box = openmc.model.RectangularParallelepiped(
    -10.0, 10.0, -10.0, 10.0, -10.0, 10.0,
    boundary_type='reflective'
)
cell = openmc.Cell(fill=material, region=-box)
geometry = openmc.Geometry(openmc.Universe(cells=[cell]))

settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 1000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box([-5, -5, -5], [5, 5, 5])
)
"""

    def _get_example_2_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: Slab Geometry
import openmc
openmc.reset_auto_ids()

fuel = openmc.Material(name='UO2 Fuel')
fuel.add_element('U', 1.0, enrichment=3.0)
fuel.add_element('O', 2.0)
fuel.set_density('g/cm3', 10.0)
materials = openmc.Materials([fuel])

box = openmc.model.RectangularParallelepiped(
    -5.0, 5.0, -10.0, 10.0, -10.0, 10.0
)
box.xmin.boundary_type = 'vacuum'
box.xmax.boundary_type = 'vacuum'
box.ymin.boundary_type = 'reflective'
box.ymax.boundary_type = 'reflective'
box.zmin.boundary_type = 'reflective'
box.zmax.boundary_type = 'reflective'

cell = openmc.Cell(fill=fuel, region=-box)
geometry = openmc.Geometry(openmc.Universe(cells=[cell]))

settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 1000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box([-4.9, -9.0, -9.0], [4.9, 9.0, 9.0])
)
"""

    def _get_example_3_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: Bare Sphere
import openmc
openmc.reset_auto_ids()

u235 = openmc.Material(name='Enriched Uranium')
u235.add_nuclide('U235', 1.0)
u235.set_density('g/cm3', 18.5)
materials = openmc.Materials([u235])

sphere = openmc.Sphere(r=6.0, boundary_type='vacuum')
cell = openmc.Cell(fill=u235, region=-sphere)
geometry = openmc.Geometry(openmc.Universe(cells=[cell]))

settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 2000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Point((0.0, 0.0, 0.0))
)
"""

    def _get_example_4_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: BWR Pin Cell
import openmc
openmc.reset_auto_ids()

fuel = openmc.Material(name='UO2 Fuel')
fuel.add_element('U', 1.0, enrichment=2.4)
fuel.add_element('O', 2.0)
fuel.set_density('g/cm3', 10.2)

clad = openmc.Material(name='Zircaloy-2')
clad.add_element('Zr', 1.0)
clad.set_density('g/cm3', 6.55)

water = openmc.Material(name='Light Water')
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 0.74)
water.add_s_alpha_beta('c_H_in_H2O')

materials = openmc.Materials([fuel, clad, water])

fuel_or = openmc.ZCylinder(r=0.41)
clad_or = openmc.ZCylinder(r=0.475)
box = openmc.model.RectangularParallelepiped(
    -0.64, 0.64, -0.64, 0.64, -5.0, 5.0,
    boundary_type='reflective'
)

fuel_cell = openmc.Cell(fill=fuel, region=-fuel_or & -box)
clad_cell = openmc.Cell(fill=clad, region=+fuel_or & -clad_or & -box)
water_cell = openmc.Cell(fill=water, region=+clad_or & -box)

geometry = openmc.Geometry(
    openmc.Universe(cells=[fuel_cell, clad_cell, water_cell])
)

settings = openmc.Settings()
settings.batches = 80
settings.inactive = 20
settings.particles = 1500
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box([-0.6, -0.6, -4.0], [0.6, 0.6, 4.0])
)
"""

    def _get_example_5_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: PWR Fuel Assembly
import openmc
openmc.reset_auto_ids()

# --- Materials ---
fuel = openmc.Material(name='UO2 Fuel')
fuel.add_element('U', 1.0, enrichment=3.1)
fuel.add_element('O', 2.0)
fuel.set_density('g/cm3', 10.3)

water = openmc.Material(name='Water')
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 0.72)
water.add_s_alpha_beta('c_H_in_H2O')

zircaloy = openmc.Material(name='Zircaloy')
zircaloy.add_element('Zr', 1.0)
zircaloy.set_density('g/cm3', 6.55)

materials = openmc.Materials([fuel, water, zircaloy])

# --- Geometry ---
fuel_or = openmc.ZCylinder(r=0.4095)
clad_or = openmc.ZCylinder(r=0.475)

# Outer universe for the lattice
water_outer = openmc.Cell(fill=water)
outer_univ = openmc.Universe(cells=[water_outer])

fuel_cell = openmc.Cell(fill=fuel, region=-fuel_or)
clad_cell = openmc.Cell(fill=zircaloy, region=+fuel_or & -clad_or)
water_cell = openmc.Cell(fill=water, region=+clad_or)
pin_universe = openmc.Universe(cells=[fuel_cell, clad_cell, water_cell])

lattice = openmc.RectLattice()
lattice.lower_left = (-5.67, -5.67)
lattice.pitch = (1.26, 1.26)
lattice.universes = [[pin_universe] * 9 for _ in range(9)]
lattice.outer = outer_univ

outer_box = openmc.model.RectangularParallelepiped(
    -5.67, 5.67, -5.67, 5.67, -20.0, 20.0,
    boundary_type='reflective'
)
assembly_cell = openmc.Cell(fill=lattice, region=-outer_box)
geometry = openmc.Geometry(openmc.Universe(cells=[assembly_cell]))

# --- Settings ---
settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 1000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box([-5.5, -5.5, -10], [5.5, 5.5, 10])
)
"""

    def _get_example_6_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: C5G7 Pin Cell Demonstration
import openmc
openmc.reset_auto_ids()

fuel = openmc.Material(name='UO2 Fuel')
fuel.add_element('U', 1.0, enrichment=4.0)
fuel.add_element('O', 2.0)
fuel.set_density('g/cm3', 10.1)

clad = openmc.Material(name='Zircaloy')
clad.add_element('Zr', 1.0)
clad.set_density('g/cm3', 6.55)

water = openmc.Material(name='Water')
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 0.7)
water.add_s_alpha_beta('c_H_in_H2O')

materials = openmc.Materials([fuel, clad, water])

fuel_or = openmc.ZCylinder(r=0.54)
clad_or = openmc.ZCylinder(r=0.62)
box = openmc.model.RectangularParallelepiped(
    -0.63, 0.63, -0.63, 0.63, -10.0, 10.0,
    boundary_type='reflective'
)

cells = [
    openmc.Cell(fill=fuel, region=-fuel_or & -box),
    openmc.Cell(fill=clad, region=+fuel_or & -clad_or & -box),
    openmc.Cell(fill=water, region=+clad_or & -box),
]
geometry = openmc.Geometry(openmc.Universe(cells=cells))

settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 1000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box([-0.5, -0.5, -5], [0.5, 0.5, 5])
)
"""

    def _get_example_7_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: TRISO Fuel Particle
import openmc
openmc.reset_auto_ids()

kernel = openmc.Material(name='UO2 Kernel')
kernel.add_element('U', 1.0, enrichment=8.0)
kernel.add_element('O', 2.0)
kernel.set_density('g/cm3', 10.5)

buffer = openmc.Material(name='Porous Carbon Buffer')
buffer.add_element('C', 1.0)
buffer.set_density('g/cm3', 1.0)

pyc = openmc.Material(name='Pyrolytic Carbon')
pyc.add_element('C', 1.0)
pyc.set_density('g/cm3', 1.9)

sic = openmc.Material(name='Silicon Carbide')
sic.add_element('Si', 1.0)
sic.add_element('C', 1.0)
sic.set_density('g/cm3', 3.2)

materials = openmc.Materials([kernel, buffer, pyc, sic])

r1 = openmc.Sphere(r=0.25)
r2 = openmc.Sphere(r=0.35)
r3 = openmc.Sphere(r=0.385)
r4 = openmc.Sphere(r=0.42)

box = openmc.model.RectangularParallelepiped(
    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
    boundary_type='reflective'
)

c1 = openmc.Cell(fill=kernel, region=-r1 & -box)
c2 = openmc.Cell(fill=buffer, region=+r1 & -r2 & -box)
c3 = openmc.Cell(fill=pyc, region=+r2 & -r3 & -box)
c4 = openmc.Cell(fill=sic, region=+r3 & -r4 & -box)
c5 = openmc.Cell(fill=pyc, region=+r4 & -box)

geometry = openmc.Geometry(
    openmc.Universe(cells=[c1, c2, c3, c4, c5])
)

settings = openmc.Settings()
settings.batches = 40
settings.inactive = 10
settings.particles = 1000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Point((0.0, 0.0, 0.0))
)
"""

    def _get_example_8_code(self):
        now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        return f"""# c      Created on: {now_str}
# c      OpenMC Studio Example: HTGR Fuel Compact Demonstration
import openmc
openmc.reset_auto_ids()

fuel = openmc.Material(name='TRISO UO2')
fuel.add_element('U', 1.0, enrichment=8.0)
fuel.add_element('O', 2.0)
fuel.set_density('g/cm3', 10.5)

graphite = openmc.Material(name='Graphite')
graphite.add_element('C', 1.0)
graphite.set_density('g/cm3', 1.7)

helium = openmc.Material(name='Helium')
helium.add_element('He', 1.0)
helium.set_density('g/cm3', 0.00018)

materials = openmc.Materials([fuel, graphite, helium])

fuel_or = openmc.ZCylinder(r=0.25)
compact_or = openmc.ZCylinder(r=1.0)

# Box to reflect and bound the entire geometry
box = openmc.model.RectangularParallelepiped(
    -1.1, 1.1, -1.1, 1.1, -5.0, 5.0,
    boundary_type='reflective'
)

fuel_cell = openmc.Cell(fill=fuel, region=-fuel_or & -box)
graphite_cell = openmc.Cell(fill=graphite, region=+fuel_or & -compact_or & -box)
helium_cell = openmc.Cell(fill=helium, region=+compact_or & -box)

geometry = openmc.Geometry(
    openmc.Universe(cells=[fuel_cell, graphite_cell, helium_cell])
)

settings = openmc.Settings()
settings.batches = 40
settings.inactive = 10
settings.particles = 1000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8])
)
"""

    # --- Save project in advanced JSON format ---
    def save_project(self, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Project", "",
                                                      "OpenMC Studio Project (*.omcs);;Python Files (*.py)")

        if filepath:
            try:
                script_text = self._get_script_text()
                now_str = datetime.now().strftime("%A, %B %d, %Y at %H:%M")

                if filepath.endswith('.omcs'):
                    # Save as JSON database to ensure UI states are saved (if any)
                    ui_state = {}
                    for name, widget in [
                        ('materials', self.materials_page),
                        ('geometry', self.geometry_page),
                        ('settings', self.settings_page),
                        ('tallies', self.tallies_page),
                        ('libraries', self.libraries_page)
                    ]:
                        if hasattr(widget, 'save_state'):
                            ui_state[name] = widget.save_state()
                        elif hasattr(widget, 'to_dict'):
                            ui_state[name] = widget.to_dict()

                    project_data = {
                        "app": "OpenMC Studio",
                        "version": "1.0",
                        "timestamp": now_str,
                        "script": script_text,
                        "ui_state": ui_state
                    }
                    if hasattr(self.project_manager, 'to_dict'):
                        project_data['project_manager'] = self.project_manager.to_dict()

                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(project_data, f, indent=4)
                else:
                    # Save as standard Python file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(script_text)

                self.statusbar.showMessage(f"Project saved: {os.path.basename(filepath)}", 5000)
                self.console_widget.append_log(f"💾 Project saved successfully: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{str(e)}")

    def save_project_as(self):
        self.save_project(None)

    # --- Smart project restoration (JSON or standard Python) ---
    def open_project(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Project", "",
                                                  "OpenMC Studio Project (*.omcs);;Python Files (*.py);;All Files (*)")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                script_text = content
                is_full_project = False

                if filepath.endswith('.omcs'):
                    try:
                        project_data = json.loads(content)
                        if isinstance(project_data, dict) and "script" in project_data:
                            script_text = project_data["script"]
                            is_full_project = True

                            # Restore UI states
                            ui_state = project_data.get("ui_state", {})
                            for name, widget in [
                                ('materials', self.materials_page),
                                ('geometry', self.geometry_page),
                                ('settings', self.settings_page),
                                ('tallies', self.tallies_page),
                                ('libraries', self.libraries_page)
                            ]:
                                if name in ui_state:
                                    if hasattr(widget, 'load_state'):
                                        widget.load_state(ui_state[name])
                                    elif hasattr(widget, 'from_dict'):
                                        widget.from_dict(ui_state[name])

                            if 'project_manager' in project_data and hasattr(self.project_manager, 'from_dict'):
                                self.project_manager.from_dict(project_data['project_manager'])
                    except json.JSONDecodeError:
                        pass  # If it's an old file that doesn't support JSON, open it as usual

                if hasattr(self.script_editor, 'editor') and hasattr(self.script_editor.editor, 'setPlainText'):
                    self.script_editor.editor.setPlainText(script_text)
                elif hasattr(self.script_editor, 'setPlainText'):
                    self.script_editor.setPlainText(script_text)

                self.statusbar.showMessage(f"Project loaded: {os.path.basename(filepath)}", 5000)

                if hasattr(self, 'console_widget'):
                    if is_full_project:
                        self.console_widget.append_log(
                            f"📂 Successfully loaded FULL project: {os.path.basename(filepath)}")
                    else:
                        self.console_widget.append_log(
                            f"📂 Successfully loaded Python Script: {os.path.basename(filepath)}")

            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load project file:\n{str(e)}")

    def show_about_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About OpenMC Studio")
        dialog.setFixedSize(500, 380)

        primary = "#008000" if self.current_theme == "colored" else "#555555"
        secondary = "#FF8C00" if self.current_theme == "colored" else "#777777"
        primary_hover = "#005500" if self.current_theme == "colored" else "#333333"

        dialog.setStyleSheet(f"""
            QDialog {{ background-color: #FFFFFF; }}
            QLabel {{ color: #000000; font-size: 14px; }}
            QPushButton {{ background-color: {primary}; color: white; padding: 8px 15px; font-weight: bold; border-radius: 4px; font-size: 13px;}}
            QPushButton:hover {{ background-color: {primary_hover}; }}
        """)
        layout = QVBoxLayout(dialog)
        title_label = QLabel("<b>OpenMC Studio v1.0</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"font-size: 22px; color: {primary}; margin-bottom: 5px;")
        layout.addWidget(title_label)

        info_text = (
            "<div style='line-height: 1.5;'>"
            "<b>Developer & Copyright:</b><br>"
            "Copyright © 2026 Dr. Ayman Abu Ghazal<br><br>"
            "<b>Affiliation:</b><br>"
            "Jordan Atomic Energy Commission (JAEC)<br>"
            "Amman, Jordan"
            "</div>"
        )
        info_label = QLabel(info_text)
        info_label.setOpenExternalLinks(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        citation_group = QGroupBox("How to Cite (Research Papers)")
        citation_group.setStyleSheet(
            f"QGroupBox {{ border: 2px solid {secondary}; border-radius: 5px; margin-top: 15px; color: {secondary}; font-weight: bold; }}")
        cit_layout = QVBoxLayout(citation_group)
        citation_text = "Abu Ghazal, A. (2026). OpenMC Studio [Computer software]. Jordan Atomic Energy Commission, Amman, Jordan."
        cit_label = QLabel(citation_text)
        cit_label.setWordWrap(True)
        cit_label.setStyleSheet("font-style: italic; color: #333333; font-size: 13px; padding: 5px;")
        cit_layout.addWidget(cit_label)

        btn_copy = QPushButton("📋 Copy Citation")
        btn_copy.setStyleSheet(
            f"background-color: {secondary}; color: white; padding: 6px; font-weight: bold; border-radius: 4px;")

        def copy_to_clipboard():
            QApplication.clipboard().setText(citation_text)
            btn_copy.setText("✔️ Copied!")
            btn_copy.setStyleSheet(
                f"background-color: {primary}; color: white; padding: 6px; font-weight: bold; border-radius: 4px;")

        btn_copy.clicked.connect(copy_to_clipboard)
        cit_layout.addWidget(btn_copy, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(citation_group)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)
        dialog.exec()

    def _get_script_text(self):
        if hasattr(self.script_editor, 'editor'):
            text = self.script_editor.editor.toPlainText()
        else:
            text = self.script_editor.toPlainText()

        if "openmc.reset_auto_ids()" not in text:
            if "import openmc" in text:
                text = re.sub(r'^(import openmc.*)$', r'\1\nopenmc.reset_auto_ids()', text, flags=re.MULTILINE)
            else:
                text = "import openmc\nopenmc.reset_auto_ids()\n" + text
        return text

    def _position_building_overlay(self):
        if not hasattr(self, 'building_overlay'):
            return

        target = None

        for name in (
                "plot_view", "image_label", "canvas", "plot_canvas",
                "image_view", "viewer", "plot_display", "display_widget"
        ):
            candidate = getattr(self.plots_page, name, None)
            if candidate is not None and hasattr(candidate, "rect"):
                target = candidate
                break

        if target is None:
            try:
                for child in self.plots_page.findChildren(QWidget):
                    name = child.objectName().lower()
                    cls = child.__class__.__name__.lower()
                    if (
                            "plot" in name or "plot" in cls or
                            "image" in name or "canvas" in name or
                            "viewer" in name
                    ) and child.width() > 200 and child.height() > 150:
                        target = child
                        break
            except Exception:
                target = None

        if target is None:
            target = self.plots_page

        if self.building_overlay.parentWidget() is not target:
            self.building_overlay.setParent(target)

        self.building_overlay.raise_()

        if target.width() > 0 and target.height() > 0:
            x = max(0, (target.width() - self.building_overlay.width()) // 2)
            y = max(0, (target.height() - self.building_overlay.height()) // 2)
            self.building_overlay.move(x, y)

    def _show_building_overlay(self):
        if not hasattr(self, 'building_overlay'):
            return
        self.visual_tabs.setCurrentIndex(1)
        QApplication.processEvents()
        self.building_progress.setValue(0)
        self.building_stage.setText("Preparing model...")
        self._position_building_overlay()
        self.building_overlay.raise_()
        self.building_overlay.show()
        self._position_building_overlay()
        self.building_timer.start()
        QApplication.processEvents()

    def _advance_building_progress(self):
        if not hasattr(self, 'building_progress') or not self.building_overlay.isVisible():
            return
        value = self.building_progress.value()
        if value < 88:
            self.building_progress.setValue(value + 1)

    def _set_building_stage(self, value, text):
        if hasattr(self, 'building_progress'):
            self.building_progress.setValue(max(0, min(100, value)))
        if hasattr(self, 'building_stage'):
            self.building_stage.setText(text)
        QApplication.processEvents()

    def _hide_building_overlay(self):
        if hasattr(self, 'building_timer'):
            self.building_timer.stop()
        if hasattr(self, 'building_overlay'):
            self.building_overlay.hide()
        if hasattr(self, 'plots_page'):
            self.plots_page.setEnabled(True)
        QApplication.processEvents()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'building_overlay') and self.building_overlay.isVisible():
            self._position_building_overlay()

    def _generate_and_show_plot(self):
        self._plot_request_id = getattr(self, '_plot_request_id', 0) + 1
        request_id = self._plot_request_id
        self.statusbar.showMessage("🔧 Building model, please wait...", 0)
        self._show_building_overlay()
        self.plots_page.setEnabled(False)
        self._set_building_stage(10, "Exporting OpenMC model...")
        script_text = self._get_script_text()
        worker = ExportWorker(script_text, os.getcwd())
        self._active_workers.append(worker)
        worker.finished_signal.connect(
            lambda success, message, export_path, total_batches, rid=request_id, w=worker:
            self._on_export_for_plot_finished(success, message, export_path, total_batches, rid, w))
        worker.start()

    def _on_export_for_plot_finished(self, success, message, export_path, total_batches, request_id, worker):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        if request_id != self._plot_request_id:
            return
        if not success:
            self._hide_building_overlay()
            self.statusbar.showMessage("Ready", 3000)
            QMessageBox.critical(self, "Export Error", message)
            return
        self._set_building_stage(55, "Generating geometry plot...")
        plot_worker = PlotWorker(export_path)
        self._active_workers.append(plot_worker)
        plot_worker.finished_signal.connect(
            lambda s, m, p, rid=request_id, w=plot_worker: self._on_plot_finished(s, m, p, rid, w))
        plot_worker.start()

    def _on_plot_finished(self, success, message, ppm_path, request_id, worker):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        if request_id != self._plot_request_id:
            self._hide_building_overlay()
            return
        self.statusbar.showMessage("Ready", 3000)
        if success:
            self._set_building_stage(100, "Plot ready")
            QTimer.singleShot(180, self._hide_building_overlay)
            self.plots_page.display_image(ppm_path)
            self.visual_tabs.setCurrentIndex(1)
            self.console_widget.append_log(f"✅ {message}")
            self.statusbar.showMessage("Ready - 2D Geometry Viewer Activated", 3000)
        else:
            self.console_widget.append_log(f"❌ Plot Error: {message}")

    def _run_openmc_simulation(self):
        try:
            text, ok = QInputDialog.getText(
                self,
                "Results File Name",
                "Enter a name for the results file (leave empty for default):",
                QLineEdit.EchoMode.Normal,
                ""
            )

            if not ok:
                return

            self.custom_sp_name = text.strip()
            if self.custom_sp_name and not self.custom_sp_name.endswith('.h5'):
                self.custom_sp_name += '.h5'

            if hasattr(self, 'results_page_loaded') and hasattr(self, 'results_page'):
                for attr in dir(self.results_page):
                    val = getattr(self.results_page, attr)
                    if hasattr(val, 'close') and ('StatePoint' in str(type(val)) or 'h5py' in str(type(val)).lower()):
                        try:
                            val.close()
                        except:
                            pass
            gc.collect()

            self.action_run.setEnabled(False)
            self.statusbar.showMessage("🔧 Building model, please wait...", 0)
            self._show_building_overlay()
            self.plots_page.setEnabled(False)
            self._set_building_stage(10, "Preparing simulation model...")
            self._run_request_id = getattr(self, '_run_request_id', 0) + 1
            request_id = self._run_request_id
            script_text = self._get_script_text()
            worker = ExportWorker(script_text, os.getcwd())
            self._active_workers.append(worker)
            worker.finished_signal.connect(
                lambda success, message, export_path, total_batches, rid=request_id, w=worker:
                self._on_export_for_run_finished(success, message, export_path, total_batches, rid, w))
            worker.start()

        except Exception as e:
            self._hide_building_overlay()
            self.action_run.setEnabled(True)
            QMessageBox.critical(self, "Simulation Error", str(e))

    def _on_export_for_run_finished(self, success, message, export_path, total_batches, request_id, worker):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        if request_id != self._run_request_id:
            self._hide_building_overlay()
            self.action_run.setEnabled(True)
            return
        self.statusbar.showMessage("Ready", 3000)
        if not success:
            self._hide_building_overlay()
            self.action_run.setEnabled(True)
            QMessageBox.critical(self, "Simulation Error", message)
            return

        self._hide_building_overlay()
        self.dock_console.setVisible(True)

        self.progress_dialog = SimulationProgressDialog(self, theme=self.current_theme)
        script_text = self._get_script_text()
        is_dep = "openmc.deplete" in script_text
        sim_worker = SimulationWorker(export_path, total_batches, is_depletion=is_dep)
        self._active_workers.append(sim_worker)
        sim_worker.log_signal.connect(self.console_widget.append_log)
        sim_worker.progress_signal.connect(self.progress_dialog.update_progress)
        sim_worker.finished_signal.connect(
            lambda s, m, w=sim_worker: self._on_simulation_finished(s, m, w))

        sim_worker.start()
        self.progress_dialog.exec()

    def _on_simulation_finished(self, success, message, worker=None):
        if worker is not None and worker in self._active_workers:
            self._active_workers.remove(worker)
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.accept()

        self.action_run.setEnabled(True)

        if success:
            self.console_widget.append_log(f"✅ {message}")

            if hasattr(self, 'custom_sp_name') and self.custom_sp_name:
                try:
                    export_path = os.path.join(os.getcwd(), "export")
                    sp_files = glob.glob(os.path.join(export_path, "statepoint.*.h5"))
                    if sp_files:
                        latest_sp = max(sp_files, key=os.path.getctime)
                        new_path = os.path.join(export_path, self.custom_sp_name)

                        if os.path.exists(new_path):
                            os.remove(new_path)

                        os.rename(latest_sp, new_path)
                        self.console_widget.append_log(f"📝 Results renamed to: {self.custom_sp_name}")
                except Exception as e:
                    self.console_widget.append_log(f"⚠️ Could not rename file: {str(e)}")

            try:
                if not hasattr(self, 'results_page_loaded'):
                    self._lazy_load_tabs(4)

                self.results_page.auto_load_latest_statepoint()
                self.visual_tabs.setCurrentWidget(self.results_page)
                self.statusbar.showMessage("Ready - Simulation Results Activated", 3000)
            except Exception as e:
                self.console_widget.append_log(f"⚠️ Warning: Could not auto-load results. {str(e)}")
        else:
            self.console_widget.append_log(f"❌ Simulation Failed: {message}")
            QMessageBox.critical(self, "Simulation Error", message)