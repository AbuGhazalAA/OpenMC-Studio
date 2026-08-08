import logging
import os
import glob
import sys
import subprocess
import time
import re
import gc
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
    QTabWidget, QLabel, QStatusBar, QToolBar, QStyle, QApplication, QMessageBox, QFileDialog, QSplitter, QDialog,
    QPushButton, QGroupBox, QProgressBar, QInputDialog, QLineEdit, QComboBox
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QIcon

from ui.property_editor import PropertyEditorWidget
from ui.geometry.geometry_page import GeometryPageWidget
from ui.materials.materials_page import MaterialsPageWidget
from ui.settings.settings_page import SettingsPageWidget
from ui.project_tree import ProjectTreeWidget
from ui.console_widget import ConsoleWidget
from ui.tallies.tallies_page import TalliesPageWidget
from core.project_manager import ProjectManager
from ui.script_editor import ScriptEditorWidget
from ui.results_page import ResultsPageWidget
from ui.plots.plots_page import PlotsPageWidget
from ui.plots.voxel_page import VoxelPageWidget
from ui.tracks.tracks_page import TracksPageWidget
from ui.libraries_page import LibrariesPageWidget

import openmc

logger = logging.getLogger("OpenMC Studio")


# =================================================================
# --- Worker Threads ---
# =================================================================
class ExportWorker(QThread):
    """Writes current_script.py and executes it (in-process exec() when
    frozen, isolated subprocess otherwise) entirely off the GUI thread.
    Building/exporting a model can take a real, sometimes unpredictable
    amount of time (complex geometry, antivirus scanning newly-written
    files, etc.) -- running it directly on the GUI thread, as the
    previous version did, freezes the whole UI and can trigger Windows'
    "Not Responding" state for the entire app, even though the work
    itself is proceeding normally.
    """
    finished_signal = Signal(bool, str, str, int)  # success, error_message, export_path, total_batches

    def __init__(self, script_text, project_dir):
        super().__init__()
        self.script_text = script_text
        self.project_dir = project_dir

    def run(self):
        try:
            compatibility_shim = (
                "# --- OpenMC Studio: auto-injected compatibility shim ---\n"
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

            script_path = os.path.join(self.project_dir, "current_script.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(compatibility_shim)
                f.write(self.script_text)

            export_path = os.path.join(self.project_dir, "export")
            if not os.path.exists(export_path):
                os.makedirs(export_path)

            if getattr(sys, 'frozen', False):
                # داخل الملف المُجمَّد، sys.executable يشير إلى الملف
                # التنفيذي نفسه، فاستدعاء subprocess به يُعيد تشغيل
                # التطبيق بأكمله. لذا هنا فقط، نُنفّذ السكربت داخل نفس
                # العملية (لكن على هذا الخيط الخلفي، لا خيط الواجهة).
                original_dir = os.getcwd()
                try:
                    os.chdir(export_path)
                    exec(compatibility_shim + self.script_text, {'__name__': '__main__'})
                finally:
                    os.chdir(original_dir)
            else:
                result = subprocess.run(
                    [sys.executable, script_path], cwd=export_path,
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    details = (result.stderr or result.stdout or "(no output captured)").strip()
                    self.finished_signal.emit(
                        False, f"Script execution failed (exit code {result.returncode}):\n{details}", "", 0)
                    return

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

    def __init__(self, export_dir, total_batches):
        super().__init__()
        self.export_dir = export_dir
        self.total_batches = total_batches

    def run(self):
        try:
            # creationflags=CREATE_NO_WINDOW (Windows only -- never
            # evaluated elsewhere) stops a new console window from
            # popping up for the openmc.exe child process when this app
            # is a frozen --windowed exe with no console of its own for
            # that console-mode child to inherit.
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            process = subprocess.Popen(
                ["openmc"],
                cwd=self.export_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags
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
                self.finished_signal.emit(True, "Simulation completed successfully! StatePoint generated.")
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
            # Direct subprocess call instead of openmc.plot_geometry() --
            # that wrapper's internal subprocess spawn shows a new,
            # visible console window when this app is a frozen
            # --windowed exe (no console for the openmc.exe child, which
            # is itself a console app, to inherit). CREATE_NO_WINDOW
            # suppresses that popup; harmless no-op on non-Windows.
            # Using cwd= directly also avoids the os.chdir() pattern the
            # previous version used, which is unsafe if another thread
            # (e.g. SimulationWorker) is relying on the working directory
            # at the same time.
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(
                ["openmc", "--plot"], cwd=self.export_dir,
                capture_output=True, text=True, creationflags=creationflags
            )
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "(no output captured)").strip()
                self.finished_signal.emit(
                    False, f"Plot generation failed (exit code {result.returncode}):\n{details}", "")
                return

            png_path = os.path.join(self.export_dir, "geometry_plot.png")
            if not os.path.exists(png_path):
                self.finished_signal.emit(False, "geometry_plot.png was not generated.", "")
                return

            self.finished_signal.emit(True, "Plot generated successfully!", png_path)
        except Exception as e:
            self.finished_signal.emit(False, str(e), "")


# =================================================================
# --- Simulation Progress Dialog ---
# =================================================================
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
        # مرجع قوي دائم لكل خيط خلفي (Worker) قيد التشغيل حالياً --
        # يمنع بايثون من حذف كائن الخيط من الذاكرة أثناء عمله الفعلي
        # (وهذا يسبب انهياراً كاملاً في PySide6/Qt، وليس مجرد استثناء
        # بايثون عادي). كل عامل يُضاف هنا عند إنشائه، ويُحذَف منه فقط
        # داخل معالج انتهائه الخاص، بعد التأكد أنه انتهى فعلياً.
        self._active_workers = []

        # ينشأ هنا مباشرة (وليس داخل _setup_tool_windows) لأن تلك الدالة
        # ستصبح قابلة للاستدعاء مجدداً من new_project() لإعادة بناء
        # نوافذ الأدوات المنبثقة بالكامل -- لكن محرر السكربت نفسه يجب ألا
        # يُعاد إنشاؤه هناك (نصه يُمسَح مباشرة في مكان آخر بدلاً من ذلك).
        self.script_editor = ScriptEditorWidget()

        self._setup_tool_windows()
        self._setup_central_widget()
        self._setup_dock_widgets()
        self._setup_actions()
        self._setup_menus()
        self._setup_toolbars()
        self._setup_statusbar()
        self._apply_theme()

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
            QTabBar::tab {{ background: {primary}; color: white; padding: 10px 20px; border-top-left-radius: 5px; border-top-right-radius: 5px; margin-right: 2px; font-weight: bold; }}
            QTabBar::tab:selected {{ background: {primary_hover}; }}
            QTabWidget::pane {{ border: 2px solid {primary}; }}
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

    def _setup_tool_windows(self):
        # قابلة للاستدعاء أكثر من مرة (تُستدعى مجدداً من new_project()
        # لإعادة بناء نوافذ الأدوات المنبثقة بالكامل). إن وجدت نسخ
        # سابقة، أغلقها إن كانت ظاهرة على الشاشة ثم جدولها للحذف قبل
        # إنشاء نسخ جديدة تماماً.
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
        self.geometry_page.script_generated.connect(self.script_editor.append_code)
        self.settings_page.script_generated.connect(self.script_editor.append_code)
        self.tallies_page.script_generated.connect(self.script_editor.append_code)

        self.win_materials = self._create_popup_window("Materials Editor", self.materials_page)
        self.win_settings = self._create_popup_window("Simulation Settings", self.settings_page)
        self.win_tallies = self._create_popup_window("Tallies & Detectors", self.tallies_page)
        self.win_libraries = self._create_popup_window("Nuclear Data Libraries Manager", self.libraries_page)

        self.win_geometry = QDialog(self)
        self.win_geometry.setWindowTitle("Geometry Builder")
        self.win_geometry.setMinimumSize(800, 600)
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

    def _create_popup_window(self, title, widget):
        win = QDialog(self)
        win.setWindowTitle(title)
        win.setMinimumSize(800, 600)
        layout = QVBoxLayout(win)
        layout.addWidget(widget)
        return win

    def _setup_central_widget(self):
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.visual_tabs = QTabWidget()
        self.plots_page = PlotsPageWidget()
        self.voxel_page = VoxelPageWidget()
        self.results_page = ResultsPageWidget()
        self.tracks_page = TracksPageWidget()

        self.plots_page.script_generated.connect(self.script_editor.append_code)
        self.voxel_page.script_generated.connect(self.script_editor.append_code)
        self.tracks_page.script_generated.connect(self.script_editor.append_code)

        self.visual_tabs.addTab(self.plots_page, "🎨 2D Geometry Viewer")
        self.visual_tabs.addTab(self.voxel_page, "🧊 3D Voxel Viewer")
        self.visual_tabs.addTab(self.tracks_page, "🎯 Particle Tracks")
        self.visual_tabs.addTab(self.results_page, "📊 Simulation Results")

        self.main_splitter.addWidget(self.visual_tabs)

        script_container = QWidget()
        script_layout = QVBoxLayout(script_container)
        script_layout.setContentsMargins(0, 0, 0, 0)

        script_header_layout = QHBoxLayout()
        self.script_header = QLabel("🔥 Python Script Engine")

        script_header_layout.addWidget(self.script_header)
        script_header_layout.addStretch()

        script_header_container = QWidget()
        script_header_container.setLayout(script_header_layout)

        script_layout.addWidget(script_header_container)
        script_layout.addWidget(self.script_editor)

        self.main_splitter.addWidget(script_container)
        self.main_splitter.setSizes([850, 550])
        self.setCentralWidget(self.main_splitter)

    def _setup_dock_widgets(self):
        self.dock_project = QDockWidget("Project Explorer", self)
        self.project_tree = ProjectTreeWidget()
        self.dock_project.setWidget(self.project_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_project)
        self.dock_project.hide()

        self.dock_console = QDockWidget("Live Output Console", self)
        self.console_widget = ConsoleWidget()
        self.dock_console.setWidget(self.console_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_console)

        self.geometry_page.log_signal.connect(self.console_widget.append_log)

    def _setup_actions(self):
        style = QApplication.style()
        self.action_new = QAction(style.standardIcon(QStyle.StandardPixmap.SP_FileIcon), "New Project", self)
        self.action_new.triggered.connect(self.new_project)
        self.action_open = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open Project...",
                                   self)
        self.action_open.triggered.connect(self.open_project)
        self.action_save = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
        self.action_save.triggered.connect(self.save_project)
        self.action_run = QAction("▶️ Run OpenMC", self)
        self.action_run.setShortcut("F5")
        self.action_run.triggered.connect(self._run_openmc_simulation)
        self.action_validate = QAction("✔️ Validate Geometry & Plot", self)
        self.action_validate.triggered.connect(self._generate_and_show_plot)
        self.action_about = QAction("ℹ️ About OpenMC Studio", self)
        self.action_about.triggered.connect(self.show_about_dialog)
        self.action_theme = QAction("🌗 Toggle Grayscale Theme", self)
        self.action_theme.triggered.connect(self.toggle_theme)

    def _setup_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_new)
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_save)

        build_menu = menubar.addMenu("&Build (Setup)")
        act_mat = QAction("🧪 Materials Editor", self)
        # Lambda, not a direct .show reference: _setup_tool_windows() can
        # run again later (from new_project()), replacing win_materials
        # with a fresh object. A direct reference here would freeze on
        # the object that existed when this menu was first built, and
        # break once that object is replaced. The lambda looks up
        # self.win_materials fresh at click time instead.
        act_mat.triggered.connect(lambda: self.win_materials.show())
        build_menu.addAction(act_mat)
        act_geo = QAction("📐 Geometry Builder", self)
        act_geo.triggered.connect(lambda: self.win_geometry.show())
        build_menu.addAction(act_geo)
        act_set = QAction("⚙️ Simulation Settings", self)
        act_set.triggered.connect(lambda: self.win_settings.show())
        build_menu.addAction(act_set)
        act_tal = QAction("🎯 Tallies & Detectors", self)
        act_tal.triggered.connect(lambda: self.win_tallies.show())
        build_menu.addAction(act_tal)
        build_menu.addSeparator()
        act_lib = QAction("📚 Data Libraries Manager", self)
        act_lib.triggered.connect(lambda: self.win_libraries.show())
        build_menu.addAction(act_lib)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.dock_project.toggleViewAction())
        view_menu.addAction(self.dock_console.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self.action_theme)

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
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.action_run)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready - VISED Layout Mode", 5000)

    # =================================================================
    # --- دالة إضافة المايكروبوديز ---
    # =================================================================
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

    # =================================================================
    # --- دوال إدارة المشاريع ---
    # =================================================================
    def new_project(self):
        # Script editor
        if hasattr(self.script_editor, 'editor'):
            self.script_editor.editor.clear()
        else:
            self.script_editor.clear()

        # Live console -- a freshly-launched app has an empty one too
        if hasattr(self, 'console_widget'):
            try:
                self.console_widget.clear_log()
            except Exception:
                pass  # if ConsoleWidget doesn't expose .clear(), skip silently rather than crash

        # Rebuild the four visual tabs from scratch rather than trying
        # to clear each one's internal state field-by-field -- a fresh
        # instance IS its own default/empty state by construction, so
        # this is the most reliable way to guarantee "looks exactly
        # like the app was just reopened" for the 2D/3D/Tracks/Results
        # views, matching what was explicitly requested.
        old_index = self.visual_tabs.currentIndex()

        new_plots_page = PlotsPageWidget()
        new_voxel_page = VoxelPageWidget()
        new_tracks_page = TracksPageWidget()
        new_results_page = ResultsPageWidget()

        new_plots_page.script_generated.connect(self.script_editor.append_code)
        new_voxel_page.script_generated.connect(self.script_editor.append_code)
        new_tracks_page.script_generated.connect(self.script_editor.append_code)

        for old_widget in (self.plots_page, self.voxel_page, self.tracks_page, self.results_page):
            idx = self.visual_tabs.indexOf(old_widget)
            if idx != -1:
                self.visual_tabs.removeTab(idx)
            # VoxelPageWidget wraps a pyvistaqt.QtInteractor -- a VTK/
            # OpenGL render context, which (unlike a plain Qt widget) is
            # known to be sensitive to deletion without first closing
            # its render pipeline explicitly. Skipping this can crash or
            # leak GPU resources, the same class of risk as deleting a
            # still-running QThread we hit earlier in this project.
            if hasattr(old_widget, 'plotter'):
                try:
                    old_widget.plotter.close()
                except Exception:
                    pass
            old_widget.deleteLater()

        self.plots_page = new_plots_page
        self.voxel_page = new_voxel_page
        self.tracks_page = new_tracks_page
        self.results_page = new_results_page

        self.visual_tabs.addTab(self.plots_page, "🎨 2D Geometry Viewer")
        self.visual_tabs.addTab(self.voxel_page, "🧊 3D Voxel Viewer")
        self.visual_tabs.addTab(self.tracks_page, "🎯 Particle Tracks")
        self.visual_tabs.addTab(self.results_page, "📊 Simulation Results")

        self.visual_tabs.setCurrentIndex(min(old_index, self.visual_tabs.count() - 1))

        # Rebuild the five popup tool windows too (Materials, Geometry,
        # Settings, Tallies, Libraries) -- these accumulate their own
        # internal state (e.g. GeometryPageWidget's surface list,
        # TalliesPageWidget's created-tallies list) that persisted
        # across "New Project" until now, even though the script itself
        # had already been cleared. _setup_tool_windows() closes any of
        # these that are currently open before replacing them.
        self._setup_tool_windows()
        # This one connection lives in _setup_dock_widgets() instead of
        # _setup_tool_windows() because, at __init__ time, console_widget
        # doesn't exist yet when _setup_tool_windows() first runs. Safe
        # to reconnect directly here since console_widget already exists
        # by the time new_project() can be called.
        self.geometry_page.log_signal.connect(self.console_widget.append_log)

        self.statusbar.showMessage("New project started -- all views reset.", 5000)

    def save_project(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Project", "",
                                                  "OpenMC Studio Project (*.omcs);;Python Files (*.py)")
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                if hasattr(self.script_editor, 'editor'):
                    f.write(self.script_editor.editor.toPlainText())
                else:
                    f.write(self.script_editor.toPlainText())
            self.statusbar.showMessage(f"Project saved: {os.path.basename(filepath)}", 5000)

    def open_project(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Project", "",
                                                  "OpenMC Studio Project (*.omcs);;Python Files (*.py);;All Files (*)")
        if filepath:
            try:
                # استخدام 'r' للقراءة بشكل آمن وتجنب مسح الملفات
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                if hasattr(self.script_editor, 'editor') and hasattr(self.script_editor.editor, 'setPlainText'):
                    self.script_editor.editor.setPlainText(file_content)
                elif hasattr(self.script_editor, 'setPlainText'):
                    self.script_editor.setPlainText(file_content)

                self.statusbar.showMessage(f"Project loaded: {os.path.basename(filepath)}", 5000)
                if hasattr(self, 'console_widget') and self.console_widget:
                    self.console_widget.append_log(f"📂 Successfully loaded project: {os.path.basename(filepath)}")

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
            "© 2026 Dr. Ayman Abu Ghazal. All rights reserved.<br><br>"
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

    # =================================================================
    # --- التصدير والتشغيل الخارجي الآمن للسكربت (على خيط خلفي) ---
    # =================================================================
    def _get_script_text(self):
        if hasattr(self.script_editor, 'editor'):
            return self.script_editor.editor.toPlainText()
        return self.script_editor.toPlainText()

    def _generate_and_show_plot(self):
        # كل طلب رسم يأخذ رقماً تسلسلياً جديداً؛ فقط نتيجة أحدث طلب
        # تُعرَض فعلياً -- هذا يمنع نتيجة قديمة متأخرة الوصول (مثلاً من
        # نموذج سابق كان لا يزال قيد البناء عند تحميل مثال جديد
        # والضغط على Validate Geometry مجدداً بسرعة) من الكتابة فوق
        # نتيجة أحدث، بغض النظر عن ترتيب اكتمال الخيوط الخلفية الفعلي.
        self._plot_request_id = getattr(self, '_plot_request_id', 0) + 1
        request_id = self._plot_request_id
        self.statusbar.showMessage("🔧 Building model, please wait...", 0)
        script_text = self._get_script_text()
        worker = ExportWorker(script_text, os.getcwd())
        self._active_workers.append(worker)  # يبقيه حياً طوال عمله الفعلي
        worker.finished_signal.connect(
            lambda success, message, export_path, total_batches, rid=request_id, w=worker:
                self._on_export_for_plot_finished(success, message, export_path, total_batches, rid, w))
        worker.start()

    def _on_export_for_plot_finished(self, success, message, export_path, total_batches, request_id, worker):
        if worker in self._active_workers:
            self._active_workers.remove(worker)  # انتهى فعلياً -- آمن الآن للتحرير
        if request_id != self._plot_request_id:
            return  # طلب أقدم تجاوزه طلب أحدث -- تجاهله
        if not success:
            self.statusbar.showMessage("Ready", 3000)
            QMessageBox.critical(self, "Export Error", message)
            return
        plot_worker = PlotWorker(export_path)
        self._active_workers.append(plot_worker)
        plot_worker.finished_signal.connect(
            lambda s, m, p, rid=request_id, w=plot_worker: self._on_plot_finished(s, m, p, rid, w))
        plot_worker.start()

    def _on_plot_finished(self, success, message, ppm_path, request_id, worker):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        if request_id != self._plot_request_id:
            return  # طلب أقدم -- تجاهله
        self.statusbar.showMessage("Ready", 3000)
        if success:
            self.plots_page.display_image(ppm_path)
            self.visual_tabs.setCurrentWidget(self.plots_page)
            self.console_widget.append_log(f"✅ {message}")
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

            if hasattr(self, 'results_page'):
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
            self.action_run.setEnabled(True)
            QMessageBox.critical(self, "Simulation Error", str(e))

    def _on_export_for_run_finished(self, success, message, export_path, total_batches, request_id, worker):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        if request_id != self._run_request_id:
            self.action_run.setEnabled(True)
            return  # طلب أقدم -- تجاهله
        self.statusbar.showMessage("Ready", 3000)
        if not success:
            self.action_run.setEnabled(True)
            QMessageBox.critical(self, "Simulation Error", message)
            return

        self.dock_console.setVisible(True)

        self.progress_dialog = SimulationProgressDialog(self, theme=self.current_theme)
        sim_worker = SimulationWorker(export_path, total_batches)
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
                self.results_page.auto_load_latest_statepoint()
                self.visual_tabs.setCurrentWidget(self.results_page)
            except Exception as e:
                self.console_widget.append_log(f"⚠️ Warning: Could not auto-load results. {str(e)}")
        else:
            self.console_widget.append_log(f"❌ Simulation Failed: {message}")
            QMessageBox.critical(self, "Simulation Error", message)
