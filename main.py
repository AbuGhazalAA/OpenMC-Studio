import sys
import time
import six

# Workaround for a PyInstaller / six incompatibility: six installs a meta
# path importer whose _path attribute a frozen build cannot resolve.
for p in sys.meta_path:
    if p.__class__.__name__ == "_SixMetaPathImporter":
        setattr(p, "_path", [])

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QProgressBar, QFrame, QPushButton, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QSurfaceFormat


class LoaderThread(QThread):
    """
    Background thread that imports the heavy libraries.

    Doing this off the main thread keeps the interface responsive, so the
    minimise and close buttons on the splash screen work at any moment
    instead of freezing until loading finishes.

    Any failure is reported through load_failed rather than raised: an
    exception here would abort run(), yet Qt still emits finished, so the
    launch would go ahead as if nothing had happened and fail later with
    no explanation of the real cause.
    """
    progress_update = Signal(int, str)
    load_failed = Signal(str, str)      # module name, traceback

    def run(self):
        stage = "startup"
        try:
            # 1. Core numerics
            stage = "numpy / pandas"
            self.progress_update.emit(15, "Loading Base & Math Libraries...")
            import numpy
            import pandas
            time.sleep(0.2)  # brief pause so the progress bar reads smoothly

            # 2. SciPy -- the single slowest import of the set
            stage = "scipy"
            self.progress_update.emit(35, "Initializing Data Analysis Engine (SciPy)...")
            import scipy
            import scipy.signal
            import scipy.optimize

            # 3. OpenMC
            stage = "openmc"
            self.progress_update.emit(60, "Loading OpenMC Physics Engine... (Please wait)")
            import openmc

            # 4. Interface components
            self.progress_update.emit(85, "Building User Interface Components...")
            # Once these imports complete the modules are in sys.modules,
            # so the main window finds them already loaded and appears
            # almost instantly rather than importing them itself.
            time.sleep(0.3)
        except Exception:
            import traceback
            self.load_failed.emit(stage, traceback.format_exc())


class SplashScreen(QWidget):
    """
    Frameless splash screen with its own minimise and close buttons.
    """

    def __init__(self):
        super().__init__()
        # Frameless and always on top
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(550, 340)

        # Main frame
        frame = QFrame(self)
        frame.setFixedSize(550, 340)
        frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 3px solid #008000;
                border-radius: 12px;
            }
        """)
        main_layout = QVBoxLayout(frame)
        main_layout.setContentsMargins(15, 10, 15, 20)

        # --- Window controls (minimise / close) ---
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self.btn_min = QPushButton("-")
        self.btn_min.setFixedSize(30, 30)
        self.btn_min.setToolTip("Minimize")
        self.btn_min.setStyleSheet("""
            QPushButton { background-color: #e0e0e0; color: black; font-weight: bold; border-radius: 15px; font-size: 16px; border: none; }
            QPushButton:hover { background-color: #d0d0d0; }
        """)
        self.btn_min.clicked.connect(self.showMinimized)

        self.btn_close = QPushButton("×")
        self.btn_close.setFixedSize(30, 30)
        self.btn_close.setToolTip("Cancel & Exit")
        self.btn_close.setStyleSheet("""
            QPushButton { background-color: #ff4c4c; color: white; font-weight: bold; border-radius: 15px; font-size: 18px; border: none; }
            QPushButton:hover { background-color: #ff1a1a; }
        """)
        self.btn_close.clicked.connect(self.close_app)

        top_layout.addWidget(self.btn_min)
        top_layout.addWidget(self.btn_close)
        main_layout.addLayout(top_layout)

        # --- Text ---
        title = QLabel("OpenMC Studio")
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #008000; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        version = QLabel("Version 1.0 - Professional Edition")
        version.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF8C00; border: none;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)

        author = QLabel("Copyright © 2026 Dr. Ayman Abu Ghazal")
        author.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333; border: none; margin-top: 10px;")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)

        inst = QLabel("Jordan Atomic Energy Commission")
        inst.setStyleSheet("font-size: 11px; font-weight: bold; color: #555555; border: none;")
        inst.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Progress bar ---
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #008000;
                border-radius: 5px;
                text-align: center;
                color: #000000;
                font-weight: bold;
                font-size: 13px;
                background-color: #F8F9FA;
            }
            QProgressBar::chunk {
                background-color: #FF8C00;
                border-radius: 3px;
            }
        """)
        self.progress.setFixedHeight(28)
        self.progress.setValue(0)

        self.status = QLabel("Initializing...")
        self.status.setStyleSheet("font-size: 13px; font-weight: bold; color: #008000; border: none; margin-top: 5px;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Assemble
        main_layout.addWidget(title)
        main_layout.addWidget(version)
        main_layout.addSpacing(10)
        main_layout.addWidget(author)
        main_layout.addWidget(inst)
        main_layout.addSpacing(20)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.status)
        main_layout.addSpacing(10)

        # --- Wire up the background loader ---
        self._load_error = None
        self.loader_thread = LoaderThread()
        self.loader_thread.progress_update.connect(self.update_progress)
        self.loader_thread.load_failed.connect(self._remember_load_error)
        self.loader_thread.finished.connect(self.launch_main_window)

    def _remember_load_error(self, stage, details):
        self._load_error = (stage, details)

    def close_app(self):
        """Exit cleanly if the user cancels during loading."""
        if self.loader_thread.isRunning():
            self.loader_thread.terminate()
        sys.exit(0)

    def start_loading(self):
        """Start the real loading work."""
        self.loader_thread.start()

    def update_progress(self, value, text):
        """Update the progress bar and the status line beneath it."""
        self.progress.setValue(value)
        self.status.setText(text)

    def launch_main_window(self):
        """Build and show the main window once the libraries are loaded.

        Everything here runs inside a Qt slot. An exception raised in a slot
        does not propagate: Qt prints it to stderr and carries on. In a
        windowed build there is no stderr, so a failure to build the main
        window left the splash sitting at "Ready!" forever with no window
        and no message -- the one failure mode that tells the user nothing
        at all. Any error is now shown, with the traceback, instead.
        """
        self.update_progress(100, "Ready!")

        if self._load_error is not None:
            stage, details = self._load_error
            self._report_startup_failure(
                f"A required library failed to load: {stage}", details)
            return

        try:
            from ui.main_window import MainWindow
            self.main_win = MainWindow()
            self.main_win.show()
        except Exception:
            import traceback
            self._report_startup_failure(
                "The main window could not be created.", traceback.format_exc())
            return

        self.close()  # dismiss the splash screen

    def _report_startup_failure(self, summary, details):
        """Show why startup failed, with a traceback that can be copied."""
        from PySide6.QtWidgets import QMessageBox

        self.update_progress(100, "Startup failed")
        self.status.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #c0392b; border: none;")

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("OpenMC Studio — Startup Failed")
        box.setText(summary)
        box.setInformativeText(
            "The application cannot continue. The technical details below "
            "name the missing piece — copy them when reporting this.")
        box.setDetailedText(details)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # Also write it beside the executable: a dialog can be dismissed
        # before it is read, and a packaged build has nowhere else to log.
        try:
            import os
            base = (os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
                    else os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(base, "startup_error.log"), "w",
                      encoding="utf-8") as fh:
                fh.write(f"{summary}\n\n{details}")
        except Exception:
            pass

        box.exec()
        sys.exit(1)


def main():
    # =================================================================
    # --- Request OpenGL 3.2 before the application exists, so the 3D
    # --- voxel viewer (PyVista/VTK) has the context it needs and does
    # --- not crash the process when that tab is first opened. ---
    # =================================================================
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)
    # =================================================================

    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()
    splash.start_loading()  # kick off the background loader

    sys.exit(app.exec())


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()