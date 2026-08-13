import sys
import time
import six

# خدعة برمجية لحل مشكلة PyInstaller مع مكتبة six
for p in sys.meta_path:
    if p.__class__.__name__ == "_SixMetaPathImporter":
        setattr(p, "_path", [])

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QProgressBar, QFrame, QPushButton, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QSurfaceFormat


class LoaderThread(QThread):
    """
    خيط خلفي (Background Thread) لتحميل المكتبات الثقيلة.
    فصل هذه العملية يضمن عدم تجميد الواجهة (Main Thread)،
    مما يسمح لأزرار الإغلاق والتصغير بالعمل الفوري في أي لحظة.
    """
    progress_update = Signal(int, str)

    def run(self):
        # 1. تحميل الأساسيات
        self.progress_update.emit(15, "Loading Base & Math Libraries...")
        import numpy
        import pandas
        time.sleep(0.2)  # مهلة بسيطة لجمالية العرض

        # 2. تحميل SciPy (وهي ما كانت تسبب التأخير الأكبر)
        self.progress_update.emit(35, "Initializing Data Analysis Engine (SciPy)...")
        import scipy
        import scipy.signal
        import scipy.optimize

        # 3. تحميل OpenMC
        self.progress_update.emit(60, "Loading OpenMC Physics Engine... (Please wait)")
        import openmc

        # 4. بناء الواجهات
        self.progress_update.emit(85, "Building User Interface Components...")
        # بمجرد أن تنتهي هذه الخيوط من التحميل، ستكون المكتبات موجودة في (sys.modules)
        # وعندما تستدعيها النافذة الرئيسية ستفتح في جزء من الثانية!
        time.sleep(0.3)


class SplashScreen(QWidget):
    """
    شاشة ترحيب وتحميل احترافية (بدون حواف) مع أزرار تحكم وإغلاق.
    """

    def __init__(self):
        super().__init__()
        # إزالة حواف النافذة وجعلها دائماً في المقدمة
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(550, 340)

        # الإطار الرئيسي
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

        # --- أزرار التحكم (تصغير وإغلاق) ---
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

        # --- النصوص ---
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

        # --- شريط التحميل (Progress Bar) ---
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

        # إضافة العناصر
        main_layout.addWidget(title)
        main_layout.addWidget(version)
        main_layout.addSpacing(10)
        main_layout.addWidget(author)
        main_layout.addWidget(inst)
        main_layout.addSpacing(20)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.status)
        main_layout.addSpacing(10)

        # --- ربط الخيط الخلفي ---
        self.loader_thread = LoaderThread()
        self.loader_thread.progress_update.connect(self.update_progress)
        self.loader_thread.finished.connect(self.launch_main_window)

    def close_app(self):
        """إغلاق البرنامج بشكل آمن إذا قرر المستخدم الإلغاء"""
        if self.loader_thread.isRunning():
            self.loader_thread.terminate()
        sys.exit(0)

    def start_loading(self):
        """بدء عملية التحميل الواقعية"""
        self.loader_thread.start()

    def update_progress(self, value, text):
        """تحديث شريط التحميل والنص السفلي"""
        self.progress.setValue(value)
        self.status.setText(text)

    def launch_main_window(self):
        """استدعاء الواجهة الرئيسية بعد اكتمال تحميل المكتبات"""
        self.update_progress(100, "Ready!")
        from ui.main_window import MainWindow
        self.main_win = MainWindow()
        self.main_win.show()
        self.close()  # إغلاق شاشة الترحيب


def main():
    # =================================================================
    # --- إعداد بيئة OpenGL 3.2 قبل بناء التطبيق لمنع انهيار PyVista ---
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
    splash.start_loading()  # تشغيل خيط التحميل

    sys.exit(app.exec())


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()