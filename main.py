import sys
import six

# خدعة برمجية لحل مشكلة PyInstaller مع مكتبة six
for p in sys.meta_path:
    if p.__class__.__name__ == "_SixMetaPathImporter":
        setattr(p, "_path", [])
import sys
import time
import matplotlib.pyplot as plt  # لحل مشكلة التوافق مع PySide6 كما فعلنا سابقاً

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class SplashScreen(QWidget):
    """
    شاشة ترحيب وتحميل احترافية (بدون حواف) تظهر قبل فتح الواجهة الرئيسية.
    """
    def __init__(self):
        super().__init__()
        # إزالة حواف النافذة (Frameless) وجعلها دائماً في المقدمة
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(550, 320)

        # الإطار الرئيسي (الخلفية البيضاء مع حدود خضراء)
        frame = QFrame(self)
        frame.setFixedSize(550, 320)
        frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 3px solid #008000;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- النصوص ---
        title = QLabel("OpenMC Studio")
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #008000; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        version = QLabel("Version 1.0 - Academic Edition")
        version.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF8C00; border: none;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)

        author = QLabel("Developed by: Dr. Ayman Abu Ghazal")
        author.setStyleSheet("font-size: 18px; font-weight: bold; color: #333333; border: none; margin-top: 15px;")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)

        inst = QLabel("Jordan Atomic Energy Commission")
        inst.setStyleSheet("font-size: 15px; font-weight: bold; color: #555555; border: none;")
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

        # إضافة العناصر للواجهة وتنسيق المسافات
        layout.addSpacing(15)
        layout.addWidget(title)
        layout.addWidget(version)
        layout.addSpacing(10)
        layout.addWidget(author)
        layout.addWidget(inst)
        layout.addSpacing(25)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addSpacing(10)

    def update_progress(self, value, text):
        """تحديث شريط التحميل والنص السفلي"""
        self.progress.setValue(value)
        self.status.setText(text)

def main():
    app = QApplication(sys.argv)

    # 1. إظهار شاشة الترحيب أولاً
    splash = SplashScreen()
    splash.show()
    app.processEvents() # إجبار الشاشة على التحديث الفوري

    # 2. محاكاة تحميل المكونات الأساسية وتحديث الشريط
    splash.update_progress(15, "Loading Python Modules & Graphics Engines...")
    time.sleep(0.4)
    app.processEvents()

    splash.update_progress(40, "Connecting to OpenMC Core...")
    time.sleep(0.5)
    app.processEvents()

    splash.update_progress(65, "Building VISED Layout Interfaces...")
    time.sleep(0.4)
    app.processEvents()

    # استدعاء الواجهة الرئيسية هنا لكي لا يتجمد البرنامج في البداية
    from ui.main_window import MainWindow

    splash.update_progress(85, "Initializing 3D Voxel Engine (PyVista)...")
    window = MainWindow()
    time.sleep(0.6)
    app.processEvents()

    splash.update_progress(100, "Ready!")
    time.sleep(0.3)
    app.processEvents()

    # 3. إغلاق شاشة الترحيب وفتح البرنامج الفعلي
    splash.close()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()