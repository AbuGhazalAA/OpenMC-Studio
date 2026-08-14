import os
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QMessageBox, QHeaderView
)
from PySide6.QtCore import Signal


class LibrariesPageWidget(QWidget):
    """
    واجهة إدارة واستكشاف مكتبات البيانات النووية (Nuclear Data Libraries Manager)
    تسمح بتحديد مسار ملف cross_sections.xml وتصفح النظائر المتاحة.
    """
    library_changed = Signal(str)
    # إضافة الإشارة الجديدة لإرسال الكود إلى الشاشة البرتقالية
    script_generated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 1. صندوق إعدادات المسار المتغير ---
        path_group = QGroupBox("Cross Sections Path Configuration")
        path_layout = QHBoxLayout(path_group)

        self.path_field = QLineEdit()
        self.path_field.setPlaceholderText("Select path to cross_sections.xml...")

        # جلب المسار الحالي إن وجد في متغيرات النظام
        env_path = os.environ.get("OPENMC_CROSS_SECTIONS", "")
        if env_path:
            self.path_field.setText(env_path)

        btn_browse = QPushButton("📁 Browse XML")
        btn_browse.setStyleSheet("background-color: #008000; color: white; font-weight: bold; padding: 6px;")
        btn_browse.clicked.connect(self.browse_cross_sections)

        btn_set = QPushButton("⚡ Apply to Environment")
        btn_set.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold; padding: 6px;")
        btn_set.clicked.connect(self.apply_library_path)

        path_layout.addWidget(QLabel("XML Path:"))
        path_layout.addWidget(self.path_field)
        path_layout.addWidget(btn_browse)
        path_layout.addWidget(btn_set)
        layout.addWidget(path_group)

        # --- 2. صندوق استكشاف النظائر (Isotope Explorer) ---
        explorer_group = QGroupBox("Isotope & Cross-Sections Explorer")
        exp_layout = QVBoxLayout(explorer_group)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Nuclide / Material", "Alias", "AWR", "Data File Path"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        exp_layout.addWidget(self.table)

        btn_load_isotopes = QPushButton("🔍 Scan Library & List Isotopes")
        btn_load_isotopes.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 6px;")
        btn_load_isotopes.clicked.connect(self.scan_isotopes)
        exp_layout.addWidget(btn_load_isotopes)

        layout.addWidget(explorer_group)

    def browse_cross_sections(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select cross_sections.xml", "",
                                                  "XML Files (cross_sections.xml);;All Files (*)")
        if filepath:
            self.path_field.setText(filepath)

    def apply_library_path(self):
        path = self.path_field.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid cross_sections.xml file path.")
            return

        # 1. تعيين متغير البيئة ليقرأه محرك OpenMC في الويندوز (في حال أردنا التشغيل محلياً لاحقاً)
        os.environ["OPENMC_CROSS_SECTIONS"] = path

        # 2. التحويل الذكي لمسار لينكس (WSL)
        wsl_path = path.replace('\\', '/')
        if len(wsl_path) > 2 and wsl_path[1] == ':':
            drive = wsl_path[0].lower()
            wsl_path = f"/mnt/{drive}" + wsl_path[2:]

        # 3. توليد الكود للشاشة البرتقالية
        code = f"\n# --- Cross Sections Library Setup (WSL Compatible) ---\n"
        code += f"openmc.config['cross_sections'] = '{wsl_path}'\n"

        # إرسال الكود والإشارات
        self.script_generated.emit(code)
        self.library_changed.emit(path)

        QMessageBox.information(
            self, "Success",
            f"Environment set for Windows.\n\nScript automatically updated for WSL with path:\n{wsl_path}"
        )

    def scan_isotopes(self):
        path = self.path_field.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", "Please specify a valid cross_sections.xml file first.")
            return

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            self.table.setRowCount(0)
            rows = []

            # قراءة العناصر والنظائر من ملف الـ XML
            for child in root.findall('nuclide') + root.findall('element') + root.findall('relaxation'):
                name = child.get('name', 'Unknown')
                alias = child.get('alias', '-')
                awr = child.get('awr', '-')
                filepath = child.get('path', '-')
                if name != 'Unknown':
                    rows.append((name, alias, awr, filepath))

            self.table.setRowCount(len(rows))
            for row_idx, (n, a, w, f) in enumerate(rows[:600]):  # عرض أول 600 مدخل لسرعة الواجهة
                self.table.setItem(row_idx, 0, QTableWidgetItem(n))
                self.table.setItem(row_idx, 1, QTableWidgetItem(a))
                self.table.setItem(row_idx, 2, QTableWidgetItem(w))
                self.table.setItem(row_idx, 3, QTableWidgetItem(f))

            QMessageBox.information(self, "Scan Complete",
                                    f"Successfully loaded {len(rows)} isotopes/materials from library.")
        except Exception as e:
            QMessageBox.critical(self, "Parsing Error", f"Failed to parse XML file:\n{str(e)}")