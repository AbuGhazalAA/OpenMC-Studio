import os
import re
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QPlainTextEdit, QFileDialog, QMessageBox
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#00008B"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            r"\bimport\b", r"\bfrom\b", r"\bclass\b", r"\bdef\b",
            r"\bif\b", r"\belif\b", r"\belse\b", r"\btry\b", r"\bexcept\b",
            r"\bfor\b", r"\bin\b", r"\bwhile\b", r"\breturn\b", r"\bpass\b",
            r"\bTrue\b", r"\bFalse\b", r"\bNone\b", r"\band\b", r"\bor\b", r"\bnot\b"
        ]
        for word in keywords:
            self.highlighting_rules.append((re.compile(word), keyword_format))

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#006400"))
        string_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((re.compile(r"\".*\""), string_format))
        self.highlighting_rules.append((re.compile(r"'.*'"), string_format))

        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#8B0000"))
        number_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((re.compile(r"\b[0-9]+\.?[0-9]*\b"), number_format))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#111111"))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((re.compile(r"#[^\n]*"), comment_format))

    def highlightBlock(self, text):
        for pattern, text_format in self.highlighting_rules:
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - match.start()
                self.setFormat(start, length, text_format)


class ScriptEditorWidget(QWidget):
    script_executed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ Run / Sync to GUI")
        self.btn_run.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px;")
        self.btn_run.clicked.connect(self.run_script)

        self.btn_save = QPushButton("💾 Save Script")
        self.btn_save.setStyleSheet("background-color: #555555; color: white; font-weight: bold; padding: 5px;")
        self.btn_save.clicked.connect(self.save_script)

        self.btn_load = QPushButton("📂 Load Script")
        self.btn_load.setStyleSheet("background-color: #555555; color: white; font-weight: bold; padding: 5px;")
        self.btn_load.clicked.connect(self.load_script)

        toolbar_layout.addWidget(self.btn_run)
        toolbar_layout.addWidget(self.btn_save)
        toolbar_layout.addWidget(self.btn_load)
        layout.addLayout(toolbar_layout)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("ScriptArea")

        layout.addWidget(self.editor)
        self.highlighter = PythonHighlighter(self.editor.document())

    def set_theme(self, theme):
        """دالة للتبديل بين لون السكريبت البرتقالي أو الرمادي"""
        bg_color = "#FF8C00" if theme == "colored" else "#E5E5E5"
        self.editor.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {bg_color}; 
                color: #000000;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 15px;
                font-weight: bold;
                border: none;
            }}
        """)

    def append_code(self, code):
        self.editor.appendPlainText(code)

    def run_script(self):
        code = self.editor.toPlainText()
        namespace = {}
        try:
            # حماية .plot() التفاعلية (تحتاج openmc.lib، غالباً غير متوفرة/معطوبة
            # خارج بناء conda/Linux كامل) — نفس الحماية المطبقة في
            # PlotsPageWidget، مطلوبة هنا أيضاً لأن هذا الزر ينفّذ exec() منفصلاً
            # تماماً عن _export_simulation_xml.
            try:
                import openmc as _omcs_shim_openmc

                def _omcs_safe_plot_stub(self, *args, **kwargs):
                    print('[OpenMC Studio] Skipped interactive .plot() call '
                          '(needs openmc.lib) -- use "Validate Geometry & Plot" instead.')
                    return None

                for _cls_name in ('Universe', 'Model', 'Geometry', 'Cell', 'Region'):
                    _cls = getattr(_omcs_shim_openmc, _cls_name, None)
                    if _cls is not None and hasattr(_cls, 'plot'):
                        _cls.plot = _omcs_safe_plot_stub
            except ImportError:
                pass

            exec(code, namespace)
            self.script_executed.emit(namespace)
        except Exception as e:
            QMessageBox.warning(self, "Script Error", str(e))

    def save_script(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Python Script", "", "Python Files (*.py)")
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.editor.toPlainText())

    def load_script(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Python Script", "", "Python Files (*.py)")
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.editor.setPlainText(f.read())