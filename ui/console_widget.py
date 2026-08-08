from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit


class ConsoleWidget(QWidget):
    """
    Live Console output viewer for OpenMC Engine.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        # تنسيق بألوان الكونسول الاحترافي (خلفية داكنة وخط أخضر/أبيض)
        self.text_edit.setStyleSheet(
            "background-color: #1e1e1e; color: #dcdcaa; font-family: Consolas, Courier New, monospace; font-size: 13px;")
        layout.addWidget(self.text_edit)

    def append_log(self, text):
        """إضافة نص جديد مع التمرير التلقائي للأسفل"""
        self.text_edit.append(text)
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """مسح الشاشة قبل تشغيل محاكاة جديدة"""
        self.text_edit.clear()