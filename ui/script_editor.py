import os
import re
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QPlainTextEdit, QFileDialog, QMessageBox, QInputDialog, QLineEdit
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

    def replace_tagged_block(self, tag, code):
        """Like append_code, but for a page (currently only the Geometry
        Builder) that re-emits its ENTIRE current state on every single
        add/delete/undo action rather than one incremental chunk per
        click. Plain append_code would leave every earlier version of
        that full block sitting in the script too -- so clicking Undo
        never actually removed the undone surface/cell from the visible
        script text, it just appended yet another (correct) block on top
        of the still-present stale ones.

        Wraps `code` in `# >>> BEGIN {tag} BLOCK <<<` / `# >>> END {tag}
        BLOCK <<<` markers. If a block with this tag already exists in
        the script, replaces just that region in place (preserving
        anything the user typed before/after it, including manual edits
        to the OTHER pages' blocks); otherwise appends a new tagged block
        at the end, same as append_code would.
        """
        begin_marker = f"# >>> BEGIN {tag} BLOCK <<<"
        end_marker = f"# >>> END {tag} BLOCK <<<"
        wrapped = f"{begin_marker}\n{code}\n{end_marker}\n"

        text = self.editor.toPlainText()
        start = text.find(begin_marker)
        end = text.find(end_marker)
        if start != -1 and end != -1 and end > start:
            end += len(end_marker)
            new_text = text[:start] + wrapped.rstrip('\n') + text[end:]
            self.editor.setPlainText(new_text)
        else:
            self.editor.appendPlainText(wrapped)

    def run_script(self):
        code = self.editor.toPlainText()

        # --- ميزة حماية الملفات وإعادة التسمية التلقائية (Auto-Renaming) ---
        sim_name, ok = QInputDialog.getText(
            self, "Simulation Name",
            "Enter a name for this simulation (to protect output files from being overwritten):",
            QLineEdit.EchoMode.Normal, "MyReactor_Sim"
        )

        # إذا أدخل المستخدم اسماً، نقوم بإضافة كود مخفي في نهاية السكربت لإعادة تسمية الملفات
        if ok and sim_name.strip():
            safe_name = re.sub(r'\W', '_', sim_name.strip())  # تعقيم الاسم برمجياً
            rename_code_snippet = f"""
# --- OpenMC Studio Auto-Rename Protection ---
import os
import glob
try:
    prefix = '{safe_name}'
    # تغيير اسم ملف الاستنزاف
    if os.path.exists('depletion_results.h5'):
        os.rename('depletion_results.h5', f'{{prefix}}_depletion_results.h5')
        print(f'\\n[OpenMC Studio] ✅ Protected: depletion_results.h5 -> {{prefix}}_depletion_results.h5')

    # تغيير اسم ملفات الـ Statepoint
    for sp_file in glob.glob('statepoint.*.h5'):
        if not sp_file.startswith(prefix):
            os.rename(sp_file, f'{{prefix}}_{{sp_file}}')
            print(f'[OpenMC Studio] ✅ Protected: {{sp_file}} -> {{prefix}}_{{sp_file}}')
except Exception as e:
    print(f'[OpenMC Studio] ⚠️ Warning: Auto-rename failed: {{e}}')
"""
            code_to_run = code + "\n" + rename_code_snippet
        else:
            # إذا لم يُدخل اسماً أو ضغط إلغاء، يعمل السكربت كالمعتاد
            code_to_run = code

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

                # يُعيد ضبط سجل المعرّفات التلقائية قبل كل exec() -- بدون هذا،
                # تشغيل هذا الزر أكثر من مرة (أو بعد أي صفحة أخرى نفّذت السكريبت
                # مسبقاً، مثل Validate Geometry) يترك معرّفات Material/Surface/
                # Cell القديمة مسجّلة، فتتصادم مع المعرّفات الصريحة الجديدة
                # ويطبع OpenMC عشرات تحذيرات IDWarning. نفس النمط المستخدم في
                # كل مسارات exec() الأخرى بالتطبيق (plots_page.py، إلخ).
                _omcs_shim_openmc.reset_auto_ids()
                # Every other exec() path in the app (plots_page.py,
                # voxel_page.py, tracks_page.py, results_page.py) seeds
                # its namespace with {'openmc': openmc, ...} before
                # exec()'ing, so a script built purely through the GUI
                # builders (which never insert an `import openmc` line of
                # their own -- see materials_page.py/geometry_page.py/etc,
                # they only emit `openmc.Material(...)`-style calls) still
                # runs. This path used to be the one exception: `namespace`
                # started empty, so the SAME GUI-built script that runs
                # fine everywhere else failed here with "NameError: name
                # 'openmc' is not defined" the moment it used a bare
                # `openmc.X` call.
                namespace['openmc'] = _omcs_shim_openmc
            except ImportError:
                pass

            # تنفيذ الكود المحسن (code_to_run) بدلاً من القديم
            exec(code_to_run, namespace)
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