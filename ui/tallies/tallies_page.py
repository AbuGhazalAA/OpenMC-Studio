import openmc
import re  # تمت الإضافة لتعقيم أسماء المتغيرات
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QListWidget, QAbstractItemView, QMessageBox, QFormLayout, QStackedWidget
)
from PySide6.QtCore import Signal


class TalliesPageWidget(QWidget):
    """
    Advanced OpenMC Tallies Configuration Page.
    Supports every commonly-used OpenMC Filter type, combined multi-filter
    tallies (tally.filters = [f1, f2, ...]), arbitrary/MT-reaction scores,
    an optional nuclides restriction, and a Pulse Height (Detector F8)
    preset for HPGe/NaI-style detector simulations.
    """
    script_generated = Signal(str)

    # كل أنواع الفلاتر المدعومة، مبنية ديناميكياً (نفس نمط SURFACE_TYPES في
    # صفحة الهندسة): kind يحدد شكل حقول الإدخال المطلوبة.
    #   varlist   -> قائمة أسماء متغيرات مفصولة بفواصل (خلايا/مواد/أكوان/أسطح)
    #   singlevar -> متغير واحد فقط
    #   floatlist -> قائمة أرقام عشرية مفصولة بفواصل (تُدرج كما هي في الكود)
    #   intlist   -> قائمة أعداد صحيحة مفصولة بفواصل
    #   strlist   -> قائمة نصوص (توضع بين علامتي اقتباس تلقائياً)
    #   mesh      -> بُعد/زاوية سفلية/زاوية علوية لبناء RegularMesh
    FILTER_TYPES = {
        "CellFilter": {"kind": "varlist", "ctor": "CellFilter",
                       "label": "Cells (comma-separated variables):", "default": "c1"},
        "CellFromFilter": {"kind": "varlist", "ctor": "CellFromFilter",
                            "label": "Cells being entered (comma-separated):", "default": "c1"},
        "CellbornFilter": {"kind": "varlist", "ctor": "CellbornFilter",
                            "label": "Birth cells (comma-separated):", "default": "c1"},
        "MaterialFilter": {"kind": "varlist", "ctor": "MaterialFilter",
                            "label": "Materials (comma-separated variables):", "default": "WATER"},
        "MaterialFromFilter": {"kind": "varlist", "ctor": "MaterialFromFilter",
                                "label": "Materials being entered (comma-separated):", "default": "WATER"},
        "UniverseFilter": {"kind": "varlist", "ctor": "UniverseFilter",
                            "label": "Universes (comma-separated variables):", "default": "root"},
        "SurfaceFilter": {"kind": "varlist", "ctor": "SurfaceFilter",
                           "label": "Surfaces (comma-separated variables):", "default": "s1"},
        "DistribcellFilter": {"kind": "singlevar", "ctor": "DistribcellFilter",
                               "label": "Distributed Cell (single variable):", "default": "c1"},
        "EnergyFilter": {"kind": "floatlist", "ctor": "EnergyFilter",
                          "label": "Energy bin boundaries (eV, comma-separated):",
                          "default": "0.0, 1.0e6, 2.0e7"},
        "EnergyoutFilter": {"kind": "floatlist", "ctor": "EnergyoutFilter",
                             "label": "Outgoing-energy bin boundaries (eV, comma-separated):",
                             "default": "0.0, 1.0e6, 2.0e7"},
        "DelayedGroupFilter": {"kind": "intlist", "ctor": "DelayedGroupFilter",
                                "label": "Delayed groups (comma-separated integers):",
                                "default": "1, 2, 3, 4, 5, 6"},
        "ParticleFilter": {"kind": "strlist", "ctor": "ParticleFilter",
                            "label": "Particle types (comma-separated: neutron, photon, electron, positron):",
                            "default": "neutron"},
        "MeshFilter": {"kind": "mesh", "ctor": "MeshFilter"},
        "MeshSurfaceFilter": {"kind": "mesh", "ctor": "MeshSurfaceFilter"},
        "LegendreFilter": {"kind": "fields", "ctor": "LegendreFilter", "fields": [
            ("order", "Order (integer >= 0):", "3")]},
        "SpatialLegendreFilter": {"kind": "fields", "ctor": "SpatialLegendreFilter", "fields": [
            ("order", "Order (integer >= 0):", "3"), ("axis", "Axis (x, y, or z):", "x"),
            ("minimum", "Minimum (cm):", "-10.0"), ("maximum", "Maximum (cm):", "10.0")]},
        "SphericalHarmonicsFilter": {"kind": "fields", "ctor": "SphericalHarmonicsFilter", "fields": [
            ("order", "Order (integer >= 0):", "3"), ("cosine", "Cosine basis ('scatter' or 'particle'):", "scatter")]},
        "ZernikeFilter": {"kind": "fields", "ctor": "ZernikeFilter", "fields": [
            ("order", "Order (integer >= 0):", "4"), ("x", "Center x (cm):", "0.0"),
            ("y", "Center y (cm):", "0.0"), ("r", "Radius (cm):", "10.0")]},
        "ZernikeRadialFilter": {"kind": "fields", "ctor": "ZernikeRadialFilter", "fields": [
            ("order", "Order (integer >= 0):", "4"), ("x", "Center x (cm):", "0.0"),
            ("y", "Center y (cm):", "0.0"), ("r", "Radius (cm):", "10.0")]},
    }
    PULSE_HEIGHT_LABEL = "Pulse Height (Detector F8) [adds CellFilter + EnergyFilter]"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pending_filters = []  # list of dicts: {type, var, code, label}
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # ==========================================
        # --- قسم تعريف العداد (Tally Definition) ---
        # ==========================================
        tally_group = QGroupBox("Define Tally")
        tally_layout = QFormLayout(tally_group)

        self.tally_name_field = QLineEdit("tally1")
        tally_layout.addRow("Tally Name:", self.tally_name_field)

        self.scores_list = QListWidget()
        self.scores_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # "capture" ليست درجة صحيحة في OpenMC؛ المكافئ الصحيح للأسر الإشعاعي
        # هو تفاعل (n,gamma). أضفنا أيضاً بعض تفاعلات MT الشائعة و"heating"
        # التي تعتمد عليها صفحة Results. أي درجة أخرى (بما فيها أرقام MT
        # الخام) يمكن إضافتها عبر حقل "Custom Score" أدناه.
        scores = ["flux", "absorption", "fission", "total", "scatter", "nu-fission",
                  "heating", "heating-local", "kappa-fission",
                  "(n,gamma)", "(n,p)", "(n,a)", "(n,2n)", "(n,3n)",
                  "pulse-height"]
        self.scores_list.addItems(scores)
        self.scores_list.setFixedHeight(120)
        self.scores_list.setToolTip(
            "Hold Ctrl to select multiple scores. Select 'pulse-height' for HPGe/NaI detector simulations.")
        tally_layout.addRow("Scores to Measure:", self.scores_list)

        custom_score_layout = QHBoxLayout()
        self.custom_score_field = QLineEdit()
        self.custom_score_field.setPlaceholderText("e.g. (n,gamma), MT number as string, or any valid OpenMC score")
        btn_add_custom_score = QPushButton("+ Add")
        btn_add_custom_score.clicked.connect(self._add_custom_score)
        custom_score_layout.addWidget(self.custom_score_field, stretch=1)
        custom_score_layout.addWidget(btn_add_custom_score)
        tally_layout.addRow("Custom / MT-Reaction Score:", custom_score_layout)

        self.nuclides_field = QLineEdit()
        self.nuclides_field.setPlaceholderText("(optional) comma-separated nuclides, e.g. U235, H1 -- blank = all")
        tally_layout.addRow("Restrict to Nuclides (optional):", self.nuclides_field)

        main_layout.addWidget(tally_group)

        # ==========================================
        # --- قسم الفلاتر (يدعم فلاتر متعددة لكل عداد) ---
        # ==========================================
        filter_group = QGroupBox("Filters (a tally may combine multiple filters, e.g. Cell + Energy)")
        filter_layout = QFormLayout(filter_group)

        self.filter_type_combo = QComboBox()
        filter_names = list(self.FILTER_TYPES.keys()) + [self.PULSE_HEIGHT_LABEL]
        self.filter_type_combo.addItems(filter_names)
        self.filter_type_combo.currentTextChanged.connect(self._on_filter_type_changed)
        filter_layout.addRow("Filter Type:", self.filter_type_combo)

        self.filter_param_stack = QStackedWidget()
        self.filter_param_widgets = {}
        for type_name, spec in self.FILTER_TYPES.items():
            page = QWidget()
            page_form = QFormLayout(page)
            page_form.setContentsMargins(0, 0, 0, 0)
            if spec["kind"] == "mesh":
                dim = QLineEdit("10, 10, 1")
                ll = QLineEdit("-10.0, -10.0, -10.0")
                ur = QLineEdit("10.0, 10.0, 10.0")
                page_form.addRow("Dimension (Nx, Ny, Nz):", dim)
                page_form.addRow("Lower Left (x, y, z):", ll)
                page_form.addRow("Upper Right (x, y, z):", ur)
                self.filter_param_widgets[type_name] = {"dim": dim, "ll": ll, "ur": ur}
            elif spec["kind"] == "fields":
                field_widgets = {}
                for key, label, default in spec["fields"]:
                    edit = QLineEdit(default)
                    page_form.addRow(label, edit)
                    field_widgets[key] = edit
                self.filter_param_widgets[type_name] = field_widgets
            else:
                edit = QLineEdit(spec["default"])
                page_form.addRow(spec["label"], edit)
                self.filter_param_widgets[type_name] = {"value": edit}
            self.filter_param_stack.addWidget(page)

        # آخر صفحة: نموذج محاكاة الكاشف (Pulse Height preset)
        self.detector_widget = QWidget()
        d_layout = QFormLayout(self.detector_widget)
        d_layout.setContentsMargins(0, 0, 0, 0)
        self.d_cell_field = QLineEdit("c1")
        self.d_bins_field = QLineEdit("np.linspace(0, 3e6, 3000)")
        self.d_bins_field.setToolTip("Use numpy for bins, e.g., np.linspace(min, max, num_channels)")
        self.d_geb_a = QLineEdit("0.0")
        self.d_geb_b = QLineEdit("0.0001")
        self.d_geb_c = QLineEdit("0.0")
        d_layout.addRow("Detector Cell Variable:", self.d_cell_field)
        d_layout.addRow("Energy Channels (eV):", self.d_bins_field)
        d_layout.addRow(QLabel("Gaussian Energy Broadening (GEB): FWHM = a + b*sqrt(E + c*E^2)"))
        d_layout.addRow("GEB 'a':", self.d_geb_a)
        d_layout.addRow("GEB 'b':", self.d_geb_b)
        d_layout.addRow("GEB 'c':", self.d_geb_c)
        self.filter_param_stack.addWidget(self.detector_widget)

        filter_layout.addRow("Filter Settings:", self.filter_param_stack)

        filter_btn_layout = QHBoxLayout()
        btn_add_filter = QPushButton("➕ Add Filter to Tally")
        btn_add_filter.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        btn_add_filter.clicked.connect(self.add_filter_action)
        btn_remove_filter = QPushButton("🗑️ Remove Selected Filter")
        btn_remove_filter.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        btn_remove_filter.clicked.connect(self.remove_filter_action)
        filter_btn_layout.addWidget(btn_add_filter)
        filter_btn_layout.addWidget(btn_remove_filter)
        filter_layout.addRow(filter_btn_layout)

        filter_layout.addRow(QLabel("Filters queued for this tally (empty = Global tally):"))
        self.pending_filters_list = QListWidget()
        self.pending_filters_list.setFixedHeight(80)
        filter_layout.addRow(self.pending_filters_list)

        main_layout.addWidget(filter_group)

        # ==========================================
        # --- الأزرار والقائمة السفلية ---
        # ==========================================
        btn_add = QPushButton("Add Tally & Generate Script")
        btn_add.setStyleSheet("background-color: #0e639c; color: white; font-weight: bold; padding: 10px;")
        btn_add.clicked.connect(self.add_tally_action)
        main_layout.addWidget(btn_add)

        main_layout.addWidget(QLabel("<b>Created Tallies:</b>"))
        self.created_tallies_list = QListWidget()
        main_layout.addWidget(self.created_tallies_list)

    def _on_filter_type_changed(self, text):
        names = list(self.FILTER_TYPES.keys()) + [self.PULSE_HEIGHT_LABEL]
        if text in names:
            self.filter_param_stack.setCurrentIndex(names.index(text))

    def _add_custom_score(self):
        score = self.custom_score_field.text().strip()
        if not score:
            return
        existing = [self.scores_list.item(i).text() for i in range(self.scores_list.count())]
        if score not in existing:
            self.scores_list.addItem(score)
        for i in range(self.scores_list.count()):
            if self.scores_list.item(i).text() == score:
                self.scores_list.item(i).setSelected(True)
                break
        self.custom_score_field.clear()

    # ==========================================
    # --- Pending filter queue (multi-filter tallies) ---
    # ==========================================
    def add_filter_action(self):
        ftype = self.filter_type_combo.currentText()

        if ftype == self.PULSE_HEIGHT_LABEL:
            target_cell = self.d_cell_field.text().strip()
            energy_bins = self.d_bins_field.text().strip()
            if not target_cell or not energy_bins:
                QMessageBox.warning(self, "Warning", "Detector Cell Variable and Energy Channels are required.")
                return
            idx = len(self.pending_filters) + 1
            cell_var = f"f{idx}_cell"
            energy_var = f"f{idx}_energy"
            code = (f"{cell_var} = openmc.CellFilter([{target_cell}])\n"
                    f"{energy_var}_bins = {energy_bins}\n"
                    f"{energy_var} = openmc.EnergyFilter({energy_var}_bins)\n"
                    f"# GEB Parameters: a={self.d_geb_a.text()}, b={self.d_geb_b.text()}, c={self.d_geb_c.text()}\n")
            self.pending_filters.append({
                "type": "CellFilter", "var": cell_var, "code": "",
                "label": f"CellFilter({target_cell})"})
            self.pending_filters.append({
                "type": "EnergyFilter", "var": energy_var, "code": code,
                "label": f"EnergyFilter(detector channels)"})
            self.pending_filters_list.addItem(f"CellFilter({target_cell})")
            self.pending_filters_list.addItem(f"EnergyFilter(detector channels)")
            for i in range(self.scores_list.count()):
                if self.scores_list.item(i).text() == "pulse-height":
                    self.scores_list.item(i).setSelected(True)
                    break
            return

        var, code, label, error = self._build_filter(ftype)
        if error:
            QMessageBox.warning(self, "Warning", error)
            return

        self.pending_filters.append({"type": ftype, "var": var, "code": code, "label": label})
        self.pending_filters_list.addItem(label)

    def _build_filter(self, ftype):
        """Returns (var_name, code_text, display_label, error_message)."""
        spec = self.FILTER_TYPES[ftype]
        idx = len(self.pending_filters) + 1
        var = f"f{idx}_{spec['ctor']}"

        if spec["kind"] == "mesh":
            w = self.filter_param_widgets[ftype]
            dim, ll, ur = w["dim"].text().strip(), w["ll"].text().strip(), w["ur"].text().strip()
            if not dim or not ll or not ur:
                return None, None, None, "Mesh Dimension, Lower Left and Upper Right are all required."
            mesh_var = f"{var}_mesh"
            code = (f"{mesh_var} = openmc.RegularMesh()\n"
                    f"{mesh_var}.dimension = ({dim})\n"
                    f"{mesh_var}.lower_left = ({ll})\n"
                    f"{mesh_var}.upper_right = ({ur})\n"
                    f"{var} = openmc.{spec['ctor']}({mesh_var})\n")
            return var, code, f"{ftype}(dim={dim})", None

        if spec["kind"] == "fields":
            w = self.filter_param_widgets[ftype]
            vals = {}
            for key, edit in w.items():
                vals[key] = edit.text().strip()
                if not vals[key]:
                    return None, None, None, f"{ftype}: '{key}' cannot be empty."

            try:
                order = int(vals["order"])
                if order < 0:
                    raise ValueError
            except ValueError:
                return None, None, None, f"{ftype}: Order must be a non-negative integer."

            if ftype == "LegendreFilter":
                code = f"{var} = openmc.LegendreFilter(order={order})\n"
                return var, code, f"LegendreFilter(order={order})", None

            if ftype == "SpatialLegendreFilter":
                axis = vals["axis"].strip().lower()
                if axis not in ("x", "y", "z"):
                    return None, None, None, "SpatialLegendreFilter: Axis must be 'x', 'y', or 'z'."
                try:
                    minimum, maximum = float(vals["minimum"]), float(vals["maximum"])
                except ValueError:
                    return None, None, None, "SpatialLegendreFilter: Minimum/Maximum must be numbers."
                if minimum >= maximum:
                    return None, None, None, "SpatialLegendreFilter: Minimum must be less than Maximum."
                code = (f"{var} = openmc.SpatialLegendreFilter(order={order}, axis='{axis}', "
                        f"minimum={minimum}, maximum={maximum})\n")
                return var, code, f"SpatialLegendreFilter(order={order}, axis={axis})", None

            if ftype == "SphericalHarmonicsFilter":
                cosine = vals["cosine"].strip().lower()
                if cosine not in ("scatter", "particle"):
                    return None, None, None, "SphericalHarmonicsFilter: Cosine basis must be 'scatter' or 'particle'."
                code = f"{var} = openmc.SphericalHarmonicsFilter(order={order}, cosine='{cosine}')\n"
                return var, code, f"SphericalHarmonicsFilter(order={order}, cosine={cosine})", None

            if ftype in ("ZernikeFilter", "ZernikeRadialFilter"):
                try:
                    x, y, r = float(vals["x"]), float(vals["y"]), float(vals["r"])
                except ValueError:
                    return None, None, None, f"{ftype}: x/y/r must be numbers."
                if r <= 0:
                    return None, None, None, f"{ftype}: Radius must be > 0."
                code = f"{var} = openmc.{spec['ctor']}(order={order}, x={x}, y={y}, r={r})\n"
                return var, code, f"{ftype}(order={order}, x={x}, y={y}, r={r})", None

        value = self.filter_param_widgets[ftype]["value"].text().strip()
        if not value:
            return None, None, None, f"{ftype} requires a value."

        if spec["kind"] == "varlist":
            code = f"{var} = openmc.{spec['ctor']}([{value}])\n"
            return var, code, f"{ftype}({value})", None
        if spec["kind"] == "singlevar":
            code = f"{var} = openmc.{spec['ctor']}({value})\n"
            return var, code, f"{ftype}({value})", None
        if spec["kind"] == "floatlist":
            try:
                [float(x) for x in value.split(",")]
            except ValueError:
                return None, None, None, f"Invalid numeric bin list for {ftype}: '{value}'"
            code = f"{var} = openmc.{spec['ctor']}([{value}])\n"
            return var, code, f"{ftype}([{value}])", None
        if spec["kind"] == "intlist":
            try:
                [int(x) for x in value.split(",")]
            except ValueError:
                return None, None, None, f"Invalid integer list for {ftype}: '{value}'"
            code = f"{var} = openmc.{spec['ctor']}([{value}])\n"
            return var, code, f"{ftype}([{value}])", None
        if spec["kind"] == "strlist":
            names = [x.strip() for x in value.split(",") if x.strip()]
            quoted = ", ".join(f"'{n}'" for n in names)
            code = f"{var} = openmc.{spec['ctor']}([{quoted}])\n"
            return var, code, f"{ftype}([{quoted}])", None

        return None, None, None, f"Unknown filter kind for {ftype}."

    def remove_filter_action(self):
        row = self.pending_filters_list.currentRow()
        if row < 0:
            return
        self.pending_filters_list.takeItem(row)
        self.pending_filters.pop(row)

    def _clear_pending_filters(self):
        self.pending_filters = []
        self.pending_filters_list.clear()

    # ==========================================
    # --- Tally creation ---
    # ==========================================
    def add_tally_action(self):
        raw_name = self.tally_name_field.text().strip()
        selected_items = self.scores_list.selectedItems()

        if not raw_name:
            QMessageBox.warning(self, "Warning", "Please provide a valid Tally Name.")
            return

        # --- التعقيم الذكي (Sanitization) ---
        # تحويل أي حرف غير صالح لبايثون إلى شرطة سفلية، وضمان عدم البدء برقم
        t_var_name = re.sub(r'\W', '_', raw_name)
        if re.match(r'^\d', t_var_name):
            t_var_name = f"t_{t_var_name}"

        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one Score (e.g., flux).")
            return

        selected_scores = [item.text() for item in selected_items]

        # OpenMC restricts 'pulse-height' tallies to exactly one CellFilter
        # plus one EnergyFilter (the EnergyFilter defines the deposited-
        # energy channel binning for the pulse-height spectrum, not the
        # incident particle energy). Any other filter combination is
        # invalid and would otherwise only fail once the script runs.
        if "pulse-height" in selected_scores:
            ftypes = sorted(f["type"] for f in self.pending_filters)
            if ftypes != ["CellFilter", "EnergyFilter"]:
                QMessageBox.warning(
                    self, "Warning",
                    "The 'pulse-height' score requires exactly one CellFilter and one EnergyFilter "
                    "(use the 'Pulse Height (Detector F8)' filter type, which adds both at once).")
                return

        # --- توليد كود البايثون (Script Generation) ---
        py_code = f"\n# --- Tally Definition: {raw_name} ---\n"
        for f in self.pending_filters:
            if f["code"]:
                py_code += f["code"]

        # نستخدم t_var_name المتغير المعقم في الكود، و raw_name كاسم للعرض في OpenMC
        py_code += f"{t_var_name} = openmc.Tally(name='{raw_name}')\n"
        scores_str = ", ".join([f"'{s}'" for s in selected_scores])
        py_code += f"{t_var_name}.scores = [{scores_str}]\n"

        nuclides_txt = self.nuclides_field.text().strip()
        if nuclides_txt:
            nuclide_names = [n.strip() for n in nuclides_txt.split(",") if n.strip()]
            nuclides_str = ", ".join(f"'{n}'" for n in nuclide_names)
            py_code += f"{t_var_name}.nuclides = [{nuclides_str}]\n"

        if self.pending_filters:
            filters_str = ", ".join(f["var"] for f in self.pending_filters)
            py_code += f"{t_var_name}.filters = [{filters_str}]\n"
            filter_desc = " + ".join(f["label"] for f in self.pending_filters)
        else:
            filter_desc = "Global"

        py_code += (
            f"try:\n"
            f"    tallies.append({t_var_name})\n"
            f"except NameError:\n"
            f"    tallies = openmc.Tallies([{t_var_name}])\n"
        )

        self.script_generated.emit(py_code)
        self.created_tallies_list.addItem(
            f"Tally: {raw_name} | Filters: {filter_desc} | Scores: {', '.join(selected_scores)}")
        QMessageBox.information(self, "Success", f"Tally '{raw_name}' added to script successfully!")
        self._clear_pending_filters()