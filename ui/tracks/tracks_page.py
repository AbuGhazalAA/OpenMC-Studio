import os
import shutil
import subprocess
import sys
import time
import h5py
import numpy as np
import openmc

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QFormLayout, QSplitter, QSpinBox, QApplication, QCheckBox,
    QDialog, QProgressBar, QMessageBox
)
from PySide6.QtCore import Signal, QThread, QTimer, Qt
from PySide6.QtGui import QPixmap

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import matplotlib.image as mpimg
import matplotlib.patheffects as pe

from ui.plots.plots_page import ZoomableImageViewer


# Color coding matches common HP/ViSED-style conventions (not an OpenMC
# standard -- just a readable default): neutron=blue, photon=green,
# electron=red, positron=orange. Anything else falls back to gray.
def _xyz_field_to_array(field):
    """OpenMC's Track states['r'] and states['u'] are NOT plain (N, 3)
    float arrays -- they are structured arrays with named sub-fields
    'x', 'y', 'z' (dtype [('x','<f8'), ('y','<f8'), ('z','<f8')]).
    Confirmed directly from a real runtime error:
        ufunc 'minimum' did not contain a loop with signature matching
        types (dtype([('x','<f8'),('y','<f8'),('z','<f8')]), ...)
    This converts either a single structured record (0-d, one state) or
    an array of them (1-d, N states) into a plain (N, 3) float array,
    so all downstream code (indexing, np.einsum, min/max, etc.) can work
    with ordinary numeric arrays instead of field-name access everywhere.
    """
    field = np.atleast_1d(field)  # promotes a 0-d single-state record to shape (1,)
    return np.stack([field['x'], field['y'], field['z']], axis=-1)


PARTICLE_COLORS = {
    'neutron': '#1f77b4',
    'photon': '#2ca02c',
    'electron': '#d62728',
    'positron': '#ff7f0e',
}

# ── Reading tracks.h5 without depending on the openmc Python package ──
#
# OpenMC's C++ writer (src/track_output.cpp) stores the particle type as a
# PDG Monte Carlo number:
#     write_attribute(dset, "particles", particles);   // pdg_number()
# while openmc/tracks.py turns that straight back into a type:
#     ptype = ParticleType(particle)
#
# Those two halves only agree when the binary and the Python package come
# from the same era. In OpenMC 0.16 ParticleType became a PDG-backed class;
# before that it was an IntEnum numbered 0..3. Pair a PDG-writing binary
# with a pre-PDG package -- which is what the openmc-windows-beta build
# ships -- and reading any track file dies with:
#     np.int32(22) is not a valid ParticleType
# (22 being the PDG number for a photon). Nothing is wrong with the run or
# the file; only the reader disagrees.
#
# So the file is also readable directly through h5py, accepting BOTH
# conventions. Same approach already used for depletion_results.h5 in
# ui/results_page.py, and for the same reason.
_PDG_TO_NAME = {
    2112: 'neutron',
    22: 'photon',
    11: 'electron',
    -11: 'positron',
    2212: 'proton',
    1000010020: 'deuteron',
    1000010030: 'triton',
    1000020040: 'alpha',
}

# Pre-0.16 files stored a bare 0..3 index here instead of a PDG number.
# The two encodings cannot collide: no legacy index is a PDG number in the
# table above, so trying PDG first and falling back is unambiguous.
_LEGACY_INDEX_TO_PDG = {0: 2112, 1: 22, 2: 11, 3: -11}


class _DirectParticleType:
    """Stands in for openmc.ParticleType, exposing just the `.name` the
    rendering code reads."""
    __slots__ = ('name', 'pdg_number')

    def __init__(self, name, pdg_number):
        self.name = name
        self.pdg_number = pdg_number

    def __repr__(self):
        return self.name


class _DirectParticleTrack:
    """Mirrors openmc.ParticleTrack: a particle type plus its states."""
    __slots__ = ('particle', 'states')

    def __init__(self, particle, states):
        self.particle = particle
        self.states = states


class _DirectTrack:
    """Mirrors openmc.Track: one source history and everything it spawned."""
    __slots__ = ('particle_tracks', 'identifier')

    def __init__(self, particle_tracks, identifier=(0, 0, 0)):
        self.particle_tracks = particle_tracks
        self.identifier = identifier


def _track_sort_key(dset_name):
    """Datasets are named track_<batch>_<generation>_<particle id>; sort by
    those numbers so histories come out in run order rather than the
    lexicographic order that would put track_10 before track_2."""
    try:
        _, batch, gen, particle = dset_name.split('_')
        return (int(batch), int(gen), int(particle))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _particle_name_from_code(code):
    """PDG number (or legacy 0..3 index) -> lowercase particle name."""
    code = int(code)
    if code in _PDG_TO_NAME:
        return _PDG_TO_NAME[code]
    if code in _LEGACY_INDEX_TO_PDG:
        return _PDG_TO_NAME[_LEGACY_INDEX_TO_PDG[code]]
    return f'particle {code}'


def _read_tracks_h5(filepath, particle_filter='all'):
    """Read tracks.h5 with plain h5py, returning objects shaped like the
    ones openmc.Tracks yields.

    Layout, per OpenMC's own writer and reader:
        /track_<batch>_<gen>_<id>   Dataset of states
            attrs['offsets']    where each particle's states start,
                                with a final entry closing the last slice
            attrs['particles']  one particle code per slice
    """
    tracks = []
    with h5py.File(filepath, 'r') as fh:
        for dset_name in sorted(fh, key=_track_sort_key):
            dset = fh[dset_name]
            if not isinstance(dset, h5py.Dataset):
                continue
            states = dset[()]
            if 'offsets' not in dset.attrs or 'particles' not in dset.attrs:
                continue

            offsets = np.atleast_1d(dset.attrs['offsets']).astype(np.int64)
            particles = np.atleast_1d(dset.attrs['particles']).astype(np.int64)

            # OpenMC writes one more offset than particles so the last
            # slice is closed. Tolerate a file that omits that final entry
            # rather than dropping the last particle of every history.
            if offsets.size == particles.size:
                offsets = np.append(offsets, len(states))

            particle_tracks = []
            for code, start, end in zip(particles, offsets[:-1], offsets[1:]):
                name = _particle_name_from_code(code)
                if particle_filter != 'all' and name != particle_filter:
                    continue
                particle_tracks.append(_DirectParticleTrack(
                    _DirectParticleType(name, int(code)),
                    states[int(start):int(end)]))

            if particle_tracks:
                tracks.append(_DirectTrack(particle_tracks, _track_sort_key(dset_name)))
    return tracks

# Interaction (collision) detection uses OpenMC's own recorded cell_id/
# material_id per state, not a guessed angle threshold: a state transition
# where BOTH stay the same as the previous state means the particle didn't
# cross a surface into a new cell -- OpenMC only writes a new state on a
# real event, so "new state, same cell+material" is definitionally a
# collision (scatter, absorption+reemission, etc.) rather than a boundary
# crossing. This replaced an earlier direction-angle-change heuristic,
# which could both miss small-angle scatters and misfire on accumulated
# floating-point noise across many boundary crossings.


def _neutralize_interactive_plot_methods():
    """See ui/plots/plots_page.py for the full rationale: OpenMC's
    interactive Universe.plot()/Model.plot()/etc. need openmc.lib, which
    is frequently unavailable/broken outside a full conda/Linux build.
    Duplicated here (rather than imported) so this module has no import
    dependency on plots_page.py beyond ZoomableImageViewer, keeping the
    two pages independently robust if one is edited without the other.
    """
    def _safe_plot_stub(self, *args, **kwargs):
        print('[OpenMC Studio] Skipped interactive .plot() call '
              '(needs openmc.lib) -- use "Force Render Now" instead.')
        return None

    for cls_name in ('Universe', 'Model', 'Geometry', 'Cell', 'Region'):
        cls = getattr(openmc, cls_name, None)
        if cls is not None and hasattr(cls, 'plot'):
            cls.plot = _safe_plot_stub


def _discover_model_objects(script_code):
    """Exec script_code in-process and return whatever OpenMC model
    objects it defines, matched by TYPE (not variable name) so this
    works regardless of naming convention (geometry/geom, materials/mats,
    etc.) -- same robust approach used for geometry discovery elsewhere
    in this app.

    Returns (geometry, materials_list, settings, tallies, error_message).
    error_message is None on success.
    """
    _neutralize_interactive_plot_methods()
    # Reset OpenMC's global auto-ID registry before exec'ing -- without
    # this, re-running the SAME script (e.g. after already using "Force
    # Render Now" in the 2D Geometry Viewer earlier in this session)
    # collides with the still-registered Material/Surface/Cell IDs from
    # the previous exec() and floods the console with IDWarnings. Same
    # pattern already used correctly in PlotsPageWidget.render_live_plot().
    openmc.reset_auto_ids()
    namespace = {'openmc': openmc, 'os': os, '__name__': '__main__'}
    try:
        exec(script_code, namespace)
    except Exception as e:
        msg = str(e).strip() or f"{type(e).__name__} (no message attached to the exception)"
        return None, [], None, None, msg

    geometry = None
    materials_list = []
    settings_obj = None
    tallies_obj = None
    for val in namespace.values():
        if isinstance(val, openmc.Geometry):
            geometry = val
        elif isinstance(val, openmc.Material):
            materials_list.append(val)
        elif isinstance(val, openmc.Settings):
            settings_obj = val
        elif isinstance(val, openmc.Tallies):
            tallies_obj = val

    if geometry is None:
        return None, [], None, None, "No 'openmc.Geometry' object was found in the code."
    if settings_obj is None:
        return None, [], None, None, ("No 'openmc.Settings' object was found -- a tracked run needs a "
                                       "valid source, so 'settings' (with .source set) must be defined.")
    return geometry, materials_list, settings_obj, tallies_obj, None


def _render_overlay(geometry_png_path, tracks, basis, origin, width, out_png_path,
                     show_tracks=True, show_source=True, show_collisions=True,
                     source_marker_size=7, reveal_fraction=1.0, reveal_time=None, dpi=150):
    """Composite the background geometry PNG with real particle-track
    polylines, source-birth markers, and interaction-point markers, all
    aligned in PHYSICAL (cm) coordinates via matplotlib's `extent` -- this
    sidesteps needing to hand-compute pixel offsets, and automatically
    stays correct for any Basis/Origin/Width combination.

    The four show_* flags are independent toggles (any combination,
    including all four or just one), matching how the UI checkboxes work.

    Track lines are colored by particle type (see PARTICLE_COLORS) but
    drawn as a LineCollection whose per-segment ALPHA is driven by the
    real recorded energy (states['E']) relative to that particle's own
    starting energy -- full energy is fully opaque, and the line fades as
    the particle loses energy along its path, so energy loss is visible
    directly on the track itself.

    `reveal_time`, when given (absolute seconds, matching states['time']),
    truncates each track to only the states recorded at or before that
    physical time -- used for time-accurate animation playback (see
    TrackSimulationWorker.run) so two tracks of different real duration
    reveal proportionally to actual elapsed simulation time rather than
    their own array-index fraction. `reveal_fraction` (index-based) is
    kept as a fallback for the rare case where no usable time data exists;
    reveal_time takes precedence whenever it is not None.

    `dpi` is exposed so animation frames (which don't need to be
    razor-sharp -- they're each on screen for a fraction of a second) can
    be rendered cheaper than the final "kept" frame, reducing the total
    CPU work of a multi-frame animated run.
    """
    img = mpimg.imread(geometry_png_path)

    h_idx, v_idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[basis]
    h_center, v_center = origin[h_idx], origin[v_idx]
    h_half, v_half = width[0] / 2.0, width[1] / 2.0
    extent = (h_center - h_half, h_center + h_half, v_center - v_half, v_center + v_half)

    fig = Figure(figsize=(10, 10), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    # origin='upper': row 0 of the PNG is the TOP of the image, matching
    # OpenMC's own render convention (top of image = max value of the
    # vertical basis coordinate) -- consistent with how Origin/Width are
    # already used for the 2D Geometry Viewer elsewhere in this app.
    ax.imshow(img, extent=extent, origin='upper')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    seen_particles = set()
    if show_tracks or show_source or show_collisions:
        for track in tracks:
            for i, ptrack in enumerate(track.particle_tracks):
                r = _xyz_field_to_array(ptrack.states['r'])  # always a plain (N, 3) float array now
                # For animated playback: only reveal the states recorded
                # up to a cutoff. Prefer real elapsed TIME (reveal_time)
                # over array-index fraction whenever available -- see the
                # docstring above. A single-state track is unaffected
                # (nothing to truncate); always keep at least 1 state so
                # it can still register.
                if r.shape[0] > 1:
                    if reveal_time is not None:
                        times = np.atleast_1d(ptrack.states['time']).astype(float)
                        n_keep = max(1, int(np.searchsorted(times, reveal_time, side='right')))
                        r = r[:n_keep]
                    elif reveal_fraction < 1.0:
                        n_keep = max(1, int(np.ceil(r.shape[0] * reveal_fraction)))
                        r = r[:n_keep]
                pname = ptrack.particle.name.lower()
                color = PARTICLE_COLORS.get(pname, '#888888')
                label = pname.capitalize() if pname not in seen_particles else None

                # A particle with only ONE recorded state -> r.shape[0]==1
                # after the conversion above. Drawn as a small marker
                # (when show_source is on) rather than silently dropped --
                # itself diagnostic: only markers and no lines means
                # particles are being absorbed at/near birth.
                if r.shape[0] < 2:
                    if show_source:
                        seen_particles.add(pname)
                        ax.plot(r[0, h_idx], r[0, v_idx], marker='x', color=color, markersize=8,
                                markeredgewidth=2.0, alpha=0.95, zorder=5, label=label,
                                path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
                    continue

                h = r[:, h_idx]
                v = r[:, v_idx]

                if show_tracks:
                    seen_particles.add(pname)
                    # Energy-loss gradient: alpha per segment follows this
                    # particle's OWN energy relative to its starting
                    # energy (states['E'], eV) -- full energy = opaque,
                    # fully spent = faint. Falls back to a flat, fully
                    # opaque line if no positive starting energy is
                    # recorded (e.g. a build/version without E populated).
                    energies = np.atleast_1d(ptrack.states['E']).astype(float)[:r.shape[0]]
                    e0 = energies[0] if energies.size and energies[0] > 0 else 0.0
                    if e0 > 0:
                        frac = np.clip(energies / e0, 0.0, 1.0)
                        seg_frac = (frac[:-1] + frac[1:]) / 2.0
                        min_alpha = 0.35
                        seg_alpha = min_alpha + (1.0 - min_alpha) * seg_frac
                    else:
                        seg_alpha = np.full(max(0, len(h) - 1), 0.95)

                    points = np.stack([h, v], axis=1).reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    seg_colors = np.tile(np.array(to_rgba(color)), (len(segments), 1))
                    seg_colors[:, 3] = seg_alpha

                    lc = LineCollection(
                        segments, colors=seg_colors, linewidths=1.6, zorder=3, label=label,
                        path_effects=[pe.Stroke(linewidth=2.6, foreground='white', alpha=0.6), pe.Normal()])
                    ax.add_collection(lc)

                # Source-birth marker: only the FIRST particle_track entry
                # in a Track is the primary (source) particle; the rest
                # are secondaries (e.g. Compton electrons) born
                # mid-geometry. Small filled circle, not a star -- with
                # ~20 source points clustered close together a star's
                # 5-point spikes overlap into a messy blob; a circle
                # stays a clean, compact mark at any density.
                if show_source and i == 0:
                    seen_particles.add(pname)
                    ax.plot(h[0], v[0], marker='o', color=color, markersize=source_marker_size,
                            markeredgecolor='black', markeredgewidth=0.7, zorder=5,
                            label=label if not show_tracks else None,
                            path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])

                # Interaction markers: a state is a real collision when its
                # cell_id AND material_id both match the PREVIOUS state --
                # meaning this new recorded state happened without crossing
                # a surface into a different cell (see the module-level
                # comment above for why that's a reliable, data-grounded
                # test rather than a guessed angle threshold).
                if show_collisions:
                    cell_ids = np.atleast_1d(ptrack.states['cell_id'])[:r.shape[0]]
                    mat_ids = np.atleast_1d(ptrack.states['material_id'])[:r.shape[0]]
                    if cell_ids.shape[0] > 2:
                        same_cell = (cell_ids[1:] == cell_ids[:-1]) & (mat_ids[1:] == mat_ids[:-1])
                        hit_idx = np.where(same_cell)[0] + 1
                        if len(hit_idx) > 0:
                            ax.plot(h[hit_idx], v[hit_idx], '.', color=color,
                                    markersize=5, alpha=0.95, zorder=4,
                                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    if seen_particles:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    axis_labels = {'xy': ('X [cm]', 'Y [cm]'), 'xz': ('X [cm]', 'Z [cm]'), 'yz': ('Y [cm]', 'Z [cm]')}
    xl, yl = axis_labels[basis]
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    title_bits = []
    if show_source:
        title_bits.append("\u25cf source birth")
    if show_collisions:
        title_bits.append("\u00b7 interaction")
    suffix = (" -- " + ", ".join(title_bits)) if title_bits else ""
    ax.set_title(f"Particle Tracks ({basis} slice){suffix}")

    fig.tight_layout()
    canvas.print_png(out_png_path)
    fig.clear()  # drop all axes/artists now rather than waiting for GC,
                 # since this function may be called many times in a row
                 # (once per animation frame) within a single run


def _looks_transient_io_error(text):
    """Heuristic: does this error text look like a passing file-lock/IO
    hiccup (antivirus real-time scan, cloud sync, Windows Search
    indexing, etc.) rather than a genuine model/configuration error?
    Used to decide whether a retry is worth attempting -- retrying a
    REAL error (e.g. missing cross-section data, invalid settings) would
    just waste time repeating the same failure.
    """
    text_lower = (text or '').lower()
    return any(s in text_lower for s in (
        'write error', 'being used by another process', 'access is denied',
        'permission denied', 'sharing violation', 'resource temporarily unavailable',
        'device or resource busy',
    ))


class TrackSimulationWorker(QThread):
    """Runs a REAL (but short: 1 batch, N particles) OpenMC transport
    calculation with settings.track enabled, then builds the combined
    geometry+tracks overlay image. Runs off the GUI thread since this is
    an actual simulation, not an instant preview render.
    """
    # success, message, list of frame PNG paths (empty on failure), diagnostics
    # dict. The diagnostics carry the real bounding box of every recorded
    # track point so the page can tell the user when a run genuinely
    # succeeded but every track fell outside the requested Origin/Width --
    # which renders a perfectly correct image with nothing visible on it.
    finished_signal = Signal(bool, str, list, dict)
    # Emitted for every line OpenMC itself prints during the transport
    # run, as it's printed -- so the Live Output Console shows real
    # activity WHILE the run is in progress instead of staying silent
    # until everything finishes and the tracks suddenly appear. This
    # doesn't make the drawing itself live (OpenMC only writes tracks.h5
    # once the whole run completes -- there's no incremental track data
    # to draw mid-run), but it replaces a silent wait with visible proof
    # the run is actually progressing.
    log_signal = Signal(str)

    def __init__(self, script_code, n_particles, particle_filter,
                 basis, origin, width, pixels, color_by, export_root,
                 show_tracks=True, show_source=True, show_collisions=True,
                 source_marker_size=7):
        super().__init__()
        self.script_code = script_code
        self.n_particles = n_particles
        self.particle_filter = particle_filter  # 'all' or a ParticleType name
        self.basis = basis
        self.origin = origin
        self.width = width
        self.pixels = pixels
        self.color_by = color_by
        self.export_root = export_root
        self.show_tracks = show_tracks
        self.show_source = show_source
        self.show_collisions = show_collisions
        self.source_marker_size = source_marker_size

    def run(self):
        try:
            track_dir = os.path.join(self.export_root, "tracks")
            os.makedirs(track_dir, exist_ok=True)

            # Pre-flight: the whole page shells out to the `openmc`
            # executable twice (transport run, then the background plot).
            # If it isn't on PATH, Popen raises FileNotFoundError deep
            # inside the run and the user is left with an empty viewer and
            # a cryptic message. Check once, up front, and say exactly what
            # is missing and how to fix it.
            if shutil.which("openmc") is None:
                self.finished_signal.emit(
                    False,
                    "The 'openmc' executable was not found on your PATH.\n\n"
                    "The Particle Tracks page runs a real OpenMC transport calculation "
                    "as a separate process, so the openmc program itself must be "
                    "launchable from a terminal -- the Python 'openmc' package alone "
                    "is not enough.\n\n"
                    "To check, open a terminal and run:  openmc --version\n"
                    "If that fails, add the folder containing openmc.exe to your PATH "
                    "environment variable and restart OpenMC Studio.",
                    [], {})
                return

            geometry, materials_list, settings_obj, tallies_obj, err = \
                _discover_model_objects(self.script_code)
            if err:
                self.finished_signal.emit(False, err, [], {})
                return

            # Override settings for a FAST, track-focused run only -- this
            # writes exclusively into export/tracks/, never touching the
            # user's own export/ results from a real production run.
            settings_obj.track = [(1, 1, i) for i in range(1, self.n_particles + 1)]
            settings_obj.batches = 1
            if getattr(settings_obj, 'run_mode', None) == 'eigenvalue':
                settings_obj.inactive = 0
            # ALWAYS shrink to exactly n_particles -- do not keep the
            # original script's production-scale value (e.g. 1.7 million
            # for a real GEM50-83 run). This isn't just about speed: some
            # OpenMC builds (e.g. the unofficial "openmc-windows-beta"
            # used here) impose a hard cap -- "Batch or Particle limit
            # exceeded... Max particles = 10000" -- and a stray max()
            # against the original setting silently kept that huge value
            # and tripped it.
            settings_obj.particles = self.n_particles
            # Fix the plot color seed: without this, OpenMC assigns
            # RANDOM material colors on every run, which is why two runs
            # with identical Origin/Width/Basis can render completely
            # differently -- the geometry never changes, only which
            # random colors happened to land on which material.
            settings_obj.plot_seed = 1

            openmc.Materials(materials_list).export_to_xml(os.path.join(track_dir, "materials.xml"))
            geometry.export_to_xml(os.path.join(track_dir, "geometry.xml"))
            settings_obj.export_to_xml(os.path.join(track_dir, "settings.xml"))
            (tallies_obj if tallies_obj is not None else openmc.Tallies()).export_to_xml(
                os.path.join(track_dir, "tallies.xml"))

            # Direct subprocess call instead of the openmc.run() wrapper:
            # the wrapper raises RuntimeError with a message it extracts
            # from OpenMC's own output using patterns tuned for the
            # official build. On non-standard builds (e.g. this Windows
            # beta) that extraction can come up EMPTY, giving a useless
            # blank error. Capturing stdout/stderr ourselves guarantees
            # real diagnostic text is always available.
            # creationflags=CREATE_NO_WINDOW (Windows only -- never
            # evaluated elsewhere) stops a new console window from
            # popping up for the openmc.exe child process when this app
            # is a frozen --windowed exe with no console of its own for
            # that console-mode child to inherit.
            # Retry ONLY if the failure looks like a transient file-lock/
            # IO hiccup (see _looks_transient_io_error) rather than a real
            # configuration error -- retrying a genuine error (e.g.
            # missing cross-section data) would just waste time repeating
            # the same failure three times before reporting it.
            # Streamed via Popen (not subprocess.run's capture_output,
            # which only hands back output once the process has fully
            # exited) so each line OpenMC prints reaches the Live Output
            # Console the moment it's produced -- see log_signal above.
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            max_attempts = 3
            returncode = None
            combined_lines = []
            for attempt in range(1, max_attempts + 1):
                combined_lines = []
                if attempt > 1:
                    self.log_signal.emit(f"[OpenMC Studio] Retrying transport run (attempt {attempt}/{max_attempts})...")
                try:
                    process = subprocess.Popen(
                        ["openmc"], cwd=track_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, creationflags=creationflags
                    )
                except OSError as launch_err:
                    # The PATH pre-flight above catches the usual case, but
                    # the executable can still be unlaunchable (wrong
                    # architecture, missing DLL, blocked by policy). Report
                    # that as itself rather than as a mystery empty viewer.
                    self.finished_signal.emit(
                        False,
                        f"Could not start the 'openmc' process: {launch_err}\n\n"
                        "The program was found on your PATH but refused to launch. "
                        "Try running 'openmc --version' in a terminal to see the "
                        "underlying error.",
                        [], {})
                    return
                for line in process.stdout:
                    clean_line = line.rstrip()
                    combined_lines.append(clean_line)
                    self.log_signal.emit(clean_line)
                process.wait()
                returncode = process.returncode
                if returncode == 0:
                    break
                combined = "\n".join(combined_lines)
                if attempt < max_attempts and _looks_transient_io_error(combined):
                    time.sleep(0.6 * attempt)
                    continue
                break

            if returncode != 0:
                details = "\n".join(combined_lines).strip() or "(no output captured)"
                self.finished_signal.emit(
                    False, f"Transport run failed (exit code {returncode}):\n{details}", [], {})
                return

            tracks_h5 = os.path.join(track_dir, "tracks.h5")
            if not os.path.exists(tracks_h5):
                self.finished_signal.emit(
                    False,
                    "tracks.h5 was not generated -- the run may have failed silently. "
                    "Check that the model has valid cross-section data and a proper source.",
                    [], {})
                return

            # Background geometry plot -- same underlying mechanism as
            # the 2D Geometry Viewer, but with its OWN filename
            # ("tracks_bg", not "geometry_plot") so there is no chance of
            # colliding with that page's own file of the same name, even
            # if some quirk of this build's cwd handling ever pointed
            # both at the same location.
            plot = openmc.Plot()
            plot.filename = 'tracks_bg'
            plot.basis = self.basis
            plot.origin = self.origin
            plot.width = self.width
            plot.pixels = self.pixels
            plot.color_by = self.color_by
            openmc.Plots([plot]).export_to_xml(os.path.join(track_dir, "plots.xml"))

            geometry_png = os.path.join(track_dir, "tracks_bg.png")
            geometry_ppm = os.path.join(track_dir, "tracks_bg.ppm")

            # An OpenMC built WITHOUT libpng -- which includes the Windows
            # v0.16.0 build this app is commonly used with -- writes its
            # plot as a .ppm, not a .png, and still exits 0. Waiting for a
            # .png that is never going to appear made this page retry three
            # times and then report "failed after 3 attempts (exit code 0)"
            # on a plot run that had in fact succeeded. ui/plots/plots_page.py
            # already converts the .ppm with Pillow, which is exactly why
            # the 2D Geometry Viewer worked while this page did not; the
            # same conversion is done here.
            #
            # Retry on failure: a libpng "Write Error" can be a passing
            # hiccup (antivirus real-time scan, OneDrive/cloud sync,
            # Windows Search indexing briefly locking the file), OR --
            # if retrying alone doesn't help -- a corrupted/partially
            # written file LEFT BEHIND by the failed attempt itself,
            # which then makes every subsequent attempt fail the same
            # way. Deleting any existing file before each attempt
            # guarantees a genuinely clean slate every time either way.
            # Same creationflags fix as the transport run above.
            max_attempts = 3
            plot_result = None
            convert_error = None
            for attempt in range(1, max_attempts + 1):
                for stale in (geometry_png, geometry_ppm):
                    if os.path.exists(stale):
                        try:
                            os.remove(stale)
                        except OSError:
                            pass  # if this itself is locked, let the write attempt surface that
                plot_result = subprocess.run(
                    ["openmc", "--plot"], cwd=track_dir, capture_output=True, text=True,
                    creationflags=creationflags
                )
                if not os.path.exists(geometry_png) and os.path.exists(geometry_ppm):
                    try:
                        from PIL import Image
                        with Image.open(geometry_ppm) as im:
                            im.save(geometry_png, "PNG")
                        self.log_signal.emit(
                            "[OpenMC Studio] This OpenMC build wrote a .ppm plot "
                            "(no libpng); converted it to .png.")
                    except Exception as conv_err:
                        convert_error = conv_err
                if plot_result.returncode == 0 and os.path.exists(geometry_png):
                    break
                if attempt < max_attempts:
                    time.sleep(0.6 * attempt)

            if plot_result.returncode != 0 or not os.path.exists(geometry_png):
                details = (plot_result.stderr or plot_result.stdout or "(no output captured)").strip()
                if convert_error is not None:
                    details = (f"OpenMC wrote {os.path.basename(geometry_ppm)}, but converting it "
                               f"to PNG failed: {convert_error}\n"
                               f"Pillow is required to read this build's plot output "
                               f"(pip install Pillow).\n\n") + details
                elif os.path.exists(geometry_ppm):
                    details = ("OpenMC produced a .ppm plot that could not be converted "
                               "to PNG.\n\n") + details
                self.finished_signal.emit(
                    False,
                    f"Background plot generation failed after {max_attempts} attempts "
                    f"(exit code {plot_result.returncode}):\n{details}",
                    [], {})
                return

            # Prefer OpenMC's own reader -- it tracks any future change to
            # the file format -- but fall back to reading the HDF5 directly
            # when the installed package cannot decode what the binary
            # wrote (see _read_tracks_h5 for the PDG/IntEnum mismatch that
            # makes this necessary on some builds).
            try:
                tracks = openmc.Tracks(tracks_h5)
                if self.particle_filter != 'all':
                    tracks = tracks.filter(particle=self.particle_filter)
            except Exception as read_err:
                self.log_signal.emit(
                    f"[OpenMC Studio] openmc.Tracks could not read the file ({read_err}); "
                    "falling back to reading tracks.h5 directly.")
                try:
                    tracks = _read_tracks_h5(tracks_h5, self.particle_filter)
                except Exception as direct_err:
                    self.finished_signal.emit(
                        False,
                        f"tracks.h5 could not be read.\n\n"
                        f"OpenMC's own reader failed with: {read_err}\n"
                        f"Reading the file directly also failed with: {direct_err}\n\n"
                        "The file may be truncated or written in an unrecognised format.",
                        [], {})
                    return

            if len(tracks) == 0:
                if self.particle_filter != 'all':
                    message = (
                        f"No {self.particle_filter} tracks were recorded.\n\n"
                        "The run itself worked, but no particle of this type was "
                        "tracked. Set 'Show Particle Type' back to 'all' to see "
                        "what was actually produced.")
                else:
                    message = ("tracks.h5 was written but contains no particle "
                               "histories at all. Check that the source produces "
                               "particles inside the geometry.")
                self.finished_signal.emit(False, message, [], {})
                return

            # Diagnostic summary -- printed via the worker's own stdout,
            # which the app's Live Output Console does not capture, so
            # also hand it back in the success message where the GUI
            # will show it. This settles directly whether real, moving
            # track data exists (vs. every particle being absorbed at
            # its very first recorded point, or landing outside the
            # current Origin/Width view) rather than guessing from the
            # rendered image alone.
            n_primary = len(tracks)
            n_secondary_total = 0
            n_single_state = 0
            pt_mins = np.array([np.inf, np.inf, np.inf])
            pt_maxs = np.array([-np.inf, -np.inf, -np.inf])
            has_any_points = False
            for trk in tracks:
                for pt in trk.particle_tracks:
                    n_secondary_total += 1
                    pts_2d = _xyz_field_to_array(pt.states['r'])  # plain (N, 3) now
                    if pts_2d.shape[0] < 2:
                        n_single_state += 1
                    if pts_2d.size > 0:
                        has_any_points = True
                        pt_mins = np.minimum(pt_mins, pts_2d.min(axis=0))
                        pt_maxs = np.maximum(pt_maxs, pts_2d.max(axis=0))
            diag = (f"Diagnostics: {n_primary} primary histories, "
                    f"{n_secondary_total} particle_tracks total "
                    f"({n_single_state} with <2 recorded states -- drawn as points only).")
            if has_any_points:
                diag += (f" Position range: x[{pt_mins[0]:.2f},{pt_maxs[0]:.2f}] "
                         f"y[{pt_mins[1]:.2f},{pt_maxs[1]:.2f}] z[{pt_mins[2]:.2f},{pt_maxs[2]:.2f}] cm.")
            else:
                diag += " No position states recorded at all."

            # Generate a SEQUENCE of frames instead of one static final
            # image -- the widget plays these back with a QTimer for an
            # animated "watch the tracks grow" effect. True live updates
            # DURING the OpenMC run itself aren't possible here (would
            # need openmc.lib, the in-process C API -- confirmed broken on
            # this specific openmc-windows-beta build earlier in this
            # session); this replay happens right after the (now fast,
            # ~20-particle) run completes, which is the closest achievable
            # approximation.
            #
            # Frames are spaced by real elapsed simulation TIME
            # (states['time'], seconds), not array-index fraction: t_max
            # is the latest recorded time across every displayed particle,
            # and frame i reveals everything up through t_max * i/n_frames.
            # Since the QTimer below fires each frame at a fixed interval,
            # this makes the on-screen relative motion between frames
            # match real relative timing (e.g. a particle that free-streams
            # a long way between two collisions visibly moves further in
            # a frame than one that scatters immediately) instead of every
            # track simply reaching the same fraction of its own length at
            # the same time regardless of how long it actually took.
            # Falls back to the previous index-fraction method (reveal_time
            # left None) if no positive time data is available at all.
            t_max = 0.0
            for trk in tracks:
                for pt in trk.particle_tracks:
                    times = np.atleast_1d(pt.states['time']).astype(float)
                    if times.size:
                        t_max = max(t_max, float(times.max()))

            # n_frames halved from an earlier version (12->6), and only
            # the FINAL frame renders at full quality (dpi=150) -- the
            # other 5 are transient (on screen for a fraction of a
            # second each) and rendered at dpi=80 instead, roughly
            # halving the pixel count per frame. Together this cuts the
            # total matplotlib rendering workload by roughly 4-5x, which
            # is what was making the whole app feel sluggish during a
            # tracked run (Python's GIL means heavy computation on this
            # background thread still competes with the GUI thread for
            # CPU time, even though it's not literally blocking it).
            n_frames = 6
            frame_paths = []
            for frame_i in range(1, n_frames + 1):
                fraction = frame_i / n_frames
                is_last = (frame_i == n_frames)
                frame_path = os.path.join(track_dir, f"tracks_frame_{frame_i:02d}.png")
                _render_overlay(geometry_png, tracks, self.basis, self.origin,
                                 self.width, frame_path,
                                 show_tracks=self.show_tracks, show_source=self.show_source,
                                 show_collisions=self.show_collisions,
                                 source_marker_size=self.source_marker_size,
                                 reveal_fraction=fraction,
                                 reveal_time=(t_max * fraction) if t_max > 0 else None,
                                 dpi=150 if is_last else 80)
                frame_paths.append(frame_path)

            # Hand the raw numbers back too, so the page can tell apart
            # "nothing was tracked" from "everything was tracked but lies
            # outside the requested view" -- two situations that produce
            # the exact same blank-looking image.
            diagnostics = {
                'n_primary': n_primary,
                'n_secondary_total': n_secondary_total,
                'n_single_state': n_single_state,
                'has_any_points': bool(has_any_points),
                'mins': pt_mins.tolist() if has_any_points else None,
                'maxs': pt_maxs.tolist() if has_any_points else None,
            }
            self.finished_signal.emit(
                True, f"Rendered {len(tracks)} tracked source-particle histories. {diag}",
                frame_paths, diagnostics)

        except Exception as e:
            msg = str(e).strip() or f"{type(e).__name__} (no message attached to the exception)"
            self.finished_signal.emit(False, msg, [], {})


class _RunningNoticeDialog(QDialog):
    """A genuinely SEPARATE window (not embedded in the main panel) shown
    while a tracked simulation is running, warning the user not to close
    the application. Non-modal (.show(), not .exec()) so the rest of the
    app -- including this same tab -- stays usable/visible while it's up;
    it is closed programmatically the moment the worker thread finishes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracked Simulation Running")
        self.setModal(False)
        self.setFixedSize(400, 130)
        # Keep it on top and give it its own taskbar-visible window
        # rather than a tool/utility style -- easier to notice.
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        label = QLabel(
            "\u23f3  Tracked simulation is running\n\n"
            "A real OpenMC transport process is in progress.\n"
            "Please do NOT close OpenMC Studio until this\n"
            "window closes automatically."
        )
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        layout.addWidget(label)
        self.setStyleSheet("background-color: #f39c12;")


class TracksPageWidget(QWidget):
    script_generated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter()

        left_panel = QWidget()
        form = QFormLayout(left_panel)

        self.basis_combo = QComboBox()
        self.basis_combo.addItems(["xy", "xz", "yz"])
        form.addRow("Basis (Plane):", self.basis_combo)

        self.origin_field = QLineEdit("0.0, 0.0, 0.0")
        form.addRow("Origin (cm):", self.origin_field)

        self.width_field = QLineEdit("40.0, 40.0")
        form.addRow("Width (cm):", self.width_field)

        self.pixels_field = QLineEdit("1000, 1000")
        form.addRow("Resolution:", self.pixels_field)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["material", "cell"])
        form.addRow("Color By:", self.color_combo)

        self.n_particles_spin = QSpinBox()
        # Upper bound matches the hard cap this specific OpenMC build
        # enforces ("Batch or Particle limit exceeded... Max particles =
        # 10000" -- see TrackSimulationWorker.run for where that was hit).
        # Values in the low thousands are still a real transport run (not
        # an instant preview) and each additional particle/secondary is
        # its own matplotlib LineCollection in the overlay, so very high
        # counts will visibly slow down both the run and the render step.
        self.n_particles_spin.setRange(1, 10000)
        self.n_particles_spin.setValue(20)
        self.n_particles_spin.setToolTip(
            "Number of source particles to track (batch 1, generation 1,\n"
            "particles 1..N). Each may spawn secondaries (e.g. Compton\n"
            "electrons) which are tracked too.\n"
            "Keep this modest for quick iteration -- it's a real transport\n"
            "run, not an instant preview, and rendering gets slower with\n"
            "more tracked histories. 10000 is this OpenMC build's hard cap."
        )
        form.addRow("Particles to Track:", self.n_particles_spin)

        self.particle_filter_combo = QComboBox()
        self.particle_filter_combo.addItems(["all", "neutron", "photon", "electron", "positron"])
        form.addRow("Show Particle Type:", self.particle_filter_combo)

        # Independent display-layer toggles -- any combination, so e.g.
        # "source only", "collisions only", "tracks only", or all
        # together are all just different checkbox states, no separate
        # modes to pick between.
        self.chk_tracks = QCheckBox("Track lines")
        self.chk_tracks.setChecked(True)
        form.addRow("Show:", self.chk_tracks)

        self.chk_source = QCheckBox("Source birth points")
        self.chk_source.setChecked(True)
        form.addRow("", self.chk_source)

        self.chk_collisions = QCheckBox("Interactions/collisions")
        self.chk_collisions.setChecked(True)
        form.addRow("", self.chk_collisions)

        self.source_size_spin = QSpinBox()
        self.source_size_spin.setRange(2, 30)
        self.source_size_spin.setValue(7)
        self.source_size_spin.setToolTip("Marker size (points) for the source-birth circles.")
        form.addRow("Source Marker Size:", self.source_size_spin)

        self.btn_run = QPushButton("\U0001F3AF Run Tracked Simulation")
        self.btn_run.setStyleSheet(
            "background-color: #8e44ad; color: white; font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self.run_tracked_simulation)
        form.addRow(self.btn_run)

        # Indeterminate (busy) bar shown only while the transport run and
        # background plot are in progress -- OpenMC only writes tracks.h5
        # once the whole run finishes, so there's no real percentage to
        # report for a single-batch tracked run; this is visual proof
        # something is happening rather than a silent wait, alongside the
        # live-streamed lines now going to the Live Output Console (see
        # TrackSimulationWorker.log_signal).
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        form.addRow(self.progress_bar)

        self.btn_append = QPushButton("\U0001F4BE Append Track Settings to Script")
        self.btn_append.setStyleSheet(
            "background-color: #4a4a4a; color: white; padding: 5px;")
        self.btn_append.clicked.connect(self.append_track_code)
        form.addRow(self.btn_append)


        self.viewer = ZoomableImageViewer()
        self.viewer.setMinimumSize(400, 400)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.viewer)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter)

    def _log(self, msg):
        try:
            for widget in QApplication.topLevelWidgets():
                if hasattr(widget, 'console_widget'):
                    widget.console_widget.append_log(msg)
                    return
        except Exception:
            pass
        print(msg)

    def _fail(self, title, message):
        """Report a failure BOTH in the console log and in a dialog.

        Every failure path on this page used to end at a single _log()
        line. If the Live Output Console happened to be collapsed or the
        user was looking at the viewer, the entire visible outcome of a
        failed run was: the button re-enables, the orange notice window
        disappears, and the image area stays empty -- indistinguishable
        from the app doing nothing at all. A tracked run takes real time
        and real CPU, so it must never fail invisibly.
        """
        self._log(f"⚠️ {title}: {message}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        # Long OpenMC output goes in the detail pane so the dialog stays a
        # readable size, and stays selectable so it can be copied out.
        if len(message) > 300 or message.count("\n") > 6:
            head, _, tail = message.partition("\n")
            box.setText(head.strip() or "The tracked simulation could not be completed.")
            box.setDetailedText(tail.strip() or message)
        else:
            box.setText(message)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    def _parse_fields(self):
        o_str = self.origin_field.text().strip()
        w_str = self.width_field.text().strip()
        p_str = self.pixels_field.text().strip()
        try:
            origin = tuple(map(float, o_str.split(',')))
            width = tuple(map(float, w_str.split(',')))
            pixels = tuple(map(int, p_str.split(',')))
        except ValueError:
            return None, None, None, "Origin, Width, or Resolution format is incorrect."
        if len(origin) != 3:
            return None, None, None, f"Origin must have 3 values, got {len(origin)}."
        if len(width) != 2:
            return None, None, None, f"Width must have 2 values, got {len(width)}."
        if len(pixels) != 2:
            return None, None, None, f"Resolution must have 2 values, got {len(pixels)}."
        return origin, width, pixels, None

    def run_tracked_simulation(self):
        main_win = self.window()
        if not hasattr(main_win, 'script_editor'):
            self._fail("Particle Tracks", "No script editor was found in the main window.")
            return
        code = main_win.script_editor.editor.toPlainText()
        if not code.strip():
            self._fail(
                "Script Is Empty",
                "There is no model to track.\n\n"
                "Build a model first (Materials, Geometry, Settings) or open an "
                "existing script, then run the tracked simulation.")
            return

        origin, width, pixels, err = self._parse_fields()
        if err:
            self._fail("Invalid Plot Settings", err)
            return

        export_root = os.path.abspath(os.path.join(os.getcwd(), "export"))
        os.makedirs(export_root, exist_ok=True)

        self.btn_run.setEnabled(False)
        self.btn_run.setText("\u23f3 Running...")
        self.progress_bar.setVisible(True)
        # Separate, standalone window (not part of this panel) -- see
        # _RunningNoticeDialog. Kept as an attribute so it isn't
        # garbage-collected while shown, and so _on_tracked_run_finished
        # can close it.
        self._running_dialog = _RunningNoticeDialog(self)
        self._running_dialog.show()
        self._log("\U0001F3AF Starting tracked simulation...")

        self._worker = TrackSimulationWorker(
            code, self.n_particles_spin.value(), self.particle_filter_combo.currentText(),
            self.basis_combo.currentText(), origin, width, pixels,
            self.color_combo.currentText(), export_root,
            show_tracks=self.chk_tracks.isChecked(),
            show_source=self.chk_source.isChecked(),
            show_collisions=self.chk_collisions.isChecked(),
            source_marker_size=self.source_size_spin.value(),
        )
        self._worker.log_signal.connect(self._log)
        self._worker.finished_signal.connect(self._on_tracked_run_finished)
        self._worker.start()

    def _on_tracked_run_finished(self, success, message, frame_paths, diagnostics=None):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("\U0001F3AF Run Tracked Simulation")
        self.progress_bar.setVisible(False)
        if hasattr(self, '_running_dialog') and self._running_dialog is not None:
            self._running_dialog.close()
            self._running_dialog = None

        if not success:
            self._fail("Tracked Simulation Failed", message)
            return

        self._log(f"\u2705 {message}")
        if not frame_paths:
            # Previously a bare `return`: the run reported success but the
            # viewer stayed empty with nothing written anywhere at all.
            self._fail(
                "No Image Was Produced",
                "The run reported success but produced no image frames. "
                "Check the export/tracks folder for a partially written file, "
                "then run again.")
            return

        # Animated "replay": step through the progressively-revealed
        # frames on a timer rather than jumping straight to the final
        # image -- this is the closest achievable approximation of
        # watching tracks/interactions appear in real time (true live
        # updates DURING the OpenMC run itself would need openmc.lib,
        # confirmed broken on this build earlier in this session).
        self._anim_frames = frame_paths
        self._anim_index = 0
        if not hasattr(self, '_anim_timer'):
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._advance_animation_frame)
        self._anim_timer.start(270)  # ms per frame (6 frames now, was 12 at 180ms)
        self._warn_if_nothing_visible(diagnostics or {})

    def _warn_if_nothing_visible(self, diagnostics):
        """Explain a technically-correct image that still looks empty.

        A run can succeed completely and still show nothing, for reasons
        the image itself cannot convey:
          * no position states were recorded at all;
          * every particle was absorbed at birth (one state each, so there
            is no line to draw -- only a dot);
          * the tracks are real and were drawn, but lie entirely outside
            the requested Origin/Width, so they fall off the edge of the
            canvas.
        The third is the easy one to mistake for a broken feature, and it
        is also the one we can fix for the user: the bounding box of the
        recorded points gives the exact Origin/Width that would frame
        them, so offer those numbers instead of just reporting a problem.
        """
        if not diagnostics:
            return

        if not diagnostics.get('has_any_points'):
            self._fail(
                "No Track Data Recorded",
                "The run finished, but OpenMC recorded no particle positions at all.\n\n"
                "This usually means the source produced no particles inside the "
                "geometry. Check that settings.source is defined and that its "
                "position lies inside a cell of the model.")
            return

        mins = np.asarray(diagnostics.get('mins'), dtype=float)
        maxs = np.asarray(diagnostics.get('maxs'), dtype=float)

        basis = self.basis_combo.currentText()
        h_idx, v_idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[basis]
        origin, width, _pixels, err = self._parse_fields()
        if err:
            return

        h_lo, h_hi = origin[h_idx] - width[0] / 2.0, origin[h_idx] + width[0] / 2.0
        v_lo, v_hi = origin[v_idx] - width[1] / 2.0, origin[v_idx] + width[1] / 2.0

        # Do the track extents overlap the visible window at all? Compared
        # as intervals rather than testing individual points, so a track
        # that merely passes through the view still counts as visible.
        overlaps = (mins[h_idx] <= h_hi and maxs[h_idx] >= h_lo and
                    mins[v_idx] <= v_hi and maxs[v_idx] >= v_lo)
        if overlaps:
            n_single = diagnostics.get('n_single_state', 0)
            n_total = diagnostics.get('n_secondary_total', 0)
            if n_total and n_single == n_total:
                self._fail(
                    "No Track Lines to Draw",
                    "Every tracked particle has only one recorded state, so there "
                    "are no lines to draw -- only the birth points.\n\n"
                    "This means each particle was absorbed at the moment it was "
                    "born. It is a real physical result, not a display error: "
                    "check the source energy and the material at the source "
                    "position.")
            return

        # Centre the view on the tracks and pad the span by 10% so the
        # extremes do not sit exactly on the border.
        new_origin = list(origin)
        for idx in (h_idx, v_idx):
            new_origin[idx] = (mins[idx] + maxs[idx]) / 2.0
        span_h = max(maxs[h_idx] - mins[h_idx], 1.0) * 1.1
        span_v = max(maxs[v_idx] - mins[v_idx], 1.0) * 1.1

        suggested_origin = ", ".join(f"{v:.2f}" for v in new_origin)
        suggested_width = f"{span_h:.2f}, {span_v:.2f}"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Tracks Are Outside the Current View")
        box.setText(
            "The simulation worked and the tracks were drawn, but all of them "
            "lie outside the region you asked to see, so the image looks empty.")
        box.setInformativeText(
            f"Tracks occupy:\n"
            f"    x [{mins[0]:.2f}, {maxs[0]:.2f}]   "
            f"y [{mins[1]:.2f}, {maxs[1]:.2f}]   "
            f"z [{mins[2]:.2f}, {maxs[2]:.2f}]  cm\n\n"
            f"Current view ({basis} plane):\n"
            f"    horizontal [{h_lo:.2f}, {h_hi:.2f}]   "
            f"vertical [{v_lo:.2f}, {v_hi:.2f}]  cm\n\n"
            f"Suggested settings:\n"
            f"    Origin: {suggested_origin}\n"
            f"    Width:  {suggested_width}")
        apply_btn = box.addButton("Use These Settings", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Keep Mine", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        if box.clickedButton() is apply_btn:
            self.origin_field.setText(suggested_origin)
            self.width_field.setText(suggested_width)
            self._log(f"🎯 View reframed to Origin ({suggested_origin}), Width ({suggested_width}). "
                      "Run the tracked simulation again to see the tracks.")

    def _advance_animation_frame(self):
        if self._anim_index >= len(self._anim_frames):
            self._anim_timer.stop()
            return
        self.viewer.set_image(QPixmap(self._anim_frames[self._anim_index]))
        self._anim_index += 1

    def append_track_code(self):
        n = self.n_particles_spin.value()
        py_code = (
            f"\n# ==========================================\n"
            f"# --- Particle Track Settings (auto-appended) ---\n"
            f"# ==========================================\n"
            f"settings.track = [(1, 1, i) for i in range(1, {n} + 1)]\n"
            f"# NOTE: for a track-focused run, also consider: settings.batches = 1\n"
        )
        self.script_generated.emit(py_code)
        self._log("\u2705 Track settings code appended to script editor.")