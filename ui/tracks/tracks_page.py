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
    QDialog, QProgressBar, QMessageBox, QScrollArea, QSlider, QFileDialog
)
from PySide6.QtCore import Signal, QThread, QTimer, Qt
from PySide6.QtGui import QPixmap

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba, LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import matplotlib.patheffects as pe

from ui.plots.plots_page import ZoomableImageViewer
from ui.tracks.tracks_help import TracksHelpDialog


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


def _contiguous_runs(mask):
    """Yield (start, stop) index pairs for each maximal run of True.

    Used by the slice filter: a track that leaves the slab and comes back
    must be drawn as two separate polylines. Joining them would draw a
    straight line across the gap that the particle never travelled.
    """
    if not mask.any():
        return
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = np.concatenate(([0], edges + 1))
    stops = np.concatenate((edges + 1, [mask.size]))
    for start, stop in zip(starts, stops):
        if mask[start]:
            yield int(start), int(stop)


def _collision_indices(states, n_states):
    """Indices of states that are real collisions rather than surface
    crossings: a new state whose cell_id AND material_id both match the
    previous state's did not cross into a new cell, and OpenMC only writes
    a state on a real event."""
    cell_ids = np.atleast_1d(states['cell_id'])[:n_states]
    mat_ids = np.atleast_1d(states['material_id'])[:n_states]
    if cell_ids.shape[0] <= 2:
        return np.empty(0, dtype=int)
    same_cell = (cell_ids[1:] == cell_ids[:-1]) & (mat_ids[1:] == mat_ids[:-1])
    return np.flatnonzero(same_cell) + 1


def _classify_fate(last_point, model_bbox):
    """'escaped' if a particle's final position sits on (or beyond) the
    model's outer boundary, 'absorbed' if it stopped inside, None when the
    model has no finite bounding box to compare against.

    The track file records positions, not why a history ended, so the
    boundary is what separates "left the problem" from "stopped here" --
    and it is the distinction shielding work actually needs.
    """
    if model_bbox is None:
        return None
    lower, upper = np.asarray(model_bbox[0], dtype=float), np.asarray(model_bbox[1], dtype=float)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    span = float(np.max(upper - lower))
    if not np.isfinite(span) or span <= 0:
        return None
    tol = 1e-3 * span
    beyond = np.any(last_point < lower - tol) or np.any(last_point > upper + tol)
    on_face = np.any((np.abs(last_point - lower) <= tol) |
                     (np.abs(last_point - upper) <= tol))
    return 'escaped' if (beyond or on_face) else 'absorbed'


def _render_overlay(geometry_png_path, tracks, basis, origin, width, out_png_path,
                     show_tracks=True, show_source=True, show_collisions=True,
                     source_marker_size=7, reveal_fraction=1.0, reveal_time=None, dpi=150,
                     slab_thickness=None, color_mode='particle',
                     collision_mode='points', collision_marker_size=5,
                     show_primaries=True, show_secondaries=True, heatmap_bins=60,
                     energy_min=None, energy_max=None,
                     show_fate=False, model_bbox=None):
    """Composite the background geometry PNG with real particle-track
    polylines, source-birth markers, and interaction points, all aligned in
    PHYSICAL (cm) coordinates via matplotlib's `extent` -- this sidesteps
    needing to hand-compute pixel offsets, and automatically stays correct
    for any Basis/Origin/Width combination.

    The show_* flags are independent toggles (any combination), matching
    how the UI checkboxes work.

    `slab_thickness` (cm) restricts drawing to states within +/- half that
    distance of the plane along the axis the basis does not cover. Without
    it the whole depth of a 3D model is flattened onto one plane, so tracks
    centimetres away from the background slice are drawn on top of it as
    though they were in it -- the picture becomes denser than the physics
    and stops matching the geometry underneath. None keeps the old
    project-everything behaviour.

    `color_mode`:
      'particle' -- one colour per particle type (PARTICLE_COLORS), with
                    per-segment alpha following the particle's own energy
                    relative to its starting energy.
      'energy'   -- segments coloured by absolute energy on a shared
                    logarithmic scale with a colour bar, so energies are
                    comparable between tracks rather than only within one.

    `collision_mode`:
      'points'  -- one marker per interaction.
      'heatmap' -- a 2D histogram of interaction sites. With thousands of
                   collisions the individual markers merge into noise;
                   binned, the same data shows where interaction actually
                   concentrates.

    `show_primaries` / `show_secondaries` split source particles from the
    secondaries they create (Compton electrons, fluorescence photons).
    Secondaries usually outnumber primaries several times over, so being
    able to drop them is often what makes a plot readable at all.

    `energy_min` / `energy_max` (eV) keep only the states recorded inside
    that window. This is applied per STATE, not per track, so a photon that
    starts at 1 MeV and degrades to 50 keV appears only along the stretch
    where it was still above the threshold -- which is what separates the
    penetrating radiation that matters for shielding from the low-energy
    scatter filling the same space.

    `show_fate` marks where each particle history ended, split by whether
    it escaped the model or stopped inside it, using `model_bbox`
    (lower_left, upper_right from openmc.Geometry.bounding_box).

    Returns a dict of counts describing what was actually drawn.

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
    be rendered cheaper than the final "kept" frame.
    """
    img = mpimg.imread(geometry_png_path)

    h_idx, v_idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[basis]
    # The axis the basis does NOT cover -- the slice normal. Indices are a
    # permutation of {0,1,2}, so the missing one is whatever is left over.
    n_idx = 3 - h_idx - v_idx
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

    # \u2500\u2500 Pass 1: collect what will be drawn \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # Gathering first means the energy colour scale and the interaction
    # heatmap can both be built from the WHOLE displayed set. Normalising
    # per-track instead would make two tracks with the same colour mean
    # different energies, which is exactly what the old alpha fade did.
    polylines = []       # (h, v, seg_energy, pname, is_primary)
    single_points = []   # (h, v, pname)
    source_points = []   # (h, v, pname)
    collision_points = []  # (h, v, pname)
    fate_points = {'escaped': [], 'absorbed': []}
    fate_counts = {'escaped': 0, 'absorbed': 0, 'unknown': 0}

    for track in tracks:
        for i, ptrack in enumerate(track.particle_tracks):
            is_primary = (i == 0)
            if is_primary and not show_primaries:
                continue
            if not is_primary and not show_secondaries:
                continue

            r = _xyz_field_to_array(ptrack.states['r'])  # plain (N, 3) float array
            # For animated playback: only reveal the states recorded up to
            # a cutoff. Prefer real elapsed TIME over array-index fraction
            # whenever available -- see the docstring. Always keep at least
            # one state so the particle can still register.
            if r.shape[0] > 1:
                if reveal_time is not None:
                    times = np.atleast_1d(ptrack.states['time']).astype(float)
                    n_keep = max(1, int(np.searchsorted(times, reveal_time, side='right')))
                    r = r[:n_keep]
                elif reveal_fraction < 1.0:
                    n_keep = max(1, int(np.ceil(r.shape[0] * reveal_fraction)))
                    r = r[:n_keep]

            pname = ptrack.particle.name.lower()
            energies = np.atleast_1d(ptrack.states['E']).astype(float)[:r.shape[0]]

            # Slice filter: keep only the states lying within the slab.
            if slab_thickness is not None and slab_thickness > 0:
                in_slab = np.abs(r[:, n_idx] - origin[n_idx]) <= (slab_thickness / 2.0)
            else:
                in_slab = np.ones(r.shape[0], dtype=bool)

            # Energy window, applied per state so a track appears only
            # along the stretch where it was still inside the window.
            visible = in_slab
            if energy_min is not None:
                visible = visible & (energies >= energy_min)
            if energy_max is not None:
                visible = visible & (energies <= energy_max)
            if not visible.any():
                continue

            h_all, v_all = r[:, h_idx], r[:, v_idx]

            # Where this history ended. Classified from the last recorded
            # state regardless of the energy window (the fate is a property
            # of the history, not of the slice being viewed), but only
            # drawn when that point is inside the slab being shown.
            if show_fate and r.shape[0] >= 1:
                fate = _classify_fate(r[-1], model_bbox)
                fate_counts[fate if fate else 'unknown'] += 1
                if fate and in_slab[-1]:
                    fate_points[fate].append((h_all[-1], v_all[-1], pname))

            if show_source and is_primary and visible[0]:
                source_points.append((h_all[0], v_all[0], pname))

            if show_collisions:
                hit_idx = _collision_indices(ptrack.states, r.shape[0])
                hit_idx = hit_idx[visible[hit_idx]]
                for k in hit_idx:
                    collision_points.append((h_all[k], v_all[k], pname))

            if not show_tracks:
                continue

            # Each run of consecutive in-slab states is its own polyline;
            # a run of one state has no segment to draw, so it becomes a
            # point instead of being dropped silently.
            for start, stop in _contiguous_runs(visible):
                if stop - start < 2:
                    single_points.append((h_all[start], v_all[start], pname))
                    continue
                seg_e = (energies[start:stop - 1] + energies[start + 1:stop]) / 2.0
                polylines.append((h_all[start:stop], v_all[start:stop],
                                  seg_e, pname, is_primary))

    # \u2500\u2500 Energy colour scale, shared across every drawn segment \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    energy_norm = None
    if color_mode == 'energy' and polylines:
        all_e = np.concatenate([p[2] for p in polylines])
        positive = all_e[all_e > 0]
        if positive.size:
            e_lo, e_hi = float(positive.min()), float(positive.max())
            # A colour bar needs a real range; a monoenergetic run would
            # otherwise collapse to a single value LogNorm cannot map.
            if e_hi <= e_lo * 1.001:
                e_hi = e_lo * 10.0
            energy_norm = LogNorm(vmin=e_lo, vmax=e_hi)
    cmap = matplotlib.colormaps['viridis']

    # \u2500\u2500 Pass 2: draw \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    seen_particles = set()
    for h, v, seg_e, pname, is_primary in polylines:
        seen_particles.add(pname)
        points = np.stack([h, v], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Secondaries are drawn thinner and dashed so they stay visually
        # subordinate to the source particles even when both are shown.
        linewidth = 1.6 if is_primary else 1.0
        linestyle = 'solid' if is_primary else (0, (3, 2))

        if energy_norm is not None:
            lc = LineCollection(segments, cmap=cmap, norm=energy_norm,
                                linewidths=linewidth, linestyles=linestyle, zorder=3)
            lc.set_array(np.clip(seg_e, energy_norm.vmin, energy_norm.vmax))
        else:
            color = PARTICLE_COLORS.get(pname, '#888888')
            # Per-segment alpha follows this particle's OWN energy relative
            # to its start: full energy opaque, fully spent faint.
            e0 = seg_e[0] if seg_e.size and seg_e[0] > 0 else 0.0
            if e0 > 0:
                frac = np.clip(seg_e / e0, 0.0, 1.0)
                seg_alpha = 0.35 + 0.65 * frac
            else:
                seg_alpha = np.full(len(segments), 0.95)
            seg_colors = np.tile(np.array(to_rgba(color)), (len(segments), 1))
            seg_colors[:, 3] = seg_alpha
            lc = LineCollection(segments, colors=seg_colors, linewidths=linewidth,
                                linestyles=linestyle, zorder=3,
                                path_effects=[pe.Stroke(linewidth=linewidth + 1.0,
                                                        foreground='white', alpha=0.6),
                                              pe.Normal()])
        ax.add_collection(lc)

    # A particle with a single usable state has no line to draw. Marked
    # with an x rather than dropped -- only markers and no lines is itself
    # diagnostic: particles are being absorbed at or near birth.
    for h, v, pname in single_points:
        seen_particles.add(pname)
        ax.plot(h, v, marker='x', color=PARTICLE_COLORS.get(pname, '#888888'),
                markersize=8, markeredgewidth=2.0, alpha=0.95, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    # Source-birth markers: a small filled circle, not a star -- with many
    # source points clustered together a star's spikes overlap into a blob.
    for h, v, pname in source_points:
        seen_particles.add(pname)
        ax.plot(h, v, marker='o', color=PARTICLE_COLORS.get(pname, '#888888'),
                markersize=source_marker_size, markeredgecolor='black',
                markeredgewidth=0.7, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])

    # End-of-history markers. Escaped is drawn hollow (the particle left --
    # nothing remains there) and absorbed solid (the energy stopped here),
    # in fixed colours rather than per particle type, because the question
    # they answer is about fate, not species.
    for h, v, _pname in fate_points['absorbed']:
        ax.plot(h, v, marker='s', color='#b91c1c', markersize=5,
                markeredgecolor='white', markeredgewidth=0.6, zorder=7)
    for h, v, _pname in fate_points['escaped']:
        ax.plot(h, v, marker='^', markerfacecolor='none',
                markeredgecolor='#0ea5e9', markersize=6, markeredgewidth=1.4,
                zorder=7)

    collision_mesh = None
    if collision_points:
        ch = np.array([c[0] for c in collision_points])
        cv = np.array([c[1] for c in collision_points])
        if collision_mode == 'heatmap':
            counts, h_edges, v_edges = np.histogram2d(
                ch, cv, bins=heatmap_bins,
                range=[[extent[0], extent[1]], [extent[2], extent[3]]])
            # Empty bins stay fully transparent so the geometry shows
            # through everywhere nothing actually happened.
            masked = np.ma.masked_less_equal(counts.T, 0)
            collision_mesh = ax.pcolormesh(
                h_edges, v_edges, masked, cmap='inferno', alpha=0.75,
                zorder=4, shading='auto')
        else:
            for pname in sorted({c[2] for c in collision_points}):
                sel = np.array([c[2] == pname for c in collision_points])
                ax.plot(ch[sel], cv[sel], '.',
                        color=PARTICLE_COLORS.get(pname, '#888888'),
                        markersize=collision_marker_size, alpha=0.95, zorder=5,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # \u2500\u2500 Legend / colour bars \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    handles = []
    if energy_norm is not None:
        bar = fig.colorbar(ScalarMappable(norm=energy_norm, cmap=cmap),
                           ax=ax, fraction=0.046, pad=0.03)
        bar.set_label('Particle energy [eV]', fontsize=9)
    elif seen_particles:
        handles = [Line2D([], [], color=PARTICLE_COLORS.get(p, '#888888'),
                          linewidth=2.0, label=p.capitalize())
                   for p in sorted(seen_particles)]
        if show_primaries and show_secondaries and any(not p[4] for p in polylines):
            handles.append(Line2D([], [], color='#444444', linewidth=1.0,
                                  linestyle=(0, (3, 2)), label='Secondary'))

    # Fate keys are worth showing even in energy mode, where the colour bar
    # replaces the particle legend but says nothing about how tracks ended.
    if fate_points['absorbed']:
        handles.append(Line2D([], [], linestyle='none', marker='s', color='#b91c1c',
                              markersize=5, label=f"Absorbed ({fate_counts['absorbed']})"))
    if fate_points['escaped']:
        handles.append(Line2D([], [], linestyle='none', marker='^',
                              markerfacecolor='none', markeredgecolor='#0ea5e9',
                              markersize=6, label=f"Escaped ({fate_counts['escaped']})"))
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.85)

    if collision_mesh is not None:
        bar2 = fig.colorbar(collision_mesh, ax=ax, fraction=0.046, pad=0.03,
                            location='left')
        bar2.set_label('Interactions per bin', fontsize=9)

    axis_labels = {'xy': ('X [cm]', 'Y [cm]'), 'xz': ('X [cm]', 'Z [cm]'), 'yz': ('Y [cm]', 'Z [cm]')}
    xl, yl = axis_labels[basis]
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

    title_bits = []
    if show_source:
        title_bits.append("\u25cf source birth")
    if show_collisions:
        title_bits.append("interaction heatmap" if collision_mode == 'heatmap'
                          else "\u00b7 interaction")
    if not show_secondaries:
        title_bits.append("primaries only")
    elif not show_primaries:
        title_bits.append("secondaries only")
    if energy_min is not None or energy_max is not None:
        lo = f"{energy_min:.3g}" if energy_min is not None else "0"
        hi = f"{energy_max:.3g}" if energy_max is not None else "\u221e"
        title_bits.append(f"E in [{lo}, {hi}] eV")
    suffix = (" -- " + ", ".join(title_bits)) if title_bits else ""

    normal_axis = 'XYZ'[n_idx]
    if slab_thickness is not None and slab_thickness > 0:
        depth = (f"{normal_axis} = {origin[n_idx]:.2f} \u00b1 {slab_thickness / 2.0:.2f} cm")
    else:
        depth = f"all {normal_axis} projected"
    ax.set_title(f"Particle Tracks ({basis} slice, {depth}){suffix}")

    fig.tight_layout()
    canvas.print_png(out_png_path)
    fig.clear()  # drop all axes/artists now rather than waiting for GC,
                 # since this function may be called many times in a row
                 # (once per animation frame) within a single run

    return {
        'polylines': len(polylines),
        'source_points': len(source_points),
        'collisions': len(collision_points),
        'escaped': fate_counts['escaped'],
        'absorbed': fate_counts['absorbed'],
        'fate_unknown': fate_counts['unknown'],
    }


def _generate_background_plot(track_dir, basis, origin, width, pixels, color_by,
                              filename='tracks_bg', log=None):
    """Render the geometry slice OpenMC draws under the tracks.

    Returns (png_path, error_text); error_text is None on success.

    Extracted from the simulation worker because the slice-position slider
    needs it too: moving the tracks' slab without moving the background
    would put the tracks over a picture of a different plane -- exactly the
    mismatch the slab filter exists to remove.

    An OpenMC built WITHOUT libpng -- including the Windows v0.16.0 build
    this app is commonly used with -- writes its plot as a .ppm, not a
    .png, and still exits 0. Waiting for a .png that is never going to
    appear reported "failed after 3 attempts (exit code 0)" on a plot run
    that had in fact succeeded, so the .ppm is converted with Pillow, as
    ui/plots/plots_page.py already does.

    Retry on failure: a libpng "Write Error" can be a passing hiccup
    (antivirus real-time scan, cloud sync, Windows Search indexing briefly
    locking the file), OR -- if retrying alone doesn't help -- a corrupted
    file LEFT BEHIND by the failed attempt, which then makes every later
    attempt fail the same way. Deleting before each attempt guarantees a
    clean slate either way.
    """
    plot = openmc.Plot()
    plot.filename = filename
    plot.basis = basis
    plot.origin = tuple(origin)
    plot.width = tuple(width)
    plot.pixels = tuple(pixels)
    plot.color_by = color_by
    openmc.Plots([plot]).export_to_xml(os.path.join(track_dir, "plots.xml"))

    png_path = os.path.join(track_dir, f"{filename}.png")
    ppm_path = os.path.join(track_dir, f"{filename}.ppm")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

    max_attempts = 3
    plot_result = None
    convert_error = None
    for attempt in range(1, max_attempts + 1):
        for stale in (png_path, ppm_path):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass  # if this itself is locked, let the write attempt surface that
        plot_result = subprocess.run(
            ["openmc", "--plot"], cwd=track_dir, capture_output=True, text=True,
            creationflags=creationflags)
        if not os.path.exists(png_path) and os.path.exists(ppm_path):
            try:
                from PIL import Image
                with Image.open(ppm_path) as im:
                    im.save(png_path, "PNG")
                if log is not None:
                    log("[OpenMC Studio] This OpenMC build wrote a .ppm plot "
                        "(no libpng); converted it to .png.")
            except Exception as conv_err:
                convert_error = conv_err
        if plot_result.returncode == 0 and os.path.exists(png_path):
            return png_path, None
        if attempt < max_attempts:
            time.sleep(0.6 * attempt)

    details = (plot_result.stderr or plot_result.stdout or "(no output captured)").strip()
    if convert_error is not None:
        details = (f"OpenMC wrote {os.path.basename(ppm_path)}, but converting it "
                   f"to PNG failed: {convert_error}\n"
                   f"Pillow is required to read this build's plot output "
                   f"(pip install Pillow).\n\n") + details
    elif os.path.exists(ppm_path):
        details = ("OpenMC produced a .ppm plot that could not be converted "
                   "to PNG.\n\n") + details
    return None, (f"Background plot generation failed after {max_attempts} attempts "
                  f"(exit code {plot_result.returncode}):\n{details}")


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
                 display=None, **legacy_display):
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
        # Every drawing choice travels as one dict so it can be handed
        # straight to _render_overlay and, more importantly, reused later
        # by the redraw path without the two falling out of step.
        # legacy_display keeps the old show_tracks=... keyword form working.
        self.display = dict(display or {})
        self.display.update(legacy_display)

        # Kept so the page can redraw from memory instead of re-running a
        # whole transport calculation just to toggle a checkbox.
        self.tracks = None
        self.geometry_png = None
        self.t_max = 0.0
        self.model_bbox = None

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

            # The model's outer boundary, used to tell a particle that left
            # the problem from one that stopped inside it. Unbounded models
            # give infinities, in which case fate stays unclassified rather
            # than guessed.
            try:
                bbox = geometry.bounding_box
                self.model_bbox = (np.asarray(bbox[0], dtype=float),
                                   np.asarray(bbox[1], dtype=float))
            except Exception:
                self.model_bbox = None

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
            geometry_png, plot_err = _generate_background_plot(
                track_dir, self.basis, self.origin, self.width, self.pixels,
                self.color_by, log=self.log_signal.emit)
            if plot_err:
                self.finished_signal.emit(False, plot_err, [], {})
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
                                 reveal_fraction=fraction,
                                 reveal_time=(t_max * fraction) if t_max > 0 else None,
                                 dpi=150 if is_last else 80,
                                 model_bbox=self.model_bbox,
                                 **self.display)
                frame_paths.append(frame_path)

            # Retained so a display toggle can redraw straight from these
            # instead of paying for another transport run.
            self.tracks = tracks
            self.geometry_png = geometry_png
            self.t_max = t_max

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


class TrackRedrawWorker(QThread):
    """Re-renders the overlay from track data already in memory.

    Changing how tracks are drawn used to mean re-running the whole
    transport calculation, because rendering only ever happened inside
    TrackSimulationWorker. Hiding the interaction dots therefore cost a
    full OpenMC run -- for a result that was already sitting in memory and
    would come back identical. The track data does not depend on any of
    the display settings, so redrawing needs nothing but matplotlib.

    Still a thread: several thousand polylines take long enough to render
    that doing it on the GUI thread would visibly freeze the window.
    """
    finished_signal = Signal(bool, str, str)  # success, message, frame path

    def __init__(self, tracks, geometry_png, basis, origin, width,
                 out_path, display, model_bbox=None, replot=None):
        super().__init__()
        self.tracks = tracks
        self.geometry_png = geometry_png
        self.basis = basis
        self.origin = origin
        self.width = width
        self.out_path = out_path
        self.display = dict(display)
        self.model_bbox = model_bbox
        # When the slice moves along the normal axis the background has to
        # move with it. Redrawing tracks at a new depth over a picture of
        # the original plane would reintroduce exactly the mismatch the
        # slab filter exists to remove, so a re-plot is requested instead.
        self.replot = replot
        self.stats = {}

    def run(self):
        try:
            geometry_png = self.geometry_png
            if self.replot:
                new_png, err = _generate_background_plot(
                    self.replot['track_dir'], self.basis, self.origin,
                    self.width, self.replot['pixels'], self.replot['color_by'],
                    filename='tracks_bg_slice')
                if err:
                    self.finished_signal.emit(False, err, "")
                    return
                geometry_png = new_png
            self.stats = _render_overlay(
                geometry_png, self.tracks, self.basis, self.origin,
                self.width, self.out_path, dpi=150,
                model_bbox=self.model_bbox, **self.display) or {}
            self.finished_signal.emit(True, "", self.out_path)
        except Exception as e:
            msg = str(e).strip() or f"{type(e).__name__} (no message attached)"
            self.finished_signal.emit(False, msg, "")


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

        # Help first, so it is the first thing a new user sees rather than
        # something to hunt for after the plot has already confused them.
        self.btn_help = QPushButton("❓ Help — How Particle Tracking Works")
        self.btn_help.setStyleSheet(
            "background-color: #0e7490; color: white; font-weight: bold; padding: 6px;")
        self.btn_help.setToolTip(
            "Full guide to this page: every control, how to read the plot,\n"
            "and what to do when it does not look right.")
        self.btn_help.clicked.connect(self.show_help)
        form.addRow(self.btn_help)

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

        # Primaries vs secondaries. A run of 1000 source particles produced
        # 4103 tracks -- so three quarters of what is on screen is
        # secondaries, and being able to drop them is often what makes the
        # plot readable at all.
        self.chk_primaries = QCheckBox("Primary (source) particles")
        self.chk_primaries.setChecked(True)
        self.chk_primaries.setToolTip(
            "The particles emitted by the source itself.")
        form.addRow("", self.chk_primaries)

        self.chk_secondaries = QCheckBox("Secondary particles")
        self.chk_secondaries.setChecked(True)
        self.chk_secondaries.setToolTip(
            "Particles created during transport -- Compton electrons,\n"
            "fluorescence photons, and so on. Drawn thinner and dashed\n"
            "so they stay visually subordinate to the primaries.")
        form.addRow("", self.chk_secondaries)

        self.source_size_spin = QSpinBox()
        self.source_size_spin.setRange(2, 30)
        self.source_size_spin.setValue(7)
        self.source_size_spin.setToolTip("Marker size (points) for the source-birth circles.")
        form.addRow("Source Marker Size:", self.source_size_spin)

        self.collision_size_spin = QSpinBox()
        self.collision_size_spin.setRange(1, 20)
        self.collision_size_spin.setValue(4)
        self.collision_size_spin.setToolTip(
            "Marker size (points) for interaction dots. Lower this before\n"
            "hiding them entirely -- at 1-2 they read as texture on the\n"
            "track rather than as clutter over the geometry.")
        form.addRow("Interaction Dot Size:", self.collision_size_spin)

        self.collision_mode_combo = QComboBox()
        self.collision_mode_combo.addItems(["Points", "Density heatmap"])
        self.collision_mode_combo.setToolTip(
            "Points        -- one dot per interaction.\n"
            "Density heatmap -- the same interactions binned into a 2D\n"
            "                histogram. With thousands of collisions the\n"
            "                individual dots merge into noise; binned,\n"
            "                they show where interaction concentrates.")
        form.addRow("Interactions As:", self.collision_mode_combo)

        # Slice thickness. Without it the full depth of a 3D model is
        # flattened onto one plane, so tracks centimetres away from the
        # background slice are drawn on top of it as though they were in
        # it -- denser than the physics, and not matching the geometry
        # underneath.
        self.chk_fate = QCheckBox("End of history (absorbed / escaped)")
        self.chk_fate.setChecked(False)
        self.chk_fate.setToolTip(
            "Mark where each particle history ended:\n"
            "  filled red square  -- stopped inside the model (absorbed)\n"
            "  hollow blue triangle -- reached the outer boundary (escaped)\n"
            "The legend carries the counts, which is the number a shielding\n"
            "study is usually after: how much actually got through.")
        form.addRow("", self.chk_fate)

        self.slab_field = QLineEdit("")
        self.slab_field.setPlaceholderText("blank = project all depth")
        self.slab_field.setToolTip(
            "Draw only the parts of a track lying within +/- half this\n"
            "distance of the plane, along the axis the Basis does not\n"
            "cover (z for xy, y for xz, x for yz).\n\n"
            "Leave blank to keep projecting the entire depth onto the\n"
            "plane -- convenient, but it shows tracks that are nowhere\n"
            "near the geometry slice drawn underneath them.")
        form.addRow("Slice Thickness (cm):", self.slab_field)

        # Slice position. Redrawing no longer costs a transport run, so the
        # slab can be swept through the model to explore it in depth -- the
        # closest this 2D page gets to a 3D view. Moving it re-plots the
        # geometry background at the new depth as well, which does cost an
        # `openmc --plot`, hence sliderReleased rather than valueChanged.
        slice_row = QWidget()
        slice_layout = QHBoxLayout(slice_row)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.setToolTip(
            "Sweep the slice through the model along the axis the Basis\n"
            "does not cover. Needs a Slice Thickness to be set, and a run\n"
            "to have produced tracks. Releasing the handle re-plots the\n"
            "geometry background at the new depth too, so the tracks and\n"
            "the picture underneath always describe the same plane.")
        self.slice_slider.sliderReleased.connect(self._on_slice_moved)
        self.lbl_slice_pos = QLabel("--")
        self.lbl_slice_pos.setMinimumWidth(64)
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addWidget(self.lbl_slice_pos)
        form.addRow("Slice Position:", slice_row)

        # Energy window. Applied per state, so a photon that starts at
        # 1 MeV and degrades appears only along the stretch where it was
        # still above the threshold -- which is how the penetrating
        # radiation gets separated from the low-energy scatter around it.
        energy_row = QWidget()
        energy_layout = QHBoxLayout(energy_row)
        energy_layout.setContentsMargins(0, 0, 0, 0)
        self.energy_min_field = QLineEdit("")
        self.energy_min_field.setPlaceholderText("min eV")
        self.energy_max_field = QLineEdit("")
        self.energy_max_field.setPlaceholderText("max eV")
        for field in (self.energy_min_field, self.energy_max_field):
            field.setToolTip(
                "Energy window in eV; leave either side blank for no limit.\n"
                "Example: 100000 in 'min' shows only where particles were\n"
                "still above 100 keV.")
            energy_layout.addWidget(field)
        form.addRow("Energy Filter:", energy_row)

        self.track_color_combo = QComboBox()
        self.track_color_combo.addItems(["Particle type", "Energy (log scale)"])
        self.track_color_combo.setToolTip(
            "Particle type      -- one colour per particle; the line fades\n"
            "                   as that particle loses its own energy.\n"
            "Energy (log scale) -- absolute energy on a shared colour bar,\n"
            "                   so energies are comparable BETWEEN tracks\n"
            "                   and not only along one.")
        form.addRow("Color Tracks By:", self.track_color_combo)

        # Redrawing needs only matplotlib and data already in memory, so
        # every one of these is instant -- no second transport run.
        for widget in (self.chk_tracks, self.chk_source, self.chk_collisions,
                       self.chk_primaries, self.chk_secondaries, self.chk_fate):
            widget.toggled.connect(self.redraw_from_cache)
        for widget in (self.source_size_spin, self.collision_size_spin):
            widget.valueChanged.connect(self.redraw_from_cache)
        for widget in (self.collision_mode_combo, self.track_color_combo):
            widget.currentIndexChanged.connect(self.redraw_from_cache)
        for widget in (self.slab_field, self.energy_min_field, self.energy_max_field):
            widget.editingFinished.connect(self.redraw_from_cache)

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

        save_row = QWidget()
        save_layout = QHBoxLayout(save_row)
        save_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_save_image = QPushButton("🖼 Save Image")
        self.btn_save_image.setToolTip(
            "Save the picture currently on screen as a PNG (1500x1500).")
        self.btn_save_image.clicked.connect(self.save_current_image)
        self.btn_save_gif = QPushButton("🎞 Save Animation")
        self.btn_save_gif.setToolTip(
            "Save the replay -- the frames that reveal the tracks by\n"
            "elapsed simulation time -- as an animated GIF.")
        self.btn_save_gif.clicked.connect(self.save_animation_gif)
        save_layout.addWidget(self.btn_save_image)
        save_layout.addWidget(self.btn_save_gif)
        form.addRow("Export:", save_row)

        self.btn_append = QPushButton("\U0001F4BE Append Track Settings to Script")
        self.btn_append.setStyleSheet(
            "background-color: #4a4a4a; color: white; padding: 5px;")
        self.btn_append.clicked.connect(self.append_track_code)
        form.addRow(self.btn_append)


        self.viewer = ZoomableImageViewer()
        self.viewer.setMinimumSize(400, 400)

        # The control column grew past the height of a 768-pixel screen, so
        # the Run button fell off the bottom with no way to reach it. A
        # scroll area keeps every control reachable at any window height.
        # setWidgetResizable makes the panel follow the splitter's width
        # instead of staying at its size hint, so only a VERTICAL bar
        # appears -- a horizontal one would just hide the labels instead.
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(left_panel)
        controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setMinimumWidth(260)

        splitter.addWidget(controls_scroll)
        splitter.addWidget(self.viewer)
        splitter.setSizes([320, 700])
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

    def _fail(self, title, message, help_anchor="trouble"):
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
        # A failure is exactly when the explanation is worth having, so the
        # relevant help section is one click away instead of buried behind
        # the button at the top of a panel the user may have scrolled past.
        box.addButton(QMessageBox.StandardButton.Ok)
        help_button = box.addButton("Open Help", QMessageBox.ButtonRole.HelpRole)
        box.exec()
        if box.clickedButton() is help_button:
            self.show_help(help_anchor)

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

    def display_options(self):
        """Every drawing choice, in the exact keyword form _render_overlay
        takes. One place to read the widgets means the initial render and
        every later redraw cannot drift apart."""
        slab_text = self.slab_field.text().strip()
        slab = None
        if slab_text:
            try:
                slab = float(slab_text)
                if slab <= 0:
                    slab = None
            except ValueError:
                slab = None  # _parse_fields reports bad numbers; ignore here
        return {
            'show_tracks': self.chk_tracks.isChecked(),
            'show_source': self.chk_source.isChecked(),
            'show_collisions': self.chk_collisions.isChecked(),
            'show_primaries': self.chk_primaries.isChecked(),
            'show_secondaries': self.chk_secondaries.isChecked(),
            'source_marker_size': self.source_size_spin.value(),
            'collision_marker_size': self.collision_size_spin.value(),
            'collision_mode': ('heatmap'
                               if self.collision_mode_combo.currentText().startswith("Density")
                               else 'points'),
            'color_mode': ('energy'
                           if self.track_color_combo.currentText().startswith("Energy")
                           else 'particle'),
            'slab_thickness': slab,
            'show_fate': self.chk_fate.isChecked(),
            'energy_min': self._parse_optional_float(self.energy_min_field.text()),
            'energy_max': self._parse_optional_float(self.energy_max_field.text()),
        }

    def render_origin(self):
        """The Origin to draw with: the typed one, but with the slice
        normal replaced by the slider when it is active. Only the normal
        component moves -- the in-plane components still frame the view."""
        origin, _width, _pixels, err = self._parse_fields()
        if err:
            return None
        if not self.slice_slider.isEnabled():
            return origin
        h_idx, v_idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[
            self.basis_combo.currentText()]
        n_idx = 3 - h_idx - v_idx
        moved = list(origin)
        moved[n_idx] = self._slice_position()
        return tuple(moved)

    def show_help(self, anchor="contents"):
        """Open the help window, optionally at one section.

        Held as an attribute so it is not garbage-collected while open, and
        so a second press raises the existing window instead of stacking
        another copy on top of it.
        """
        if getattr(self, '_help_dialog', None) is None:
            self._help_dialog = TracksHelpDialog(self)
        self._help_dialog.show_section(anchor)

    def _parse_optional_float(self, text):
        text = (text or '').strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _on_slice_moved(self):
        """Slider released: show the new depth and redraw (background too)."""
        cache = getattr(self, '_track_cache', None)
        if not cache:
            return
        self.lbl_slice_pos.setText(f"{self._slice_position():.2f} cm")
        self.redraw_from_cache(replot_background=True)

    def _slice_position(self):
        """Slider value in cm along the slice normal."""
        return self.slice_slider.value() / 100.0

    def _configure_slice_slider(self, mins, maxs):
        """Give the slider the model's own depth range.

        Qt sliders are integers, so the range is held in hundredths of a
        centimetre -- fine enough for any geometry this page renders.
        """
        h_idx, v_idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[
            self.basis_combo.currentText()]
        n_idx = 3 - h_idx - v_idx
        lo, hi = float(mins[n_idx]), float(maxs[n_idx])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            self.slice_slider.setEnabled(False)
            self.lbl_slice_pos.setText("--")
            return
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(int(lo * 100), int(hi * 100))
        origin, _w, _p, err = self._parse_fields()
        start = origin[n_idx] if not err else (lo + hi) / 2.0
        self.slice_slider.setValue(int(np.clip(start, lo, hi) * 100))
        self.slice_slider.blockSignals(False)
        self.slice_slider.setEnabled(True)
        self.lbl_slice_pos.setText(f"{self._slice_position():.2f} cm")

    def save_current_image(self):
        """Copy the picture on screen to wherever the user wants it."""
        source = getattr(self, '_current_frame', None)
        if not source or not os.path.exists(source):
            self._fail("Nothing to Save",
                       "There is no rendered image yet. Run a tracked "
                       "simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Track Image", "particle_tracks.png", "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith('.png'):
            path += '.png'
        try:
            shutil.copyfile(source, path)
        except OSError as e:
            self._fail("Could Not Save Image", str(e))
            return
        self._log(f"💾 Track image saved to {path}")

    def save_animation_gif(self):
        """Write the replay frames out as an animated GIF."""
        frames = getattr(self, '_anim_frames', None)
        if not frames:
            self._fail("Nothing to Save",
                       "There is no animation yet. Run a tracked simulation "
                       "first -- the replay frames are produced by the run.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Track Animation", "particle_tracks.gif", "Animated GIF (*.gif)")
        if not path:
            return
        if not path.lower().endswith('.gif'):
            path += '.gif'
        try:
            from PIL import Image
            images = [Image.open(f).convert('P', palette=Image.Palette.ADAPTIVE)
                      for f in frames if os.path.exists(f)]
            if not images:
                raise ValueError("none of the frame files could be opened")
            # The last frame is the complete picture, so it is held longer
            # than the partial ones that build up to it.
            durations = [270] * (len(images) - 1) + [1500]
            images[0].save(path, save_all=True, append_images=images[1:],
                           duration=durations, loop=0)
        except Exception as e:
            self._fail("Could Not Save Animation",
                       f"{e}\n\nSaving a GIF needs Pillow (pip install Pillow).")
            return
        self._log(f"🎞 Track animation saved to {path} ({len(frames)} frames)")

    def redraw_from_cache(self, *_args, replot_background=False):
        """Re-render the overlay from track data already in memory.

        None of the display settings affect the transport calculation, so
        changing one has no business re-running OpenMC. Silently does
        nothing until a run has actually produced tracks.
        """
        cache = getattr(self, '_track_cache', None)
        if not cache:
            return
        if getattr(self, '_redraw_worker', None) is not None and self._redraw_worker.isRunning():
            # A redraw is already in flight. Remember that the settings
            # moved on so the result is refreshed once it lands, rather
            # than queueing a thread per keystroke.
            self._redraw_pending = True
            self._redraw_pending_replot = (
                getattr(self, '_redraw_pending_replot', False) or replot_background)
            return

        # Stop the replay animation: it would otherwise overwrite the new
        # image a few hundred milliseconds later with a stale frame.
        if getattr(self, '_anim_timer', None) is not None:
            self._anim_timer.stop()

        origin = self.render_origin() or cache['origin']
        replot = None
        if replot_background and cache.get('track_dir'):
            replot = {'track_dir': cache['track_dir'],
                      'pixels': cache['pixels'],
                      'color_by': cache['color_by']}

        self._redraw_pending = False
        self._redraw_pending_replot = False
        self.progress_bar.setVisible(True)
        out_path = os.path.join(os.path.dirname(cache['geometry_png']), "tracks_redraw.png")
        self._redraw_worker = TrackRedrawWorker(
            cache['tracks'], cache['geometry_png'], cache['basis'],
            origin, cache['width'], out_path, self.display_options(),
            model_bbox=cache.get('model_bbox'), replot=replot)
        self._redraw_worker.finished_signal.connect(self._on_redraw_finished)
        self._redraw_worker.start()

    def _on_redraw_finished(self, success, message, frame_path):
        self.progress_bar.setVisible(False)
        if not success:
            self._fail("Redraw Failed", message)
            return
        self.viewer.set_image(QPixmap(frame_path))
        self._current_frame = frame_path

        stats = getattr(self._redraw_worker, 'stats', {}) or {}
        if stats.get('escaped') or stats.get('absorbed'):
            total = stats['escaped'] + stats['absorbed']
            if total:
                self._log(f"🎯 Fate of {total} tracked particles: "
                          f"{stats['absorbed']} absorbed, {stats['escaped']} escaped "
                          f"({100.0 * stats['escaped'] / total:.1f}% got out).")

        if getattr(self, '_redraw_pending', False):
            # Settings changed while this ran; carry any pending background
            # re-plot into the follow-up rather than dropping it.
            self.redraw_from_cache(
                replot_background=getattr(self, '_redraw_pending_replot', False))

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
                "existing script, then run the tracked simulation.",
                help_anchor="quick")
            return

        origin, width, pixels, err = self._parse_fields()
        if err:
            self._fail("Invalid Plot Settings", err, help_anchor="view")
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
            display=self.display_options())
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

        # Keep the track data so display toggles can redraw from it. The
        # worker has finished by the time this runs, so reading its
        # attributes here needs no extra synchronisation.
        worker = getattr(self, '_worker', None)
        if worker is not None and getattr(worker, 'tracks', None):
            self._track_cache = {
                'tracks': worker.tracks,
                'geometry_png': worker.geometry_png,
                'basis': worker.basis,
                'origin': worker.origin,
                'width': worker.width,
                'pixels': worker.pixels,
                'color_by': worker.color_by,
                'track_dir': os.path.dirname(worker.geometry_png),
                'model_bbox': worker.model_bbox,
            }
            diag = diagnostics or {}
            if diag.get('has_any_points'):
                self._configure_slice_slider(diag['mins'], diag['maxs'])

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
                "position lies inside a cell of the model.",
                help_anchor="trouble")
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
                    "position.",
                    help_anchor="trouble")
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
        frame = self._anim_frames[self._anim_index]
        self.viewer.set_image(QPixmap(frame))
        self._current_frame = frame   # what "Save Image" writes out
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