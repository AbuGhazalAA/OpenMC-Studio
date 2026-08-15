"""Help content and viewer for the Particle Tracks page.

Kept in its own module so the documentation can grow without burying the
page's logic, and so the same dialog shell can later host help for other
tabs by handing it a different HTML body.

The text is written against what the page actually does -- including the
parts that surprise people (the slice projecting the whole model depth,
the reader falling back to raw HDF5, the .ppm plots) -- rather than being
a list of widget names.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, QLineEdit,
    QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextDocument


TRACKS_HELP_HTML = """
<style>
  body      { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt;
              color: #1f2937; line-height: 1.55; }
  h1        { font-size: 17pt; color: #6b21a8; margin-bottom: 2px; }
  h2        { font-size: 13pt; color: #6b21a8; margin-top: 22px;
              border-bottom: 2px solid #ddd6fe; padding-bottom: 3px; }
  h3        { font-size: 11.5pt; color: #374151; margin-top: 16px; margin-bottom: 4px; }
  code      { background: #f3f4f6; color: #b91c1c; padding: 1px 4px;
              font-family: Consolas, monospace; }
  pre       { background: #f3f4f6; padding: 8px; font-family: Consolas, monospace;
              font-size: 9.5pt; }
  .lead     { color: #4b5563; font-size: 11pt; }
  .toc      { background: #faf5ff; border: 1px solid #e9d5ff; padding: 8px 14px; }
  .toc a    { color: #6b21a8; text-decoration: none; }
  .key      { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 8px 12px;
              margin: 10px 0; }
  .warn     { background: #fef2f2; border-left: 4px solid #dc2626; padding: 8px 12px;
              margin: 10px 0; }
  .tip      { background: #f0fdf4; border-left: 4px solid #16a34a; padding: 8px 12px;
              margin: 10px 0; }
  table     { border-collapse: collapse; margin: 8px 0; }
  th, td    { border: 1px solid #d1d5db; padding: 5px 9px; text-align: left;
              vertical-align: top; }
  th        { background: #f3f4f6; }
  .sym      { font-weight: bold; }
</style>

<a name="contents"></a>
<h1>Particle Tracks</h1>
<p class="lead">Run a short OpenMC transport calculation, record the exact path of
every tracked particle, and draw those paths over a slice of your geometry.</p>

<div class="toc">
<b>Contents</b><br>
1. <a href="#what">What this page does</a><br>
2. <a href="#quick">Quick start</a><br>
3. <a href="#view">The view: Basis, Origin, Width, Resolution, Color By</a><br>
4. <a href="#run">The run: Particles to Track, Show Particle Type</a><br>
5. <a href="#layers">Display layers: what to draw</a><br>
6. <a href="#slice">Slice Thickness and Slice Position</a><br>
7. <a href="#color">Color Tracks By</a><br>
8. <a href="#energy">Energy Filter</a><br>
9. <a href="#fate">End of history: absorbed or escaped</a><br>
10. <a href="#interactions">Interactions: points or heatmap</a><br>
11. <a href="#reading">Reading the plot</a><br>
12. <a href="#export">Export</a><br>
13. <a href="#how">How it works underneath</a><br>
14. <a href="#speed">Performance</a><br>
15. <a href="#trouble">Troubleshooting</a><br>
16. <a href="#glossary">Glossary</a>
</div>

<a name="what"></a>
<h2>1. What this page does</h2>
<p>Most of OpenMC Studio reports <i>averages</i> -- a flux, a dose, a reaction rate
summed over millions of histories. This page does the opposite: it follows a
handful of individual particles and shows you where each one actually went.</p>
<p>That makes it a debugging and teaching tool rather than a scoring tool. It answers
questions a tally cannot:</p>
<ul>
  <li>Is my source emitting where I think it is, and in the direction I think?</li>
  <li>Do particles reach the detector at all, or stop in the shield first?</li>
  <li>Is that gap in my geometry a real streaming path?</li>
  <li>Where in the shield is the energy actually deposited?</li>
</ul>
<div class="key">
This is a <b>real</b> transport calculation, not a preview. It writes its own
XML into <code>export/tracks/</code> and runs the <code>openmc</code>
executable there. It never touches the results of a production run you did
from the Run menu.
</div>

<a name="quick"></a>
<h2>2. Quick start</h2>
<ol>
  <li>Build a model (Materials, Geometry, Settings) or load a script. The script
      must define a geometry <i>and</i> settings with a source.</li>
  <li>Set <b>Width</b> so the region you care about fits on screen.</li>
  <li>Leave <b>Particles to Track</b> at 20 for the first look. Raise it later.</li>
  <li>Press <b>Run Tracked Simulation</b> and watch the Live Output Console.</li>
  <li>When the image appears, change any display option -- it redraws instantly,
      without running OpenMC again.</li>
</ol>
<div class="tip">
Only the settings above the Run button affect the calculation. Everything below
it only affects drawing, so you can explore freely once a run is done.
</div>

<a name="view"></a>
<h2>3. The view</h2>

<h3>Basis (Plane)</h3>
<p>Which plane to slice through the model: <code>xy</code>, <code>xz</code> or
<code>yz</code>. This sets both the geometry picture and the two axes the tracks
are projected onto. The third axis -- the one the basis does not name -- is the
<b>slice normal</b>, and it is the axis <a href="#slice">Slice Thickness</a>
works along.</p>

<h3>Origin (cm)</h3>
<p>Three numbers: the centre of the view in model coordinates. The two components
matching the basis position the picture; the third decides <i>which</i> slice you
are looking at. For <code>xy</code>, an origin of <code>0, 0, 12</code> shows the
plane at z = 12 cm.</p>

<h3>Width (cm)</h3>
<p>Two numbers: how much of the model is visible, horizontally and vertically. Too
large and everything shrinks into the middle; too small and your tracks fall
outside the frame -- if that happens the page tells you and offers the numbers
that would frame them.</p>

<h3>Resolution</h3>
<p>Pixels in the background geometry image. 1000 x 1000 is plenty. Higher values
make OpenMC's plotting step slower without improving the tracks, which are drawn
as vectors on top and stay sharp regardless.</p>

<h3>Color By</h3>
<p><code>material</code> colours each region by the material filling it;
<code>cell</code> colours by cell. Use <code>cell</code> when two different cells
share a material and you need to tell them apart.</p>
<div class="key">
Colours are assigned by OpenMC from a fixed seed, so the same model always
renders the same way. They carry no physical meaning -- they only separate
regions.
</div>

<a name="run"></a>
<h2>4. The run</h2>

<h3>Particles to Track</h3>
<p>How many <b>source</b> particles to follow (batch 1, generation 1, particles
1..N). Each may create secondaries, which are tracked too -- so the number of
paths drawn is usually several times this number. A run of 1000 source photons
easily produces 4000 tracks.</p>
<table>
  <tr><th>Value</th><th>Use</th></tr>
  <tr><td>1 - 50</td><td>Checking a source position or a geometry gap. Fast, and
      you can follow individual paths with your eye.</td></tr>
  <tr><td>100 - 1000</td><td>Seeing a pattern: where the beam goes, where it
      stops.</td></tr>
  <tr><td>1000+</td><td>Density-style pictures. Use the interaction heatmap and
      consider hiding secondaries.</td></tr>
</table>
<p>The upper limit of 10000 matches a hard cap some OpenMC builds enforce
(<i>"Batch or Particle limit exceeded"</i>).</p>

<h3>Show Particle Type</h3>
<p>Keeps only one species: <code>neutron</code>, <code>photon</code>,
<code>electron</code> or <code>positron</code>. Applied while reading the track
file, so filtering does not require another run. If a type produced nothing, the
page says so rather than drawing an empty frame.</p>

<a name="layers"></a>
<h2>5. Display layers</h2>
<p>Every checkbox is independent -- any combination is valid, including all of
them or just one. Changing any of them redraws immediately from data already in
memory; OpenMC is not run again.</p>
<table>
  <tr><th>Layer</th><th>Shows</th></tr>
  <tr><td><b>Track lines</b></td>
      <td>The paths themselves, as polylines between recorded states.</td></tr>
  <tr><td><b>Source birth points</b></td>
      <td>A filled circle where each primary particle was born. Confirms your
          source is where you intended.</td></tr>
  <tr><td><b>Interactions/collisions</b></td>
      <td>A dot at each real interaction -- see
          <a href="#interactions">section 10</a>.</td></tr>
  <tr><td><b>Primary (source) particles</b></td>
      <td>Particles emitted by the source itself.</td></tr>
  <tr><td><b>Secondary particles</b></td>
      <td>Everything created during transport: Compton electrons, fluorescence
          photons, and so on. Drawn thinner and dashed so they stay
          subordinate.</td></tr>
  <tr><td><b>End of history</b></td>
      <td>Where each path ended -- see <a href="#fate">section 9</a>.</td></tr>
</table>
<div class="tip">
Turning off <b>Secondary particles</b> is often the single biggest improvement to
a crowded plot. In a typical photon run three quarters of the lines are
secondaries.
</div>

<h3>Marker sizes</h3>
<p><b>Source Marker Size</b> and <b>Interaction Dot Size</b> are in points. Before
hiding interaction dots entirely, try size 1 or 2 -- at that size they read as
texture along the track instead of clutter over the geometry.</p>

<a name="slice"></a>
<h2>6. Slice Thickness and Slice Position</h2>
<p>This is the most important idea on the page, and the least obvious.</p>

<h3>The problem</h3>
<p>The coloured background is a genuine slice: an infinitely thin cut through your
model at the plane you chose. Your particles, however, move in three dimensions.
With no slice thickness set, <b>every</b> track is drawn regardless of its depth,
flattening the whole model onto that one plane.</p>
<div class="warn">
A run through a 10 cm model reported positions spanning
<code>z[-4.82, 4.81] cm</code>. Every one of those tracks was drawn on top of a
picture of the plane at z = 0. A photon interacting 4.8 cm away appeared to
interact inside a region it never entered. The picture is denser than the
physics, and the material under a track may not be the material it hit.
</div>

<h3>Slice Thickness (cm)</h3>
<p>Enter a number and only the parts of a track lying within <b>half that
distance</b> of the plane are drawn. Entering <code>2</code> on an
<code>xy</code> basis with origin z = 0 draws only what lies in
<code>z = 0 &plusmn; 1 cm</code>. The title always states the depth being shown,
so a saved figure is never ambiguous.</p>
<p>A track that leaves the slab and comes back is drawn as <b>two separate
lines</b>, not one line jumping the gap -- the particle never travelled that
shortcut.</p>
<table>
  <tr><th>Leave blank when</th><th>Set a value when</th></tr>
  <tr><td>You want a visual survey: where does the radiation go overall? The full
      projection gives a much larger sample in one image.</td>
      <td>You are drawing any conclusion about a particular plane, or comparing
      tracks against the materials underneath them.</td></tr>
</table>

<h3>Slice Position</h3>
<p>The slider sweeps the slab along the slice normal, letting you explore the model
in depth from a 2D page -- much like scrolling through a CT scan. Its range comes
from the actual extent of the recorded tracks, and it stays disabled until a run
has produced some.</p>
<div class="key">
Releasing the slider also <b>re-plots the geometry background</b> at the new
depth. Moving the tracks without moving the picture underneath would recreate
exactly the mismatch slice thickness exists to remove. That re-plot calls
<code>openmc --plot</code>, so it takes a moment -- which is why it happens when
you release the handle, not while you drag it.
</div>

<a name="color"></a>
<h2>7. Color Tracks By</h2>

<h3>Particle type</h3>
<p>One colour per species -- neutron blue, photon green, electron red, positron
orange. Each line also fades along its length as that particle loses energy,
relative to <i>its own</i> starting energy.</p>
<div class="warn">
That fade is <b>relative</b>. Two lines at the same shade may be at completely
different energies: one at half of 10 MeV, another at half of 50 keV.
</div>

<h3>Energy (log scale)</h3>
<p>Colours every segment by its <b>absolute</b> energy on one shared logarithmic
scale, with a colour bar in eV. Now the same colour means the same energy
everywhere in the picture, so you can see where particles are still penetrating
and where they have thermalised. The logarithmic scale is used because particle
energies span many decades.</p>

<a name="energy"></a>
<h2>8. Energy Filter</h2>
<p>Two boxes, in <b>eV</b>; leave either blank for no limit. Type
<code>100000</code> in the first to keep only energies above 100 keV.</p>
<div class="key">
The filter applies <b>per recorded state</b>, not per track. A photon born at
1 MeV that degrades to 50 keV is drawn only along the stretch where it was still
above your threshold, then disappears. That is what separates the penetrating
radiation that matters for shielding from the low-energy scatter filling the same
space.
</div>
<p>If nothing is drawn, your window is probably outside the energies present --
the title always states the window in use.</p>

<a name="fate"></a>
<h2>9. End of history: absorbed or escaped</h2>
<p>Marks where each particle's history ended:</p>
<table>
  <tr><th>Marker</th><th>Meaning</th></tr>
  <tr><td class="sym" style="color:#b91c1c">&#9632; filled red square</td>
      <td><b>Absorbed</b> -- the particle stopped inside the model.</td></tr>
  <tr><td class="sym" style="color:#0ea5e9">&#9651; hollow blue triangle</td>
      <td><b>Escaped</b> -- it reached the model's outer boundary and left the
          problem.</td></tr>
</table>
<p>The legend carries the counts, and the console reports the percentage that got
out. For a shielding study that percentage is often the whole point: <i>how much
actually made it through?</i></p>
<div class="key">
The track file records <b>positions</b>, not the reason a history ended. The
classification therefore compares the final position against the geometry's
bounding box. If your model has no finite bounding box, fates are reported as
unclassified rather than guessed.
</div>

<a name="interactions"></a>
<h2>10. Interactions: points or heatmap</h2>
<p>An interaction is detected from the data, not estimated from the shape of the
path: OpenMC writes a new state only on a real event, so a state whose
<code>cell_id</code> <i>and</i> <code>material_id</code> both match the previous
one cannot be a surface crossing -- it is a collision.</p>
<table>
  <tr><th>Mode</th><th>Use</th></tr>
  <tr><td><b>Points</b></td>
      <td>One dot per interaction. Best at low particle counts, where you can
          follow what happened to each particle.</td></tr>
  <tr><td><b>Density heatmap</b></td>
      <td>The same interactions binned into a 2D histogram with its own colour
          bar. At high particle counts individual dots merge into noise; binned,
          they show where interaction actually concentrates. Empty bins stay
          transparent so the geometry shows through.</td></tr>
</table>
<div class="tip">
The heatmap is the natural way to see where a shield is doing its work. Combine
it with a slice thickness so the map describes one plane rather than the whole
depth.
</div>

<a name="reading"></a>
<h2>11. Reading the plot</h2>
<ul>
  <li><b>Title</b> -- always states the plane, the depth being shown
      (<code>Z = 0.00 &plusmn; 1.00 cm</code> or <code>all Z projected</code>),
      any energy window, and which layers are on. A saved figure is
      self-describing.</li>
  <li><b>Legend</b> (upper right) -- particle types, a dashed entry when
      secondaries are shown, and the fate counts.</li>
  <li><b>Right colour bar</b> -- particle energy in eV, when colouring by
      energy.</li>
  <li><b>Left colour bar</b> -- interactions per bin, when using the
      heatmap.</li>
  <li><b>&times; markers</b> -- a particle with only one recorded state, so
      there is no line to draw. Many of these means particles are being absorbed
      essentially at birth.</li>
</ul>
<p>After a run, the image plays back as a short animation revealing the tracks by
<b>elapsed simulation time</b> -- not by fraction of each path. A particle that
free-streams a long way between collisions visibly covers more ground per frame
than one that scatters immediately.</p>

<a name="export"></a>
<h2>12. Export</h2>
<p><b>Save Image</b> writes the picture currently on screen as a PNG at
1500 x 1500. <b>Save Animation</b> assembles the replay frames into an animated
GIF, holding the final complete frame longer than the partial ones.</p>
<p><b>Append Track Settings to Script</b> adds the equivalent
<code>settings.track = ...</code> line to your script, so a production run can
record the same particles.</p>

<a name="how"></a>
<h2>13. How it works underneath</h2>
<ol>
  <li>Your script is executed in memory and the geometry, materials, settings and
      tallies objects are picked out <i>by type</i>, so variable names do not
      matter.</li>
  <li>A copy of the settings is modified for a fast, track-focused run:
      <code>settings.track</code> is set to the requested particles, batches to 1,
      and the particle count reduced to exactly what you asked for.</li>
  <li>That model is written to <code>export/tracks/</code> and the
      <code>openmc</code> executable is run there, streaming its output into the
      Live Output Console.</li>
  <li>OpenMC writes <code>tracks.h5</code>, which is read back and drawn over a
      geometry plot rendered separately by <code>openmc --plot</code>.</li>
</ol>
<div class="key">
Track data is only written when the whole run finishes, so the drawing cannot be
live. The animation is a replay produced immediately afterwards.
</div>
<p>Two compatibility fallbacks run automatically and report themselves in the
console:</p>
<ul>
  <li>Builds compiled without <code>libpng</code> write the plot as
      <code>.ppm</code>. It is converted to PNG automatically.</li>
  <li>Builds whose Python package is older than their binary cannot decode the
      particle type in the track file (<i>"22 is not a valid ParticleType"</i> --
      22 is the PDG number for a photon). The file is then read directly with
      h5py, accepting both encodings.</li>
</ul>

<a name="speed"></a>
<h2>14. Performance</h2>
<ul>
  <li><b>Particles to Track</b> dominates everything. Each path is drawn
      individually, so both the run and the drawing scale with it.</li>
  <li>Display changes are free -- they never re-run OpenMC. Explore as much as
      you like once a run is done.</li>
  <li><b>Slice Position</b> is the one exception: it re-plots the geometry, which
      costs an <code>openmc --plot</code>.</li>
  <li>A slice thickness makes drawing <i>faster</i> as well as more honest, since
      out-of-slab states are dropped before anything is drawn.</li>
  <li>Resolution affects only the background image.</li>
</ul>

<a name="trouble"></a>
<h2>15. Troubleshooting</h2>

<h3>Nothing appears at all</h3>
<p>Every failure now opens a dialog; check the Live Output Console for the full
text. The most common cause is that the <code>openmc</code> executable is not on
your PATH -- the page checks this before starting and tells you. Confirm with
<code>openmc --version</code> in a terminal.</p>

<h3>The image looks empty but the run succeeded</h3>
<p>Three causes, and the page distinguishes them for you:</p>
<ul>
  <li><b>Tracks outside the view</b> -- they were drawn, but off the edge. You are
      offered the Origin and Width that would frame them.</li>
  <li><b>No positions recorded</b> -- the source is producing nothing inside the
      geometry. Check <code>settings.source</code>.</li>
  <li><b>Every particle absorbed at birth</b> -- a real physical result, not a
      display fault. Check the source energy and the material at its
      position.</li>
</ul>

<h3>The plot is a dense unreadable blob</h3>
<p>In order of effect: hide secondaries, set a slice thickness, switch
interactions to the heatmap, reduce the interaction dot size, lower the particle
count.</p>

<h3>Tracks do not line up with the geometry</h3>
<p>Almost always the depth. Set a slice thickness -- see
<a href="#slice">section 6</a>.</p>

<a name="glossary"></a>
<h2>16. Glossary</h2>
<table>
  <tr><td><b>History</b></td><td>One source particle and everything it
      creates.</td></tr>
  <tr><td><b>Primary</b></td><td>A particle emitted by the source.</td></tr>
  <tr><td><b>Secondary</b></td><td>A particle created during transport.</td></tr>
  <tr><td><b>State</b></td><td>One recorded point along a path: position,
      direction, energy, time, weight, cell and material.</td></tr>
  <tr><td><b>Slice normal</b></td><td>The axis the basis does not name -- z for
      <code>xy</code>, y for <code>xz</code>, x for <code>yz</code>.</td></tr>
  <tr><td><b>Leakage</b></td><td>The fraction of particles escaping the model.
      OpenMC prints it in the console; the fate markers show it
      spatially.</td></tr>
  <tr><td><b>PDG number</b></td><td>The standard particle code stored in the track
      file: 2112 neutron, 22 photon, 11 electron, -11 positron.</td></tr>
</table>

<p><br><i>OpenMC Studio -- Particle Tracks</i></p>
"""


class TracksHelpDialog(QDialog):
    """Scrollable help viewer with a find box.

    Modeless on purpose: the point is to read an explanation while changing
    the very control it describes, which a modal dialog would block.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Particle Tracks — Help")
        # Give it real window controls: this is a document to be read
        # alongside the page, so it should maximise like any other window.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint)
        self.resize(900, 720)

        layout = QVBoxLayout(self)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        # Internal #anchor links drive the table of contents; letting the
        # browser follow them itself would try to load them as documents.
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._go_to_anchor)
        self.browser.setHtml(TRACKS_HELP_HTML)
        layout.addWidget(self.browser)

        controls = QHBoxLayout()
        self.find_field = QLineEdit()
        self.find_field.setPlaceholderText("Find in help…")
        self.find_field.returnPressed.connect(self._find_next)
        controls.addWidget(QLabel("🔍"))
        controls.addWidget(self.find_field)

        btn_find = QPushButton("Find Next")
        btn_find.clicked.connect(self._find_next)
        controls.addWidget(btn_find)

        btn_top = QPushButton("↑ Contents")
        btn_top.clicked.connect(lambda: self.browser.scrollToAnchor("contents"))
        controls.addWidget(btn_top)

        controls.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        controls.addWidget(btn_close)
        layout.addLayout(controls)

    def _go_to_anchor(self, url):
        self.browser.scrollToAnchor(url.toString().lstrip('#'))

    def _find_next(self):
        """Search forward, wrapping to the top when the end is reached so a
        second press never looks like the term simply is not there."""
        text = self.find_field.text().strip()
        if not text:
            return
        if self.browser.find(text):
            return
        cursor = self.browser.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        self.browser.setTextCursor(cursor)
        self.browser.find(text)

    def show_section(self, anchor):
        """Open (or raise) the dialog scrolled to one section."""
        self.show()
        self.raise_()
        self.activateWindow()
        self.browser.scrollToAnchor(anchor)
