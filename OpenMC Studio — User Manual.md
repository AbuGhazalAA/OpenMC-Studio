# OpenMC Studio — User Manual

A practical guide to building, running, and visualizing Monte Carlo radiation transport simulations with OpenMC Studio.

---

## 1. What This Tool Does

OpenMC Studio wraps the OpenMC Monte Carlo transport code in a graphical interface. Instead of writing every material, surface, and setting by hand in a Python script, you build your model piece by piece through dedicated editor pages — and the corresponding Python code is written into the script editor for you automatically. You can then run the simulation, and inspect the geometry, particle behavior, and results without leaving the application.

This makes it suited to reactor physics modeling, shielding calculations, criticality problems, and — with its dedicated pulse-height tally mode — radiation detector response simulation (HPGe, NaI, and similar detectors).

---

## 2. The Main Window

When you launch OpenMC Studio, you'll see two main areas side by side:

- **Left side** — a set of tabs: **2D Geometry Viewer**, **3D Voxel Viewer**, **Particle Tracks**, and **Simulation Results**. This is where you *see* your model.
- **Right side** — the **Python Script Engine**, a live view of the Python code describing your model. You rarely need to type here directly — the editor pages write to it for you — but you can always read, edit, save, or load it as a plain `.py` file.

Along the top, the **Build (Setup)** menu opens the editor pages for Materials, Geometry, Settings, Tallies, and the Nuclear Data Library manager, each as a separate window. The toolbar gives you quick access to **Validate Geometry & Plot** and **Run OpenMC**.

---

## 3. Building a Model, Step by Step

A model needs four things before it can run: materials, geometry, settings (including a source), and — usually — at least one tally to record results.

### 3.1 Materials

Open **Build (Setup) → Materials Editor**. You have two ways to add materials:

- **Built-in library**: pick from common materials (water, heavy water, B₄C, lead, stainless steel, concrete, polyethylene, enriched UO₂, HPGe, NaI) and click **Add Library Material to Script**. Densities and compositions are pre-filled with standard values.
- **Custom builder**: name your material, set its density and units, then add nuclides or elements one at a time with their fraction (atomic or weight). Click **Generate Custom Material Script** when done.

Each addition appends the matching Python code to the script editor immediately — you can watch it happen.

### 3.2 Geometry

Open **Build (Setup) → Geometry Builder**. Define surfaces (sphere, cylinder, or plane) by name, type, and radius/coordinate, with an optional boundary condition (vacuum, reflective, or transmission — the default). Use **Undo** to step back if you make a mistake.

For common shapes, the **Insert Macrobody to Script** control at the top of this window drops in a ready-made cylinder, box, sphere, or truncated cone directly into your script — useful if you're used to MCNP-style macrobodies.

Note: this page defines *surfaces*. Turning surfaces into filled *cells* (combining regions and assigning materials) is done directly in the script editor, using standard OpenMC syntax (`openmc.Cell(fill=..., region=...)`).

### 3.3 Settings and Source

Open **Build (Setup) → Simulation Settings**. This page covers:

- **Cross-section library path** — where OpenMC finds nuclear data. If `OPENMC_CROSS_SECTIONS` is already set in your environment, it's pre-filled.
- **Run mode** — *fixed source* (e.g. a detector or shielding problem) or *eigenvalue* (a criticality/reactor problem). Eigenvalue mode reveals an extra "Inactive Batches" field.
- **Batches / Particles per batch** — how much statistics you want. More particles means better statistics but longer run time.
- **Visual Tracking** — check this and set a particle count to enable the data needed for the Particle Tracks tab (see §4.3). Keep this number modest; it's a real transport calculation, not a preview.
- **Active Particles** — enable photon and/or electron/positron transport if your problem needs them (for example, gamma emission from a decaying source, or electron energy deposition).
- **Advanced Source Definition** (fixed-source mode only) — particle type, spatial distribution (point or box), and energy distribution (a single discrete energy, or a Watt fission spectrum).

Click **Apply Settings & Generate Script** to write everything to the script.

### 3.4 Tallies

Open **Build (Setup) → Tallies & Detectors**. Name your tally, choose one or more scores (flux, absorption, fission, scatter, etc.), and pick a filter:

- **CellFilter / MaterialFilter** — restrict the tally to a specific cell or material (enter its script variable name, e.g. `c1`).
- **EnergyFilter** — bin results by energy (comma-separated bin boundaries in eV).
- **MeshFilter** — a 3D regular mesh over a defined region.
- **Pulse Height (Detector F8)** — the detector-specific mode. This automatically selects the `pulse-height` score, and asks for the detector cell, energy channel bins, and Gaussian Energy Broadening parameters (`FWHM = a + b·√(E + c·E²)`), matching how a real HPGe or NaI detector's resolution is characterized.

Click **Add Tally & Generate Script**.

---

## 4. Visualizing Your Model

### 4.1 2D Geometry Viewer

Set a basis plane (xy/xz/yz), origin, width, resolution, and coloring (by material or by cell), then click **Force Render Now**. Two conveniences:

- **View → Fit Full Geometry** automatically frames your entire model.
- **Zoom to Cell** jumps to a specific cell's own bounding box, populated automatically from whatever cells your script currently defines.

**Append Plot Code to Script** saves your current view settings into the script permanently, if you want to keep it.

### 4.2 3D Voxel Viewer

Click **1. Generate Voxel Script** to add a voxel plot definition, then **2. Render Real-Time 3D (GPU)** to build and display an interactive, GPU-accelerated 3D model of your geometry — rotate, pan, and zoom freely. This reflects your script's actual geometry each time you render it, not a cached image.

### 4.3 Particle Tracks

This tab runs a small, real OpenMC transport calculation (using the particle count you set in §3.3) and overlays the actual simulated particle paths on your geometry:

- **● (circle)** — where a particle was born (source point)
- **thin line** — the particle's path
- **· (dot)** — a point where the particle actually interacted with the material (scattered, was absorbed and re-emitted, etc.) — distinct from simply crossing a boundary

Each display layer (tracks, source points, interactions) can be toggled independently, and results play back as a short animation once the run completes. A separate window reminds you not to close the application while this real calculation is in progress.

### 4.4 Simulation Results

After a run completes, the latest statepoint file loads automatically. Tally results are available here for inspection, including spectrum visualization with GEB broadening applied for pulse-height tallies — so you can see a simulated detector spectrum the way it would actually appear on real instrumentation.

#### Depletion / burnup runs

When the run was a depletion calculation, the depletion results file (`depletion_results.h5`) is the one that loads — not the last per-step statepoint — and the **Depletion / Burnup Results Table** appears filled in, without any further clicks. It holds every step the file contains, whether the script asked for four timesteps or eighty:

| Column | Meaning |
| --- | --- |
| Step | Index of the depletion step as stored in the file |
| Time (days) | Cumulative irradiation time |
| Burnup (MWd/kgHM = GWd/MTU) | Cumulative burnup — the two units are the same number. Shown as *Energy (MWd)* instead if the inventory contains no heavy metal to divide by |
| k-effective, Std Dev (k) | Multiplication factor of each step and its statistical uncertainty |
| Reactivity (pcm), Std Dev (pcm) | ρ = (k−1)/k × 10⁵ |

Switching **View** to *Nuclide inventory* keeps the same rows and replaces the k-effective columns with one column per selected nuclide, in atoms, grams, weight %, atom density (a/b-cm) or N/N₀, for one material or all of them summed.

Everything in the table is copyable: **📋 Copy Whole Table** puts it on the clipboard as tab-separated text that pastes into Excel or OriginLab as proper cells, **Ctrl+C** (or **📋 Copy Selection**) copies just the selected block, and **💾 Export This Table (CSV)** saves it at full numerical precision. The line above the table summarises what was extracted — number of steps, time and burnup range, how k moved, and the burnup at which k crosses 1.

---

## 5. Running a Simulation

Click **✔ Validate Geometry & Plot** first — this builds your model and shows the 2D geometry, letting you catch definition errors before committing to a full run. Once you're satisfied, click **▶ Run OpenMC**. You'll be asked for an optional custom name for the results file; leave it blank to use OpenMC's default naming.

A progress window shows elapsed time and estimated time remaining, and the Live Output Console (bottom panel) streams OpenMC's own output as the run proceeds. When finished, results load automatically and the view switches to the Simulation Results tab.

---

## 6. Working with the Script Directly

Everything the editor pages generate lands in the Python Script Engine panel on the right. You can:

- Edit it by hand at any point — OpenMC Studio doesn't lock you out of the underlying code.
- **Save Script** / **Load Script** as a plain `.py` file.
- **▶ Run / Sync to GUI** executes the script directly, independent of the Materials/Geometry/etc. pages — useful for testing a script you've written or modified yourself.

---

## 7. Managing Projects

- **File → New Project** clears everything back to a fresh state — script, all visualizations, the console, and every editor window (Materials, Geometry, Settings, Tallies, Libraries) — equivalent to closing and reopening the application.
- **File → Save** writes your current script to an `.omcs` or `.py` file.
- **File → Open Project** loads a previously saved `.omcs` or `.py` file back into the script editor.

---

## 8. Tips for Detector Simulation Work

If you're modeling a detector (HPGe, NaI, or similar):

1. Use the **Pulse Height (Detector F8)** tally mode (§3.4) rather than a plain flux or absorption tally — it's purpose-built for this and handles the energy-channel binning for you.
2. Remember to enable **photon transport** in Settings if your source emits gammas (§3.3) — pulse-height tallies without photon transport enabled will not produce meaningful results.
3. GEB parameters (a, b, c) should reflect your *actual characterized detector resolution*, not default placeholder values — these determine how sharply your simulated peaks resemble a real spectrum.
4. Use **Particle Tracks** (§4.3) with a modest particle count to visually sanity-check that particles are actually reaching and interacting with your detector volume before committing to a full production run.

---

## 9. Troubleshooting

- **A script error dialog appears when running**: read the message — it's the actual Python/OpenMC error text. Common causes are a missing `settings` or `geometry` object, or forgetting to activate photon transport for a pulse-height tally.
- **The geometry plot looks wrong or empty**: check your Origin and Width fields — you may be viewing a region of space your geometry doesn't occupy. Try **View → Fit Full Geometry** to reorient.
- **The 3D Voxel or Particle Tracks tab shows nothing**: these require a real transport/plotting step to complete first — check the Live Output Console for errors, and confirm your cross-section library path is set correctly (Nuclear Data Libraries Manager).
