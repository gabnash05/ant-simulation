"""controller.py — Controller functions for visual and headless modes."""

import numpy as np

from config import (
    SPECIES_LIST,
    SPECIES_PARAMS,
    NCR_SITES,
    GRID_W,
    GRID_H,
    TIMESTEPS,
    N_COLONY,
    FOOD_POS,
    FOOD_RADIUS,
    NEST_POSITIONS,
    SEARCHING,
    RETURNING,
    N_MC_RUNS,
)

from shade import generate_shade
from model import AntForagingModel
from simulation import run_monte_carlo
from utils.visualization_utils import (
    plot_foraging_efficiency_distributions,
    plot_collective_order,
    plot_dominance_heatmap,
    plot_shannon_entropy,
    plot_thermal_performance_curves,
    plot_thermal_stress_exposure,
    compute_mean_dominance_probability,
    generate_site_summary_table,
)

RUN_GUI = False  # module-level toggle; overridden by --visualize CLI flag

# ══════════════════════════════════════════════════════════════════════
# VISUAL MODE
# ══════════════════════════════════════════════════════════════════════


def run_visual_mode():
    """
    Interactive visual verification (PyQtGraph).

    Single-site run with live agent map, pheromone field, and E_i(t) (Eq. 17).
    """

    print("\n[VISUAL MODE] Launching interactive visualization…")
    import os

    os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    import numpy as np

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    site = NCR_SITES[
        13
    ]  # one of the more stable sites and better for visual verification

    T_base = np.random.normal(
        site["T_base_mean"],
        1.5,
        (GRID_W, GRID_H),
    )

    shade = generate_shade(
        site["G"],
        site["W"],
        site["B"],
    )

    model = AntForagingModel(
        T_base=T_base,
        shade=shade,
        run_id=0,
    )

    # ──────────────────────────────────────────────────────────────
    # DARK COLORMAP  (dark teal → dark amber, perceptually even)
    # Maps cool temps to deep teal-blues, warm temps to deep ambers,
    # keeping the mid-range visible while ants stay legible on top.
    # ──────────────────────────────────────────────────────────────

    _dark_pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    _dark_colors = [
        (0, 55, 75, 255),  # very dark teal     (cool)
        (0, 80, 90, 255),  # dark teal-green
        (60, 60, 0, 255),  # dark olive-yellow   (mid)
        (110, 50, 0, 255),  # dark burnt-orange
        (120, 20, 0, 255),  # very dark red-amber (hot)
    ]
    _dark_cmap = pg.ColorMap(pos=_dark_pos, color=_dark_colors)

    # ──────────────────────────────────────────────────────────────
    # WINDOW
    # ──────────────────────────────────────────────────────────────

    win = pg.GraphicsLayoutWidget(title="ANTS Visual Mode")
    win.resize(1600, 780)
    win.setBackground("#0d0d0d")
    win.show()

    # ──────────────────────────────────────────────────────────────
    # SIMULATION INFO HEADER
    # Displayed as a text label spanning the top of the window.
    # ──────────────────────────────────────────────────────────────

    header_label = pg.LabelItem(
        text=(
            f"<span style='color:#c8c8c8; font-size:11pt; font-family:monospace;'>"
            f"Site: <b>{site.get('name', 'NCR-0')}</b> &nbsp;|&nbsp; "
            f"Grid: <b>{GRID_W} × {GRID_H} cells</b> &nbsp;|&nbsp; "
            f"Species: <b>{', '.join(SPECIES_LIST)}</b> &nbsp;|&nbsp; "
            f"T<sub>base</sub> mean: <b>{site['T_base_mean']} °C</b> &nbsp;|&nbsp; "
            f"Timesteps: <b>{TIMESTEPS}</b> &nbsp;|&nbsp; "
            f"G={site['G']}  W={site['W']}  B={site['B']}"
            f"</span>"
        ),
        justify="left",
    )
    win.addItem(header_label, row=0, col=0, colspan=3)

    # ──────────────────────────────────────────────────────────────
    # PANELS
    # ──────────────────────────────────────────────────────────────

    agent_plot = win.addPlot(row=1, col=0)
    phi_plot = win.addPlot(row=1, col=1)
    eff_plot = win.addPlot(row=1, col=2)  # Repurposed from omega_plot

    for p in (agent_plot, phi_plot):
        p.setAspectLocked(True)
        p.setXRange(0, GRID_W)
        p.setYRange(0, GRID_H)

    # ──────────────────────────────────────────────────────────────
    # AXIS LABELS / UNITS
    # ──────────────────────────────────────────────────────────────

    agent_plot.setTitle(
        "<span style='color:#e0e0e0;'>Agent Map</span>",
        size="10pt",
    )
    agent_plot.setLabel("bottom", "X position", units="cells")
    agent_plot.setLabel("left", "Y position", units="cells")

    phi_plot.setTitle(
        "<span style='color:#e0e0e0;'>Pheromone Field  Σφ</span>",
        size="10pt",
    )
    phi_plot.setLabel("bottom", "X position", units="cells")
    phi_plot.setLabel("left", "Y position", units="cells")

    eff_plot.setTitle(
        "<span style='color:#e0e0e0;'>Foraging Efficiency  E<sub>i</sub>(t)</span>",
        size="10pt",
    )
    eff_plot.setLabel("bottom", "Timestep", units="steps")
    eff_plot.setLabel("left", "E_i", units="trips/capita")

    # ──────────────────────────────────────────────────────────────
    # TEMPERATURE BACKGROUND
    # ──────────────────────────────────────────────────────────────

    temp_img = pg.ImageItem()
    temp_img.setLevels((25, 45))
    temp_img.setColorMap(_dark_cmap)
    agent_plot.addItem(temp_img)

    # ──────────────────────────────────────────────────────────────
    # SHADE OVERLAY
    # ──────────────────────────────────────────────────────────────

    shade_img = pg.ImageItem()
    shade_img.setLevels((0.0, 1.0))
    shade_img.setOpacity(0.45)
    agent_plot.addItem(shade_img)

    # ──────────────────────────────────────────────────────────────
    # AGENT SCATTERS
    # ──────────────────────────────────────────────────────────────

    scatters = {}

    for sp in SPECIES_LIST:
        scat = pg.ScatterPlotItem(
            size=5,
            brush=SPECIES_PARAMS[sp]["color"],
            pen=None,
        )
        agent_plot.addItem(scat)
        scatters[sp] = scat

    # ──────────────────────────────────────────────────────────────
    # FOOD
    # ──────────────────────────────────────────────────────────────

    food = pg.ScatterPlotItem(
        [FOOD_POS[0]],
        [FOOD_POS[1]],
        size=FOOD_RADIUS * 2,
        symbol="s",
        brush="white",
        pen=pg.mkPen("black"),
        pxMode=False,
    )
    agent_plot.addItem(food)

    # ──────────────────────────────────────────────────────────────
    # NESTS
    # ──────────────────────────────────────────────────────────────

    for sp in SPECIES_LIST:
        nx, ny = NEST_POSITIONS[sp]
        nest = pg.ScatterPlotItem(
            [nx],
            [ny],
            size=10,
            symbol="s",
            brush=SPECIES_PARAMS[sp]["color"],
            pen=pg.mkPen("white"),
        )
        agent_plot.addItem(nest)

    # ──────────────────────────────────────────────────────────────
    # PHEROMONE FIELD
    # ──────────────────────────────────────────────────────────────

    phi_img = pg.ImageItem()
    phi_plot.addItem(phi_img)

    # ──────────────────────────────────────────────────────────────
    # STEP COUNTER LABEL
    # ──────────────────────────────────────────────────────────────

    step_label = pg.TextItem(
        text="Step: 0 / " + str(TIMESTEPS),
        color=(255, 255, 255),
        anchor=(0, 0),
        fill=pg.mkBrush(0, 0, 0, 160),
    )
    step_label.setFont(pg.QtGui.QFont("Courier New", 10, pg.QtGui.QFont.Weight.Bold))
    step_label.setPos(2, GRID_H - 2)
    step_label.setZValue(100)
    agent_plot.addItem(step_label)

    # ──────────────────────────────────────────────────────────────
    # EFFICIENCY CURVES
    # ──────────────────────────────────────────────────────────────

    eff_history = {sp: [] for sp in SPECIES_LIST}
    eff_curves = {}

    eff_plot.setXRange(0, TIMESTEPS)
    # Removed rigid Y-bounds since trips/capita scales dynamically past 1.0
    eff_plot.enableAutoRange(axis="y", enable=True)
    eff_plot.addLegend(offset=(10, 10))

    for sp in SPECIES_LIST:
        curve = eff_plot.plot(
            pen=pg.mkPen(SPECIES_PARAMS[sp]["color"], width=2),
            name=SPECIES_PARAMS[sp]["label"],
        )
        eff_curves[sp] = curve

    # ──────────────────────────────────────────────────────────────
    # UPDATE LOOP
    # ──────────────────────────────────────────────────────────────

    def update():

        if model.step_num >= TIMESTEPS:
            timer.stop()
            return

        model.step()

        # ──────────────────────────────────────────────────────────
        # STEP COUNTER UPDATE
        # ──────────────────────────────────────────────────────────

        step_label.setText(f"Step: {model.step_num} / {TIMESTEPS}")

        # ──────────────────────────────────────────────────────────
        # TEMPERATURE
        # ──────────────────────────────────────────────────────────

        temp_img.setImage(
            model.env.T_field.T,
            autoLevels=False,
        )

        # ──────────────────────────────────────────────────────────
        # SHADE
        # ──────────────────────────────────────────────────────────

        shade_img.setImage(
            model.env.shade.T,
            autoLevels=False,
        )

        # ──────────────────────────────────────────────────────────
        # AGENTS
        # ──────────────────────────────────────────────────────────

        for sp in SPECIES_LIST:
            active = [a for a in model.agents[sp] if a.state in (SEARCHING, RETURNING)]
            xs = [a.x for a in active]
            ys = [a.y for a in active]
            scatters[sp].setData(xs, ys)

        # ──────────────────────────────────────────────────────────
        # PHEROMONES
        # ──────────────────────────────────────────────────────────

        phi_sum = model.env.phi.sum(axis=0)
        phi_min = float(phi_sum.min())
        phi_max = float(phi_sum.max())
        if phi_max <= phi_min:
            phi_max = phi_min + 1e-6
        phi_img.setImage(phi_sum, autoLevels=False, levels=(phi_min, phi_max))

        # ──────────────────────────────────────────────────────────
        # EFFICIENCY (E_i) UPDATE
        # ──────────────────────────────────────────────────────────

        rec = model.records[-1]

        for sp in SPECIES_LIST:
            eff_val = rec.get(f"E_{sp}", 0.0)
            eff_history[sp].append(eff_val)
            eff_curves[sp].setData(eff_history[sp])

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(30)

    app.exec()


# ══════════════════════════════════════════════════════════════════════
# HEADLESS MODE
# ══════════════════════════════════════════════════════════════════════


def run_headless_batch(
    n_runs: int = N_MC_RUNS,
    n_sites: int = len(NCR_SITES),
    out_dir: str = "ants_output",
    verbose: bool = True,
):
    """
    Headless Monte Carlo batch mode

    Runs simulation across NCR sites, writes figures, validation table, and CSV export.
    """

    import os
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    sites = NCR_SITES[:n_sites]

    print("\n" + "═" * 65)
    print("  ANTS SIMULATION — HIGH-THROUGHPUT HEADLESS BATCH MODE")
    print(
        f"  Sites : {len(sites)}   Runs/site : {n_runs}   "
        f"Total runs : {len(sites) * n_runs}"
    )
    print(f"  Output: {out_dir}/")
    print("═" * 65 + "\n")

    all_results = run_monte_carlo(
        sites=sites,
        n_runs=n_runs,
        verbose=verbose,
    )

    print("\n[Reference runs] Mechanistic validation simulations (one per site)…")

    all_ref_records: list[dict] = []

    for site in sites:
        T_ref = np.random.normal(
            site["T_base_mean"],
            1.5,
            (GRID_W, GRID_H),
        )

        sh_ref = generate_shade(
            site["G"],
            site["W"],
            site["B"],
        )

        ref_m = AntForagingModel(
            T_base=T_ref,
            shade=sh_ref,
            run_id=-1,
        )

        ref_m.run(steps=TIMESTEPS)

        all_ref_records.extend(ref_m.records)

        print(f"    {site['name']}")

    print("\n[Figures] Generating output plots…")

    ground_truth = {s["name"]: s["dominant_species"] for s in sites}

    plot_thermal_performance_curves(out_dir)

    plot_foraging_efficiency_distributions(all_results, out_dir)

    plot_thermal_stress_exposure(all_results, out_dir)

    plot_dominance_heatmap(
        all_results,
        ground_truth,
        out_dir,
    )

    plot_shannon_entropy(
        all_results,
        out_dir,
    )

    plot_collective_order(
        all_ref_records,
        out_dir,
    )

    print("\n[Tables] Generating output tables…")

    generate_site_summary_table(
        all_results,
        ground_truth,
        out_dir,
    )

    # ── Mean dominance probability for each predicted-winner species ───────
    print("\n" + "─" * 50)
    print("  Mean Dominance Probability  P̄ᵢ (sites where species predicted dominant)")
    print("─" * 50)

    mean_dom_probs = compute_mean_dominance_probability(all_results)

    for label, prob in mean_dom_probs.items():
        print(f"  {label:<20s} P̄ᵢ = {prob:.4f}")

    # ── Mean thermal stress exposure per species ───────────────────────────
    print("\n" + "─" * 60)
    print("  Mean Thermal Stress Exposure  (proportion of colony-timesteps)")
    print(
        f"  {'Species':<22s}  {'τ_stress (T > T_opt)':>22s}  {'τ_critical (T ≥ CT_max)':>24s}"
    )
    print("─" * 60)

    all_runs = pd.concat(list(all_results.values()), ignore_index=True)

    for sp in SPECIES_LIST:
        label = SPECIES_PARAMS[sp]["label"]
        mean_stress = all_runs[f"tau_stress_{sp}"].mean()
        mean_critical = all_runs[f"tau_critical_{sp}"].mean()
        print(f"  {label:<22s}  {mean_stress:>22.4f}  {mean_critical:>24.4f}")

    print("\n[Export] Saving raw Monte Carlo dataset…")

    raw = pd.concat(
        list(all_results.values()),
        ignore_index=True,
    )

    raw_path = os.path.join(
        out_dir,
        "monte_carlo_raw_data.csv",
    )

    raw.to_csv(raw_path, index=False)

    print(f"  Saved: {raw_path}")

    print("\n" + "═" * 65)

    print("  SIMULATION COMPLETE")

    print(f"  All outputs saved to: {os.path.abspath(out_dir)}/")

    print("═" * 65 + "\n")
