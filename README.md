# Thermally Constrained Pheromone-Mediated Foraging Simulation

Agent-based model of urban ant foraging under thermal stress and species-specific pheromone chemistry, with Monte Carlo runs across NCR (Philippines) sites. Implements *A Stochastic Agent-Based Model of Thermally Constrained Pheromone-Mediated Foraging in Urban Ant Communities*.

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt` (NumPy, pandas, geopandas, matplotlib, PyQt5, pyqtgraph, Mesa, etc.)
- Processed site data: `data/processed/sites_with_lst.csv` (see Installation)

## Installation

```bash
pip install -r requirements.txt
```

*Note: This next step is only required if you need to regenerate the site inputs from scratch (which you dont). The sites already come preprocessed.*

Prepare site inputs (from project root):

```bash
python src/scripts/extract_lst.py
```

Place raw geospatial inputs under `data/raw/` (`Manila.shp`, `Manila_csv.tab`) as expected by the extraction script.


## Running the Simulation

All simulation commands must be executed from the `src/` directory to ensure that relative python module paths and configurations resolve correctly:

```bash
cd src
```

## CLI Arguments Reference

The table below breaks down the available Command Line Interface (CLI) parameters defined in `main.py`:

| **Parameter** | **Type** | **Default Value** | **Description** |
| --- | --- | --- | --- |
| **`--visualize`** | Flag | *None* | Disables the batch processing pipeline and launches the graphical UI view of agent tracking and real-time pheromone grid dynamics for a single site. |
| **`--runs`** | Integer | `N_MC_RUNS` | The number of Monte Carlo simulation runs to execute per study site. High values improve statistical density and stability. |
| **`--sites`** | Integer | `len(NCR_SITES)` | The total number of geographical sites (from your config) to sequentially process during a headless batch. |
| **`--out`** | String | `"ants_output"` | The target output directory where generated figures, diagnostic charts, and summary statistics CSV files will be stored. |
| **`--quiet`** | Flag | *None* | Suppresses step-by-step console logging for individual simulation runs. Highly recommended for cleaning up automated runner logs. |

## Execution Patterns

### 1. Headless Batch Mode (Default)

Run multi-site Monte Carlo simulations completely in the background. The script processes each site and outputs compiled results (such as data tables and figures) directly to disk.

* **Standard Production Simulation** (runs 500 Monte Carlo runs across 16 sites, saving plots to the default directory):
```bash
python main.py --runs 500 --sites 16 --out ants_output
```


* **Fast Diagnostic Run** (runs a lightweight test with only 50 iterations across 2 sites, suppressing console clutter):
```bash
python main.py --runs 50 --sites 2 --out test_output --quiet
```



### 2. Visual Mode

Launches a real-time single-site interactive graphical display (utilizing PyQtGraph/Matplotlib as configured) for immediate visual confirmation of agent locations, pheromone networks, and foraging efficiency.

```bash
python main.py --visualize
```


## Project Structure

| Path | Role |
|------|------|
| `src/main.py` | CLI entry point |
| `src/model.py` | Simulation controller, recruitment, metrics |
| `src/agents.py` | Off-lattice ant agents and behavior |
| `src/environment.py` | Thermal field and pheromone layers |
| `src/thermal.py` | Thermal performance and decay rates |
| `src/shade.py` | Stochastic shade grid generation |
| `src/simulation.py` | Monte Carlo runner |
| `src/controller.py` | Headless batch and visual modes |
| `src/config.py` | Parameters and NCR site loading |
| `src/config/*.toml` | Species, grid, and site definitions |
| `src/utils/visualization_utils.py` | Figures and validation tables |
| `src/scripts/extract_lst.py` | LST / land-cover preprocessing |

## Outputs

Headless runs create an output directory (default `ants_output/`) containing:

Here are short descriptions for each output:

---

**`monte_carlo_raw_data.csv`**
Raw per-run results across all Monte Carlo runs and sites. Each row records one run's foraging efficiency, collective order, thermal exposure, and dominant species — the base dataset for all figures and tables.

**`fig1_thermal_performance_curves.png`**
Gaussian TPC curves for all four species plotted against temperature. Shows each species' performance peak (Topt), thermal breadth, and where their curves fall relative to NCR's observed LST range.

**`fig2_foraging_efficiency_distributions.png`**
Box plots of normalized foraging efficiency (E_i) per species across all 16 sites, sorted by mean LST. Illustrates the widening gap between invasive and native species as temperatures rise.

**`fig3_thermal_stress_exposure.png`**
Grouped bar chart of mean τ_stress and τ_critical per species. Shows what proportion of simulation time each species spends above Topt and at or above CTmax.

**`fig4_dominance_probability_heatmap.png`**
Heatmap of P_i (win probability) per species across all 16 sites sorted by LST. Highlights the ~47–48°C transition where dominance shifts from P. longicornis to S. geminata.

**`fig5_shannon_entropy.png`**
Bar chart of Shannon dominance entropy (H) per site. Peaks at the competitive transition zone, identifying sites where small microclimate differences could tip dominance outcomes.

**`fig6_collective_order_validation.png`**
Ω (collective order parameter) vs. temperature curves per species, with Topt, CTmax, and Tcollapse marked. Mechanistically validates that pheromone network collapse falls within biologically realistic thermal bounds.

**`table1_site_validation_summary.csv`**
Side-by-side comparison of observed vs. predicted dominant species per site. The basis for the 62.5% classification accuracy reported in the paper.
