"""
╔══════════════════════════════════════════════════════════════════════════╗
║   ANTS — Thermally Constrained Pheromone-Mediated Foraging Simulation   ║
║   Agent-Based Model (Mesa) + Monte Carlo Simulation                     ║
║   Based on: ANTS-METHODOLOGY (NCR, Philippines)                         ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python ants_simulation.py                     # headless batch (500 runs × 16 sites)
    python ants_simulation.py --visualize          # interactive matplotlib animation
    python ants_simulation.py --runs 50 --out out  # quick test run
    python ants_simulation.py --sites 2 --runs 10  # two sites, 10 runs each
"""

import argparse
import warnings
from config import N_MC_RUNS, NCR_SITES
from controller import run_visual_mode, run_headless_batch, RUN_GUI

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ANTS: Thermally Constrained Pheromone-Mediated Foraging ABM"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch matplotlib visual verification mode (single site)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_MC_RUNS,
        help=f"Monte Carlo runs per site  (default: {N_MC_RUNS})",
    )
    parser.add_argument(
        "--sites",
        type=int,
        default=len(NCR_SITES),
        help=f"Number of NCR sites to process  (default: {len(NCR_SITES)})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ants_output",
        help="Output directory for figures and tables  (default: ants_output)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress output",
    )
    args = parser.parse_args()

    if args.visualize or RUN_GUI:
        run_visual_mode()
    else:
        run_headless_batch(
            n_runs=args.runs,
            n_sites=args.sites,
            out_dir=args.out,
            verbose=not args.quiet,
        )
