"""simulation.py — Monte Carlo simulation runner."""

import numpy as np
import pandas as pd
from config import NCR_SITES, N_MC_RUNS, GRID_W, GRID_H, TIMESTEPS
from shade import generate_shade
from model import AntForagingModel


def run_monte_carlo(
    sites: list = None,
    n_runs: int = N_MC_RUNS,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Execute the full Monte Carlo simulation across all NCR sites.
    Returns {site_name: DataFrame of per-run metrics}.
    """
    if sites is None:
        sites = NCR_SITES

    all_results: dict[str, pd.DataFrame] = {}

    for s_idx, site in enumerate(sites):
        name = site["name"]
        T_mean = site["T_base_mean"]
        G, W, B = site["G"], site["W"], site["B"]

        if verbose:
            print(
                f"  [{s_idx + 1:>2}/{len(sites)}] {name}  "
                f"T_mean={T_mean:.1f}°C  G={G:.2f}  W={W:.2f}  B={B:.2f}"
            )

        run_rows = []
        for run_id in range(n_runs):
            # Stochastic T_base: ±σ variation around site mean LST
            T_base = np.random.normal(T_mean, 1.5, (GRID_W, GRID_H))
            T_base = np.clip(T_base, T_mean - 4.0, T_mean + 4.0)

            shade = generate_shade(G, W, B)
            model = AntForagingModel(T_base=T_base, shade=shade, run_id=run_id)
            model.run(steps=TIMESTEPS)
            metrics = model.compute_run_metrics()
            metrics["run_id"] = run_id
            metrics["site"] = name
            run_rows.append(metrics)

            if verbose and (run_id + 1) % 100 == 0:
                dom_counts = {}
                for r in run_rows:
                    d = r["dominant"]
                    dom_counts[d] = dom_counts.get(d, 0) + 1
                top = max(dom_counts, key=dom_counts.get)
                print(
                    f"       Run {run_id + 1:>3}/{n_runs}  "
                    f"leading: {top.replace('_', ' ')} ({dom_counts[top]})"
                )

        all_results[name] = pd.DataFrame(run_rows)

    return all_results
