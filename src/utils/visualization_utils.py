"""visualization.py — utility library for graph and table plotting and validation utilities."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    NCR_SITES,
    SPECIES_LIST,
    SPECIES_PARAMS,
    MIN_BIN_SIZE,
)
from thermal import thermal_multiplier
from model import AntForagingModel


# ═══════════════════════════════════════════════════════════════════════════
# SITE ORDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _build_site_temp_index() -> dict[str, dict]:
    """
    Returns a dict keyed by site name with 'toml_idx' (1-based position in
    sites.toml) and 'T' (T_base_mean), drawn from NCR_SITES (config.py).
    NCR_SITES preserves the sites.toml file order, so enumerate gives the
    correct 1–16 site numbers.
    """
    return {
        s["name"]: {"toml_idx": i + 1, "T": s["T_base_mean"]}
        for i, s in enumerate(NCR_SITES)
    }


def _temp_sorted_sites(site_names) -> list[str]:
    """
    Sort a collection of site names by T_base_mean descending
    (highest temperature first).
    """
    idx = _build_site_temp_index()
    return sorted(
        site_names,
        key=lambda s: idx.get(s, {}).get("T", 0.0),
        reverse=True,
    )


def _site_axis_label(site_name: str) -> str:
    """
    Return 'Site {N} ({T:.1f}°C)' where N is the 1-based index of the site
    in sites.toml (i.e. 1–16 in file order).
    Falls back to the raw name if the site is not found in NCR_SITES.
    """
    idx = _build_site_temp_index()
    info = idx.get(site_name)
    if info is None:
        return site_name
    return f"Site {info['toml_idx']} ({info['T']:.1f}°C)"


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION METRICS
# ═══════════════════════════════════════════════════════════════════════════


def site_dominance(df: pd.DataFrame) -> dict:
    """
    Section III-G — Site-level dominance summary

    Eq. (20) P_i, Eq. (21) Ŝ = argmax P_i, Eq. (23) Shannon entropy H.
    """

    n = len(df)

    cnt = df["dominant"].value_counts()

    P_i = {sp: cnt.get(sp, 0) / n for sp in SPECIES_LIST}

    S_hat = max(P_i, key=P_i.get)

    H = -sum(p * math.log(p) for p in P_i.values() if p > 1e-12)

    return {
        "P_i": P_i,
        "S_hat": S_hat,
        "H": H,
    }


def classification_accuracy(
    site_dominance_dict: dict[str, dict],
    ground_truth: dict[str, str],
) -> float:
    """
    Eq. (22): Accuracy = N_correct / N_sites

    Fraction of sites where predicted dominant species matches observation.
    """

    print()

    n_correct = sum(
        1
        for site, dom in site_dominance_dict.items()
        if ground_truth.get(site).replace("_", " ") == dom["S_hat"].replace("_", " ")
    )

    return n_correct / len(site_dominance_dict) if site_dominance_dict else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — THERMAL PERFORMANCE CURVES
# ═══════════════════════════════════════════════════════════════════════════


def plot_thermal_performance_curves(out_dir: str):
    """
    Section III-D.1 — Thermal performance curves (Eq. 4) for all species.
    """
    T_range = np.linspace(0, 50, 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    for sp in SPECIES_LIST:
        p = SPECIES_PARAMS[sp]

        f_v = [thermal_multiplier(sp, T) for T in T_range]

        ax.plot(
            T_range,
            f_v,
            lw=3,
            color=p["color"],
            label=p["label"],
        )

        # T_opt
        ax.axvline(
            p["T_opt"],
            color=p["color"],
            ls=":",
            lw=1.8,
            alpha=0.8,
        )

        # CT_max
        ax.axvline(
            p["CT_max"],
            color=p["color"],
            ls="--",
            lw=1.8,
            alpha=0.7,
        )

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Thermal Performance Multiplier  fᵢ(T)", fontsize=12)

    ax.set_title(
        "Species Thermal Performance Curves",
        fontsize=14,
        weight="bold",
    )

    ax.grid(alpha=0.25)

    ax.legend(
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "fig1_thermal_performance_curves.png",
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — FORAGING EFFICIENCY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════


def plot_foraging_efficiency_distributions(
    all_results: dict,
    out_dir: str,
):
    """
    Eq. (17) — Per-species foraging efficiency distributions across NCR sites.

    Sites sorted by descending T_base_mean (highest LST first).
    """

    # ──────────────────────────────────────────────────────────────────────
    # Site ordering — temperature-descending (highest LST first)
    # ──────────────────────────────────────────────────────────────────────

    ranked_sites = _temp_sorted_sites(all_results.keys())

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
    )

    axes = axes.flatten()

    # ──────────────────────────────────────────────────────────────────────
    # First pass: collect all data to determine shared y-axis maximum
    # ──────────────────────────────────────────────────────────────────────

    global_ymax = 0.0

    for sp in SPECIES_LIST:
        for site in ranked_sites:
            vals = all_results[site][f"E_{sp}"].values
            if len(vals) > 0:
                global_ymax = max(global_ymax, np.max(vals))

    # Add 5% headroom
    global_ymax *= 1.05

    # ──────────────────────────────────────────────────────────────────────
    # Second pass: plot with shared y-axis
    # ──────────────────────────────────────────────────────────────────────

    for idx, sp in enumerate(SPECIES_LIST):
        p = SPECIES_PARAMS[sp]
        ax = axes[idx]

        data = [all_results[site][f"E_{sp}"].values for site in ranked_sites]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2.0),
            whiskerprops=dict(color=p["color"], linewidth=1.2, alpha=0.7),
            capprops=dict(color=p["color"], linewidth=1.2, alpha=0.7),
            flierprops=dict(
                marker=".",
                color=p["color"],
                alpha=0.25,
                markersize=3,
            ),
            boxprops=dict(linewidth=0),
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(p["color"])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(ranked_sites) + 1))

        ax.set_xticklabels(
            [_site_axis_label(s) for s in ranked_sites],
            rotation=45,
            fontsize=8,
        )

        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")

        ax.set_ylabel("Foraging Efficiency  Eᵢ", fontsize=10)
        ax.set_xlabel("NCR Site", fontsize=10)

        ax.set_title(
            p["label"],
            fontsize=11,
            color=p["color"],
            weight="bold",
        )

        ax.set_ylim(0, global_ymax)
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle(
        "Foraging Efficiency Distributions Across NCR Sites",
        fontsize=14,
        weight="bold",
        y=1.01,
    )

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "fig2_foraging_efficiency_distributions.png",
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"    Saved: {path}")


def compute_mean_foraging_efficiency(all_results: dict) -> dict:
    """
    Eq. (17) — Mean foraging efficiency E_i pooled across sites and Monte Carlo runs.
    """
    means = {}

    for sp in SPECIES_LIST:
        p = SPECIES_PARAMS[sp]
        all_vals = []

        for site in all_results:
            vals = all_results[site][f"E_{sp}"].values
            all_vals.extend(vals)

        means[p["label"]] = np.mean(all_vals) if all_vals else np.nan

    return means


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — THERMAL STRESS EXPOSURE
# ═══════════════════════════════════════════════════════════════════════════


def plot_thermal_stress_exposure(
    all_results: dict,
    out_dir: str,
):
    """
    Fig. 3 — Mean proportion of simulation timesteps spent above thermal
    thresholds for each species, pooled across all NCR sites and Monte
    Carlo runs.

    Eq. (19): τ_stress = t(T>T_opt)/t_total, τ_critical = t(T≥CT_max)/t_total per species.
    """

    # ── collect per-species stress and critical arrays ─────────────────────
    stress_means, stress_sds = [], []
    critical_means, critical_sds = [], []
    species_labels = []

    for sp in SPECIES_LIST:
        p = SPECIES_PARAMS[sp]
        stress_vals, critical_vals = [], []

        for site_df in all_results.values():
            s_col = f"tau_stress_{sp}"
            c_col = f"tau_critical_{sp}"

            if s_col in site_df.columns:
                stress_vals.extend(site_df[s_col].dropna().tolist())
            if c_col in site_df.columns:
                critical_vals.extend(site_df[c_col].dropna().tolist())

        stress_means.append(np.mean(stress_vals) if stress_vals else 0.0)
        stress_sds.append(np.std(stress_vals, ddof=1) if len(stress_vals) > 1 else 0.0)
        critical_means.append(np.mean(critical_vals) if critical_vals else 0.0)
        critical_sds.append(
            np.std(critical_vals, ddof=1) if len(critical_vals) > 1 else 0.0
        )
        species_labels.append(p["label"])

    n_sp = len(SPECIES_LIST)
    x = np.arange(n_sp)
    width = 0.35

    # IEEE grayscale palette ──────────────────────────────────────────────
    DARK_GRAY = "#4d4d4d"
    LIGHT_GRAY = "#b3b3b3"

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_stress = ax.bar(
        x - width / 2,
        stress_means,
        width,
        yerr=stress_sds,
        label=r"Thermal stress ($T > T_{opt}$)",
        color=DARK_GRAY,
        edgecolor="black",
        linewidth=0.7,
        capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor="black"),
    )

    bars_critical = ax.bar(
        x + width / 2,
        critical_means,
        width,
        yerr=critical_sds,
        label=r"Critical exposure ($T \geq CT_{max}$)",
        color=LIGHT_GRAY,
        edgecolor="black",
        linewidth=0.7,
        capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(species_labels, fontsize=10)
    ax.set_ylabel("Proportion of timesteps", fontsize=11)
    ax.set_ylim(0, min(1.0, max(stress_means + critical_means) * 1.35 + 0.05))
    ax.set_title(
        "Mean Thermal Stress and Critical Exposure per Species",
        fontsize=13,
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bar values inside bars (avoids overlap with error bars)
    for bar in bars_stress:
        h = bar.get_height()
        if h > 0.015:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 0.5,
                f"{h:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )
    for bar in bars_critical:
        h = bar.get_height()
        if h > 0.015:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 0.5,
                f"{h:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#333333",
                fontweight="bold",
            )

    # Legend as a patch row below the species tick labels
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=DARK_GRAY,
            edgecolor="black",
            linewidth=0.7,
            label=r"Thermal stress ($T > T_{opt}$)",
        ),
        Patch(
            facecolor=LIGHT_GRAY,
            edgecolor="black",
            linewidth=0.7,
            label=r"Critical exposure ($T \geq CT_{max}$)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    path = os.path.join(out_dir, "fig3_thermal_stress_exposure.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def compute_mean_dominance_probability(all_results: dict) -> dict:
    """
    Eq. (20) — Mean dominance probability P_i at sites where species is predicted winner.
    """
    winner_probs: dict[str, list[float]] = {sp: [] for sp in SPECIES_LIST}

    for site_df in all_results.values():
        dom = site_dominance(site_df)
        winner = dom["S_hat"]
        winner_probs[winner].append(dom["P_i"][winner])

    result = {}
    for sp in SPECIES_LIST:
        probs = winner_probs[sp]
        if probs:
            p = SPECIES_PARAMS[sp]
            result[p["label"]] = np.mean(probs)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — DOMINANCE PROBABILITY HEATMAP
# ═══════════════════════════════════════════════════════════════════════════


def plot_dominance_heatmap(
    all_results: dict,
    ground_truth: dict,
    out_dir: str,
):
    """
    Eq. (20) — Species dominance probability P_i heatmap across NCR sites.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Sort sites by temperature-descending (highest LST first)
    # ──────────────────────────────────────────────────────────────────────

    ranked_sites = _temp_sorted_sites(all_results.keys())

    mat = np.zeros(
        (
            len(SPECIES_LIST),
            len(ranked_sites),
        )
    )

    for j, site in enumerate(ranked_sites):
        dom = site_dominance(all_results[site])

        for i, sp in enumerate(SPECIES_LIST):
            mat[i, j] = dom["P_i"][sp]

    fig, ax = plt.subplots(figsize=(16, 5.5))

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(ranked_sites)))

    ax.set_xticklabels(
        [_site_axis_label(s) for s in ranked_sites],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_yticks(range(len(SPECIES_LIST)))

    ax.set_yticklabels(
        [SPECIES_PARAMS[sp]["label"] for sp in SPECIES_LIST],
        fontsize=10,
    )

    ax.set_title(
        "Species Dominance Probability Across NCR Sites",
        fontsize=14,
        weight="bold",
    )

    # ──────────────────────────────────────────────────────────────────────
    # Ground-truth markers
    # ──────────────────────────────────────────────────────────────────────

    species_to_idx = {sp: i for i, sp in enumerate(SPECIES_LIST)}

    for j, site in enumerate(ranked_sites):
        actual = ground_truth.get(site)

        if actual in species_to_idx:
            i = species_to_idx[actual]

            ax.scatter(
                j,
                i,
                s=120,
                facecolors="none",
                edgecolors="#1a1a1a",
                linewidths=2.5,
            )

    cbar = fig.colorbar(im, ax=ax)

    cbar.set_label(
        "Dominance Probability  Pᵢ",
        fontsize=11,
    )

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "fig4_dominance_probability_heatmap.png",
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — SHANNON ENTROPY
# ═══════════════════════════════════════════════════════════════════════════


def plot_shannon_entropy(
    all_results: dict,
    out_dir: str,
):
    """
    Eq. (23) — Shannon dominance entropy H across sites (competitive uncertainty).
    """

    threshold = math.log(2)

    ranked = _temp_sorted_sites(all_results.keys())

    entropies = [site_dominance(all_results[s])["H"] for s in ranked]

    fig, ax = plt.subplots(figsize=(14, 5))

    site_labels = [_site_axis_label(s) for s in ranked]

    bars = ax.bar(
        site_labels,
        entropies,
        alpha=0.85,
    )

    ax.axhline(
        threshold,
        color="red",
        ls="--",
        lw=2,
        label=f"H = ln(2) ≈ {threshold:.3f}",
    )

    ax.set_ylabel(
        "Shannon Dominance Entropy  H",
        fontsize=11,
    )

    ax.set_xlabel(
        "NCR Site",
        fontsize=11,
    )

    ax.set_title(
        "Competitive Uncertainty Across NCR Sites",
        fontsize=14,
        weight="bold",
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")

    ax.legend(frameon=False)

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "fig5_shannon_entropy.png",
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — COLLECTIVE ORDER COLLAPSE
# ═══════════════════════════════════════════════════════════════════════════


def plot_collective_order(
    all_ref_records: list[dict],
    out_dir: str,
):
    """
    Eq. (18) — Mechanistic validation of collective order Ω vs temperature.

    Accepts pooled per-step records from one reference run per site,
    giving full thermal range coverage for Ω vs temperature binning.

    T_collapse detection is restricted to temperatures above T_opt to
    exclude the initialization phase (Ω is low at run-start when no
    trails exist yet, regardless of temperature) and to enforce the
    validity criterion: T_opt < T_collapse ≤ CT_max.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    axes = axes.flatten()

    # ──────────────────────────────────────────────────────────────────────
    # First pass: collect all binned Ω values to determine shared y-axis
    # ──────────────────────────────────────────────────────────────────────

    all_by = []

    for idx, sp in enumerate(SPECIES_LIST):
        p = SPECIES_PARAMS[sp]

        T_s = np.array([r.get(f"mean_T_{sp}", np.nan) for r in all_ref_records])

        Om_s = np.array([r[f"omega_{sp}"] for r in all_ref_records])

        N_s = np.array([r.get(f"n_active_{sp}", 0) for r in all_ref_records])

        MIN_ACTIVE = max(1, round(0.10 * 100))

        valid = ~np.isnan(T_s) & (N_s >= MIN_ACTIVE)
        T_v = T_s[valid]
        Om_v = Om_s[valid]

        if len(T_v) > 10:
            bins = np.linspace(T_v.min(), T_v.max(), 25)

            for bi in range(len(bins) - 1):
                mask = (T_v >= bins[bi]) & (T_v < bins[bi + 1])

                if mask.sum() >= MIN_BIN_SIZE:
                    all_by.append(Om_v[mask].mean())

    # Determine shared y-axis range from the pooled binned means
    if all_by:
        y_min_data = min(all_by)
        y_max_data = max(all_by)
        margin = (y_max_data - y_min_data) * 0.15 if y_max_data > y_min_data else 0.05
        global_ymin = max(0.0, y_min_data - margin)
        global_ymax = min(1.0, y_max_data + margin)
    else:
        global_ymin = 0.0
        global_ymax = 1.0

    # ──────────────────────────────────────────────────────────────────────
    # Second pass: plot with shared y-axis
    # ──────────────────────────────────────────────────────────────────────

    for idx, sp in enumerate(SPECIES_LIST):
        p = SPECIES_PARAMS[sp]
        ax = axes[idx]

        T_s = np.array([r.get(f"mean_T_{sp}", np.nan) for r in all_ref_records])

        Om_s = np.array([r[f"omega_{sp}"] for r in all_ref_records])

        N_s = np.array([r.get(f"n_active_{sp}", 0) for r in all_ref_records])

        MIN_ACTIVE = max(1, round(0.10 * 100))

        valid = ~np.isnan(T_s) & (N_s >= MIN_ACTIVE)
        T_v = T_s[valid]
        Om_v = Om_s[valid]

        bx = []
        by = []

        if len(T_v) > 10:
            bins = np.linspace(T_v.min(), T_v.max(), 25)

            for bi in range(len(bins) - 1):
                mask = (T_v >= bins[bi]) & (T_v < bins[bi + 1])

                if mask.sum() >= MIN_BIN_SIZE:
                    bx.append((bins[bi] + bins[bi + 1]) / 2)
                    by.append(Om_v[mask].mean())

            ax.plot(
                bx,
                by,
                lw=3,
                color=p["color"],
                label="Mean Ω",
            )

        # Ω = 0.5 threshold
        ax.axhline(
            0.5,
            color="red",
            ls="--",
            lw=1.5,
        )

        # T_opt boundary
        if "T_opt" in p:
            ax.axvline(
                p["T_opt"],
                color="green",
                ls=":",
                lw=2,
                label="T_opt",
            )

        # CT_max boundary
        ax.axvline(
            p["CT_max"],
            color="black",
            ls=":",
            lw=2,
            label="CT_max",
        )

        # T_collapse: first bin ABOVE T_opt where mean Ω < 0.5.
        T_col = None

        for bx_val, by_val in zip(bx, by):
            if bx_val > p["T_opt"] and by_val < 0.5:
                T_col = bx_val
                break

        if T_col is not None:
            ax.axvline(
                T_col,
                color="orange",
                ls="-.",
                lw=2,
                label="T_collapse",
            )

            title = f"{p['label']}\nTcollapse = {T_col:.1f}°C"

        else:
            title = f"{p['label']}\nNo Collapse Detected"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Collective Order Ω")
        ax.set_ylim(global_ymin, global_ymax)
        ax.grid(alpha=0.2)

    # Single global legend from last panel
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.suptitle(
        "Mechanistic Validation of Pheromone Network Collapse",
        fontsize=14,
        weight="bold",
        y=1.01,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    path = os.path.join(
        out_dir,
        "fig6_collective_order_validation.png",
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 1 — SITE VALIDATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════


def generate_site_summary_table(
    all_results: dict,
    ground_truth: dict,
    out_dir: str = ".",
):
    """
    Section III-G — Site validation table (Eq. 21–22, Eq. 23 entropy).
    """

    rows = []
    dom_dict = {}
    _site_idx = _build_site_temp_index()

    for site_name, df in all_results.items():
        dom = site_dominance(df)
        dom_dict[site_name] = dom

        predicted = dom["S_hat"]
        actual = ground_truth.get(site_name, "—")
        info = _site_idx.get(site_name, {})

        rows.append(
            {
                "Site No.": info.get("toml_idx", "—"),
                "Site": site_name,
                "Temp (°C)": round(info["T"], 1) if "T" in info else "—",
                "Observed": actual.replace("_", " "),
                "Predicted": predicted.replace("_", " "),
                "Entropy H": round(dom["H"], 3),
                "Match": "✓"
                if actual.replace("_", " ") == predicted.replace("_", " ")
                else "✗",
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by="Entropy H",
        ascending=False,
    )

    print("\n" + "═" * 90)
    print("  TABLE 1 — SITE-LEVEL VALIDATION SUMMARY")
    print("═" * 90)

    print(summary.to_string(index=False))

    acc = classification_accuracy(dom_dict, ground_truth)
    n_correct = int(acc * len(all_results))

    print(f"\n  Classification Accuracy: {n_correct}/{len(all_results)} = {acc:.3f}")

    print("\n" + "─" * 50)
    print("  Mean Foraging Efficiency (Eᵢ) Across All Sites")
    print("─" * 50)

    for sp in SPECIES_LIST:
        p = SPECIES_PARAMS[sp]
        all_vals = []
        for site in all_results:
            vals = all_results[site][f"E_{sp}"].values
            all_vals.extend(vals)
        mean_E = np.mean(all_vals) if all_vals else np.nan
        print(f"  {p['label']:<20s} E = {mean_E:.4f}")

    path = os.path.join(out_dir, "table1_site_validation_summary.csv")
    summary.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    return summary
