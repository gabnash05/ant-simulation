"""visualization.py — Publication-grade plotting and validation utilities."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    SPECIES_LIST,
    SPECIES_PARAMS,
)
from thermal import thermal_multiplier
from model import AntForagingModel


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def site_dominance(df: pd.DataFrame) -> dict:
    """
    Eq. (17): P_i = N_wins_i / N_runs
    Eq. (18): Ŝ = argmax P_i
    Eq. (20): H = −Σ P_i ln P_i
    """

    n = len(df)

    cnt = df["dominant"].value_counts()

    P_i = {
        sp: cnt.get(sp, 0) / n
        for sp in SPECIES_LIST
    }

    S_hat = max(P_i, key=P_i.get)

    H = -sum(
        p * math.log(p)
        for p in P_i.values()
        if p > 1e-12
    )

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
    Eq. (19): Accuracy = N_correct / N_sites
    """

    print()

    n_correct = sum(
        1
        for site, dom in site_dominance_dict.items()
        if ground_truth.get(site).replace("_", " ") == dom["S_hat"].replace("_", " ")
    )

    return (
        n_correct / len(site_dominance_dict)
        if site_dominance_dict
        else 0.0
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — THERMAL PERFORMANCE CURVES
# ═══════════════════════════════════════════════════════════════════════════

def plot_thermal_performance_curves(out_dir: str):
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
    Fig. 5 — Per-species foraging efficiency (Eq. 14) distributions
    across all NCR sites.

    Sites are sorted by descending Shannon entropy (Eq. 20), consistent
    with the ordering used in Figs. 2 and 3, so high-uncertainty
    transition zones appear on the left of every panel.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Site ordering — entropy-descending, same as Figs. 2 & 3
    # ──────────────────────────────────────────────────────────────────────

    ranked_sites = sorted(
        all_results.keys(),
        key=lambda s: site_dominance(all_results[s])["H"],
        reverse=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
    )

    axes = axes.flatten()

    for idx, sp in enumerate(SPECIES_LIST):
        p = SPECIES_PARAMS[sp]
        ax = axes[idx]

        data = [
            all_results[site][f"E_{sp}"].values
            for site in ranked_sites
        ]

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
            ranked_sites,
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

        ax.set_ylim(bottom=0)

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


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — DOMINANCE PROBABILITY HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_dominance_heatmap(
    all_results: dict,
    ground_truth: dict,
    out_dir: str,
):
    """
    Main Monte Carlo result figure.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Sort sites by entropy
    # ──────────────────────────────────────────────────────────────────────

    ranked_sites = sorted(
        all_results.keys(),
        key=lambda s: site_dominance(all_results[s])["H"],
        reverse=True,
    )

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
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(ranked_sites)))

    ax.set_xticklabels(
        ranked_sites,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_yticks(range(len(SPECIES_LIST)))

    ax.set_yticklabels(
        [
            SPECIES_PARAMS[sp]["label"]
            for sp in SPECIES_LIST
        ],
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

    species_to_idx = {
        sp: i
        for i, sp in enumerate(SPECIES_LIST)
    }

    for j, site in enumerate(ranked_sites):

        actual = ground_truth.get(site)

        if actual in species_to_idx:

            i = species_to_idx[actual]

            ax.scatter(
                j,
                i,
                s=120,
                facecolors="none",
                edgecolors="white",
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
        "fig3_dominance_probability_heatmap.png",
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
    Transition-zone uncertainty figure.
    """

    threshold = math.log(2)

    ranked = sorted(
        all_results.keys(),
        key=lambda s: site_dominance(all_results[s])["H"],
        reverse=True,
    )

    entropies = [
        site_dominance(all_results[s])["H"]
        for s in ranked
    ]

    fig, ax = plt.subplots(figsize=(14, 5))

    bars = ax.bar(
        ranked,
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

    # Annotate highest-entropy sites
    for i in range(min(3, len(ranked))):

        ax.text(
            i,
            entropies[i] + 0.03,
            "Transition Zone",
            ha="center",
            fontsize=9,
            rotation=90,
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
        label.set_ha('right')
        label.set_rotation_mode('anchor')

    ax.legend(frameon=False)

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "fig4_shannon_entropy.png",
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
    ref_model: AntForagingModel,
    out_dir: str,
):
    """
    Mechanistic validation figure with a single global legend.
    """

    collapse = ref_model.collapse_temperatures()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    axes = axes.flatten()

    for idx, sp in enumerate(SPECIES_LIST):

        p = SPECIES_PARAMS[sp]
        ax = axes[idx]

        T_s = np.array(
            [
                r.get(f"mean_T_{sp}", np.nan)
                for r in ref_model.records
            ]
        )

        Om_s = np.array(
            [
                r[f"omega_{sp}"]
                for r in ref_model.records
            ]
        )

        valid = ~np.isnan(T_s)
        T_v = T_s[valid]
        Om_v = Om_s[valid]

        if len(T_v) > 10:

            bins = np.linspace(
                T_v.min(),
                T_v.max(),
                25,
            )

            bx = []
            by = []

            for bi in range(len(bins) - 1):

                mask = (
                    (T_v >= bins[bi])
                    &
                    (T_v < bins[bi + 1])
                )

                if mask.sum() > 0:

                    bx.append(
                        (bins[bi] + bins[bi + 1]) / 2
                    )

                    by.append(
                        Om_v[mask].mean()
                    )

            ax.plot(
                bx,
                by,
                lw=3,
                color=p["color"],
                label="Mean Ω",
            )

        # Ω threshold
        ax.axhline(
            0.5,
            color="red",
            ls="--",
            lw=1.5,
            label="Ω = 0.5 Threshold",
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

        # Collapse temperature
        T_col = collapse.get(sp)

        if T_col is not None:

            ax.axvline(
                T_col,
                color="orange",
                ls="-.",
                lw=2,
                label="T_collapse",
            )

            title = (
                f"{p['label']}\n"
                f"Tcollapse = {T_col:.1f}°C"
            )

        else:

            title = (
                f"{p['label']}\n"
                f"No Collapse Detected"
            )

        ax.set_title(
            title,
            fontsize=10,
        )

        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Collective Order Ω")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

    # 1. GRAB HANDLES AND LABELS FROM THE LAST PLOTTED PANEL
    handles, labels = ax.get_legend_handles_labels()

    # 2. CREATE A SINGLE GLOBAL LEGEND AT THE BOTTOM
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.03)  # Positions it safely below the plots
    )

    plt.suptitle(
        "Mechanistic Validation of Pheromone Network Collapse",
        fontsize=14,
        weight="bold",
        y=1.01,
    )

    # tight_layout adjusted slightly to leave breathing room for the bottom legend
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    path = os.path.join(
        out_dir,
        "fig5_collective_order_validation.png",
    )

    # bbox_inches="tight" ensures the new bottom legend isn't clipped out of the saved PNG
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
    Main validation table.
    """

    rows = []

    dom_dict = {}

    for site_name, df in all_results.items():

        dom = site_dominance(df)

        dom_dict[site_name] = dom

        predicted = dom["S_hat"]

        actual = ground_truth.get(site_name, "—")

        rows.append(
            {
                "Site": site_name,
                "Observed": actual.replace("_", " "),
                "Predicted": predicted.replace("_", " "),
                "Entropy H": round(dom["H"], 3),
                "Match": "✓" if actual.replace("_", " ") == predicted.replace("_", " ") else "✗",
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

    acc = classification_accuracy(
        dom_dict,
        ground_truth,
    )

    n_correct = int(acc * len(all_results))

    print(
        f"\n  Classification Accuracy:"
        f" {n_correct}/{len(all_results)} = {acc:.3f}"
    )

    path = os.path.join(
        out_dir,
        "table1_site_validation_summary.csv",
    )

    summary.to_csv(
        path,
        index=False,
    )

    print(f"\n  Saved: {path}")

    return summary
