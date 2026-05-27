"""visualization.py — All plotting functions and visual mode runner."""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    SPECIES_LIST,
    SPECIES_PARAMS,
)
from thermal import thermal_multiplier
from model import AntForagingModel


# ── Eqs. (17)–(20): site-level dominance, accuracy, entropy ──────────────


def site_dominance(df: pd.DataFrame) -> dict:
    """
    Eq. (17): P_i = N_wins_i / 500
    Eq. (18): Ŝ = argmax P_i
    Eq. (20): H = −Σ P_i ln P_i   (Shannon dominance entropy)
    """
    n = len(df)
    cnt = df["dominant"].value_counts()
    P_i = {sp: cnt.get(sp, 0) / n for sp in SPECIES_LIST}
    S_hat = max(P_i, key=P_i.get)
    H = -sum(p * math.log(p) for p in P_i.values() if p > 1e-12)
    return {"P_i": P_i, "S_hat": S_hat, "H": H}


def classification_accuracy(
    site_dominance_dict: dict[str, dict],
    ground_truth: dict[str, str],
) -> float:
    """Eq. (19): Accuracy = N_correct / N_sites."""
    n_correct = sum(
        1
        for site, dom in site_dominance_dict.items()
        if ground_truth.get(site) == dom["S_hat"]
    )
    return n_correct / len(site_dominance_dict) if site_dominance_dict else 0.0


# ── Graph 1: Thermal Performance Curves ──────────────────────────────────


def plot_thermal_performance_curves(out_dir: str):
    T_range = np.linspace(0, 50, 500)
    fig, ax = plt.subplots(figsize=(10, 6))
    for sp in SPECIES_LIST:
        p = SPECIES_PARAMS[sp]
        f_v = [thermal_multiplier(sp, T) for T in T_range]
        ax.plot(T_range, f_v, color=p["color"], lw=2.5, label=p["label"])
        ax.axvline(p["T_opt"], color=p["color"], ls=":", alpha=0.55, lw=1)
        ax.axvline(p["CT_max"], color=p["color"], ls="--", alpha=0.45, lw=1)
        ax.axvline(p["CT_min"], color=p["color"], ls="--", alpha=0.45, lw=1)
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Thermal Performance Multiplier  f_i(T)", fontsize=12)
    ax.set_title("Gaussian Thermal Performance Curves — All Four Species", fontsize=13)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_thermal_performance_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Graph 2: Foraging Efficiency Boxplots per Site ────────────────────────


def plot_foraging_efficiency_boxplots(all_results: dict, out_dir: str):
    n_sites = len(all_results)
    ncols = 4
    nrows = math.ceil(n_sites / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4.5), sharey=False)
    axes = np.array(axes).flatten()

    for idx, (site_name, df) in enumerate(all_results.items()):
        ax = axes[idx]
        data = [df[f"E_{sp}"].values for sp in SPECIES_LIST]
        labels = [SPECIES_PARAMS[sp]["label"] for sp in SPECIES_LIST]
        colors = [SPECIES_PARAMS[sp]["color"] for sp in SPECIES_LIST]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            labels=labels,
            medianprops=dict(color="black", lw=2),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        dom = site_dominance(df)
        ax.set_title(
            f"{site_name}\nŜ = {dom['S_hat'].replace('_', ' ')}  H={dom['H']:.2f}",
            fontsize=9,
        )
        ax.set_ylabel("E_i" if idx % ncols == 0 else "")
        ax.tick_params(axis="x", rotation=40, labelsize=7)

    for ax in axes[n_sites:]:
        ax.set_visible(False)

    plt.suptitle(
        "Foraging Efficiency  E_i  Distribution across NCR Sites (500 Monte Carlo Runs)",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_foraging_efficiency_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Graph 3: Species Dominance Probability Heatmap ───────────────────────


def plot_dominance_heatmap(all_results: dict, out_dir: str):
    site_names = list(all_results.keys())
    mat = np.zeros((len(SPECIES_LIST), len(site_names)))

    for j, name in enumerate(site_names):
        dom = site_dominance(all_results[name])
        for i, sp in enumerate(SPECIES_LIST):
            mat[i, j] = dom["P_i"].get(sp, 0.0)

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(site_names)))
    ax.set_xticklabels(site_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(SPECIES_LIST)))
    ax.set_yticklabels(
        [SPECIES_PARAMS[sp]["label"] for sp in SPECIES_LIST], fontsize=10
    )
    ax.set_title("Species Dominance Probability  P_i  Across 16 NCR Sites", fontsize=13)
    fig.colorbar(im, ax=ax, label="P_i")
    # Annotate cells
    for i in range(len(SPECIES_LIST)):
        for j in range(len(site_names)):
            val = mat[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if val < 0.7 else "white",
            )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_dominance_probability_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Graph 4: Shannon Entropy per Site ────────────────────────────────────


def plot_shannon_entropy(all_results: dict, out_dir: str):
    site_names = list(all_results.keys())
    entropies = [site_dominance(all_results[n])["H"] for n in site_names]
    threshold = math.log(2)  # H = ln(2) ≈ 0.693  transition threshold
    colors = ["#E74C3C" if h > threshold else "#2980B9" for h in entropies]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(site_names, entropies, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(
        threshold,
        color="orange",
        ls="--",
        lw=2,
        label=f"H = ln(2) ≈ {threshold:.3f}  transition threshold",
    )
    ax.set_xlabel("NCR Site", fontsize=11)
    ax.set_ylabel("Shannon Dominance Entropy  H", fontsize=11)
    ax.set_title(
        "Site-Level Competitive Uncertainty — Red: high-uncertainty transition zones",
        fontsize=13,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_shannon_entropy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Graph 5: Collective Order Ω vs Temperature (mechanistic validation) ──


def plot_collective_order(ref_model: AntForagingModel, out_dir: str):
    collapse = ref_model.collapse_temperatures()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, sp in enumerate(SPECIES_LIST):
        p = SPECIES_PARAMS[sp]
        ax = axes[idx]
        T_s = np.array([r.get(f"mean_T_{sp}", np.nan) for r in ref_model.records])
        Om_s = np.array([r[f"omega_{sp}"] for r in ref_model.records])

        valid = ~np.isnan(T_s)
        ax.scatter(
            T_s[valid],
            Om_s[valid],
            alpha=0.25,
            s=8,
            color=p["color"],
            label="Ω per step",
        )

        # Binned mean curve
        T_v = T_s[valid]
        Om_v = Om_s[valid]
        if len(T_v) > 10:
            bins = np.linspace(T_v.min(), T_v.max(), 25)
            bx, by = [], []
            for bi in range(len(bins) - 1):
                mask = (T_v >= bins[bi]) & (T_v < bins[bi + 1])
                if mask.sum() > 0:
                    bx.append((bins[bi] + bins[bi + 1]) / 2)
                    by.append(Om_v[mask].mean())
            ax.plot(bx, by, "k-o", lw=2, ms=4, label="Ω binned mean")

        ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Ω = 0.5")
        ax.axvline(
            p["T_opt"], color="green", ls=":", lw=1.5, label=f"T_opt = {p['T_opt']}°C"
        )
        ax.axvline(
            p["CT_max"],
            color="darkred",
            ls="--",
            lw=1.5,
            label=f"CT_max = {p['CT_max']}°C",
        )

        T_col = collapse.get(sp)
        if T_col is not None:
            ok = p["T_opt"] < T_col <= p["CT_max"]
            label = f"T_collapse = {T_col:.1f}°C  {'✓ VALID' if ok else '✗ INVALID'}"
            ax.axvline(T_col, color="orange", ls="-.", lw=2, label=label)
            ax.set_title(f"{p['label']}  —  {label}", fontsize=10)
        else:
            ax.set_title(
                f"{p['label']}  —  Ω stable (no collapse detected)", fontsize=10
            )

        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Collective Order Ω")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    plt.suptitle(
        "Collective Order Ω vs Temperature — Mechanistic Validation",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_collective_order_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ── Graph 6: Thermal Exposure Distribution per Species ───────────────────


def plot_thermal_exposure(all_results: dict, out_dir: str):
    records = []

    for _, df in all_results.items():
        for sp in SPECIES_LIST:
            records.extend([row[f"tau_stress_{sp}"] for _, row in df.iterrows()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stress_data = [
        np.concatenate([all_results[s][f"tau_stress_{sp}"].values for s in all_results])
        for sp in SPECIES_LIST
    ]

    critical_data = [
        np.concatenate(
            [all_results[s][f"tau_critical_{sp}"].values for s in all_results]
        )
        for sp in SPECIES_LIST
    ]

    for ax, data, title in zip(
        axes, [stress_data, critical_data], ["τ_stress", "τ_critical"]
    ):
        vp = ax.violinplot(data, showmeans=True, showextrema=False)

        for body, sp in zip(vp["bodies"], SPECIES_LIST):
            body.set_facecolor(SPECIES_PARAMS[sp]["color"])
            body.set_alpha(0.75)

        ax.set_xticks(range(1, len(SPECIES_LIST) + 1))
        ax.set_xticklabels(
            [SPECIES_PARAMS[sp]["label"] for sp in SPECIES_LIST], rotation=20
        )

        ax.set_ylabel("Proportion")
        ax.set_title(title)

    plt.tight_layout()

    path = os.path.join(out_dir, "fig6_thermal_exposure.png")

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"    Saved: {path}")


# ── Table 1: Site-Level Summary ───────────────────────────────────────────


def generate_site_summary_table(
    all_results: dict,
    ground_truth: dict = None,
    out_dir: str = ".",
):
    rows = []
    dom_dict = {}
    for site_name, df in all_results.items():
        dom = site_dominance(df)
        dom_dict[site_name] = dom
        row = {
            "Site": site_name,
            "Predicted (Ŝ)": dom["S_hat"].replace("_", " "),
            "H": round(dom["H"], 3),
        }
        for sp in SPECIES_LIST:
            row[f"P({SPECIES_PARAMS[sp]['label']})"] = round(dom["P_i"].get(sp, 0.0), 3)
        if ground_truth:
            actual = ground_truth.get(site_name, "—")
            row["Actual"] = actual
            row["Correct?"] = "✓" if actual == dom["S_hat"] else "✗"
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Terminal output
    print("\n" + "═" * 90)
    print("  TABLE: SITE-LEVEL SPECIES DOMINANCE SUMMARY")
    print("═" * 90)
    print(summary.to_string(index=False))

    # Classification accuracy
    if ground_truth:
        acc = classification_accuracy(dom_dict, ground_truth)
        n_c = int(acc * len(all_results))
        print(
            f"\n  Eq.(19) Classification Accuracy: {n_c}/{len(all_results)} = {acc:.3f}"
        )

    path = os.path.join(out_dir, "table1_site_summary.csv")
    summary.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return summary


# ── Table 2: Mechanistic Validation Summary ───────────────────────────────


def generate_mechanistic_table(
    collapse_temps: dict,
    out_dir: str = ".",
):
    rows = []
    print("\n" + "═" * 75)
    print("  TABLE: MECHANISTIC VALIDATION — T_collapse vs T_opt and CT_max")
    print("═" * 75)
    hdr = f"  {'Species':<22} {'T_opt':>7} {'T_collapse':>12} {'CT_max':>8} {'Criterion':>16}"
    print(hdr)
    print("  " + "─" * 68)

    for sp in SPECIES_LIST:
        p = SPECIES_PARAMS[sp]
        T_col = collapse_temps.get(sp)
        if T_col is not None:
            ok = p["T_opt"] < T_col <= p["CT_max"]
            crit = "✓ PASS" if ok else "✗ FAIL"
            print(
                f"  {p['label']:<22} {p['T_opt']:>7.1f} {T_col:>12.2f}"
                f" {p['CT_max']:>8.1f} {crit:>16}"
            )
            rows.append(
                {
                    "Species": p["label"],
                    "T_opt (°C)": p["T_opt"],
                    "T_collapse": round(T_col, 2),
                    "CT_max (°C)": p["CT_max"],
                    "Valid": crit,
                }
            )
        else:
            print(
                f"  {p['label']:<22} {p['T_opt']:>7.1f} {'STABLE':>12}"
                f" {p['CT_max']:>8.1f} {'✓ STABLE':>16}"
            )
            rows.append(
                {
                    "Species": p["label"],
                    "T_opt (°C)": p["T_opt"],
                    "T_collapse": "STABLE",
                    "CT_max (°C)": p["CT_max"],
                    "Valid": "✓ STABLE",
                }
            )

    mech_df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "table2_mechanistic_validation.csv")
    mech_df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")
    return mech_df
