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
import math
import os
import random
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL PARAMETERS & CONFIGURATION  (loaded from TOML)
# ══════════════════════════════════════════════════════════════════════════

_CFG_DIR = Path(__file__).parent / "config"


def _load_toml(filename: str) -> dict:
    with open(_CFG_DIR / filename, "rb") as _f:
        return tomllib.load(_f)


# ─── species.toml → SPECIES_PARAMS ────────────────────────────────────────
def _build_species_params(raw: dict) -> dict:
    """Derive v_max_grid (cells/step) from v_max_mm_s at load-time."""
    out = {}
    for sp, vals in raw.items():
        entry = dict(vals)
        entry["v_max_grid"] = entry["v_max_mm_s"] * 60 / 1000
        out[sp] = entry
    return out

SPECIES_PARAMS = _build_species_params(_load_toml("species.toml"))
SPECIES_LIST   = list(SPECIES_PARAMS.keys())
SPECIES_INDEX  = {sp: i for i, sp in enumerate(SPECIES_LIST)}

# ─── simulation.toml → global constants ───────────────────────────────────
_sim = _load_toml("simulation.toml")

_comm           = _sim["communication"]
BETA_0          = _comm["BETA_0"]
T_REF           = _comm["T_REF"]
C_THRESH        = _comm["C_THRESH"]
N_NONLIN        = _comm["N_NONLIN"]
P_DROP          = _comm["P_DROP"]
K_ATTRACT       = _comm["K_ATTRACT"]
B_REC           = _comm["B_REC"]
P_TRICKLE       = _comm["P_TRICKLE"]
N_ACT_MIN       = _comm["N_ACT_MIN"]

_grid           = _sim["grid"]
GRID_W          = _grid["GRID_W"]
GRID_H          = _grid["GRID_H"]
TIMESTEPS       = _grid["TIMESTEPS"]
DT              = _grid["DT"]
A_DIURNAL       = _grid["A_DIURNAL"]
FOOD_POS        = tuple(_grid["FOOD_POS"])
FOOD_RADIUS     = _grid["FOOD_RADIUS"]
DELTA_THETA     = math.radians(_grid["DELTA_THETA_DEG"])

_col            = _sim["colony"]
N_COLONY        = _col["N_COLONY"]
N_SCOUTS        = _col["N_SCOUTS"]

N_MC_RUNS       = _sim["monte_carlo"]["N_MC_RUNS"]

NEST_POSITIONS  = {
    sp: tuple(coords)
    for sp, coords in _sim["nest_positions"].items()
}

# ─── sites.toml → NCR_SITES ───────────────────────────────────────────────
NCR_SITES = _load_toml("sites.toml")["site"]


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — THERMAL PERFORMANCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def sigma(sp: str) -> float:
    """Eq. (6): Thermal breadth σ_i = (CT_max - CT_min) / 6."""
    p = SPECIES_PARAMS[sp]
    return (p["CT_max"] - p["CT_min"]) / 6.0


def thermal_multiplier(sp: str, T: float) -> float:
    """Eq. (5): Gaussian thermal performance multiplier f_i(T) ∈ [0, 1]."""
    p = SPECIES_PARAMS[sp]
    sig = sigma(sp)
    return math.exp(-((T - p["T_opt"]) ** 2) / (2.0 * sig ** 2))


def effective_velocity(sp: str, T: float) -> float:
    """Eq. (4): v_i(T) = v_max,i · f_i(T)  [cells / timestep]."""
    return SPECIES_PARAMS[sp]["v_max_grid"] * thermal_multiplier(sp, T)


def species_decay_rate(sp: str, T: float) -> float:
    """Eq. (1): α_i(T) = β_0 · κ_i · (T + T_ref)."""
    p = SPECIES_PARAMS[sp]
    return BETA_0 * p["kappa"] * (T + T_REF)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPATIAL ENVIRONMENT (NumPy layers)
# ══════════════════════════════════════════════════════════════════════════

class EnvironmentLayers:
    """
    Manages:
      • T_base(x,y)    — static base temperature from LST dataset
      • s(x,y)         — static shade coefficient ∈ [0, 1]
      • T_field(x,y,t) — dynamic thermal field (updated each step)
      • φ_k(x,y)       — 4 pheromone layers (one per species, chemically isolated)
    """

    def __init__(self, T_base: np.ndarray, shade: np.ndarray):
        assert T_base.shape == (GRID_W, GRID_H), "T_base must be (100, 100)"
        assert shade.shape  == (GRID_W, GRID_H), "shade must be (100, 100)"
        self.T_base  = T_base.astype(np.float64)
        self.shade   = shade.astype(np.float64)
        self.T_field = T_base.copy().astype(np.float64)
        # 4 pheromone layers, initialized to zero
        self.phi     = np.zeros((4, GRID_W, GRID_H), dtype=np.float64)

    # ── Eq. (3): diurnal temperature update ─────────────────────────────
    def update_temperature(self, t: int):
        factor = A_DIURNAL * (1.0 - self.shade) * math.sin(math.pi * t / TIMESTEPS)
        self.T_field = self.T_base + factor

    # ── Eq. (2): vectorized per-species pheromone decay ─────────────────
    def update_pheromones(self):
        """
        φ_k(x,y,t+1) = φ_k(x,y,t) · exp(−α_i(T))
        Uses mean grid temperature per species layer.
        """
        mean_T = float(self.T_field.mean())
        for idx, sp in enumerate(SPECIES_LIST):
            alpha         = species_decay_rate(sp, mean_T)
            decay_factor  = math.exp(-alpha)
            self.phi[idx] *= decay_factor  # vectorized NumPy op

    # ── Cell-level accessors ─────────────────────────────────────────────
    def _clip(self, x: int, y: int):
        return max(0, min(GRID_W - 1, x)), max(0, min(GRID_H - 1, y))

    def get_T(self, x: int, y: int) -> float:
        cx, cy = self._clip(x, y)
        return float(self.T_field[cx, cy])

    def get_phi(self, sp_idx: int, x: int, y: int) -> float:
        cx, cy = self._clip(x, y)
        return float(self.phi[sp_idx, cx, cy])

    def deposit_phi(self, sp_idx: int, x: int, y: int, amount: float):
        cx, cy = self._clip(x, y)
        self.phi[sp_idx, cx, cy] += amount


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANT AGENT (off-lattice, state machine)
# ══════════════════════════════════════════════════════════════════════════

# Agent state constants
INACTIVE      = 0
SEARCHING     = 1
RETURNING     = 2
INCAPACITATED = 3


class AntAgent:
    """
    Off-lattice ant agent.
    • Continuous (x, y) coordinates; floor-mapped to discrete grid cells.
    • State machine: INACTIVE → SEARCHING → RETURNING → (loop)
    • Incapacitated when local T ≥ CT_max (Eq. 8).
    """

    __slots__ = [
        "uid", "sp", "sp_idx", "nest_x", "nest_y", "model",
        "x", "y", "theta", "state",
        "trips", "steps_active", "steps_above_topt", "steps_at_ctmax",
        "trail_steps", "chose_pheromone",
    ]

    def __init__(self, uid: int, sp: str, sp_idx: int, nest: tuple, model):
        self.uid     = uid
        self.sp      = sp
        self.sp_idx  = sp_idx
        self.nest_x  = float(nest[0])
        self.nest_y  = float(nest[1])
        self.model   = model

        self.x      = float(nest[0])
        self.y      = float(nest[1])
        self.theta  = random.uniform(0.0, 2.0 * math.pi)
        self.state  = INACTIVE

        # Telemetry accumulators
        self.trips          = 0
        self.steps_active   = 0
        self.steps_above_topt = 0
        self.steps_at_ctmax   = 0
        self.trail_steps    = 0
        self.chose_pheromone = False

    # ── helpers ──────────────────────────────────────────────────────────

    def _cell(self):
        return int(math.floor(self.x)), int(math.floor(self.y))

    def _local_T(self) -> float:
        cx, cy = self._cell()
        return self.model.env.get_T(cx, cy)

    def _reflect(self, nx: float, ny: float) -> tuple:
        """Wall reflection: reverse the affected heading component."""
        if nx < 0:
            nx = -nx
            self.theta = math.pi - self.theta
        elif nx >= GRID_W:
            nx = 2.0 * GRID_W - nx - 1e-9
            self.theta = math.pi - self.theta
        if ny < 0:
            ny = -ny
            self.theta = -self.theta
        elif ny >= GRID_H:
            ny = 2.0 * GRID_H - ny - 1e-9
            self.theta = -self.theta
        return nx, ny

    # ── forward-projected pheromone sensing ──────────────────────────────

    def _sense_candidates(self, v: float) -> list:
        """
        Project 3 candidate cells at R_sense = v_i(T) ahead, at angles
        θ − Δθ, θ, θ + Δθ. Returns [(phi_val, angle_offset), ...].
        """
        candidates = []
        for dθ in (-DELTA_THETA, 0.0, DELTA_THETA):
            ang = self.theta + dθ
            px  = self.x + v * math.cos(ang)
            py  = self.y + v * math.sin(ang)
            cx  = int(math.floor(max(0.0, min(GRID_W - 1, px))))
            cy  = int(math.floor(max(0.0, min(GRID_H - 1, py))))
            phi = self.model.env.get_phi(self.sp_idx, cx, cy)
            candidates.append((phi, dθ))
        return candidates   # [(phi_left, -Δθ), (phi_center, 0), (phi_right, +Δθ)]

    # ── Eq. (11): stochastic trail selection ─────────────────────────────

    def _stochastic_choice(self, candidates: list) -> float:
        """
        P(k) = (K + C_k)^n / Σ_j (K + C_j)^n
        Returns the angle offset of the chosen candidate.
        """
        weights = [(K_ATTRACT + phi) ** N_NONLIN for phi, _ in candidates]
        total   = sum(weights)
        probs   = [w / total for w in weights]
        r, cum  = random.random(), 0.0
        for (phi, dθ), p in zip(candidates, probs):
            cum += p
            if r <= cum:
                return dθ
        return candidates[-1][1]

    # ── main per-timestep step ────────────────────────────────────────────

    def step(self):
        if self.state == INACTIVE:
            return

        T_local = self._local_T()
        p       = SPECIES_PARAMS[self.sp]

        # Eq. (8): incapacitation if T ≥ CT_max
        if T_local >= p["CT_max"]:
            self.state = INCAPACITATED
            return
        # Recover from incapacitation if temperature has dropped
        if self.state == INCAPACITATED:
            self.state = SEARCHING

        self.steps_active += 1
        self.chose_pheromone = False

        # Thermal exposure telemetry (Eq. 16)
        if T_local > p["T_opt"]:
            self.steps_above_topt += 1
        if T_local >= p["CT_max"]:
            self.steps_at_ctmax += 1

        v = effective_velocity(self.sp, T_local)
        if v < 1e-9:
            return  # thermally frozen

        if self.state == SEARCHING:
            self._search_step(v)
        elif self.state == RETURNING:
            self._return_step(v)

    # ── SEARCHING behavior ───────────────────────────────────────────────

    def _search_step(self, v: float):
        candidates = self._sense_candidates(v)

        # Eq. (9): probabilistic detection — Michaelis-Menten
        max_phi  = max(c[0] for c in candidates)
        P_detect = max_phi / (max_phi + C_THRESH)

        if max_phi > 0 and random.random() < P_detect:
            dθ = self._stochastic_choice(candidates)
            self.chose_pheromone = True
            self.trail_steps += 1
        else:
            # Unguided: biased random walk with small angular noise
            dθ = random.choice((-DELTA_THETA, 0.0, DELTA_THETA))
            dθ += random.gauss(0.0, math.radians(8.0))

        self.theta += dθ
        nx = self.x + v * math.cos(self.theta)
        ny = self.y + v * math.sin(self.theta)
        self.x, self.y = self._reflect(nx, ny)   # Eq. (7) + boundary

        # Food contact check
        cx, cy = self._cell()
        if abs(cx - FOOD_POS[0]) <= FOOD_RADIUS and abs(cy - FOOD_POS[1]) <= FOOD_RADIUS:
            self.state = RETURNING
            dx = self.nest_x - self.x
            dy = self.nest_y - self.y
            self.theta = math.atan2(dy, dx)

    # ── RETURNING behavior ───────────────────────────────────────────────

    def _return_step(self, v: float):
        # Eq. (10): pheromone deposition
        cx, cy = self._cell()
        self.model.env.deposit_phi(self.sp_idx, cx, cy, P_DROP)

        # Directed movement toward nest
        dx   = self.nest_x - self.x
        dy   = self.nest_y - self.y
        dist = math.hypot(dx, dy)

        if dist <= v:
            # Arrived at nest
            self.x, self.y = self.nest_x, self.nest_y
            self.state     = SEARCHING
            self.theta     = random.uniform(0.0, 2.0 * math.pi)
            self.trips    += 1
            # Pheromone-driven recruitment batch (Section E.4)
            self.model.recruit(self.sp, B_REC)
        else:
            self.theta = math.atan2(dy, dx)
            nx = self.x + v * math.cos(self.theta)
            ny = self.y + v * math.sin(self.theta)
            self.x, self.y = self._reflect(nx, ny)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — ANT FORAGING MODEL
# ══════════════════════════════════════════════════════════════════════════

class AntForagingModel:
    """
    Main model class.
    Manages: execution clock, agent activation, environment updates,
    recruitment logic, and per-step data collection.
    """

    def __init__(self, T_base: np.ndarray, shade: np.ndarray, run_id: int = 0):
        self.run_id   = run_id
        self.step_num = 0
        self.env      = EnvironmentLayers(T_base, shade)

        # Agent lists indexed by species name
        self.agents: dict[str, list[AntAgent]] = {sp: [] for sp in SPECIES_LIST}
        self._uid = 0

        # Initialise colonies
        for sp_idx, sp in enumerate(SPECIES_LIST):
            nest = NEST_POSITIONS[sp]
            for _ in range(N_COLONY):
                a = AntAgent(self._uid, sp, sp_idx, nest, self)
                self.agents[sp].append(a)
                self._uid += 1
            # Activate initial scouts
            for a in self.agents[sp][:N_SCOUTS]:
                a.state = SEARCHING
                a.theta = random.uniform(0.0, 2.0 * math.pi)

        # Per-step metric records: list of dicts
        self.records: list[dict] = []

    # ── recruitment ──────────────────────────────────────────────────────

    def recruit(self, sp: str, n: int):
        """Activate up to n INACTIVE agents from species sp."""
        count = 0
        for a in self.agents[sp]:
            if count >= n:
                break
            if a.state == INACTIVE:
                a.state = SEARCHING
                a.theta = random.uniform(0.0, 2.0 * math.pi)
                count  += 1

    def _trickle_release(self):
        """
        Section E.4 — stochastic trickle: if N_active < 15, release 1
        backup agent with P_trickle = 0.2 per timestep.
        """
        for sp in SPECIES_LIST:
            n_active = sum(
                1 for a in self.agents[sp]
                if a.state in (SEARCHING, RETURNING)
            )
            if n_active < N_ACT_MIN and random.random() < P_TRICKLE:
                self.recruit(sp, 1)

    # ── main simulation step ─────────────────────────────────────────────

    def step(self):
        self.step_num += 1
        t = self.step_num

        # Environment updates
        self.env.update_temperature(t)
        self.env.update_pheromones()

        # Trickle release check
        self._trickle_release()

        # Activate and shuffle all agents
        all_agents: list[AntAgent] = []
        for sp in SPECIES_LIST:
            all_agents.extend(self.agents[sp])
        random.shuffle(all_agents)
        for a in all_agents:
            a.step()

        # ── Per-step data collection ─────────────────────────────────────
        rec = {"step": t}
        for sp in SPECIES_LIST:
            active = [a for a in self.agents[sp] if a.state in (SEARCHING, RETURNING)]
            trail  = [a for a in active if a.chose_pheromone]
            n_a    = len(active)
            n_t    = len(trail)
            # Eq. (15): Ω = N_trail / N_active
            omega  = n_t / n_a if n_a > 0 else 0.0
            rec[f"n_active_{sp}"] = n_a
            rec[f"omega_{sp}"]    = omega
            # Mean local temperature across active agents
            if active:
                mean_T = sum(
                    self.env.get_T(int(math.floor(a.x)), int(math.floor(a.y)))
                    for a in active
                ) / n_a
            else:
                mean_T = float("nan")
            rec[f"mean_T_{sp}"] = mean_T
        self.records.append(rec)

    def run(self, steps: int = TIMESTEPS):
        """Run the model for the given number of timesteps."""
        for _ in range(steps):
            self.step()

    # ── per-run aggregate metrics ────────────────────────────────────────

    def compute_run_metrics(self) -> dict:
        """
        Returns dict of per-species Ei, Ω, τ_stress, τ_critical,
        plus the run-level dominant species (Eq. 14–16).
        """
        metrics = {}
        for sp in SPECIES_LIST:
            # Eq. (14): E_i = N_trips,i / N_colony,i
            total_trips = sum(a.trips for a in self.agents[sp])
            E_i         = total_trips / N_COLONY
            metrics[f"E_{sp}"] = E_i

            # Eq. (15): Ω mean across timesteps
            omega_vals           = [r[f"omega_{sp}"] for r in self.records]
            metrics[f"Omega_{sp}"] = float(np.mean(omega_vals))

            # Eq. (16): τ_stress and τ_critical
            tot_active = sum(a.steps_active       for a in self.agents[sp])
            tot_stress = sum(a.steps_above_topt   for a in self.agents[sp])
            tot_crit   = sum(a.steps_at_ctmax     for a in self.agents[sp])
            metrics[f"tau_stress_{sp}"]   = tot_stress / tot_active if tot_active else 0.0
            metrics[f"tau_critical_{sp}"] = tot_crit   / tot_active if tot_active else 0.0

        # Dominant species for this run
        E_vals             = {sp: metrics[f"E_{sp}"] for sp in SPECIES_LIST}
        metrics["dominant"] = max(E_vals, key=E_vals.get)
        return metrics

    # ── mechanistic validation helper ────────────────────────────────────

    def collapse_temperatures(self) -> dict:
        """
        For each species, find T_collapse: mean temperature during the first
        window of ≥50 consecutive timesteps where Ω < 0.5.
        Validity criterion: T_opt < T_collapse ≤ CT_max.
        """
        result = {}
        for sp in SPECIES_LIST:
            omega_arr = np.array([r[f"omega_{sp}"] for r in self.records])
            T_arr     = np.array([r.get(f"mean_T_{sp}", np.nan) for r in self.records])
            consec    = 0
            T_col     = None
            for i, om in enumerate(omega_arr):
                if om < 0.5:
                    consec += 1
                    if consec >= 50:
                        start = i - 49
                        T_col = float(np.nanmean(T_arr[start : i + 1]))
                        break
                else:
                    consec = 0
            result[sp] = T_col
        return result


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — SHADE GENERATION (Monte Carlo environment builder)
# ══════════════════════════════════════════════════════════════════════════

def generate_shade(G: float, W: float, B: float) -> np.ndarray:
    """
    Eq. (12)–(13): Stochastic shade field from land-cover covariates.
      p_shade = max(0, 0.8G + 0.1W − 0.3B)
    Binary seed cells are box-blurred into a continuous gradient.
    Fully vectorized with NumPy.
    """
    p_shade = max(0.0, 0.8 * G + 0.1 * W - 0.3 * B)
    # Random seed positions
    seed    = (np.random.rand(GRID_W, GRID_H) < p_shade).astype(np.float64)
    # Eq. (13): 3×3 box blur via cumulative sum (O(n) instead of O(9n))
    padded  = np.pad(seed, 1, mode="edge")
    smooth  = np.zeros((GRID_W, GRID_H), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            smooth += padded[di : di + GRID_W, dj : dj + GRID_H]
    smooth /= 9.0
    return smooth


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════

# NCR_SITES and N_MC_RUNS are loaded from config/sites.toml and
# config/simulation.toml respectively (see Section 1 above).


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
        name   = site["name"]
        T_mean = site["T_base_mean"]
        G, W, B = site["G"], site["W"], site["B"]

        if verbose:
            print(f"  [{s_idx + 1:>2}/{len(sites)}] {name}  "
                  f"T_mean={T_mean:.1f}°C  G={G:.2f}  W={W:.2f}  B={B:.2f}")

        run_rows = []
        for run_id in range(n_runs):
            # Stochastic T_base: ±σ variation around site mean LST
            T_base = np.random.normal(T_mean, 1.5, (GRID_W, GRID_H))
            T_base = np.clip(T_base, T_mean - 4.0, T_mean + 4.0)

            shade  = generate_shade(G, W, B)
            model  = AntForagingModel(T_base=T_base, shade=shade, run_id=run_id)
            model.run(steps=TIMESTEPS)
            metrics = model.compute_run_metrics()
            metrics["run_id"] = run_id
            metrics["site"]   = name
            run_rows.append(metrics)

            if verbose and (run_id + 1) % 100 == 0:
                dom_counts = {}
                for r in run_rows:
                    d = r["dominant"]
                    dom_counts[d] = dom_counts.get(d, 0) + 1
                top = max(dom_counts, key=dom_counts.get)
                print(f"       Run {run_id + 1:>3}/{n_runs}  "
                      f"leading: {top.replace('_', ' ')} ({dom_counts[top]})")

        all_results[name] = pd.DataFrame(run_rows)

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTPUT METRICS & VALIDATION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════

# ── Eqs. (17)–(20): site-level dominance, accuracy, entropy ──────────────

def site_dominance(df: pd.DataFrame) -> dict:
    """
    Eq. (17): P_i = N_wins_i / 500
    Eq. (18): Ŝ = argmax P_i
    Eq. (20): H = −Σ P_i ln P_i   (Shannon dominance entropy)
    """
    n   = len(df)
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
        1 for site, dom in site_dominance_dict.items()
        if ground_truth.get(site) == dom["S_hat"]
    )
    return n_correct / len(site_dominance_dict) if site_dominance_dict else 0.0


# ── Graph 1: Thermal Performance Curves ──────────────────────────────────

def plot_thermal_performance_curves(out_dir: str):
    T_range = np.linspace(0, 50, 500)
    fig, ax = plt.subplots(figsize=(10, 6))
    for sp in SPECIES_LIST:
        p    = SPECIES_PARAMS[sp]
        f_v  = [thermal_multiplier(sp, T) for T in T_range]
        ax.plot(T_range, f_v, color=p["color"], lw=2.5, label=p["label"])
        ax.axvline(p["T_opt"],  color=p["color"], ls=":",  alpha=0.55, lw=1)
        ax.axvline(p["CT_max"], color=p["color"], ls="--", alpha=0.45, lw=1)
        ax.axvline(p["CT_min"], color=p["color"], ls="--", alpha=0.45, lw=1)
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Thermal Performance Multiplier  f_i(T)", fontsize=12)
    ax.set_title("Gaussian Thermal Performance Curves — All Four Species", fontsize=13)
    ax.set_xlim(0, 50); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_thermal_performance_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path}")


# ── Graph 2: Foraging Efficiency Boxplots per Site ────────────────────────

def plot_foraging_efficiency_boxplots(all_results: dict, out_dir: str):
    n_sites = len(all_results)
    ncols   = 4
    nrows   = math.ceil(n_sites / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4.5), sharey=False)
    axes = np.array(axes).flatten()

    for idx, (site_name, df) in enumerate(all_results.items()):
        ax     = axes[idx]
        data   = [df[f"E_{sp}"].values for sp in SPECIES_LIST]
        labels = [SPECIES_PARAMS[sp]["label"] for sp in SPECIES_LIST]
        colors = [SPECIES_PARAMS[sp]["color"] for sp in SPECIES_LIST]
        bp     = ax.boxplot(data, patch_artist=True, labels=labels,
                            medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.72)
        dom     = site_dominance(df)
        ax.set_title(f"{site_name}\nŜ = {dom['S_hat'].replace('_', ' ')}  H={dom['H']:.2f}",
                     fontsize=9)
        ax.set_ylabel("E_i" if idx % ncols == 0 else "")
        ax.tick_params(axis="x", rotation=40, labelsize=7)

    for ax in axes[n_sites:]:
        ax.set_visible(False)

    plt.suptitle("Foraging Efficiency  E_i  Distribution across NCR Sites (500 Monte Carlo Runs)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_foraging_efficiency_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path}")


# ── Graph 3: Species Dominance Probability Heatmap ───────────────────────

def plot_dominance_heatmap(all_results: dict, out_dir: str):
    site_names = list(all_results.keys())
    mat        = np.zeros((len(SPECIES_LIST), len(site_names)))

    for j, name in enumerate(site_names):
        dom = site_dominance(all_results[name])
        for i, sp in enumerate(SPECIES_LIST):
            mat[i, j] = dom["P_i"].get(sp, 0.0)

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(site_names)))
    ax.set_xticklabels(site_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(SPECIES_LIST)))
    ax.set_yticklabels([SPECIES_PARAMS[sp]["label"] for sp in SPECIES_LIST], fontsize=10)
    ax.set_title("Species Dominance Probability  P_i  Across 16 NCR Sites", fontsize=13)
    fig.colorbar(im, ax=ax, label="P_i")
    # Annotate cells
    for i in range(len(SPECIES_LIST)):
        for j in range(len(site_names)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 0.7 else "white")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_dominance_probability_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path}")


# ── Graph 4: Shannon Entropy per Site ────────────────────────────────────

def plot_shannon_entropy(all_results: dict, out_dir: str):
    site_names = list(all_results.keys())
    entropies  = [site_dominance(all_results[n])["H"] for n in site_names]
    threshold  = math.log(2)   # H = ln(2) ≈ 0.693  transition threshold
    colors     = ["#E74C3C" if h > threshold else "#2980B9" for h in entropies]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(site_names, entropies, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(threshold, color="orange", ls="--", lw=2,
               label=f"H = ln(2) ≈ {threshold:.3f}  transition threshold")
    ax.set_xlabel("NCR Site", fontsize=11)
    ax.set_ylabel("Shannon Dominance Entropy  H", fontsize=11)
    ax.set_title("Site-Level Competitive Uncertainty — Red: high-uncertainty transition zones",
                 fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_shannon_entropy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path}")


# ── Graph 5: Collective Order Ω vs Temperature (mechanistic validation) ──

def plot_collective_order(ref_model: AntForagingModel, out_dir: str):
    collapse = ref_model.collapse_temperatures()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.flatten()

    for idx, sp in enumerate(SPECIES_LIST):
        p    = SPECIES_PARAMS[sp]
        ax   = axes[idx]
        T_s  = np.array([r.get(f"mean_T_{sp}", np.nan) for r in ref_model.records])
        Om_s = np.array([r[f"omega_{sp}"]              for r in ref_model.records])

        valid = ~np.isnan(T_s)
        ax.scatter(T_s[valid], Om_s[valid], alpha=0.25, s=8,
                   color=p["color"], label="Ω per step")

        # Binned mean curve
        T_v   = T_s[valid]
        Om_v  = Om_s[valid]
        if len(T_v) > 10:
            bins  = np.linspace(T_v.min(), T_v.max(), 25)
            bx, by = [], []
            for bi in range(len(bins) - 1):
                mask = (T_v >= bins[bi]) & (T_v < bins[bi + 1])
                if mask.sum() > 0:
                    bx.append((bins[bi] + bins[bi + 1]) / 2)
                    by.append(Om_v[mask].mean())
            ax.plot(bx, by, "k-o", lw=2, ms=4, label="Ω binned mean")

        ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Ω = 0.5")
        ax.axvline(p["T_opt"],  color="green",   ls=":",  lw=1.5,
                   label=f"T_opt = {p['T_opt']}°C")
        ax.axvline(p["CT_max"], color="darkred", ls="--", lw=1.5,
                   label=f"CT_max = {p['CT_max']}°C")

        T_col = collapse.get(sp)
        if T_col is not None:
            ok    = p["T_opt"] < T_col <= p["CT_max"]
            label = f"T_collapse = {T_col:.1f}°C  {'✓ VALID' if ok else '✗ INVALID'}"
            ax.axvline(T_col, color="orange", ls="-.", lw=2, label=label)
            ax.set_title(f"{p['label']}  —  {label}", fontsize=10)
        else:
            ax.set_title(f"{p['label']}  —  Ω stable (no collapse detected)", fontsize=10)

        ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Collective Order Ω")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)

    plt.suptitle("Collective Order Ω vs Temperature — Mechanistic Validation",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_collective_order_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {path}")


# ── Graph 6: Thermal Exposure Distribution per Species ───────────────────

def plot_thermal_exposure(all_results: dict, out_dir: str):
    """Violin plot of τ_stress and τ_critical across all sites and runs."""
    records = []
    for site_name, df in all_results.items():
        for sp in SPECIES_LIST:
            for _, row in df.iterrows():
                records.append({
                    "site":     site_name,
                    "species":  SPECIES_PARAMS[sp]["label"],
                    "tau_stress":   row.get(f"tau_stress_{sp}", np.nan),
                    "tau_critical": row.get(f"tau_critical_{sp}", np.nan),
                })
    tdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col, title in zip(
        axes,
        ["tau_stress", "tau_critical"],
        ["τ_stress  (T > T_opt)", "τ_critical  (T ≥ CT_max)"]
    ):
        sns.violinplot(
            data=tdf, x="species", y=col, ax=ax, inner="quartile",
            palette={SPECIES_PARAMS[sp]["label"]: SPECIES_PARAMS[sp]["color"]
                     for sp in SPECIES_LIST}
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(""); ax.set_ylabel("Proportion of timesteps")
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Thermal Exposure Distribution across All Monte Carlo Runs", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig6_thermal_exposure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
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
        dom       = site_dominance(df)
        dom_dict[site_name] = dom
        row = {
            "Site":             site_name,
            "Predicted (Ŝ)":   dom["S_hat"].replace("_", " "),
            "H":                round(dom["H"], 3),
        }
        for sp in SPECIES_LIST:
            row[f"P({SPECIES_PARAMS[sp]['label']})"] = round(dom["P_i"].get(sp, 0.0), 3)
        if ground_truth:
            actual          = ground_truth.get(site_name, "—")
            row["Actual"]   = actual
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
        print(f"\n  Eq.(19) Classification Accuracy: {n_c}/{len(all_results)} = {acc:.3f}")

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
    print(hdr); print("  " + "─" * 68)

    for sp in SPECIES_LIST:
        p     = SPECIES_PARAMS[sp]
        T_col = collapse_temps.get(sp)
        if T_col is not None:
            ok   = p["T_opt"] < T_col <= p["CT_max"]
            crit = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {p['label']:<22} {p['T_opt']:>7.1f} {T_col:>12.2f}"
                  f" {p['CT_max']:>8.1f} {crit:>16}")
            rows.append({
                "Species":    p["label"],
                "T_opt (°C)": p["T_opt"],
                "T_collapse": round(T_col, 2),
                "CT_max (°C)":p["CT_max"],
                "Valid":      crit,
            })
        else:
            print(f"  {p['label']:<22} {p['T_opt']:>7.1f} {'STABLE':>12}"
                  f" {p['CT_max']:>8.1f} {'✓ STABLE':>16}")
            rows.append({
                "Species":    p["label"],
                "T_opt (°C)": p["T_opt"],
                "T_collapse": "STABLE",
                "CT_max (°C)":p["CT_max"],
                "Valid":      "✓ STABLE",
            })

    mech_df = pd.DataFrame(rows)
    path    = os.path.join(out_dir, "table2_mechanistic_validation.csv")
    mech_df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")
    return mech_df


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — DUAL-MODE EXECUTION CONTROLLER
# ══════════════════════════════════════════════════════════════════════════

RUN_GUI = False   # module-level toggle; overridden by --visualize CLI flag


def run_visual_mode():
    """
    Visual Verification Mode (--visualize).
    Runs a single site simulation with a real-time matplotlib animation
    showing agent positions, pheromone heatmap, and temperature overlay.
    """
    print("\n[VISUAL MODE] Launching interactive matplotlib animation…")
    print("  Site: NCR_01   Close the window to exit.\n")

    site   = NCR_SITES[0]
    T_base = np.random.normal(site["T_base_mean"], 1.5, (GRID_W, GRID_H))
    shade  = generate_shade(site["G"], site["W"], site["B"])
    model  = AntForagingModel(T_base=T_base, shade=shade, run_id=0)

    import matplotlib.animation as animation

    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax_agents = fig.add_subplot(gs[0, 0])
    ax_phi    = fig.add_subplot(gs[0, 1])
    ax_omega  = fig.add_subplot(gs[0, 2])

    omega_history = {sp: [] for sp in SPECIES_LIST}

    def _update(frame):
        model.step()
        t = model.step_num

        # ── Panel 1: agent map ──────────────────────────────────────
        ax_agents.cla()
        ax_agents.imshow(
            model.env.T_field.T, origin="lower", cmap="RdYlBu_r",
            vmin=25, vmax=45, alpha=0.35, aspect="auto",
            extent=[0, GRID_W, 0, GRID_H]
        )
        for sp in SPECIES_LIST:
            p  = SPECIES_PARAMS[sp]
            xs = [a.x for a in model.agents[sp] if a.state in (SEARCHING, RETURNING)]
            ys = [a.y for a in model.agents[sp] if a.state in (SEARCHING, RETURNING)]
            if xs:
                ax_agents.scatter(xs, ys, s=6, c=p["color"], alpha=0.75, label=p["label"])
        ax_agents.plot(*FOOD_POS, "w*", ms=14, label="Food")
        for sp in SPECIES_LIST:
            nx, ny = NEST_POSITIONS[sp]
            ax_agents.plot(nx, ny, "s",
                           color=SPECIES_PARAMS[sp]["color"], ms=9, mec="white")
        ax_agents.set_xlim(0, GRID_W); ax_agents.set_ylim(0, GRID_H)
        ax_agents.set_title(f"Agents  (step {t}/{TIMESTEPS})", fontsize=10)
        ax_agents.legend(fontsize=7, loc="upper right")

        # ── Panel 2: pheromone heatmap ──────────────────────────────
        ax_phi.cla()
        phi_sum = model.env.phi.sum(axis=0)
        ax_phi.imshow(phi_sum.T, origin="lower", cmap="hot",
                      vmin=0, vmax=max(1.0, phi_sum.max()), aspect="auto")
        ax_phi.set_title("Pheromone Field (sum)", fontsize=10)

        # ── Panel 3: Ω time series ─────────────────────────────────
        ax_omega.cla()
        rec = model.records[-1] if model.records else {}
        for sp in SPECIES_LIST:
            om = rec.get(f"omega_{sp}", 0.0)
            omega_history[sp].append(om)
        steps = list(range(1, len(omega_history[SPECIES_LIST[0]]) + 1))
        for sp in SPECIES_LIST:
            p = SPECIES_PARAMS[sp]
            ax_omega.plot(steps, omega_history[sp], color=p["color"], lw=1.5, label=p["label"])
        ax_omega.axhline(0.5, color="red", ls="--", lw=1, alpha=0.6)
        ax_omega.set_xlim(0, TIMESTEPS); ax_omega.set_ylim(-0.05, 1.05)
        ax_omega.set_xlabel("Timestep"); ax_omega.set_ylabel("Ω")
        ax_omega.set_title("Collective Order Ω", fontsize=10)
        ax_omega.legend(fontsize=7)

        return []

    ani = animation.FuncAnimation(
        fig, _update, frames=TIMESTEPS, interval=30, blit=False, repeat=False
    )
    plt.suptitle("ANTS — Thermally Constrained Pheromone Foraging (Visual Mode)", fontsize=12)
    plt.tight_layout()
    plt.show()


def run_headless_batch(
    n_runs:  int  = N_MC_RUNS,
    n_sites: int  = len(NCR_SITES),
    out_dir: str  = "ants_output",
    verbose: bool = True,
):
    """
    High-Throughput Batch Mode.
    Executes the full Monte Carlo, generates all validation figures and tables,
    and exports raw data to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    sites = NCR_SITES[:n_sites]

    print("\n" + "═" * 65)
    print("  ANTS SIMULATION — HIGH-THROUGHPUT HEADLESS BATCH MODE")
    print(f"  Sites : {len(sites)}   Runs/site : {n_runs}   "
          f"Total runs : {len(sites) * n_runs}")
    print(f"  Output: {out_dir}/")
    print("═" * 65 + "\n")

    # ── Monte Carlo runs ─────────────────────────────────────────────────
    all_results = run_monte_carlo(sites=sites, n_runs=n_runs, verbose=verbose)

    # ── Single reference run for mechanistic validation ──────────────────
    print("\n[Reference run] Mechanistic validation simulation…")
    ref    = sites[0]
    T_ref  = np.random.normal(ref["T_base_mean"], 1.5, (GRID_W, GRID_H))
    sh_ref = generate_shade(ref["G"], ref["W"], ref["B"])
    ref_m  = AntForagingModel(T_base=T_ref, shade=sh_ref, run_id=-1)
    ref_m.run(steps=TIMESTEPS)
    col_temps = ref_m.collapse_temperatures()

    # ── Figures ──────────────────────────────────────────────────────────
    print("\n[Figures] Generating output plots…")
    plot_thermal_performance_curves(out_dir)
    plot_foraging_efficiency_boxplots(all_results, out_dir)
    plot_dominance_heatmap(all_results, out_dir)
    plot_shannon_entropy(all_results, out_dir)
    plot_collective_order(ref_m, out_dir)
    plot_thermal_exposure(all_results, out_dir)

    # ── Tables ───────────────────────────────────────────────────────────
    print("\n[Tables] Generating output tables…")
    # Placeholder ground-truth: replace with actual dominant species from [7]
    gt_placeholder = {s["name"]: "P_longicornis" for s in sites}
    generate_site_summary_table(all_results, ground_truth=gt_placeholder, out_dir=out_dir)
    generate_mechanistic_table(col_temps, out_dir=out_dir)

    # ── Raw data export ──────────────────────────────────────────────────
    print("\n[Export] Saving raw Monte Carlo dataset…")
    raw = pd.concat(list(all_results.values()), ignore_index=True)
    raw_path = os.path.join(out_dir, "monte_carlo_raw_data.csv")
    raw.to_csv(raw_path, index=False)
    print(f"  Saved: {raw_path}")

    print("\n" + "═" * 65)
    print("  SIMULATION COMPLETE")
    print(f"  All outputs saved to: {os.path.abspath(out_dir)}/")
    print("═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ANTS: Thermally Constrained Pheromone-Mediated Foraging ABM"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch matplotlib visual verification mode (single site)",
    )
    parser.add_argument(
        "--runs", type=int, default=N_MC_RUNS,
        help=f"Monte Carlo runs per site  (default: {N_MC_RUNS})",
    )
    parser.add_argument(
        "--sites", type=int, default=len(NCR_SITES),
        help=f"Number of NCR sites to process  (default: {len(NCR_SITES)})",
    )
    parser.add_argument(
        "--out", type=str, default="ants_output",
        help="Output directory for figures and tables  (default: ants_output)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run progress output",
    )
    args = parser.parse_args()

    if args.visualize or RUN_GUI:
        run_visual_mode()
    else:
        run_headless_batch(
            n_runs  = args.runs,
            n_sites = args.sites,
            out_dir = args.out,
            verbose = not args.quiet,
        )