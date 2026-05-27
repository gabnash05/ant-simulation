"""model.py — Ant Foraging Model (main simulation controller)."""

import math
import random
import numpy as np
from config import (
    SPECIES_LIST,
    SPECIES_PARAMS,
    NEST_POSITIONS,
    N_COLONY,
    N_SCOUTS,
    TIMESTEPS,
    N_ACT_MIN,
    P_TRICKLE,
    INACTIVE,
    SEARCHING,
    RETURNING,
)
from environment import EnvironmentLayers
from agents import AntAgent


class AntForagingModel:
    """
    Main model class.
    Manages: execution clock, agent activation, environment updates,
    recruitment logic, and per-step data collection.
    """

    def __init__(self, T_base: np.ndarray, shade: np.ndarray, run_id: int = 0):
        self.run_id = run_id
        self.step_num = 0
        self.env = EnvironmentLayers(T_base, shade)

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

        self.all_agents = [a for sp in SPECIES_LIST for a in self.agents[sp]]
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
                count += 1

    def _trickle_release(self):
        """
        Section E.4 — stochastic trickle: if N_active < 15, release 1
        backup agent with P_trickle = 0.2 per timestep.
        """
        for sp in SPECIES_LIST:
            n_active = sum(
                1 for a in self.agents[sp] if a.state in (SEARCHING, RETURNING)
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
        random.shuffle(self.all_agents)
        for a in self.all_agents:
            a.step()

        # ── Per-step data collection ─────────────────────────────────────
        rec = {"step": t}
        for sp in SPECIES_LIST:
            active = [a for a in self.agents[sp] if a.state in (SEARCHING, RETURNING)]
            trail = [a for a in active if a.trail_detected]
            n_a = len(active)
            n_t = len(trail)

            # Eq. (15): Ω = N_trail / N_active
            omega = n_t / n_a if n_a > 0 else 0.0
            rec[f"n_active_{sp}"] = n_a
            rec[f"omega_{sp}"] = omega

            total_trips = sum(a.trips for a in self.agents[sp])
            rec[f"E_{sp}"] = total_trips / N_COLONY

            if active:
                mean_T = (
                    sum(
                        self.env.get_T(int(math.floor(a.x)), int(math.floor(a.y)))
                        for a in active
                    )
                    / n_a
                )
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
            E_i = total_trips / N_COLONY
            metrics[f"E_{sp}"] = E_i

            # Eq. (15): Ω mean across timesteps
            omega_vals = [r[f"omega_{sp}"] for r in self.records]
            metrics[f"Omega_{sp}"] = float(np.mean(omega_vals))

            # Eq. (16): τ_stress and τ_critical
            tot_active = sum(a.steps_active for a in self.agents[sp])
            tot_stress = sum(a.steps_above_topt for a in self.agents[sp])
            tot_crit = sum(a.steps_at_ctmax for a in self.agents[sp])
            metrics[f"tau_stress_{sp}"] = tot_stress / tot_active if tot_active else 0.0
            metrics[f"tau_critical_{sp}"] = tot_crit / tot_active if tot_active else 0.0

        # Dominant species for this run
        E_vals = {sp: metrics[f"E_{sp}"] for sp in SPECIES_LIST}
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
            T_arr = np.array([r.get(f"mean_T_{sp}", np.nan) for r in self.records])
            consec = 0
            T_col = None
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
