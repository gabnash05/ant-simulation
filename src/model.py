"""model.py — Ant Foraging Model (main simulation controller)."""

import math
import random
import numpy as np
from config import (
    SPECIES_LIST,
    SPECIES_PARAMS,
    NEST_POSITIONS,
    FOOD_POS,
    N_COLONY,
    N_SCOUTS,
    TIMESTEPS,
    N_ACT_MIN,
    P_TRICKLE,
    T_THRESH_SCALE,
    INACTIVE,
    SEARCHING,
    RETURNING,
)
from environment import EnvironmentLayers
from agents import AntAgent


class AntForagingModel:
    """
    Main simulation controller.

    Manages execution clock, agent activation, environment updates,
    recruitment (Section III-D.4), and per-step metric collection (Section III-G).
    """

    def __init__(self, T_base: np.ndarray, shade: np.ndarray, run_id: int = 0):
        self.run_id = run_id
        self.step_num = 0
        self.env = EnvironmentLayers(T_base, shade)

        self.agents: dict[str, list[AntAgent]] = {sp: [] for sp in SPECIES_LIST}
        self._uid = 0

        self.last_delivery_step: dict[str, int] = {sp: 0 for sp in SPECIES_LIST}
        D_avg = {
            sp: math.hypot(
                FOOD_POS[0] - NEST_POSITIONS[sp][0],
                FOOD_POS[1] - NEST_POSITIONS[sp][1],
            )
            for sp in SPECIES_LIST
        }

        # Per-species temporal trickle threshold (timesteps), computed once
        # from grid geometry and each species' mean velocity.
        self.t_thresh: dict[str, float] = {
            sp: T_THRESH_SCALE * (2.0 * D_avg[sp] * SPECIES_PARAMS[sp]["v_mean_grid"])
            for sp in SPECIES_LIST
        }

        for sp_idx, sp in enumerate(SPECIES_LIST):
            nest = NEST_POSITIONS[sp]
            for _ in range(N_COLONY):
                a = AntAgent(self._uid, sp, sp_idx, nest, self)
                self.agents[sp].append(a)
                self._uid += 1
            for a in self.agents[sp][:N_SCOUTS]:
                a.state = SEARCHING
                a.theta = random.uniform(0.0, 2.0 * math.pi)

        self.all_agents = [a for sp in SPECIES_LIST for a in self.agents[sp]]
        self.records: list[dict] = []

    # ═════════════════════════════════════════════════════════════
    # Recruitment
    # ═════════════════════════════════════════════════════════════

    def recruit(self, sp: str, n: int):
        """
        Section III-D.4 — Batch Recruitment

        Activates up to n INACTIVE agents (B_rec = 3 on food delivery).
        """
        count = 0
        for a in self.agents[sp]:
            if count >= n:
                break
            if a.state == INACTIVE:
                a.state = SEARCHING
                a.theta = random.uniform(0.0, 2.0 * math.pi)
                count += 1

    def notify_delivery(self, sp: str):
        """Record timestep of most recent food delivery for species sp."""
        self.last_delivery_step[sp] = self.step_num

    def _trickle_release(self):
        """
        Section III-D.4 — Stochastic Trickle Recruitment

        Deploys one INACTIVE agent with P_trickle = 0.2 when N_active < 15
        or elapsed since delivery exceeds species-specific t_thresh.
        """
        for sp in SPECIES_LIST:
            n_active = sum(
                1 for a in self.agents[sp] if a.state in (SEARCHING, RETURNING)
            )
            elapsed_since_delivery = self.step_num - self.last_delivery_step[sp]

            population_starved = n_active < N_ACT_MIN
            delivery_stalled = elapsed_since_delivery > self.t_thresh[sp]

            if (population_starved or delivery_stalled) and random.random() < P_TRICKLE:
                self.recruit(sp, 1)

    # ═════════════════════════════════════════════════════════════
    # Main Simulation Step
    # ═════════════════════════════════════════════════════════════

    def step(self):
        """Advance one timestep: environment, recruitment, agents, and metrics."""
        self.step_num += 1
        t = self.step_num

        self.env.update_temperature(t)
        self.env.update_pheromones()

        self._trickle_release()

        random.shuffle(self.all_agents)
        for a in self.all_agents:
            a.step()

        # Thermal exposure tracking (Eq. 19 numerators)
        for a in self.all_agents:
            if a.state == INACTIVE:
                continue
            T_here = self.env.get_T(int(math.floor(a.x)), int(math.floor(a.y)))
            sp_p = SPECIES_PARAMS[a.sp]
            if T_here >= sp_p["CT_max"]:
                a.steps_at_ctmax += 1
            if T_here > sp_p["T_opt"]:
                a.steps_above_topt += 1

        rec = {"step": t}
        for sp in SPECIES_LIST:
            active = [a for a in self.agents[sp] if a.state in (SEARCHING, RETURNING)]
            trail = [a for a in active if a.trail_detected]
            n_a = len(active)
            n_t = len(trail)

            # Eq. (18): Ω = N_trail / N_active
            omega = n_t / n_a if n_a > 0 else 0.0
            rec[f"n_active_{sp}"] = n_a
            rec[f"omega_{sp}"] = omega

            total_trips = sum(a.trips for a in self.agents[sp])
            # Eq. (17): E_i = N_trips,i / N_colony,i
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

    # ═════════════════════════════════════════════════════════════
    # Per-run Aggregate Metrics
    # ═════════════════════════════════════════════════════════════

    def compute_run_metrics(self) -> dict:
        """
        Section III-G — Output Metrics

        Per-species E_i (Eq. 17), Ω (Eq. 18), τ_stress and τ_critical (Eq. 19),
        plus run-level dominant species by max E_i.
        """
        metrics = {}
        for sp in SPECIES_LIST:
            total_trips = sum(a.trips for a in self.agents[sp])
            # Eq. (17): E_i = N_trips,i / N_colony,i
            E_i = total_trips / N_COLONY
            metrics[f"E_{sp}"] = E_i

            omega_vals = [r[f"omega_{sp}"] for r in self.records]
            metrics[f"Omega_{sp}"] = float(np.mean(omega_vals))

            # Eq. (19): τ_stress = t(T>T_opt)/t_total, τ_critical = t(T≥CT_max)/t_total
            tot_stress = sum(a.steps_above_topt for a in self.agents[sp])
            tot_crit = sum(a.steps_at_ctmax for a in self.agents[sp])
            tot_field = sum(a.steps_active + a.steps_at_ctmax for a in self.agents[sp])
            metrics[f"tau_stress_{sp}"] = tot_stress / tot_field if tot_field else 0.0
            metrics[f"tau_critical_{sp}"] = tot_crit / tot_field if tot_field else 0.0

        E_vals = {sp: metrics[f"E_{sp}"] for sp in SPECIES_LIST}
        metrics["dominant"] = max(E_vals, key=E_vals.get)
        return metrics

    # ═════════════════════════════════════════════════════════════
    # Mechanistic Validation
    # ═════════════════════════════════════════════════════════════

    def collapse_temperatures(self) -> dict:
        """
        Mechanistic validation helper.

        T_collapse: mean temperature during first ≥50 consecutive timesteps with Ω < 0.5.
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
