"""agents.py — Ant agent class."""

import math
import random
from config import (
    SPECIES_PARAMS,
    GRID_W,
    GRID_H,
    DELTA_THETA,
    C_THRESH,
    K_ATTRACT,
    N_NONLIN,
    P_DROP,
    B_REC,
    FOOD_POS,
    FOOD_RADIUS,
    INACTIVE,
    SEARCHING,
    RETURNING,
    INCAPACITATED,
)
from thermal import effective_velocity


class AntAgent:
    """
    Off-lattice ant agent.
    • Continuous (x, y) coordinates; floor-mapped to discrete grid cells.
    • State machine: INACTIVE → SEARCHING → RETURNING → (loop)
    • Incapacitated when local T ≥ CT_max (Eq. 8).
    """

    __slots__ = [
        "uid",
        "sp",
        "sp_idx",
        "nest_x",
        "nest_y",
        "model",
        "x",
        "y",
        "theta",
        "state",
        "trips",
        "steps_active",
        "steps_above_topt",
        "steps_at_ctmax",
        "trail_steps",
        "chose_pheromone",
        "trail_detected",
    ]

    def __init__(self, uid: int, sp: str, sp_idx: int, nest: tuple, model):
        self.uid = uid
        self.sp = sp
        self.sp_idx = sp_idx
        self.nest_x = float(nest[0])
        self.nest_y = float(nest[1])
        self.model = model

        self.x = float(nest[0])
        self.y = float(nest[1])
        self.theta = random.uniform(0.0, 2.0 * math.pi)
        self.state = INACTIVE

        # Telemetry accumulators
        self.trips = 0
        self.steps_active = 0
        self.steps_above_topt = 0
        self.steps_at_ctmax = 0
        self.trail_steps = 0
        self.chose_pheromone = False
        self.trail_detected = False

    # ── helpers ──────────────────────────────────────────────────────────

    def _cell(self):
        return int(math.floor(self.x)), int(math.floor(self.y))

    def _local_T(self) -> float:
        cx, cy = self._cell()
        return self.model.env.get_T(cx, cy)

    def _reflect(self, nx: float, ny: float) -> tuple:
        """Wall reflection: reverse the affected heading component."""
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            return nx, ny

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
        """Returns list of (phi, dtheta) for left, center, right candidates for pheromone sensing."""
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        # Center cell
        cx_c = int(math.floor(max(0.0, min(GRID_W - 1, self.x + v * cos_t))))
        cy_c = int(math.floor(max(0.0, min(GRID_H - 1, self.y + v * sin_t))))
        phi_c = self.model.env.get_phi(self.sp_idx, cx_c, cy_c)

        # Left and right: rotate the center offset by ±Δθ
        cos_d = math.cos(DELTA_THETA)
        sin_d = math.sin(DELTA_THETA)

        # Left candidate: rotate (cos_t, sin_t) by -Δθ
        cos_l = cos_t * cos_d + sin_t * sin_d  # cos(θ - Δθ)
        sin_l = sin_t * cos_d - cos_t * sin_d  # sin(θ - Δθ)
        cx_l = int(math.floor(max(0.0, min(GRID_W - 1, self.x + v * cos_l))))
        cy_l = int(math.floor(max(0.0, min(GRID_H - 1, self.y + v * sin_l))))
        phi_l = self.model.env.get_phi(self.sp_idx, cx_l, cy_l)

        # Right candidate: rotate (cos_t, sin_t) by +Δθ
        cos_r = cos_t * cos_d - sin_t * sin_d  # cos(θ + Δθ)
        sin_r = sin_t * cos_d + cos_t * sin_d  # sin(θ + Δθ)
        cx_r = int(math.floor(max(0.0, min(GRID_W - 1, self.x + v * cos_r))))
        cy_r = int(math.floor(max(0.0, min(GRID_H - 1, self.y + v * sin_r))))
        phi_r = self.model.env.get_phi(self.sp_idx, cx_r, cy_r)

        return [(phi_l, -DELTA_THETA), (phi_c, 0.0), (phi_r, DELTA_THETA)]

    # ── Eq. (11): stochastic trail selection ─────────────────────────────

    def _stochastic_choice(self, candidates: list) -> float:
        """
        P(k) = (K + C_k)^n / Σ_j (K + C_j)^n
        Returns the angle offset of the chosen candidate.
        """
        for phi, dθ in candidates:
            tx = self.x + math.cos(self.theta + dθ)
            ty = self.y + math.sin(self.theta + dθ)

            if (
                abs(tx - FOOD_POS[0]) <= FOOD_RADIUS
                and abs(ty - FOOD_POS[1]) <= FOOD_RADIUS
            ):
                return dθ

        weights = [(K_ATTRACT + phi) ** N_NONLIN for phi, _ in candidates]
        total = sum(weights)
        probs = [w / total for w in weights]
        r, cum = random.random(), 0.0
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
        p = SPECIES_PARAMS[self.sp]

        if self.state == INCAPACITATED:
            if T_local < p["CT_max"]:
                self.state = SEARCHING
            else:
                return

        if T_local >= p["CT_max"]:
            self.state = INCAPACITATED
            self.steps_at_ctmax += 1
            return

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

        max_phi = max(c[0] for c in candidates)
        P_detect = max_phi / (max_phi + C_THRESH)

        self.trail_detected = P_detect > 0.5

        if max_phi > 0 and random.random() < P_detect:
            dtheta = self._stochastic_choice(candidates)
            self.chose_pheromone = True
            self.trail_steps += 1
        else:
            dtheta = random.choice((-DELTA_THETA, 0.0, DELTA_THETA))
            dtheta += random.gauss(0.0, math.radians(8.0))

        self.theta += dtheta

        nx = self.x + v * math.cos(self.theta)
        ny = self.y + v * math.sin(self.theta)

        self.x, self.y = self._reflect(nx, ny)

        cx, cy = self._cell()

        if (
            abs(cx - FOOD_POS[0]) <= FOOD_RADIUS
            and abs(cy - FOOD_POS[1]) <= FOOD_RADIUS
        ):
            self.state = RETURNING
            self.theta = math.atan2(self.nest_y - self.y, self.nest_x - self.x)

    # ── RETURNING behavior ───────────────────────────────────────────────

    def _return_step(self, v: float):
        # Eq. (10): pheromone deposition
        cx, cy = self._cell()
        self.model.env.deposit_phi(self.sp_idx, cx, cy, P_DROP)

        # Directed movement toward nest
        dx = self.nest_x - self.x
        dy = self.nest_y - self.y
        dist = math.hypot(dx, dy)

        if dist <= v:
            # Arrived at nest
            self.x, self.y = self.nest_x, self.nest_y
            self.state = SEARCHING
            self.theta = random.uniform(0.0, 2.0 * math.pi)
            self.trips += 1
            self.model.notify_delivery(self.sp)
            # Pheromone-driven recruitment batch (Section E.4)
            self.model.recruit(self.sp, B_REC)
        else:
            self.theta = math.atan2(dy, dx)
            nx = self.x + v * math.cos(self.theta)
            ny = self.y + v * math.sin(self.theta)
            self.x, self.y = self._reflect(nx, ny)
