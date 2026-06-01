"""environment.py — Spatial environment layers."""

import math
import numpy as np
from config import GRID_W, GRID_H, A_DIURNAL, TIMESTEPS, SPECIES_LIST
from thermal import species_decay_rate


class EnvironmentLayers:
    """
    Section III-C — Simulation Environment

    Manages:
      • T_base(x,y)    — static base temperature from LST dataset
      • s(x,y)         — static shade coefficient ∈ [0, 1]
      • T_field(x,y,t) — dynamic thermal field (updated each step)
      • φ_k(x,y)       — species-specific pheromone layers (chemically isolated)
    """

    def __init__(self, T_base: np.ndarray, shade: np.ndarray):
        assert T_base.shape == (GRID_W, GRID_H), "T_base must be (100, 100)"
        assert shade.shape == (GRID_W, GRID_H), "shade must be (100, 100)"
        self.T_base = T_base.astype(np.float32)
        self.shade = shade.astype(np.float32)
        self.T_field = T_base.copy().astype(np.float32)
        self.phi = np.zeros((4, GRID_W, GRID_H), dtype=np.float32)

    # ═════════════════════════════════════════════════════════════
    # Field Update Functions
    # ═════════════════════════════════════════════════════════════

    def update_temperature(self, t: int):
        """
        Eq. (1): T(x,y,t) = T_base(x,y) − A(1−s(x,y)) cos(2πt / TIMESTEPS)

        Updates the diurnal thermal field for timestep t.
        """
        factor = (
            A_DIURNAL * (1.0 - self.shade) * math.cos((2 * math.pi * t) / TIMESTEPS)
        )
        self.T_field = self.T_base - factor

    def update_pheromones(self):
        """
        Eq. (2): φ_k(t+1) = φ_k(t) · exp(−α_i(T))

        Applies temperature-dependent pheromone decay per species layer.
        """
        mean_T = float(self.T_field.mean())

        decay = np.array(
            [math.exp(-species_decay_rate(sp, mean_T)) for sp in SPECIES_LIST],
            dtype=np.float32,
        )[:, None, None]

        self.phi *= decay

    # ═════════════════════════════════════════════════════════════
    # Cell-level Accessors
    # ═════════════════════════════════════════════════════════════

    def _clip(self, x: int, y: int):
        """Clamp grid indices to valid bounds."""
        return max(0, min(GRID_W - 1, x)), max(0, min(GRID_H - 1, y))

    def get_T(self, x: int, y: int) -> float:
        """Sample T_field at cell (x, y)."""
        cx, cy = self._clip(x, y)
        return float(self.T_field[cx, cy])

    def get_phi(self, sp_idx: int, x: int, y: int) -> float:
        """Sample pheromone layer φ_k at cell (x, y) for species index sp_idx."""
        cx, cy = self._clip(x, y)
        return float(self.phi[sp_idx, cx, cy])

    def deposit_phi(self, sp_idx: int, x: int, y: int, amount: float):
        """
        Eq. (9): φ_k ← φ_k + P_DROP

        Deposits pheromone amount on species layer sp_idx at cell (x, y).
        """
        cx, cy = self._clip(x, y)
        self.phi[sp_idx, cx, cy] += amount
