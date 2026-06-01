"""thermal.py — Thermal performance functions."""

import math
from config import SPECIES_PARAMS, BETA_0, T_REF


# ═════════════════════════════════════════════════════════════
# Thermal Performance Functions
# ═════════════════════════════════════════════════════════════


def thermal_multiplier(sp: str, T: float) -> float:
    """
    Eq. (4): f_i(T) = exp(-(T - T_opt)² / (2σ²))

    Gaussian thermal performance multiplier for species sp at temperature T.
    """
    p = SPECIES_PARAMS[sp]
    return math.exp(-((T - p["T_opt"]) ** 2) / p["_thermal_denom"])


def effective_velocity(sp: str, T: float) -> float:
    """
    Eq. (3): v_i(T) = v_mean,i · f_i(T)

    Temperature-adjusted movement speed on the grid.
    """
    return SPECIES_PARAMS[sp]["v_mean_grid"] * thermal_multiplier(sp, T)


def species_decay_rate(sp: str, T: float) -> float:
    """
    Eq. (14): α_i(T) = β₀ κ_i (T + T_ref)

    Temperature-dependent pheromone evaporation rate for species sp.
    """
    p = SPECIES_PARAMS[sp]
    return BETA_0 * p["kappa"] * (T + T_REF)
