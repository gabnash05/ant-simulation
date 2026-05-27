"""thermal.py — Thermal performance functions."""

import math
from config import SPECIES_PARAMS, BETA_0, T_REF


def thermal_multiplier(sp: str, T: float) -> float:
    p = SPECIES_PARAMS[sp]
    return math.exp(-((T - p["T_opt"]) ** 2) / p["_thermal_denom"])


def effective_velocity(sp: str, T: float) -> float:
    return SPECIES_PARAMS[sp]["v_max_grid"] * thermal_multiplier(sp, T)


def species_decay_rate(sp: str, T: float) -> float:
    p = SPECIES_PARAMS[sp]
    return BETA_0 * p["kappa"] * (T + T_REF)
