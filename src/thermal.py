"""thermal.py — Thermal performance functions."""

import math
from config import SPECIES_PARAMS, BETA_0, T_REF


def thermal_multiplier(sp: str, T: float) -> float:
    # Calculates a Gaussian-shaped modifier based on current temperature.
    # Returns a value between 0.0 and 1.0 depending on distance from optimal temp.
    # Drop-off rate is controlled by the species-specific thermal denominator.
    p = SPECIES_PARAMS[sp]
    return math.exp(-((T - p["T_opt"]) ** 2) / p["_thermal_denom"])


def effective_velocity(sp: str, T: float) -> float:
    # Adjusts the baseline grid velocity based on the thermal multiplier.
    # Models thermal throttling: agents move fastest at their optimal temperature.
    # Velocity approaches zero as environmental temperature extremes are reached.
    return SPECIES_PARAMS[sp]["v_mean_grid"] * thermal_multiplier(sp, T)


def species_decay_rate(sp: str, T: float) -> float:
    # Calculates the pheromone or agent evaporation/decay rate.
    # Scales baseline decay (BETA_0) by species sensitivity (kappa).
    # Temperature-dependent: higher temperatures accelerate the decay rate.
    p = SPECIES_PARAMS[sp]
    return BETA_0 * p["kappa"] * (T + T_REF)