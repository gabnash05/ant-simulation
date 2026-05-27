"""config.py — Load all configuration once. Import from here everywhere."""

import math
import tomllib
from pathlib import Path

_CFG_DIR = Path(__file__).parent / "config"


def _load_toml(filename: str) -> dict:
    with open(_CFG_DIR / filename, "rb") as _f:
        return tomllib.load(_f)


def _build_species_params(raw: dict) -> dict:
    out = {}
    for sp, vals in raw.items():
        entry = dict(vals)
        entry["v_max_grid"] = entry["v_max_mm_s"] * 60 / 1000
        sig = (entry["CT_max"] - entry["CT_min"]) / 6.0
        entry["_thermal_denom"] = 2.0 * sig**2
        out[sp] = entry
    return out


# ── Species ────────────────────────────────────────────────────────────
SPECIES_PARAMS = _build_species_params(_load_toml("species.toml"))
SPECIES_LIST = list(SPECIES_PARAMS.keys())
SPECIES_INDEX = {sp: i for i, sp in enumerate(SPECIES_LIST)}

# ── Simulation ─────────────────────────────────────────────────────────
_sim = _load_toml("simulation.toml")

_comm = _sim["communication"]
BETA_0 = _comm["BETA_0"]
T_REF = _comm["T_REF"]
C_THRESH = _comm["C_THRESH"]
N_NONLIN = _comm["N_NONLIN"]
P_DROP = _comm["P_DROP"]
K_ATTRACT = _comm["K_ATTRACT"]
B_REC = _comm["B_REC"]
P_TRICKLE = _comm["P_TRICKLE"]
N_ACT_MIN = _comm["N_ACT_MIN"]

_grid = _sim["grid"]
GRID_W = _grid["GRID_W"]
GRID_H = _grid["GRID_H"]
TIMESTEPS = _grid["TIMESTEPS"]
DT = _grid["DT"]
A_DIURNAL = _grid["A_DIURNAL"]
FOOD_POS = tuple(_grid["FOOD_POS"])
FOOD_RADIUS = _grid["FOOD_RADIUS"]
DELTA_THETA = math.radians(_grid["DELTA_THETA_DEG"])

_col = _sim["colony"]
N_COLONY = _col["N_COLONY"]
N_SCOUTS = _col["N_SCOUTS"]

N_MC_RUNS = _sim["monte_carlo"]["N_MC_RUNS"]

NEST_POSITIONS = {sp: tuple(coords) for sp, coords in _sim["nest_positions"].items()}

# ── Sites ──────────────────────────────────────────────────────────────
NCR_SITES = _load_toml("sites.toml")["site"]

# ── Agent states ───────────────────────────────────────────────────────
INACTIVE = 0
SEARCHING = 1
RETURNING = 2
INCAPACITATED = 3
