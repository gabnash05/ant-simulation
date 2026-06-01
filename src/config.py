"""config.py — Load all configuration once. Import from here everywhere."""

import math
import tomllib
import csv
from pathlib import Path

_CFG_DIR = Path(__file__).parent / "config"
_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def _load_toml(filename: str) -> dict:
    """Load a TOML file from src/config/."""
    with open(_CFG_DIR / filename, "rb") as _f:
        return tomllib.load(_f)


def _build_species_params(raw: dict) -> dict:
    """
    Section III-D.1 — Species parameter derivation

    Converts species.toml entries into grid velocity and Gaussian width used by thermal.py.
    """
    out = {}
    for sp, vals in raw.items():
        entry = dict(vals)
        entry["v_mean_grid"] = entry["v_mean"] * 60 / 1000
        # Eq. (5): σ_i = (CT_max − CT_min) / 6
        sig = (entry["CT_max"] - entry["CT_min"]) / 6.0
        entry["_thermal_denom"] = 2.0 * sig**2
        out[sp] = entry
    return out


def _load_sites_with_lst() -> list[dict]:
    """
    Section III-C — Site configuration loader

    Merge sites.toml (name, dms) with the processed CSV (LST grid values).

    The CSV is produced by src/scripts/extract_lst.py and contains one row
    per site with columns: site_name, id, meanLST, Green_area, Waterbody,
    Built_area, Elevation, EHcluster.

    Each returned dict exposes the field names the model uses:
        name        — site identifier (matches sites.toml)
        dms         — original DMS string
        T_base_mean — meanLST (°C, mean 2000-2023)
        G           — Green_area (ha of greenery in the 1 km² cell)
        W           — Waterbody (0/1, intersects inland water body)
        B           — Built_area (ha of built-up land cover)
        grid_id     — grid cell identifier
        Elevation   — metres above sea level
        EHcluster   — 0/1 extreme heat cluster membership
    """

    csv_path = _DATA_DIR / "sites_with_lst.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed site data not found: {csv_path}\n"
            "Run src/scripts/extract_lst.py first."
        )

    lst_by_name: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lst_by_name[row["site_name"]] = row

    toml_sites = _load_toml("sites.toml")["site"]
    merged = []
    for s in toml_sites:
        name = s["name"]
        lst = lst_by_name.get(name, {})
        if not lst:
            raise KeyError(
                f"Site '{name}' found in sites.toml but missing from "
                f"{csv_path.name}. Re-run extract_lst.py."
            )
        merged.append(
            {
                # Identity
                "name": name,
                "dms": s["dms"],
                "site_no": s["site_no"],
                "dominant_species": s["dominant_species"],
                # Model inputs (mapped from grid column names)
                "T_base_mean": float(lst["meanLST_2022"]),
                "G": float(lst["Green_area"]),
                "W": float(lst["Waterbody"]),
                "B": float(lst["Built_area"]),
                # Additional context (available if model needs them later)
                "grid_id": int(float(lst["id"])),
                "Elevation": float(lst["Elevation"]),
                "EHcluster": int(float(lst["EHcluster"])),
            }
        )
    return merged


# ═════════════════════════════════════════════════════════════
# Species
# ═════════════════════════════════════════════════════════════
SPECIES_PARAMS = _build_species_params(_load_toml("species.toml"))
SPECIES_LIST = list(SPECIES_PARAMS.keys())
SPECIES_INDEX = {sp: i for i, sp in enumerate(SPECIES_LIST)}

# ═════════════════════════════════════════════════════════════
# Simulation
# ═════════════════════════════════════════════════════════════
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
T_THRESH_SCALE = _comm["T_THRESH_SCALE"]

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
MIN_BIN_SIZE = _sim["visualization"]["MIN_BIN_SIZE"]

NEST_POSITIONS = {sp: tuple(coords) for sp, coords in _sim["nest_positions"].items()}

# ═════════════════════════════════════════════════════════════
# Sites (processed LST dataset)
# ═════════════════════════════════════════════════════════════
NCR_SITES = _load_sites_with_lst()

# ═════════════════════════════════════════════════════════════
# Agent States
# ═════════════════════════════════════════════════════════════
INACTIVE = 0
SEARCHING = 1
RETURNING = 2
INCAPACITATED = 3
