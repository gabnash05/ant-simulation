"""
extract_lst.py
--------------
For each site in sites.toml, finds the single grid cell (polygon) that
contains it and extracts the columns the simulation model needs.
Additionally, merges temperature data for 2022 from an external tab file.

Output: data/processed/sites_with_lst.csv  — exactly one row per site.

Columns in output
-----------------
  site_name   : from sites.toml
  id          : grid cell identifier
  meanLST     : mean LST 2000-2023  (retained for reference)
  Green_area  : greenery area (ha)  → used as G
  Waterbody   : 0/1 water flag      → used as W
  Built_area  : built-up area (ha)  → used as B
  Elevation   : metres above sea level
  EHcluster   : 0/1 extreme heat cluster membership
  2022-* : Monthly temperature variables for 2022
  meanLST_2022: Annual mean temperature for the 2022 study period
                → used as T_base_mean in the model

Run from the project root:
    python src/scripts/extract_lst.py
"""

import sys
import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))
from utils.geometry_utils import dms_to_decimal

CONFIG_DIR = SRC_DIR / "config"
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

GRID_COLS = [
    "id",
    "meanLST",
    "Green_area",
    "Waterbody",
    "Built_area",
    "Elevation",
    "EHcluster",
]

LST_CSV_FILENAME = "Manila_csv.tab"
LST_MAX_VALID = 100


def load_sites() -> gpd.GeoDataFrame:
    with open(CONFIG_DIR / "sites.toml", "rb") as f:
        config = tomllib.load(f)
    records = []
    for s in config["site"]:
        lat, lon = dms_to_decimal(s["dms"])
        records.append({"site_name": s["name"], "latitude": lat, "longitude": lon})
    df = pd.DataFrame(records)
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )


def main() -> None:
    # 1. Sites from TOML
    sites = load_sites()
    log.info("Sites loaded: %d", len(sites))

    # 2. Grid shapefile — only load the columns we need, not all 26
    grid = gpd.read_file(
        DATA_RAW / "Manila.shp",
        columns=GRID_COLS,
    )
    log.info("Grid loaded: %d polygons, CRS=%s", len(grid), grid.crs)

    # 3. Reproject sites to grid CRS (UTM Zone 51N) before joining
    sites_proj = sites.to_crs(grid.crs)

    # 4. Point-in-polygon: each site → the one cell that contains it
    joined = gpd.sjoin(
        sites_proj,
        grid[["geometry"] + GRID_COLS],
        how="left",
        predicate="within",
    )

    # Fallback: sites sitting exactly on a polygon boundary use nearest cell
    missed = joined["id"].isna()
    if missed.any():
        log.warning(
            "Boundary fallback for: %s",
            joined.loc[missed, "site_name"].tolist(),
        )
        nearest = gpd.sjoin_nearest(
            sites_proj[missed.values],
            grid[["geometry"] + GRID_COLS],
            how="left",
        )
        for col in GRID_COLS:
            joined.loc[missed, col] = nearest[col].values

    # 5. Extract 2022 temperatures from the TAB file and merge
    csv_path = DATA_RAW / LST_CSV_FILENAME
    if csv_path.exists():
        lst_df = pd.read_csv(csv_path, sep="\t")

        cols_2022 = [col for col in lst_df.columns if str(col).startswith("2022-")]
        if not cols_2022:
            log.error(
                "No '2022-*' columns found in %s. Check column names.",
                LST_CSV_FILENAME,
            )

        lst_2022 = lst_df[["id"] + cols_2022].copy()

        # ── Filter: replace values > LST_MAX_VALID with NaN ──────────
        n_filtered = 0
        for col in cols_2022:
            mask = lst_2022[col] > LST_MAX_VALID
            n_filtered += mask.sum()
            lst_2022.loc[mask, col] = np.nan

        if n_filtered:
            log.info(
                "Filtered %d temperature value(s) above %d°C → set to NaN.",
                n_filtered,
                LST_MAX_VALID,
            )

        # ── Compute mean, ignoring NaN months ────────────────────────
        lst_2022["meanLST_2022"] = lst_2022[cols_2022].mean(axis=1)

        joined = joined.merge(lst_2022, on="id", how="left")

        n_missing = joined["meanLST_2022"].isna().sum()
        if n_missing:
            log.warning(
                "%d site(s) have no valid 2022 LST data — T_base_mean will be NaN.",
                n_missing,
            )

        log.info(
            "Merged 2022 temperature data from %s (%d monthly columns).",
            LST_CSV_FILENAME,
            len(cols_2022),
        )
    else:
        log.error(
            "Data file %s not found. Proceeding without 2022 data.",
            csv_path,
        )
        cols_2022 = []
        joined["meanLST_2022"] = pd.NA

    # 6. Keep only the columns config.py will read (now including 2022 data)
    final_cols = (
        ["site_name", "latitude", "longitude"]
        + GRID_COLS
        + cols_2022
        + ["meanLST_2022"]
    )
    out = joined[final_cols].reset_index(drop=True)

    unmatched = out["id"].isna().sum()
    if unmatched:
        log.error(
            "No grid cell found for %d site(s): %s",
            unmatched,
            out.loc[out["id"].isna(), "site_name"].tolist(),
        )

    out_path = DATA_PROCESSED / "sites_with_lst.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved %d rows → %s", len(out), out_path)

    display_cols = [c for c in [
        "site_name", "id", "meanLST", "meanLST_2022", "Green_area", "EHcluster",
    ] if c in out.columns]
    log.info("\n%s", out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()