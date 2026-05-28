"""
extract_lst.py
--------------
For each site in sites.toml, finds the single grid cell (polygon) that
contains it and extracts the columns the simulation model needs.

Output: data/processed/sites_with_lst.csv  — exactly one row per site.

Columns in output
-----------------
  site_name   : from sites.toml
  id          : grid cell identifier
  meanLST     : mean LST 2000-2023  → used as T_base_mean in the model
  Green_area  : greenery area (ha)  → used as G
  Waterbody   : 0/1 water flag      → used as W
  Built_area  : built-up area (ha)  → used as B
  Elevation   : metres above sea level
  EHcluster   : 0/1 extreme heat cluster membership

Run from the project root:
    python src/scripts/extract_lst.py
"""

import sys
import logging
import tomllib
from pathlib import Path

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
        columns=GRID_COLS,  # geopandas 0.12+ supports column filtering
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

    # 5. Keep only the columns config.py will read
    out = joined[["site_name", "latitude", "longitude"] + GRID_COLS].reset_index(
        drop=True
    )

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
    log.info(
        "\n%s",
        out[
            [
                "site_name",
                "id",
                "meanLST",
                "Green_area",
                "Waterbody",
                "Built_area",
                "EHcluster",
            ]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
