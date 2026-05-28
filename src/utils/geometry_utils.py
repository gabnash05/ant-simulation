"""geometry_utils.py — Coordinate conversion helpers for the LST extraction pipeline."""

import re
from typing import Tuple


def dms_to_decimal(dms_str: str) -> Tuple[float, float]:
    """
    Convert a DMS string to (latitude, longitude) in decimal degrees.

    Supports formats like:
        "14° 41' 35\" N, 120° 58' 19\" E"
        "14°41'35\"N, 120°58'19\"E"

    Returns
    -------
    (lat, lon) : tuple of floats
        Positive = North/East, negative = South/West.

    Examples
    --------
    >>> lat, lon = dms_to_decimal("14° 41' 35\\\" N, 120° 58' 19\\\" E")
    >>> round(lat, 4), round(lon, 4)
    (14.6931, 120.9719)
    """

    pattern = r"(\d+)°\s*(\d+)'\s*(\d+(?:\.\d+)?)\"\s*([NSEWnsew])"
    matches = re.findall(pattern, dms_str)

    if len(matches) != 2:
        raise ValueError(
            f"Expected 2 coordinate components (lat + lon), "
            f"found {len(matches)} in: '{dms_str}'"
        )

    decimals = []
    for deg, minute, second, direction in matches:
        decimal = int(deg) + int(minute) / 60 + float(second) / 3600
        if direction.upper() in ("S", "W"):
            decimal = -decimal
        decimals.append(decimal)

    return (decimals[0], decimals[1])  # (lat, lon)


def decimal_to_dms(lat: float, lon: float) -> str:
    """
    Convert decimal degrees back to a DMS string (for display/verification).
    """

    def _fmt(value: float, pos_dir: str, neg_dir: str) -> str:
        direction = pos_dir if value >= 0 else neg_dir
        value = abs(value)
        deg = int(value)
        mins = int((value - deg) * 60)
        secs = round(((value - deg) * 60 - mins) * 60, 1)
        return f"{deg}° {mins}' {secs}\" {direction}"

    return f"{_fmt(lat, 'N', 'S')}, {_fmt(lon, 'E', 'W')}"
