"""shade.py — Stochastic shade grid generation."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from config import GRID_W, GRID_H


# ═════════════════════════════════════════════════════════════
# Monte Carlo Shade Generation
# ═════════════════════════════════════════════════════════════


def generate_shade(G: float, W: float, B: float) -> np.ndarray:
    """
    Section III-F — Monte Carlo Shade Generation

    Builds the static shade coefficient field s(x,y) from land-cover inputs.
    """
    # Eq. (15): p_shade = max(0, 0.8G + 0.1W − 0.3B)
    p_shade = max(0.0, 0.8 * G + 0.1 * W - 0.3 * B)

    seed = (np.random.rand(GRID_W, GRID_H) < p_shade).astype(np.float32)

    padded = np.pad(seed, 1, mode="edge")

    # Eq. (16): s(x,y) = (1/9) Σ neighborhood (3×3 box blur)
    return sliding_window_view(padded, (3, 3)).mean(axis=(-1, -2)).astype(np.float32)
