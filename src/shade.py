"""shade.py — Stochastic shade grid generation."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from config import GRID_W, GRID_H


def generate_shade(G: float, W: float, B: float) -> np.ndarray:
    p_shade = max(0.0, 0.8 * G + 0.1 * W - 0.3 * B)

    seed = (np.random.rand(GRID_W, GRID_H) < p_shade).astype(np.float32)

    padded = np.pad(seed, 1, mode="edge")

    return sliding_window_view(padded, (3, 3)).mean(axis=(-1, -2)).astype(np.float32)
