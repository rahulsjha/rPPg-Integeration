"""rPPG-Toolbox POS backend integration.

This module implements the POS_WANG unsupervised method pattern from:
https://github.com/ubicomplab/rPPG-Toolbox

The implementation here is adapted to run as a lightweight backend in a
chunked near real-time pipeline.
"""

import math

import numpy as np


def process_video(frames: np.ndarray) -> np.ndarray:
    """Calculate per-frame RGB spatial means."""
    rgb = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        rgb.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(rgb, dtype=np.float64)


def pos_wang(frames: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """POS_WANG rPPG signal extraction."""
    rgb = process_video(frames)
    n = rgb.shape[0]
    if n < 4:
        return np.zeros(n, dtype=np.float64)

    h = np.zeros((1, n), dtype=np.float64)
    l = max(2, int(math.ceil(win_sec * fs)))

    for t in range(n):
        m = t - l
        if m >= 0:
            c = rgb[m:t, :]
            c = np.true_divide(c, np.mean(c, axis=0, keepdims=True) + 1e-8)
            c = np.asarray(c).T

            s = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64), c)
            denom = np.std(s[1, :]) + 1e-8
            p = s[0, :] + (np.std(s[0, :]) / denom) * s[1, :]
            p = p - np.mean(p)
            h[0, m:t] += p

    out = np.asarray(h).reshape(-1)
    out = out - np.mean(out)
    std = np.std(out)
    if std > 1e-8:
        out = out / std
    return out
