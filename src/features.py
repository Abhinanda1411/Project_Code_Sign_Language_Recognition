"""
features.py - turns hand landmarks into model inputs.

THE SAME FILE is used in Colab (training) and in the website (live), so a
feature can never be computed differently in the two places.
No project imports on purpose: it must work anywhere.
"""
import numpy as np

WRIST = 0
MIDDLE_MCP = 9                 # base of the middle finger = palm size reference
TIPS = [4, 8, 12, 16, 20]      # thumb, index, middle, ring, pinky tips

# Static feature sets that the notebook compares on the validation session
STATIC_KINDS = ["raw", "norm", "norm_mirror", "norm_mirror_dist"]
DYNAMIC_STEPS = 20             # time points a J/Z gesture is resampled to


def _points(points):
    return np.asarray(points, dtype=float).reshape(21, 3)


def palm_scale(p):
    """Distance wrist -> middle-finger base: makes features independent of hand size / distance."""
    d = float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]))
    return d if d > 1e-6 else 1e-6


def static_features(points, hand, kind="norm_mirror_dist"):
    """
    points: 21 x (x, y, z).  hand: "Right" or "Left".
      raw               63 raw picture coordinates (old style, baseline)
      norm              wrist moved to (0,0,0) and divided by palm size
      norm_mirror       norm + Left hands flipped so they look like Right hands
      norm_mirror_dist  norm_mirror + distances between fingertips / wrist
    """
    p = _points(points)

    if kind == "raw":
        return p.reshape(-1)

    rel = (p - p[WRIST]) / palm_scale(p)

    if kind in ("norm_mirror", "norm_mirror_dist") and hand == "Left":
        rel[:, 0] *= -1

    features = rel.reshape(-1)

    if kind == "norm_mirror_dist":
        tips = rel[TIPS]
        extra = []
        for i in range(5):
            extra.append(np.linalg.norm(tips[i] - rel[WRIST]))
            for j in range(i + 1, 5):
                extra.append(np.linalg.norm(tips[i] - tips[j]))
        features = np.concatenate([features, extra])

    return features


def dynamic_features(points_seq, times, hand, steps=DYNAMIC_STEPS):
    """
    One J / Z / NONE gesture -> fixed-length vector.

    points_seq: T x 21 x 3 landmarks, times: T seconds.
    All 5 fingertips are used (not one), positions are measured from the
    wrist at the START of the gesture in palm-size units, and the path is
    resampled by TIME to `steps` points so frame-rate differences vanish.
    """
    P = np.asarray(points_seq, dtype=float).reshape(-1, 21, 3)
    t = np.asarray(times, dtype=float)

    order = np.argsort(t, kind="stable")
    P, t = P[order], t[order]

    origin = P[0, WRIST]
    scale = palm_scale(P[0])
    rel = (P[:, TIPS, :] - origin) / scale          # T x 5 x 3

    if hand == "Left":
        rel[:, :, 0] *= -1

    if t[-1] - t[0] < 1e-6:                         # degenerate: all same time
        grid = np.arange(len(t), dtype=float)
        t = grid
    grid = np.linspace(t[0], t[-1], steps)

    out = np.empty((steps, 5, 3))
    for k in range(5):
        for c in range(3):
            out[:, k, c] = np.interp(grid, t, rel[:, k, c])

    return out.reshape(-1)