"""
make_sign_guide.py - draws one picture per letter for the website's Sign Guide.

Usage (from the src folder):   python make_sign_guide.py

The pictures come from YOUR recordings: for each letter it takes the training-session
samples, finds the typical one (the sample closest to the middle of all of them) and
draws its 21 hand points as a stick hand. J and Z are drawn as the path the fingertip
takes. Files go to  frontend/signs/A.svg ... Z.svg.

The pictures are drawn as you see yourself in the camera (like a mirror), with finger
colours: thumb orange, index blue, middle green, ring purple, little finger red.
Bigger dots are closer to the camera.

Want real photos instead? Put your own photos in  frontend/signs/photos/A.jpg ...
(or .png): the guide page shows a photo first when one exists.
"""
import glob
import os

import numpy as np
import pandas as pd

import config as C
import features as F

OUT_DIR = os.path.join(C.PROJECT_ROOT, "frontend", "signs")
TRAIN = [s for s, split in C.SESSION_SPLIT.items() if split == "train"]
COORDS = [f"{c}{i}" for i in range(21) for c in "xyz"]

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
         (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
         (5, 9), (9, 13), (13, 17)]
PALM = {(0, 5), (0, 17), (5, 9), (9, 13), (13, 17)}
FINGER_OF = {i: f for f, idx in enumerate([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                                          (13, 14, 15, 16), (17, 18, 19, 20)]) for i in idx}
COLORS = ["#C9822B", "#2F6DB5", "#2E8B63", "#8E5BB5", "#C0504D"]
PALM_COLOR = "#8a96a3"


def session_files(label):
    out = []
    for s in TRAIN:
        out += glob.glob(os.path.join(C.RAW_DIR, f"session_{s:02d}", f"{label}.csv"))
        out += glob.glob(os.path.join(C.RAW_DIR, f"session_{s}", f"{label}.csv"))
    return sorted(set(out))


def typical_hand(label):
    """Typical sample (as 21x3, right-hand view, wrist at 0, palm size 1) or None."""
    shapes = []
    for path in session_files(label):
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            p = r[COORDS].to_numpy(float).reshape(21, 3)
            rel = (p - p[0]) / F.palm_scale(p)
            if r["hand"] == "Left":
                rel[:, 0] *= -1                     # show every hand as a right hand
            shapes.append(rel)
    if not shapes:
        return None
    arr = np.array(shapes)
    median = np.median(arr, axis=0)
    best = int(np.argmin(((arr - median) ** 2).sum(axis=(1, 2))))
    return arr[best]


def typical_path(label, tip):
    """Median fingertip path (steps x 2) of J or Z, in palm sizes from the start wrist."""
    paths = []
    for path in session_files(label):
        d = pd.read_csv(path)
        d = d[(d.frame >= 0) & (d.frame < C.DYNAMIC_FRAMES)]
        for (_, _), g in d.groupby(["hand", "sample_id"]):
            hand = g["hand"].iloc[0]
            v = F.dynamic_features(g[COORDS].to_numpy(float), g["t"].to_numpy(float), hand)
            paths.append(v.reshape(F.DYNAMIC_STEPS, 5, 3)[:, tip, :2])
    if not paths:
        return None
    return np.median(np.array(paths), axis=0)


# ---------------- drawing ----------------
def hand_svg(rel, box):
    x0, y0, w, h = box
    lines, dots = [], []
    for a, b in BONES:
        col = PALM_COLOR if (a, b) in PALM else COLORS[FINGER_OF[b]]
        lines.append(f'<line x1="{rel[a][0]:.3f}" y1="{rel[a][1]:.3f}" x2="{rel[b][0]:.3f}" '
                     f'y2="{rel[b][1]:.3f}" stroke="{col}" stroke-width="0.075" stroke-linecap="round"/>')
    order = np.argsort(-rel[:, 2])                  # far joints first, near joints on top
    for i in order:
        near = float(np.clip(-rel[i][2], -0.5, 0.5))
        r = 0.065 + 0.07 * near
        col = PALM_COLOR if i == 0 else COLORS[FINGER_OF[i]]
        dots.append(f'<circle cx="{rel[i][0]:.3f}" cy="{rel[i][1]:.3f}" r="{r:.3f}" fill="{col}" '
                    f'stroke="white" stroke-width="0.03"/>')
    return wrap("".join(lines + dots), box)


def path_svg(pts, box, color):
    x0, y0, w, h = box
    d = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
    a, b = pts[-2], pts[-1]
    ang = np.arctan2(b[1] - a[1], b[0] - a[0])
    tip = np.array(b)
    left = tip - 0.28 * np.array([np.cos(ang - 0.5), np.sin(ang - 0.5)])
    right = tip - 0.28 * np.array([np.cos(ang + 0.5), np.sin(ang + 0.5)])
    body = (f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="0.08" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<polygon points="{tip[0]:.3f},{tip[1]:.3f} {left[0]:.3f},{left[1]:.3f} {right[0]:.3f},{right[1]:.3f}" fill="{color}"/>'
            f'<circle cx="{pts[0][0]:.3f}" cy="{pts[0][1]:.3f}" r="0.13" fill="#2E8B63" stroke="white" stroke-width="0.03"/>')
    return wrap(body, box)


def wrap(inner, box):
    x0, y0, w, h = box
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.3f} {y0:.3f} {w:.3f} {h:.3f}" '
            f'width="300" height="{300 * h / w:.0f}">{inner}</svg>')


def common_box(items, margin=0.35):
    pts = np.vstack([np.asarray(i)[:, :2] for i in items])
    x_half = max(abs(pts[:, 0].min()), abs(pts[:, 0].max())) + margin
    y_lo, y_hi = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    return (-x_half, y_lo, 2 * x_half, y_hi - y_lo)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hands = {L: typical_hand(L) for L in C.STATIC_LETTERS}
    have = {L: h for L, h in hands.items() if h is not None}
    done, missing = [], [L for L in C.STATIC_LETTERS if L not in have]

    if have:
        box = common_box(have.values())              # same scale for every letter
        for L, rel in have.items():
            with open(os.path.join(OUT_DIR, f"{L}.svg"), "w") as f:
                f.write(hand_svg(rel, box))
            done.append(L)

    for L, tip, color in (("J", 4, "#C0504D"), ("Z", 1, "#2F6DB5")):   # pinky tip / index tip
        pts = typical_path(L, tip)
        if pts is None:
            missing.append(L)
            continue
        box = common_box([pts], margin=0.6)
        with open(os.path.join(OUT_DIR, f"{L}.svg"), "w") as f:
            f.write(path_svg(pts, box, color))
        done.append(L)

    print(f"drawn: {' '.join(sorted(done))}")
    if missing:
        print(f"no recordings found for: {' '.join(missing)} (those cards show text only)")
    print("saved in", os.path.relpath(OUT_DIR, C.PROJECT_ROOT))


if __name__ == "__main__":
    main()