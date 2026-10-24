"""
gestures.py - hand gestures that are commands, not letters.

  open palm  (both hands, held)  -> clear the text
  thumbs up  (held)              -> speak the text

These are simple geometry rules on the 21 landmarks, not a trained model.
The numbers below are starting values: check them with your own hands using the
"Open palms" and "Thumbs up" indicators on the web page.

OPEN PALM : four fingers stretched out AND the thumb stretched out
            (a letter such as B keeps the thumb folded, so it does not count).
THUMBS UP : four fingers curled, thumb straight, and the thumb tip clearly
            ABOVE the knuckles (an upright hand). Letter A keeps the thumb
            beside the fist, so its thumb tip is not that high.
"""
import numpy as np

FINGER_RATIO = 1.25        # open palm: tip this much farther from the wrist than the middle joint
THUMB_RATIO = 1.10         # thumb tip farther from the pinky base than the thumb's own joint
THUMB_UP_REACH = 0.60      # thumbs up: thumb tip this far (in palm sizes) from the index knuckle
THUMB_UP_MARGIN = 0.50     # thumbs up: thumb tip this far ABOVE the index knuckle (palm sizes)

HOLD_SECONDS = 0.8         # a command gesture must be held this long
GRACE_FRAMES = 3           # a detection flicker this short does not restart the hold

FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))     # (middle joint, tip) of index..pinky


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _points(points):
    return np.asarray(points, dtype=float).reshape(21, 3)


def _thumb_straight(p):
    return bool(_d(p[4], p[17]) > THUMB_RATIO * _d(p[3], p[17]))


def is_open_palm(points):
    p = _points(points)
    wrist = p[0]

    if _d(wrist, p[9]) < 1e-6:
        return False

    for pip, tip in FINGERS:
        if _d(p[tip], wrist) < FINGER_RATIO * _d(p[pip], wrist):
            return False

    return bool(_thumb_straight(p))


def is_thumbs_up(points):
    p = _points(points)
    wrist = p[0]
    scale = _d(wrist, p[9])

    if scale < 1e-6:
        return False

    for pip, tip in FINGERS:                          # all four fingers curled
        if _d(p[tip], wrist) >= _d(p[pip], wrist):
            return False

    if not _thumb_straight(p):
        return False

    if _d(p[4], p[5]) < THUMB_UP_REACH * scale:       # thumb sticks out from the fist
        return False

    return bool((p[5][1] - p[4][1]) > THUMB_UP_MARGIN * scale)   # image y grows downward


class HoldGesture:
    """
    True ONCE when `active` has stayed true for hold_seconds. The gesture must be
    released (for more than grace_frames frames) before it can fire again.
    """

    def __init__(self, hold_seconds=HOLD_SECONDS, grace_frames=GRACE_FRAMES):
        self.hold_seconds = hold_seconds
        self.grace_frames = grace_frames
        self.reset()

    def reset(self):
        self.start = None
        self.misses = 0
        self.armed = True
        self.elapsed = 0.0

    def update(self, active, now):
        if active:
            self.misses = 0
            if self.start is None:
                self.start = now
        else:
            self.misses += 1
            if self.misses > self.grace_frames:
                self.start = None
                self.armed = True

        self.elapsed = (now - self.start) if self.start is not None else 0.0

        if self.armed and self.start is not None and self.elapsed >= self.hold_seconds:
            self.armed = False
            return True
        return False

    @property
    def progress(self):
        return min(self.elapsed / self.hold_seconds, 1.0) if self.armed else 0.0


ClearGesture = HoldGesture      # old name