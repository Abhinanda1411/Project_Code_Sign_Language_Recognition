"""
motion.py  -  the dynamic-gesture start rule.

The collection script and (later) the live app both use this class, so a
gesture starts under exactly the same condition when recording data and
when recognizing.
"""

import math
from collections import deque

import config as C


def landmark_motion(current_points, previous_points):
    """Average distance moved by the 21 landmarks between two frames."""
    if previous_points is None:
        return 0.0

    total = 0.0

    for (cx, cy, cz), (px, py, pz) in zip(current_points, previous_points):
        dx = cx - px
        dy = cy - py
        dz = cz - pz
        total += math.sqrt(dx * dx + dy * dy + dz * dz)

    return total / 21.0


class MotionTrigger:
    """
    Fires when the 5-frame average motion has been >= MOTION_THRESHOLD
    for DYNAMIC_START_FRAMES frames in a row.

    quiet_frames counts how many frames in a row were BELOW the threshold
    (used to wait for the hand to settle between recordings).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_points = None
        self.motion_history = deque(maxlen=C.MOTION_WINDOW)
        self.start_history = deque(maxlen=C.DYNAMIC_START_FRAMES)
        self.quiet_frames = 0

    def update(self, points):
        motion = landmark_motion(points, self.previous_points)
        self.previous_points = points
        self.motion_history.append(motion)

        average = sum(self.motion_history) / len(self.motion_history)
        moving = average >= C.MOTION_THRESHOLD

        self.start_history.append(moving)
        self.quiet_frames = 0 if moving else self.quiet_frames + 1

        triggered = (
            len(self.start_history) >= C.DYNAMIC_START_FRAMES
            and all(self.start_history)
        )

        return average, triggered