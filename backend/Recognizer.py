"""
recognizer.py - turns a stream of hand detections into typed letters.

Pure logic: no camera, no web. It gets one detection per video frame and says
what (if anything) should be typed. Everything that decides "when is a gesture
a letter" lives here, so it can be tested without a webcam.

Static letters : the same letter must be predicted STABLE_FRAMES frames in a row.
J / Z          : same trigger and 30-frame window as in collect_data.py.
"""
import os
import sys
from collections import Counter, deque

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config as C                      # noqa: E402
import features as F                    # noqa: E402
from motion import MotionTrigger        # noqa: E402

# ---- live settings (starting values - tune in the live trials) ----
STABLE_FRAMES = 15      # a letter must be predicted this many frames in a row (~0.5 s)
UNLOCK_FRAMES = 5       # a DIFFERENT pose this long lets the same letter be typed again
LOST_FRAMES = 10        # no hand for this long also lets the same letter repeat
QUIET_FRAMES = 3        # hand must be nearly still before a static letter is predicted


class Recognizer:
    def __init__(self, static_bundle, dynamic_bundle):
        self.static_model = static_bundle["model"]
        self.static_kind = static_bundle["feature_kind"]
        self.dynamic_model = dynamic_bundle["model"]
        self.steps = dynamic_bundle.get("steps", F.DYNAMIC_STEPS)
        self.reset()

    def reset(self):
        self.trigger = MotionTrigger()
        self.state = "SETTLE"            # SETTLE -> ARMED -> RECORDING -> SETTLE
        self.settle_since = None
        self.recorded = []               # (time, points, label) of the current J/Z attempt
        self.history = deque(maxlen=STABLE_FRAMES)
        self.locked = None               # letter already typed, waiting for a change
        self.unlock_count = 0
        self.pending_lock = False        # after J/Z: ignore the pose the hand ends in
        self.lost = 0

    # ------------------------------------------------------------
    @staticmethod
    def _inside(points):
        low, high = C.EDGE_MARGIN, 1.0 - C.EDGE_MARGIN
        return all(low <= x <= high and low <= y <= high for x, y, _ in points)

    def _static_ok(self, result):
        """Same quality gate that the training samples had to pass."""
        return (result.score >= C.MIN_HAND_SCORE
                and self._inside(result.points))

    # ------------------------------------------------------------
    def update(self, result, now):
        """
        result: HandResult or None (one video frame).
        Returns dict: hand, state, prediction, hold (0-1), committed, dynamic.
        'committed' is the letter to type now (or None).
        """
        out = dict(hand=result is not None, state=self.state, prediction=None,
                   hold=0.0, committed=None, dynamic=None)

        usable = result is not None and result.label in ("Left", "Right")

        # hand lost -> the same letter may be typed again
        self.lost = 0 if result is not None else self.lost + 1
        if self.lost >= LOST_FRAMES:
            self.locked = None
            self.unlock_count = 0

        if usable:
            _, triggered = self.trigger.update(result.points)
        else:
            self.trigger.reset()
            triggered = False

        if self.settle_since is None:
            self.settle_since = now

        # ---------------- J / Z state machine ----------------
        if self.state == "RECORDING":
            if not usable:
                self._discard(now)
            else:
                self.recorded.append((now, result.points, result.label))
                if len(self.recorded) >= C.DYNAMIC_FRAMES:
                    self._finish_recording(out, now)

        elif self.state == "SETTLE":
            if (usable and self.trigger.quiet_frames >= C.SETTLE_FRAMES
                    and now - self.settle_since >= C.MIN_REARM_SECONDS):
                self.state = "ARMED"

        elif self.state == "ARMED":
            if usable and triggered:
                self.state = "RECORDING"
                self.recorded = [(now, result.points, result.label)]
                self.pending_lock = False
                self.history.clear()

        out["state"] = self.state

        # ---------------- static letters ----------------
        if (usable and self.state != "RECORDING" and out["committed"] is None
                and self._static_ok(result)
                and self.trigger.quiet_frames >= QUIET_FRAMES):
            self._static(result, out)
        else:
            self.history.clear()

        return out

    # ------------------------------------------------------------
    def _discard(self, now):
        self.recorded = []
        self.state = "SETTLE"
        self.settle_since = now

    def _finish_recording(self, out, now):
        labels = [lab for _, _, lab in self.recorded]
        hand = Counter(labels).most_common(1)[0][0]
        t0 = self.recorded[0][0]
        points = [p for _, p, _ in self.recorded]
        times = [t - t0 for t, _, _ in self.recorded]

        x = F.dynamic_features(points, times, hand, self.steps)
        label = str(self.dynamic_model.predict([x])[0])
        out["dynamic"] = label

        if label in ("J", "Z"):
            out["committed"] = label
            self.pending_lock = True     # the pose the hand ends in must not be typed
            self.locked = None
            self.unlock_count = 0

        self.recorded = []
        self.state = "SETTLE"
        self.settle_since = now
        self.history.clear()

    def _static(self, result, out):
        x = F.static_features(result.points, result.label, self.static_kind)
        pred = str(self.static_model.predict([x])[0])
        self.history.append(pred)

        run = 0
        for p in reversed(self.history):
            if p != pred:
                break
            run += 1

        out["prediction"] = pred
        out["hold"] = min(run / STABLE_FRAMES, 1.0)

        # a different pose long enough -> the locked letter may be typed again
        if self.locked is not None:
            if pred != self.locked:
                self.unlock_count += 1
                if self.unlock_count >= UNLOCK_FRAMES:
                    self.locked = None
                    self.unlock_count = 0
            else:
                self.unlock_count = 0

        if run < STABLE_FRAMES:
            return

        if pred == "REST":
            self.locked = None           # REST = "reset": the same letter can follow again
            self.pending_lock = False
            return

        if pred == self.locked:
            return

        if self.pending_lock:            # end pose of a J/Z: remember it, do not type it
            self.locked = pred
            self.pending_lock = False
            return

        out["committed"] = pred
        self.locked = pred
        self.unlock_count = 0