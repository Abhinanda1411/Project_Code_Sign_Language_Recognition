"""
hand_tracker.py  -  the ONLY file that talks to MediaPipe.

Every script (collection now, live app later) gets hand landmarks through
this wrapper. If the MediaPipe interface ever has to change, only this file
changes.

Uses the classic mp.solutions.hands interface, so the project needs a
MediaPipe version that still has it (for example: pip install mediapipe==0.10.21).
"""

import cv2
import mediapipe as mp

import config as C


class HandResult:
    """One detected hand."""

    def __init__(self, points, label, score, landmarks=None):
        self.points = points          # 21 x (x, y, z), picture units
        self.label = label            # "Right" or "Left"
        self.score = score            # MediaPipe's left/right confidence
        self.landmarks = landmarks    # raw MediaPipe object (for drawing)


def read_frame(cap):
    """Read one camera frame and mirror it (selfie view). None on failure."""
    ok, frame = cap.read()

    if not ok:
        return None

    return cv2.flip(frame, 1)


class HandTracker:
    def __init__(self, max_hands=1):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe version has no 'mp.solutions'. "
                "Install a version that still has it, e.g.:\n"
                "    pip uninstall -y mediapipe\n"
                "    pip install mediapipe==0.10.21"
            )

        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=C.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=C.MIN_TRACKING_CONFIDENCE
        )

    def process_all(self, rgb_frame):
        """Return a list of HandResult (0, 1 or 2 hands)."""
        results = self._hands.process(rgb_frame)
        hands = []

        if not results.multi_hand_landmarks:
            return hands

        for i, landmarks in enumerate(results.multi_hand_landmarks):
            points = [(p.x, p.y, p.z) for p in landmarks.landmark]

            label = "Unknown"
            score = 0.0

            if results.multi_handedness and i < len(results.multi_handedness):
                top = results.multi_handedness[i].classification[0]
                label = top.label
                score = float(top.score)

            if C.FLIP_HANDEDNESS and label in ("Left", "Right"):
                label = "Left" if label == "Right" else "Right"

            hands.append(HandResult(points, label, score, landmarks))

        return hands

    def process(self, rgb_frame):
        """Return the first HandResult, or None if no hand is visible."""
        hands = self.process_all(rgb_frame)
        return hands[0] if hands else None

    def draw(self, frame, result):
        self._drawer.draw_landmarks(
            frame,
            result.landmarks,
            self._mp_hands.HAND_CONNECTIONS
        )

    def close(self):
        self._hands.close()