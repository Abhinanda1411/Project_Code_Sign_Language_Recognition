"""
config.py  -  ONE place for every setting shared by the whole project.

The collection script, the training notebooks and the live app all read
their numbers from here (or from a copy of this file), so a value can never
silently differ between recording and recognition.
"""

import os

# ============================================================
# FOLDERS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SESSIONS_FILE = os.path.join(PROJECT_ROOT, "data", "sessions.csv")

# ============================================================
# CLASSES
# ============================================================

# Static letters (J and Z need movement, so they are dynamic)
STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# REST = relaxed hand, pointing, in-between poses ("not a letter")
STATIC_CLASSES = STATIC_LETTERS + ["REST"]

# NONE = hand movement that is NOT a J and NOT a Z
DYNAMIC_CLASSES = ["J", "Z", "NONE"]

# Recording order inside one session and one hand
ALL_CLASSES = STATIC_CLASSES + DYNAMIC_CLASSES

# ============================================================
# SESSIONS AND HANDS
# ============================================================

NUM_SESSIONS = 6

# Hands to record in EVERY session. Right block first, then Left block.
# Demo with one hand only?  ->  HANDS = ["Right"]   (or ["Left"])
HANDS = ["Right", "Left"]

# How the sessions will be used in Colab (documentation + sessions.csv)
SESSION_SPLIT = {1: "train", 2: "train", 3: "train", 4: "train",
                 5: "validation", 6: "test"}

# Samples per class, per session, PER HAND
STATIC_SAMPLES_PER_HAND = 25
DYNAMIC_SAMPLES_PER_HAND = 30

# ============================================================
# CAMERA / MEDIAPIPE  (use the SAME values in the live app)
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# The picture is mirrored (like a selfie) before MediaPipe sees it.
# Check on first run: show your RIGHT hand -> the screen must say "Right".
# If it says "Left", set this to True.
FLIP_HANDEDNESS = False

# ============================================================
# STATIC SAMPLE QUALITY  (starting values - tune after the pilot)
# ============================================================

# A new static sample is saved only if the hand changed at least this much
# since the last saved one (average landmark movement, picture units).
DIVERSITY_THRESHOLD = 0.03
MIN_SAVE_INTERVAL = 0.15      # seconds

MIN_HAND_SCORE = 0.8          # MediaPipe's left/right confidence
EDGE_MARGIN = 0.02            # whole hand must stay inside the picture

# ============================================================
# DYNAMIC (J / Z / NONE) - must match the live app
# ============================================================

DYNAMIC_FRAMES = 30
MOTION_WINDOW = 5
MOTION_THRESHOLD = 0.015
DYNAMIC_START_FRAMES = 6

# Frames saved from BEFORE the trigger fired (frame numbers -10 ... -1)
PRE_ROLL_FRAMES = 10

# After a recording, the hand must be quiet this long before re-arming
SETTLE_FRAMES = 8
MIN_REARM_SECONDS = 0.6