"""
collect_data.py  -  record the sign language dataset (static + dynamic).

Run from PyCharm:  right-click this file -> Run.

FILES CREATED
    data/raw/session_01/A.csv, B.csv, ... Y.csv, REST.csv   (static: 1 row = 1 sample)
    data/raw/session_01/J.csv, Z.csv, NONE.csv              (dynamic: 1 sample = many rows,
                                                             grouped by sample_id)
    data/sessions.csv                                       (one row per session)

ORDER ("continue where I left off")
    session 1: Right hand -> every class, then Left hand -> every class
    session 2: ...
  A session is one sitting under one set of conditions, so finish a whole
  session before starting the next one (ideally on a different day / light).

EVERY sample is written to disk the moment it is captured. Pausing, ending,
closing the window or even a crash never loses what was already saved.

CONTROLS (click the buttons or use the keys)
    START (S)   begin collecting
    PAUSE (P)   stop collecting (saved samples stay)
    UNDO  (U)   delete the last sample saved in this run
    END   (E)   stop and go back to the menu  (Q also works)
    SPACE       start / pause
"""

import csv
import importlib.metadata
import os
import time
from collections import deque
from datetime import datetime

import cv2

import config as C
from hand_tracker import HandTracker, read_frame
from motion import MotionTrigger, landmark_motion

WINDOW = "Sign language data collection"

COORD_NAMES = [f"{c}{i}" for i in range(21) for c in "xyz"]
META_COLUMNS = ["label", "hand", "hand_score", "session", "recorded_at"]

STATIC_HEADER = COORD_NAMES + META_COLUMNS
DYNAMIC_HEADER = ["sample_id", "frame", "t"] + COORD_NAMES + META_COLUMNS

SESSIONS_HEADER = [
    "session", "date", "person", "split", "lighting", "background",
    "distance_cm", "camera_resolution", "mediapipe_version",
    "opencv_version", "notes",
]

NONE_HINTS = [
    "Switch between two static letters (e.g. A -> B)",
    "Move your open hand across the screen",
    "Point with your index finger - do NOT draw a Z",
    "Hold a letter and shake it slightly",
    "Wave your hand",
    "Bring your hand in and settle into a letter",
    "Rotate your wrist slowly (no J shape)",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def is_dynamic(label):
    return label in C.DYNAMIC_CLASSES


def target_for(label):
    if is_dynamic(label):
        return C.DYNAMIC_SAMPLES_PER_HAND

    return C.STATIC_SAMPLES_PER_HAND


def session_dir(session):
    return os.path.join(C.RAW_DIR, f"session_{session:02d}")


def class_path(session, label):
    return os.path.join(session_dir(session), f"{label}.csv")


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def flat(points):
    values = []

    for x, y, z in points:
        values.extend((round(x, 6), round(y, 6), round(z, 6)))

    return values


# ============================================================
# FILES
# ============================================================

def ensure_file(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(header)
        return

    with open(path, "r", newline="", encoding="utf-8") as file:
        existing = next(csv.reader(file), None)

    if existing != header:
        raise SystemExit(
            f"\nERROR: {path}\nhas an unexpected header. Rename or move "
            "that file and run again (nothing was changed)."
        )


def append_rows(path, rows):
    with open(path, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


def count_samples(session, label, hand):
    """Samples already saved for this session / class / hand."""
    path = class_path(session, label)

    if not os.path.exists(path):
        return 0

    count = 0

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)

        if not header or "hand" not in header:
            return 0

        hand_index = header.index("hand")
        frame_index = header.index("frame") if "frame" in header else None

        for row in reader:
            if len(row) != len(header) or row[hand_index] != hand:
                continue

            if frame_index is None:
                count += 1
            elif row[frame_index] == "0":
                count += 1          # one dynamic sample = one frame-0 row

    return count


def next_sample_id(path):
    if not os.path.exists(path):
        return 1

    highest = 0

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:
            try:
                highest = max(highest, int(row[0]))
            except (ValueError, IndexError):
                pass

    return highest + 1


def undo_last(path, dynamic):
    """Remove the last sample from a class file. Returns True if removed."""
    if not os.path.exists(path):
        return False

    with open(path, "r", newline="", encoding="utf-8") as file:
        rows = [row for row in csv.reader(file) if row]

    if len(rows) <= 1:
        return False

    header, body = rows[0], rows[1:]

    if dynamic:
        last_id = body[-1][0]

        while body and body[-1][0] == last_id:
            body.pop()
    else:
        body.pop()

    temp = path + ".tmp"

    with open(temp, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(body)

    os.replace(temp, path)
    return True


def save_static(session, label, hand, score, points):
    path = class_path(session, label)
    ensure_file(path, STATIC_HEADER)

    row = flat(points) + [label, hand, round(score, 4), session, now_iso()]
    append_rows(path, [row])


def dynamic_row(sample_id, frame_number, t, points, score, session, label, hand):
    return (
        [sample_id, frame_number, round(t, 4)]
        + flat(points)
        + [label, hand, round(score, 4), session, now_iso()]
    )


# ============================================================
# PROGRESS AND RESUME
# ============================================================

def find_next():
    """First (session, hand, class) that is not complete yet, or None."""
    for session in range(1, C.NUM_SESSIONS + 1):
        for hand in C.HANDS:
            for label in C.ALL_CLASSES:
                if count_samples(session, label, hand) < target_for(label):
                    return session, hand, label

    return None


def describe(item):
    session, hand, label = item
    return f"Session {session:02d} | {hand} hand | {label}"


def print_progress():
    print("\n==================== PROGRESS ====================")

    total_classes = len(C.ALL_CLASSES)

    for session in range(1, C.NUM_SESSIONS + 1):
        parts = []

        for hand in C.HANDS:
            done = sum(
                1 for label in C.ALL_CLASSES
                if count_samples(session, label, hand) >= target_for(label)
            )
            parts.append(f"{hand}: {done}/{total_classes} classes")

        split = C.SESSION_SPLIT.get(session, "-")
        print(f"Session {session:02d} [{split:<10}] " + " | ".join(parts))

    nxt = find_next()

    if nxt is None:
        print("\nEverything is complete.")
    else:
        print(f"\nNext to record: {describe(nxt)}")

    print("==================================================")


# ============================================================
# sessions.csv
# ============================================================

def session_is_logged(session):
    if not os.path.exists(C.SESSIONS_FILE):
        return False

    with open(C.SESSIONS_FILE, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("session") == str(session):
                return True

    return False


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def ask(prompt, default=""):
    text = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return text or default


def ensure_session_logged(session, width, height):
    """First time a session is used: note the recording conditions."""
    if session_is_logged(session):
        return

    print(f"\n--- First recording of session {session:02d}: describe the conditions ---")
    print("(press Enter to skip any question; you can edit data/sessions.csv later)")

    person = ask("Who is recording (name or nickname)", "me")
    lighting = ask("Lighting (e.g. daylight, lamp, dim)")
    background = ask("Background (e.g. plain wall, bookshelf)")
    distance = ask("Roughly how far is your hand from the camera, in cm")
    notes = ask("Any notes")

    row = [
        session,
        datetime.now().strftime("%Y-%m-%d"),
        person,
        C.SESSION_SPLIT.get(session, ""),
        lighting,
        background,
        distance,
        f"{width}x{height}",
        package_version("mediapipe"),
        getattr(cv2, "__version__", "unknown"),
        notes,
    ]

    os.makedirs(os.path.dirname(C.SESSIONS_FILE), exist_ok=True)

    new_file = (
        not os.path.exists(C.SESSIONS_FILE)
        or os.path.getsize(C.SESSIONS_FILE) == 0
    )

    with open(C.SESSIONS_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if new_file:
            writer.writerow(SESSIONS_HEADER)

        writer.writerow(row)

    print("Session details saved.\n")


# ============================================================
# WINDOW: BUTTONS, KEYS, TEXT
# ============================================================

BUTTON_ORDER = ["start", "pause", "undo", "end"]
BUTTON_LABELS = {
    "start": "START (S)",
    "pause": "PAUSE (P)",
    "undo": "UNDO (U)",
    "end": "END (E)",
}


def button_rects(height, width):
    gap = 8
    button_w = (width - gap * (len(BUTTON_ORDER) + 1)) // len(BUTTON_ORDER)
    top = height - 50
    bottom = height - 10

    rects = {}

    for i, action in enumerate(BUTTON_ORDER):
        left = gap + i * (button_w + gap)
        rects[action] = (left, top, left + button_w, bottom)

    return rects


class UI:
    def __init__(self):
        self.pending = None          # action from a mouse click
        self.collecting = False
        self.size = (C.CAMERA_HEIGHT, C.CAMERA_WIDTH)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for action, (x1, y1, x2, y2) in button_rects(*self.size).items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.pending = action
                return


def open_window(ui):
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, ui.on_mouse)


def close_window():
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def read_action(ui):
    """Mouse click or key press -> 'start' / 'pause' / 'undo' / 'end' / None."""
    key = cv2.waitKey(1) & 0xFF

    action = ui.pending
    ui.pending = None

    if key == ord("s"):
        action = "start"
    elif key == ord("p"):
        action = "pause"
    elif key == ord("u"):
        action = "undo"
    elif key in (ord("e"), ord("q")):
        action = "end"
    elif key == 32:
        action = "pause" if ui.collecting else "start"

    return action


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text[:70], (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_buttons(frame, ui):
    height, width = frame.shape[:2]

    for action, (x1, y1, x2, y2) in button_rects(height, width).items():
        active = (
            (action == "start" and ui.collecting)
            or (action == "pause" and not ui.collecting)
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 150, 0) if active else (80, 80, 80), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, BUTTON_LABELS[action], (x1 + 8, y1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_common(frame, ui, title, status_line, status_color, hint, message):
    height = frame.shape[0]

    put(frame, title, 28, (0, 255, 0), 0.65)
    put(frame, ("COLLECTING" if ui.collecting else "PAUSED  -  press START"),
        56, (0, 255, 255) if ui.collecting else (0, 165, 255), 0.65)
    put(frame, status_line, 84, status_color, 0.55)

    if message:
        put(frame, message, height - 95, (0, 255, 255), 0.6)

    put(frame, hint, height - 65, (255, 0, 255), 0.5)
    draw_buttons(frame, ui)

    cv2.imshow(WINDOW, frame)


def grab(cap, tracker, ui):
    """One camera frame + hand result (hand skeleton already drawn)."""
    frame = read_frame(cap)

    if frame is None:
        return None, None

    ui.size = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = tracker.process(rgb)

    if result is not None:
        tracker.draw(frame, result)

    return frame, result


def check_hand(result, required_hand, strict):
    """(is_good, reason). strict also checks confidence and frame edges."""
    if result is None:
        return False, "No hand detected"

    if result.label != required_hand:
        return False, (f"Show your {required_hand.upper()} hand "
                       f"(camera sees {result.label})")

    if not strict:
        return True, ""

    if result.score < C.MIN_HAND_SCORE:
        return False, "Hand not clear - improve light or position"

    low, high = C.EDGE_MARGIN, 1.0 - C.EDGE_MARGIN

    for x, y, _ in result.points:
        if not (low <= x <= high and low <= y <= high):
            return False, "Keep your WHOLE hand inside the picture"

    return True, ""


# ============================================================
# STATIC COLLECTION  (A-Y letters and REST)
# ============================================================

def run_static(session, hand, label, target, cap, tracker, ui, start_collecting=False):
    path = class_path(session, label)
    ensure_file(path, STATIC_HEADER)

    count = count_samples(session, label, hand)
    ui.collecting = start_collecting

    last_points = None
    last_time = 0.0
    saved_this_run = 0
    message, message_until = "", 0.0

    if label == "REST":
        hint = "REST: relaxed hand, pointing, fist, in-between poses"
    else:
        hint = f"Sign '{label}' clearly, then drift: near/far, tilt, rotate"

    while True:
        frame, result = grab(cap, tracker, ui)

        if frame is None:
            print("\nERROR: could not read the camera.")
            return "error"

        now = time.time()
        good, reason = check_hand(result, hand, strict=True)
        change = None

        if good and last_points is not None:
            change = landmark_motion(result.points, last_points)

        if ui.collecting and good:
            first = last_points is None
            enough = (
                change is not None
                and change >= C.DIVERSITY_THRESHOLD
                and now - last_time >= C.MIN_SAVE_INTERVAL
            )

            if first or enough:
                save_static(session, label, hand, result.score, result.points)
                count += 1
                saved_this_run += 1
                last_points = result.points
                last_time = now
                print(f"\r{describe((session, hand, label))}: {count}/{target}   ",
                      end="", flush=True)

                if count >= target:
                    print(f"\n{label} ({hand}) complete.")
                    return "done"

        if not good:
            status, color = reason, (0, 0, 255)
        elif ui.collecting:
            need = f"{change:.3f}/{C.DIVERSITY_THRESHOLD}" if change is not None else "-"
            status, color = f"Good. Move/rotate your hand a little (change {need})", (0, 255, 0)
        else:
            status, color = "Hand OK - press START", (0, 255, 0)

        if now > message_until:
            message = ""

        draw_common(frame, ui,
                    f"S{session:02d} | {hand.upper()} | {label}   {count}/{target}",
                    status, color, hint, message)

        action = read_action(ui)

        if action == "start":
            ui.collecting = True
        elif action == "pause":
            ui.collecting = False
        elif action == "undo":
            if saved_this_run > 0 and undo_last(path, dynamic=False):
                count -= 1
                saved_this_run -= 1
                last_points = None
                message, message_until = f"Undone - now {count}/{target}", now + 2.0
            else:
                message, message_until = "Nothing to undo in this run", now + 2.0
        elif action == "end":
            print(f"\nEnded. {label} ({hand}): {count}/{target} saved.")
            return "end"


# ============================================================
# DYNAMIC COLLECTION  (J, Z, NONE)
# ============================================================

def run_dynamic(session, hand, label, target, cap, tracker, ui, start_collecting=False):
    path = class_path(session, label)
    ensure_file(path, DYNAMIC_HEADER)

    count = count_samples(session, label, hand)
    next_id = next_sample_id(path)
    ui.collecting = start_collecting

    trigger = MotionTrigger()
    recent = deque(maxlen=C.PRE_ROLL_FRAMES)

    state = "SETTLE"                 # SETTLE -> ARMED -> RECORDING -> SETTLE
    settle_since = time.time()
    recorded = []                    # (time, points, score)
    pre_roll = []
    t0 = 0.0

    saved_this_run = 0
    message, message_until = "", 0.0

    def discard(text):
        nonlocal recorded, pre_roll, message, message_until
        recorded, pre_roll = [], []
        message, message_until = text, time.time() + 2.5

    while True:
        frame, result = grab(cap, tracker, ui)

        if frame is None:
            print("\nERROR: could not read the camera.")
            return "error"

        now = time.time()
        good, reason = check_hand(result, hand, strict=False)
        average, triggered = 0.0, False

        if good:
            average, triggered = trigger.update(result.points)
        else:
            trigger.reset()
            recent.clear()

        if not ui.collecting:
            if state == "RECORDING":
                discard("Paused - gesture discarded")

            state = "SETTLE"
            settle_since = now

        elif good:
            if state == "SETTLE":
                if (trigger.quiet_frames >= C.SETTLE_FRAMES
                        and now - settle_since >= C.MIN_REARM_SECONDS):
                    state = "ARMED"

            elif state == "ARMED":
                if triggered:
                    pre_roll = list(recent)
                    recorded = [(now, result.points, result.score)]
                    t0 = now
                    state = "RECORDING"

            elif state == "RECORDING":
                recorded.append((now, result.points, result.score))

                if len(recorded) >= C.DYNAMIC_FRAMES:
                    rows = []

                    for k, (t, points, score) in enumerate(pre_roll):
                        rows.append(dynamic_row(
                            next_id, k - len(pre_roll), t - t0,
                            points, score, session, label, hand))

                    for k, (t, points, score) in enumerate(recorded):
                        rows.append(dynamic_row(
                            next_id, k, t - t0,
                            points, score, session, label, hand))

                    append_rows(path, rows)

                    duration = recorded[-1][0] - t0
                    count += 1
                    next_id += 1
                    saved_this_run += 1
                    recorded, pre_roll = [], []
                    state = "SETTLE"
                    settle_since = now
                    message, message_until = (
                        f"Saved {count}/{target} ({duration:.2f}s)", now + 2.0)
                    print(f"{describe((session, hand, label))}: {count}/{target} "
                          f"({duration:.2f}s)")

                    if count >= target:
                        print(f"\n{label} ({hand}) complete.")
                        return "done"

        else:
            if state == "RECORDING":
                discard("Hand lost or changed - gesture discarded")
                state = "SETTLE"
                settle_since = now

        if good:
            recent.append((now, result.points, result.score))

        if not good:
            status, color = reason, (0, 0, 255)
        elif not ui.collecting:
            status, color = "Hand OK - press START", (0, 255, 0)
        elif state == "SETTLE":
            status, color = "Hold still for a moment...", (0, 255, 255)
        elif state == "ARMED":
            status, color = "READY - start your gesture now", (0, 255, 0)
        else:
            status, color = f"RECORDING {len(recorded)}/{C.DYNAMIC_FRAMES}", (0, 0, 255)

        if label == "J":
            hint = "J: draw the J in the air with your pinky"
        elif label == "Z":
            hint = "Z: draw the Z in the air with your index finger"
        else:
            hint = "NONE: " + NONE_HINTS[count % len(NONE_HINTS)]

        if now > message_until:
            message = ""

        draw_common(frame, ui,
                    f"S{session:02d} | {hand.upper()} | {label}   {count}/{target}",
                    status, color, hint, message)

        action = read_action(ui)

        if action == "start":
            ui.collecting = True
        elif action == "pause":
            ui.collecting = False
        elif action == "undo":
            if state == "RECORDING":
                discard("Gesture discarded")
                state = "SETTLE"
                settle_since = now
            elif saved_this_run > 0 and undo_last(path, dynamic=True):
                count -= 1
                saved_this_run -= 1
                message, message_until = f"Undone - now {count}/{target}", now + 2.0
            else:
                message, message_until = "Nothing to undo in this run", now + 2.0
        elif action == "end":
            print(f"\nEnded. {label} ({hand}): {count}/{target} saved.")
            return "end"


def run_class(session, hand, label, target, cap, tracker, ui, start_collecting=False):
    if is_dynamic(label):
        return run_dynamic(session, hand, label, target, cap, tracker, ui, start_collecting)

    return run_static(session, hand, label, target, cap, tracker, ui, start_collecting)


# ============================================================
# MENU MODES
# ============================================================

def camera_size(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def continue_mode(cap, tracker):
    ui = UI()
    open_window(ui)

    try:
        while True:
            nxt = find_next()

            if nxt is None:
                print("\nEverything is complete!")
                return

            session, hand, label = nxt
            ensure_session_logged(session, *camera_size(cap))

            print(f"\n>>> {describe(nxt)}   (press START in the window)")

            outcome = run_class(session, hand, label, target_for(label),
                                cap, tracker, ui, start_collecting=False)

            if outcome != "done":
                return
    finally:
        close_window()


def ask_choice(prompt, valid):
    while True:
        text = input(prompt).strip()

        if text.upper() in valid:
            return text.upper()

        print(f"Please enter one of: {', '.join(valid)}")


def manual_mode(cap, tracker):
    session = int(ask_choice(
        f"\nSession number (1-{C.NUM_SESSIONS}): ",
        [str(s) for s in range(1, C.NUM_SESSIONS + 1)]))

    if len(C.HANDS) == 1:
        hand = C.HANDS[0]
    else:
        letter = ask_choice("Hand - R for Right, L for Left: ", ["R", "L"])
        hand = "Right" if letter == "R" else "Left"

    label = ask_choice(
        f"Class ({', '.join(C.ALL_CLASSES)}): ", C.ALL_CLASSES)

    count = count_samples(session, label, hand)
    target = target_for(label)

    print(f"\nAlready saved: {count}/{target}")

    if count >= target:
        answer = input("This one is complete. Record MORE samples? (y/n): ").strip().lower()

        if answer != "y":
            return

        extra = int(ask("How many extra samples", "10"))
        target = count + extra

    ensure_session_logged(session, *camera_size(cap))

    ui = UI()
    open_window(ui)

    try:
        print(f"\n>>> {describe((session, hand, label))}   (press START in the window)")
        run_class(session, hand, label, target, cap, tracker, ui)
    finally:
        close_window()


def main():
    print("\n==========================================")
    print("   SIGN LANGUAGE DATA COLLECTION")
    print("==========================================")
    print(f"Saving to: {C.RAW_DIR}")

    cap = cv2.VideoCapture(C.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("\nERROR: could not open the camera.")
        return

    tracker = HandTracker()

    try:
        while True:
            print_progress()
            nxt = find_next()

            print("\n  1. Continue where I left off"
                  + (f"  ->  {describe(nxt)}" if nxt else ""))
            print("  2. Choose session, hand and letter myself")
            print("  3. Show progress again")
            print("  4. Quit")

            choice = input("\nChoose 1-4: ").strip()

            if choice == "1":
                continue_mode(cap, tracker)
            elif choice == "2":
                manual_mode(cap, tracker)
            elif choice == "3":
                continue
            elif choice == "4":
                break
            else:
                print("Please enter 1, 2, 3 or 4.")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    print("\nProgram closed. All saved data is kept.")


if __name__ == "__main__":
    main()
