"""
engine.py - owns the webcam. A background thread reads frames, finds the hand,
asks the Recognizer what was signed, and keeps the typed text.

The web server only reads from it (latest picture + state) and sends it
commands (space, backspace, clear, camera on/off). Only this thread touches the
camera, so turning it off really releases it (the camera light goes out).

Gestures that are commands, not letters:
  both open palms held  -> clear the text
  thumbs up held        -> the page speaks the text
"""
import threading
import time

import cv2
import numpy as np

from gestures import HoldGesture, is_open_palm, is_thumbs_up


class Engine:
    def __init__(self, recognizer, capture, tracker, read_frame, suggest_fn=None, clock=time.time,
                 pair_tracker=None, open_camera=None):
        self.recognizer = recognizer
        self.cap = capture
        self.open_camera = open_camera      # function that opens the camera again after "off"
        self.tracker = tracker
        self.pair_tracker = pair_tracker    # 2-hand tracker, used ONLY for the clear gesture
        self.read_frame = read_frame        # must mirror the frame like collect_data.py does
        self.suggest_fn = suggest_fn        # NLP hook: text -> list of words (added later)
        self.clock = clock

        self.lock = threading.Lock()
        self.frame_ready = threading.Condition()
        self.jpeg = None
        self.running = False
        self.thread = None

        self.want_camera = capture is not None
        self.text = ""
        self.suggestions = []
        self.error = ""
        self.fps = 0.0
        self.last_out = self._idle_out()
        self.last_typed = ""
        self.last_dynamic = ""
        self.last_dynamic_at = 0.0

        self.clear_gesture = HoldGesture()
        self.speak_gesture = HoldGesture()
        self.palms = 0
        self.thumb = False
        self.cleared_at = -10.0
        self.spoken_at = -10.0
        self.speak_seq = 0                  # the page speaks the text whenever this number changes
        self._refresh_suggestions()         # starting suggestions (empty text)

    @staticmethod
    def _idle_out():
        return dict(hand=False, state="SETTLE", prediction=None, hold=0.0)

    # ---------------- thread control ----------------
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self._release_camera()
        for t in (self.tracker, self.pair_tracker):
            try:
                if t is not None:
                    t.close()
            except Exception:
                pass

    # ---------------- camera on / off ----------------
    def _release_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.frame_ready:
                self.jpeg = buf.tobytes()
                self.frame_ready.notify_all()

    def _show_off_picture(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, "Camera off", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 2)
        self._publish(frame)

    def _reset_tracking(self):
        self.recognizer.reset()
        self.clear_gesture.reset()
        self.speak_gesture.reset()
        with self.lock:
            self.last_out = self._idle_out()
            self.palms = 0
            self.thumb = False
            self.fps = 0.0

    def apply_camera_state(self):
        """Called by the loop: makes the real camera match what the user asked for."""
        if not self.want_camera and self.cap is not None:
            self._release_camera()
            self._reset_tracking()
            self._show_off_picture()
        elif self.want_camera and self.cap is None:
            cap = self.open_camera() if self.open_camera else None
            if cap is None:
                self.want_camera = False
                with self.lock:
                    self.error = "Could not open the camera (is another program using it?)"
                self._show_off_picture()
            else:
                self.cap = cap
                self._reset_tracking()

    # ---------------- main loop ----------------
    def step(self):
        """One frame. Returns False if the camera failed."""
        frame = self.read_frame(self.cap)
        if frame is None:
            self.error = "Could not read the camera (is another program using it?)"
            return False

        now = self.clock()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normal signing uses the fast ONE-hand tracker (same as in training).
        hands = self.tracker.process_all(rgb)
        primary = hands[0] if hands else None

        # Only when that hand is an open palm do we pay for a 2-hand look.
        pair = []
        open_count = 0
        if primary is not None and is_open_palm(primary.points):
            open_count = 1
            if self.pair_tracker is not None:
                pair = self.pair_tracker.process_all(rgb)
                open_count = max(1, sum(1 for h in pair if is_open_palm(h.points)))

        both_open = len(pair) == 2 and open_count == 2
        thumb = primary is not None and not both_open and is_thumbs_up(primary.points)

        clear_now = self.clear_gesture.update(both_open, now)
        speak_now = self.speak_gesture.update(thumb, now)

        # letters are paused while a command gesture is shown
        out = self.recognizer.update(None if (both_open or thumb) else primary, now)

        for h in (pair if pair else hands):
            self.tracker.draw(frame, h)

        with self.lock:
            self.error = ""
            self.last_out = out
            self.palms = open_count
            self.thumb = thumb
            if clear_now:
                self.text = ""
                self.last_typed = ""
                self.cleared_at = now
                self._refresh_suggestions()
            if speak_now and self.text.strip():
                self.speak_seq += 1
                self.spoken_at = now
            if out.get("dynamic"):
                self.last_dynamic = out["dynamic"]
                self.last_dynamic_at = now
            if out.get("committed"):
                self._type(out["committed"])

        self._publish(frame)
        return True

    def _loop(self):
        last = self.clock()
        while self.running:
            self.apply_camera_state()

            if self.cap is None:                 # camera is off
                time.sleep(0.1)
                last = self.clock()
                continue

            if not self.step():
                time.sleep(0.5)
                continue

            now = self.clock()
            dt = max(now - last, 1e-6)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            last = now

    # ---------------- text editing / commands ----------------
    def _refresh_suggestions(self):
        if self.suggest_fn is None:
            self.suggestions = []
            return
        try:
            self.suggestions = list(self.suggest_fn(self.text))[:5]
        except Exception:
            self.suggestions = []

    def _type(self, letter):
        self.text += letter
        self.last_typed = letter
        self._refresh_suggestions()

    def action(self, name, value=""):
        if name == "camera_on":
            self.want_camera = True
            return True
        if name == "camera_off":
            self.want_camera = False
            return True

        with self.lock:
            if name == "space":
                if self.text and not self.text.endswith(" "):
                    self.text += " "
            elif name == "backspace":
                self.text = self.text[:-1]
            elif name == "clear":
                self.text = ""
                self.last_typed = ""
            elif name == "accept":              # replace the word being typed
                head = self.text.rsplit(" ", 1)[0] + " " if " " in self.text else ""
                self.text = head + value + " "
            else:
                return False
            self._refresh_suggestions()
        return True

    # ---------------- what the web page reads ----------------
    def snapshot(self):
        with self.lock:
            out = self.last_out
            return dict(
                text=self.text,
                suggestions=self.suggestions,
                camera_on=self.want_camera and self.cap is not None,
                hand=bool(out.get("hand")),
                state=out.get("state"),
                prediction=out.get("prediction"),
                hold=round(float(out.get("hold", 0.0)), 2),
                last_typed=self.last_typed,
                dynamic=self.last_dynamic if self.clock() - self.last_dynamic_at < 2.5 else "",
                fps=round(self.fps, 1),
                palms=self.palms,
                clear_hold=round(self.clear_gesture.progress, 2),
                cleared=self.clock() - self.cleared_at < 1.5,
                thumb=self.thumb,
                speak_hold=round(self.speak_gesture.progress, 2),
                speak_seq=self.speak_seq,
                speaking=self.clock() - self.spoken_at < 1.5,
                error=self.error,
            )

    def mjpeg(self):
        """Generator for the live picture."""
        while True:
            with self.frame_ready:
                self.frame_ready.wait(timeout=2)
                data = self.jpeg
            if data is None:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")