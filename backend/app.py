"""
Run from the project root:      python backend/app.py
Then open in the browser:       http://127.0.0.1:5000
"""
import os
import sys

from flask import Flask, Response, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)        # backend first: its files win if src/ has a file with the same name

FRONTEND = os.path.join(ROOT, "frontend")
MODELS = os.path.join(ROOT, "models")


def create_app(engine):
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND, "index.html")

    @app.route("/<path:name>")
    def files(name):
        return send_from_directory(FRONTEND, name)

    @app.route("/video")
    def video():
        return Response(engine.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/state")
    def state():
        return jsonify(engine.snapshot())

    @app.route("/action/<name>", methods=["POST"])
    def action(name):
        value = (request.get_json(silent=True) or {}).get("value", "")
        if not engine.action(name, str(value)):
            return jsonify(error="unknown action"), 400
        return jsonify(engine.snapshot())

    return app


def load_models():
    import joblib
    import sklearn

    static = joblib.load(os.path.join(MODELS, "static_model.joblib"))
    dynamic = joblib.load(os.path.join(MODELS, "dynamic_model.joblib"))

    trained = static.get("versions", {}).get("sklearn")
    if trained and trained != sklearn.__version__:
        print(f"WARNING: models were trained with scikit-learn {trained} but you have "
              f"{sklearn.__version__}. If predictions fail, install the same version:\n"
              f"    pip install scikit-learn=={trained}")
    return static, dynamic


def build_engine():
    import cv2
    import config as C
    from engine import Engine
    from hand_tracker import HandTracker, read_frame
    from Recognizer import Recognizer

    from nlp import load_suggester

    static, dynamic = load_models()
    suggester = load_suggester()

    def open_camera():
        cap = cv2.VideoCapture(C.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_HEIGHT)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    cap = open_camera()
    if cap is None:
        raise SystemExit("Could not open the camera. Close other programs that use it.")

    return Engine(Recognizer(static, dynamic), cap, HandTracker(max_hands=1), read_frame,
                  suggest_fn=suggester.suggest if suggester else None,
                  pair_tracker=HandTracker(max_hands=2), open_camera=open_camera)


if __name__ == "__main__":
    engine = build_engine()
    engine.start()
    app = create_app(engine)
    print("\nOpen  http://127.0.0.1:5000  in your browser  (Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
