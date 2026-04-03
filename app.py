from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import threading
import json
import time
import os

# ── Tensorflow / Keras ────────────────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False

app = Flask(__name__)

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_face_mesh   = mp.solutions.face_mesh
mp_pose        = mp.solutions.pose
mp_hands       = mp.solutions.hands

# ── Global state (shared across threads) ─────────────────────────────────────
state = {
    "sentence":       [],
    "predictions":    [],
    "current_action": "",
    "confidence":     0.0,
    "fps":            0,
    "is_running":     False,
    "frame_count":    0,
}
state_lock = threading.Lock()

actions       = np.array(['hello', 'thanks', 'iloveyou'])
THRESHOLD     = 0.7
MODEL_PATH    = "action.h5"
sequence_buf  = []
sentence_buf  = []

# Load model if available
model = None
if MODEL_LOADED and os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")

# ── MediaPipe helpers ─────────────────────────────────────────────────────────
def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

def extract_keypoints(results):
    pose = (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    lh   = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3))
    rh   = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3))
    face = (np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(468 * 3))
    return np.concatenate([pose, lh, rh, face])

# ── Video stream generator ────────────────────────────────────────────────────
def gen_frames():
    global sequence_buf, sentence_buf

    cap = cv2.VideoCapture(0)
    fps_time  = time.time()
    fps_count = 0
    res        = np.zeros(len(actions))

    with state_lock:
        state["is_running"] = True

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Keypoint extraction & prediction
            keypoints = extract_keypoints(results)
            sequence_buf.insert(0, keypoints)
            sequence_buf = sequence_buf[:30]

            if len(sequence_buf) == 30 and model is not None:
                res = model.predict(np.expand_dims(sequence_buf, axis=0), verbose=0)[0]
                best_idx    = int(np.argmax(res))
                best_action = actions[best_idx]
                best_conf   = float(res[best_idx])

                if best_conf > THRESHOLD:
                    if len(sentence_buf) == 0 or best_action != sentence_buf[-1]:
                        sentence_buf.append(best_action)
                if len(sentence_buf) > 5:
                    sentence_buf = sentence_buf[-5:]

                with state_lock:
                    state["predictions"]    = res.tolist()
                    state["current_action"] = best_action if best_conf > THRESHOLD else ""
                    state["confidence"]     = best_conf
                    state["sentence"]       = sentence_buf.copy()

            # FPS overlay
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                with state_lock:
                    state["fps"]         = fps_count
                    state["frame_count"] += fps_count
                fps_count = 0
                fps_time  = time.time()

            # Overlay text bar
            cv2.rectangle(image, (0, 0), (640, 38), (15, 15, 20), -1)
            cv2.putText(image, ' '.join(sentence_buf), (6, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 170), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    with state_lock:
        state["is_running"] = False

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           actions=actions.tolist(),
                           model_loaded=(model is not None))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(state)

@app.route('/reset', methods=['POST'])
def reset():
    global sequence_buf, sentence_buf
    sequence_buf = []
    sentence_buf = []
    with state_lock:
        state["sentence"]       = []
        state["predictions"]    = []
        state["current_action"] = ""
        state["confidence"]     = 0.0
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
