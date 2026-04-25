"""
app.py — FacePlay main server.
Handles: Flask UI, QR registration, Spotify, face detection loop, hand gesture volume.

Usage:
    py -3.12 app.py

Then open http://localhost:5000 in your browser.
"""

import cv2
import numpy as np
import pickle
import os
import time
import threading
import base64
import qrcode
import io
import socket
import subprocess

import pygame
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ENCODINGS_FILE = "known_faces.pkl"
COOLDOWN = 240
SIMILARITY_THRESHOLD = 0.65
PROCESS_EVERY_N = 1
DISPLAY_DURATION = 240
SONG_BUFFER = 5
UPLOAD_FOLDER = "static/photos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("songs", exist_ok=True)

# Gesture config
GESTURE_COOLDOWN = 0.5
VOLUME_STEP = 10
MOVEMENT_THRESHOLD = 0.04
GESTURE_HISTORY = 6

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Spotify ───────────────────────────────────────────────────────────────────
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
    scope="user-read-playback-state user-modify-playback-state user-read-currently-playing"
))

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "current_person": None,
    "current_song": None,
    "last_seen": [],
    "last_played": {},
    "display_until": 0,
    "volume_display": "",
    "volume_display_until": 0,
}
state_lock = threading.Lock()

# ── Face model ────────────────────────────────────────────────────────────────
print("Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("Model ready.")

# ── MediaPipe hands ───────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )
    GESTURE_ENABLED = True
    print("Gesture control enabled.")
except Exception as e:
    print(f"Gesture disabled: {e}")
    GESTURE_ENABLED = False
    hands_detector = None

gesture_state = {
    "volume": 50,
    "last_gesture": 0,
    "wrist_history": [],
}

# ── Audio ─────────────────────────────────────────────────────────────────────
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()


def load_known_faces():
    if not os.path.exists(ENCODINGS_FILE):
        return []
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_known_faces(faces):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(faces, f)


def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)


def match_face(embedding, known_faces):
    best_name, best_data, best_dist = None, None, float("inf")
    for person in known_faces:
        dist = cosine_distance(embedding, person["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_name = person["name"]
            best_data = person
    if best_dist < SIMILARITY_THRESHOLD:
        return best_name, best_data, best_dist
    return None, None, None


SPOTIFY_EXE = os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe")


def launch_spotify():
    try:
        if os.path.exists(SPOTIFY_EXE):
            subprocess.Popen([SPOTIFY_EXE])
        else:
            os.startfile("spotify:")
    except Exception as e:
        print(f"Could not launch Spotify: {e}")


def play_spotify(track_uri, device_id=None):
    try:
        devices = sp.devices()
        available = devices.get("devices", [])
        if not available:
            print("Launching Spotify...")
            launch_spotify()
            time.sleep(4)
            devices = sp.devices()
            available = devices.get("devices", [])
        if not available:
            print("Spotify still not available.")
            return False
        active = [d for d in available if d.get("is_active")]
        target = device_id or (active[0]["id"] if active else available[0]["id"])
        sp.start_playback(device_id=target, uris=[track_uri])
        print(f"♪ Spotify playing: {track_uri}")
        return True
    except Exception as e:
        print(f"Spotify error: {e}")
        return False


def play_local(path, name):
    if not os.path.exists(path):
        print(f"WARNING: Song not found: {path}")
        return
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        print(f"♪ Local playing for {name}: {path}")
    except Exception as e:
        print(f"Audio error: {e}")


def trigger_song(person_data, name):
    song = person_data.get("song", "")
    if song.startswith("spotify:track:"):
        success = play_spotify(song)
        if not success:
            local = person_data.get("local_song", "")
            if local:
                play_local(local, name)
    else:
        play_local(song, name)


def set_spotify_volume(volume_pct):
    try:
        volume_pct = max(0, min(100, volume_pct))
        devices = sp.devices()
        available = devices.get("devices", [])
        if not available:
            return
        active = [d for d in available if d.get("is_active")]
        target = active[0]["id"] if active else available[0]["id"]
        sp.volume(volume_pct, device_id=target)
        gesture_state["volume"] = volume_pct
        print(f"🔊 Volume: {volume_pct}%")
    except Exception as e:
        print(f"Volume error: {e}")


# ── Detection loop ────────────────────────────────────────────────────────────
def detection_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    print("Detection loop started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        if frame_count % PROCESS_EVERY_N != 0:
            continue

        known_faces = load_known_faces()
        if not known_faces:
            time.sleep(0.5)
            continue

        faces = face_app.get(frame)

        for face in faces:
            embedding = face.embedding
            name, person_data, dist = match_face(embedding, known_faces)
            now = time.time()

            with state_lock:
                current_person = state["current_person"]
                last_triggered_any = max(state["last_played"].values()) if state["last_played"] else 0
                is_new_person = name != current_person

                if not is_new_person and now - last_triggered_any < SONG_BUFFER:
                    continue

                last = state["last_played"].get(name or "__unknown__", 0)
                if now - last > COOLDOWN:
                    if name and person_data:
                        trigger_song(person_data, name)
                        state["last_played"][name] = now
                        state["current_person"] = name
                        state["current_song"] = person_data.get("song_name", "Unknown")
                        state["display_until"] = now + DISPLAY_DURATION
                        entry = {
                            "name": name,
                            "time": time.strftime("%H:%M:%S"),
                            "photo": person_data.get("photo", "")
                        }
                        state["last_seen"].insert(0, entry)
                        state["last_seen"] = state["last_seen"][:10]
                    else:
                        unknown_song = "songs/unknown.mp3"
                        if os.path.exists(unknown_song):
                            play_local(unknown_song, "???")
                        state["last_played"]["__unknown__"] = now
                        state["current_person"] = "???"
                        state["current_song"] = "Who are you?"
                        state["display_until"] = now + DISPLAY_DURATION

        time.sleep(0.03)

    cap.release()


# ── Gesture loop ──────────────────────────────────────────────────────────────
def gesture_loop():
    if not GESTURE_ENABLED:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gesture: webcam not found, skipping.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Gesture loop started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            wrist_y = hand.landmark[0].y

            history = gesture_state["wrist_history"]
            history.append(wrist_y)
            if len(history) > GESTURE_HISTORY:
                history.pop(0)

            if len(history) >= GESTURE_HISTORY:
                mid = len(history) // 2
                early_avg = sum(history[:mid]) / mid
                recent_avg = sum(history[mid:]) / (len(history) - mid)
                delta = early_avg - recent_avg

                now = time.time()
                if abs(delta) > MOVEMENT_THRESHOLD and now - gesture_state["last_gesture"] > GESTURE_COOLDOWN:
                    if delta > 0:
                        new_vol = gesture_state["volume"] + VOLUME_STEP
                        label = f"🔊 {min(new_vol, 100)}%"
                        print(f"👆 Hand UP → Volume {min(new_vol, 100)}%")
                    else:
                        new_vol = gesture_state["volume"] - VOLUME_STEP
                        label = f"🔉 {max(new_vol, 0)}%"
                        print(f"👇 Hand DOWN → Volume {max(new_vol, 0)}%")

                    set_spotify_volume(new_vol)
                    gesture_state["last_gesture"] = now
                    gesture_state["wrist_history"] = []

                    with state_lock:
                        state["volume_display"] = label
                        state["volume_display_until"] = now + 2
        else:
            gesture_state["wrist_history"] = []

        time.sleep(0.03)

    cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    known = load_known_faces()
    return render_template("index.html", people=known)


@app.route("/display")
def display():
    return render_template("display.html")


@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@app.route("/api/register/qr")
def register_qr():
    local_ip = socket.gethostbyname(socket.gethostname())
    url = f"http://{local_ip}:5000/register/mobile"
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return jsonify({"qr": b64, "url": url})


@app.route("/register/mobile")
def register_mobile():
    return render_template("register_mobile.html")


@app.route("/api/register", methods=["POST"])
def api_register():
    name = request.form.get("name", "").strip()
    song_uri = request.form.get("song_uri", "").strip()
    song_name = request.form.get("song_name", "Unknown").strip()
    photo_file = request.files.get("photo")

    if not name or not photo_file:
        return jsonify({"error": "Name and photo required"}), 400

    photo_path = f"{UPLOAD_FOLDER}/{name.lower().replace(' ', '_')}.jpg"
    photo_file.save(photo_path)

    img = cv2.imread(photo_path)
    if img is None:
        return jsonify({"error": "Could not read photo"}), 400

    faces = face_app.get(img)
    if not faces:
        return jsonify({"error": "No face detected. Try better lighting."}), 400
    if len(faces) > 1:
        return jsonify({"error": "Multiple faces detected. Use a solo photo."}), 400

    embedding = faces[0].embedding

    if not song_uri:
        song_uri = "spotify:track:4cOdK2wGLETKBW3PvgPWqT"
        song_name = "Never Gonna Give You Up - Rick Astley"

    known = load_known_faces()
    known = [p for p in known if p["name"].lower() != name.lower()]
    known.append({
        "name": name,
        "embedding": embedding,
        "song": song_uri,
        "song_name": song_name,
        "photo": f"/static/photos/{name.lower().replace(' ', '_')}.jpg"
    })
    save_known_faces(known)
    return jsonify({"success": True, "name": name, "song": song_name})


@app.route("/api/search_song")
def search_song():
    query = request.args.get("q", "")
    if not query:
        return jsonify([])
    try:
        results = sp.search(q=query, type="track", limit=5)
        tracks = []
        for t in results["tracks"]["items"]:
            tracks.append({
                "uri": t["uri"],
                "name": t["name"],
                "artist": t["artists"][0]["name"],
                "album_art": t["album"]["images"][0]["url"] if t["album"]["images"] else ""
            })
        return jsonify(tracks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify({
            "current_person": state["current_person"],
            "current_song": state["current_song"],
            "display_until": state["display_until"],
            "last_seen": state["last_seen"],
            "now": time.time(),
            "volume_display": state["volume_display"],
            "volume_display_until": state["volume_display_until"],
        })


@app.route("/api/people")
def api_people():
    known = load_known_faces()
    return jsonify([{
        "name": p["name"],
        "song_name": p.get("song_name", ""),
        "photo": p.get("photo", "")
    } for p in known])


@app.route("/api/delete/<name>", methods=["DELETE"])
def delete_person(name):
    known = load_known_faces()
    known = [p for p in known if p["name"].lower() != name.lower()]
    save_known_faces(known)
    return jsonify({"success": True})


@app.route("/api/update_song", methods=["POST"])
def update_song():
    data = request.get_json()
    name = data.get("name", "").strip()
    uri = data.get("song_uri", "").strip()
    song_name = data.get("song_name", "").strip()
    if not name or not uri:
        return jsonify({"error": "Name and song required"}), 400
    known = load_known_faces()
    updated = False
    for p in known:
        if p["name"].lower() == name.lower():
            p["song"] = uri
            p["song_name"] = song_name
            updated = True
            break
    if not updated:
        return jsonify({"error": "Person not found"}), 404
    save_known_faces(known)
    return jsonify({"success": True})


@app.route("/api/devices")
def api_devices():
    try:
        devices = sp.devices()
        return jsonify(devices.get("devices", []))
    except Exception as e:
        return jsonify([])


@app.route("/api/now_playing")
def api_now_playing():
    try:
        current = sp.current_playback()
        if current and current.get("item"):
            item = current["item"]
            images = item.get("album", {}).get("images", [])
            art = images[0]["url"] if images else None
            return jsonify({
                "album_art": art,
                "track": item["name"],
                "artist": item["artists"][0]["name"]
            })
    except Exception as e:
        print(f"now_playing error: {e}")
    return jsonify({"album_art": None})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    if GESTURE_ENABLED:
        g = threading.Thread(target=gesture_loop, daemon=True)
        g.start()
    print("\nFacePlay running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)