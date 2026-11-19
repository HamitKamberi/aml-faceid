import csv
import io
import os
import pickle
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from insightface.app import FaceAnalysis

from faceeid import execution_providers, open_usb_camera

warnings.filterwarnings("ignore", category=FutureWarning, message=r"`estimate` is deprecated")

MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_s")
DATA_DIR = os.environ.get("DATA_DIR", ".")

CONFIG = {
    "DB_PATH": os.path.join(DATA_DIR, f"embeddings_{MODEL_NAME}.pkl"),
    "ATTENDANCE_DB": os.path.join(DATA_DIR, "attendance.db"),
    "TIMEZONE": os.environ.get("ATTENDANCE_TZ", "Europe/Zurich"),
    "LEAVE_AFTER": os.environ.get("LEAVE_AFTER", "15:30"),
    "MATCH_THRESHOLD": 0.70,
    "CONFIRM_HITS": 2,
    "CONFIRM_WINDOW": 3.0,
    "LEAVE_UPDATE_EVERY": 10,
    "MIN_STAY_MINUTES": float(os.environ.get("MIN_STAY_MINUTES", "1")),
    "MAX_YAW": 0.22,
    "MAX_ROLL_DEG": 15,
    "MIN_FACE_PX": 80,
    "INFER_FPS": 10,
    "LIVENESS": os.environ.get("LIVENESS", "1") != "0",
    "BLINK_RATIO": 0.65,
    "LIVENESS_WINDOW": 4.0,
    "LIVE_VALID_SECONDS": 6.0,
    "STREAM_FPS": 10,
    "BANNER_SECONDS": 4,
    "WELCOME_SECONDS": 3,
    "REPEAT_MESSAGE_EVERY": 15,
    "FLASK_PORT": 5002,
}

TZ = ZoneInfo(CONFIG["TIMEZONE"])
GREEN = (80, 200, 80)
BLUE = (220, 140, 40)
AMBER = (0, 170, 255)
RED = (60, 60, 220)


def now_local():
    return datetime.now(TZ)


class AttendanceBook:
    """One row per person per day: arrival time and departure time (after LEAVE_AFTER)."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.leave_after = datetime.strptime(CONFIG["LEAVE_AFTER"], "%H:%M").time()
        self.last_leave_write = {}
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS attendance (
                    day TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arrival TEXT NOT NULL,
                    arrival_conf REAL NOT NULL,
                    departure TEXT,
                    departure_conf REAL,
                    PRIMARY KEY (day, name))"""
            )
        self._import_old_csv(os.path.join(os.path.dirname(path) or ".", "attendance.csv"))

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _import_old_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return
        with open(csv_path, newline="") as f, self._connect() as db:
            for row in csv.DictReader(f):
                db.execute(
                    "INSERT OR IGNORE INTO attendance (day, name, arrival, arrival_conf) VALUES (?, ?, ?, ?)",
                    (row["Date"], row["Name"], row["Arrival"], float(row["Confidence"])),
                )
        os.rename(csv_path, csv_path + ".imported")
        print(f"[INFO] Imported {csv_path} into {self.path}.")

    def status(self, name):
        """Returns (arrival, departure) for today, or (None, None)."""
        day = now_local().strftime("%Y-%m-%d")
        with self.lock, self._connect() as db:
            row = db.execute(
                "SELECT arrival, departure FROM attendance WHERE day = ? AND name = ?", (day, name)
            ).fetchone()
        return (row["arrival"], row["departure"]) if row else (None, None)

    def record_sighting(self, name, score):
        """Registers arrival (first sighting of the day) or departure.
        Departure = the latest sighting after LEAVE_AFTER, at least MIN_STAY_MINUTES after arrival;
        every later sighting moves it forward.
        Returns ("arrival" | "departure" | "departure_update" | None, time string)."""
        now = now_local()
        day, clock = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        with self.lock, self._connect() as db:
            row = db.execute(
                "SELECT arrival, departure FROM attendance WHERE day = ? AND name = ?", (day, name)
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO attendance (day, name, arrival, arrival_conf) VALUES (?, ?, ?, ?)",
                    (day, name, clock, score),
                )
                return "arrival", clock

            arrived = datetime.strptime(f"{day} {row['arrival']}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
            stayed = now - arrived >= timedelta(minutes=CONFIG["MIN_STAY_MINUTES"])
            if now.time() < self.leave_after or not stayed:
                return None, row["arrival"]

            last = self.last_leave_write.get(name, 0)
            if row["departure"] and time.time() - last < CONFIG["LEAVE_UPDATE_EVERY"]:
                return None, row["departure"]
            db.execute(
                "UPDATE attendance SET departure = ?, departure_conf = ? WHERE day = ? AND name = ?",
                (clock, score, day, name),
            )
            self.last_leave_write[name] = time.time()
            return ("departure_update" if row["departure"] else "departure"), clock

    def _rows(self, where, args):
        with self.lock, self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM attendance WHERE {where} ORDER BY day DESC, arrival", args
            ).fetchall()
        out = []
        for r in rows:
            hours = None
            if r["departure"]:
                a = datetime.strptime(r["arrival"], "%H:%M:%S")
                d = datetime.strptime(r["departure"], "%H:%M:%S")
                hours = round((d - a).total_seconds() / 3600, 2)
            out.append(
                {
                    "day": r["day"],
                    "name": r["name"],
                    "arrival": r["arrival"],
                    "arrival_conf": round(r["arrival_conf"] * 100),
                    "departure": r["departure"],
                    "departure_conf": round(r["departure_conf"] * 100) if r["departure_conf"] else None,
                    "hours": hours,
                }
            )
        return out

    def by_day(self, day):
        return self._rows("day = ?", (day,))

    def by_person(self, name, days=31):
        since = (now_local() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self._rows("name = ? AND day >= ?", (name, since))

    def all_rows(self):
        return self._rows("1 = 1", ())

    def people(self):
        with self.lock, self._connect() as db:
            return [r[0] for r in db.execute("SELECT DISTINCT name FROM attendance ORDER BY name")]


class IdentityMatcher:
    """Compares a face embedding with every sub-centre of every known person."""

    def __init__(self, db_path):
        with open(db_path, "rb") as f:
            db = pickle.load(f)
        names, centres = [], []
        for name, data in db.items():
            for centre in data if isinstance(data, list) else [data]:
                centre = np.asarray(centre, dtype=np.float32).ravel()
                names.append(name)
                centres.append(centre / np.linalg.norm(centre))
        if not centres:
            raise RuntimeError(f"{db_path} contains no identities. Run try.py first.")
        self.names = names
        self.known = sorted(set(names))
        self.centres = np.stack(centres)
        print(f"[INFO] Loaded {len(self.known)} people, {len(names)} reference vectors.")

    def best_match(self, embedding):
        emb = embedding / np.linalg.norm(embedding)
        scores = self.centres @ emb
        i = int(np.argmax(scores))
        return self.names[i], float(scores[i])


class SharedState:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        self.detections = []
        self.det_lock = threading.Lock()
        self.banner = None
        self.event = None
        self.event_id = 0
        self.event_lock = threading.Lock()
        self.known_people = []
        self.status = "Starting camera..."
        self.running = True


STATE = SharedState()


def announce(kind, name, clock):
    """Message for the dashboard's welcome panel."""
    with STATE.event_lock:
        STATE.event_id += 1
        STATE.event = {"id": STATE.event_id, "kind": kind, "name": name, "time": clock, "ts": time.time()}
BOOK = AttendanceBook(CONFIG["ATTENDANCE_DB"])


def looking_at_camera(kps):
    """Uses the 5 face landmarks (eyes, nose, mouth corners) to check the face is turned to the camera."""
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1:
        return False
    yaw = (nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_dist
    roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    return abs(yaw) <= CONFIG["MAX_YAW"] and abs(roll) <= CONFIG["MAX_ROLL_DEG"]


LEFT_EYE = [33, 35, 36, 37, 39, 40, 41, 42]
RIGHT_EYE = [87, 89, 90, 91, 93, 94, 95, 96]


def eye_openness(landmarks):
    """Height / width of both eye outlines from the 106-point landmarks; drops sharply during a blink."""
    ratios = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        pts = landmarks[idx]
        w = np.ptp(pts[:, 0])
        if w > 0:
            ratios.append(np.ptp(pts[:, 1]) / w)
    return float(np.mean(ratios)) if ratios else 0.0


class BlinkLiveness:
    """A person counts as live after one blink: eyes clearly close, then open again.
    A printed photo or a still picture on a phone cannot do this."""

    def __init__(self):
        self.samples = {}
        self.live_until = {}
        self.warned = False

    def is_live(self, name, now):
        return now < self.live_until.get(name, 0)

    def update(self, name, now, face):
        if self.is_live(name, now):
            return True
        landmarks = getattr(face, "landmark_2d_106", None)
        if landmarks is None:
            if not self.warned:
                print("[WARN] No 106-point landmarks - liveness check disabled. Use buffalo_s or buffalo_l.")
                self.warned = True
            return True
        window = CONFIG["LIVENESS_WINDOW"]
        samples = [(t, v) for t, v in self.samples.get(name, []) if now - t <= window]
        samples.append((now, eye_openness(landmarks)))
        self.samples[name] = samples
        if len(samples) < 5:
            return False
        values = np.array([v for _, v in samples])
        open_level = np.percentile(values, 80)
        lowest = int(np.argmin(values))
        blinked = values[lowest] < CONFIG["BLINK_RATIO"] * open_level
        reopened = lowest < len(values) - 1 and values[-1] > 0.85 * open_level
        if blinked and reopened:
            self.live_until[name] = now + CONFIG["LIVE_VALID_SECONDS"]
            self.samples[name] = []
            print(f"[LIVE] Blink detected for {name}")
            return True
        return False


def capture_worker():
    try:
        cap = open_usb_camera()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        STATE.status = f"Camera error: {exc}"
        return
    if STATE.status.startswith("Starting camera"):
        STATE.status = "Loading face model..."
    while STATE.running:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        with STATE.frame_lock:
            STATE.frame = frame
    cap.release()


def inference_worker():
    candidates = [CONFIG["DB_PATH"], os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(CONFIG["DB_PATH"]))]
    db_path = next((p for p in candidates if os.path.exists(p)), None)
    if db_path is None:
        msg = f"Face database {CONFIG['DB_PATH']} not found - run try.py"
        print(f"[ERROR] {msg}")
        STATE.status = msg
        return
    try:
        matcher = IdentityMatcher(db_path)
        STATE.known_people = matcher.known
        providers, ctx_id = execution_providers()
        modules = ["detection", "recognition", "landmark_2d_106"]
        model = FaceAnalysis(name=MODEL_NAME, root=".", providers=providers, allowed_modules=modules)
        model.prepare(ctx_id=ctx_id, det_size=(640, 640))
    except Exception as exc:
        print(f"[ERROR] Model failed to load: {exc}")
        STATE.status = f"Model error: {exc}"
        return
    print(f"[INFO] Model {MODEL_NAME} ready on {'GPU' if ctx_id >= 0 else 'CPU'} using {db_path}.")
    STATE.status = ""

    hits = {}
    last_message = {}
    liveness = BlinkLiveness()
    interval = 1.0 / CONFIG["INFER_FPS"]
    last = 0.0

    while STATE.running:
        now = time.time()
        if now - last < interval:
            time.sleep(0.01)
            continue
        last = now

        with STATE.frame_lock:
            frame = None if STATE.frame is None else STATE.frame.copy()
        if frame is None:
            continue

        results = []
        for face in model.get(frame):
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            box = [x1, y1, x2, y2]
            if min(x2 - x1, y2 - y1) < CONFIG["MIN_FACE_PX"]:
                results.append({"bbox": box, "label": "Come closer", "color": AMBER})
                continue

            name, score = matcher.best_match(face.embedding)
            pct = int(round(score * 100))

            if score >= 0.5 and not looking_at_camera(face.kps):
                results.append({"bbox": box, "label": f"{name}? Look straight at the camera", "color": AMBER})
                continue

            if score < CONFIG["MATCH_THRESHOLD"]:
                if score >= 0.5:
                    label, color = f"{name}? {pct}% - look at the camera", AMBER
                else:
                    label, color = f"Unknown {pct}%", RED
                results.append({"bbox": box, "label": label, "color": color})
                continue

            if CONFIG["LIVENESS"] and not liveness.update(name, now, face):
                eyes = eye_openness(face.landmark_2d_106) if getattr(face, "landmark_2d_106", None) is not None else 0
                results.append({"bbox": box, "label": f"{name} {pct}% - please blink (eyes {eyes:.2f})", "color": AMBER})
                continue

            recent = [t for t in hits.get(name, []) if now - t <= CONFIG["CONFIRM_WINDOW"]]
            recent.append(now)
            hits[name] = recent

            if len(recent) >= CONFIG["CONFIRM_HITS"]:
                event, clock = BOOK.record_sighting(name, score)
                if event == "arrival":
                    print(f"[ARRIVAL] {name} at {clock} ({pct}%)")
                    STATE.banner = (f"Welcome {name} - arrived {clock}", GREEN, now)
                    announce("arrival", name, clock)
                    last_message[name] = now
                elif event in ("departure", "departure_update"):
                    print(f"[DEPARTURE] {name} at {clock} ({pct}%)")
                    if now - last_message.get(name, 0) > CONFIG["REPEAT_MESSAGE_EVERY"] or event == "departure":
                        STATE.banner = (f"Goodbye {name} - left {clock}", BLUE, now)
                        announce("departure", name, clock)
                        last_message[name] = now
                elif now - last_message.get(name, 0) > CONFIG["REPEAT_MESSAGE_EVERY"]:
                    arrival, departure = BOOK.status(name)
                    announce("already", name, arrival)
                    last_message[name] = now

            arrival, departure = BOOK.status(name)
            if departure:
                label, color = f"{name} {pct}% - left {departure}", BLUE
            elif arrival:
                label, color = f"{name} {pct}% - in since {arrival}", GREEN
            else:
                label, color = f"{name} {pct}% - hold still...", GREEN
            results.append({"bbox": box, "label": label, "color": color})

        with STATE.det_lock:
            STATE.detections = results


def draw(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), d["color"], 3)
        (tw, th), _ = cv2.getTextSize(d["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, y1 - th - 18), (x1 + tw + 12, y1), d["color"], -1)
        cv2.putText(frame, d["label"], (x1 + 6, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if STATE.status:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), RED, -1)
        cv2.putText(frame, STATE.status[:90], (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    banner = STATE.banner
    if banner and time.time() - banner[2] < CONFIG["BANNER_SECONDS"]:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 90), (w, h), banner[1], -1)
        cv2.putText(frame, banner[0], (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return frame


def generate_frames():
    interval = 1.0 / CONFIG["STREAM_FPS"]
    last = 0.0
    while STATE.running:
        now = time.time()
        if now - last < interval:
            time.sleep(0.005)
            continue
        last = now
        with STATE.frame_lock:
            frame = None if STATE.frame is None else STATE.frame.copy()
        if frame is None:
            continue
        with STATE.det_lock:
            detections = list(STATE.detections)
        ok, buf = cv2.imencode(".jpg", draw(frame, detections), [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Attendance</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0F1B2D;--panel:#16263D;--line:#2A3C55;--text:#F6F5F1;--muted:#8FA3B5;--accent:#4FD1C5;--leave:#7FB3FF}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif}
header{display:flex;align-items:center;gap:24px;padding:16px 24px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:22px;margin:0}
nav{display:flex;gap:8px}
nav button{background:none;border:1px solid var(--line);color:var(--muted);padding:8px 16px;border-radius:999px;cursor:pointer;font-size:15px}
nav button.on{background:var(--accent);border-color:var(--accent);color:var(--bg);font-weight:600}
.clock{margin-left:auto;color:var(--muted)}
.view{display:none;padding:24px} .view.on{display:block}
.live{display:grid;grid-template-columns:2fr 1fr;gap:24px}
img{width:100%;border-radius:12px;background:var(--panel)}
.panel{background:var(--panel);border-radius:12px;padding:20px}
h2{margin:0 0 4px;font-size:20px} .sub{color:var(--muted);margin:0 0 16px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:16px}
th,td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line)}
th{color:var(--accent);font-weight:600;font-size:14px}
.left{color:var(--leave)} .in{color:var(--accent)} .none{color:var(--muted)}
.bar{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.bar button,.bar input,.bar select,.bar a{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:15px;text-decoration:none}
.stats{display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap}
.stat{background:var(--panel);border-radius:12px;padding:14px 18px;min-width:150px}
.stat b{display:block;font-size:26px} .stat span{color:var(--muted);font-size:13px}
.kiosk{position:relative;background:var(--panel);border-radius:12px;padding:40px 32px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;min-height:420px;transition:background .3s}
.kiosk.ok{background:#1E7F57}
.kiosk.bye{background:#2A5F9E}
.k-idle,.k-msg{display:flex;flex-direction:column;align-items:center;gap:14px}
.k-msg[hidden],.k-idle[hidden]{display:none}
.k-ring{width:96px;height:96px;border-radius:50%;border:4px solid var(--line);border-top-color:var(--accent);animation:spin 2.4s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.k-check{width:110px;height:110px;border-radius:50%;background:rgba(255,255,255,.18);font-size:64px;line-height:110px;font-weight:700}
.k-title{font-size:36px;margin:0;line-height:1.2}
.k-text{font-size:20px;margin:0;color:#DDE7EE;max-width:340px}
.k-time{font-size:44px;font-weight:700;margin:4px 0 0;font-variant-numeric:tabular-nums}
.k-foot{color:var(--muted);margin:12px 0 0}
.k-rule{position:absolute;bottom:14px;left:0;right:0;margin:0;font-size:13px;color:rgba(255,255,255,.55)}
@media (prefers-reduced-motion:reduce){.k-ring{animation:none}}
@media (max-width:900px){.live{grid-template-columns:1fr}}
</style></head>
<body>
<header><h1>Attendance</h1>
<nav><button data-v="live" class="on">Live</button><button data-v="day">Day by day</button><button data-v="person">Person</button></nav>
<span class="clock" id="clock"></span></header>

<section class="view on" id="live"><div class="live">
<div><p id="status" style="display:none;background:#7A2E2E;padding:12px 16px;border-radius:10px;margin:0 0 12px"></p>
<img src="/video_feed" alt="Live camera"></div>
<div class="kiosk" id="kiosk">
<div class="k-idle" id="kIdle">
<div class="k-ring"></div>
<h2 class="k-title">Please look at the camera</h2>
<p class="k-text">Look straight into the camera and <b>blink once</b> to get registered.</p>
<p class="k-foot" id="kCount"></p>
</div>
<div class="k-msg" id="kMsg" hidden>
<div class="k-check">&#10003;</div>
<h2 class="k-title" id="kHello"></h2>
<p class="k-text" id="kThanks"></p>
<p class="k-time" id="kTime"></p>
</div>
<p class="k-rule">Match &ge; {{ threshold }}% &middot; departures after {{ leave_after }}</p>
</div>
</div></section>

<section class="view" id="day">
<div class="bar"><button id="prev">&larr;</button><input type="date" id="dayPick"><button id="next">&rarr;</button>
<a href="/export.csv">Export all (CSV)</a></div>
<div class="stats" id="dayStats"></div>
<div class="panel"><table><thead><tr><th>Name</th><th>Arrived</th><th>Left</th><th>Hours</th><th>Match in / out</th></tr></thead><tbody id="dayRows"></tbody></table></div>
</section>

<section class="view" id="person">
<div class="bar"><select id="who"></select><span class="sub" style="margin:0">Last 31 days</span></div>
<div class="stats" id="personStats"></div>
<div class="panel"><table><thead><tr><th>Day</th><th>Arrived</th><th>Left</th><th>Hours</th></tr></thead><tbody id="personRows"></tbody></table></div>
</section>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const left = r => r.departure ? `<span class="left">${r.departure}</span>` : `<span class="none">not yet</span>`;
const hrs = h => h == null ? '<span class="none">-</span>' : `${Math.floor(h)}h ${String(Math.round((h % 1) * 60)).padStart(2,'0')}m`;
const empty = n => `<tr><td colspan="${n}" class="none">No records</td></tr>`;
let today = '{{ today }}', view = 'live';

document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('on', v.id === b.dataset.v));
  view = b.dataset.v; refresh();
});

async function get(url){ const r = await fetch(url); return r.json(); }

async function loadToday(){
  const d = await get('/api/day/' + today);
  const n = d.rows.length;
  $('kCount').textContent = n ? `${n} ${n === 1 ? 'person' : 'people'} registered today` : 'No one registered yet today';
}

let lastEvent = null, hideTimer = null;
const SHOW_MS = {{ welcome_ms }};
function showEvent(e){
  const k = $('kiosk');
  if (e.kind === 'arrival') {
    $('kHello').textContent = `Welcome, ${e.name}!`;
    $('kThanks').textContent = 'Thank you, you are registered.';
    $('kTime').textContent = e.time;
    k.className = 'kiosk ok';
  } else if (e.kind === 'departure') {
    $('kHello').textContent = `Goodbye, ${e.name}!`;
    $('kThanks').textContent = 'Thank you, your departure is registered.';
    $('kTime').textContent = e.time;
    k.className = 'kiosk bye';
  } else {
    $('kHello').textContent = `Welcome, ${e.name}!`;
    $('kThanks').textContent = `You are already registered today (arrived ${e.time}).`;
    $('kTime').textContent = new Date().toTimeString().slice(0, 8);
    k.className = 'kiosk ok';
  }
  $('kIdle').hidden = true; $('kMsg').hidden = false;
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => { k.className = 'kiosk'; $('kMsg').hidden = true; $('kIdle').hidden = false; loadToday(); }, SHOW_MS);
}
async function pollEvent(){
  try{
    const d = await get('/api/event');
    if (!d.event) return;
    if (lastEvent === null) { lastEvent = d.event.id; if (d.age > SHOW_MS / 1000) return; }
    else if (d.event.id === lastEvent) return;
    lastEvent = d.event.id;
    showEvent(d.event);
  }catch(e){}
}
setInterval(pollEvent, 400);

async function loadDay(){
  const day = $('dayPick').value || today;
  const d = await get('/api/day/' + day);
  const done = d.rows.filter(r => r.hours != null);
  const avg = done.length ? done.reduce((a, r) => a + r.hours, 0) / done.length : null;
  const first = d.rows.length ? d.rows.map(r => r.arrival).sort()[0] : '-';
  $('dayStats').innerHTML =
    `<div class="stat"><b>${d.rows.length}</b><span>people present</span></div>` +
    `<div class="stat"><b>${first}</b><span>first arrival</span></div>` +
    `<div class="stat"><b>${done.length}</b><span>departures registered</span></div>` +
    `<div class="stat"><b>${hrs(avg)}</b><span>average hours</span></div>`;
  $('dayRows').innerHTML = d.rows.length ? d.rows.map(r =>
    `<tr><td>${esc(r.name)}</td><td class="in">${r.arrival}</td><td>${left(r)}</td><td>${hrs(r.hours)}</td>` +
    `<td>${r.arrival_conf}% / ${r.departure_conf == null ? '-' : r.departure_conf + '%'}</td></tr>`).join('') : empty(5);
}

async function loadPeople(){
  const d = await get('/api/people');
  const keep = $('who').value;
  $('who').innerHTML = d.people.map(p => `<option>${esc(p)}</option>`).join('');
  if (keep && d.people.includes(keep)) $('who').value = keep;
}

async function loadPerson(){
  if (!$('who').options.length) await loadPeople();
  const name = $('who').value; if (!name) { $('personRows').innerHTML = empty(4); return; }
  const d = await get('/api/person/' + encodeURIComponent(name));
  const done = d.rows.filter(r => r.hours != null);
  const total = done.reduce((a, r) => a + r.hours, 0);
  const avgIn = d.rows.length ? (() => { const s = d.rows.map(r => r.arrival.split(':').map(Number)).map(([h,m,s]) => h*3600+m*60+s);
    const a = s.reduce((x,y)=>x+y,0)/s.length; return `${String(Math.floor(a/3600)).padStart(2,'0')}:${String(Math.floor(a%3600/60)).padStart(2,'0')}`; })() : '-';
  $('personStats').innerHTML =
    `<div class="stat"><b>${d.rows.length}</b><span>days present</span></div>` +
    `<div class="stat"><b>${avgIn}</b><span>average arrival</span></div>` +
    `<div class="stat"><b>${hrs(total)}</b><span>total hours (with departure)</span></div>`;
  $('personRows').innerHTML = d.rows.length ? d.rows.map(r =>
    `<tr><td>${r.day}</td><td class="in">${r.arrival}</td><td>${left(r)}</td><td>${hrs(r.hours)}</td></tr>`).join('') : empty(4);
}

function shift(n){ const d = new Date(($('dayPick').value || today) + 'T12:00:00'); d.setDate(d.getDate() + n);
  $('dayPick').value = d.toISOString().slice(0,10); loadDay(); }
$('prev').onclick = () => shift(-1); $('next').onclick = () => shift(1);
$('dayPick').onchange = loadDay; $('who').onchange = loadPerson;

async function refresh(){
  try{
    const c = await get('/api/clock'); today = c.today; $('clock').textContent = c.now + ' ' + c.tz;
    const msg = c.status || (c.camera ? '' : 'No camera picture yet - check the camera device');
    $('status').style.display = msg ? 'block' : 'none'; $('status').textContent = msg;
    if (!$('dayPick').value) $('dayPick').value = today;
    if (view === 'live') await loadToday();
    if (view === 'day') await loadDay();
    if (view === 'person') { await loadPeople(); await loadPerson(); }
  }catch(e){}
}
refresh(); setInterval(refresh, 5000);
</script></body></html>"""

app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(
        PAGE_HTML,
        threshold=int(CONFIG["MATCH_THRESHOLD"] * 100),
        leave_after=CONFIG["LEAVE_AFTER"],
        today=now_local().strftime("%Y-%m-%d"),
        welcome_ms=int(CONFIG["WELCOME_SECONDS"] * 1000),
    )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/event")
def api_event():
    with STATE.event_lock:
        event = dict(STATE.event) if STATE.event else None
    return jsonify({"event": event, "age": time.time() - event["ts"] if event else None})


@app.route("/api/clock")
def api_clock():
    now = now_local()
    return jsonify({"today": now.strftime("%Y-%m-%d"), "now": now.strftime("%H:%M"), "tz": now.strftime("%Z"),
                    "status": STATE.status, "camera": STATE.frame is not None})


@app.route("/api/day/<day>")
def api_day(day):
    return jsonify({"day": day, "rows": BOOK.by_day(day)})


@app.route("/api/person/<name>")
def api_person(name):
    days = request.args.get("days", default=31, type=int)
    return jsonify({"name": name, "rows": BOOK.by_person(name, days)})


@app.route("/api/people")
def api_people():
    return jsonify({"people": sorted(set(BOOK.people()) | set(STATE.known_people))})


@app.route("/export.csv")
def export_csv():
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Date", "Name", "Arrival", "Departure", "Hours", "ArrivalMatch%", "DepartureMatch%"])
    for r in BOOK.all_rows():
        writer.writerow([r["day"], r["name"], r["arrival"], r["departure"] or "", r["hours"] or "",
                         r["arrival_conf"], r["departure_conf"] or ""])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=attendance.csv"})


if __name__ == "__main__":
    threading.Thread(target=capture_worker, daemon=True).start()
    time.sleep(1)
    threading.Thread(target=inference_worker, daemon=True).start()
    print(f"[INFO] Dashboard: http://localhost:{CONFIG['FLASK_PORT']}  (times in {CONFIG['TIMEZONE']})")
    try:
        app.run(host="0.0.0.0", port=CONFIG["FLASK_PORT"], debug=False, threaded=True)
    finally:
        STATE.running = False
