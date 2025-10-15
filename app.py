import os
import io
import csv
import json
import time
import smtplib
from email.message import EmailMessage
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import av
import pandas as pd
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from dotenv import load_dotenv
import pyrebase

# App / folders / env

st.set_page_config(page_title="Weapon Detection", page_icon="🛡️", layout="wide")
load_dotenv(override=True)

DATA_DIR = Path("data")
SNAP_DIR = Path("snapshots")
DATA_DIR.mkdir(exist_ok=True)
SNAP_DIR.mkdir(exist_ok=True)
ALERT_LOG_CSV = DATA_DIR / "alerts.csv"

DETECTION_THRESHOLD = 0.70
EMAIL_COOLDOWN_SECONDS = 60
WEAPON_LABELS_DEFAULT = {"gun", "knife", "pistol", "rifle", "grenade", "explosive"}


def get_env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v

SMTP_USER = get_env("SMTP_USER")
SMTP_PASSWORD = get_env("SMTP_PASSWORD")
SMTP_HOST = get_env("SMTP_HOST", "smtp.gmail.com")
try:
    SMTP_PORT = int(get_env("SMTP_PORT", "587"))
except ValueError:
    SMTP_PORT = 587

# Firebase (email/password)

def init_firebase_auth():
    cfg_path = "firebase_config.json"
    if not os.path.exists(cfg_path):
        st.error("Missing firebase_config.json next to app.py")
        st.stop()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pb_config = {
        "apiKey": cfg.get("apiKey"),
        "authDomain": cfg.get("authDomain"),
        "databaseURL": cfg.get("databaseURL", ""),
        "storageBucket": cfg.get("storageBucket"),
    }
    try:
        return pyrebase.initialize_app(pb_config).auth()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.stop()

auth = init_firebase_auth()

@dataclass
class SessionUser:
    email: str
    id_token: str

# Model

@st.cache_resource
def load_model() -> YOLO:
    model_path = "models/weapon.pt"
    if not os.path.exists(model_path):
        st.error("Model file not found at models/weapon.pt")
        st.stop()
    return YOLO(model_path)

# Email 

def send_email(subject: str, body: str, attachments: Tuple[Tuple[str, bytes], ...], to_email: str) -> bool:
    """
    Sends FROM SMTP_USER to to_email.
    TLS if SMTP_PORT=587, SSL if SMTP_PORT=465.
    attachments expected as tuple of (filename, bytes)
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        st.error("Missing SMTP_USER / SMTP_PASSWORD in .env (use Gmail App Password: 16 chars, no spaces).")
        return False
    if not to_email:
        st.error("Recipient email missing.")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg.set_content(body)

    for filename, blob in attachments:
        msg.add_attachment(blob, maintype="image", subtype="png", filename=filename)

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as s:
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        elif SMTP_PORT == 587:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
                s.ehlo()
                s.starttls()
                s.ehlo()
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        else:
            st.error("Unsupported SMTP_PORT. Use 587 (TLS) or 465 (SSL).")
            return False

        st.success(f"📧 Alert email sent to {to_email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        st.error("SMTPAuthenticationError: Google rejected the login. "
                 "Use a Gmail **App Password** (16 characters, no spaces), and make sure SMTP_USER is the same Gmail that created it.")
        st.error(f"Details: {e}")
        return False
    except Exception as e:
        st.error(f"Email send failed: {type(e).__name__}: {e}")
        return False

# Snapshots & logging

def image_to_png_bytes(bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def save_png(bgr: np.ndarray, path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
    return image_to_png_bytes(bgr)

def append_log(ts: datetime, user_email: str, label: str, conf: float,
               snapshot_path: Path, crop_path: Optional[Path], camera: str) -> None:
    header = ["timestamp", "user_email", "label", "confidence",
              "snapshot_path", "crop_path", "camera"]
    new_file = not ALERT_LOG_CSV.exists()
    with open(ALERT_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow([
            ts.isoformat(),
            user_email,
            label,
            f"{conf:.4f}",
            str(snapshot_path),
            str(crop_path) if crop_path else "",
            camera
        ])


# Auth Views

def signup_view():
    st.title("🔐 Create account")
    with st.form("signup_form"):
        email = st.text_input("Email")
        pw1 = st.text_input("Password", type="password")
        pw2 = st.text_input("Confirm password", type="password")
        ok = st.form_submit_button("Sign up")

    if ok:
        if not email or not pw1 or not pw2:
            st.warning("Fill all fields.")
            return
        if pw1 != pw2:
            st.error("Passwords do not match.")
            return
        try:
            auth.create_user_with_email_and_password(email, pw1)
            st.success("Account created. Please log in.")
            st.session_state.auth_mode = "login"
            st.rerun()
        except Exception as e:
            st.error(f"Sign up failed: {e}")

def login_view():
    st.title("🔐 Sign in")
    with st.form("login_form"):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")

    if ok:
        try:
            u = auth.sign_in_with_email_and_password(email, pw)
            st.session_state.user = SessionUser(email=email, id_token=u["idToken"])
            st.success("Signed in.")
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

    st.caption("New here?")
    if st.button("Create an account"):
        st.session_state.auth_mode = "signup"
        st.rerun()

def logout_button():
    if st.button("Log out"):
        st.session_state.pop("user", None)
        st.success("Logged out.")
        st.rerun()

# Detection 

class WeaponDetector(VideoTransformerBase):
    def __init__(self, model: YOLO, target_labels: set, conf_thres: float,
                 email_cooldown: int, current_user_email: str, camera_desc: str,
                 persist_frames: int = 5, min_area_ratio: float = 0.02):
        self.model = model
        self.labels = {l.lower() for l in target_labels}
        self.conf_thres = conf_thres
        self.email_cooldown = email_cooldown
        self.last_alert_ts = 0.0
        self.user_email = current_user_email
        self.camera_desc = camera_desc

        self.persist_frames = max(1, int(persist_frames))
        self.min_area_ratio = float(min_area_ratio)
        self.seen_counts = defaultdict(int)

    def _png_bytes(self, bgr: np.ndarray) -> bytes:
        return image_to_png_bytes(bgr)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        min_area = self.min_area_ratio * (h * w)

        results = self.model.predict(img, conf=self.conf_thres, verbose=False)
        annotated = img.copy()

        present_this_frame = set()
        best = {"label": None, "conf": 0.0, "crop": None, "box": None}

        if results and len(results) > 0:
            r = results[0]
            if getattr(r, "boxes", None) is not None and getattr(r.boxes, "xyxy", None) is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_arr = r.boxes.cls.cpu().numpy()
                conf_arr = r.boxes.conf.cpu().numpy()
                for (coords, cls_idx, conf_val) in zip(xyxy, cls_arr, conf_arr):
                    x1, y1, x2, y2 = map(int, coords)
                    area = max(0, (x2 - x1) * (y2 - y1))
                    label = r.names[int(cls_idx)].lower()
                    score = float(conf_val)

                    color = (0, 255, 0)
                    if label in self.labels and score >= self.conf_thres and area >= min_area:
                        color = (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{label} {score:.2f}", (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if label in self.labels and score >= self.conf_thres and area >= min_area:
                        present_this_frame.add(label)
                        if score > best["conf"]:
                            best["label"] = label
                            best["conf"] = score
                            best["box"] = (x1, y1, x2, y2)
                            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
                            best["crop"] = crop if crop.size else None

        existing_labels = set(self.seen_counts.keys())
        for lbl in (existing_labels | present_this_frame):
            if lbl in present_this_frame:
                self.seen_counts[lbl] = min(self.persist_frames, self.seen_counts[lbl] + 1)
            else:
                self.seen_counts[lbl] = max(0, self.seen_counts[lbl] - 1)
                if self.seen_counts[lbl] == 0:
                    self.seen_counts.pop(lbl, None)

        confirmed_label = None
        for lbl, cnt in self.seen_counts.items():
            if cnt >= self.persist_frames:
                confirmed_label = lbl
                break

        now = time.time()
        if confirmed_label and best["label"] == confirmed_label and (now - self.last_alert_ts >= self.email_cooldown):
            self.last_alert_ts = now
            ts = datetime.now()
            stamp = ts.strftime("%Y%m%d_%H%M%S")

            snap_path = SNAP_DIR / f"{stamp}_frame.png"
            crop_path = SNAP_DIR / f"{stamp}_crop_{confirmed_label}.png"

            full_png = save_png(annotated, snap_path)
            crop_png = None
            if best["crop"] is not None and best["crop"].size:
                crop_png = save_png(best["crop"], crop_path)
            else:
                crop_path = None

            append_log(
                ts=ts,
                user_email=self.user_email,
                label=confirmed_label,
                conf=best["conf"],
                snapshot_path=snap_path,
                crop_path=crop_path,
                camera=self.camera_desc,
            )

            subject = f"[ALERT {stamp}] {confirmed_label} detected ({best['conf']:.2f})"
            body = (
                f"Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"User: {self.user_email}\n"
                f"Camera: {self.camera_desc}\n"
                f"Detected: {confirmed_label} ({best['conf']:.2f})\n"
                f"Snapshot: {snap_path}\n"
                f"Crop: {crop_path if crop_path else 'N/A'}\n"
            )

            attachments = [("frame.png", full_png)]
            if crop_png is not None:
                attachments.append(("crop.png", crop_png))

            try:
                send_email(subject, body, tuple(attachments), to_email=self.user_email)
            except Exception:
                pass

            self.seen_counts[confirmed_label] = 0

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Views

def detection_view():
    st.title("🎯 Real-time Weapon Detection")
    user: SessionUser = st.session_state["user"]
    st.caption(f"Signed in as: {user.email}")
    logout_button()

    st.sidebar.header("Camera")
    cam_index = st.sidebar.number_input("Webcam index", min_value=0, max_value=9, value=0)

    conf = st.sidebar.slider("Confidence threshold", 0.30, 0.95, DETECTION_THRESHOLD, 0.01)
    persist_frames = st.sidebar.slider("Frames to confirm (anti-noise)", 1, 15, 5)
    min_area_pct = st.sidebar.slider("Min box area (% of frame)", 0.1, 50.0, 2.0)
    cooldown = st.sidebar.slider("Email cooldown (sec)", 10, 600, EMAIL_COOLDOWN_SECONDS, 10)

    if st.sidebar.button("Send test email"):
        send_email(
            subject="Test alert from Weapon Detection",
            body="This is a test email to verify your SMTP settings.",
            attachments=tuple(),
            to_email=user.email
        )

    model = load_model()

    st.write("### Live Stream")
    def factory():
        return WeaponDetector(
            model=model,
            target_labels=WEAPON_LABELS_DEFAULT,
            conf_thres=conf,
            email_cooldown=cooldown,
            current_user_email=user.email,
            camera_desc=f"webcam:{cam_index}",
            persist_frames=persist_frames,
            min_area_ratio=(min_area_pct / 100.0),
        )

    webrtc_streamer(
        key="weapon-detect-webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=factory,
        media_stream_constraints={"video": {"deviceId": None}, "audio": False},
    )

def dashboard_view():
    st.title("📊 Dashboard")
    user: SessionUser = st.session_state["user"]

    if not ALERT_LOG_CSV.exists():
        st.info("No detections yet. Run the detector to log alerts.")
        return

    df = pd.read_csv(ALERT_LOG_CSV)
    if df.empty:
        st.info("No detections yet.")
        return

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        pass

    my = df[df["user_email"].str.lower() == user.email.lower()].copy()
    if my.empty:
        st.info("No detections for this account yet.")
        return

    total = len(my)
    last_time = my["timestamp"].max()
    top_label = my["label"].value_counts().idxmax() if not my["label"].empty else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("My detections", f"{total}")
    c2.metric("Last alert", str(last_time))
    c3.metric("Top label", top_label)

    st.subheader("Recent alerts")
    table_df = my[["timestamp", "label", "confidence"]].copy()
    table_df.rename(columns={
        "timestamp": "Detection Time",
        "label": "Weapon Type",
        "confidence": "Confidence"
    }, inplace=True)
    table_df["Detection Time"] = pd.to_datetime(table_df["Detection Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(table_df.sort_values("Detection Time", ascending=False).head(25), use_container_width=True)

    st.subheader("Latest snapshots")
    latest = my.sort_values("timestamp", ascending=False).head(8)
    imgs = []
    for p in latest["snapshot_path"].tolist():
        pth = Path(p)
        if pth.exists():
            imgs.append(str(pth))
    if imgs:
        st.image(imgs, caption=[Path(i).name for i in imgs], width=240)
    else:
        st.caption("No snapshot files found on disk.")

# Router

def router():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    if st.session_state.user is None:
        st.sidebar.title("Navigation")
        st.sidebar.write("Please sign in.")
        if st.session_state.auth_mode == "signup":
            signup_view()
        else:
            login_view()
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Detection", "Dashboard"], index=0)
    if page == "Detection":
        detection_view()
    else:
        dashboard_view()

router()
