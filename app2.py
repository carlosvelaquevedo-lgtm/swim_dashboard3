import streamlit as st
import cv2
import numpy as np
import math
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import tempfile
import os
import statistics

# ─────────────────────────────────────────────
# ENUMS & DATA MODELS
# ─────────────────────────────────────────────

class SwimPhase(Enum):
    ENTRY = "Entry"
    PULL = "Pull"
    PUSH = "Push"
    RECOVERY = "Recovery"

class CameraView(Enum):
    SIDE = "Side"
    FRONT = "Front"
    DIAGONAL = "Diagonal"
    UNKNOWN = "Unknown"

@dataclass
class AthleteProfile:
    height_cm: float
    discipline: str  # "pool" or "triathlon"

    def roll_tolerance(self):
        return 1.15 if self.discipline == "triathlon" else 1.0

# ─────────────────────────────────────────────
# GEOMETRY UTILITIES
# ─────────────────────────────────────────────

def angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def detect_camera_view(lm_norm):
    sw = abs(lm_norm["left_shoulder"][0] - lm_norm["right_shoulder"][0])
    sh = abs(lm_norm["left_shoulder"][1] - lm_norm["right_shoulder"][1])
    if sw < 1e-6:
        return CameraView.UNKNOWN
    r = sh / sw
    if r > 0.9:
        return CameraView.FRONT
    if r < 0.4:
        return CameraView.SIDE
    return CameraView.DIAGONAL

def compute_torso_lean(lm_pixel):
    mid_shoulder = (
        (lm_pixel["left_shoulder"][0] + lm_pixel["right_shoulder"][0]) / 2,
        (lm_pixel["left_shoulder"][1] + lm_pixel["right_shoulder"][1]) / 2
    )
    mid_hip = (
        (lm_pixel["left_hip"][0] + lm_pixel["right_hip"][0]) / 2,
        (lm_pixel["left_hip"][1] + lm_pixel["right_hip"][1]) / 2
    )
    dy = mid_shoulder[1] - mid_hip[1]
    dx = mid_shoulder[0] - mid_hip[0]
    return math.degrees(math.atan2(dy, dx))

def compute_forearm_to_vertical(lm_pixel):
    dx = lm_pixel["left_wrist"][0] - lm_pixel["left_elbow"][0]
    dy = lm_pixel["left_wrist"][1] - lm_pixel["left_elbow"][1]
    return abs(math.degrees(math.atan2(dx, -dy)))

def get_zone_color(value, good_range, ok_range):
    if good_range[0] <= value <= good_range[1]:
        return (0, 180, 0)    # green
    elif ok_range[0] <= value <= ok_range[1]:
        return (0, 220, 220)  # yellow/cyan
    else:
        return (0, 0, 220)    # red

# ─────────────────────────────────────────────
# SIMPLIFIED BODY SILHOUETTE
# ─────────────────────────────────────────────

def draw_simplified_silhouette(frame, origin_x, base_y, color=(180, 180, 180), thickness=3):
    cv2.circle(frame, (origin_x, base_y - 50), 18, color, thickness)
    cv2.line(frame, (origin_x, base_y - 32), (origin_x, base_y + 70), color, thickness + 2)
    cv2.line(frame, (origin_x, base_y - 10), (origin_x - 45, base_y + 30), color, thickness)
    cv2.line(frame, (origin_x, base_y - 10), (origin_x + 45, base_y + 30), color, thickness)
    cv2.line(frame, (origin_x, base_y + 70), (origin_x - 35, base_y + 130), color, thickness)
    cv2.line(frame, (origin_x, base_y + 70), (origin_x + 35, base_y + 130), color, thickness)

# ─────────────────────────────────────────────
# DRAW TECHNIQUE COMPARISON PANEL (now includes body rotation)
# ─────────────────────────────────────────────

def draw_technique_panel(
    frame,
    origin_x,
    title,
    torso_angle,
    forearm_angle,
    roll_angle,
    phase,
    is_ideal=False
):
    h, w = frame.shape[:2]
    panel_x, panel_y = origin_x - 140, 30
    panel_w, panel_h = 280, 300  # taller for extra metric
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame, title.upper(), (panel_x + 10, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255) if not is_ideal else (200,200,255), 2)

    # Silhouette
    silhouette_color = (160,160,160) if is_ideal else (200,200,200)
    draw_simplified_silhouette(frame, panel_x + 140, panel_y + 140, color=silhouette_color, thickness=2)

    # Torso lean
    torso_color = get_zone_color(abs(torso_angle), (4, 12), (0, 18))
    torso_len = 80
    torso_dx = torso_len * math.sin(math.radians(torso_angle))
    torso_dy = torso_len * math.cos(math.radians(torso_angle))
    cv2.line(frame, (panel_x + 140, panel_y + 80),
             (int(panel_x + 140 + torso_dx), int(panel_y + 80 + torso_dy)),
             torso_color, 6)
    cv2.putText(frame, f"Torso Lean: {torso_angle:.1f}°", (panel_x + 10, panel_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, torso_color, 2)

    # Forearm catch
    if phase in (SwimPhase.PULL.value, SwimPhase.PUSH.value):
        catch_color = get_zone_color(forearm_angle, (0, 35), (0, 60))
        catch_text = f"Forearm to Vertical: {forearm_angle:.1f}°"
    else:
        catch_color = (180, 180, 180)
        catch_text = "Forearm n/a"
    catch_len = 80
    catch_dx = catch_len * math.sin(math.radians(forearm_angle))
    catch_dy = catch_len * math.cos(math.radians(forearm_angle))
    cv2.line(frame, (panel_x + 220, panel_y + 80),
             (int(panel_x + 220 + catch_dx), int(panel_y + 80 + catch_dy)),
             catch_color, 6)
    cv2.putText(frame, catch_text, (panel_x + 10, panel_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, catch_color, 2)

    # Body rotation (new)
    roll_color = get_zone_color(roll_angle, (35, 55), (25, 65))  # Fine-tuned: good 35-55°, ok 25-65°
    cv2.putText(frame, f"Body Rotation: {roll_angle:.1f}°", (panel_x + 10, panel_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, roll_color, 2)
    # Rotation indicator line (horizontal plane)
    rot_x = panel_x + 140
    rot_y = panel_y + 220
    rot_len = 60
    rot_dx = rot_len * math.cos(math.radians(roll_angle))  # horizontal component
    rot_dy = rot_len * math.sin(math.radians(roll_angle))  # vertical for tilt
    cv2.line(frame, (rot_x, rot_y), (int(rot_x + rot_dx), int(rot_y + rot_dy)), roll_color, 5)

    # Status
    status_text = "IDEAL REFERENCE" if is_ideal else "YOUR STROKE"
    cv2.putText(frame, status_text, (panel_x + 10, panel_y + panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

PHASE_WEIGHTS = {
    SwimPhase.PULL.value: 1.0,
    SwimPhase.PUSH.value: 0.9,
    SwimPhase.ENTRY.value: 0.6,
    SwimPhase.RECOVERY.value: 0.3,
}

VIEW_METRICS = {
    CameraView.SIDE: {"elbow", "roll"},
    CameraView.DIAGONAL: {"elbow"},
    CameraView.FRONT: {"symmetry"},
}

def technique_score(elbow_dev, roll_dev, sym_dev):
    score = 100.0
    score -= elbow_dev * 0.8
    score -= roll_dev * 0.6
    score -= sym_dev * 0.5
    return max(score, 0.0)

# ─────────────────────────────────────────────
# ANALYZER
# ─────────────────────────────────────────────

class SwimAnalyzer:
    def __init__(self, athlete: AthleteProfile):
        self.athlete = athlete
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.view_counts = {v: 0 for v in CameraView}
        self.score_sum = 0.0
        self.score_weight = 0.0
        self.roll_history = []  # (time, signed roll) for analysis

    def process(self, frame, t):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_pose.process(rgb)
        if not res.pose_landmarks:
            return frame, None

        h, w, _ = frame.shape

        lm_pixel = {}
        lm_norm = {}
        landmarks_names = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip"
        ]
        vis = []
        for name in landmarks_names:
            idx = getattr(mp.solutions.pose.PoseLandmark, name.upper())
            p = res.pose_landmarks.landmark[idx]
            x_pixel = p.x * w
            y_pixel = p.y * h
            lm_pixel[name] = (x_pixel, y_pixel)
            lm_norm[name] = (p.x, p.y)
            vis.append(p.visibility)

        confidence = np.mean(vis) if vis else 0.0

        view = detect_camera_view(lm_norm)
        self.view_counts[view] += 1
        dominant_view = max(self.view_counts, key=self.view_counts.get)

        elbow_angle = angle(
            lm_pixel["left_shoulder"],
            lm_pixel["left_elbow"],
            lm_pixel["left_wrist"]
        )

        ls = lm_pixel["left_shoulder"]
        rs = lm_pixel["right_shoulder"]
        dy = ls[1] - rs[1]
        dx = ls[0] - rs[0]
        roll = math.degrees(math.atan2(dy, dx)) if dx != 0 else (90.0 if dy > 0 else -90.0)
        self.roll_history.append((t, roll))  # store signed roll

        symmetry = abs(lm_pixel["left_hip"][0] - lm_pixel["right_hip"][0]) / w * 100

        wrist_y = lm_pixel["left_wrist"][1]
        shoulder_y = lm_pixel["left_shoulder"][1]
        underwater = wrist_y > shoulder_y + 20
        if underwater:
            if elbow_angle > 130:
                phase = SwimPhase.ENTRY
            elif elbow_angle > 90:
                phase = SwimPhase.PULL
            else:
                phase = SwimPhase.PUSH
        else:
            phase = SwimPhase.RECOVERY

        self.mp_drawing.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        enabled = VIEW_METRICS.get(dominant_view, set())
        elbow_dev = abs(elbow_angle - 110) if "elbow" in enabled else 0
        roll_dev = abs(abs(roll) - 45 * self.athlete.roll_tolerance()) if "roll" in enabled else 0
        sym_dev = symmetry if "symmetry" in enabled else 0

        base_score = technique_score(elbow_dev, roll_dev, sym_dev)
        weighted_score = base_score * PHASE_WEIGHTS[phase.value] * max(confidence, 0.2)

        self.score_sum += weighted_score
        self.score_weight += max(confidence, 0.2)

        # New metrics
        torso_lean = compute_torso_lean(lm_pixel)
        forearm_vertical = compute_forearm_to_vertical(lm_pixel)
        roll_abs = abs(roll)

        # Overlay
        roll_color = get_zone_color(roll_abs, (35, 55), (25, 65))
        cv2.putText(frame, f"Body Roll: {roll_abs:.1f}°", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, roll_color, 2)

        # Panels
        draw_technique_panel(
            frame,
            origin_x = w - 200,
            title = "YOUR STROKE",
            torso_angle = torso_lean,
            forearm_angle = forearm_vertical,
            roll_angle = roll_abs,
            phase = phase.value,
            is_ideal = False
        )

        draw_technique_panel(
            frame,
            origin_x = 200,
            title = "IDEAL REFERENCE",
            torso_angle = 8.0,
            forearm_angle = 20.0,
            roll_angle = 45.0,  # typical ideal peak roll
            phase = "PULL",
            is_ideal = True
        )

        cv2.putText(frame, f"Phase: {phase.value} | Score: {int(weighted_score)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame, weighted_score

    def get_body_rotation_metrics(self):
        if not self.roll_history:
            return {"avg_abs": 0.0, "max_abs": 0.0, "status": "No data", "color": "#94a3b8"}

        abs_rolls = [abs(r) for t, r in self.roll_history]
        avg_abs = statistics.mean(abs_rolls)
        max_abs = max(abs_rolls)
        status = "Good" if 35 <= avg_abs <= 55 else ("Too flat" if avg_abs < 35 else "Excessive")
        color = "#22c55e" if "Good" in status else "#ef4444"
        return {"avg_abs": avg_abs, "max_abs": max_abs, "status": status, "color": color}

    def final_score(self):
        return self.score_sum / self.score_weight if self.score_weight > 0 else 0.0

# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Freestyle Swim Analyzer v3.4")
st.title("🏊 Freestyle Swim Technique Analyzer — v3.4 (with Body Rotation Analysis)")

st.caption("Added:\n"
           "• Body rotation tracking & analysis (avg/max roll, status)\n"
           "• Body Roll in overlays & panels (green 35-55°, yellow 25-65°, red outside)\n"
           "• Rotation line indicator in panels")

with st.sidebar:
    st.header("Athlete Profile")
    height = st.slider("Height (cm)", 150, 200, 170)
    discipline = st.selectbox("Discipline", ["pool", "triathlon"])

athlete = AthleteProfile(height, discipline)
analyzer = SwimAnalyzer(athlete)

uploaded = st.file_uploader("Upload swimming video (MP4/MOV)", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded.getvalue())
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Cannot open video file.")
        if os.path.exists(input_path):
            os.unlink(input_path)
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fd_out, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd_out)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0

    try:
        with st.spinner("Analyzing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, _ = analyzer.process(frame, frame_count / fps)
                writer.write(annotated)
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Frame {frame_count}/{total_frames} ({progress:.0%})")

        writer.release()
        cap.release()
        if os.path.exists(input_path):
            os.unlink(input_path)

        final_score = analyzer.final_score()
        rot_metrics = analyzer.get_body_rotation_metrics()

        st.success(f"**Final Technique Score: {final_score:.1f} / 100**")

        st.markdown(f"<div style='font-size:24px; color: {rot_metrics['color']}; margin:20px 0;'>"
                    f"**Body Rotation Analysis**<br>"
                    f"Average: {rot_metrics['avg_abs']:.1f}° | Max: {rot_metrics['max_abs']:.1f}°<br>"
                    f"Status: {rot_metrics['status']} (Ideal avg 35-55°)</div>", unsafe_allow_html=True)

        st.video(out_path)

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
