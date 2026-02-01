import streamlit as st
import cv2
import numpy as np
import math
import statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import tempfile
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import io
import zipfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# ─────────────────────────────────────────────
# MEDIAPIPE IMPORT WITH ERROR HANDLING
# ─────────────────────────────────────────────

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    st.error(f"MediaPipe import failed: {e}")

# ─────────────────────────────────────────────
# CUSTOM CSS (modern glassmorphism UI)
# ─────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(100, 116, 139, 0.3);
        margin-bottom: 16px;
    }
    
    .metric-card-green { border-left: 4px solid #22c55e; }
    .metric-card-red   { border-left: 4px solid #ef4444; }
    .metric-card-yellow{ border-left: 4px solid #eab308; }
    
    .score-card {
        background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        margin-bottom: 24px;
    }
    
    .drill-card {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(100, 116, 139, 0.3);
        margin-bottom: 12px;
    }
    
    .rec-high   { background: rgba(127, 29, 29, 0.3); border-left: 4px solid #ef4444; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
    .rec-medium { background: rgba(113, 63, 18, 0.3); border-left: 4px solid #eab308; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
    .rec-low    { background: rgba(20, 83, 45, 0.3); border-left: 4px solid #22c55e; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.3);
    }
    
    .css-1d391kg { background: rgba(15, 23, 42, 0.9); }
    
    h1, h2, h3 { color: #f8fafc !important; }
    p, span, label { color: #cbd5e1; }
    
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .legend-dot-green { background: #22c55e; }
    .legend-dot-yellow { background: #eab308; }
    .legend-dot-red { background: #ef4444; }
    .legend-dot-white { background: #ffffff; }
</style>
"""

# ─────────────────────────────────────────────
# CONFIG CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────

DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_YAW_THRESHOLD = 0.15
MIN_BREATH_GAP_S = 1.0
MIN_BREATH_HOLD_FRAMES = 4

DEFAULT_TORSO_GOOD = (4, 12)
DEFAULT_TORSO_OK = (0, 18)
DEFAULT_FOREARM_GOOD = (0, 35)
DEFAULT_FOREARM_OK = (0, 60)
DEFAULT_ROLL_GOOD = (35, 55)
DEFAULT_ROLL_OK = (25, 65)
DEFAULT_KICK_SYM_MAX_GOOD = 15
DEFAULT_KICK_DEPTH_GOOD = (0.25, 0.6)

# ─────────────────────────────────────────────
# ENUMS & DATA MODELS
# ─────────────────────────────────────────────

class SwimPhase(Enum):
    ENTRY = "Entry"
    PULL = "Pull"
    PUSH = "Push"
    RECOVERY = "Recovery"

@dataclass
class AthleteProfile:
    height_cm: float
    discipline: str

@dataclass
class FrameMetrics:
    time_s: float
    elbow_angle: float
    knee_left: float
    knee_right: float
    kick_symmetry: float
    kick_depth_proxy: float
    symmetry_hips: float
    score: float
    body_roll: float
    torso_lean: float
    forearm_vertical: float
    phase: str
    breath_state: str
    confidence: float = 1.0

@dataclass
class TrainingDrill:
    title: str
    description: str
    sets: str
    focus: str

@dataclass
class Recommendation:
    title: str
    description: str
    priority: str

@dataclass
class SessionSummary:
    duration_s: float
    avg_score: float
    avg_body_roll: float
    max_body_roll: float
    stroke_rate: float
    breaths_per_min: float
    breath_left: int
    breath_right: int
    total_strokes: int
    avg_kick_symmetry: float
    avg_kick_depth: float
    kick_status: str
    avg_confidence: float
    best_frame_bytes: Optional[bytes] = None
    worst_frame_bytes: Optional[bytes] = None
    drills: List[TrainingDrill] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def compute_torso_lean(lm_pixel):
    mid_s = ((lm_pixel["left_shoulder"][0] + lm_pixel["right_shoulder"][0]) / 2,
             (lm_pixel["left_shoulder"][1] + lm_pixel["right_shoulder"][1]) / 2)
    mid_h = ((lm_pixel["left_hip"][0] + lm_pixel["right_hip"][0]) / 2,
             (lm_pixel["left_hip"][1] + lm_pixel["right_hip"][1]) / 2)
    dy = mid_s[1] - mid_h[1]
    dx = mid_s[0] - mid_h[0]
    return math.degrees(math.atan2(dy, dx))

def compute_forearm_vertical(lm_pixel):
    dx = lm_pixel["left_wrist"][0] - lm_pixel["left_elbow"][0]
    dy = lm_pixel["left_wrist"][1] - lm_pixel["left_elbow"][1]
    return abs(math.degrees(math.atan2(dx, -dy)))

def compute_kick_depth_proxy(lm_pixel):
    hip_y = (lm_pixel["left_hip"][1] + lm_pixel["right_hip"][1]) / 2
    knee_l_y = lm_pixel["left_knee"][1]
    knee_r_y = lm_pixel["right_knee"][1]
    sh_dist = abs(lm_pixel["left_shoulder"][1] - lm_pixel["left_hip"][1]) or 1.0
    return ((abs(knee_l_y - hip_y) + abs(knee_r_y - hip_y)) / 2) / sh_dist

def get_zone_color(val, good, ok):
    if good[0] <= val <= good[1]: return (0, 180, 0)
    if ok[0] <= val <= ok[1]: return (0, 220, 220)
    return (0, 0, 220)

def detect_local_minimum(arr, threshold=10):
    if len(arr) < 3: return False
    mid = len(arr) // 2
    return arr[mid] < min(arr[:mid] + arr[mid+1:]) and (arr[mid] + threshold) <= min(arr[:mid] + arr[mid+1:])

# ─────────────────────────────────────────────
# VISUAL PANELS
# ─────────────────────────────────────────────

def draw_simplified_silhouette(frame, x, y, color=(180,180,180), th=3):
    cv2.circle(frame, (x, y-50), 18, color, th)
    cv2.line(frame, (x, y-32), (x, y+70), color, th+2)
    cv2.line(frame, (x, y-10), (x-45, y+30), color, th)
    cv2.line(frame, (x, y-10), (x+45, y+30), color, th)
    cv2.line(frame, (x, y+70), (x-35, y+130), color, th)
    cv2.line(frame, (x, y+70), (x+35, y+130), color, th)

def draw_technique_panel(frame, origin_x, title, torso, forearm, roll, kick_depth, phase, is_ideal=False, breath_side='N'):
    h, w = frame.shape[:2]
    px, py = origin_x - 140, 30
    pw, ph = 280, 440
    ov = frame.copy()
    cv2.rectangle(ov, (px, py), (px+pw, py+ph), (0,0,0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, title.upper(), (px+10, py+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255,255,255) if not is_ideal else (200,200,255), 2)

    draw_simplified_silhouette(frame, px+140, py+200, (160,160,160) if is_ideal else (200,200,200), 2)

    tc = get_zone_color(abs(torso), DEFAULT_TORSO_GOOD, DEFAULT_TORSO_OK)
    tlen = 80
    tdx = tlen * math.sin(math.radians(torso))
    tdy = tlen * math.cos(math.radians(torso))
    cv2.line(frame, (px+140, py+100), (int(px+140+tdx), int(py+100+tdy)), tc, 6)
    cv2.putText(frame, f"Torso Lean: {torso:.1f}°", (px+10, py+70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tc, 2)

    if phase in ("Pull", "Push"):
        fc = get_zone_color(forearm, DEFAULT_FOREARM_GOOD, DEFAULT_FOREARM_OK)
        ftxt = f"Forearm to Vert: {forearm:.1f}°"
    else:
        fc = (180,180,180)
        ftxt = "Forearm n/a"
    cdx = 80 * math.sin(math.radians(forearm))
    cdy = 80 * math.cos(math.radians(forearm))
    cv2.line(frame, (px+220, py+100), (int(px+220+cdx), int(py+100+cdy)), fc, 6)
    cv2.putText(frame, ftxt, (px+10, py+110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2)

    rc = get_zone_color(roll, DEFAULT_ROLL_GOOD, DEFAULT_ROLL_OK)
    cv2.putText(frame, f"Body Rot: {roll:.1f}°", (px+10, py+150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc, 2)
    rdx = 60 * math.cos(math.radians(roll))
    rdy = 60 * math.sin(math.radians(roll))
    cv2.line(frame, (px+140, py+260), (int(px+140+rdx), int(py+260+rdy)), rc, 5)

    kdc = get_zone_color(kick_depth * 100, DEFAULT_KICK_DEPTH_GOOD, (0.1, 0.8))
    cv2.putText(frame, f"Kick Depth: {kick_depth:.2f}", (px+10, py+190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kdc, 2)
    kdx = 60 * min(kick_depth / 1.0, 1.0)
    cv2.line(frame, (px+140, py+320), (int(px+140+kdx), py+320), kdc, 5)

    if breath_side != 'N':
        bcolor = (255,165,0) if breath_side == 'L' else (0,191,255)
        btxt = f"Breath: {'Left' if breath_side == 'L' else 'Right'}"
        cv2.putText(frame, btxt, (px+10, py+350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bcolor, 2)
        arrow_x = px + 220
        arrow_y = py + 350
        cv2.arrowedLine(frame, (arrow_x, arrow_y), (arrow_x - 40 if breath_side == 'L' else arrow_x + 40, arrow_y), bcolor, 2, tipLength=0.4)
    else:
        cv2.putText(frame, "Breath: Neutral", (px+10, py+350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

    stxt = "IDEAL REFERENCE" if is_ideal else "YOUR STROKE"
    cv2.putText(frame, stxt, (px+10, py+ph-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

# ─────────────────────────────────────────────
# ANALYZER CLASS (FIXED FOR STREAMLIT CLOUD)
# ─────────────────────────────────────────────

class SwimAnalyzer:
    def __init__(self, athlete: AthleteProfile, conf_thresh, yaw_thresh):
        self.athlete = athlete
        self.conf_thresh = conf_thresh
        self.yaw_thresh = yaw_thresh
        self.pose = None
        self.drawing = None
        self.styles = None
        
        # Initialize MediaPipe with error handling
        self._init_mediapipe()

        self.metrics: List[FrameMetrics] = []
        self.stroke_times = []
        self.breath_l = self.breath_r = 0
        self.breath_side = 'N'
        self.breath_persist = 0
        self.last_breath = -1000
        self.elbow_win = deque(maxlen=9)
        self.time_win = deque(maxlen=9)
        self.best_dev = float('inf')
        self.worst_dev = -float('inf')
        self.best_bytes = self.worst_bytes = None

        self.torso_buffer = deque(maxlen=7)
        self.forearm_buffer = deque(maxlen=7)
        self.kick_depth_buffer = deque(maxlen=7)

    def _init_mediapipe(self):
        """Initialize MediaPipe with multiple fallback options for cloud compatibility."""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not available")
        
        # Try different configurations in order of preference
        configs = [
            # Config 1: Lite model, lower thresholds (most compatible)
            {"model_complexity": 0, "min_detection_confidence": 0.3, "min_tracking_confidence": 0.3, "static_image_mode": False},
            # Config 2: Lite model, static mode
            {"model_complexity": 0, "min_detection_confidence": 0.3, "min_tracking_confidence": 0.3, "static_image_mode": True},
            # Config 3: Lite model only
            {"model_complexity": 0},
        ]
        
        last_error = None
        for i, config in enumerate(configs):
            try:
                self.pose = mp.solutions.pose.Pose(**config)
                self.drawing = mp.solutions.drawing_utils
                self.styles = mp.solutions.drawing_styles
                # Test with a dummy image to ensure it works
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                self.pose.process(test_img)
                return  # Success!
            except Exception as e:
                last_error = e
                if self.pose:
                    try:
                        self.pose.close()
                    except:
                        pass
                    self.pose = None
                continue
        
        # If all configs failed, raise the last error
        raise RuntimeError(f"Failed to initialize MediaPipe after trying all configurations. Last error: {last_error}")

    def process(self, frame, t):
        if self.pose is None:
            return frame, None
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            res = self.pose.process(rgb)
        except Exception as e:
            st.warning(f"Pose processing error: {e}")
            return frame, None
            
        if not res.pose_landmarks:
            return frame, None

        h, w = frame.shape[:2]
        lm_pixel = {}
        names = ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow",
                 "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        vis_sum = 0
        for name in names:
            idx = getattr(mp.solutions.pose.PoseLandmark, name.upper())
            p = res.pose_landmarks.landmark[idx]
            lm_pixel[name] = (p.x * w, p.y * h)
            vis_sum += p.visibility

        conf = vis_sum / len(names)
        if conf < self.conf_thresh:
            return frame, None

        if "left_hip" in lm_pixel and "left_shoulder" in lm_pixel:
            if lm_pixel["left_hip"][1] < lm_pixel["left_shoulder"][1]:
                frame = cv2.flip(frame, -1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    res = self.pose.process(rgb)
                except:
                    return frame, None
                if not res.pose_landmarks:
                    return frame, None
                for name in names:
                    idx = getattr(mp.solutions.pose.PoseLandmark, name.upper())
                    p = res.pose_landmarks.landmark[idx]
                    lm_pixel[name] = (p.x * w, p.y * h)

        elbow = min(
            calculate_angle(lm_pixel["left_shoulder"], lm_pixel["left_elbow"], lm_pixel["left_wrist"]),
            calculate_angle(lm_pixel["right_shoulder"], lm_pixel["right_elbow"], lm_pixel["right_wrist"])
        )

        roll = abs(math.degrees(math.atan2(
            lm_pixel["left_shoulder"][1] - lm_pixel["right_shoulder"][1],
            lm_pixel["left_shoulder"][0] - lm_pixel["right_shoulder"][0]
        )))

        knee_l = calculate_angle(lm_pixel["left_hip"], lm_pixel["left_knee"], lm_pixel["left_ankle"])
        knee_r = calculate_angle(lm_pixel["right_hip"], lm_pixel["right_knee"], lm_pixel["right_ankle"])
        kick_sym = abs(knee_l - knee_r)
        kick_depth_raw = compute_kick_depth_proxy(lm_pixel)

        symmetry_hips = abs(lm_pixel["left_hip"][0] - lm_pixel["right_hip"][0]) / w * 100

        wrist_y = lm_pixel["left_wrist"][1]
        shoulder_y = lm_pixel["left_shoulder"][1]
        underwater = wrist_y > shoulder_y + 20
        phase = "Recovery"
        if underwater:
            if elbow > 130: phase = "Entry"
            elif elbow > 90: phase = "Pull"
            else: phase = "Push"

        yaw = 0
        if "nose" in lm_pixel:
            mid_s = (lm_pixel["left_shoulder"][0] + lm_pixel["right_shoulder"][0]) / 2
            yaw = (lm_pixel["nose"][0] - mid_s) / abs(lm_pixel["right_shoulder"][0] - lm_pixel["left_shoulder"][0] or 1)
        if abs(yaw) > self.yaw_thresh:
            side = 'R' if yaw > 0 else 'L'
            if side == self.breath_side:
                self.breath_persist += 1
            else:
                self.breath_persist = 1
                self.breath_side = side
            if self.breath_persist >= MIN_BREATH_HOLD_FRAMES and t - self.last_breath >= MIN_BREATH_GAP_S:
                if side == 'L': self.breath_l += 1
                else: self.breath_r += 1
                self.last_breath = t

        self.elbow_win.append(elbow)
        self.time_win.append(t)
        if len(self.elbow_win) >= 9 and detect_local_minimum(list(self.elbow_win)):
            ct = self.time_win[4]
            if not self.stroke_times or ct - self.stroke_times[-1] >= 0.5:
                self.stroke_times.append(ct)

        torso_raw = compute_torso_lean(lm_pixel)
        forearm_raw = compute_forearm_vertical(lm_pixel)
        kick_depth_raw = compute_kick_depth_proxy(lm_pixel)

        self.torso_buffer.append(torso_raw)
        self.forearm_buffer.append(forearm_raw)
        self.kick_depth_buffer.append(kick_depth_raw)

        torso = statistics.mean(self.torso_buffer) if self.torso_buffer else torso_raw
        forearm = statistics.mean(self.forearm_buffer) if self.forearm_buffer else forearm_raw
        kick_depth = statistics.mean(self.kick_depth_buffer) if self.kick_depth_buffer else kick_depth_raw
        roll_abs = abs(roll)

        if self.drawing and res.pose_landmarks:
            self.drawing.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=self.styles.get_default_pose_landmarks_style())

        draw_technique_panel(frame, w-200, "YOUR STROKE", torso, forearm, roll_abs, kick_depth, phase, False, self.breath_side)
        draw_technique_panel(frame, 200, "IDEAL REFERENCE", 8.0, 20.0, 45.0, 0.4, "PULL", True, 'N')

        if phase == "Pull":
            dev = abs(elbow - 110)
            if dev < self.best_dev:
                self.best_dev = dev
                _, buf = cv2.imencode('.jpg', frame)
                self.best_bytes = buf.tobytes()
            if dev > self.worst_dev:
                self.worst_dev = dev
                _, buf = cv2.imencode('.jpg', frame)
                self.worst_bytes = buf.tobytes()

        torso_dev = max(0, min(abs(torso - 8), 20)) / 20 * 30
        forearm_dev = max(0, min(forearm - 35, 65 - forearm)) / 65 * 25
        roll_dev = max(0, min(abs(roll_abs - 45), 20)) / 20 * 20
        kick_sym_dev = max(0, kick_sym / DEFAULT_KICK_SYM_MAX_GOOD) * 15
        kick_depth_dev = 0 if DEFAULT_KICK_DEPTH_GOOD[0] <= kick_depth <= DEFAULT_KICK_DEPTH_GOOD[1] else 10
        score = max(0, 100 - (torso_dev + forearm_dev + roll_dev + kick_sym_dev + kick_depth_dev))

        metrics = FrameMetrics(t, elbow, knee_l, knee_r, kick_sym, kick_depth,
                               symmetry_hips, score, roll_abs, torso, forearm, phase,
                               self.breath_side if self.breath_side != 'N' else "-", conf)
        self.metrics.append(metrics)

        return frame, score

    def close(self):
        """Clean up MediaPipe resources."""
        if self.pose:
            try:
                self.pose.close()
            except:
                pass

    def get_summary(self):
        if not self.metrics: return SessionSummary(0,0,0,0,0,0,0,0,0,0,0,"No data",1.0,None,None)

        d = self.metrics[-1].time_s
        scores = [m.score for m in self.metrics if m.confidence >= DEFAULT_CONF_THRESHOLD]
        rolls = [m.body_roll for m in self.metrics if m.confidence >= DEFAULT_CONF_THRESHOLD]
        ksyms = [m.kick_symmetry for m in self.metrics if m.confidence >= DEFAULT_CONF_THRESHOLD]
        kdepths = [m.kick_depth_proxy for m in self.metrics if m.confidence >= DEFAULT_CONF_THRESHOLD]
        confs = [m.confidence for m in self.metrics]

        sr = 0
        if len(self.stroke_times) >= 2:
            dur = self.stroke_times[-1] - self.stroke_times[0]
            if dur > 0.1: sr = 60 * (len(self.stroke_times)-1) / dur

        bpm = (self.breath_l + self.breath_r) / (d/60) if d > 0 else 0

        avg_kick_sym = statistics.mean(ksyms) if ksyms else 0
        avg_kick_depth = statistics.mean(kdepths) if kdepths else 0
        kick_status = "Good" if avg_kick_sym < DEFAULT_KICK_SYM_MAX_GOOD and DEFAULT_KICK_DEPTH_GOOD[0] < avg_kick_depth < DEFAULT_KICK_DEPTH_GOOD[1] else "Needs Work"

        return SessionSummary(
            duration_s=d,
            avg_score=statistics.mean(scores) if scores else 0,
            avg_body_roll=statistics.mean(rolls) if rolls else 0,
            max_body_roll=max(rolls) if rolls else 0,
            stroke_rate=sr,
            breaths_per_min=bpm,
            breath_left=self.breath_l,
            breath_right=self.breath_r,
            total_strokes=len(self.stroke_times),
            avg_kick_symmetry=avg_kick_sym,
            avg_kick_depth=avg_kick_depth,
            kick_status=kick_status,
            avg_confidence=statistics.mean(confs) if confs else 1.0,
            best_frame_bytes=self.best_bytes,
            worst_frame_bytes=self.worst_bytes
        )

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def generate_plots(analyzer):
    if not analyzer.metrics: return io.BytesIO()
    times = [m.time_s for m in analyzer.metrics]
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    axs[0].plot(times, [m.body_roll for m in analyzer.metrics], label="Body Roll", color='purple')
    axs[0].axhspan(DEFAULT_ROLL_GOOD[0], DEFAULT_ROLL_GOOD[1], color='green', alpha=0.2)
    axs[0].set_title("Body Roll Over Time"); axs[0].legend()
    axs[1].plot(times, [m.kick_symmetry for m in analyzer.metrics], label="Kick Symmetry", color='orange')
    axs[1].axhline(DEFAULT_KICK_SYM_MAX_GOOD, color='red', linestyle='--'); axs[1].set_title("Kick Symmetry (°)"); axs[1].legend()
    axs[2].plot(times, [m.kick_depth_proxy for m in analyzer.metrics], label="Kick Depth Proxy", color='cyan')
    axs[2].axhspan(DEFAULT_KICK_DEPTH_GOOD[0], DEFAULT_KICK_DEPTH_GOOD[1], color='green', alpha=0.2); axs[2].set_title("Kick Depth Proxy (norm)"); axs[2].legend()
    axs[3].plot(times, [m.score for m in analyzer.metrics], label="Technique Score", color='lime'); axs[3].set_title("Technique Score Over Time"); axs[3].legend()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────

def generate_pdf_report(summary: SessionSummary, filename: str, plot_buffer: io.BytesIO) -> io.BytesIO:
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontSize=24, textColor=colors.HexColor('#06b6d4'), spaceAfter=20))

    story = []
    story.append(Paragraph("Freestyle Swimming Technique Analysis Report", styles['CustomTitle']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Session Information", styles['Heading2']))
    session_data = [
        ['File', filename],
        ['Duration', f"{summary.duration_s:.1f}s"],
        ['Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ['Avg Confidence', f"{summary.avg_confidence*100:.1f}%"]
    ]
    story.append(Table(session_data, colWidths=[2*inch, 4*inch]))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Performance Metrics", styles['Heading2']))
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Technique Score', f"{summary.avg_score:.1f}/100", 'Good' if summary.avg_score >= 70 else 'Needs Work'],
        ['Stroke Rate', f"{summary.stroke_rate:.1f} spm", 'Good'],
        ['Breaths/min', f"{summary.breaths_per_min:.1f}", 'Balanced' if abs(summary.breath_left - summary.breath_right) <= 5 else 'Asymmetric'],
        ['Avg Body Roll', f"{summary.avg_body_roll:.1f}°", 'Good' if DEFAULT_ROLL_GOOD[0] <= summary.avg_body_roll <= DEFAULT_ROLL_GOOD[1] else 'Check'],
        ['Max Body Roll', f"{summary.max_body_roll:.1f}°", 'Good' if summary.max_body_roll <= DEFAULT_ROLL_GOOD[1] else 'Excessive'],
        ['Kick Symmetry', f"{summary.avg_kick_symmetry:.1f}°", summary.kick_status],
        ['Kick Depth Proxy', f"{summary.avg_kick_depth:.2f}", 'Good']
    ]
    t = Table(metrics_data, colWidths=[2*inch, 2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ]))
    story.append(t)

    if summary.best_frame_bytes or summary.worst_frame_bytes:
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Best & Worst Frames (Pull Phase)", styles['Heading2']))
        if summary.best_frame_bytes:
            best_img = RLImage(io.BytesIO(summary.best_frame_bytes))
            best_img.drawWidth = 3*inch
            best_img.drawHeight = 2*inch
            story.append(Paragraph("Best Pull Frame", styles['Normal']))
            story.append(best_img)
        if summary.worst_frame_bytes:
            worst_img = RLImage(io.BytesIO(summary.worst_frame_bytes))
            worst_img.drawWidth = 3*inch
            worst_img.drawHeight = 2*inch
            story.append(Paragraph("Worst Pull Frame", styles['Normal']))
            story.append(worst_img)
        story.append(Spacer(1, 0.2*inch))

    if summary.drills:
        story.append(Paragraph("Recommended Drills", styles['Heading2']))
        for d in summary.drills:
            story.append(Paragraph(f"<b>{d.title}</b><br/>{d.description}<br/><i>{d.sets} | {d.focus}</i>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

    if summary.recommendations:
        story.append(Paragraph("Recommendations", styles['Heading2']))
        for r in summary.recommendations:
            story.append(Paragraph(f"<b>[{r.priority.upper()}] {r.title}</b><br/>{r.description}", styles['Normal']))

    if plot_buffer.getvalue():
        story.append(PageBreak())
        story.append(Paragraph("Analysis Charts", styles['Heading2']))
        plot_buffer.seek(0)
        img = RLImage(plot_buffer)
        img.drawWidth = 7*inch
        img.drawHeight = 9*inch
        story.append(img)

    pdf.build(story)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# CSV & ZIP
# ─────────────────────────────────────────────

def export_to_csv(analyzer):
    data = {
        'time_s': [m.time_s for m in analyzer.metrics],
        'elbow_angle': [m.elbow_angle for m in analyzer.metrics],
        'kick_symmetry': [m.kick_symmetry for m in analyzer.metrics],
        'kick_depth_proxy': [m.kick_depth_proxy for m in analyzer.metrics],
        'body_roll': [m.body_roll for m in analyzer.metrics],
        'torso_lean': [m.torso_lean for m in analyzer.metrics],
        'forearm_vertical': [m.forearm_vertical for m in analyzer.metrics],
        'phase': [m.phase for m in analyzer.metrics],
        'breath_state': [m.breath_state for m in analyzer.metrics],
        'confidence': [m.confidence for m in analyzer.metrics],
        'score': [m.score for m in analyzer.metrics]
    }
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def create_results_bundle(video_path, csv_buf, pdf_buf, plot_buf, timestamp, analyzer):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                zipf.writestr(f"annotated_{timestamp}.mp4", f.read())
        zipf.writestr(f"report_{timestamp}.pdf", pdf_buf.getvalue())
        zipf.writestr(f"charts_{timestamp}.png", plot_buf.getvalue())
        zipf.writestr(f"data_{timestamp}.csv", csv_buf.getvalue())
        if analyzer.best_bytes:
            zipf.writestr(f"best_frame_{timestamp}.jpg", analyzer.best_bytes)
        if analyzer.worst_bytes:
            zipf.writestr(f"worst_frame_{timestamp}.jpg", analyzer.worst_bytes)
    zip_buf.seek(0)
    return zip_buf

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(layout="wide", page_title="Freestyle Swim Analyzer Pro")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🏊 Freestyle Swim Technique Analyzer Pro")
    st.markdown("AI-powered analysis with modern UI & detailed coaching feedback")

    # Check MediaPipe availability
    if not MEDIAPIPE_AVAILABLE:
        st.error("❌ MediaPipe is not available. Please check installation.")
        st.code("pip install mediapipe>=0.10.0")
        return

    with st.sidebar:
        st.header("Athlete & Settings")
        height = st.slider("Height (cm)", 150, 200, 170)
        discipline = st.selectbox("Discipline", ["pool", "triathlon"])
        conf_thresh = st.slider("Confidence Threshold", 0.3, 0.7, DEFAULT_CONF_THRESHOLD, 0.05)
        yaw_thresh = st.slider("Yaw Threshold", 0.05, 0.3, DEFAULT_YAW_THRESHOLD, 0.01)

    athlete = AthleteProfile(height, discipline)

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov"])

    if uploaded:
        # Initialize analyzer with error handling
        try:
            analyzer = SwimAnalyzer(athlete, conf_thresh, yaw_thresh)
        except Exception as e:
            st.error(f"❌ Failed to initialize pose detection: {e}")
            st.info("This may be a temporary issue. Please try refreshing the page.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded.getvalue())
            input_path = tmp_in.name

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        progress = st.progress(0)
        status = st.empty()

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                annotated, _ = analyzer.process(frame, frame_idx / fps)
                writer.write(annotated)
                frame_idx += 1
                if total > 0:
                    progress.progress(min(frame_idx / total, 1.0))
                status.text(f"Processing frame {frame_idx}/{total}")

            cap.release()
            writer.release()
            
            # Clean up input file
            try:
                os.unlink(input_path)
            except:
                pass

            summary = analyzer.get_summary()
            plot_buf = generate_plots(analyzer)
            pdf_buf = generate_pdf_report(summary, uploaded.name, plot_buf)
            csv_buf = export_to_csv(analyzer)
            zip_buf = create_results_bundle(out_path, csv_buf, pdf_buf, plot_buf,
                                            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), analyzer)

            # Clean up analyzer
            analyzer.close()

            st.success("✅ Analysis complete!")

            # Score card
            score_color = "#22c55e" if summary.avg_score >= 70 else "#eab308" if summary.avg_score >= 50 else "#ef4444"
            st.markdown(f"""
            <div class="score-card">
                <h2 style="margin:0; color:white;">Technique Score</h2>
                <div style="font-size: 48px; font-weight: bold; color: {score_color};">{summary.avg_score:.1f}/100</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            cols = st.columns(4)
            cols[0].metric("Stroke Rate", f"{summary.stroke_rate:.1f} spm")
            cols[1].metric("Breaths/min", f"{summary.breaths_per_min:.1f}")
            cols[2].metric("Avg Body Roll", f"{summary.avg_body_roll:.1f}°")
            cols[3].metric("Kick Depth", f"{summary.avg_kick_depth:.2f}", delta=summary.kick_status)

            # Best/Worst frames
            col1, col2 = st.columns(2)
            with col1:
                if summary.best_frame_bytes:
                    st.image(summary.best_frame_bytes, caption="Best Pull Frame")
            with col2:
                if summary.worst_frame_bytes:
                    st.image(summary.worst_frame_bytes, caption="Worst Pull Frame")

            # Video display
            if os.path.exists(out_path):
                st.video(out_path)

            st.download_button("📦 Download Full Bundle (ZIP)", zip_buf, "swim_analysis.zip", "application/zip")

            # Clean up output file
            try:
                if os.path.exists(out_path):
                    os.unlink(out_path)
            except:
                pass

        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            
            # Cleanup on error
            try:
                cap.release()
                writer.release()
                analyzer.close()
            except:
                pass
            try:
                if os.path.exists(input_path): os.unlink(input_path)
            except:
                pass
            try:
                if os.path.exists(out_path): os.unlink(out_path)
            except:
                pass

if __name__ == "__main__":
    main()
