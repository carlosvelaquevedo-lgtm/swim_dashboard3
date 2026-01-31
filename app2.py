import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
import tempfile
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io
import zipfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from collections import deque

# ────────────── CONFIG DEFAULTS ──────────────
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

# ────────────── DATA CLASSES ──────────────
@dataclass
class AthleteProfile:
    height_cm: float
    discipline: str

    def roll_tolerance(self):
        return 1.15 if self.discipline == "triathlon" else 1.0

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

# ────────────── HELPERS ──────────────
def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(np.arccos(np.clip(cosang, -1, 1)))

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

# ────────────── SWIM ANALYZER ──────────────
class SwimAnalyzerClassic:
    def __init__(self, athlete: AthleteProfile, conf_thresh, yaw_thresh):
        self.athlete = athlete
        self.conf_thresh = conf_thresh
        self.yaw_thresh = yaw_thresh
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
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

    def process(self, frame, t):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if not results.pose_landmarks:
            return frame, None
        lm_pixel = {}
        h, w = frame.shape[:2]
        for name in ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow",
                     "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]:
            lm = getattr(self.mp_pose.PoseLandmark, name.upper())
            landmark = results.pose_landmarks.landmark[lm]
            lm_pixel[name] = (landmark.x*w, landmark.y*h)
        conf = statistics.mean([results.pose_landmarks.landmark[getattr(self.mp_pose.PoseLandmark, n.upper())].visibility
                               for n in ["left_shoulder","right_shoulder","left_hip","right_hip"]])
        if conf < self.conf_thresh:
            return frame, None
        # Flip upside-down
        if lm_pixel["left_hip"][1] < lm_pixel["left_shoulder"][1]:
            frame = cv2.flip(frame, -1)
        # Metrics
        elbow = min(calculate_angle(lm_pixel["left_shoulder"], lm_pixel["left_elbow"], lm_pixel["left_wrist"]),
                    calculate_angle(lm_pixel["right_shoulder"], lm_pixel["right_elbow"], lm_pixel["right_wrist"]))
        roll = abs(math.degrees(math.atan2(lm_pixel["left_shoulder"][1]-lm_pixel["right_shoulder"][1],
                                           lm_pixel["left_shoulder"][0]-lm_pixel["right_shoulder"][0])))
        knee_l = calculate_angle(lm_pixel["left_hip"], lm_pixel["left_knee"], lm_pixel["left_ankle"])
        knee_r = calculate_angle(lm_pixel["right_hip"], lm_pixel["right_knee"], lm_pixel["right_ankle"])
        kick_sym = abs(knee_l - knee_r)
        kick_depth = compute_kick_depth_proxy(lm_pixel)
        symmetry_hips = abs(lm_pixel["left_hip"][0]-lm_pixel["right_hip"][0])/w*100
        wrist_y = lm_pixel["left_wrist"][1]; shoulder_y = lm_pixel["left_shoulder"][1]
        underwater = wrist_y > shoulder_y + 20
        phase = "Recovery"
        if underwater:
            if elbow>130: phase="Entry"
            elif elbow>90: phase="Pull"
            else: phase="Push"
        yaw = (lm_pixel["nose"][0] - (lm_pixel["left_shoulder"][0]+lm_pixel["right_shoulder"][0])/2)/abs(lm_pixel["right_shoulder"][0]-lm_pixel["left_shoulder"][0] or 1)
        if abs(yaw) > self.yaw_thresh:
            side = 'R' if yaw>0 else 'L'
            if side==self.breath_side:
                self.breath_persist+=1
            else:
                self.breath_persist=1
                self.breath_side=side
            if self.breath_persist>=MIN_BREATH_HOLD_FRAMES and t-self.last_breath>=MIN_BREATH_GAP_S:
                if side=='L': self.breath_l+=1
                else: self.breath_r+=1
                self.last_breath=t
        self.elbow_win.append(elbow); self.time_win.append(t)
        if len(self.elbow_win)>=9 and detect_local_minimum(list(self.elbow_win)):
            ct=self.time_win[4]
            if not self.stroke_times or ct-self.stroke_times[-1]>=0.5:
                self.stroke_times.append(ct)
        torso_raw = compute_torso_lean(lm_pixel)
        forearm_raw = compute_forearm_vertical(lm_pixel)
        self.torso_buffer.append(torso_raw)
        self.forearm_buffer.append(forearm_raw)
        self.kick_depth_buffer.append(kick_depth)
        torso = statistics.mean(self.torso_buffer)
        forearm = statistics.mean(self.forearm_buffer)
        kick_depth = statistics.mean(self.kick_depth_buffer)
        roll_abs = abs(roll)
        draw_technique_panel(frame, w-200, "YOUR STROKE", torso, forearm, roll_abs, kick_depth, phase, False, self.breath_side)
        draw_technique_panel(frame, 200, "IDEAL REFERENCE", 8.0, 20.0, 45.0, 0.4, "PULL", True, 'N')
        # Score
        torso_dev = max(0, min(abs(torso-8),20))/20*30
        forearm_dev = max(0, min(forearm-35,65-forearm))/65*25
        roll_dev = max(0, min(abs(roll_abs-45),20))/20*20
        kick_sym_dev = max(0, kick_sym/DEFAULT_KICK_SYM_MAX_GOOD)*15
        kick_depth_dev = 0 if DEFAULT_KICK_DEPTH_GOOD[0]<=kick_depth<=DEFAULT_KICK_DEPTH_GOOD[1] else 10
        score = max(0, 100-(torso_dev+forearm_dev+roll_dev+kick_sym_dev+kick_depth_dev))
        metrics = FrameMetrics(t, elbow, knee_l, knee_r, kick_sym, kick_depth, symmetry_hips, score, roll_abs, torso, forearm, phase,
                               self.breath_side if self.breath_side!='N' else "-", conf)
        self.metrics.append(metrics)
        return frame, score

    def get_summary(self):
        if not self.metrics: return SessionSummary(0,0,0,0,0,0,0,0,0,0,0,"No data",1.0)
        d = self.metrics[-1].time_s
        scores = [m.score for m in self.metrics if m.confidence>=DEFAULT_CONF_THRESHOLD]
        rolls = [m.body_roll for m in self.metrics if m.confidence>=DEFAULT_CONF_THRESHOLD]
        ksyms = [m.kick_symmetry for m in self.metrics if m.confidence>=DEFAULT_CONF_THRESHOLD]
        kdepths = [m.kick_depth_proxy for m in self.metrics if m.confidence>=DEFAULT_CONF_THRESHOLD]
        confs = [m.confidence for m in self.metrics]
        sr = 0
        if len(self.stroke_times)>=2:
            dur=self.stroke_times[-1]-self.stroke_times[0]
            if dur>0.1: sr=60*(len(self.stroke_times)-1)/dur
        bpm=(self.breath_l+self.breath_r)/(d/60) if d>0 else 0
        avg_kick_sym=statistics.mean(ksyms) if ksyms else 0
        avg_kick_depth=statistics.mean(kdepths) if kdepths else 0
        kick_status="Good" if avg_kick_sym<DEFAULT_KICK_SYM_MAX_GOOD and DEFAULT_KICK_DEPTH_GOOD[0]<avg_kick_depth<DEFAULT_KICK_DEPTH_GOOD[1] else "Needs Work"
        return SessionSummary(d, statistics.mean(scores), statistics.mean(rolls), max(rolls),
                              sr, bpm, self.breath_l, self.breath_r, len(self.stroke_times),
                              avg_kick_sym, avg_kick_depth, kick_status, statistics.mean(confs),
                              self.best_bytes, self.worst_bytes)

# ────────────── STREAMLIT APP ──────────────
def main():
    st.set_page_config(layout="wide", page_title="Freestyle Swim Analyzer Classic")
    st.title("Freestyle Swim Analyzer Classic – MediaPipe Pose")
    with st.sidebar:
        st.header("Athlete Settings")
        height = st.slider("Height (cm)", 150, 200, 170)
        discipline = st.selectbox("Discipline", ["pool", "triathlon"])
        conf_thresh = st.slider("Min Detection Confidence", 0.3, 0.7, DEFAULT_CONF_THRESHOLD, 0.05)
        yaw_thresh = st.slider("Breath Yaw Threshold", 0.05, 0.3, DEFAULT_YAW_THRESHOLD, 0.01)
    athlete = AthleteProfile(height, discipline)
    analyzer = SwimAnalyzerClassic(athlete, conf_thresh, yaw_thresh)
    uploaded = st.file_uploader("Upload swimming video", type=["mp4","mov"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.getvalue()); input_path=tmp.name
        cap=cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = tempfile.mktemp(suffix=".mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        progress = st.progress(0); status = st.empty()
        frame_idx=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            annotated, _ = analyzer.process(frame, frame_idx/fps)
            writer.write(annotated)
            frame_idx+=1
            progress.progress(frame_idx/total)
            status.text(f"Frame {frame_idx}/{total}")
        cap.release(); writer.release(); os.unlink(input_path)
        st.success("Analysis complete!")
        summary = analyzer.get_summary()
        cols = st.columns(4)
        cols[0].metric("Stroke Rate", f"{summary.stroke_rate:.1f} spm")
        cols[1].metric("Breaths/min", f"{summary.breaths_per_min:.1f}")
        cols[2].metric("Avg Body Roll", f"{summary.avg_body_roll:.1f}°")
        cols[3].metric("Kick Status", summary.kick_status)
        st.video(out_path)
        # CSV
        df = pd.DataFrame([m.__dict__ for m in analyzer.metrics])
        csv_buf = io.StringIO(); df.to_csv(csv_buf,index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), "metrics.csv")
        # ZIP with video + CSV
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("metrics.csv", csv_buf.getvalue())
            with open(out_path,"rb") as f: zf.writestr("annotated_video.mp4", f.read())
        st.download_button("Download ZIP", zip_buf.getvalue(), "swim_analysis.zip")
        os.unlink(out_path)

if __name__=="__main__":
    main()
