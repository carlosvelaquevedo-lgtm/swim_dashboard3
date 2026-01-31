import streamlit as st
import cv2
import numpy as np
import math
import statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
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
import mediapipe as mp

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
# GEOMETRY & HELPERS
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

def get_zone_color(val, good=(0,0), ok=(0,0)):
    if good[0] <= val <= good[1]: return (0, 180, 0)
    if ok[0] <= val <= ok[1]: return (0, 220, 220)
    return (0, 0, 220)

def detect_view(lm_norm):
    sw = abs(lm_norm["left_shoulder"][0] - lm_norm["right_shoulder"][0])
    sh = abs(lm_norm["left_shoulder"][1] - lm_norm["right_shoulder"][1])
    nose_vis = lm_norm.get("nose", (0,0))[0] != 0
    if sw < 1e-6: return CameraView.UNKNOWN
    r = sh / sw
    if r > 0.9 and nose_vis: return CameraView.FRONT
    if r < 0.4: return CameraView.SIDE
    return CameraView.DIAGONAL

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

def draw_technique_panel(frame, origin_x, title, torso, forearm, roll, phase, is_ideal=False):
    h, w = frame.shape[:2]
    px, py = origin_x - 140, 30
    pw, ph = 280, 360
    ov = frame.copy()
    cv2.rectangle(ov, (px, py), (px+pw, py+ph), (0,0,0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, title.upper(), (px+10, py+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255,255,255) if not is_ideal else (200,200,255), 2)

    draw_simplified_silhouette(frame, px+140, py+180, (160,160,160) if is_ideal else (200,200,200), 2)

    tc = get_zone_color(abs(torso), (4,12), (0,18))
    tlen = 80
    tdx = tlen * math.sin(math.radians(torso))
    tdy = tlen * math.cos(math.radians(torso))
    cv2.line(frame, (px+140, py+80), (int(px+140+tdx), int(py+80+tdy)), tc, 6)
    cv2.putText(frame, f"Torso: {torso:.1f}°", (px+10, py+70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tc, 2)

    if phase in ("Pull", "Push"):
        fc = get_zone_color(forearm, (0,35), (0,60))
        ftxt = f"Forearm: {forearm:.1f}°"
    else:
        fc = (180,180,180)
        ftxt = "Forearm n/a"
    cdx = 80 * math.sin(math.radians(forearm))
    cdy = 80 * math.cos(math.radians(forearm))
    cv2.line(frame, (px+220, py+80), (int(px+220+cdx), int(py+80+cdy)), fc, 6)
    cv2.putText(frame, ftxt, (px+10, py+110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2)

    rc = get_zone_color(roll, (35,55), (25,65))
    cv2.putText(frame, f"Rotation: {roll:.1f}°", (px+10, py+150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc, 2)
    rdx = 60 * math.cos(math.radians(roll))
    rdy = 60 * math.sin(math.radians(roll))
    cv2.line(frame, (px+140, py+260), (int(px+140+rdx), int(py+260+rdy)), rc, 5)

    stxt = "IDEAL REFERENCE" if is_ideal else "YOUR STROKE"
    cv2.putText(frame, stxt, (px+10, py+ph-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

# ─────────────────────────────────────────────
# ANALYZER CLASS
# ─────────────────────────────────────────────

class SwimAnalyzer:
    def __init__(self, athlete: AthleteProfile):
        self.athlete = athlete
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

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

    def process(self, frame, t):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return frame, None

        h, w = frame.shape[:2]
        lm = {}
        for name in ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow",
                     "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]:
            idx = getattr(mp.solutions.pose.PoseLandmark, name.upper())
            p = res.pose_landmarks.landmark[idx]
            lm[name] = (p.x * w, p.y * h)

        conf = statistics.mean([res.pose_landmarks.landmark[getattr(mp.solutions.pose.PoseLandmark, n.upper())].visibility
                                for n in ["left_shoulder","right_shoulder","left_hip","right_hip"]])

        if conf < 0.4:
            return frame, None

        elbow = min(
            calculate_angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"]),
            calculate_angle(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])
        )

        roll = abs(math.degrees(math.atan2(
            lm["left_shoulder"][1] - lm["right_shoulder"][1],
            lm["left_shoulder"][0] - lm["right_shoulder"][0]
        )))

        knee_l = calculate_angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
        knee_r = calculate_angle(lm["right_hip"], lm["right_knee"], lm["right_ankle"])
        kick_sym = abs(knee_l - knee_r)
        kick_depth = compute_kick_depth_proxy(lm)

        symmetry_hips = abs(lm["left_hip"][0] - lm["right_hip"][0]) / w * 100

        wrist_y = lm["left_wrist"][1]
        shoulder_y = lm["left_shoulder"][1]
        underwater = wrist_y > shoulder_y + 20
        phase = "Recovery"
        if underwater:
            if elbow > 130: phase = "Entry"
            elif elbow > 90: phase = "Pull"
            else: phase = "Push"

        yaw = 0
        if "nose" in lm:
            mid_s = (lm["left_shoulder"][0] + lm["right_shoulder"][0]) / 2
            yaw = (lm["nose"][0] - mid_s) / abs(lm["right_shoulder"][0] - lm["left_shoulder"][0] or 1)
        if abs(yaw) > 0.15:
            side = 'R' if yaw > 0 else 'L'
            if side == self.breath_side:
                self.breath_persist += 1
            else:
                self.breath_persist = 1
                self.breath_side = side
            if self.breath_persist >= 4 and t - self.last_breath >= 1.0:
                if side == 'L': self.breath_l += 1
                else: self.breath_r += 1
                self.last_breath = t

        self.elbow_win.append(elbow)
        self.time_win.append(t)
        if len(self.elbow_win) >= 9 and detect_local_minimum(list(self.elbow_win), 10)[0]:
            ct = self.time_win[4]
            if not self.stroke_times or ct - self.stroke_times[-1] >= 0.5:
                self.stroke_times.append(ct)

        torso = compute_torso_lean(lm)
        forearm = compute_forearm_vertical(lm)
        roll_abs = abs(roll)

        self.drawing.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.styles.get_default_pose_landmarks_style())

        draw_technique_panel(frame, w-200, "YOUR STROKE", torso, forearm, roll_abs, phase, False)
        draw_technique_panel(frame, 200, "IDEAL REFERENCE", 8.0, 20.0, 45.0, "PULL", True)

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

        metrics = FrameMetrics(t, elbow, knee_l, knee_r, kick_sym, kick_depth,
                               symmetry_hips, 80.0, roll_abs, torso, forearm, phase,
                               self.breath_side if self.breath_side != 'N' else "-", conf)
        self.metrics.append(metrics)

        return frame, 80.0

    def get_summary(self):
        if not self.metrics: return SessionSummary(0,0,0,0,0,0,0,0,0,0,0,"No data",1.0)

        d = self.metrics[-1].time_s
        scores = [m.score for m in self.metrics if m.confidence >= 0.4]
        rolls = [m.body_roll for m in self.metrics if m.confidence >= 0.4]
        ksyms = [m.kick_symmetry for m in self.metrics if m.confidence >= 0.4]
        kdepths = [m.kick_depth_proxy for m in self.metrics if m.confidence >= 0.4]
        confs = [m.confidence for m in self.metrics]

        sr = 0
        if len(self.stroke_times) >= 2:
            dur = self.stroke_times[-1] - self.stroke_times[0]
            if dur > 0.1: sr = 60 * (len(self.stroke_times)-1) / dur

        bpm = (self.breath_l + self.breath_r) / (d/60) if d > 0 else 0

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
            avg_kick_symmetry=statistics.mean(ksyms) if ksyms else 0,
            avg_kick_depth=statistics.mean(kdepths) if kdepths else 0,
            kick_status="Good" if statistics.mean(ksyms)<15 and 0.25<statistics.mean(kdepths)<0.6 else "Needs Work",
            avg_confidence=statistics.mean(confs) if confs else 1.0,
            best_frame_bytes=self.best_bytes,
            worst_frame_bytes=self.worst_bytes
        )

# ─────────────────────────────────────────────
# PLOTS (expanded)
# ─────────────────────────────────────────────

def generate_plots(analyzer):
    if not analyzer.metrics: return io.BytesIO()

    times = [m.time_s for m in analyzer.metrics]
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    axs[0].plot(times, [m.body_roll for m in analyzer.metrics], label="Body Roll", color='purple')
    axs[0].axhspan(35, 55, color='green', alpha=0.2)
    axs[0].set_title("Body Roll Over Time")
    axs[0].legend()

    axs[1].plot(times, [m.kick_symmetry for m in analyzer.metrics], label="Kick Symmetry", color='orange')
    axs[1].axhline(15, color='red', linestyle='--')
    axs[1].set_title("Kick Symmetry (°)")
    axs[1].legend()

    axs[2].plot(times, [m.kick_depth_proxy for m in analyzer.metrics], label="Kick Depth Proxy", color='cyan')
    axs[2].axhspan(0.25, 0.6, color='green', alpha=0.2)
    axs[2].set_title("Kick Depth Proxy (normalized)")
    axs[2].legend()

    axs[3].plot(times, [m.score for m in analyzer.metrics], label="Technique Score", color='lime')
    axs[3].set_title("Technique Score Over Time")
    axs[3].legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# PDF REPORT (expanded with kick metrics)
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
        ['Confidence', f"{summary.avg_confidence*100:.1f}%"]
    ]
    story.append(Table(session_data).setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)])))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Performance Metrics", styles['Heading2']))
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Technique Score', f"{summary.avg_score:.1f}/100", 'Good' if summary.avg_score >= 70 else 'Needs Work'],
        ['Stroke Rate', f"{summary.stroke_rate:.1f} spm", 'Good'],
        ['Breaths/min', f"{summary.breaths_per_min:.1f}", 'Balanced' if abs(summary.breath_left - summary.breath_right) <= 5 else 'Asymmetric'],
        ['Avg Body Roll', f"{summary.avg_body_roll:.1f}°", 'Good' if 35 <= summary.avg_body_roll <= 55 else 'Check'],
        ['Max Body Roll', f"{summary.max_body_roll:.1f}°", 'Good' if summary.max_body_roll <= 55 else 'Excessive'],
        ['Kick Symmetry', f"{summary.avg_kick_symmetry:.1f}°", summary.kick_status],
        ['Kick Depth Proxy', f"{summary.avg_kick_depth:.2f}", 'Good']
    ]
    story.append(Table(metrics_data).setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)])))

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
        img = RLImage(plot_buffer)
        img.drawWidth = 7*inch
        img.drawHeight = 5*inch
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
        'confidence': [m.confidence for m in analyzer.metrics]
    }
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def create_results_bundle(video_path, csv_buf, pdf_buf, plot_buf, timestamp):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with open(video_path, 'rb') as f:
            zipf.writestr(f"annotated_{timestamp}.mp4", f.read())
        zipf.writestr(f"report_{timestamp}.pdf", pdf_buf.getvalue())
        zipf.writestr(f"charts_{timestamp}.png", plot_buf.getvalue())
        zipf.writestr(f"data_{timestamp}.csv", csv_buf.getvalue())
        if analyzer.best_frame_bytes:
            zipf.writestr(f"best_frame_{timestamp}.jpg", analyzer.best_frame_bytes)
        if analyzer.worst_frame_bytes:
            zipf.writestr(f"worst_frame_{timestamp}.jpg", analyzer.worst_frame_bytes)
    zip_buf.seek(0)
    return zip_buf

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(layout="wide", page_title="Freestyle Swim Analyzer Pro v4.0 – Final")
    st.title("Freestyle Swim Technique Analyzer Pro – Final Merged & Fixed")

    with st.sidebar:
        height = st.slider("Height (cm)", 150, 200, 170)
        discipline = st.selectbox("Discipline", ["pool", "triathlon"])

    athlete = AthleteProfile(height, discipline)
    analyzer = SwimAnalyzer(athlete)

    uploaded = st.file_uploader("Upload swimming video", type=["mp4", "mov"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded.getvalue())
            input_path = tmp_in.name

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.mktemp(suffix=".mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

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
                progress.progress(frame_idx / total)
                status.text(f"Frame {frame_idx}/{total}")

            cap.release()
            writer.release()
            os.unlink(input_path)

            summary = analyzer.get_summary()
            plot_buf = generate_plots(analyzer)
            pdf_buf = generate_pdf_report(summary, uploaded.name, plot_buf)
            csv_buf = export_to_csv(analyzer)
            zip_buf = create_results_bundle(out_path, csv_buf, pdf_buf, plot_buf, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

            st.success("Analysis complete!")

            cols = st.columns(4)
            cols[0].metric("Stroke Rate", f"{summary.stroke_rate:.1f} spm")
            cols[1].metric("Breaths/min", f"{summary.breaths_per_min:.1f}")
            cols[2].metric("Avg Body Roll", f"{summary.avg_body_roll:.1f}°")
            cols[3].metric("Kick Depth", f"{summary.avg_kick_depth:.2f}", delta=summary.kick_status)

            col1, col2 = st.columns(2)
            with col1:
                if summary.best_frame_bytes:
                    st.image(summary.best_frame_bytes, caption="Best Pull Frame")
            with col2:
                if summary.worst_frame_bytes:
                    st.image(summary.worst_frame_bytes, caption="Worst Pull Frame")

            st.video(out_path)

            st.download_button("Download Full Analysis (ZIP)", zip_buf, f"swim_analysis.zip", "application/zip")

            os.unlink(out_path)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            if os.path.exists(input_path): os.unlink(input_path)
            if os.path.exists(out_path): os.unlink(out_path)

if __name__ == "__main__":
    main()
