import streamlit as st
import cv2
import numpy as np
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List
import tempfile
import pandas as pd
import io
import zipfile
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
import mediapipe as mp

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_TORSO_GOOD = (4, 12)
DEFAULT_TORSO_OK = (0, 18)
DEFAULT_FOREARM_GOOD = (0, 35)
DEFAULT_FOREARM_OK = (0, 60)
DEFAULT_ROLL_GOOD = (35, 55)
DEFAULT_ROLL_OK = (25, 65)
DEFAULT_KICK_DEPTH_GOOD = (0.25, 0.6)

# ──────────────────────────────
# DATA CLASSES
# ──────────────────────────────
@dataclass
class FrameMetrics:
    time_s: float
    elbow_angle: float
    torso_lean: float
    forearm_vertical: float
    kick_depth_proxy: float
    body_roll: float
    phase: str
    confidence: float = 1.0

@dataclass
class AthleteProfile:
    height_cm: float
    discipline: str

# ──────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────
def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
    return np.degrees(np.arccos(np.clip(cosang,-1,1)))

def compute_torso_lean(lm):
    mid_sh = ((lm["LEFT_SHOULDER"][0]+lm["RIGHT_SHOULDER"][0])/2,
              (lm["LEFT_SHOULDER"][1]+lm["RIGHT_SHOULDER"][1])/2)
    mid_hip = ((lm["LEFT_HIP"][0]+lm["RIGHT_HIP"][0])/2,
               (lm["LEFT_HIP"][1]+lm["RIGHT_HIP"][1])/2)
    dy = mid_sh[1]-mid_hip[1]
    dx = mid_sh[0]-mid_hip[0]
    return math.degrees(math.atan2(dy,dx))

def compute_forearm_vertical(lm):
    dx = lm["LEFT_WRIST"][0]-lm["LEFT_ELBOW"][0]
    dy = lm["LEFT_WRIST"][1]-lm["LEFT_ELBOW"][1]
    return abs(math.degrees(math.atan2(dx,-dy)))

def compute_kick_depth_proxy(lm):
    hip_y = (lm["LEFT_HIP"][1]+lm["RIGHT_HIP"][1])/2
    knee_l_y = lm["LEFT_KNEE"][1]
    knee_r_y = lm["RIGHT_KNEE"][1]
    sh_dist = abs(lm["LEFT_SHOULDER"][1]-lm["LEFT_HIP"][1]) or 1.0
    return ((abs(knee_l_y-hip_y)+abs(knee_r_y-hip_y))/2)/sh_dist

def get_zone_color(val, good, ok):
    if good[0]<=val<=good[1]: return (0,180,0)
    if ok[0]<=val<=ok[1]: return (0,220,220)
    return (0,0,220)

def draw_technique_panel(frame, x_origin, title, torso, forearm, roll, kick_depth):
    px, py = x_origin-140, 30
    pw, ph = 280, 440
    overlay = frame.copy()
    cv2.rectangle(overlay,(px,py),(px+pw,py+ph),(0,0,0),-1)
    cv2.addWeighted(overlay,0.65,frame,0.35,0,frame)
    cv2.putText(frame,title,(px+10,py+30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    tc = get_zone_color(abs(torso), DEFAULT_TORSO_GOOD, DEFAULT_TORSO_OK)
    cv2.putText(frame,f"Torso Lean: {torso:.1f}°",(px+10,py+70),cv2.FONT_HERSHEY_SIMPLEX,0.7,tc,2)
    fc = get_zone_color(forearm, DEFAULT_FOREARM_GOOD, DEFAULT_FOREARM_OK)
    cv2.putText(frame,f"Forearm Vert: {forearm:.1f}°",(px+10,py+110),cv2.FONT_HERSHEY_SIMPLEX,0.7,fc,2)
    rc = get_zone_color(roll, DEFAULT_ROLL_GOOD, DEFAULT_ROLL_OK)
    cv2.putText(frame,f"Body Roll: {roll:.1f}°",(px+10,py+150),cv2.FONT_HERSHEY_SIMPLEX,0.7,rc,2)
    kdc = get_zone_color(kick_depth*100, DEFAULT_KICK_DEPTH_GOOD,(0.1,0.8))
    cv2.putText(frame,f"Kick Depth: {kick_depth:.2f}",(px+10,py+190),cv2.FONT_HERSHEY_SIMPLEX,0.7,kdc,2)

# ──────────────────────────────
# ANALYZER CLASS
# ──────────────────────────────
class SwimAnalyzerClassic:
    def __init__(self, athlete: AthleteProfile):
        self.athlete = athlete
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=DEFAULT_CONF_THRESHOLD,
                                      min_tracking_confidence=DEFAULT_CONF_THRESHOLD)
        self.metrics: List[FrameMetrics] = []

    def analyze_frame(self, frame, time_s):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        lm = {}
        h, w, _ = frame.shape
        for id, lm_data in enumerate(results.pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(id).name
            lm[name] = (int(lm_data.x*w), int(lm_data.y*h))
        elbow_angle = calculate_angle(lm["LEFT_SHOULDER"], lm["LEFT_ELBOW"], lm["LEFT_WRIST"])
        torso = compute_torso_lean(lm)
        forearm = compute_forearm_vertical(lm)
        kick_depth = compute_kick_depth_proxy(lm)
        body_roll = 45
        phase = "Pull" if elbow_angle<150 else "Recovery"
        fm = FrameMetrics(time_s, elbow_angle, torso, forearm, kick_depth, body_roll, phase)
        self.metrics.append(fm)
        draw_technique_panel(frame, frame.shape[1]-200,"Stroke", torso, forearm, body_roll, kick_depth)
        draw_technique_panel(frame, 200,"IDEAL", 8, 35, 45, 0.4)
        return fm

    def get_dataframe(self):
        return pd.DataFrame([vars(fm) for fm in self.metrics])

# ──────────────────────────────
# PLOTS & PDF
# ──────────────────────────────
def generate_plots(analyzer):
    df = analyzer.get_dataframe()
    if df.empty: return io.BytesIO()
    plt.style.use("dark_background")
    fig, axs = plt.subplots(2,1,figsize=(10,6))
    axs[0].plot(df.time_s, df.elbow_angle,label="Elbow Angle",color="orange")
    axs[0].set_title("Elbow Angle Over Time"); axs[0].legend()
    axs[1].plot(df.time_s, df.kick_depth_proxy,label="Kick Depth",color="cyan")
    axs[1].set_title("Kick Depth Over Time"); axs[1].legend()
    buf = io.BytesIO()
    plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig)
    buf.seek(0)
    return buf

def generate_pdf(analyzer, video_name, plot_buf):
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer,pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Swim Analysis Report - {video_name}",styles['Title']))
    story.append(Spacer(1,0.2*inch))
    df = analyzer.get_dataframe()
    if not df.empty:
        table_data = [df.columns.tolist()] + df.values.tolist()
        story.append(Table(table_data, style=TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)])))
    if plot_buf.getvalue():
        story.append(PageBreak())
        story.append(Paragraph("Plots",styles['Heading2']))
        img = RLImage(plot_buf)
        img.drawWidth = 6*inch
        img.drawHeight = 4*inch
        story.append(img)
    pdf.build(story)
    buffer.seek(0)
    return buffer

# ──────────────────────────────
# STREAMLIT APP
# ──────────────────────────────
st.set_page_config(layout="wide", page_title="SwimAnalyzer Live ZIP")
st.title("🏊 SwimAnalyzer Classic Pose – Live + ZIP Export")

st.sidebar.header("Athlete Profile")
height_cm = st.sidebar.number_input("Height (cm)", 140, 220, 175)
discipline = st.sidebar.selectbox("Discipline", ["triathlon","swimming"])
athlete = AthleteProfile(height_cm, discipline)

uploaded_file = st.file_uploader("Upload swimming video", type=["mp4","mov","avi"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.flush()
    cap = cv2.VideoCapture(tfile.name)
    analyzer = SwimAnalyzerClassic(athlete)
    stframe = st.empty()
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    progress = st.progress(0)
    frame_dir = tempfile.TemporaryDirectory()

    zip_buffer = io.BytesIO()
    zipf = zipfile.ZipFile(zip_buffer, 'w')

    while True:
        ret, frame = cap.read()
        if not ret: break
        time_s = frame_idx/fps
        analyzer.analyze_frame(frame, time_s)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")
        frame_path = os.path.join(frame_dir.name,f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        zipf.write(frame_path, f"frames/frame_{frame_idx:04d}.png")
        frame_idx += 1
        progress.progress(frame_idx/ int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()

    # CSV + PDF into ZIP
    df = analyzer.get_dataframe()
    csv_bytes = df.to_csv(index=False).encode()
    zipf.writestr("swim_analysis.csv", csv_bytes)
    plot_buf = generate_plots(analyzer)
    pdf_buf = generate_pdf(analyzer, uploaded_file.name, plot_buf)
    zipf.writestr("swim_report.pdf", pdf_buf.getvalue())
    zipf.close()
    zip_buffer.seek(0)

    st.subheader("Analysis Data")
    st.dataframe(df)
    st.image(plot_buf, caption="Plots")
    st.download_button("Download Full ZIP Bundle", zip_buffer.getvalue(),
                       "swim_analysis_live.zip","application/zip")
