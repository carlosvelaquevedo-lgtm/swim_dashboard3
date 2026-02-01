# 🏊 Freestyle Swim Technique Analyzer Pro

AI-powered swimming technique analysis using computer vision and pose estimation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-time Pose Analysis**: Uses MediaPipe Pose for accurate body landmark detection
- **Technique Metrics**: Analyzes torso lean, forearm angle, body roll, and kick mechanics
- **Breathing Detection**: Tracks breathing patterns and bilateral balance
- **Stroke Phase Recognition**: Identifies Entry, Pull, Push, and Recovery phases
- **Visual Comparison**: Side-by-side overlay comparing swimmer to ideal form
- **Comprehensive Reports**: PDF reports, CSV data exports, and annotated video output

## Demo

Upload a freestyle swimming video and get instant analysis:

| Metric | Description |
|--------|-------------|
| Technique Score | Overall score (0-100) based on form |
| Stroke Rate | Strokes per minute |
| Body Roll | Shoulder rotation angle |
| Kick Symmetry | Left/right leg balance |
| Kick Depth | Amplitude of kick motion |

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/swim-analyzer.git
cd swim-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

### Docker Setup

```bash
docker build -t swim-analyzer .
docker run -p 8501:8501 swim-analyzer
```

## Usage

1. **Upload Video**: Select an MP4 or MOV file of freestyle swimming
2. **Configure Settings**: Adjust athlete height, discipline, and detection thresholds
3. **Analyze**: Click process and wait for frame-by-frame analysis
4. **Review Results**: View technique score, metrics, and visual feedback
5. **Export**: Download the full analysis bundle (ZIP) containing:
   - Annotated video
   - PDF report
   - CSV data
   - Analysis charts
   - Best/worst frame snapshots

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Height (cm) | 170 | Athlete's height for scaling |
| Discipline | pool | `pool` or `triathlon` |
| Confidence Threshold | 0.5 | Minimum pose detection confidence |
| Yaw Threshold | 0.15 | Head turn threshold for breath detection |

## Technique Zones

The analyzer uses color-coded zones for feedback:

| Zone | Color | Meaning |
|------|-------|---------|
| Good | 🟢 Green | Optimal range |
| OK | 🟡 Yellow | Acceptable, minor adjustment needed |
| Needs Work | 🔴 Red | Outside optimal range |

### Ideal Ranges

- **Torso Lean**: 4-12° (good), 0-18° (acceptable)
- **Forearm Vertical**: 0-35° (good), 0-60° (acceptable)
- **Body Roll**: 35-55° (good), 25-65° (acceptable)
- **Kick Symmetry**: < 15° difference between legs
- **Kick Depth**: 0.25-0.6 (normalized)

## Project Structure

```
swim-analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── .gitignore            # Git ignore rules
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # This file
└── LICENSE               # MIT License
```

## Technical Details

### Pose Estimation
- Uses MediaPipe Pose with `model_complexity=0` (lite model) for cloud compatibility
- Tracks 13 key landmarks for swimming analysis
- Applies temporal smoothing with 7-frame buffers

### Stroke Detection
- Identifies strokes via elbow angle local minima
- Requires minimum 0.5s gap between detected strokes
- Calculates stroke rate from timing data

### Breathing Analysis
- Detects head rotation using nose-to-shoulder midpoint offset
- Requires 4+ consecutive frames above yaw threshold
- Enforces minimum 1.0s gap between breath counts

## Limitations

- Best results with side-view or slightly angled camera footage
- Single swimmer analysis only
- Requires clear visibility of the swimmer's body
- May struggle with heavy water splash or poor lighting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for video processing
- [ReportLab](https://www.reportlab.com/) for PDF generation

## Support

If you encounter any issues or have questions:
- Open an [Issue](https://github.com/yourusername/swim-analyzer/issues)
- Check existing issues for solutions

---

Made with ❤️ for swimmers and coaches
