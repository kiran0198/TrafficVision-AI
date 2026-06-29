# 🚦 TrafficVision-AI

A Deep Learning based Traffic Near-Miss Analytics System developed using **YOLO**, **ByteTrack**, **OpenCV**, and **PyTorch**.

The system performs real-time vehicle detection, tracking, speed estimation, interaction analysis, and collision risk assessment from traffic surveillance videos.

---

## Features

- Vehicle Detection (YOLO)
- Multi-Object Tracking (ByteTrack)
- Vehicle Speed Estimation
- Pixel-to-Meter Scaling
- Near-Miss Detection
- Collision Risk Analysis
- Real-Time Analytics Dashboard
- Live Conflict Waveform
- Vehicle Speed Logging
- GPU (CUDA) Support

---

## Demo

### Input

Traffic surveillance video

↓

### Output

Annotated traffic video with:

- Vehicle IDs
- Speed (km/h)
- Collision Risk
- Interaction Lines
- Dashboard
- Live Waveform

↓

Vehicle speed log

```
Vehicle Speeds

ID: 2  Speed: 34.8 km/h
ID: 7  Speed: 49.5 km/h
ID: 11 Speed: 26.7 km/h
```

---

# Folder Structure

```
TrafficVision-AI/
├── models/
├── videos/
├── output/
├── screenshots/
├── docs/
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone https://github.com/yourusername/TrafficVision-AI.git

cd TrafficVision-AI

pip install -r requirements.txt
```

---

# Run

```bash
python main.py
```

---

# Technologies

- Python
- PyTorch
- YOLO
- ByteTrack
- OpenCV
- CUDA

---

# Outputs

- Annotated Video
- Vehicle Speed Log
- Traffic Dashboard
- Collision Risk Analysis
- Live Interaction Waveform

---

# Future Improvements

- Perspective Transformation
- Homography-based Speed Estimation
- Automatic Incident Detection
- Vehicle Counting
- Lane Detection
- Traffic Density Estimation
- Heatmap Visualization
- Export CSV Analytics

---

# Credits

## Developer

**Kiran Kumar Sahu**

Designed and implemented the complete software architecture, source code, analytics pipeline, visualization, and documentation.

---

## Sample Video

Traffic footage courtesy of **Pexels**.

https://www.pexels.com/video/aerial-view-of-urban-traffic-flow-30862941/

Used for research and demonstration purposes.

---

## License

MIT License
