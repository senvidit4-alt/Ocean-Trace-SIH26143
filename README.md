# 🌊 OceanTrace (Ocean-Trace-v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4+-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![OpenDrift](https://img.shields.io/badge/OpenDrift-1.14+-007ACC.svg)](https://opendrift.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Autonomous Satellite SAR Oil Spill Detection, Hydrodynamic Backward Drift Reconstruction & AIS Vessel Attribution Pipeline**

---

## 🧭 About the Project (Ye Project Kya Hai?)

**OceanTrace** is a full-stack forensic intelligence platform designed to detect marine oil spills from satellite imagery, trace their origin backwards through ocean currents and winds, and identify the specific vessels responsible using AIS (Automatic Identification System) maritime telemetry.

When vessels illegally dump oil (bilge dumping) or suffer accidents in the open ocean, the slick drifts away due to ocean currents and surface winds. By the time satellites detect the spill, the polluting ship is often hundreds of miles away. 

**OceanTrace solves this attribution challenge through an automated 3-stage physics-informed AI pipeline.**

```
   🛰️ Sentinel-1 SAR Image
              │
              ▼
┌────────────────────────────────────────┐
│  MODULE 1: SAR Oil Spill Detection     │
│  • ResNet-34 U-Net (PyTorch)           │  ──► detection.json
│  • Calibrated backscatter segmentation │      (Polygon GeoJSON, Area km², Confidence)
│  • False-positive / look-alike filter  │
└────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────┐
│  MODULE 2: Backward Drift Simulation   │
│  • OpenDrift / CMEMS Current + ERA5    │  ──► reconstruction.json & reconstruction.nc
│  • Lagrangian reverse particle physics │      (Spill Origin, Release Window, Uncertainty)
│  • 68% & 95% Confidence Ellipses       │
└────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────┐
│  MODULE 3: AIS Vessel Attribution      │
│  • Spatio-Temporal KD-Tree Matcher     │  ──► attribution.json
│  • 5-Factor Transparent Evidence Score │      (Ranked Vessel Dossiers, Proximity,
│  • Speed & Course Anomaly Analysis     │       Temporal Alignment, Investigation Report)
└────────────────────────────────────────┘
```

---

## 🎯 Target Users (Kiske Liye Hai?)

1. **Coast Guards & Maritime Surveillance Authorities (ICG / USCG / EMSA)**:
   - Rapidly detect offshore pollution events and generate court-admissible forensic evidence dossiers.
2. **Environmental Protection Agencies (EPA / Pollution Control Boards)**:
   - Hold polluters accountable for illegal bilge dumps and environmental damage.
3. **Port & Harbor Authorities**:
   - Monitor near-shore anchorage zones and verify vessel compliance.
4. **Maritime & Satellite Oceanography Researchers**:
   - Modular, reproducible architecture combining Earth Observation (EO) data and Lagrangian physics modeling.

---

## ✨ Key Features (Highlights)

- 🛰️ **Deep Learning SAR Segmentation**: Custom ResNet-34 U-Net trained on Sentinel-1 SAR imagery with focal Tversky loss to isolate true oil slicks from look-alikes (algal blooms, low wind zones).
- 🔄 **Physics-Informed Reverse Drift**: Reverse Lagrangian particle simulation driven by hydrodynamic current velocities and wind drag.
- 🚢 **Multi-Factor Vessel Attribution**: Evaluates spatial distance, temporal overlap, trajectory heading, drift vector consistency, and operational speed anomalies.
- 💻 **Interactive Full-Stack Web Console**: Modern geospatial dashboard with interactive Leaflet map, slick overlay, particle origin cloud, vessel trajectories, and PDF/printable forensic reports.
- 🔌 **Standardized REST API**: Complete FastAPI service with CORS, file upload validation, and OpenAPI documentation (`/docs`).

---

## 🚀 Foolproof Setup & Run Guide (Teammates Clone Guide)

Follow these exact steps to run the complete project on **Windows**, **macOS**, or **Linux** without any missing dependencies.

### 📋 Prerequisites
- **Git** installed ([Download Git](https://git-scm.com/))
- **Python 3.10+** ([Download Python](https://www.python.org/))
- **Node.js 18+** & **npm** ([Download Node.js](https://nodejs.org/))

---

### Step 1: Clone Repository
```bash
git clone https://github.com/senvidit4-alt/Ocean-Trace-v2.0.git
cd Ocean-Trace-v2.0
```

---

### Step 2: Backend Setup (FastAPI & AI Pipeline)

Open a terminal inside the project root directory:

#### On Windows (PowerShell):
```powershell
# 1. Create Python virtual environment
python -m venv venv

# 2. Allow script execution (if PowerShell restricts scripts)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install all Python dependencies
pip install -r requirements.txt

# 5. Start the backend server
python main.py
```

#### On Linux / macOS:
```bash
# 1. Create Python virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install all Python dependencies
pip install -r requirements.txt

# 4. Start the backend server
python3 main.py
```

> ✅ **Backend Server live at:** `http://localhost:8000`  
> 📖 **Interactive Swagger Docs:** `http://localhost:8000/docs`  
> 🩺 **Health Check Endpoint:** `http://localhost:8000/api/health`

---

### Step 3: Frontend Setup (Interactive Web Console)

Open a **NEW terminal window**, navigate to `ocean-trace-frontend`:

```bash
# 1. Enter the frontend directory
cd ocean-trace-frontend

# 2. Install frontend dependencies
npm install

# 3. Launch Vite development server
npm run dev
```

> 🖥️ **Ocean Trace Web Console live at:** `http://localhost:5173` (Open in your browser!)

---

## 🧪 Automated Verification & Testing

Verify that all endpoints, model weights, and simulation pipelines function properly on your system with one command:

```bash
# Run the complete test suite (from repo root with venv activated)
python test_backend_api.py
```

Expected output:
```
============================================================
ALL API ENDPOINT INTEGRATION TESTS PASSED SUCCESSFULLY!
============================================================
```

To run the pipeline end-to-end via CLI on sample data:
```bash
python run_pipeline.py --input-synthetic --output-dir outputs
```

---

## 📁 Repository Structure

```
Ocean-Trace-v2.0/
├── backend/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST API endpoints (/detect, /trace, /attribute)
├── ocean-trace-frontend/           # Interactive Web UI Console
│   ├── index.html                  # Main UI layout & geospatial map
│   ├── src/styles.css              # Glassmorphism & dark-mode styling
│   ├── vite.config.js              # Vite server & proxy configuration
│   ├── .env.example                # Frontend environment variable template
│   └── package.json
├── modules/
│   ├── 01_detection/               # SAR AI model inference, train & validation scripts
│   ├── 02_drift/                   # Hydrodynamic environmental forcing & OpenDrift simulation
│   └── 03_attribution/             # AIS ingestion & multi-factor evidence ranking
├── data/
│   ├── samples/                    # Sample AIS CSV & synthetic forcing NetCDF
│   └── AIS_178834011589976755_1814-1788340116592.csv # Gulf of Mexico AIS telemetry dataset
├── contracts/                      # JSON schemas defining inter-module handoffs
├── docs/                           # Architecture specifications & API documentation
├── unet_spill_best.pth             # Pre-trained ResNet-34 U-Net weights (included!)
├── run_pipeline.py                 # Full CLI pipeline runner
├── test_backend_api.py             # Integration test suite for backend API
├── requirements.txt                # Production Python dependencies
├── .env.example                    # Backend environment configuration template
└── main.py                         # Root entrypoint (`python main.py`)
```

---

## ⚙️ REST API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` or `/api/health` | Service health status & connected module readiness |
| `POST` | `/detect-spill` | Module 1: Upload SAR TIFF to get slick polygon & area |
| `POST` | `/trace-origin` | Module 2: Run backward drift simulation from detection polygon |
| `POST` | `/attribute-vessel` | Module 3: Match candidate vessels against source trajectory |
| `POST` | `/run-full-pipeline` | Full end-to-end pipeline execution in a single request |

---

## 🛠️ Troubleshooting

- **PowerShell Execution Policy Error on Windows?**
  Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell before running `.\venv\Scripts\Activate.ps1`.
- **Port 8000 or 5173 already in use?**
  You can run the backend on a different port: `uvicorn main:app --port 8080` and update `VITE_API_URL` in `ocean-trace-frontend/.env`.
- **Pretrained Weights Check**:
  `unet_spill_best.pth` is already present in the root directory. The backend automatically detects and loads it on startup.

---

## 👥 Authors & Team
- **Team Ocean Trace** (Smart India Hackathon)
- Repository: [https://github.com/senvidit4-alt/Ocean-Trace-v2.0.git](https://github.com/senvidit4-alt/Ocean-Trace-v2.0.git)
