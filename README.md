# OceanTrace (Ocean-Trace-v2.0)

> **Autonomous Satellite Oil Spill Detection, Backward Drift Reconstruction & AIS Vessel Attribution Pipeline**

OceanTrace is an end-to-end intelligence system for marine pollution monitoring and maritime vessel attribution. It combines Sentinel-1 SAR satellite deep learning (U-Net), backward Lagrangian hydrodynamic drift modeling (OpenDrift/CMEMS/ERA5), and spatio-temporal AIS vessel telemetry correlation.

---

## 🚀 Quick Start Guide (For Teammates)

Get the entire full-stack system running locally in under 3 minutes.

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm**
- **Git**

---

### 2. Clone Repository
```bash
git clone https://github.com/senvidit4-alt/Ocean-Trace-v2.0.git
cd Ocean-Trace-v2.0
```

---

### 3. Backend Setup (FastAPI + AI Pipeline)

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 2. Install all Python dependencies
pip install -r requirements.txt

# 3. (Optional) Copy environment template if custom configs are needed
cp .env.example .env

# 4. Start the backend server
python main.py
```
> 🛰️ Backend runs at: **`http://localhost:8000`**  
> 📖 Interactive Swagger API Docs: **`http://localhost:8000/docs`**  
> 🩺 Health Check: **`http://localhost:8000/api/health`**

---

### 4. Frontend Setup (Interactive Ocean Trace Console)

In a new terminal window:

```bash
# 1. Navigate to the frontend directory
cd ocean-trace-frontend

# 2. Install frontend dependencies
npm install

# 3. Start the Vite dev server
npm run dev
```
> 🖥️ Web Console live at: **`http://localhost:5173`** (or port indicated in terminal)

---

## 🏗️ System Architecture & Data Flow

The system operates across three decoupled modules communicating via explicit JSON contracts:

```
[ Sentinel-1 SAR Image ]
          │
          ▼
┌────────────────────────────────────────┐
│  MODULE 1: SAR Oil Spill Detection     │  ──► detection.json
│  • ResNet-34 U-Net (PyTorch)           │      (Polygons, area, confidence)
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  MODULE 2: Backward Drift Simulation   │  ──► reconstruction.json & .nc
│  • OpenDrift / CMEMS Current + ERA5    │      (Origin area, release time, radius)
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  MODULE 3: AIS Vessel Attribution      │  ──► attribution.json
│  • Spatio-Temporal KD-Tree Matcher     │      (Ranked vessel candidates,
│  • Transparent 5-Factor Evidence Score │       proximity, speed anomalies)
└────────────────────────────────────────┘
```

---

## 📦 Directory Structure

```
Ocean-Trace-v2.0/
├── backend/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST API endpoints
├── ocean-trace-frontend/           # Interactive Web UI Console
│   ├── index.html                  # Dashboard & geospatial visualization
│   ├── src/styles.css              # Custom styling
│   ├── vite.config.js              # Vite server & proxy configuration
│   └── package.json
├── modules/
│   ├── 01_detection/               # SAR AI model inference, train & validation
│   ├── 02_drift/                   # Hydrodynamic environmental forcing & OpenDrift
│   └── 03_attribution/             # AIS ingestion & multi-factor evidence ranking
├── data/
│   ├── samples/                    # Sample AIS CSV & synthetic forcing NetCDF
│   └── AIS_178834011589976755_1814-1788340116592.csv # Gulf of Mexico AIS telemetry
├── contracts/                      # JSON schemas for inter-module handoffs
├── docs/                           # Architecture specifications & validation guides
├── unet_spill_best.pth             # Trained deep learning model checkpoint
├── run_pipeline.py                 # Full CLI pipeline runner
├── test_backend_api.py             # Integration test suite for backend API
├── requirements.txt                # Unified Python dependencies
├── .env.example                    # Environment variable template
└── main.py                         # Root entrypoint (`python main.py`)
```

---

## 🧪 Testing & Verification

Run the automated backend test suite to verify all endpoints, AI inference, and simulation handoffs:

```bash
python test_backend_api.py
```

Run the complete pipeline end-to-end via CLI:

```bash
python run_pipeline.py --input-synthetic --output-dir outputs
```

---

## 👥 Team
- **Team Ocean Trace (SIH)**
