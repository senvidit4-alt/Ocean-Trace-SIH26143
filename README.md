# 🌊 OceanTrace (Ocean-Trace-v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4+-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![OpenDrift](https://img.shields.io/badge/OpenDrift-1.14+-007ACC.svg)](https://opendrift.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Autonomous Satellite SAR Oil Spill Detection, Hydrodynamic Backward Drift Reconstruction & AIS Vessel Attribution Pipeline**

---

## 🧭 About the Project

**OceanTrace** is a full-stack forensic intelligence platform that detects marine oil spills from satellite radar imagery, reconstructs backward through ocean currents and winds to locate their point of origin, and identifies the specific vessels responsible using real AIS (Automatic Identification System) maritime telemetry — turning raw satellite data into a court-ready attribution report.

**The problem it solves:** When a vessel illegally discharges oil (bilge dumping) or suffers an accident at sea, the slick drifts with currents and wind. By the time a satellite captures the spill, the responsible ship may be hundreds of kilometers away and long gone. Manual investigation is slow, low-resolution, and rarely converges on a specific vessel with defensible evidence.

OceanTrace automates this entire chain — detection → origin reconstruction → attribution — through a physics-informed, evidence-based 3-stage pipeline, producing a transparent, auditable report rather than an opaque black-box verdict.

```
            🛰️ Sentinel-1 SAR Image
                      │
                      ▼
┌──────────────────────────────────────────┐
│ MODULE 1 — SAR Oil Spill Detection       │
│ • ResNet-34 U-Net (PyTorch, pretrained)  │ ──► detection.json
│ • dB-calibrated backscatter segmentation │     (Polygon GeoJSON, Area km², Confidence)
│ • Coastal & look-alike false-positive    │
│   filtering (calm water, algae, wetlands)│
└──────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│ MODULE 2 — Backward Drift Reconstruction │
│ • OpenDrift Lagrangian particle engine   │ ──► reconstruction.json / .nc
│ • Real CMEMS ocean current + ERA5 wind   │     (Spill Origin, Release Window,
│ • Reverse-time trajectory simulation     │      Uncertainty Ellipses)
└──────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│ MODULE 3 — AIS Vessel Attribution        │
│ • Spatio-temporal vessel matching        │ ──► attribution.json
│ • 5-factor transparent evidence scoring  │     (Ranked Vessel Dossiers,
│   (Spatial, Temporal, Trajectory,        │      Proximity, Investigation Report)
│   Drift-consistency, Speed/Course)       │
└──────────────────────────────────────────┘
                      │
                      ▼
 📄 Forensic PDF Report (SHA-256 sealed, QR-verifiable) + 🗄️ PostgreSQL Incident & Vessel Registry
```

---

## 🎯 Target Users

| User | Use Case |
|---|---|
| **Coast Guards & Maritime Surveillance Authorities (ICG / USCG / EMSA)** | Rapidly detect offshore pollution events and generate forensic-grade evidence dossiers for enforcement action. |
| **Environmental Protection Agencies / Pollution Control Boards** | Hold polluters accountable for illegal bilge dumps and quantify environmental damage. |
| **Port & Harbor Authorities** | Monitor near-shore anchorage zones and verify vessel environmental compliance. |
| **Maritime & Satellite Oceanography Researchers** | Reproducible, modular architecture combining Earth Observation data with Lagrangian ocean-physics modeling. |

---

## ✨ Key Features

- 🛰️ **Deep Learning SAR Segmentation** — A ResNet-34 U-Net (transfer-learned from ImageNet, fine-tuned on real Sentinel-1 SAR imagery) trained with a custom Hybrid Focal Tversky loss to handle extreme class imbalance and isolate true oil slicks from look-alikes (calm-water zones, algal blooms, coastal wetlands).
- 🔄 **Physics-Informed Reverse Drift** — Backward Lagrangian particle simulation driven by real hydrodynamic current velocity and wind-drag fields (CMEMS / ECMWF), estimating spill origin location and release time window with confidence ellipses.
- 🚢 **Transparent Multi-Factor Vessel Attribution** — Evaluates each candidate vessel across five independently-weighted, explainable signals (spatial proximity, temporal alignment, trajectory consistency, drift-path match, speed/course anomaly) instead of an opaque black-box score — every attribution is auditable and defensible.
- 💻 **Interactive Full-Stack Web Console** — A modern geospatial dashboard with an interactive map, slick overlay visualization, drift particle rendering, ranked vessel trajectories, live analytics, and printable/exportable forensic reports.
- 🔐 **Court-Ready Forensic PDF Export** — Client-side generated incident report with a SHA-256 tamper-evidence hash and a QR-verifiable case seal — no extra backend infrastructure required.
- 🗄️ **Persistent Incident Database** — Every detection and attribution report is stored in PostgreSQL, with automatic vessel-ownership enrichment (since raw AIS data has no ownership field) via a maritime registry lookup.
- 🔌 **Standardized REST API** — Complete FastAPI service with CORS support, dual JSON/file-upload input, and interactive OpenAPI documentation (`/docs`).

---

## 🏗️ Architecture & Design Principles

- **Modular, contract-driven pipeline** — Each module (Detection, Drift, Attribution) is independently developed, tested, and owned, with a formally versioned JSON schema (`contracts/CONTRACTS.md`) defining the handoff between them. This means any module can be improved or swapped without breaking the others.
- **Real data, not synthetic** — Trained on a published, peer-reviewed Sentinel-1 SAR benchmark dataset; validated against a real post-Hurricane-Ida Gulf of Mexico scene; attribution tested against real NOAA MarineCadastre AIS records (264,646 points, 396 vessels).
- **Explainability over black-box confidence** — Vessel attribution uses a transparent, weighted evidence-fusion score rather than an opaque ML classifier, so every result can be justified signal-by-signal — critical for any real investigative or legal use.
- **Honest engineering** — False-positive suppression (5 km coastal exclusion buffer + minimum polygon area filtering) was validated by reducing a real scene from 69,176 km² of false positives down to a confirmed, honest zero open-water detections.

---

## 🚀 Setup & Run Guide

Follow these exact steps to run the complete project on Windows, macOS, or Linux.

### 📋 Prerequisites
- **Git** ([Download](https://git-scm.com/downloads))
- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+ & npm** ([Download](https://nodejs.org/))
- **PostgreSQL** (optional — required only for persistent incident storage)

---

### Step 1 — Clone the Repository
```bash
git clone https://github.com/senvidit4-alt/Ocean-Trace-v2.0.git
cd Ocean-Trace-v2.0
```

---

### Step 2 — Backend Setup (FastAPI & AI Pipeline)

**Windows (PowerShell):**
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

**Linux / macOS:**
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

- ✅ **Backend live at:** http://localhost:8000
- 📖 **Interactive Swagger Docs:** http://localhost:8000/docs
- 🩺 **Health Check:** http://localhost:8000/api/health

---

### Step 3 — Frontend Setup (Interactive Web Console)

Open a new terminal window:
```bash
cd ocean-trace-frontend
npm install
npm run dev
```

- 🖥️ **Web Console live at:** http://localhost:5173

---

### Step 4 — (Optional) Database Setup

To enable persistent incident storage:
```bash
cp .env.example .env
# Fill in your PostgreSQL credentials in .env
python report_db.py
# Initializes the schema
```

---

## 🧪 Automated Verification & Testing

Run the full integration test suite (from repo root, with venv activated):
```bash
python test_backend_api.py
```

Expected output:
```
============================================================
ALL API ENDPOINT INTEGRATION TESTS PASSED SUCCESSFULLY!
============================================================
```

Run the full pipeline end-to-end via CLI on sample data:
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
│   ├── 01_detection/               # SAR AI model inference, training & validation scripts
│   ├── 02_drift/                   # Hydrodynamic environmental forcing & OpenDrift simulation
│   └── 03_attribution/             # AIS ingestion & multi-factor evidence ranking
├── data/
│   ├── samples/                    # Sample AIS CSV & synthetic forcing NetCDF
│   └── AIS_*.csv                   # Real Gulf of Mexico AIS telemetry dataset
├── contracts/                      # JSON schemas defining inter-module handoffs
├── docs/                           # Architecture specifications & API documentation
├── unet_spill_best.pth             # Pre-trained ResNet-34 U-Net weights (included)
├── report_db.py                    # PostgreSQL incident/vessel storage interface
├── run_pipeline.py                 # Full CLI pipeline runner
├── test_backend_api.py             # Integration test suite for backend API
├── requirements.txt                # Production Python dependencies
├── .env.example                    # Backend environment configuration template
└── main.py                         # Root entrypoint (`python main.py`)
```

---

## ⚙️ REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` or `/api/health` | Service health status & connected module readiness |
| `POST` | `/detect-spill` | Module 1 — Upload SAR TIFF, get slick polygon, area & confidence |
| `POST` | `/trace-origin` | Module 2 — Run backward drift simulation from a detection polygon |
| `POST` | `/attribute-vessel` | Module 3 — Match candidate vessels against the source trajectory |
| `POST` | `/run-full-pipeline` | Full end-to-end pipeline execution in a single request |

*Full interactive API reference available at `/docs` once the backend is running.*

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Detection Model** | PyTorch, segmentation_models_pytorch (ResNet-34 U-Net), rasterio, OpenCV |
| **Drift Simulation** | OpenDrift, NetCDF, CMEMS / ECMWF ocean-current & wind data |
| **Vessel Attribution** | pandas, GeoPandas, Shapely, NOAA MarineCadastre AIS data |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Database** | PostgreSQL, psycopg2, JSONB |
| **Frontend** | Vite, HTML/CSS/JavaScript, Leaflet-style map rendering |
| **Forensic Export** | html2pdf.js, Web Crypto API (SHA-256), qrcode.js |
| **Deployment** | Render (backend), Vercel (frontend) |

---

## 🛠️ Troubleshooting

- **PowerShell execution policy error (Windows)** — Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating the virtual environment.
- **Port 8000 or 5173 already in use** — Run the backend on a different port: `uvicorn main:app --port 8080`, and update `VITE_API_URL` in `ocean-trace-frontend/.env` accordingly.
- **Pretrained weights check** — `unet_spill_best.pth` is already included in the repository root; the backend automatically detects and loads it on startup.
- **Database connection errors** — Ensure PostgreSQL is running locally and credentials in `.env` match your local setup; the app runs fully without a database, just without persistent storage.

---

## 🔮 Future Scope

- Live global vessel traffic integration via real-time AIS streaming (e.g. AISstream.io) for continuous maritime monitoring beyond historical incident review.
- Extension of the SAR look-alike classifier to cover additional edge cases such as iceberg-related backscatter dampening in polar regions.
- Deployment against real-time Indian Ocean satellite coverage as a live regional monitoring roadmap.

---

## 👥 Team

**Team Ocean Trace** — Smart India Hackathon 2026  
**Repository:** [github.com/senvidit4-alt/Ocean-Trace-v2.0](https://github.com/senvidit4-alt/Ocean-Trace-v2.0)

---

## 📄 License

This project is licensed under the MIT License.
