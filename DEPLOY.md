# OceanTrace Deployment Guide

This document outlines the deployment strategy, runtime configurations, hardcoded path audit, and model artifact provisioning for deploying the OceanTrace system to cloud environments.

---

## 🏗️ Architecture Overview

The OceanTrace deployment separates the intensive computational backend from the user-facing web interface:

| Component | Target Platform | Tech Stack | Responsibilities |
|---|---|---|---|
| **Backend API** | **Render** or **Railway** | Python 3.11 + FastAPI + Uvicorn | Satellite SAR detection (PyTorch U-Net), OpenOil Lagrangian drift simulation, NOAA AIS vessel candidate attribution |
| **Frontend Web App** | **Vercel** | React / Next.js / Vite | Interactive geospatial map (deck.gl / Mapbox / Leaflet), incident reports, vessel candidate triage dashboard |

---

## ⚙️ Backend Deployment (Render / Railway)

### 1. Build & Start Commands

- **Build Command:**
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```
- **Start Command (Procfile):**
  ```bash
  uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
  ```
*(Render and Railway dynamically bind the `$PORT` environment variable to incoming HTTP traffic).*

---

### 2. Model Checkpoint Strategy (`unet_spill_best.pth`)

The trained PyTorch U-Net model checkpoint (`unet_spill_best.pth`, ~69.1 MB) is excluded from standard Git tracking (`.gitignore`) to keep the repository lightweight. When deploying to Render or Railway, use one of the following methods to supply the weights:

#### Recommended: Pre-Deploy Download Script (or GitHub Release Asset)
Host `unet_spill_best.pth` as a private/public GitHub Release asset or on cloud storage (AWS S3, Google Cloud Storage, Cloudflare R2).

In Render or Railway, update the **Build Command** to download the weights automatically before starting:
```bash
curl -L -o unet_spill_best.pth "https://storage.googleapis.com/your-bucket/unet_spill_best.pth" && pip install -r requirements.txt
```

#### Alternative: Git LFS (Large File Storage)
1. Install Git LFS: `git lfs install`
2. Track the checkpoint: `git lfs track "*.pth"`
3. Commit `.gitattributes` and push. Railway and Render natively clone Git LFS files if LFS is enabled on the repository.

#### Alternative: Docker Container
If deploying via Docker, use a multi-stage Dockerfile that copies the checkpoint directly into the image:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🔍 Hardcoded Local Paths Audit

The following table documents all static or default paths currently referenced in the codebase. In future refactoring iterations, these can be migrated to read from environment variables (`.env`) without altering pipeline contracts:

| Configuration Item | Current Hardcoded Location | Defined In | Recommended Env Var |
|---|---|---|---|
| **Model Weights** | `"unet_spill_best.pth"` | `run_pipeline.py`, `backend/main.py`, `inference.py` | `MODEL_PATH` |
| **Environmental NetCDF** | `"data/samples/synthetic_env.nc"` | `run_pipeline.py`, `backend/main.py` | `ENV_NETCDF_PATH` |
| **AIS Telemetry Dataset** | `"data/AIS_178834011589976755_1814-1788340116592.csv"` | `run_pipeline.py`, `backend/main.py` | `AIS_DATA_PATH` |
| **Land / Coastline Mask** | `"gulf_coast_land.geojson"` | `modules/01_detection/src/inference.py` | `LAND_MASK_PATH` |
| **Upload Directory** | `"uploads"` | `backend/main.py` | `UPLOAD_DIR` |
| **Server Port** | `8000` | `main.py` | `PORT` |
| **Server Host** | `"0.0.0.0"` | `main.py` | `HOST` |
| **CORS Origins** | `["*"]` | `backend/main.py` | `CORS_ORIGINS` |
| **Handoff Output Contracts** | `"detection.json"`, `"reconstruction.json"`, `"attribution.json"` | `run_pipeline.py`, `backend/main.py` | `OUTPUT_DIR` |

---

## 🌐 Frontend Deployment (Vercel)

When the frontend is ready to deploy to Vercel:

1. **Environment Variables**: Set the backend API URL in Vercel project settings:
   - For Next.js: `NEXT_PUBLIC_API_URL=https://oceantrace-api.onrender.com`
   - For Vite: `VITE_API_URL=https://oceantrace-api.onrender.com`
2. **CORS Alignment**:
   - Ensure the backend's `CORS_ORIGINS` environment variable includes your Vercel deployment URL (e.g. `https://oceantrace.vercel.app`).
3. **File Upload Limits**:
   - For direct SAR TIFF uploads through the frontend, verify client-side upload payload limits, or use multipart chunking if scenes exceed 50 MB.
