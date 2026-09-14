# OceanTrace Backend API Documentation

The OceanTrace REST API exposes the full satellite oil spill detection, backward drift reconstruction, and AIS vessel attribution pipeline for frontend integrations and automated workflows.

---

## 🚀 Running the Server

### Option 1: Direct Python Execution
```bash
python main.py
```

### Option 2: Using Uvicorn Directly
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation (Swagger UI) is available at:
- **Swagger Docs:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 🌐 CORS Configuration

CORS middleware is configured with permissive defaults for development (`allow_origins=["*"]`), allowing any frontend (Vite, Next.js, React) to make cross-origin requests directly.

---

## 📡 Endpoints Specification

### 1. Health Check

Checks backend uptime and operational status.

- **Method:** `GET`
- **Path:** `/health` (or `/`)
- **Headers:** None
- **Input:** None

#### Example Request
```bash
curl -X GET http://localhost:8000/health
```

#### Example Response (`200 OK`)
```json
{
  "status": "ok",
  "service": "OceanTrace Forensic API",
  "version": "1.0.0",
  "timestamp": "2026-09-11T11:14:04.631685+00:00"
}
```

---

### 2. Detect Spill (Module 1 Only)

Runs the U-Net segmentation model on a Sentinel-1 SAR GeoTIFF image.

- **Method:** `POST`
- **Path:** `/detect-spill`
- **Content-Type:** `application/json` OR `multipart/form-data`

#### Input Formats

##### A. JSON Body
```json
{
  "image_path": "dataset/real_dataset/images/00002.tif",
  "threshold": 0.70,
  "input_scale": "auto"
}
```

##### B. Multipart Form Upload
- `file`: GeoTIFF file upload (.tif)
- `threshold` (optional, float): Detection confidence threshold (default: 0.70)
- `input_scale` (optional, string): `"linear"`, `"db"`, or `"auto"` (default: `"auto"`)

#### Example Request (JSON)
```bash
curl -X POST http://localhost:8000/detect-spill \
  -H "Content-Type: application/json" \
  -d '{"image_path": "dataset/real_dataset/images/00002.tif", "threshold": 0.70}'
```

#### Example Request (File Upload)
```bash
curl -X POST http://localhost:8000/detect-spill \
  -F "file=@/path/to/sentinel1_scene.tif" \
  -F "threshold=0.70"
```

#### Example Response (`200 OK - Detection Found`)
```json
{
  "status": "success",
  "detection": {
    "contract_version": "1.0",
    "observation_time": "2026-09-11T11:14:04Z",
    "polygon": {
      "type": "Polygon",
      "coordinates": [[[6.02, 55.48], [6.04, 55.49], ...]]
    },
    "area_km2": 10.44,
    "confidence": 0.73,
    "estimated_age_hours": 12.0,
    "georeference_status": "REAL"
  }
}
```

#### Example Response (`200 OK - No Spill Detected`)
```json
{
  "status": "no_detection",
  "message": "No oil spill detected above threshold in the provided scene.",
  "detection": {
    "area_km2": 0.0,
    "confidence": 0.0,
    "polygon": null
  }
}
```

---

### 3. Trace Origin (Module 2 Only)

Runs OpenOil backward Lagrangian drift simulation to reconstruct the probable spill origin location and time window.

- **Method:** `POST`
- **Path:** `/trace-origin`
- **Content-Type:** `application/json`

#### Input Format
```json
{
  "detection": { ... },
  "search_window_hours": 6.0,
  "number_particles": 1000
}
```
*Note: If `"detection"` is omitted, the API will automatically look for the latest `detection.json` on disk.*

#### Example Request
```bash
curl -X POST http://localhost:8000/trace-origin \
  -H "Content-Type: application/json" \
  -d '{
    "detection_path": "detection.json",
    "search_window_hours": 6.0,
    "number_particles": 1000
  }'
```

#### Example Response (`200 OK`)
```json
{
  "status": "success",
  "reconstruction": {
    "contract_version": "1.0",
    "observation_time": "2026-09-11T11:14:04Z",
    "origin_time": "2026-09-11T05:14:04Z",
    "search_window_hours": 6.0,
    "origin_centroid": [6.0109, 55.4859],
    "uncertainty": {
      "semi_major_m": 880.0,
      "semi_minor_m": 520.0,
      "orientation_deg": 6.6,
      "radius_68_m": 1100.0,
      "radius_95_m": 1700.0,
      "method": "covariance_ellipse"
    },
    "n_particles_seeded": 1000,
    "n_particles_valid": 1000
  }
}
```

---

### 4. Attribute Vessel (Module 3 Only)

Evaluates vessel trajectories from NOAA AIS dataset against the reconstructed release window and origin point across 5 forensic evidence dimensions.

- **Method:** `POST`
- **Path:** `/attribute-vessel`
- **Content-Type:** `application/json`

#### Input Format
```json
{
  "reconstruction_path": "reconstruction.json",
  "ais_csv_path": "data/AIS_178834011589976755_1814-1788340116592.csv"
}
```
*(Both fields are optional: defaults to `reconstruction.json` and auto-detects AIS CSV in `data/`)*

#### Example Request
```bash
curl -X POST http://localhost:8000/attribute-vessel \
  -H "Content-Type: application/json" \
  -d '{"reconstruction_path": "reconstruction.json"}'
```

#### Example Response (`200 OK`)
```json
{
  "status": "success",
  "attribution": {
    "contract_version": "1.0",
    "analysis_metadata": {
      "search_radius_km": 3.3,
      "search_time_window": ["2026-09-11T05:14:16Z", "2026-09-11T11:14:16Z"],
      "n_vessels_considered": 396,
      "caveats": []
    },
    "candidates": [
      {
        "mmsi": "367123456",
        "vessel_name": "CARGO_VESSEL_ALPHA",
        "category": "Candidate",
        "score": 0.5842,
        "features": [
          {
            "name": "spatial_proximity",
            "value": 0.82,
            "raw_value": 0.6,
            "raw_unit": "km",
            "weight": 0.30,
            "contribution": 0.246
          }
        ],
        "data_confidence": {
          "coverage_fraction": 0.94,
          "note": "Complete telemetry during incident window."
        }
      }
    ]
  },
  "report": "OceanTrace Incident Report\n========================================\n..."
}
```

---

### 5. Run Full Pipeline (End-to-End)

Runs the entire chain: Module 1 Detection $\to$ Module 2 Drift $\to$ Module 3 Attribution $\to$ Final Forensic Report.

- **Method:** `POST`
- **Path:** `/run-full-pipeline`
- **Content-Type:** `application/json` OR `multipart/form-data`

#### Input Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_path` | string | Yes (or file) | None | Server path to GeoTIFF image |
| `file` | file | Yes (or image_path) | None | Uploaded GeoTIFF file |
| `ais_csv_path` | string | No | Auto in `data/` | Path to NOAA AIS CSV |
| `threshold` | float | No | 0.70 | Detection confidence threshold |
| `search_window_hours` | float | No | 6.0 | Backward drift simulation hours |
| `number_particles` | int | No | 1000 | OpenOil particle count |

#### Example Request (JSON)
```bash
curl -X POST http://localhost:8000/run-full-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "dataset/real_dataset/images/00002.tif",
    "threshold": 0.70,
    "search_window_hours": 6.0
  }'
```

#### Example Request (File Upload via curl)
```bash
curl -X POST http://localhost:8000/run-full-pipeline \
  -F "file=@dataset/real_dataset/images/00002.tif" \
  -F "threshold=0.70"
```

#### Example Response (Detection Found)
```json
{
  "status": "success",
  "detection": {
    "contract_version": "1.0",
    "observation_time": "2026-09-11T11:14:16Z",
    "area_km2": 10.44,
    "confidence": 0.73,
    "polygon": { ... }
  },
  "reconstruction": {
    "origin_centroid": [6.0109, 55.4859],
    "origin_time": "2026-09-11T05:14:16Z",
    "uncertainty": { ... }
  },
  "attribution": {
    "analysis_metadata": { ... },
    "candidates": [ ... ]
  },
  "report": "OceanTrace Incident Report\n========================================\n..."
}
```

#### Example Response (No Detection Found)
```json
{
  "status": "no_detection",
  "message": "No oil spill detected above threshold in the provided scene.",
  "detection": {
    "area_km2": 0.0,
    "confidence": 0.0,
    "polygon": null
  },
  "reconstruction": null,
  "attribution": null,
  "report": null
}
```
