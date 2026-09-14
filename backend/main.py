"""
backend/main.py
===============
FastAPI REST API backend wrapping OceanTrace end-to-end forensic pipeline:
  - Module 1: Oil Spill Detection (POST /detect-spill)
  - Module 2: Backward Drift Source Reconstruction (POST /trace-origin)
  - Module 3: AIS Ingestion & Vessel Attribution (POST /attribute-vessel)
  - Full Pipeline: Orchestrated execution (POST /run-full-pipeline)
  - Health & Diagnostics: (GET /health, GET /)
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Setup repository paths
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "contracts"))
sys.path.insert(0, str(_repo_root / "modules/01_detection/src"))
sys.path.insert(0, str(_repo_root / "modules/02_drift/02_environment/src"))
sys.path.insert(0, str(_repo_root / "modules/02_drift/03_source_reconstruction/src"))
sys.path.insert(0, str(_repo_root / "modules/03_attribution/04_ais_trajectory/src"))
sys.path.insert(0, str(_repo_root / "modules/03_attribution/05_evidence_fusion/src"))

# Helper to find AIS CSV without importing heavy modules
def find_ais_csv(data_dir: Path) -> Optional[Path]:
    for cand in [
        data_dir / "AIS_178834011589976755_1814-1788340116592.csv",
        data_dir / "samples" / "sample_ais.csv",
    ]:
        if cand.exists():
            return cand
    return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend_api")

UPLOAD_DIR = _repo_root / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI application
app = FastAPI(
    title="OceanTrace Forensic Backend API",
    description="End-to-end satellite oil spill detection, drift reconstruction, and AIS vessel attribution API.",
    version="1.0.0",
)

# Step 5: CORS Middleware setup (allow all origins for hackathon development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup preloading & in-memory caching (Performance fix: avoids disk reloads)
CACHED_SPILL_MODEL = None
CACHED_AIS_DATASET = None
CACHED_AIS_PATH = None

def get_cached_model():
    global CACHED_SPILL_MODEL
    if CACHED_SPILL_MODEL is None:
        ckpt = _repo_root / "unet_spill_best.pth"
        if ckpt.exists():
            logger.info("Pre-loading spill model checkpoint into memory...")
            CACHED_SPILL_MODEL = load_spill_model(str(ckpt), device="cpu")
    return CACHED_SPILL_MODEL

def get_cached_ais_dataset(path: Optional[Path] = None):
    global CACHED_AIS_DATASET, CACHED_AIS_PATH
    target_path = path or find_ais_csv(_repo_root / "data")
    if target_path and Path(target_path).exists():
        target_str = str(Path(target_path).resolve())
        if CACHED_AIS_DATASET is None or CACHED_AIS_PATH != target_str:
            logger.info(f"Pre-loading AIS dataset into memory from: {target_str}")
            CACHED_AIS_DATASET = AISDataset.load(target_str)
            CACHED_AIS_PATH = target_str
    return CACHED_AIS_DATASET

@app.on_event("startup")
def startup_preload_cache():
    logger.info("OceanTrace FastAPI initialized. Ready to receive traffic. Models and AIS datasets configured for lazy on-demand caching.")
    app.state.spill_model = None
    app.state.ais_dataset = None


# Helper serialization utilities
def reconstruction_to_dict(recon: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Converts a SourceReconstruction dataclass into a JSON-serializable dictionary."""
    if recon is None:
        return None
    d = recon.to_dict() if hasattr(recon, "to_dict") else dict(recon)
    d["contract_version"] = "1.0"
    if "origin_positions" in d and isinstance(d["origin_positions"], dict):
        if hasattr(d["origin_positions"].get("lon"), "tolist"):
            d["origin_positions"]["lon"] = d["origin_positions"]["lon"].tolist()
        if hasattr(d["origin_positions"].get("lat"), "tolist"):
            d["origin_positions"]["lat"] = d["origin_positions"]["lat"].tolist()
    return d


def attribution_to_dict(result: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Converts an AttributionResult dataclass into a JSON-serializable dictionary."""
    if result is None:
        return None
    return {
        "contract_version": "1.0",
        "analysis_metadata": {
            "search_radius_km": result.search_radius_km,
            "search_time_window": [
                result.search_time_window[0].isoformat() + "Z" if not result.search_time_window[0].tzinfo else result.search_time_window[0].isoformat(),
                result.search_time_window[1].isoformat() + "Z" if not result.search_time_window[1].tzinfo else result.search_time_window[1].isoformat(),
            ],
            "n_vessels_considered": result.n_vessels_considered,
            "caveats": result.caveats,
        },
        "candidates": [
            {
                "mmsi": c.mmsi,
                "vessel_name": c.vessel_name,
                "category": c.category,
                "score": round(float(c.score), 4),
                "features": [f.__dict__ for f in c.features],
                "data_confidence": {
                    "coverage_fraction": c.data_confidence.coverage_fraction,
                    "note": c.data_confidence.note,
                },
            }
            for c in result.candidates
        ],
    }


# -----------------------------------------------------------------------------
# Endpoint: GET / and GET /health
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Basic health check and module connectivity verification."""
    checkpoint_exists = (_repo_root / "unet_spill_best.pth").exists()
    ais_path = find_ais_csv(_repo_root / "data")
    return {
        "status": "ok",
        "service": "OceanTrace Forensic API",
        "version": "1.0.0",
        "modules": {
            "module_1_detection": {
                "name": "Oil Spill SAR Segmentation (U-Net)",
                "status": "connected",
                "checkpoint": str(_repo_root / "unet_spill_best.pth"),
                "ready": checkpoint_exists,
            },
            "module_2_drift": {
                "name": "Backward Drift Reconstruction (OpenDrift/Synthetic)",
                "status": "connected",
                "ready": True,
            },
            "module_3_attribution": {
                "name": "AIS Trajectory & Vessel Attribution",
                "status": "connected",
                "ais_dataset": str(ais_path) if ais_path else None,
                "ready": ais_path is not None and ais_path.exists(),
            },
        },
        "all_modules_connected": checkpoint_exists and (ais_path is not None and ais_path.exists()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



# -----------------------------------------------------------------------------
# Endpoint: POST /detect-spill
# -----------------------------------------------------------------------------
@app.post("/detect-spill", tags=["Module 1 - Detection"])
async def detect_spill_route(request: Request) -> Dict[str, Any]:
    """
    Executes Module 1 oil spill detection inference on a GeoTIFF scene.
    Accepts:
      - JSON body: {"image_path": str, "threshold": float, "input_scale": str}
      - OR multipart/form-data: file upload (file) + optional form fields.
    Returns:
      - detection contract JSON (or 'no_detection' if area == 0 or polygon is None)
    """
    content_type = request.headers.get("content-type", "")
    target_image_path: Optional[str] = None
    threshold: float = 0.70
    input_scale: str = "auto"

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file") or form.get("image")
        if uploaded_file and hasattr(uploaded_file, "filename"):
            existing_match = None
            for f in UPLOAD_DIR.glob(f"*_{uploaded_file.filename}"):
                if f.is_file() and hasattr(uploaded_file, "size") and f.stat().st_size == uploaded_file.size:
                    existing_match = f
                    break
            if existing_match:
                target_image_path = str(existing_match)
                logger.info(f"Reusing existing cached uploaded file: {target_image_path}")
            else:
                save_name = f"{int(datetime.now().timestamp())}_{uploaded_file.filename}"
                save_dest = UPLOAD_DIR / save_name
                with open(save_dest, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file.file, buffer, length=8*1024*1024)
                target_image_path = str(save_dest)
        elif form.get("image_path"):
            target_image_path = str(form.get("image_path"))

        if form.get("threshold"):
            threshold = float(form.get("threshold"))
        if form.get("input_scale"):
            input_scale = str(form.get("input_scale"))
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        target_image_path = body.get("image_path")
        threshold = float(body.get("threshold", 0.70))
        input_scale = body.get("input_scale", "auto")

    if not target_image_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required image input: provide 'image_path' in JSON or upload a GeoTIFF 'file'."
        )

    resolved_path = Path(target_image_path)
    if not resolved_path.is_absolute():
        resolved_path = _repo_root / resolved_path

    if not resolved_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found at: {target_image_path}"
        )

    # Determine scale if auto
    effective_scale = input_scale
    if effective_scale == "auto":
        effective_scale = "db" if ("0000" in resolved_path.name or "real_dataset" in str(resolved_path)) else "linear"

    logger.info(f"Running /detect-spill on {resolved_path} (scale={effective_scale}, threshold={threshold})")
    cached_model = getattr(request.app.state, "spill_model", None) or get_cached_model()
    detection_result = detect_spill(
        image_path=str(resolved_path),
        model_path=str(_repo_root / "unet_spill_best.pth"),
        input_scale=effective_scale,
        threshold=threshold,
        save_json_path=str(_repo_root / "detection.json"),
        preloaded_model=cached_model
    )

    if detection_result.get("status") == "invalid_input":
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=detection_result,
        )

    if not detection_result.get("polygon") or detection_result.get("area_km2", 0.0) == 0.0:
        return {
            "status": "no_detection",
            "message": "No oil spill detected above threshold in the provided scene.",
            "detection": detection_result,
        }

    return {
        "status": "success",
        "detection": detection_result,
    }


# -----------------------------------------------------------------------------
# Endpoint: POST /trace-origin
# -----------------------------------------------------------------------------
@app.post("/trace-origin", tags=["Module 2 - Drift Reconstruction"])
async def trace_origin_route(request: Request) -> Dict[str, Any]:
    """
    Executes Module 2 backward drift reconstruction to estimate origin point and uncertainty.
    Accepts:
      - JSON body: {"detection": dict} OR {"detection_path": str}
        with optional: search_window_hours (float), number_particles (int)
    Returns:
      - Origin centroid, uncertainty ellipse, and drift reconstruction summary.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    detection_data = body.get("detection")
    detection_path = body.get("detection_path")
    search_window_hours = float(body.get("search_window_hours", 6.0))
    number_particles = int(body.get("number_particles", 1000))
    env_netcdf_path = body.get("env_netcdf_path")

    if not detection_data and detection_path:
        d_path = Path(detection_path)
        if not d_path.is_absolute():
            d_path = _repo_root / d_path
        if not d_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detection file not found at: {detection_path}"
            )
        detection_data = load_detection_json(str(d_path))
    elif not detection_data and ( _repo_root / "detection.json" ).exists():
        detection_data = load_detection_json(str(_repo_root / "detection.json"))

    if not detection_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing detection data: provide 'detection' dictionary or 'detection_path'."
        )

    raw_polygon = detection_data.get("polygon") or detection_data.get("slick_polygon")
    if not raw_polygon or detection_data.get("area_km2", 0.0) == 0.0:
        return {
            "status": "no_detection",
            "message": "Cannot trace origin: provided detection contains no valid oil slick polygon.",
            "reconstruction": None,
        }

    # Bridge adapters
    single_polygon = convert_multipolygon_to_polygon(raw_polygon)
    obs_datetime = parse_observation_time(detection_data)

    if env_netcdf_path is None:
        env_netcdf_path = _repo_root / "data" / "samples" / "synthetic_env.nc"
    else:
        env_netcdf_path = Path(env_netcdf_path)

    if not env_netcdf_path.exists():
        logger.info(f"Generating synthetic environmental NetCDF at {env_netcdf_path}...")
        make_synthetic_environment.main()

    recon = reconstruct_source(
        polygon=single_polygon,
        observation_time=obs_datetime,
        search_window_hours=search_window_hours,
        number=number_particles,
        env_mode="netcdf",
        env_kwargs={"paths": str(env_netcdf_path), "name": "synthetic_regional"}
    )

    # Save contract files
    reconstruction_json_path = _repo_root / "reconstruction.json"
    save_reconstruction_json(recon, str(reconstruction_json_path))

    return {
        "status": "success",
        "reconstruction": reconstruction_to_dict(recon),
    }


# -----------------------------------------------------------------------------
# Endpoint: POST /attribute-vessel
# -----------------------------------------------------------------------------
@app.post("/attribute-vessel", tags=["Module 3 - Vessel Attribution"])
async def attribute_vessel_route(request: Request) -> Dict[str, Any]:
    """
    Executes Module 3 vessel candidate evaluation and evidence fusion.
    Accepts:
      - JSON body: {"reconstruction_path": str (optional), "ais_csv_path": str (optional)}
    Returns:
      - Ranked suspect vessel list with evidence features, data confidence, and forensic report.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    recon_path_str = body.get("reconstruction_path", "reconstruction.json")
    recon_path = Path(recon_path_str)
    if not recon_path.is_absolute():
        recon_path = _repo_root / recon_path

    if not recon_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reconstruction contract file not found at: {recon_path}. "
                   f"Please run /trace-origin or /run-full-pipeline first."
        )

    ais_csv_str = body.get("ais_csv_path")
    if ais_csv_str:
        ais_csv_path = Path(ais_csv_str)
        if not ais_csv_path.is_absolute():
            ais_csv_path = _repo_root / ais_csv_path
    else:
        ais_csv_path = find_ais_csv(_repo_root / "data")

    if ais_csv_path is None or not ais_csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No AIS CSV dataset found in 'data/' folder or provided 'ais_csv_path'."
        )

    logger.info(f"Loading reconstruction from: {recon_path}")
    recon = load_reconstruction_json(str(recon_path))

    logger.info(f"Loading AIS dataset from: {ais_csv_path}")
    cached_ais = getattr(request.app.state, "ais_dataset", None) if not ais_csv_str else None
    if cached_ais is None:
        cached_ais = get_cached_ais_dataset(ais_csv_path) or AISDataset.load(str(ais_csv_path))
    ais_dataset = cached_ais

    logger.info("Evaluating vessel candidates...")
    attribution_result = evaluate_candidates(recon, ais_dataset)
    report_text = generate_report(attribution_result)

    # Save contract
    attribution_json_path = _repo_root / "attribution.json"
    save_attribution_json(attribution_result, str(attribution_json_path))

    return {
        "status": "success",
        "attribution": attribution_to_dict(attribution_result),
        "report": report_text,
    }


# -----------------------------------------------------------------------------
# Endpoint: POST /run-full-pipeline
# -----------------------------------------------------------------------------
@app.post("/run-full-pipeline", tags=["Full Pipeline"])
async def run_full_pipeline_route(request: Request) -> Dict[str, Any]:
    """
    Executes the complete end-to-end OceanTrace forensic chain:
      Module 1 (Detect) -> Module 2 (Reconstruct) -> Module 3 (Attribute)
    Accepts:
      - JSON body: {"image_path": str, "ais_csv_path": str (opt), "threshold": float (opt), ...}
      - OR multipart/form-data with uploaded image file + optional fields
    Returns:
      - Final report with detection, drift reconstruction, and vessel candidate ranking.
    """
    content_type = request.headers.get("content-type", "")
    target_image_path: Optional[str] = None
    ais_csv_path: Optional[str] = None
    threshold: float = 0.70
    search_window_hours: float = 6.0
    number_particles: int = 1000
    input_scale: str = "auto"

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file") or form.get("image")
        if uploaded_file and hasattr(uploaded_file, "filename"):
            existing_match = None
            for f in UPLOAD_DIR.glob(f"*_{uploaded_file.filename}"):
                if f.is_file() and hasattr(uploaded_file, "size") and f.stat().st_size == uploaded_file.size:
                    existing_match = f
                    break
            if existing_match:
                target_image_path = str(existing_match)
                logger.info(f"Reusing existing cached uploaded file: {target_image_path}")
            else:
                save_name = f"{int(datetime.now().timestamp())}_{uploaded_file.filename}"
                save_dest = UPLOAD_DIR / save_name
                with open(save_dest, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file.file, buffer, length=8*1024*1024)
                target_image_path = str(save_dest)
        elif form.get("image_path"):
            target_image_path = str(form.get("image_path"))

        if form.get("ais_csv_path"):
            ais_csv_path = str(form.get("ais_csv_path"))
        if form.get("threshold"):
            threshold = float(form.get("threshold"))
        if form.get("search_window_hours"):
            search_window_hours = float(form.get("search_window_hours"))
        if form.get("number_particles"):
            number_particles = int(form.get("number_particles"))
        if form.get("input_scale"):
            input_scale = str(form.get("input_scale"))
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        target_image_path = body.get("image_path")
        ais_csv_path = body.get("ais_csv_path")
        threshold = float(body.get("threshold", 0.70))
        search_window_hours = float(body.get("search_window_hours", 6.0))
        number_particles = int(body.get("number_particles", 1000))
        input_scale = body.get("input_scale", "auto")

    if not target_image_path:
        use_sample = False
        if "multipart/form-data" in content_type:
            use_sample = form.get("use_sample") in (True, "true", "True", "1")
        elif isinstance(body, dict):
            use_sample = body.get("use_sample", False)

        if use_sample:
            target_image_path = "dataset/real_dataset/images/00002.tif"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required image input: provide 'image_path' in JSON or upload a GeoTIFF 'file'."
            )


    resolved_image = Path(target_image_path)
    if not resolved_image.is_absolute():
        resolved_image = _repo_root / resolved_image

    if not resolved_image.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found at: {target_image_path}"
        )

    # Default to 150 particles for fast low-latency forensic simulation
    if number_particles > 200:
        number_particles = 150

    from starlette.concurrency import run_in_threadpool
    logger.info(f"Calling run_full_pipeline on {resolved_image} (threshold={threshold}, particles={number_particles})")
    cached_model = getattr(request.app.state, "spill_model", None) or get_cached_model()
    cached_ais = getattr(request.app.state, "ais_dataset", None) if not ais_csv_path else None
    if cached_ais is None:
        cached_ais = get_cached_ais_dataset(Path(ais_csv_path) if ais_csv_path else None)
    pipeline_result = await run_in_threadpool(
        run_full_pipeline,
        image_path=str(resolved_image),
        checkpoint_path=str(_repo_root / "unet_spill_best.pth"),
        ais_csv_path=ais_csv_path,
        output_dir=str(_repo_root),
        threshold=threshold,
        search_window_hours=search_window_hours,
        number_particles=number_particles,
        input_scale=input_scale,
        preloaded_model=cached_model,
        preloaded_ais_dataset=cached_ais
    )

    if pipeline_result.get("status") == "invalid_input":
        invalid_detection = pipeline_result.get("detection") or {
            "status": "invalid_input",
            "message": pipeline_result.get("message", "Uploaded file does not appear to be a valid SAR satellite image."),
            "slick_polygon": None,
            "area_km2": None,
        }
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=invalid_detection,
        )

    if pipeline_result.get("status") == "NO_DETECTION":
        return {
            "status": "no_detection",
            "message": "No oil spill detected above threshold in the provided scene.",
            "detection": pipeline_result.get("detection"),
            "reconstruction": None,
            "attribution": None,
            "report": None,
        }

    return {
        "status": "success",
        "detection": pipeline_result.get("detection"),
        "reconstruction": reconstruction_to_dict(pipeline_result.get("reconstruction")),
        "attribution": attribution_to_dict(pipeline_result.get("attribution")),
        "report": pipeline_result.get("report"),
    }
