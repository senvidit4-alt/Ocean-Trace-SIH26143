"""
run_pipeline.py
===============
End-to-end integration orchestrator and thin bridge adapter for OceanTrace.
Connects:
  Module 1 (Oil Spill Detection)
  -> Module 2 (Drift & Source Reconstruction)
  -> Module 3 (AIS Ingestion & Evidence Fusion / Attribution)

Handles contract adaptations:
  1. MultiPolygon -> largest single Polygon conversion with audit logging
  2. ISO-8601 observation_time string -> Python datetime parsing
  3. Full pipeline execution and serialization via contracts/adapters.py
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Set up repository paths
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root / "contracts"))
sys.path.insert(0, str(_repo_root / "modules/01_detection/src"))
sys.path.insert(0, str(_repo_root / "modules/02_drift/02_environment/src"))
sys.path.insert(0, str(_repo_root / "modules/02_drift/03_source_reconstruction/src"))
sys.path.insert(0, str(_repo_root / "modules/03_attribution/04_ais_trajectory/src"))
sys.path.insert(0, str(_repo_root / "modules/03_attribution/05_evidence_fusion/src"))

from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import rasterio

# Module imports
from inference import detect_spill
from adapters import (
    load_detection_json,
    save_reconstruction_json,
    save_attribution_json,
)
from source_reconstruction import reconstruct_source, SourceReconstruction
import make_synthetic_environment
from ais_trajectory import AISDataset
from evidence_fusion import evaluate_candidates, generate_report, AttributionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_pipeline")


def convert_multipolygon_to_polygon(geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Converts GeoJSON geometry to a single Polygon:
    - If geometry['type'] == 'Polygon', returns it unchanged.
    - If geometry['type'] == 'MultiPolygon', selects the largest sub-polygon
      by area (using shapely) and returns it as a single 'Polygon' geometry.
    - Logs a warning when this conversion happens, noting how many clusters
      were present and which one was kept, so this simplification is auditable.
    """
    if not geometry or not isinstance(geometry, dict):
        return geometry

    geom_type = geometry.get("type")
    if geom_type == "Polygon":
        return geometry

    if geom_type == "MultiPolygon":
        geom = shape(geometry)
        if not isinstance(geom, MultiPolygon):
            if isinstance(geom, Polygon):
                return json.loads(json.dumps(mapping(geom)))
            raise ValueError(f"Expected MultiPolygon geometry, but shapely parsed as '{geom.geom_type}'")

        polygons = list(geom.geoms)
        n_clusters = len(polygons)
        if n_clusters == 0:
            raise ValueError("MultiPolygon has 0 constituent polygons.")

        # Find largest sub-polygon by area
        largest_idx, largest_poly = max(
            enumerate(polygons),
            key=lambda item: item[1].area
        )

        logger.warning(
            f"[AUDIT] Geometry simplification: Detected MultiPolygon with {n_clusters} disjoint "
            f"slick clusters. Module 2 requires a single Polygon. Selected cluster index {largest_idx} "
            f"(largest area: {largest_poly.area:.8f} deg²) and discarded {n_clusters - 1} smaller clusters."
        )

        return json.loads(json.dumps(mapping(largest_poly)))

    raise ValueError(f"Unsupported geometry type: '{geom_type}'. Expected 'Polygon' or 'MultiPolygon'.")


def parse_observation_time(detection_json: Dict[str, Any]) -> datetime:
    """
    Extracts observation_time (or detection_timestamp, whichever is present)
    and converts it to a UTC-naive Python datetime object via datetime.fromisoformat().
    """
    time_str = detection_json.get("observation_time") or detection_json.get("detection_timestamp")
    if not time_str:
        raise KeyError("Neither 'observation_time' nor 'detection_timestamp' found in detection JSON.")

    if time_str.endswith("Z"):
        time_str = time_str[:-1] + "+00:00"

    dt = datetime.fromisoformat(time_str)

    # OpenDrift and downstream attribution treat all timestamps as UTC-naive
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def find_ais_csv(data_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Searches the specified data directory (defaults to repo 'data/') for an AIS CSV dataset.
    """
    if data_dir is None:
        data_dir = _repo_root / "data"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        return None

    # Check for any .csv files in data/ or subdirectories
    csv_candidates = list(data_dir.rglob("*.csv"))
    if csv_candidates:
        return csv_candidates[0]

    return None


def run_full_pipeline(
    image_path: str,
    checkpoint_path: str = "unet_spill_best.pth",
    ais_csv_path: Optional[str] = None,
    output_dir: str = ".",
    threshold: float = 0.70,
    env_netcdf_path: Optional[str] = None,
    search_window_hours: float = 6.0,
    number_particles: int = 1000,
    input_scale: str = "auto",
    preloaded_model: Optional[Any] = None,
    preloaded_ais_dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator executing the full end-to-end OceanTrace pipeline:
      a. Module 1: Oil spill detection on SAR scene
      b. Contract: Load detection.json
      c. Adapter: Convert MultiPolygon to largest Polygon
      d. Adapter: Parse ISO observation time to datetime
      e. Module 2: Backward drift source reconstruction
      f. Contract: Save reconstruction.json
      g. Module 3: Load AIS dataset via AISDataset.load()
      h. Module 3: Evaluate vessel candidates
      i. Module 3: Generate attribution report
      j. Contract: Save attribution.json and print report
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    detection_json_path = out_path / "detection.json"
    reconstruction_json_path = out_path / "reconstruction.json"
    attribution_json_path = out_path / "attribution.json"

    logger.info("=" * 60)
    logger.info("Starting OceanTrace End-to-End Pipeline")
    logger.info(f"Target image: {image_path}")
    logger.info(f"Model checkpoint: {checkpoint_path}")
    if threshold != 0.70:
        logger.info(f"DEMO SETTING - threshold lowered to {threshold} for demonstration purposes. Production default remains 0.70.")
        print(f"\n>>> DEMO SETTING - threshold lowered to {threshold} for demonstration purposes. Production default remains 0.70.\n")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Step a: Module 1 Inference
    # -------------------------------------------------------------------------
    logger.info("[Step 1/6] Running Module 1: Oil Spill Detection...")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Input SAR image not found at: {image_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Determine input scale if set to 'auto'
    if input_scale == "auto":
        # Check if Zenodo/real benchmark scene which is stored in dB scale
        with rasterio.open(image_path) as src:
            sample_val = src.read(1, window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)))
            # If negative values exist (e.g. -15 dB), it's in dB
            if (sample_val < 0).any() or "real_dataset" in str(image_path):
                effective_scale = "db"
            else:
                effective_scale = "linear"
    else:
        effective_scale = input_scale

    logger.info(f"Using input_scale='{effective_scale}' for SAR inference (threshold={threshold})")

    detect_result = detect_spill(
        image_path=image_path,
        model_path=checkpoint_path,
        input_scale=effective_scale,
        threshold=threshold,
        save_json_path=str(detection_json_path),
        preloaded_model=preloaded_model
    )

    if detect_result.get("status") == "invalid_input":
        logger.warning(f"Input validation rejected SAR scene: {detect_result.get('message')}")
        return {
            "status": "invalid_input",
            "message": detect_result.get("message"),
            "detection": detect_result,
            "reconstruction": None,
            "attribution": None,
            "report": None,
        }

    # -------------------------------------------------------------------------
    # Step b: Load detection.json via contract adapter
    # -------------------------------------------------------------------------
    logger.info("[Step 2/6] Loading detection contract via load_detection_json()...")
    detection_data = load_detection_json(str(detection_json_path))

    area_km2 = detection_data.get("area_km2", 0.0)
    confidence = detection_data.get("confidence", 0.0)
    raw_polygon = detection_data.get("polygon")

    logger.info(f"Detection Results: Area = {area_km2} km², Confidence = {confidence}")

    if not raw_polygon:
        logger.warning("No oil spill polygon detected above threshold. Pipeline stopped.")
        return {
            "status": "NO_DETECTION",
            "detection": detection_data,
            "reconstruction": None,
            "attribution": None
        }

    # -------------------------------------------------------------------------
    # Step c: Apply convert_multipolygon_to_polygon adapter
    # -------------------------------------------------------------------------
    logger.info("[Step 3/6] Applying MultiPolygon-to-Polygon bridge adapter...")
    single_polygon = convert_multipolygon_to_polygon(raw_polygon)

    # -------------------------------------------------------------------------
    # Step d: Parse observation time to datetime object
    # -------------------------------------------------------------------------
    obs_datetime = parse_observation_time(detection_data)
    logger.info(f"Parsed observation time: {obs_datetime} (UTC)")

    # -------------------------------------------------------------------------
    # Step e: Module 2 Source Reconstruction
    # -------------------------------------------------------------------------
    logger.info("[Step 4/6] Running Module 2: Backward Drift Source Reconstruction...")
    if env_netcdf_path is None:
        env_netcdf_path = _repo_root / "data" / "samples" / "synthetic_env.nc"
    else:
        env_netcdf_path = Path(env_netcdf_path)

    if not env_netcdf_path.exists():
        logger.info(f"Environmental NetCDF not found at {env_netcdf_path}. Generating sample dataset...")
        make_synthetic_environment.main()

    recon = reconstruct_source(
        polygon=single_polygon,
        observation_time=obs_datetime,
        search_window_hours=search_window_hours,
        number=number_particles,
        env_mode="netcdf",
        env_kwargs={"paths": str(env_netcdf_path), "name": "synthetic_regional"}
    )

    # -------------------------------------------------------------------------
    # Step f: Save reconstruction.json via contract adapter
    # -------------------------------------------------------------------------
    logger.info(f"Saving reconstruction contract to {reconstruction_json_path}...")
    save_reconstruction_json(recon, str(reconstruction_json_path))

    lon_c, lat_c = recon.origin_centroid
    logger.info(f"Reconstruction Complete! Origin Centroid: ({lat_c:.4f}°N, {lon_c:.4f}°E)")
    if recon.coverage_warnings:
        logger.info(f"Coverage warnings ({len(recon.coverage_warnings)}): {recon.coverage_warnings[0]}")

    # -------------------------------------------------------------------------
    # Step g: Check and Load AIS data via Module 3 AISDataset.load()
    # -------------------------------------------------------------------------
    logger.info("[Step 5/6] Locating and Loading AIS Dataset...")
    resolved_ais_csv = None
    if ais_csv_path:
        resolved_ais_csv = Path(ais_csv_path)
    else:
        resolved_ais_csv = find_ais_csv(_repo_root / "data")

    if resolved_ais_csv is None or not resolved_ais_csv.exists():
        msg = (
            "CRITICAL: No AIS CSV dataset found in 'data/' folder (or specified path). "
            "Module 3 cannot ingest vessel trajectories without an AIS dataset. "
            "Please check where the AIS data should be placed or provide a path to an AIS CSV."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    if preloaded_ais_dataset is not None:
        logger.info("Using cached preloaded AIS dataset.")
        ais_dataset = preloaded_ais_dataset
    else:
        logger.info(f"Loading AIS dataset from: {resolved_ais_csv}")
        ais_dataset = AISDataset.load(str(resolved_ais_csv))

    # -------------------------------------------------------------------------
    # Step h & i: Evaluate Candidates and Generate Report
    # -------------------------------------------------------------------------
    logger.info("[Step 6/6] Evaluating Vessel Candidates and Fusing Evidence...")
    attribution_result = evaluate_candidates(recon, ais_dataset)

    report_text = generate_report(attribution_result)

    # -------------------------------------------------------------------------
    # Step j: Print final report and save attribution.json
    # -------------------------------------------------------------------------
    save_attribution_json(attribution_result, str(attribution_json_path))
    logger.info(f"Saved attribution contract to {attribution_json_path}")

    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60 + "\n")

    return {
        "status": "SUCCESS",
        "detection": detection_data,
        "reconstruction": recon,
        "attribution": attribution_result,
        "report": report_text
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanTrace End-to-End Pipeline Runner")
    parser.add_argument(
        "--image",
        type=str,
        default="dataset/real_dataset/images/00002.tif",
        help="Path to SAR GeoTIFF image (default: dataset/real_dataset/images/00002.tif)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="unet_spill_best.pth",
        help="Path to U-Net model checkpoint (default: unet_spill_best.pth)"
    )
    parser.add_argument(
        "--ais-csv",
        type=str,
        default=None,
        help="Path to AIS CSV file (default: auto-detect in data/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for contracts (default: .)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Detection confidence threshold (default: 0.70)"
    )

    args = parser.parse_args()

    try:
        run_full_pipeline(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            ais_csv_path=args.ais_csv,
            output_dir=args.output_dir,
            threshold=args.threshold
        )
    except Exception as e:
        logger.error(f"Pipeline execution halted: {e}")
        sys.exit(1)
