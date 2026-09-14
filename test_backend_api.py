"""
test_backend_api.py
===================
Automated integration test for the OceanTrace FastAPI backend.
Tests:
  1. GET /health
  2. POST /run-full-pipeline with real_techie1_image129aug.tif (Graceful 'no_detection' test)
  3. POST /run-full-pipeline with 00002.tif (Full positive detection + reconstruction + attribution)
  4. POST /detect-spill
  5. POST /trace-origin
  6. POST /attribute-vessel
"""

import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_health():
    print("\n--- Testing GET /health ---")
    response = client.get("/health")
    print("Status code:", response.status_code)
    data = response.json()
    print("Response:", data)
    assert response.status_code == 200
    assert data["status"] == "ok"
    print(">>> GET /health PASSED")


def test_run_full_pipeline_no_detection():
    print("\n--- Testing POST /run-full-pipeline with real_techie1_image129aug.tif (No Detection Case) ---")
    image_path = "dataset/backup_synthetic_200/images/real_techie1_image129aug.tif"
    payload = {
        "image_path": image_path,
        "threshold": 0.70
    }
    response = client.post("/run-full-pipeline", json=payload)
    print("Status code:", response.status_code)
    data = response.json()
    print("Result status:", data.get("status"))
    print("Message:", data.get("message"))
    assert response.status_code == 200
    assert data["status"] == "no_detection"
    assert data["detection"]["area_km2"] == 0.0
    print(">>> POST /run-full-pipeline (no detection) PASSED gracefully!")


def test_run_full_pipeline_positive_detection():
    print("\n--- Testing POST /run-full-pipeline with 00002.tif (Positive Detection Case) ---")
    image_path = "dataset/real_dataset/images/00002.tif"
    payload = {
        "image_path": image_path,
        "threshold": 0.70,
        "search_window_hours": 6.0,
        "number_particles": 1000
    }
    response = client.post("/run-full-pipeline", json=payload)
    print("Status code:", response.status_code)
    data = response.json()
    print("Result status:", data.get("status"))
    assert response.status_code == 200
    assert data["status"] == "success"
    
    detection = data.get("detection", {})
    reconstruction = data.get("reconstruction", {})
    attribution = data.get("attribution", {})
    report = data.get("report", "")

    print(f"Detection: area={detection.get('area_km2')} km², confidence={detection.get('confidence')}")
    print(f"Reconstruction: origin_centroid={reconstruction.get('origin_centroid')}")
    print(f"Attribution: considered {attribution.get('analysis_metadata', {}).get('n_vessels_considered')} vessels, candidates={len(attribution.get('candidates', []))}")
    print(f"Report preview:\n{report[:300]}...\n")
    print(">>> POST /run-full-pipeline (positive detection) PASSED with full report!")


def test_modular_endpoints():
    print("\n--- Testing Modular Endpoints (/detect-spill -> /trace-origin -> /attribute-vessel) ---")
    
    # 1. /detect-spill
    print("Calling /detect-spill...")
    det_resp = client.post("/detect-spill", json={"image_path": "dataset/real_dataset/images/00002.tif", "threshold": 0.70})
    assert det_resp.status_code == 200
    det_data = det_resp.json()
    assert det_data["status"] == "success"
    print("Detection area:", det_data["detection"]["area_km2"])

    # 2. /trace-origin
    print("Calling /trace-origin...")
    trace_resp = client.post("/trace-origin", json={"detection": det_data["detection"], "search_window_hours": 6.0})
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert trace_data["status"] == "success"
    print("Reconstruction origin:", trace_data["reconstruction"]["origin_centroid"])

    # 3. /attribute-vessel
    print("Calling /attribute-vessel...")
    attr_resp = client.post("/attribute-vessel", json={"reconstruction_path": "reconstruction.json"})
    assert attr_resp.status_code == 200
    attr_data = attr_resp.json()
    assert attr_data["status"] == "success"
    print("Attribution success, candidates:", len(attr_data["attribution"]["candidates"]))
    print(">>> Modular endpoints workflow PASSED!")


def test_reject_non_sar_image():
    print("\n--- Testing Rejection of Non-Satellite Arbitrary Image (.tif) ---")
    import tempfile
    import numpy as np
    import rasterio

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_photo = Path(tmpdir) / "face_photo.tif"
        data = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)
        with rasterio.open(str(fake_photo), "w", driver="GTiff", height=100, width=100, count=3, dtype="uint8") as dst:
            dst.write(data)

        # Test /detect-spill
        resp = client.post("/detect-spill", json={"image_path": str(fake_photo)})
        print("Status code for non-SAR image (/detect-spill):", resp.status_code)
        assert resp.status_code == 422
        payload = resp.json()
        assert payload["status"] == "invalid_input"
        assert payload["slick_polygon"] is None
        assert payload["area_km2"] is None
        print("Rejection response payload:", payload)

        # Test /run-full-pipeline
        pipe_resp = client.post("/run-full-pipeline", json={"image_path": str(fake_photo)})
        print("Status code for non-SAR image (/run-full-pipeline):", pipe_resp.status_code)
        assert pipe_resp.status_code == 422
        pipe_payload = pipe_resp.json()
        assert pipe_payload["status"] == "invalid_input"
        print(">>> Rejection of non-satellite image PASSED (HTTP 422 Unprocessable Entity)!")


if __name__ == "__main__":
    test_health()
    test_run_full_pipeline_no_detection()
    test_run_full_pipeline_positive_detection()
    test_modular_endpoints()
    test_reject_non_sar_image()
    print("\n============================================================")
    print("ALL API ENDPOINT INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("============================================================\n")
