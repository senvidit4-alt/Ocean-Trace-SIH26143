"""
test_sar_validation.py
======================
Comprehensive test suite for Module 1 input validation gate.
Verifies that:
  1. Genuine Sentinel-1 SAR scenes pass validation.
  2. Non-satellite images (e.g. RGB photos, face photos, uint8 images saved as .tif) are rejected.
  3. detect_spill() returns the required invalid_input schema without running model inference.
  4. FastAPI endpoints (/detect-spill, /run-full-pipeline) return HTTP 422 Unprocessable Entity
     with the invalid_input payload when non-SAR images are uploaded/provided.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from fastapi.testclient import TestClient

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "modules/01_detection/src"))

from validation import is_valid_sar_image
from inference import detect_spill
from backend.main import app

client = TestClient(app)


def test_real_sentinel1_scenes_pass():
    """Verify all legitimate Sentinel-1 SAR test products in the repository pass validation."""
    real_scenes = [
        _repo_root / "dataset/real_dataset/images/00000.tif",
        _repo_root / "dataset/real_dataset/images/00002.tif",
        _repo_root / "dataset/backup_synthetic_200/images/real_techie1_image129aug.tif",
    ]
    for scene in real_scenes:
        if scene.exists():
            is_valid, msg = is_valid_sar_image(scene)
            print(f"[PASS CHECK] {scene.name}: is_valid={is_valid}, msg={msg}")
            assert is_valid is True, f"Legitimate scene {scene.name} was incorrectly rejected: {msg}"
            assert bool(is_valid) is True


def test_simulated_rgb_photo_rejected():
    """Verify a 3-band RGB photo saved as .tif is rejected by validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        photo_path = Path(tmpdir) / "simulated_photo_rgb.tif"
        # Simulate an RGB camera photo (3 bands, uint8, values 0-255)
        rgb_data = np.random.randint(50, 200, (3, 256, 256), dtype=np.uint8)
        with rasterio.open(
            str(photo_path),
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=3,
            dtype="uint8",
        ) as dst:
            dst.write(rgb_data)

        is_valid, msg = is_valid_sar_image(photo_path)
        print(f"[REJECT CHECK] 3-Band RGB Photo: is_valid={is_valid}, msg={msg}")
        assert is_valid is False
        assert bool(is_valid) is False
        assert "Expected Sentinel-1 characteristics" in msg

        # Test detect_spill behavior
        res = detect_spill(str(photo_path))
        print("[REJECT CHECK] detect_spill response:", res)
        assert res["status"] == "invalid_input"
        assert res["slick_polygon"] is None
        assert res["area_km2"] is None
        assert "Expected Sentinel-1 characteristics" in res["message"]


def test_simulated_grayscale_photo_rejected():
    """Verify a 1-band uint8 photo (e.g. grayscale face portrait) saved as .tif is rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gray_path = Path(tmpdir) / "simulated_portrait_gray.tif"
        gray_data = np.random.randint(40, 220, (1, 256, 256), dtype=np.uint8)
        with rasterio.open(
            str(gray_path),
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=1,
            dtype="uint8",
        ) as dst:
            dst.write(gray_data)

        is_valid, msg = is_valid_sar_image(gray_path)
        print(f"[REJECT CHECK] 1-Band Grayscale uint8 Photo: is_valid={is_valid}, msg={msg}")
        assert is_valid is False
        assert "uint8" in msg or "photographic" in msg


def test_simulated_float_photo_distribution_rejected():
    """Verify a 1-band float image with photo-level values (mean=120, max=255) is rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        float_photo = Path(tmpdir) / "simulated_float_photo.tif"
        float_data = np.random.uniform(50.0, 240.0, (1, 256, 256)).astype(np.float32)
        with rasterio.open(
            str(float_photo),
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=1,
            dtype="float32",
        ) as dst:
            dst.write(float_data)

        is_valid, msg = is_valid_sar_image(float_photo)
        print(f"[REJECT CHECK] Float32 Photo Distribution: is_valid={is_valid}, msg={msg}")
        assert is_valid is False
        assert "photographic 0-255" in msg or "Expected Sentinel-1 characteristics" in msg


def test_fastapi_endpoints_reject_non_sar():
    """Verify FastAPI backend returns HTTP 422 for non-SAR images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_photo = Path(tmpdir) / "random_photo.tif"
        rgb_data = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        with rasterio.open(
            str(fake_photo),
            "w",
            driver="GTiff",
            height=128,
            width=128,
            count=3,
            dtype="uint8",
        ) as dst:
            dst.write(rgb_data)

        # 1. Test POST /detect-spill with JSON image_path
        resp_detect = client.post("/detect-spill", json={"image_path": str(fake_photo)})
        print("[FASTAPI CHECK] /detect-spill status:", resp_detect.status_code)
        assert resp_detect.status_code == 422
        data_detect = resp_detect.json()
        assert data_detect["status"] == "invalid_input"
        assert data_detect["slick_polygon"] is None
        assert data_detect["area_km2"] is None

        # 2. Test POST /run-full-pipeline with JSON image_path
        resp_pipe = client.post("/run-full-pipeline", json={"image_path": str(fake_photo)})
        print("[FASTAPI CHECK] /run-full-pipeline status:", resp_pipe.status_code)
        assert resp_pipe.status_code == 422
        data_pipe = resp_pipe.json()
        assert data_pipe["status"] == "invalid_input"
        assert data_pipe["slick_polygon"] is None
        assert data_pipe["area_km2"] is None


def test_fastapi_endpoints_accept_real_sar():
    """Verify FastAPI backend accepts real Sentinel-1 SAR scene (HTTP 200)."""
    real_scene = "dataset/real_dataset/images/00002.tif"
    resp = client.post("/detect-spill", json={"image_path": real_scene, "threshold": 0.70})
    print("[FASTAPI CHECK] /detect-spill real scene status:", resp.status_code)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "no_detection")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Sentinel-1 SAR Input Validation Tests")
    print("=" * 60)
    test_real_sentinel1_scenes_pass()
    test_simulated_rgb_photo_rejected()
    test_simulated_grayscale_photo_rejected()
    test_simulated_float_photo_distribution_rejected()
    test_fastapi_endpoints_reject_non_sar()
    test_fastapi_endpoints_accept_real_sar()
    print("\n[SUCCESS] All Sentinel-1 input validation tests passed successfully!")
