# Module 01: Oil Spill Detection

*Internal implementation component of Product Module 1 - Detection (Trace-Oceans)*

---

## 1. Overview & Purpose
Module 1 performs automated oil spill detection from Sentinel-1 Synthetic Aperture Radar (SAR) imagery using deep learning segmentation. It produces high-confidence vector polygons, slick surface area estimates, and shape-compactness age heuristics formatted for direct handoff to Module 2 (OpenDrift drift and hindcast trajectory modeling).

---

## 2. Key Architecture & Features

### A. Real-Data Trained U-Net Model
- **Checkpoint:** `unet_spill_best.pth` (trained on 669 real Sentinel-1 SAR image/mask pairs from Zenodo).
- **Architecture:** 4-stage encoder-decoder U-Net with skip connections (`model.py`). Dynamically supports direct-mapped and modular state dictionaries via `load_spill_model()`.
- **Loss Function:** `HybridFocalLoss` (`model.py`), blending **Focal Tversky Loss** (alpha=0.3, beta=0.7, gamma=0.75) with **DiceBCELoss** to aggressively penalize false alarms from lookalikes (calm water, wind shadows) while maintaining high sensitivity to thin oil slicks.

### B. Level 1 Preprocessing Fix
- **Fixed SAR dB Normalization:** SAR linear pixel values are converted to decibels:
  dB = 10 * log10(pixel + 1e-6)
- **Fixed Clipping Window:** Clipped to [-35.0 dB, -5.0 dB] and normalized to [0.0, 1.0]. This avoids per-image min-max stretching that previously caused clean ocean scenes to be falsely detected as spills.

### C. Inference & False-Alarm Suppression Pipeline
1. **Sliding Window Tiling:** 
   - Full Sentinel-1 GRD swaths (e.g. 20,000 x 20,000 pixels) are tiled into 2048 x 2048 blocks, downsampled to 128 x 128 for inference, and upsampled back to full resolution via nearest-neighbor interpolation.
2. **Detection Threshold:**
   - Evaluated across thresholds 0.60, 0.70, 0.80; optimized permanently to **0.70** for tight, clean slick boundaries matching ground truth without haze.
3. **Coastal Land Mask & Buffer:**
   - Built-in coastal buffer filter (`gulf_coast_land.geojson` or Natural Earth 10m) masking detections within 1.0 km of coastlines and shallow wetlands to prevent shoreline false positives.
4. **Minimum Area Filter:**
   - Filters out candidate polygons smaller than 0.2 km2 to eliminate speckle noise.

---

## 3. Module Interface & Contract Compliance
The output of Module 1 conforms to the project-wide `detection.json` contract specified in `contracts/CONTRACTS.md`:

```json
{
  "contract_version": "1.0",
  "observation_time": "2026-09-10T17:53:20.595147+00:00",
  "polygon": {
    "type": "Polygon",
    "coordinates": [[[lon1, lat1], [lon2, lat2], "..."]]
  },
  "area_km2": 2.24,
  "confidence": 0.82,
  "age_hrs": 14.5,
  "georeference_status": "REAL",
  "metadata": {
    "satellite": "Sentinel-1",
    "threshold": 0.70,
    "georeference_status": "REAL"
  }
}
```

*Backward compatibility aliases (`slick_polygon`, `detection_timestamp`, `estimated_age_hours`) are also maintained in the returned dictionary.*

---

## 4. File Structure
- `src/model.py`: U-Net architectures, `DiceBCELoss`, `FocalTverskyLoss`, `HybridFocalLoss`, and `load_spill_model()`.
- `src/inference.py`: End-to-end inference pipeline, sliding window tiling, coastal land masking, polygonization, and contract output export.
- `src/data_loader.py`: SAR dataset loader with fixed dB clipping [-35.0, -5.0] and real-time augmentation.
- `src/train.py`: Training loop with early stopping, CLI flags, and `HybridFocalLoss`.
- `src/visualize_prediction.py`: 3-panel and 4-panel visualizers comparing original SAR, ground truth, and predicted masks.
- `src/threshold_sweep.py`: Threshold optimization and verification tool (0.60, 0.70, 0.80).
- `src/gulf_coast_land.geojson`: 10m Natural Earth coastal land mask for the Gulf of Mexico.
- `tests/test_inference_logic.py`: Verification of polygon generation and age heuristic calculation.
- `tests/test_pipeline.py`: Integration test of data loading, forward/backward pass, and metrics.

---

## 5. How to Run

### Inference on a SAR Scene:
```bash
python modules/01_detection/src/inference.py --image path/to/scene.tif --checkpoint unet_spill_best.pth --out detection.json
```

### Visualization:
```bash
python modules/01_detection/src/visualize_prediction.py --image path/to/scene.tif --mask path/to/ground_truth.tif --checkpoint unet_spill_best.pth --save comparison.png
```

### Run Tests:
```bash
python modules/01_detection/tests/test_inference_logic.py
python modules/01_detection/tests/test_pipeline.py
```
