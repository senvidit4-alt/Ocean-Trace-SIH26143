# Module 01: Oil Spill Detection

*Internal implementation component of Product Module 1 — Detection*



**1. Purpose:** Detect oil spills from Synthetic Aperture Radar (SAR) imagery using deep learning.
**2. Inputs:** SAR imagery (e.g., Sentinel-1).
**3. Processing:** Neural network inference to generate segmentation masks.
**4. Outputs:** Binary or probabilistic oil spill masks.
**5. Main files:** `src/model.py`, `src/inference.py`, `src/train.py`, `src/data_loader.py`
**6. How to run/test it:** Run tests in `tests/` (e.g., `pytest tests/test_pipeline.py`).
**7. Known limitations:** Model depends on pre-trained weights which are not included in the repository.
