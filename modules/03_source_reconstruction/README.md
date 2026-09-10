# Module 03: Source Reconstruction

**1. Purpose:** Estimate the probable origin and release time of an observed oil slick using backward drift propagation.
**2. Inputs:** Slick GeoJSON (or equivalent mask), observation time, environmental readers.
**3. Processing:** Simulates particle trajectories backwards in time using OpenDrift (OpenOil/OceanDrift).
**4. Outputs:** `SourceReconstruction` object containing origin centroid, time bounds, and spatial uncertainty metrics.
**5. Main files:** `src/source_reconstruction.py`
**6. How to run/test it:** Execute `tests/test_source_reconstruction.py` or `tests/test_source_reconstruction_module.py`.
**7. Known limitations:** Backward tracking is deterministic but uncertainty relies on bounding ellipse approximations.
