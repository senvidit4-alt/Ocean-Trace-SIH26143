# Module 02: Environment

*Internal implementation component of Product Module 2 — Drift*



**1. Purpose:** Provide integrated ocean current and wind data for drift modeling.
**2. Inputs:** CMEMS (currents) and ERA5 (wind) NetCDF files.
**3. Processing:** Harmonizes and standardizes environmental variables for OpenDrift compatibility.
**4. Outputs:** OpenDrift compatible reader objects.
**5. Main files:** `src/environment_readers.py`, `src/make_synthetic_environment.py`
**6. How to run/test it:** Execute `tests/test_real_environment.py`.
**7. Known limitations:** ERA5 data requires manual mapping of `valid_time` to standard CF time variables.
