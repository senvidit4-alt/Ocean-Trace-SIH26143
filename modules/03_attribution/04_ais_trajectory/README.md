# Module 04: AIS Trajectory

*Internal implementation component of Product Module 3 — Attribution*



**1. Purpose:** Ingest and process Automatic Identification System (AIS) data for vessel tracking.
**2. Inputs:** AIS data in CSV format.
**3. Processing:** Parses vessel tracks, identifies gaps in coverage, and queries vessels by spatial-temporal proximity.
**4. Outputs:** `AISDataset` and `VesselTrajectory` objects.
**5. Main files:** `src/ais_trajectory.py`
**6. How to run/test it:** Execute `tests/test_ais_load.py`.
**7. Known limitations:** Duplicate handling relies on deterministic tie-breaking. Heavy memory usage for very large unindexed CSVs.
