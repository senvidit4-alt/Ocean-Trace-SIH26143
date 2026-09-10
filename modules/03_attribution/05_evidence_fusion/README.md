# Module 05: Evidence Fusion

*Internal implementation component of Product Module 3 — Attribution*



**1. Purpose:** Score and rank candidate vessels based on their alignment with the reconstructed spill source.
**2. Inputs:** `SourceReconstruction` object (Module 03) and `AISDataset` (Module 04).
**3. Processing:** Evaluates 5 evidence dimensions (spatial proximity, temporal overlap, trajectory consistency, drift consistency, speed/course behavior).
**4. Outputs:** `AttributionResult` containing ranked candidates and a generated textual incident report.
**5. Main files:** `src/evidence_fusion.py`
**6. How to run/test it:** Execute `tests/test_evidence_fusion.py`.
**7. Known limitations:** AIS gaps and missing features are gracefully handled, but exact attribution relies on the availability of high-quality AIS and environmental data.
