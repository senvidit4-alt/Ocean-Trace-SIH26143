# OceanTrace (Trace-Oceans)

OceanTrace is a modular pipeline for oil spill attribution, combining satellite remote sensing, drift modeling, and vessel tracking to identify probable sources of marine pollution.

## System Architecture

OceanTrace is built around **THREE** functional product modules. These modules are intentionally independent and communicate through explicit JSON data contracts rather than importing each other's internal implementations.

### Module 1 — Detection
**Owner:** Techie 2
- **Input:** Sentinel-1 SAR satellite image
- **Responsibilities:** Detect oil slick, distinguish probable oil from look-alikes, produce the detected slick geometry and detection metadata.
- **Output:** `detection.json` (GeoJSON polygon, detected area, age estimate, class probability, observation time)

### Module 2 — Drift
**Owner:** Techie 3
- **Input:** `detection.json` from Module 1, ocean current data, wind data
- **Responsibilities:** Perform backward Lagrangian drift/source reconstruction, estimate where the slick most probably originated, estimate origin time, quantify uncertainty.
- **Output:** `reconstruction.json` + `trajectory.nc` (probable origin point/region, origin time, uncertainty radius/ellipse, full particle trajectory file)

### Module 3 — Attribution
**Owner:** Techie 3
- **Input:** `reconstruction.json` from Module 2, AIS vessel trajectory data
- **Responsibilities:** Identify vessels observable in the reconstructed source region/time window, correlate vessel trajectories with oil movement (evaluating spatial proximity, temporal overlap, trajectory consistency, drift consistency, and speed/course behavior), and produce a transparent evidence-based ranking.
- **Output:** `attribution.json` (ranked candidate vessel list, evidence/features for each candidate, human-readable investigation report)

---

## Internal Implementation Components

The three functional modules are implemented across five internal directories:

*   **Product Module 1 — Detection**
    *   `modules/01_oil_spill_detection/` (U-Net, PyTorch)
*   **Product Module 2 — Drift**
    *   `modules/02_environment/` (CMEMS, ERA5 readers)
    *   `modules/03_source_reconstruction/` (OpenDrift, OpenOil)
*   **Product Module 3 — Attribution**
    *   `modules/04_ais_trajectory/` (GeoPandas, AIS parsing)
    *   `modules/05_evidence_fusion/` (Transparent weighted evidence ranking)

*Disclaimer: Module 1 detects a PROBABLE oil slick; it does not magically prove oil. Module 2 produces a probable source region/time with uncertainty. Module 3 identifies investigative candidates, NOT legally guilty vessels. Proximity alone does not establish responsibility. Evidence scores are transparent triage/ranking scores, not probabilities of guilt.*
