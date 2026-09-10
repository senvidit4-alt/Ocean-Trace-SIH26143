# OceanTrace Methodology

This document outlines the scientific methodology and product principles of the OceanTrace pipeline, divided across the three functional modules.

## 1. Detection (Module 1)
**Internal Implementation:** `modules/01_oil_spill_detection/`

- **SAR Imagery:** Synthetic Aperture Radar (Sentinel-1) is used to detect damping of capillary waves on the ocean surface, characteristic of oil slicks.
- **Deep Learning:** A U-Net (ResNet34 backbone) model segments the imagery.
- **Look-alikes:** The methodology distinguishes probable oil from "look-alikes" (e.g., algal blooms, calm water patches, upwelling).
- **Core Principle:** Module 1 detects a **PROBABLE** oil slick; it does not magically prove the presence of oil or its chemical composition.

## 2. Drift (Module 2)
**Internal Implementation:** `modules/02_environment/` and `modules/03_source_reconstruction/`

- **Lagrangian Particle Tracking:** The observed oil slick geometry is treated as an initial particle cloud and propagated backward in time.
- **Environmental Forcing:** OpenDrift (OpenOil) calculates drift vectors using ocean currents (CMEMS) and wind data (ERA5).
- **Origin Estimation:** A probable source region and origin time window are calculated.
- **Uncertainty Quantification:** The particle cloud's dispersion is modeled into a covariance ellipse and percentile-based radius metrics to rigorously capture spatial uncertainty.
- **Core Principle:** Module 2 produces a probable source region/time with uncertainty. It identifies where a slick likely came from, but does not identify the vessel.

## 3. Attribution (Module 3)
**Internal Implementation:** `modules/04_ais_trajectory/` and `modules/05_evidence_fusion/`

- **AIS Processing:** Historical Automatic Identification System (AIS) data is queried around the reconstructed time and space.
- **Evidence Dimensions:**
  - **Spatial Proximity:** How close the vessel was to the reconstructed origin centroid.
  - **Temporal Overlap:** Vessel presence during the reconstructed release time window.
  - **Trajectory Consistency:** Track continuity within the probable origin boundary.
  - **Drift Consistency:** Verification against the underlying particle simulations.
  - **Speed/Course Behavior:** Deviation from baseline transit behavior (e.g., slowing down).
- **Transparent Ranking:** Evidence is fused linearly via transparent weights, avoiding black-box ML for attribution.
- **Core Principles:**
  - Module 3 identifies investigative candidates, **NOT** legally guilty vessels.
  - AIS gaps mean reduced observability/evidence, not automatic guilt.
  - Vessel type is only a weak prior.
  - Proximity alone does not establish responsibility.
  - Evidence scores are transparent triage/ranking scores, not probabilities of guilt.
  - Historical ground truth is strictly for post-hoc validation, not leaked into the attribution input.
