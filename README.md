# OceanTrace (Trace-Oceans)

OceanTrace is a modular pipeline for oil spill attribution, combining satellite remote sensing, drift modeling, and vessel tracking to identify probable sources of marine pollution.

## High-Level Pipeline

1. **Satellite oil-spill detection**: Deep learning segmentation (Module 01) detects oil slicks from SAR imagery.
2. **Environmental data**: Integration of CMEMS ocean currents and ERA5 wind data (Module 02).
3. **Source reconstruction**: OpenDrift-based backward propagation to estimate the origin of the spill (Module 03).
4. **AIS trajectory analysis**: Parsing and contextualizing historical vessel movements (Module 04).
5. **Evidence fusion**: Scoring vessel tracks against the simulated particle cloud to generate an investigative candidate ranking (Module 05).

*Disclaimer: This analysis identifies probable source regions, release-time windows, and candidate vessels for investigative priority. It does not identify legally responsible vessels.*

## Modules

- [Module 01: Oil Spill Detection](modules/01_oil_spill_detection/README.md)
- [Module 02: Environment](modules/02_environment/README.md)
- [Module 03: Source Reconstruction](modules/03_source_reconstruction/README.md)
- [Module 04: AIS Trajectory](modules/04_ais_trajectory/README.md)
- [Module 05: Evidence Fusion](modules/05_evidence_fusion/README.md)
