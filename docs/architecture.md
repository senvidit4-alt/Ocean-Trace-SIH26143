# OceanTrace Architecture Reference

OceanTrace is built on a **three-module functional architecture**, designed to separate concerns between detection, environmental modeling, and vessel attribution.

## Core Architectural Principle
The three product modules are strictly independent. They communicate through explicit, serialized JSON data contracts rather than importing each other's internal Python implementations. This ensures that:
- **Module 2 (Drift)** can be executed using a manually defined JSON detection polygon, bypassing the need for a live, trained U-Net model.
- **Module 3 (Attribution)** can process a drift reconstruction output without requiring execution of the upstream modules.

## Three-Module Data Flow

```text
Sentinel-1 SAR
      |
      v
┌──────────────────────────┐
│ MODULE 1 — DETECTION     │
│ Owner: Techie 2          │
└────────────┬─────────────┘
             |
       detection.json
             |
             v
┌──────────────────────────┐
│ MODULE 2 — DRIFT         │
│ Owner: Techie 3          │
└────────────┬─────────────┘
             |
   reconstruction.json
   + trajectory.nc artifact
             |
             v
┌──────────────────────────┐
│ MODULE 3 — ATTRIBUTION   │
│ Owner: Techie 3          │
└────────────┬─────────────┘
             |
             v
 attribution.json
 (Ranked vessels + evidence + report)
```

## Internal Implementation Mapping

The OceanTrace codebase separates the three functional product modules into **five internal implementation components** (directories). These five directories are *implementation details*, not the product architecture itself.

| Product Module | Responsibilities | Internal Implementation Component(s) |
| :--- | :--- | :--- |
| **Module 1 — Detection** | Detect oil slick, distinguish from look-alikes, produce slick geometry/metadata. | `modules/01_detection/` |
| **Module 2 — Drift** | Backward Lagrangian drift reconstruction, origin/time estimation, uncertainty quantification. | `modules/02_drift/02_environment/`<br>`modules/02_drift/03_source_reconstruction/` |
| **Module 3 — Attribution** | Correlate vessel trajectories with drift model, evaluate spatial/temporal/behavioral evidence, rank candidate vessels. | `modules/03_attribution/04_ais_trajectory/`<br>`modules/03_attribution/05_evidence_fusion/` |

## Data Contracts
See `contracts/CONTRACTS.md` for full schema documentation.
- `detection.json`: Binds Module 1 to Module 2.
- `reconstruction.json`: Binds Module 2 to Module 3 (requires accompanying NetCDF artifact for trajectories).
- `attribution.json`: Final product output.
