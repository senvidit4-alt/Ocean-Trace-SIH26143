# OceanTrace JSON Data Contracts

This directory contains the definitions and adapters for the JSON data contracts passed between the three major system modules.

## Architecture Data Flow
```text
[Product Module 1: Detection] 
       ↓ (detection.json)
[Product Module 2: Drift] 
       ↓ (reconstruction.json + trajectories.nc)
[Product Module 3: Attribution] 
       ↓ (attribution.json)
[Reporting / UI]
```

## Contracts Overview

### 1. `detection.json`
**Produced By:** Product Module 1 (Detection)
**Consumed By:** Product Module 2 (Drift)

**Fields:**
- `contract_version` (string, REQUIRED): "1.0"
- `observation_time` (string, REQUIRED): ISO-8601 timestamp (e.g. "2026-08-31T14:00:00Z").
- `polygon` (object, REQUIRED): GeoJSON-compatible geometry dictionary (typically a `Polygon`).
- `area_km2` (float, OPTIONAL): Calculated surface area of the detected slick.
- `confidence` (float, OPTIONAL): Model detection confidence/probability (0.0 to 1.0).
- `age_hrs` (float, OPTIONAL): Estimated age in hours. Not currently produced by Module 1 (set as heuristic); consumers should not rely on this.
- `metadata` (object, OPTIONAL): Additional acquisition metadata (e.g., satellite platform).

**Note for Integration Testing:** 
Module 1 can be replaced entirely by a manually created mock `detection.json` (or `synthetic_slick.geojson`) to run the downstream pipeline without requiring the trained U-Net model.

### 2. `reconstruction.json`
**Produced By:** Product Module 2 (Drift)
**Consumed By:** Product Module 3 (Attribution)

**Fields:**
- `contract_version` (string, REQUIRED): "1.0"
- `observation_time` (string, REQUIRED): ISO-8601 timestamp.
- `origin_time` (string, REQUIRED): ISO-8601 timestamp of the estimated release.
- `search_window_hours` (float, REQUIRED): Duration of the backward simulation in hours.
- `origin_centroid` (array, REQUIRED): `[lon, lat]` of the probable origin.
- `origin_positions` (object, REQUIRED): Contains `lon` and `lat` lists for the final particle cloud.
- `uncertainty` (object, REQUIRED):
  - `semi_major_m` (float)
  - `semi_minor_m` (float)
  - `orientation_deg` (float)
  - `radius_68_m` (float)
  - `radius_95_m` (float)
  - `method` (string)
- `coverage_warnings` (array of strings, OPTIONAL): Environment/fallback warnings.
- `trajectory_file_path` (string, REQUIRED): A relative path from this JSON file to the `.nc` (NetCDF) file containing the full OpenDrift xarray particle trajectories (required for attribution drift consistency).

### 3. `attribution.json`
**Produced By:** Product Module 3 (Attribution)
**Consumed By:** End User / UI / Report Generators

**Fields:**
- `contract_version` (string, REQUIRED): "1.0"
- `analysis_metadata` (object, REQUIRED):
  - `search_radius_km` (float)
  - `search_time_window` (array of 2 ISO-8601 strings)
  - `n_vessels_considered` (integer)
  - `caveats` (array of strings)
- `candidates` (array of objects, REQUIRED): Ranked investigative candidates.
  - `mmsi` (string)
  - `vessel_name` (string or null)
  - `category` (string, e.g., "High-Priority Candidate")
  - `score` (float, 0.0 to 1.0)
  - `features` (array of objects):
    - `name` (string)
    - `value` (float)
    - `raw_value` (float)
    - `raw_unit` (string)
    - `contribution` (float)
    - `explanation` (string)
    - `limitation` (string or null)
  - `data_confidence` (object):
    - `coverage_fraction` (float)
    - `note` (string)

**Important Disclaimer:** The candidate score is an investigative prioritization score, NOT a probability of guilt or legal responsibility.
