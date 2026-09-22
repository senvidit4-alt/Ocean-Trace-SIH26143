# OceanTrace Data Directory

This directory is intended for large environmental datasets, AIS archives, and SAR imagery required for OceanTrace.

**Important:** Large datasets (e.g., CMEMS .nc, ERA5 .nc, Spire AIS .csv, model weights) are intentionally excluded from version control to keep the repository lightweight.

**Expected Data Sources:**
- Environment: Copernicus Marine Service (CMEMS) ocean currents, ECMWF ERA5 wind data.
-  REAL AIS: Historical terrestrial and satellite AIS data (e.g., Spire).
- SAR: Processed Sentinel-1 imagery or oil spill detection masks.

Store your large data files here. See the `samples/` directory for synthetic/test data schemas if applicable.

Note: synthetic_slick.geojson in the samples folder is synthetic test data, and is NOT the real Bay Marchand validation input. It is used strictly for testing the pipeline.
