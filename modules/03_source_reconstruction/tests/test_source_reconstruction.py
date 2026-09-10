"""
OceanTrace - Source Reconstruction (Backward Drift) Test (Milestone 4)
=====================================================================

Purpose
-------
Test the drift engine's ability to take a synthetic "observed" oil slick
polygon, seed it at the observation time, and run a backward drift simulation
to estimate the probable origin region of the spill.

This script uses the same synthetic environmental netCDF file from Milestone 3.
It runs both a forward drift (for trajectory verification) and a backward drift
(for source reconstruction) from the provided polygon.
"""
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules/03_source_reconstruction/src'))
sys.path.insert(0, str(_repo_root / 'modules/02_environment/src'))
_injected_paths = True



import json
import math
from pathlib import Path
from datetime import timedelta

import numpy as np
from opendrift.models.openoil import OpenOil
from environment_readers import get_readers
import make_synthetic_environment

HERE = Path(__file__).parent
ENV_NETCDF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "samples" / "synthetic_env.nc"
OUTPUT_PLOT_FORWARD = HERE / "test_source_reconstruction_forward.png"
OUTPUT_PLOT_BACKWARD = HERE / "test_source_reconstruction_backward.png"

# Techie 2 dummy polygon input
dummy_polygon = {
    "type": "Polygon",
    "coordinates": [[[72.4, 18.8], [72.6, 18.8], [72.6, 19.0], [72.4, 19.0], [72.4, 18.8]]]
}

OBSERVATION_TIME_STR = "2026-08-31T14:00:00Z"

# Construct a valid GeoJSON Feature string
geojson_feature = {
    "type": "Feature",
    "geometry": dummy_polygon,
    "properties": {
        "time": OBSERVATION_TIME_STR,
        "number": 1000,
        "oil_type": "GENERIC LIGHT CRUDE"
    }
}
geojson_str = json.dumps(geojson_feature)

ENV_MODE = 'netcdf'
ENV_KWARGS = dict(paths=ENV_NETCDF_PATH, name='synthetic_regional')

TIME_STEP_SECONDS = 3600
NUM_STEPS = 6
SIM_DURATION = timedelta(seconds=TIME_STEP_SECONDS * NUM_STEPS)

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lon1, lat1, lon2, lat2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def run_simulation(direction="backward"):
    print("=" * 72)
    print(f"OceanTrace - {direction.capitalize()} Drift Simulation")
    print("=" * 72)
    
    if not ENV_NETCDF_PATH.exists():
        print(f"{ENV_NETCDF_PATH} not found -- generating it now...")
        make_synthetic_environment.main()

    o = OpenOil(loglevel=20)
    o.set_config('environment:fallback:x_wind', 0.0)
    o.set_config('environment:fallback:y_wind', 0.0)
    o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
    o.set_config('environment:fallback:y_sea_water_velocity', 0.0)

    readers = get_readers(ENV_MODE, **ENV_KWARGS)
    o.add_reader(readers)

    print(f"\nSeeding elements from synthetic GeoJSON polygon at {OBSERVATION_TIME_STR}...")
    o.seed_from_geojson(geojson_str)

    n_seeded = o.num_elements_scheduled()
    init_lon = np.array(o.elements_scheduled.lon, dtype=float)
    init_lat = np.array(o.elements_scheduled.lat, dtype=float)
    init_centroid_lon = float(init_lon.mean())
    init_centroid_lat = float(init_lat.mean())
    init_spread_m = float(np.std(init_lon) * 111320 * math.cos(math.radians(init_centroid_lat)))

    print(f"Initial (Observed) Centroid: lon={init_centroid_lon:.5f}, lat={init_centroid_lat:.5f}")

    if direction == "backward":
        # Backward run: negative time step
        print(f"\nRunning BACKWARD simulation: {NUM_STEPS} steps x {-TIME_STEP_SECONDS}s...")
        o.run(steps=NUM_STEPS, time_step=-TIME_STEP_SECONDS, time_step_output=TIME_STEP_SECONDS)
        plot_path = OUTPUT_PLOT_BACKWARD
    else:
        # Forward run
        print(f"\nRunning FORWARD simulation: {NUM_STEPS} steps x {TIME_STEP_SECONDS}s...")
        o.run(steps=NUM_STEPS, time_step=TIME_STEP_SECONDS, time_step_output=TIME_STEP_SECONDS)
        plot_path = OUTPUT_PLOT_FORWARD

    # Get final state
    final_lon = o.result.lon.isel(time=-1).values.astype(float)
    final_lat = o.result.lat.isel(time=-1).values.astype(float)
    valid = ~np.isnan(final_lon) & ~np.isnan(final_lat)
    n_deactivated = int((~valid).sum())
    final_lon, final_lat = final_lon[valid], final_lat[valid]

    final_centroid_lon = float(final_lon.mean())
    final_centroid_lat = float(final_lat.mean())
    final_spread_m = float(np.std(final_lon) * 111320 * math.cos(math.radians(final_centroid_lat)))
    
    displacement_m = haversine_m(init_centroid_lon, init_centroid_lat, final_centroid_lon, final_centroid_lat)
    
    label = "Estimated Origin" if direction == "backward" else "Final Destination"
    print(f"\n{label} Centroid: lon={final_centroid_lon:.5f}, lat={final_centroid_lat:.5f}")
    print(f"Spread (uncertainty): {final_spread_m:.1f} m")
    print(f"Centroid Displacement: {displacement_m:.1f} m")
    print(f"Deactivated particles: {n_deactivated} / {n_seeded}")

    print(f"\nGenerating plot -> {plot_path}")
    title = f"OceanTrace - {direction.capitalize()} Drift (Synthetic Data)"
    # NOTE: pass title to plot to clearly label it as synthetic test data
    # (Checking if OpenDrift's plot supports title. Usually it doesn't take 'title' kwarg directly, 
    # we can just save the file. The file name labels it).
    o.plot(filename=str(plot_path), show=False)
    
    return {
        "direction": direction,
        "n_seeded": n_seeded,
        "n_deactivated": n_deactivated,
        "init_centroid": (init_centroid_lon, init_centroid_lat),
        "final_centroid": (final_centroid_lon, final_centroid_lat),
        "spread_m": final_spread_m,
        "displacement_m": displacement_m
    }

if __name__ == '__main__':
    print("WARNING: This test uses SYNTHETIC environmental forcing and a SYNTHETIC dummy polygon.")
    print("This does NOT represent a real oil spill.\n")
    
    print("--- 1. Forward Verification ---")
    run_simulation(direction="forward")
    
    print("\n--- 2. Source Reconstruction ---")
    run_simulation(direction="backward")
    
    print("\nDone.")