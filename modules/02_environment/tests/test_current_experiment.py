"""
OceanTrace - controlled environmental movement test (Milestone 2)
===================================================================

Purpose
-------
A sanity test proving that environmental forcing (a single constant ocean
current, with wind forced to exactly zero) produces predictable, measurable
movement of a particle cloud seeded from the existing oil slick GeoJSON.

This is a SEPARATE script. It does not modify or import test_geojson.py or
test_openoil.py, and does not touch CMEMS, ERA5, AIS, attribution, backend,
frontend, database, or ML.

API note (read before editing)
-------------------------------
This script was written against opendrift==1.14.12, the version actually
installed in this environment (checked with `pip show opendrift`, not
assumed from memory or an older tutorial). Two things that differ from
older examples floating around online:

  * o.seed_from_geojson(geojson_str)  -- confirmed current signature,
    same call already used in test_geojson.py.
  * Reading trajectories back after o.run(): o.get_property(...) is
    OBSOLETE in this version (prints a deprecation warning telling you to
    use o.result.<var> instead). This script uses o.result, which is an
    xarray.Dataset with dims (trajectory, time), NOT get_property().

Current convention
-------------------
CURRENT_DIRECTION_DEG is a compass bearing (degrees clockwise from true
North) describing the direction the current flows TOWARDS (oceanographic
"set" convention), e.g. 0 = towards N, 90 = towards E, 45 = towards NE.
It is converted to the eastward/northward components OpenDrift expects via
environment:fallback:x_sea_water_velocity / y_sea_water_velocity.
"""
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules/02_environment/src'))
_injected_paths = True



import math
from pathlib import Path
from datetime import timedelta

import numpy as np
from opendrift.models.openoil import OpenOil

# ---------------------------------------------------------------------------
# Controlled experiment configuration
# ---------------------------------------------------------------------------

# Ocean current: constant magnitude and direction, isolated as the only
# forcing (see WIND below).
CURRENT_SPEED_MS = 0.5          # m/s
CURRENT_DIRECTION_DEG = 45.0    # compass bearing, flowing TOWARDS this direction (NE)

# Wind is forced to exactly zero so current is the only forcing acting on
# the particles.
WIND_X_MS = 0.0
WIND_Y_MS = 0.0

# Short, clearly defined simulation window.
TIME_STEP_SECONDS = 3600        # 1 hour
NUM_STEPS = 6                   # 6 x 1h = 6h total
SIM_DURATION = timedelta(seconds=TIME_STEP_SECONDS * NUM_STEPS)

GEOJSON_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "samples" / "synthetic_slick.geojson")
OUTPUT_PLOT = Path(__file__).parent / "test_current_displacement_output.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bearing_to_uv(speed_ms, bearing_deg):
    """(speed, compass bearing flow moves TOWARDS) -> (eastward u, northward v) m/s."""
    rad = math.radians(bearing_deg)
    u = speed_ms * math.sin(rad)
    v = speed_ms * math.cos(rad)
    return u, v


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in meters between two lon/lat points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    """Initial compass bearing from point 1 to point 2 (degrees from North)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 72)
    print("OceanTrace - controlled environmental movement test")
    print("=" * 72)

    u, v = bearing_to_uv(CURRENT_SPEED_MS, CURRENT_DIRECTION_DEG)
    print(f"Configured ocean current: {CURRENT_SPEED_MS} m/s towards "
          f"{CURRENT_DIRECTION_DEG} deg (compass bearing from N)")
    print(f"  -> x_sea_water_velocity (eastward)  = {u:.4f} m/s")
    print(f"  -> y_sea_water_velocity (northward) = {v:.4f} m/s")
    print(f"Wind forced to zero: x_wind={WIND_X_MS}, y_wind={WIND_Y_MS}")

    o = OpenOil(loglevel=20)

    o.set_config('environment:fallback:x_wind', WIND_X_MS)
    o.set_config('environment:fallback:y_wind', WIND_Y_MS)
    o.set_config('environment:fallback:x_sea_water_velocity', u)
    o.set_config('environment:fallback:y_sea_water_velocity', v)

    print(f"\nLoading slick polygon from {GEOJSON_PATH} ...")
    with open(GEOJSON_PATH, 'r') as f:
        geojson_str = f.read()
    o.seed_from_geojson(geojson_str)

    n_seeded = o.num_elements_scheduled()
    init_lon = np.array(o.elements_scheduled.lon, dtype=float)
    init_lat = np.array(o.elements_scheduled.lat, dtype=float)
    init_centroid_lon = float(init_lon.mean())
    init_centroid_lat = float(init_lat.mean())

    print(f"Seeded {n_seeded} particles from slick polygon.")
    print(f"Initial region: lon [{init_lon.min():.5f}, {init_lon.max():.5f}], "
          f"lat [{init_lat.min():.5f}, {init_lat.max():.5f}]")
    print(f"Initial centroid: lon={init_centroid_lon:.5f}, lat={init_centroid_lat:.5f}")

    print(f"\nRunning simulation: {NUM_STEPS} steps x {TIME_STEP_SECONDS}s "
          f"= {SIM_DURATION} ...")
    o.run(steps=NUM_STEPS, time_step=TIME_STEP_SECONDS,
          time_step_output=TIME_STEP_SECONDS)

    # o.get_property() is obsolete in the installed version (1.14.12) --
    # use o.result (xarray.Dataset, dims trajectory x time) instead.
    final_lon = o.result.lon.isel(time=-1).values.astype(float)
    final_lat = o.result.lat.isel(time=-1).values.astype(float)
    valid = ~np.isnan(final_lon) & ~np.isnan(final_lat)
    n_deactivated = int((~valid).sum())
    final_lon, final_lat = final_lon[valid], final_lat[valid]

    final_centroid_lon = float(final_lon.mean())
    final_centroid_lat = float(final_lat.mean())

    displacement_m = haversine_m(init_centroid_lon, init_centroid_lat,
                                  final_centroid_lon, final_centroid_lat)
    measured_bearing = bearing_deg(init_centroid_lon, init_centroid_lat,
                                    final_centroid_lon, final_centroid_lat)
    expected_displacement_m = CURRENT_SPEED_MS * SIM_DURATION.total_seconds()

    print(f"\nFinal region: lon [{final_lon.min():.5f}, {final_lon.max():.5f}], "
          f"lat [{final_lat.min():.5f}, {final_lat.max():.5f}]")
    print(f"Final centroid: lon={final_centroid_lon:.5f}, lat={final_centroid_lat:.5f}")
    if n_deactivated:
        print(f"NOTE: {n_deactivated} of {n_seeded} elements were deactivated "
              f"(e.g. stranded) before the end of the run and excluded from "
              f"the final centroid/displacement calculation.")

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"Configured current:    {CURRENT_SPEED_MS} m/s @ {CURRENT_DIRECTION_DEG:.1f} deg (towards, from N)")
    print(f"Wind:                  0 m/s (isolated)")
    print(f"Simulation duration:   {SIM_DURATION}")
    print(f"Timestep:              {TIME_STEP_SECONDS} s x {NUM_STEPS} steps")
    print(f"Particles seeded:      {n_seeded}  (valid at end: {len(final_lon)})")
    print(f"Initial region:        lon [{init_lon.min():.5f}, {init_lon.max():.5f}], "
          f"lat [{init_lat.min():.5f}, {init_lat.max():.5f}]")
    print(f"Final region:          lon [{final_lon.min():.5f}, {final_lon.max():.5f}], "
          f"lat [{final_lat.min():.5f}, {final_lat.max():.5f}]")
    print(f"Initial centroid:      ({init_centroid_lon:.5f}, {init_centroid_lat:.5f})")
    print(f"Final centroid:        ({final_centroid_lon:.5f}, {final_centroid_lat:.5f})")
    print(f"Measured displacement: {displacement_m:.1f} m")
    print(f"Measured bearing:      {measured_bearing:.1f} deg (from N)")
    print(f"Expected displacement: {expected_displacement_m:.1f} m (speed x duration, straight-line)")
    print(f"Expected bearing:      {CURRENT_DIRECTION_DEG:.1f} deg")
    print(f"Displacement error:    {abs(displacement_m - expected_displacement_m):.1f} m "
          f"({100 * abs(displacement_m - expected_displacement_m) / expected_displacement_m:.2f} %)")
    print(f"Bearing error:         {abs(measured_bearing - CURRENT_DIRECTION_DEG):.2f} deg")

    print(f"\nGenerating plot -> {OUTPUT_PLOT}")
    o.plot(filename=str(OUTPUT_PLOT), show=False)
    print("Done.")

    return {
        'n_seeded': n_seeded,
        'n_deactivated': n_deactivated,
        'init_centroid': (init_centroid_lon, init_centroid_lat),
        'final_centroid': (final_centroid_lon, final_centroid_lat),
        'displacement_m': displacement_m,
        'measured_bearing_deg': measured_bearing,
        'expected_displacement_m': expected_displacement_m,
    }


if __name__ == '__main__':
    run_test()