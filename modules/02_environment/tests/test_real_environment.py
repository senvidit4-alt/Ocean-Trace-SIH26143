"""
OceanTrace - real-environment-data test (Milestone 3)
==========================================================

Purpose
-------
Same slick.geojson and particle count as Milestones 1-2, but the drift
engine below is now provider-agnostic: it asks environment_readers.get_readers()
for a list of Reader objects and adds them to OpenOil with o.add_reader(),
without knowing or caring whether they came from a local synthetic file, a
CMEMS subset, or an ERA5 subset. Today env_mode='netcdf' points at the
small regional/time-subset file built by make_synthetic_environment.py
(see that file and the milestone report for why real CMEMS/ERA5 could not
be pulled live in this sandbox). Swapping to real data later is a one-line
change to ENV_MODE/ENV_KWARGS below -- nothing else in this script changes.

This is a SEPARATE script. It does not modify test_geojson.py,
test_openoil.py, or test_current_displacement.py (Milestone 2's controlled
constant-current test remains exactly as it was and keeps working).

No AIS, backward drift, attribution, frontend, or backend here.
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

from environment_readers import get_readers
import make_synthetic_environment  # ensures data/synthetic_env.nc exists / is regenerated

HERE = Path(__file__).parent
GEOJSON_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "samples" / "synthetic_slick.geojson"
ENV_NETCDF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "samples" / "synthetic_env.nc"
OUTPUT_PLOT = HERE / "test_real_environment_output.png"

# --- provider selection: only these two lines change to point at real data ---
ENV_MODE = 'netcdf'
ENV_KWARGS = dict(paths=ENV_NETCDF_PATH, name='synthetic_regional')
# Real CMEMS example (not runnable in this sandbox -- see report):
#   ENV_MODE = 'cmems'
#   ENV_KWARGS = dict(dataset_id='cmems_mod_glo_phy_anfc_0.083deg_PT1H-m',
#                      username=..., password=...)
# -------------------------------------------------------------------------

# Same run window as Milestone 2, for direct before/after comparison.
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


def run_test():
    print("=" * 72)
    print("OceanTrace - real-environment-data test (Milestone 3)")
    print("=" * 72)

    if not ENV_NETCDF_PATH.exists():
        print(f"{ENV_NETCDF_PATH} not found -- generating it now...")
        make_synthetic_environment.main()

    o = OpenOil(loglevel=20)

    # Fallback safety net: if a particle ever steps outside the reader's
    # spatial/temporal coverage, use 0 rather than NaN/crash. With this
    # dataset's 0.2 deg margin and a 6h/~10km max drift, this should not
    # actually be hit -- it's defensive, not load-bearing.
    o.set_config('environment:fallback:x_wind', 0.0)
    o.set_config('environment:fallback:y_wind', 0.0)
    o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
    o.set_config('environment:fallback:y_sea_water_velocity', 0.0)

    print(f"\nBuilding readers: env_mode='{ENV_MODE}', kwargs={ {k: str(v) for k, v in ENV_KWARGS.items()} }")
    readers = get_readers(ENV_MODE, **ENV_KWARGS)
    for r in readers:
        print(f"  reader: {r.name}  coverage lon[{r.xmin:.3f},{r.xmax:.3f}] "
              f"lat[{r.ymin:.3f},{r.ymax:.3f}]  time[{r.start_time},{r.end_time}]")
    o.add_reader(readers)

    print(f"\nLoading slick polygon from {GEOJSON_PATH} ...")
    with open(GEOJSON_PATH, 'r') as f:
        geojson_str = f.read()
    o.seed_from_geojson(geojson_str)

    n_seeded = o.num_elements_scheduled()
    init_lon = np.array(o.elements_scheduled.lon, dtype=float)
    init_lat = np.array(o.elements_scheduled.lat, dtype=float)
    init_centroid_lon = float(init_lon.mean())
    init_centroid_lat = float(init_lat.mean())
    init_spread_m = float(np.std(init_lon) * 111320 * math.cos(math.radians(init_centroid_lat)))

    print(f"Seeded {n_seeded} particles from slick polygon.")
    print(f"Initial region: lon [{init_lon.min():.5f}, {init_lon.max():.5f}], "
          f"lat [{init_lat.min():.5f}, {init_lat.max():.5f}]")
    print(f"Initial centroid: lon={init_centroid_lon:.5f}, lat={init_centroid_lat:.5f}")

    print(f"\nRunning simulation: {NUM_STEPS} steps x {TIME_STEP_SECONDS}s "
          f"= {SIM_DURATION} ...")
    o.run(steps=NUM_STEPS, time_step=TIME_STEP_SECONDS,
          time_step_output=TIME_STEP_SECONDS)

    final_lon = o.result.lon.isel(time=-1).values.astype(float)
    final_lat = o.result.lat.isel(time=-1).values.astype(float)
    valid = ~np.isnan(final_lon) & ~np.isnan(final_lat)
    n_deactivated = int((~valid).sum())
    final_lon, final_lat = final_lon[valid], final_lat[valid]

    final_centroid_lon = float(final_lon.mean())
    final_centroid_lat = float(final_lat.mean())
    final_spread_m = float(np.std(final_lon) * 111320 * math.cos(math.radians(final_centroid_lat)))

    displacement_m = haversine_m(init_centroid_lon, init_centroid_lat,
                                  final_centroid_lon, final_centroid_lat)
    measured_bearing = bearing_deg(init_centroid_lon, init_centroid_lat,
                                    final_centroid_lon, final_centroid_lat)

    print(f"\nFinal region: lon [{final_lon.min():.5f}, {final_lon.max():.5f}], "
          f"lat [{final_lat.min():.5f}, {final_lat.max():.5f}]")
    print(f"Final centroid: lon={final_centroid_lon:.5f}, lat={final_centroid_lat:.5f}")
    if n_deactivated:
        print(f"NOTE: {n_deactivated} of {n_seeded} elements deactivated before run end "
              f"and excluded from final centroid/displacement.")

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"Environment source:     env_mode='{ENV_MODE}' ({len(readers)} reader(s))")
    print(f"Simulation duration:    {SIM_DURATION}")
    print(f"Timestep:               {TIME_STEP_SECONDS} s x {NUM_STEPS} steps")
    print(f"Particles seeded:       {n_seeded}  (valid at end: {len(final_lon)})")
    print(f"Initial region:         lon [{init_lon.min():.5f}, {init_lon.max():.5f}], "
          f"lat [{init_lat.min():.5f}, {init_lat.max():.5f}]")
    print(f"Final region:           lon [{final_lon.min():.5f}, {final_lon.max():.5f}], "
          f"lat [{final_lat.min():.5f}, {final_lat.max():.5f}]")
    print(f"Initial centroid:       ({init_centroid_lon:.5f}, {init_centroid_lat:.5f})")
    print(f"Final centroid:         ({final_centroid_lon:.5f}, {final_centroid_lat:.5f})")
    print(f"Centroid displacement:  {displacement_m:.1f} m")
    print(f"Centroid bearing:       {measured_bearing:.1f} deg (from N)")
    print(f"Longitudinal spread (std, m):  initial={init_spread_m:.1f}  final={final_spread_m:.1f}  "
          f"(growth = non-rigid/'fanning' movement from the current's spatial shear)")

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
        'init_spread_m': init_spread_m,
        'final_spread_m': final_spread_m,
    }


if __name__ == '__main__':
    run_test()