"""
OceanTrace - test for source_reconstruction.py (Milestone 5)
==================================================================

Uses:
  - the Techie 2 dummy polygon from Milestone 4's example (bare geometry,
    no GeoJSON Feature wrapper, no embedded time property)
  - the existing synthetic environment (data/synthetic_env.nc, built in
    Milestone 3 -- regenerated if missing)
  - observation_time = 2026-08-31T14:00:00Z
  - search_window_hours = 6
  - number = 1000

This is a SEPARATE script. It does not modify test_geojson.py,
test_openoil.py, test_current_displacement.py, or test_real_environment.py.
"""
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules/03_source_reconstruction/src'))
sys.path.insert(0, str(_repo_root / 'modules/02_environment/src'))
_injected_paths = True



import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from source_reconstruction import reconstruct_source, SourceReconstruction, Uncertainty
import make_synthetic_environment

HERE = Path(__file__).parent
ENV_NETCDF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "samples" / "synthetic_env.nc"
OUTPUT_PLOT = HERE / "test_source_reconstruction_output.png"

# Techie 2's dummy polygon, exactly as given -- bare geometry dict, no
# Feature wrapper, no embedded time property.
dummy_polygon = {
    "type": "Polygon",
    "coordinates": [[[72.4, 18.8], [72.6, 18.8], [72.6, 19.0], [72.4, 19.0], [72.4, 18.8]]]
}

OBSERVATION_TIME = datetime(2026, 8, 31, 14, 0, 0)  # naive UTC, matching opendrift's own convention
SEARCH_WINDOW_HOURS = 6.0
NUMBER = 1000


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


def run_test():
    print("=" * 72)
    print("OceanTrace - source_reconstruction module test (Milestone 5)")
    print("=" * 72)

    if not ENV_NETCDF_PATH.exists():
        print(f"{ENV_NETCDF_PATH} not found -- generating it now...")
        make_synthetic_environment.main()

    print(f"\nRunning reconstruct_source(observation_time={OBSERVATION_TIME}, "
          f"search_window_hours={SEARCH_WINDOW_HOURS}, number={NUMBER}) ...")

    result = reconstruct_source(
        polygon=dummy_polygon,
        observation_time=OBSERVATION_TIME,
        search_window_hours=SEARCH_WINDOW_HOURS,
        number=NUMBER,
        env_mode='netcdf',
        env_kwargs=dict(paths=ENV_NETCDF_PATH, name='synthetic_regional'),
    )

    print("\n" + "-" * 72)
    print("VERIFICATION")
    print("-" * 72)

    # 1. Returned object type
    check("result is a SourceReconstruction instance", isinstance(result, SourceReconstruction))
    check("result.uncertainty is an Uncertainty instance", isinstance(result.uncertainty, Uncertainty))

    # 2. Origin centroid
    lon_c, lat_c = result.origin_centroid
    check("origin_centroid is a finite (lon, lat) pair",
          np.isfinite(lon_c) and np.isfinite(lat_c))
    # sanity range check -- centroid should be in the broad vicinity of the slick
    # (within the synthetic grid's bbox + a margin), not some wildly wrong value
    check("origin_centroid lon within plausible range",
          70.0 < lon_c < 75.0)
    check("origin_centroid lat within plausible range",
          17.0 < lat_c < 21.0)

    # 3. Uncertainty
    u = result.uncertainty
    check("uncertainty.semi_major_m >= semi_minor_m >= 0",
          u.semi_major_m >= u.semi_minor_m >= 0)
    check("uncertainty.radius_68_m <= radius_95_m",
          u.radius_68_m <= u.radius_95_m)
    check("uncertainty values are finite",
          all(np.isfinite([u.semi_major_m, u.semi_minor_m, u.radius_68_m, u.radius_95_m])))
    check("uncertainty method is 'covariance_ellipse' (1000 particles, non-degenerate cloud)",
          u.method == 'covariance_ellipse')

    # 4. Origin time / 5. Observation time
    check("origin_time == observation_time - search_window_hours",
          result.origin_time == result.observation_time - __import__('datetime').timedelta(
              hours=result.search_window_hours))
    check("observation_time matches requested time",
          result.observation_time == OBSERVATION_TIME)
    check("origin_time is exactly 6h before observation_time",
          (result.observation_time - result.origin_time).total_seconds() == 6 * 3600)

    # 6. Particle counts
    check("n_particles_seeded == 1000", result.n_particles_seeded == NUMBER)
    check("n_particles_valid == n_particles_seeded (no elements deactivated in open water)",
          result.n_particles_valid == result.n_particles_seeded)
    check("origin_positions arrays match n_particles_valid",
          len(result.origin_positions['lon']) == result.n_particles_valid
          and len(result.origin_positions['lat']) == result.n_particles_valid)

    # 7. Trajectory dimensions
    traj = result.trajectories
    check("trajectories has 'trajectory' and 'time' dims",
          'trajectory' in traj.lon.dims and 'time' in traj.lon.dims)
    check("trajectories trajectory-dim size == n_particles_seeded",
          traj.lon.sizes['trajectory'] == result.n_particles_seeded)
    expected_time_steps = round(SEARCH_WINDOW_HOURS) + 1  # steps + initial point
    check(f"trajectories time-dim size == {expected_time_steps} (steps+1)",
          traj.lon.sizes['time'] == expected_time_steps)

    # 8. No NaN/invalid origin positions
    check("no NaN in origin_positions['lon']", not np.isnan(result.origin_positions['lon']).any())
    check("no NaN in origin_positions['lat']", not np.isnan(result.origin_positions['lat']).any())

    # 9. Synthetic environment actually used (not silently all-fallback)
    check("env_mode echoed back is 'netcdf'", result.env_mode == 'netcdf')
    check("no coverage warnings (fully inside synthetic grid's space/time domain)",
          result.coverage_warnings == [])
    obs_centroid_lon = float(np.mean(traj.lon.isel(time=0).values))
    obs_centroid_lat = float(np.mean(traj.lat.isel(time=0).values))
    displacement_m = haversine_m(obs_centroid_lon, obs_centroid_lat, lon_c, lat_c)
    check(f"origin centroid displaced >100m from observed centroid "
          f"(got {displacement_m:.1f}m -- proves real gridded forcing moved the "
          f"particles; all-fallback/all-zero forcing would leave this ~0m)",
          displacement_m > 100)

    print("\nAll checks passed.")

    print("\n" + "-" * 72)
    print("SUMMARY")
    print("-" * 72)
    print(f"Observation time:      {result.observation_time}")
    print(f"Origin time:           {result.origin_time}")
    print(f"Search window:         {result.search_window_hours} h")
    print(f"Particles:             seeded={result.n_particles_seeded}  valid={result.n_particles_valid}")
    print(f"Observed centroid:     ({obs_centroid_lon:.5f}, {obs_centroid_lat:.5f})")
    print(f"Origin centroid:       ({lon_c:.5f}, {lat_c:.5f})")
    print(f"Centroid displacement: {displacement_m:.1f} m")
    print(f"Uncertainty (1-sigma ellipse): semi-major={u.semi_major_m:.1f}m, "
          f"semi-minor={u.semi_minor_m:.1f}m, orientation={u.orientation_deg:.1f} deg")
    print(f"Uncertainty (radii):    68%={u.radius_68_m:.1f}m, 95%={u.radius_95_m:.1f}m")
    print(f"Uncertainty method:     {u.method}")
    print(f"Coverage warnings:      {result.coverage_warnings if result.coverage_warnings else '(none)'}")
    print(f"trajectories dims:      {dict(traj.lon.sizes)}")

    make_plot(result, obs_centroid_lon, obs_centroid_lat)
    print(f"\nPlot written to {OUTPUT_PLOT}")

    return result


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def make_plot(result: SourceReconstruction, obs_centroid_lon, obs_centroid_lat):
    lon_c, lat_c = result.origin_centroid
    u = result.uncertainty
    traj = result.trajectories

    fig, ax = plt.subplots(figsize=(9, 8))

    # Backward trajectories for a readable subset of particles
    n_show = min(150, traj.lon.sizes['trajectory'])
    idx = np.linspace(0, traj.lon.sizes['trajectory'] - 1, n_show).astype(int)
    for i in idx:
        ax.plot(traj.lon.isel(trajectory=i).values, traj.lat.isel(trajectory=i).values,
                color='gray', linewidth=0.4, alpha=0.5, zorder=1)

    # Observed (t=0) polygon footprint
    obs_lon = traj.lon.isel(time=0).values
    obs_lat = traj.lat.isel(time=0).values
    ax.scatter(obs_lon, obs_lat, s=8, color='green', label='observed slick (t=obs)', zorder=2)

    # Origin cloud (t=-1, verified index)
    origin_lon = result.origin_positions['lon']
    origin_lat = result.origin_positions['lat']
    ax.scatter(origin_lon, origin_lat, s=8, color='blue', label='origin cloud (backward result)', zorder=3)

    # Centroid
    ax.scatter([lon_c], [lat_c], s=140, color='red', marker='*',
               edgecolor='black', linewidth=0.7, label='origin centroid', zorder=5)

    # Uncertainty ellipse (1-sigma), converted from local meters back to degrees
    if u.method == 'covariance_ellipse':
        width_deg = 2 * u.semi_minor_m / (111320.0 * math.cos(math.radians(lat_c))) * 2  # 2-sigma, east-west (minor)
        height_deg = 2 * u.semi_major_m / 111320.0 * 2  # 2-sigma, north-south (major)
        # angle: orientation_deg is bearing of MAJOR axis from North; matplotlib
        # Ellipse 'angle' is measured counter-clockwise from the x-axis (East).
        mpl_angle = 90 - u.orientation_deg
        ell = Ellipse((lon_c, lat_c), width=height_deg, height=width_deg,
                      angle=mpl_angle, edgecolor='red', facecolor='none',
                      linewidth=1.5, linestyle='--', label='uncertainty ellipse (2-sigma)', zorder=4)
        ax.add_patch(ell)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f"OceanTrace Milestone 5 - source reconstruction\n"
                 f"obs={result.observation_time} -> origin={result.origin_time} "
                 f"({result.search_window_hours:.0f}h backward)")
    ax.set_aspect(1 / math.cos(math.radians(lat_c)))
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    run_test()