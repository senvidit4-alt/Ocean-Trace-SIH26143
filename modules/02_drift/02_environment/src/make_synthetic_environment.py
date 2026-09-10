"""
OceanTrace - synthetic environment dataset generator (Milestone 3)
======================================================================

Why this exists
----------------
This sandbox's network egress is restricted to package registries
(pypi/npm/github/crates/ubuntu) -- it cannot reach marine.copernicus.eu or
the Copernicus Climate Data Store, so a real CMEMS or ERA5 pull cannot be
executed here (see the milestone report for the exact real-world commands).

To still deliver and test an actual working environment-reader pipeline
today, this script builds a SMALL, CF-1.6-compliant NetCDF file that has
EXACTLY the structure a small CMEMS current subset + ERA5 wind subset would
have if merged into one file: a regional/time subset (not global, not
multi-day) covering the existing slick polygon with a modest margin, at the
slick's own timestamp.

This is explicitly a stand-in for realism testing, not a realistic ocean
state. Swapping it for a real CMEMS/ERA5-derived NetCDF requires no code
change anywhere else -- see environment_readers.build_netcdf_readers(),
which this file's output is designed to be loaded by.

Fields
------
  uo  (eastward_sea_water_velocity, m/s): 0.2 -> 0.5 m/s west-to-east shear
  vo  (northward_sea_water_velocity, m/s): constant 0.1 m/s
  u10 (eastward_wind, m/s): constant 2.0 m/s
  v10 (northward_wind, m/s): constant 0.0 m/s

The current has a spatial gradient (unlike Milestone 2's spatially-uniform
constant current) specifically so the resulting particle movement is
visibly non-rigid (the cloud fans out rather than translating as a block) --
proof the reader's gridded/interpolated data path, not just its fallback
scalar path, is what's driving the simulation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).parent
SLICK_GEOJSON = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "samples" / "synthetic_slick.geojson"
OUTPUT_NC = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "samples" / "synthetic_env.nc"

GRID_MARGIN_DEG = 0.2     # margin added around the slick bbox
GRID_RESOLUTION_DEG = 0.05  # ~5.5 km grid spacing
N_HOURS = 8               # hourly steps covering the planned sim duration + margin

CURRENT_MIN_MS = 0.2
CURRENT_MAX_MS = 0.5
CURRENT_V_MS = 0.1
WIND_U_MS = 2.0
WIND_V_MS = 0.0


def load_slick_bbox_and_time():
    with open(SLICK_GEOJSON) as f:
        gj = json.load(f)
    coords = gj['geometry']['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    time_str = gj['properties']['time']
    # Same parsing OpenDrift's seed_from_geojson uses: ISO8601 'Z' -> UTC, tz-naive
    from datetime import datetime
    t0 = datetime.fromisoformat(time_str.replace("Z", "+00:00")).replace(tzinfo=None)
    return min(lons), max(lons), min(lats), max(lats), t0


def build_dataset():
    lon_min, lon_max, lat_min, lat_max, t0 = load_slick_bbox_and_time()

    grid_lon_min = round(lon_min - GRID_MARGIN_DEG, 3)
    grid_lon_max = round(lon_max + GRID_MARGIN_DEG, 3)
    grid_lat_min = round(lat_min - GRID_MARGIN_DEG, 3)
    grid_lat_max = round(lat_max + GRID_MARGIN_DEG, 3)

    lon = np.arange(grid_lon_min, grid_lon_max + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG)
    lat = np.arange(grid_lat_min, grid_lat_max + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG)
    time = pd.date_range(t0, periods=N_HOURS, freq='1h')

    nt, nlat, nlon = len(time), len(lat), len(lon)

    # East-west current shear: 0.2 m/s at grid_lon_min -> 0.5 m/s at grid_lon_max,
    # broadcast across lat and time (time-invariant field -- see module docstring).
    lon_frac = (lon - grid_lon_min) / (grid_lon_max - grid_lon_min)
    uo_2d = CURRENT_MIN_MS + (CURRENT_MAX_MS - CURRENT_MIN_MS) * lon_frac  # (nlon,)
    uo = np.broadcast_to(uo_2d, (nt, nlat, nlon)).astype('float32')
    vo = np.full((nt, nlat, nlon), CURRENT_V_MS, dtype='float32')
    u10 = np.full((nt, nlat, nlon), WIND_U_MS, dtype='float32')
    v10 = np.full((nt, nlat, nlon), WIND_V_MS, dtype='float32')

    ds = xr.Dataset(
        data_vars={
            'uo': (('time', 'latitude', 'longitude'), uo,
                   {'standard_name': 'eastward_sea_water_velocity', 'units': 'm s-1',
                    'long_name': 'Eastward sea water velocity (synthetic, CMEMS-like)'}),
            'vo': (('time', 'latitude', 'longitude'), vo,
                   {'standard_name': 'northward_sea_water_velocity', 'units': 'm s-1',
                    'long_name': 'Northward sea water velocity (synthetic, CMEMS-like)'}),
            'u10': (('time', 'latitude', 'longitude'), u10,
                    {'standard_name': 'eastward_wind', 'units': 'm s-1',
                     'long_name': '10m eastward wind (synthetic, ERA5-like)'}),
            'v10': (('time', 'latitude', 'longitude'), v10,
                    {'standard_name': 'northward_wind', 'units': 'm s-1',
                     'long_name': '10m northward wind (synthetic, ERA5-like)'}),
        },
        coords={
            'time': time,
            'latitude': ('latitude', lat, {'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'}),
            'longitude': ('longitude', lon, {'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'}),
        },
        attrs={
            'Conventions': 'CF-1.6',
            'title': 'OceanTrace synthetic regional/time subset (stand-in for CMEMS+ERA5)',
            'summary': ('Synthetic small regional/time-box current+wind field used to validate '
                        'the OpenDrift NetCDF/CF reader pipeline in an environment without '
                        'network access to CMEMS/ERA5. NOT real observational or model data.'),
        },
    )
    return ds, (grid_lon_min, grid_lon_max, grid_lat_min, grid_lat_max), t0


def main():
    OUTPUT_NC.parent.mkdir(parents=True, exist_ok=True)
    ds, bbox, t0 = build_dataset()

    encoding = {
        'time': {'units': f'hours since {t0.isoformat()}', 'calendar': 'standard'},
    }
    ds.to_netcdf(OUTPUT_NC, encoding=encoding)

    size_kb = OUTPUT_NC.stat().st_size / 1024
    print(f"Wrote {OUTPUT_NC} ({size_kb:.1f} KB)")
    print(f"Grid bbox: lon [{bbox[0]}, {bbox[1]}], lat [{bbox[2]}, {bbox[3]}]  "
          f"({len(ds.longitude)} x {len(ds.latitude)} points)")
    print(f"Time coverage: {ds.time.values[0]} .. {ds.time.values[-1]}  ({len(ds.time)} hourly steps)")
    print(f"Current (uo): {CURRENT_MIN_MS}-{CURRENT_MAX_MS} m/s eastward shear, "
          f"{CURRENT_V_MS} m/s northward (constant)")
    print(f"Wind (u10/v10): {WIND_U_MS} m/s eastward, {WIND_V_MS} m/s northward (constant)")


if __name__ == '__main__':
    main()
