"""
OceanTrace - source reconstruction module (Milestone 5)
============================================================

observed slick -> probable source region + time window + uncertainty + trajectories

This is a reusable LIBRARY module (not a test_*.py script). It is imported by
whatever needs a source estimate next (a future attribution module, a test
script, eventually a backend) and does not modify or import from any of the
Milestone 1-4 test scripts.

Reused unchanged from the existing project:
  - environment_readers.get_readers()  -- same provider-agnostic reader
    factory used in Milestone 3. This module never imports copernicusmarine,
    cdsapi, or netCDF4 directly -- only environment_readers.py does.

Verified against the INSTALLED opendrift==1.14.12 (introspected directly,
not assumed -- see comments at each call site):

  1. Backward runs: o.run(steps=N, time_step=-dt, time_step_output=dt).
     basemodel starts the run at elements_scheduled_time.max() (i.e. the
     seeding time) and auto-flips a positive time_step_output to match a
     negative time_step.

  2. o.result.time ordering for a BACKWARD run is DESCENDING
     (e.g. [14:00, 13:00, ..., 08:00]), confirmed by direct test. This means:
       o.result.isel(time=0)  -> the OBSERVATION position (seed time)
       o.result.isel(time=-1) -> the ORIGIN estimate (most-backward-propagated)
     This is the opposite of what a naive reading of "time is increasing"
     comments elsewhere in basemodel.py would suggest (those apply to a
     different code path, not to o.result) -- getting this backwards would
     silently swap "probable source" and "current observed position", so
     ORIGIN_TIME_INDEX below is the single source of truth for this.

  3. Reader coverage: reader.covers_time(time) and
     reader.covers_positions(lon, lat) -> (indices_covered, lon, lat) are
     real, existing methods on opendrift's reader base class
     (opendrift/readers/basereader/variables.py), confirmed by direct
     introspection and a live test (covers_positions returns a 3-tuple,
     not a boolean mask). Used below instead of hand-rolled xmin/xmax
     comparisons, and instead of assuming any particular attribute exists.

  4. seed_within_polygon(lons, lats, number=None, **kwargs) accepts `time`
     via **kwargs -- confirmed by successful test call. Used instead of
     seed_from_geojson() because Techie 2's polygon is not guaranteed to be
     a full GeoJSON Feature with a `time` property (the milestone's own
     example dummy_polygon is a bare geometry dict) -- observation_time is
     always the explicit, authoritative temporal anchor for this module.
"""
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules/02_environment/src'))
_injected_paths = True



import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Union

import numpy as np
import xarray as xr

from opendrift.models.openoil import OpenOil
from environment_readers import get_readers

logger = logging.getLogger(__name__)

# See docstring point 2. This is the single place that encodes which
# o.result time index is the origin, so it can't drift out of sync between
# functions.
ORIGIN_TIME_INDEX = -1
OBSERVATION_TIME_INDEX = 0

METERS_PER_DEG_LAT = 111320.0  # standard spherical approximation, matches
                                 # the local-meters conversions already used
                                 # in test_current_displacement.py / test_real_environment.py


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class Uncertainty:
    """Spatial uncertainty of the origin particle cloud. The ellipse is the
    primary, "meaningful" metric (spill clouds are rarely isotropic); the
    two radii are a simple, assumption-light fallback/summary.
    Units: meters. orientation_deg: clockwise from true North.
    method: 'covariance_ellipse' (normal case) or 'isotropic_fallback'
    (degenerate cloud -- see compute_uncertainty()).
    """
    semi_major_m: float
    semi_minor_m: float
    orientation_deg: float
    radius_68_m: float
    radius_95_m: float
    method: str


@dataclass
class SourceReconstruction:
    """observed slick -> probable source region + time window + uncertainty + trajectories"""

    observation_time: datetime
    origin_time: datetime
    search_window_hours: float

    origin_centroid: tuple           # (lon, lat) -- a SUMMARY of the origin cloud, never "the" source
    origin_positions: dict           # {'lon': np.ndarray, 'lat': np.ndarray} -- full final backward cloud
    uncertainty: Uncertainty

    trajectories: xr.Dataset         # full o.result: every particle, every timestep, preserved

    n_particles_seeded: int
    n_particles_valid: int           # after dropping any NaN/deactivated particles

    env_mode: str
    env_kwargs: dict

    coverage_warnings: list = field(default_factory=list)  # non-empty => forcing fell back to
                                                              # configured defaults somewhere in
                                                              # the run; see module docstring point 3

    def to_dict(self) -> dict:
        """JSON-friendly summary. Excludes `trajectories` (xarray.Dataset,
        not JSON-serializable) -- access reconstruction.trajectories directly
        for the full per-particle history."""
        d = asdict(self)
        d.pop('trajectories', None)
        d['observation_time'] = self.observation_time.isoformat()
        d['origin_time'] = self.origin_time.isoformat()
        return d


# ---------------------------------------------------------------------------
# Polygon parsing
# ---------------------------------------------------------------------------

def _extract_lon_lat(polygon: Union[dict, str]):
    """Accepts a bare GeoJSON geometry dict (e.g. Techie 2's dummy_polygon),
    a full GeoJSON Feature (e.g. slick.geojson from Milestones 1-3), or a
    geojson string. Returns (lons, lats) as plain lists, exterior ring only,
    without the closing duplicate point."""
    if isinstance(polygon, str):
        polygon = json.loads(polygon)

    if polygon.get('type') == 'Feature':
        geometry = polygon['geometry']
    else:
        geometry = polygon

    if geometry.get('type') != 'Polygon':
        raise ValueError(f"Expected a Polygon geometry, got '{geometry.get('type')}'")

    ring = geometry['coordinates'][0]
    # Drop closing point if the ring is explicitly closed (first == last)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    return lons, lats


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def _to_local_meters(lons, lats, ref_lon, ref_lat):
    """Flat-Earth local ENU projection, same approximation already used in
    test_current_displacement.py / test_real_environment.py. Fine at the
    tens-of-km scale these reconstructions operate at."""
    east_m = (np.asarray(lons) - ref_lon) * METERS_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    north_m = (np.asarray(lats) - ref_lat) * METERS_PER_DEG_LAT
    return east_m, north_m


def compute_uncertainty(lons, lats, centroid_lon, centroid_lat) -> Uncertainty:
    """Covariance ellipse (1-sigma) + 68th/95th percentile radii of the
    origin particle cloud, in local meters.

    Falls back to a simple isotropic (std-based) estimate if the covariance
    matrix is degenerate (too few particles, or all particles coincident) --
    preferring a reliable scalar result over a fragile/undefined ellipse.
    """
    east_m, north_m = _to_local_meters(lons, lats, centroid_lon, centroid_lat)
    distances_m = np.sqrt(east_m ** 2 + north_m ** 2)
    radius_68 = float(np.percentile(distances_m, 68))
    radius_95 = float(np.percentile(distances_m, 95))

    if len(east_m) < 3 or np.allclose(np.std(east_m), 0) and np.allclose(np.std(north_m), 0):
        # Degenerate cloud: fall back to isotropic scalar uncertainty rather
        # than risk a singular/near-singular covariance matrix.
        iso = float(np.std(distances_m)) if len(distances_m) > 0 else 0.0
        return Uncertainty(
            semi_major_m=iso, semi_minor_m=iso, orientation_deg=0.0,
            radius_68_m=radius_68, radius_95_m=radius_95,
            method='isotropic_fallback',
        )

    cov = np.cov(np.vstack([east_m, north_m]))
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    eigvals = np.clip(eigvals, 0, None)     # guard tiny negative numerical noise
    semi_minor_m, semi_major_m = np.sqrt(eigvals[0]), np.sqrt(eigvals[1])
    major_vec = eigvecs[:, 1]  # (east, north) components of major axis
    # Bearing of major axis, clockwise from North: atan2(east, north)
    orientation_deg = (math.degrees(math.atan2(major_vec[0], major_vec[1]))) % 180

    return Uncertainty(
        semi_major_m=float(semi_major_m), semi_minor_m=float(semi_minor_m),
        orientation_deg=float(orientation_deg),
        radius_68_m=radius_68, radius_95_m=radius_95,
        method='covariance_ellipse',
    )


# ---------------------------------------------------------------------------
# Coverage warnings (requirement: never silently fall back)
# ---------------------------------------------------------------------------

def _check_coverage(readers, observation_time, origin_time, lons, lats) -> list:
    """Uses each reader's own covers_time()/covers_positions() (real,
    introspected API -- see module docstring point 3) to detect whether the
    requested time window and particle footprint are actually covered.
    Returns a list of human-readable warning strings (empty if fully
    covered). Coverage gaps mean OpenDrift silently used the configured
    environment:fallback:* values for the uncovered particles/times -- this
    makes that condition visible instead of leaving it implicit.
    """
    warnings = []
    if not readers:
        warnings.append(
            "No environment readers attached -- entire run used "
            "environment:fallback:* constant values."
        )
        return warnings

    for r in readers:
        name = getattr(r, 'name', repr(r))

        for label, t in (('observation_time', observation_time), ('origin_time', origin_time)):
            if not r.covers_time(t):
                warnings.append(
                    f"Reader '{name}' does not cover {label}={t} "
                    f"(reader range: {r.start_time} .. {r.end_time}); "
                    f"fallback values were used outside this range."
                )

        covered_idx, _, _ = r.covers_positions(np.asarray(lons), np.asarray(lats))
        if len(covered_idx) < len(lons):
            n_outside = len(lons) - len(covered_idx)
            warnings.append(
                f"Reader '{name}' does not spatially cover {n_outside} of "
                f"{len(lons)} seeded particle positions "
                f"(reader coverage: {r.coverage_string()}); "
                f"fallback values were used for those particles."
            )
    return warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reconstruct_source(
    polygon: Union[dict, str],
    observation_time: datetime,
    search_window_hours: float = 6.0,
    number: int = 1000,
    env_mode: str = 'netcdf',
    env_kwargs: Optional[dict] = None,
    time_step_seconds: int = 3600,
    loglevel: int = 20,
) -> SourceReconstruction:
    """observed slick -> probable source region + time window + uncertainty + trajectories

    :param polygon: Techie 2's GeoJSON polygon -- bare geometry dict, a
        Feature dict, or a geojson string.
    :param observation_time: REQUIRED. The temporal anchor -- particles are
        seeded AT this time, and propagated backward from it. Distinct from
        `search_window_hours`: this module never requires the (unknown)
        true release time as an input.
    :param search_window_hours: how far back to search, NOT a claim about
        the true release time -- a search-horizon control, not `age_hrs`.
    :param number: particle count.
    :param env_mode: passed straight through to environment_readers.get_readers().
    :param env_kwargs: passed straight through to environment_readers.get_readers().
    :param time_step_seconds: calculation/output timestep.
    """
    env_kwargs = dict(env_kwargs or {})
    lons, lats = _extract_lon_lat(polygon)

    o = OpenOil(loglevel=loglevel)

    # Safety-net fallback (same pattern as Milestone 3): if a particle or
    # timestep falls outside reader coverage, use 0 rather than NaN/crash.
    # _check_coverage() below is what makes it VISIBLE when this safety net
    # actually gets used, rather than leaving it silent.
    o.set_config('environment:fallback:x_wind', 0.0)
    o.set_config('environment:fallback:y_wind', 0.0)
    o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
    o.set_config('environment:fallback:y_sea_water_velocity', 0.0)

    readers = get_readers(env_mode, **env_kwargs)
    if readers:
        o.add_reader(readers)

    o.seed_within_polygon(lons=lons, lats=lats, time=observation_time, number=number)
    n_seeded = o.num_elements_scheduled()

    steps = round(search_window_hours * 3600 / time_step_seconds)
    if not math.isclose(steps * time_step_seconds, search_window_hours * 3600, abs_tol=1):
        logger.warning(
            f"search_window_hours={search_window_hours} is not an exact multiple of "
            f"time_step_seconds={time_step_seconds}s; rounded to {steps} steps "
            f"({steps * time_step_seconds / 3600:.3f}h)."
        )

    o.run(steps=steps, time_step=-time_step_seconds, time_step_output=time_step_seconds)

    # See module docstring point 2: index -1, not 0, is the backward-propagated
    # origin. Verified empirically, not assumed.
    origin_lon = o.result.lon.isel(time=ORIGIN_TIME_INDEX).values.astype(float)
    origin_lat = o.result.lat.isel(time=ORIGIN_TIME_INDEX).values.astype(float)
    valid = ~np.isnan(origin_lon) & ~np.isnan(origin_lat)
    n_valid = int(valid.sum())
    origin_lon_valid = origin_lon[valid]
    origin_lat_valid = origin_lat[valid]

    origin_time = pd_to_datetime(o.result.time.values[ORIGIN_TIME_INDEX])
    observation_time_out = pd_to_datetime(o.result.time.values[OBSERVATION_TIME_INDEX])

    centroid_lon = float(np.mean(origin_lon_valid))
    centroid_lat = float(np.mean(origin_lat_valid))

    uncertainty = compute_uncertainty(origin_lon_valid, origin_lat_valid, centroid_lon, centroid_lat)

    coverage_warnings = _check_coverage(readers, observation_time_out, origin_time, lons, lats)
    for w in coverage_warnings:
        logger.warning(w)

    return SourceReconstruction(
        observation_time=observation_time_out,
        origin_time=origin_time,
        search_window_hours=steps * time_step_seconds / 3600,
        origin_centroid=(centroid_lon, centroid_lat),
        origin_positions={'lon': origin_lon_valid, 'lat': origin_lat_valid},
        uncertainty=uncertainty,
        trajectories=o.result,
        n_particles_seeded=n_seeded,
        n_particles_valid=n_valid,
        env_mode=env_mode,
        env_kwargs=env_kwargs,
        coverage_warnings=coverage_warnings,
    )


def pd_to_datetime(value) -> datetime:
    """o.result.time.values entries are numpy.datetime64; convert to a
    plain datetime so the rest of the module (and downstream consumers)
    don't need a pandas/xarray dependency just to read a timestamp."""
    import pandas as pd
    return pd.Timestamp(value).to_pydatetime()