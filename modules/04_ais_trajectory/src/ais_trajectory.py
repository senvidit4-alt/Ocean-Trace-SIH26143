"""
OceanTrace - AIS ingestion and vessel trajectory reconstruction (Milestone 6)
=================================================================================

Given: probable source region + source uncertainty + probable time window
Answer (eventually, in evidence-fusion, NOT here): which vessels were present
in/near this region during this period?

This module only answers the DATA question -- "what did AIS actually record,
and which vessels/observations match a given space+time window" -- and
returns plain, descriptive data structures. It does not score, rank,
interpret, or accuse. No evidence scoring, attribution, or forensic ranking
is implemented here (that is evidence-fusion, a later milestone).

Independence
------------
This module imports only pandas/numpy/dataclasses/datetime. It does NOT
import source_reconstruction.py or opendrift, and is not imported by them.
AIS ingestion and drift physics are separate concerns; the two are meant to
be connected later by a thin adapter living in evidence-fusion (see the
design note at the bottom of this file), not by coupling this module to
OpenDrift's dependency stack.

Dataset assumptions
--------------------
Written against the schema described for
data/AIS_178834011589976755_1814-1788340116592.csv (17 columns, matching the
standard NOAA MarineCadastre AIS export):
    MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading,
    VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft,
    Cargo, TransceiverClass
Core fields (no missing values expected): MMSI, BaseDateTime, LAT, LON, SOG,
COG, Heading. Column matching is case-insensitive and whitespace-tolerant,
since exact header casing has not been verified against the real file yet
in this environment -- see the milestone report for why.

Timestamps are treated as UTC-naive throughout, matching the convention
already used in source_reconstruction.py / OpenDrift (which also strips
tzinfo and treats all times as UTC internally) -- so timestamps compare
directly across modules without conversion.

Important behaviors, by design
-------------------------------
- No fixed sampling interval is assumed anywhere (median ~80s / mean ~5m14s
  in this dataset -- genuinely irregular). Gap detection and windowing both
  operate on actual observed timestamps, never a resampled grid.
- No position or navigation value is ever interpolated or invented. Rows
  are either real AIS observations or dropped (only for missing core
  identity: MMSI/time/lat/lon); nothing is filled in between.
- AIS gaps are reported as plain facts (start/end time, duration, bracketing
  positions) with no "suspicious" flag or interpretation of any kind.
- Duplicate (MMSI, timestamp) observations (same vessel, same instant,
  possibly differing LAT/LON -- this happens in real AIS feeds via
  base-station vs. satellite relay of the same message) are PRESERVED, not
  silently resolved to one. They are counted in LoadReport and kept in
  trajectory order using a deterministic tiebreak (original CSV row index),
  so re-running the loader on the same file always produces the same order.
- The default gap threshold (600s) is an initial prototype default, not a
  scientifically established value for this vessel population or region --
  it's a configurable parameter for exactly that reason.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ITU-R M.1371 "not available" sentinel values for AIS navigation fields.
# Converted to NaN (not dropped -- position/identity may still be valid).
SOG_NOT_AVAILABLE = 102.3
HEADING_NOT_AVAILABLE = 511

DEFAULT_GAP_THRESHOLD_SECONDS = 600  # ~7-8x this dataset's median interval (~80s).
                                       # Prototype default only -- see module docstring.

# Canonical column names this module works with internally, mapped from the
# raw NOAA MarineCadastre header names (matched case-insensitively).
_COLUMN_ALIASES = {
    'mmsi': ['mmsi'],
    'time': ['basedatetime', 'base_date_time', 'time', 'timestamp'],
    'lat': ['lat', 'latitude'],
    'lon': ['lon', 'lng', 'longitude'],
    'sog': ['sog'],
    'cog': ['cog'],
    'heading': ['heading'],
    'vessel_name': ['vesselname', 'vessel_name'],
    'imo': ['imo'],
    'call_sign': ['callsign', 'call_sign'],
    'vessel_type': ['vesseltype', 'vessel_type'],
    'status': ['status'],
    'length': ['length'],
    'width': ['width'],
    'draft': ['draft'],
    'cargo': ['cargo'],
    'transceiver_class': ['transceiverclass', 'transceiver_class'],
}
CORE_FIELDS = ['mmsi', 'time', 'lat', 'lon']
NAV_FIELDS = ['sog', 'cog', 'heading']
METADATA_FIELDS = ['vessel_name', 'imo', 'call_sign', 'vessel_type', 'status',
                    'length', 'width', 'draft', 'cargo', 'transceiver_class']


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class Gap:
    """A purely descriptive silence in one vessel's AIS record. No
    interpretation of cause or intent -- see module docstring."""
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    start_position: tuple   # (lon, lat) -- last confirmed position before the gap
    end_position: tuple     # (lon, lat) -- first confirmed position after the gap


@dataclass
class VesselTrajectory:
    """One vessel's chronological AIS record. `observations` holds only
    real, observed rows -- nothing resampled or interpolated."""
    mmsi: str
    observations: pd.DataFrame   # columns: time, lat, lon, sog, cog, heading, source_row (+ raw metadata cols)
    first_seen: datetime
    last_seen: datetime
    n_observations: int
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None
    imo: Optional[str] = None
    call_sign: Optional[str] = None
    length: Optional[float] = None
    width: Optional[float] = None
    draft: Optional[float] = None
    cargo: Optional[str] = None
    status: Optional[str] = None
    transceiver_class: Optional[str] = None

    def gaps(self, threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS) -> list:
        """Consecutive-observation gaps exceeding threshold_seconds. Only
        internal gaps (between two real observations of this vessel) --
        never the record's start/end boundary, and never an inferred
        position. See module docstring for why threshold_seconds has no
        scientifically fixed default."""
        obs = self.observations
        if len(obs) < 2:
            return []
        times = obs['time'].values
        diffs_s = np.diff(times).astype('timedelta64[s]').astype(float)
        gap_idx = np.where(diffs_s > threshold_seconds)[0]
        result = []
        for i in gap_idx:
            result.append(Gap(
                start_time=pd.Timestamp(times[i]).to_pydatetime(),
                end_time=pd.Timestamp(times[i + 1]).to_pydatetime(),
                duration_seconds=float(diffs_s[i]),
                start_position=(float(obs['lon'].iloc[i]), float(obs['lat'].iloc[i])),
                end_position=(float(obs['lon'].iloc[i + 1]), float(obs['lat'].iloc[i + 1])),
            ))
        return result

    def observations_between(self, t_start: datetime, t_end: datetime,
                              pad_before: timedelta = timedelta(0),
                              pad_after: timedelta = timedelta(0)) -> pd.DataFrame:
        """Real observations in [t_start - pad_before, t_end + pad_after].

        `pad_before`/`pad_after` exist specifically so evidence-fusion can
        pull a vessel's behavior immediately before/after a reconstructed
        spill window as CONTEXT, without conflating that context with the
        core window itself. The returned frame has an `in_core_window`
        column (True for rows inside [t_start, t_end], False for rows only
        present because of padding) so callers never have to recompute the
        distinction themselves. No rows are invented for padding -- it only
        widens which real observations are included.
        """
        window_start = t_start - pad_before
        window_end = t_end + pad_after
        mask = (self.observations['time'] >= window_start) & (self.observations['time'] <= window_end)
        result = self.observations.loc[mask].copy()
        result['in_core_window'] = (result['time'] >= t_start) & (result['time'] <= t_end)
        return result.reset_index(drop=True)

    def bbox(self) -> tuple:
        """(lon_min, lon_max, lat_min, lat_max) over this vessel's full record."""
        obs = self.observations
        return (float(obs['lon'].min()), float(obs['lon'].max()),
                float(obs['lat'].min()), float(obs['lat'].max()))


@dataclass
class VesselQueryMatch:
    """Result of AISDataset.vessels_near() for one vessel."""
    mmsi: str
    trajectory: VesselTrajectory          # FULL trajectory, unfiltered -- context is preserved
    observations_in_window: pd.DataFrame  # just the rows inside the queried space+time
    min_distance_km: float                # closest approach to the query center, within the window
    closest_approach_time: datetime
    n_observations_in_window: int


@dataclass
class LoadReport:
    """Every row the loader drops or flags, counted and surfaced -- nothing
    silent. Same discipline as SourceReconstruction.coverage_warnings."""
    rows_read: int
    rows_dropped_missing_core_fields: int
    rows_dropped_invalid_ranges: int
    exact_duplicate_rows_dropped: int
    duplicate_mmsi_timestamp_rows: int    # FLAGGED, not dropped -- see module docstring
    rows_final: int
    n_vessels: int
    time_range: tuple
    spatial_bounds: tuple
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def _resolve_columns(raw_columns) -> dict:
    """Case-insensitive, whitespace-tolerant match of raw CSV headers to
    canonical names. Returns {canonical_name: raw_column_name}. Raises with
    a clear message if a core field can't be found -- never guesses."""
    normalized = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in raw_columns}
    resolved = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.replace('_', '')
            if key in normalized:
                resolved[canonical] = normalized[key]
                break
    missing_core = [f for f in CORE_FIELDS if f not in resolved]
    if missing_core:
        raise ValueError(
            f"Could not find required core column(s) {missing_core} in CSV header "
            f"{list(raw_columns)}. Column matching is case-insensitive but exact "
            f"names must be present."
        )
    return resolved


def load_ais_csv(csv_path: str) -> tuple:
    """Read-only load: CSV -> (cleaned DataFrame, LoadReport). Never writes
    to csv_path. Cleaning order matches the approved design: parse timestamps
    -> validate core fields -> duplicate handling (both exact-row dedup and
    (mmsi, time) duplicate flagging, computed on the actual loaded data)."""
    raw = pd.read_csv(csv_path, low_memory=False)
    rows_read = len(raw)
    warnings = []

    colmap = _resolve_columns(raw.columns)
    df = pd.DataFrame({canonical: raw[raw_col] for canonical, raw_col in colmap.items()})
    df['source_row'] = np.arange(rows_read)  # provenance + deterministic tiebreak, see module docstring

    # --- 2. Timestamp parsing --------------------------------------------
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    # naive UTC throughout, matching source_reconstruction.py / OpenDrift convention
    if pd.api.types.is_datetime64tz_dtype(df['time']):
        df['time'] = df['time'].dt.tz_localize(None)

    # --- numeric coercion + sentinel handling (ITU-R M.1371) --------------
    for c in ['lat', 'lon', 'sog', 'cog', 'heading']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n_sog_sentinel = int((df['sog'] >= SOG_NOT_AVAILABLE).sum())
    df.loc[df['sog'] >= SOG_NOT_AVAILABLE, 'sog'] = np.nan
    n_heading_sentinel = int((df['heading'] == HEADING_NOT_AVAILABLE).sum())
    df.loc[df['heading'] == HEADING_NOT_AVAILABLE, 'heading'] = np.nan
    if n_sog_sentinel:
        warnings.append(f"{n_sog_sentinel} SOG values were the AIS 'not available' "
                         f"sentinel ({SOG_NOT_AVAILABLE}) -- set to NaN, rows kept.")
    if n_heading_sentinel:
        warnings.append(f"{n_heading_sentinel} Heading values were the AIS 'not available' "
                         f"sentinel ({HEADING_NOT_AVAILABLE}) -- set to NaN, rows kept.")

    # --- 3. Core navigation-field validation -------------------------------
    missing_core_mask = df['mmsi'].isna() | df['time'].isna() | df['lat'].isna() | df['lon'].isna()
    n_missing_core = int(missing_core_mask.sum())
    df = df.loc[~missing_core_mask].copy()

    invalid_range_mask = (
        (df['lat'] < -90) | (df['lat'] > 90) |
        (df['lon'] < -180) | (df['lon'] > 180) |
        (df['sog'] < 0) |
        (df['cog'] < 0) | (df['cog'] > 360) |
        (df['heading'] < 0) | (df['heading'] > 359)
    )
    n_invalid_range = int(invalid_range_mask.fillna(False).sum())
    df = df.loc[~invalid_range_mask.fillna(False)].copy()

    df['mmsi'] = df['mmsi'].astype('Int64').astype(str)

    # --- 4. Duplicate handling ---------------------------------------------
    # Exact duplicate rows (all columns identical): dropped. Computed on the
    # actual cleaned data, not assumed to be 0.
    exact_dup_mask = df.drop(columns=['source_row']).duplicated(keep='first')
    n_exact_dup = int(exact_dup_mask.sum())
    df = df.loc[~exact_dup_mask].copy()

    # (MMSI, time) duplicates with differing other fields: PRESERVED, only
    # counted -- see module docstring for why.
    n_mmsi_time_dup = int(df.duplicated(subset=['mmsi', 'time'], keep=False).sum())
    if n_mmsi_time_dup:
        warnings.append(f"{n_mmsi_time_dup} rows share an (MMSI, timestamp) pair with at "
                         f"least one other row -- all preserved (not resolved to one), "
                         f"sorted deterministically by original CSV row order.")

    # Deterministic chronological order, per vessel: time, then original row
    # index as an explicit tiebreak (never rely on an implicit stable-sort
    # assumption for this).
    df = df.sort_values(['mmsi', 'time', 'source_row']).reset_index(drop=True)

    report = LoadReport(
        rows_read=rows_read,
        rows_dropped_missing_core_fields=n_missing_core,
        rows_dropped_invalid_ranges=n_invalid_range,
        exact_duplicate_rows_dropped=n_exact_dup,
        duplicate_mmsi_timestamp_rows=n_mmsi_time_dup,
        rows_final=len(df),
        n_vessels=df['mmsi'].nunique(),
        time_range=(df['time'].min().to_pydatetime(), df['time'].max().to_pydatetime()) if len(df) else (None, None),
        spatial_bounds=(float(df['lon'].min()), float(df['lon'].max()),
                         float(df['lat'].min()), float(df['lat'].max())) if len(df) else (None, None, None, None),
        warnings=warnings,
    )
    for w in warnings:
        logger.warning(w)
    return df, report


def _build_vessel_trajectory(mmsi: str, group: pd.DataFrame) -> VesselTrajectory:
    def _mode_or_none(series):
        non_null = series.dropna()
        if non_null.empty:
            return None
        m = non_null.mode()
        return m.iloc[0] if not m.empty else None

    obs_cols = ['time', 'lat', 'lon', 'sog', 'cog', 'heading', 'source_row'] + \
               [c for c in METADATA_FIELDS if c in group.columns]
    observations = group[obs_cols].reset_index(drop=True)

    kwargs = {}
    for f in METADATA_FIELDS:
        if f in group.columns:
            kwargs[f] = _mode_or_none(group[f])

    return VesselTrajectory(
        mmsi=mmsi,
        observations=observations,
        first_seen=group['time'].min().to_pydatetime(),
        last_seen=group['time'].max().to_pydatetime(),
        n_observations=len(group),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Spatial helper
# ---------------------------------------------------------------------------

def _haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized great-circle distance in km. Same formula/style already
    used (in meters) in test_current_displacement.py / test_real_environment.py
    / source_reconstruction.py, kept consistent project-wide."""
    R = 6371.0
    lon1, lat1 = np.asarray(lon1, dtype=float), np.asarray(lat1, dtype=float)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Main container
# ---------------------------------------------------------------------------

class AISDataset:
    """396-vessel, 264k-row scale -- deliberately kept to plain pandas/numpy,
    no database, no spatial index library. All 396 vessel trajectories are
    built once at load time (cheap at this scale)."""

    def __init__(self, df: pd.DataFrame, report: LoadReport):
        self.df = df
        self.load_report = report
        self._vessels = {
            mmsi: _build_vessel_trajectory(mmsi, group)
            for mmsi, group in df.groupby('mmsi', sort=False)
        }

    @classmethod
    def load(cls, csv_path: str) -> 'AISDataset':
        df, report = load_ais_csv(csv_path)
        logger.info(f"Loaded AIS dataset: {report.rows_final}/{report.rows_read} rows kept, "
                    f"{report.n_vessels} vessels, {report.exact_duplicate_rows_dropped} exact "
                    f"duplicates dropped, {report.duplicate_mmsi_timestamp_rows} "
                    f"(mmsi,time)-duplicate rows preserved.")
        return cls(df, report)

    def vessel(self, mmsi) -> Optional[VesselTrajectory]:
        return self._vessels.get(str(mmsi))

    def all_mmsi(self) -> list:
        return list(self._vessels.keys())

    def time_range(self) -> tuple:
        return self.load_report.time_range

    def spatial_bounds(self) -> tuple:
        return self.load_report.spatial_bounds

    def vessels_near(self, center_lon: float, center_lat: float, radius_km: float,
                      time_start: datetime, time_end: datetime) -> list:
        """Which vessels have at least one real observation within
        radius_km of (center_lon, center_lat) during [time_start, time_end].
        Time-filters first (cheap), then distance-filters (vectorized
        haversine) -- see module design notes. Results sorted by closest
        approach distance, nearest first."""
        mask_time = (self.df['time'] >= time_start) & (self.df['time'] <= time_end)
        subset = self.df.loc[mask_time]
        if subset.empty:
            return []

        dist_km = _haversine_km(subset['lon'].values, subset['lat'].values, center_lon, center_lat)
        mask_dist = dist_km <= radius_km
        if not mask_dist.any():
            return []

        subset = subset.loc[mask_dist].copy()
        subset['_distance_km'] = dist_km[mask_dist]

        matches = []
        for mmsi, group in subset.groupby('mmsi', sort=False):
            idxmin = group['_distance_km'].idxmin()
            matches.append(VesselQueryMatch(
                mmsi=mmsi,
                trajectory=self._vessels[mmsi],
                observations_in_window=group.drop(columns=['_distance_km']).reset_index(drop=True),
                min_distance_km=float(group['_distance_km'].min()),
                closest_approach_time=group.loc[idxmin, 'time'].to_pydatetime(),
                n_observations_in_window=len(group),
            ))
        matches.sort(key=lambda m: m.min_distance_km)
        return matches


# ---------------------------------------------------------------------------
# Design note: connecting to source_reconstruction.py (not implemented here)
# ---------------------------------------------------------------------------
# evidence-fusion (a later milestone) will do roughly:
#
#   from source_reconstruction import reconstruct_source
#   from ais_trajectory import AISDataset
#
#   recon = reconstruct_source(polygon, observation_time, ...)
#   ais = AISDataset.load(csv_path)
#   radius_km = recon.uncertainty.radius_95_m / 1000   # evidence-fusion's choice, not this module's
#   matches = ais.vessels_near(
#       center_lon=recon.origin_centroid[0], center_lat=recon.origin_centroid[1],
#       radius_km=radius_km, time_start=recon.origin_time, time_end=recon.observation_time,
#   )
#   for m in matches:
#       context = m.trajectory.observations_between(
#           recon.origin_time, recon.observation_time,
#           pad_before=timedelta(hours=2), pad_after=timedelta(hours=2))
#       gaps = m.trajectory.gaps()
#       # -> spatial proximity, temporal match, trajectory consistency,
#       #    speed/course behaviour, AIS observability gaps computed HERE,
#       #    not in ais_trajectory.py.
#
# This module never imports source_reconstruction.py or opendrift, by design.
