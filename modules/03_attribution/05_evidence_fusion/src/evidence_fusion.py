"""
OceanTrace - Evidence Fusion / Vessel Attribution (Milestone 7)
================================================================================

Implements the frozen design specification for evidence fusion (M7_EVIDENCE_FUSION_DESIGN.md).
"""
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules/02_drift/03_source_reconstruction/src'))
sys.path.insert(0, str(_repo_root / 'modules/03_attribution/04_ais_trajectory/src'))
_injected_paths = True



import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from source_reconstruction import SourceReconstruction, Uncertainty
from ais_trajectory import AISDataset, VesselTrajectory, VesselQueryMatch, Gap, DEFAULT_GAP_THRESHOLD_SECONDS

# Prototype thresholds, see M7 design spec section F
# These are prototype presentation/triage thresholds for surfacing which candidates deserve 
# the closest look first. They are not a probability, a statistical confidence level, 
# or a validated significance threshold, and must not be described, labeled, or documented 
# as one anywhere.
THRESHOLD_HIGH_PRIORITY = 0.70
THRESHOLD_CANDIDATE = 0.45


@dataclass
class EvidenceWeights:
    spatial_proximity: float = 0.30
    temporal_overlap: float = 0.20
    trajectory_consistency: float = 0.20
    drift_consistency: float = 0.20
    speed_course_behavior: float = 0.10


@dataclass
class EvidenceFeature:
    name: str
    value: Optional[float]
    raw_value: Optional[float]
    raw_unit: Optional[str]
    weight: float
    contribution: float
    explanation: str
    limitation: Optional[str]


@dataclass
class DataConfidence:
    """Reported ALONGSIDE the evidence score. Never scored, never multiplied
    into it, never contributes to ranking."""
    coverage_fraction: float
    overlapping_gaps: list
    adjacent_gaps: list
    note: str


@dataclass
class MetadataContext:
    """Reported ALONGSIDE the score, purely descriptive. Never scored, never
    multiplied into the score, never used to rank candidates."""
    vessel_type: Optional[str]
    length_m: Optional[float]
    cargo: Optional[str]
    note: str = ("Vessel metadata is contextual information only and does not "
                 "affect this vessel's evidence score.")


@dataclass
class TimelineEvent:
    time: datetime
    phase: str
    kind: str
    lon: Optional[float]
    lat: Optional[float]
    sog: Optional[float]
    cog: Optional[float]
    heading: Optional[float]
    note: str


@dataclass
class CandidateAssessment:
    mmsi: str
    vessel_name: Optional[str]
    score: float
    category: str
    features: list
    excluded_features: list
    data_confidence: DataConfidence
    metadata_context: MetadataContext
    timeline: list
    n_observations_considered: int
    data_completeness_notes: list


@dataclass
class AttributionResult:
    reconstruction_summary: dict
    search_radius_km: float
    search_time_window: tuple
    candidates: list
    n_vessels_considered: int
    caveats: list


def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1 = np.asarray(lon1, dtype=float), np.asarray(lat1, dtype=float)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _compute_coverage_and_temporal_overlap(
    trajectory: VesselTrajectory, t_start: datetime, t_end: datetime
) -> Tuple[float, List[Gap], List[Gap]]:
    """Walks consecutive-observation intervals inside the core window to compute covered time."""
    core_duration = (t_end - t_start).total_seconds()
    if core_duration <= 0:
        return 0.0, [], []

    obs = trajectory.observations_between(t_start, t_end)
    points = []
    points.append(t_start)
    for t in obs['time']:
        points.append(t.to_pydatetime())
    points.append(t_end)

    points = sorted(list(set(points)))
    covered_duration = 0.0
    
    for i in range(len(points) - 1):
        diff_s = (points[i+1] - points[i]).total_seconds()
        if diff_s <= DEFAULT_GAP_THRESHOLD_SECONDS:
            covered_duration += diff_s

    overlap = np.clip(covered_duration / core_duration, 0.0, 1.0)
    
    all_gaps = trajectory.gaps(DEFAULT_GAP_THRESHOLD_SECONDS)
    overlapping_gaps = [g for g in all_gaps if g.start_time < t_end and g.end_time > t_start]
    
    pad_adj = timedelta(hours=2)
    adjacent_gaps = [g for g in all_gaps if 
                     (g not in overlapping_gaps) and 
                     ((t_start - pad_adj <= g.end_time <= t_start) or 
                      (t_end <= g.start_time <= t_end + pad_adj))]

    return float(overlap), overlapping_gaps, adjacent_gaps

def _build_features(
    m: VesselQueryMatch,
    recon: SourceReconstruction,
    radius_km: float,
    t_start: datetime,
    t_end: datetime,
    pad_hours: float
) -> Dict[str, dict]:
    """Computes all 5 evidence dimensions."""
    f = {}
    
    # E.1 Spatial Proximity
    v_dist = m.min_distance_km
    val_sp = max(0.0, 1.0 - min(v_dist / radius_km, 1.0))
    f['spatial_proximity'] = {
        'value': float(val_sp),
        'raw_value': float(v_dist),
        'raw_unit': 'km',
        'explanation': f"Closest approach was {v_dist:.1f} km from origin centroid.",
        'limitation': None
    }

    # E.2 Temporal Overlap
    overlap, _, _ = _compute_coverage_and_temporal_overlap(m.trajectory, t_start, t_end)
    f['temporal_overlap'] = {
        'value': overlap,
        'raw_value': overlap * 100,
        'raw_unit': '%',
        'explanation': f"Vessel trackably present for {overlap*100:.0f}% of the release window elapsed time.",
        'limitation': None
    }

    # E.3 Trajectory consistency
    if m.n_observations_in_window < 2:
        f['trajectory_consistency'] = {
            'value': None, 'raw_value': None, 'raw_unit': None,
            'explanation': "Insufficient observations inside window to determine track consistency.",
            'limitation': "Undefined for fewer than 2 in-window pings."
        }
    else:
        # Time inside region
        obs = m.observations_in_window.copy()
        obs['dist'] = _haversine_km(obs['lon'], obs['lat'], recon.origin_centroid[0], recon.origin_centroid[1])
        inside = (obs['dist'] <= radius_km)
        obs['time_diff'] = obs['time'].diff().dt.total_seconds().fillna(0)
        
        # very simplified: fraction of internal segments that start and end inside.
        inside_time = 0.0
        total_time = (obs['time'].max() - obs['time'].min()).total_seconds()
        
        if total_time > 0:
            for i in range(1, len(obs)):
                if inside.iloc[i-1] and inside.iloc[i]:
                    inside_time += obs['time_diff'].iloc[i]
                elif inside.iloc[i-1] or inside.iloc[i]:
                    inside_time += obs['time_diff'].iloc[i] / 2.0
            
            val_tc = inside_time / total_time
            f['trajectory_consistency'] = {
                'value': float(val_tc),
                'raw_value': inside_time / 3600.0,
                'raw_unit': 'hours',
                'explanation': f"Track inside search region for {val_tc*100:.0f}% of its in-window duration.",
                'limitation': None
            }
        else:
            f['trajectory_consistency'] = {
                'value': None, 'raw_value': None, 'raw_unit': None,
                'explanation': "Duration too short for track evaluation.",
                'limitation': None
            }

    # E.4 Drift consistency
    # Interpolate particle paths to vessel observation times
    # NOTE: recon.trajectories.time is DESCENDING for backward runs.
    # Must sort ascending before interpolation.
    if m.n_observations_in_window == 0:
        f['drift_consistency'] = {
            'value': None, 'raw_value': None, 'raw_unit': None,
            'explanation': "No observations in release window to match against drift model.",
            'limitation': None
        }
    else:
        try:
            ds = recon.trajectories.sortby('time') # ensure ascending
            obs = m.observations_in_window
            # only use observations that fall within the trajectory time bounds exactly
            t_min = pd.Timestamp(ds.time.values.min())
            t_max = pd.Timestamp(ds.time.values.max())
            valid_obs = obs[(obs['time'] >= t_min) & (obs['time'] <= t_max)]
            
            if len(valid_obs) == 0:
                f['drift_consistency'] = {
                    'value': None, 'raw_value': None, 'raw_unit': None,
                    'explanation': "Vessel observations fall outside the simulated trajectory time bounds.",
                    'limitation': None
                }
            else:
                target_times = xr.DataArray(valid_obs['time'].values, dims=['obs'])
                ds_interp = ds.interp(time=target_times, method='linear')
                
                # ds_interp has dims (trajectory, obs)
                p_lons = ds_interp.lon.values # shape (n_particles, n_obs)
                p_lats = ds_interp.lat.values
                
                v_lons = valid_obs['lon'].values # shape (n_obs,)
                v_lats = valid_obs['lat'].values
                
                # Compute distance to all particles per obs
                min_dists_per_obs = []
                for i in range(len(valid_obs)):
                    if np.all(np.isnan(p_lons[:, i])): continue
                    valid_p = ~np.isnan(p_lons[:, i])
                    dists = _haversine_km(p_lons[valid_p, i], p_lats[valid_p, i], v_lons[i], v_lats[i])
                    min_dists_per_obs.append(np.min(dists))
                    
                if min_dists_per_obs:
                    median_dist = float(np.median(min_dists_per_obs))
                    val_dc = max(0.0, 1.0 - min(median_dist / radius_km, 1.0))
                    
                    lim_note = None
                    if recon.coverage_warnings:
                        lim_note = "Reduced confidence: underlying drift model experienced coverage gaps/fallback values."
                        
                    f['drift_consistency'] = {
                        'value': val_dc,
                        'raw_value': median_dist,
                        'raw_unit': 'km',
                        'explanation': f"Median distance to nearest simulated particle at identical time: {median_dist:.1f} km.",
                        'limitation': lim_note
                    }
                else:
                    f['drift_consistency'] = {
                        'value': None, 'raw_value': None, 'raw_unit': None,
                        'explanation': "Simulated particles out of bounds or NaN at vessel observation times.",
                        'limitation': None
                    }
        except Exception as e:
            f['drift_consistency'] = {
                'value': None, 'raw_value': None, 'raw_unit': None,
                'explanation': f"Failed to interpolate drift model: {str(e)}",
                'limitation': None
            }

    # E.5 Speed/Course behavior
    padded_obs = m.trajectory.observations_between(t_start, t_end, timedelta(hours=pad_hours), timedelta(hours=pad_hours))
    if len(padded_obs) < 3 or m.n_observations_in_window == 0:
        f['speed_course_behavior'] = {
            'value': None, 'raw_value': None, 'raw_unit': None,
            'explanation': "Insufficient padded context to establish baseline speed/course behavior.",
            'limitation': None
        }
    else:
        in_win = padded_obs['in_core_window']
        out_win = ~in_win
        if out_win.sum() == 0:
            f['speed_course_behavior'] = {
                'value': None, 'raw_value': None, 'raw_unit': None,
                'explanation': "No observations outside the core window to establish baseline.",
                'limitation': None
            }
        else:
            sog_in = padded_obs.loc[in_win, 'sog'].dropna()
            sog_out = padded_obs.loc[out_win, 'sog'].dropna()
            
            if sog_in.empty or sog_out.empty:
                f['speed_course_behavior'] = {
                    'value': None, 'raw_value': None, 'raw_unit': None,
                    'explanation': "SOG not available to compute behavioral deviation.",
                    'limitation': "AIS SOG field missing or invalid."
                }
            else:
                mean_in = sog_in.mean()
                mean_out = sog_out.mean()
                diff = abs(mean_in - mean_out)
                val_sc = min(diff / 5.0, 1.0) # SOG deviation of 5 knots saturates score
                f['speed_course_behavior'] = {
                    'value': float(val_sc),
                    'raw_value': float(diff),
                    'raw_unit': 'knots',
                    'explanation': f"Mean SOG shifted from {mean_out:.1f} kn (context) to {mean_in:.1f} kn (in-window).",
                    'limitation': "Deviation correlates with region but has many innocent causes (traffic, anchor, weather)."
                }

    return f

def _build_timeline(
    trajectory: VesselTrajectory,
    t_start: datetime,
    t_end: datetime,
    pad_hours: float,
    center_lon: float,
    center_lat: float,
    radius_km: float
) -> List[TimelineEvent]:
    
    padded_obs = trajectory.observations_between(
        t_start, t_end, timedelta(hours=pad_hours), timedelta(hours=pad_hours)
    )
    
    events = []
    
    inside_region = False
    
    for i, row in padded_obs.iterrows():
        t = row['time'].to_pydatetime()
        if t < t_start:
            phase = 'before'
        elif t > t_end:
            phase = 'after'
        else:
            phase = 'during'
            
        dist = _haversine_km(row['lon'], row['lat'], center_lon, center_lat)
        currently_inside = bool(dist <= radius_km)
        
        if currently_inside and not inside_region:
            events.append(TimelineEvent(
                time=t, phase=phase, kind='entered_region',
                lon=row['lon'], lat=row['lat'],
                sog=row['sog'] if pd.notna(row['sog']) else None,
                cog=row['cog'] if pd.notna(row['cog']) else None,
                heading=row['heading'] if pd.notna(row['heading']) else None,
                note=f"Entered search radius ({dist:.1f} km from center)"
            ))
            inside_region = True
            
        elif not currently_inside and inside_region:
            events.append(TimelineEvent(
                time=t, phase=phase, kind='exited_region',
                lon=row['lon'], lat=row['lat'],
                sog=row['sog'] if pd.notna(row['sog']) else None,
                cog=row['cog'] if pd.notna(row['cog']) else None,
                heading=row['heading'] if pd.notna(row['heading']) else None,
                note=f"Exited search radius ({dist:.1f} km from center)"
            ))
            inside_region = False
            
        events.append(TimelineEvent(
            time=t, phase=phase, kind='observation',
            lon=row['lon'], lat=row['lat'],
            sog=row['sog'] if pd.notna(row['sog']) else None,
            cog=row['cog'] if pd.notna(row['cog']) else None,
            heading=row['heading'] if pd.notna(row['heading']) else None,
            note="AIS observation"
        ))

    pad_delta = timedelta(hours=pad_hours)
    all_gaps = trajectory.gaps(DEFAULT_GAP_THRESHOLD_SECONDS)
    for g in all_gaps:
        if g.end_time >= t_start - pad_delta and g.start_time <= t_end + pad_delta:
            mid_t = g.start_time + timedelta(seconds=g.duration_seconds/2)
            if mid_t < t_start:
                phase = 'before'
            elif mid_t > t_end:
                phase = 'after'
            else:
                phase = 'during'
            
            events.append(TimelineEvent(
                time=g.start_time, phase=phase, kind='gap',
                lon=None, lat=None, sog=None, cog=None, heading=None,
                note=f"Gap started ({g.duration_seconds/3600:.1f} hours)"
            ))

    events.sort(key=lambda e: e.time)
    return events


def evaluate_candidates(
    reconstruction: SourceReconstruction,
    ais: AISDataset,
    radius_km: Optional[float] = None,
    time_padding_hours: float = 2.0,
    weights: Optional[EvidenceWeights] = None,
) -> AttributionResult:
    
    if weights is None:
        weights = EvidenceWeights()
        
    w_dict = asdict(weights)
    
    unc = reconstruction.uncertainty
    if radius_km is None:
        radius_km = max(unc.radius_95_m, unc.semi_major_m) / 1000.0
        
    t_start = min(reconstruction.origin_time, reconstruction.observation_time)
    t_end = max(reconstruction.origin_time, reconstruction.observation_time)
    
    caveats = list(reconstruction.coverage_warnings)
    
    center_lon, center_lat = reconstruction.origin_centroid
    
    matches = ais.vessels_near(center_lon, center_lat, radius_km, t_start, t_end)
    
    if not matches:
        # One documented radius-doubling retry
        expanded_radius = radius_km * 2.0
        caveats.append(f"No vessels found at nominal radius of {radius_km:.1f} km; results use an expanded radius of {expanded_radius:.1f} km.")
        radius_km = expanded_radius
        matches = ais.vessels_near(center_lon, center_lat, radius_km, t_start, t_end)
        
    candidates = []
    
    for m in matches:
        raw_f = _build_features(m, reconstruction, radius_km, t_start, t_end, time_padding_hours)
        
        available_names = [name for name, f in raw_f.items() if f['value'] is not None]
        excluded = [name for name in w_dict.keys() if name not in available_names]
        
        features_obj = []
        
        if not available_names:
            score = 0.0
            category = 'Attribution Inconclusive'
        else:
            total_weight = sum(w_dict[n] for n in available_names)
            raw_score = sum(w_dict[n] * raw_f[n]['value'] for n in available_names) / total_weight
            score = float(np.clip(raw_score, 0.0, 1.0))
            
            for n in available_names:
                renorm_weight = w_dict[n] / total_weight
                contrib = renorm_weight * raw_f[n]['value']
                features_obj.append(EvidenceFeature(
                    name=n,
                    value=raw_f[n]['value'],
                    raw_value=raw_f[n]['raw_value'],
                    raw_unit=raw_f[n]['raw_unit'],
                    weight=renorm_weight,
                    contribution=contrib,
                    explanation=raw_f[n]['explanation'],
                    limitation=raw_f[n]['limitation']
                ))
            
            # Floor values to count as "above floor": let's say 0.20
            n_above_floor = sum(1 for f in features_obj if f.value >= 0.20)
            
            if score >= THRESHOLD_HIGH_PRIORITY and n_above_floor >= 3:
                category = 'High-Priority Candidate'
            elif score >= THRESHOLD_CANDIDATE:
                category = 'Candidate'
            elif score > 0:
                category = 'Low-Evidence Candidate'
            else:
                category = 'Attribution Inconclusive'

        overlap, over_gaps, adj_gaps = _compute_coverage_and_temporal_overlap(m.trajectory, t_start, t_end)
        
        gap_note = "An observability gap has many possible causes (equipment, coverage, congestion, port berthing, or, less commonly, deliberate action) and is not evidence of intent by itself."
        if overlap == 0.0:
            gap_note = "No AIS coverage at all during the release window — presence/absence cannot be determined from AIS. " + gap_note
            
        data_confidence = DataConfidence(
            coverage_fraction=overlap,
            overlapping_gaps=over_gaps,
            adjacent_gaps=adj_gaps,
            note=gap_note
        )
        
        meta = MetadataContext(
            vessel_type=m.trajectory.vessel_type,
            length_m=m.trajectory.length,
            cargo=m.trajectory.cargo
        )
        
        completeness = []
        if m.n_observations_in_window < 3:
            completeness.append(f"Only {m.n_observations_in_window} observations in window.")
            
        timeline = _build_timeline(m.trajectory, t_start, t_end, time_padding_hours, center_lon, center_lat, radius_km)
        
        c = CandidateAssessment(
            mmsi=m.mmsi,
            vessel_name=m.trajectory.vessel_name,
            score=score,
            category=category,
            features=features_obj,
            excluded_features=excluded,
            data_confidence=data_confidence,
            metadata_context=meta,
            timeline=timeline,
            n_observations_considered=m.n_observations_in_window,
            data_completeness_notes=completeness
        )
        candidates.append(c)
        
    candidates.sort(key=lambda x: x.score, reverse=True)
    
    return AttributionResult(
        reconstruction_summary={
            'centroid': reconstruction.origin_centroid,
            'radius_68_m': unc.radius_68_m,
            'radius_95_m': unc.radius_95_m,
            'semi_major_m': unc.semi_major_m,
            'semi_minor_m': unc.semi_minor_m,
            'orientation_deg': unc.orientation_deg
        },
        search_radius_km=radius_km,
        search_time_window=(t_start, t_end),
        candidates=candidates,
        n_vessels_considered=len(candidates),
        caveats=caveats
    )


def generate_report(result: AttributionResult, top_n: int = 5) -> str:
    lines = []
    lines.append("OceanTrace Incident Report")
    lines.append("="*40)
    
    rc = result.reconstruction_summary
    lines.append(f"\nProbable Source: {rc['centroid'][1]:.4f} N, {rc['centroid'][0]:.4f} E")
    lines.append(f"Uncertainty: {rc['radius_68_m']/1000:.1f} km (68%) / {rc['radius_95_m']/1000:.1f} km (95%);")
    lines.append(f"             ellipse {rc['semi_major_m']/1000:.1f} x {rc['semi_minor_m']/1000:.1f} km, oriented {rc['orientation_deg']:.1f}deg from N")
    
    lines.append(f"Release Window: {result.search_time_window[0]} -> {result.search_time_window[1]}")
    lines.append(f"Search: {result.search_radius_km:.1f} km (derived from 95th-percentile origin-cloud radius),")
    lines.append(f"        {result.n_vessels_considered} vessels considered")
    
    if result.caveats:
        lines.append(f"Caveats: {'; '.join(result.caveats)}")
        
    for i, c in enumerate(result.candidates[:top_n]):
        lines.append(f"\nRank {i+1}: MMSI {c.mmsi} ({c.vessel_name or 'name unknown'})")
        lines.append(f"Assessment: {c.category}    Score: {c.score:.2f}")
        lines.append("Evidence:")
        for f in c.features:
            lines.append(f"  - {f.name}: {f.explanation}")
            if f.limitation:
                lines.append(f"    (Limitation: {f.limitation})")
        if c.excluded_features:
            lines.append(f"Excluded from scoring: {', '.join(c.excluded_features)}")
        
        lines.append(f"Data confidence: {c.data_confidence.coverage_fraction*100:.0f}% AIS coverage during release window")
        for g in c.data_confidence.overlapping_gaps + c.data_confidence.adjacent_gaps:
            lines.append(f"  - AIS coverage gap of {g.duration_seconds/3600:.1f}h between {g.start_time} and {g.end_time}.")
        lines.append(f"  - Note: {c.data_confidence.note}")
            
        mc = c.metadata_context
        meta_str = f"Type: {mc.vessel_type}, Length: {mc.length_m}m, Cargo: {mc.cargo}"
        lines.append(f"Vessel context (not scored): {meta_str}")
        if c.data_completeness_notes:
            lines.append(f"Data completeness: {'; '.join(c.data_completeness_notes)}")

    if result.n_vessels_considered > top_n:
        lines.append(f"\n* {result.n_vessels_considered - top_n} additional vessels scored below the Top {top_n} and are omitted from this summary.")
        
    lines.append("\nMethodology Note:")
    lines.append("The 0.70 ('High-Priority') / 0.45 ('Candidate') category thresholds and the >=3-feature condition "
                 "are prototype presentation/triage thresholds for surfacing which candidates deserve the closest look first. "
                 "They are not a probability, a statistical confidence level, or a validated significance threshold.")

    lines.append("\nConclusion:")
    if not result.candidates:
        lines.append("Attribution remains inconclusive: this analysis identifies investigative priorities, not legal responsibility.")
    else:
        top_c = result.candidates[0]
        if top_c.features:
            top_features = sorted(top_c.features, key=lambda f: f.contribution, reverse=True)
            reasons = ", ".join([f.name for f in top_features[:2]])
        else:
            reasons = "proximity"
            
        if top_c.category == 'High-Priority Candidate':
            lines.append(f"{top_c.vessel_name or top_c.mmsi} is the highest-priority candidate for further investigation based on {reasons}. "
                         f"Attribution remains inconclusive: this analysis identifies investigative priorities, not legal responsibility.")
        else:
            lines.append(f"{top_c.vessel_name or top_c.mmsi} is the best available candidate based on {reasons}, but does not meet High-Priority criteria. "
                         f"Attribution remains inconclusive: this analysis identifies investigative priorities, not legal responsibility.")
            
    return "\n".join(lines)