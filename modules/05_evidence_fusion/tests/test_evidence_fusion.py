import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / 'modules/05_evidence_fusion/src'))
sys.path.insert(0, str(_repo_root / 'modules/03_source_reconstruction/src'))
sys.path.insert(0, str(_repo_root / 'modules/04_ais_trajectory/src'))
_injected_paths = True


import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr

# Ensure OceanTrace is in path
sys.path.append(str(Path(r"c:\Users\kusha\Desktop\folder\coding\OceanTrace")))

from evidence_fusion import (
    evaluate_candidates, generate_report, EvidenceWeights
)
from source_reconstruction import SourceReconstruction, Uncertainty
from ais_trajectory import AISDataset, LoadReport

def create_synthetic_ais():
    # Vessel 1: right in the origin at the right time
    # Vessel 2: passes by, but outside
    # Vessel 3: inside but has huge gap
    
    t_start = datetime(2021, 8, 29, 0, 0, 0)
    
    records = []
    
    # Vessel 1 (MMSI 111): good coverage, near (10.0, 10.0), SOG changes
    for i in range(20):
        t = t_start + timedelta(hours=i - 5)
        lon = 10.0 + (i - 10) * 0.001
        lat = 10.0
        sog = 10.0 if i < 8 or i > 12 else 2.0
        records.append({
            'mmsi': '111', 'time': t, 'lat': lat, 'lon': lon, 'sog': sog, 'cog': 90.0, 'heading': 90.0,
            'vessel_name': 'Test Ship 1', 'vessel_type': 'Tanker', 'length': 200, 'cargo': 'Oil'
        })
        
    # Vessel 2 (MMSI 222): far away (11.0, 11.0)
    for i in range(10):
        t = t_start + timedelta(hours=i)
        records.append({
            'mmsi': '222', 'time': t, 'lat': 11.0, 'lon': 11.0, 'sog': 15.0, 'cog': 180.0, 'heading': 180.0,
            'vessel_name': 'Far Ship', 'vessel_type': 'Cargo', 'length': 150, 'cargo': None
        })
        
    df = pd.DataFrame(records)
    report = LoadReport(
        rows_read=len(df), rows_dropped_missing_core_fields=0, rows_dropped_invalid_ranges=0,
        exact_duplicate_rows_dropped=0, duplicate_mmsi_timestamp_rows=0, rows_final=len(df),
        n_vessels=2, time_range=(df['time'].min(), df['time'].max()),
        spatial_bounds=(df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max())
    )
    
    # Needs source_row column as created by load_ais_csv
    df['source_row'] = range(len(df))
    
    return AISDataset(df, report)


def create_synthetic_recon():
    t_origin = datetime(2021, 8, 29, 0, 0, 0)
    t_obs = datetime(2021, 8, 29, 12, 0, 0)
    
    # Fake trajectory Dataset (descending time)
    times = [t_obs, t_origin]
    
    # 2 particles
    lon_data = np.array([[10.0, 10.0], [10.01, 10.01]])
    lat_data = np.array([[10.0, 10.0], [10.0, 10.0]])
    
    ds = xr.Dataset(
        data_vars=dict(
            lon=(['trajectory', 'time'], lon_data),
            lat=(['trajectory', 'time'], lat_data),
        ),
        coords=dict(
            time=times,
            trajectory=[0, 1],
        )
    )
    
    unc = Uncertainty(
        semi_major_m=5000.0, semi_minor_m=5000.0, orientation_deg=0.0,
        radius_68_m=5000.0, radius_95_m=10000.0, method='isotropic_fallback'
    )
    
    recon = SourceReconstruction(
        observation_time=t_obs,
        origin_time=t_origin,
        search_window_hours=12.0,
        origin_centroid=(10.0, 10.0),
        origin_positions={'lon': np.array([10.0, 10.01]), 'lat': np.array([10.0, 10.0])},
        uncertainty=unc,
        trajectories=ds,
        n_particles_seeded=2,
        n_particles_valid=2,
        env_mode='synthetic',
        env_kwargs={}
    )
    return recon

def main():
    print("Running Evidence Fusion Tests...")
    ais = create_synthetic_ais()
    recon = create_synthetic_recon()
    
    # Radius will be 10km (95th radius)
    result = evaluate_candidates(recon, ais)
    
    print(f"\nCandidates found: {len(result.candidates)}")
    for c in result.candidates:
        print(f"MMSI: {c.mmsi}, Score: {c.score:.2f}, Category: {c.category}")
        for f in c.features:
            print(f"  - {f.name}: {f.value:.2f}")
    
    report = generate_report(result)
    print("\nReport:")
    print(report)

if __name__ == "__main__":
    main()