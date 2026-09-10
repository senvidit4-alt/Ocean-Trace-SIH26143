import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import importlib

import xarray as xr
import pandas as pd
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent

# Safely import the modules which have numbers in their directory names
sys.path.insert(0, str(_repo_root / 'modules/02_drift/03_source_reconstruction/src'))
sys.path.insert(0, str(_repo_root / 'modules/03_attribution/04_ais_trajectory/src'))
sys.path.insert(0, str(_repo_root / 'modules/03_attribution/05_evidence_fusion/src'))

from source_reconstruction import SourceReconstruction, Uncertainty
from evidence_fusion import (
    AttributionResult, CandidateAssessment, EvidenceFeature, DataConfidence, MetadataContext, TimelineEvent
)


# --- Module 1: Detection Adapter ---

def create_detection_json(
    polygon: dict,
    observation_time: datetime,
    area_km2: float = None,
    confidence: float = None,
    age_hrs: float = None,
    metadata: dict = None,
    filepath: str = "detection.json"
):
    """
    Creates and saves a detection.json contract.
    """
    data = {
        "contract_version": "1.0",
        "observation_time": observation_time.isoformat() + "Z" if not observation_time.tzinfo else observation_time.isoformat(),
        "polygon": polygon,
    }
    if area_km2 is not None:
        data["area_km2"] = area_km2
    if confidence is not None:
        data["confidence"] = confidence
    if age_hrs is not None:
        data["age_hrs"] = age_hrs
    if metadata is not None:
        data["metadata"] = metadata
        
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_detection_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


# --- Module 2/3: Drift Adapter ---

def save_reconstruction_json(recon: SourceReconstruction, json_filepath: str, nc_filepath: str = None):
    """
    Saves SourceReconstruction to reconstruction.json.
    Dumps the trajectories to a companion .nc file and references it via a relative path.
    """
    jpath = Path(json_filepath)
    if nc_filepath is None:
        nc_filepath = jpath.with_suffix('.nc')
    else:
        nc_filepath = Path(nc_filepath)
        
    # Clean dataset attributes before saving
    ds_to_save = recon.trajectories.copy()
    for k, v in ds_to_save.attrs.items():
        if isinstance(v, type):
            ds_to_save.attrs[k] = str(v)
    for var in ds_to_save.variables:
        for k, v in ds_to_save[var].attrs.items():
            if isinstance(v, type):
                ds_to_save[var].attrs[k] = str(v)
                
    # Save the xarray dataset to netCDF
    ds_to_save.to_netcdf(nc_filepath)
    
    # Compute relative path from json directory to nc file
    try:
        rel_nc_path = nc_filepath.relative_to(jpath.parent)
    except ValueError:
        rel_nc_path = nc_filepath # fallback to absolute if not relative
        
    data = recon.to_dict()
    data['contract_version'] = "1.0"
    data['trajectory_file_path'] = str(rel_nc_path)
    
    # Convert numpy arrays in origin_positions to lists
    data['origin_positions']['lon'] = data['origin_positions']['lon'].tolist()
    data['origin_positions']['lat'] = data['origin_positions']['lat'].tolist()
    
    # Convert uncertainty dataclass to dict
    
    
    with open(jpath, 'w') as f:
        json.dump(data, f, indent=2)


def load_reconstruction_json(json_filepath: str) -> SourceReconstruction:
    jpath = Path(json_filepath)
    with open(jpath, 'r') as f:
        data = json.load(f)
        
    nc_rel_path = data.pop('trajectory_file_path')
    data.pop('contract_version', None)
    nc_full_path = jpath.parent / nc_rel_path
    
    ds = xr.open_dataset(nc_full_path)
    
    # Parse datetimes
    obs_time = pd.Timestamp(data['observation_time']).to_pydatetime()
    orig_time = pd.Timestamp(data['origin_time']).to_pydatetime()
    
    unc = Uncertainty(**data['uncertainty'])
    
    origin_positions = {
        'lon': np.array(data['origin_positions']['lon']),
        'lat': np.array(data['origin_positions']['lat'])
    }
    
    return SourceReconstruction(
        observation_time=obs_time,
        origin_time=orig_time,
        search_window_hours=data['search_window_hours'],
        origin_centroid=tuple(data['origin_centroid']),
        origin_positions=origin_positions,
        uncertainty=unc,
        trajectories=ds,
        n_particles_seeded=data['n_particles_seeded'],
        n_particles_valid=data['n_particles_valid'],
        env_mode=data['env_mode'],
        env_kwargs=data['env_kwargs'],
        coverage_warnings=data.get('coverage_warnings', [])
    )


# --- Module 4/5: Attribution Adapter ---

def save_attribution_json(result: AttributionResult, filepath: str):
    data = {
        "contract_version": "1.0",
        "analysis_metadata": {
            "search_radius_km": result.search_radius_km,
            "search_time_window": [
                result.search_time_window[0].isoformat() + "Z" if not result.search_time_window[0].tzinfo else result.search_time_window[0].isoformat(),
                result.search_time_window[1].isoformat() + "Z" if not result.search_time_window[1].tzinfo else result.search_time_window[1].isoformat()
            ],
            "n_vessels_considered": result.n_vessels_considered,
            "caveats": result.caveats
        },
        "candidates": []
    }
    
    for c in result.candidates:
        c_dict = {
            "mmsi": c.mmsi,
            "vessel_name": c.vessel_name,
            "category": c.category,
            "score": c.score,
            "features": [f.__dict__ for f in c.features],
            "data_confidence": {
                "coverage_fraction": c.data_confidence.coverage_fraction,
                "note": c.data_confidence.note
            }
        }
        data['candidates'].append(c_dict)
        
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
