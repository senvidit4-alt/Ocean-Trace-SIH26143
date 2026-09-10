import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Inject modules so adapters can import correctly
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / 'contracts'))
sys.path.insert(0, str(_repo_root / 'modules/02_drift/03_source_reconstruction/src'))
sys.path.insert(0, str(_repo_root / 'modules/03_attribution/04_ais_trajectory/src'))
sys.path.insert(0, str(_repo_root / 'modules/03_attribution/05_evidence_fusion/src'))

from adapters import (
    create_detection_json, load_detection_json,
    save_reconstruction_json, load_reconstruction_json,
    save_attribution_json
)
from source_reconstruction import SourceReconstruction, Uncertainty
from evidence_fusion import (
    AttributionResult, CandidateAssessment, EvidenceFeature, DataConfidence, MetadataContext, TimelineEvent
)
import xarray as xr


def test_detection_contract(tmp_path):
    out_file = tmp_path / "detection.json"
    poly = {
        "type": "Polygon",
        "coordinates": [[[72.2, 18.6], [72.3, 18.6], [72.3, 18.7], [72.2, 18.7], [72.2, 18.6]]]
    }
    obs_time = datetime(2026, 8, 31, 14, 0, 0)
    
    create_detection_json(
        polygon=poly,
        observation_time=obs_time,
        area_km2=10.5,
        confidence=0.92,
        age_hrs=1.5,
        metadata={"satellite": "Sentinel-1"},
        filepath=str(out_file)
    )
    
    assert out_file.exists()
    
    loaded = load_detection_json(str(out_file))
    assert loaded["contract_version"] == "1.0"
    assert loaded["observation_time"] == "2026-08-31T14:00:00Z"
    assert loaded["polygon"] == poly
    assert loaded["area_km2"] == 10.5
    assert loaded["confidence"] == 0.92
    assert loaded["age_hrs"] == 1.5
    assert loaded["metadata"] == {"satellite": "Sentinel-1"}
    print("Detection contract test passed.")


def test_reconstruction_contract(tmp_path):
    # Mock trajectories dataset
    times = [np.datetime64('2026-08-31T14:00:00'), np.datetime64('2026-08-31T08:00:00')]
    ds = xr.Dataset({
        'lon': (['trajectory', 'time'], np.array([[72.5, 72.4]])),
        'lat': (['trajectory', 'time'], np.array([[18.9, 18.8]]))
    }, coords={'time': times, 'trajectory': [0]})
    
    unc = Uncertainty(
        semi_major_m=1200.0, semi_minor_m=800.0, orientation_deg=45.0,
        radius_68_m=1000.0, radius_95_m=1500.0, method="covariance_ellipse"
    )
    
    recon = SourceReconstruction(
        observation_time=datetime(2026, 8, 31, 14, 0, 0),
        origin_time=datetime(2026, 8, 31, 8, 0, 0),
        search_window_hours=6.0,
        origin_centroid=(72.4, 18.8),
        origin_positions={'lon': np.array([72.4]), 'lat': np.array([18.8])},
        uncertainty=unc,
        trajectories=ds,
        n_particles_seeded=1,
        n_particles_valid=1,
        env_mode='netcdf',
        env_kwargs={'paths': 'dummy.nc'},
        coverage_warnings=["Test warning"]
    )
    
    out_json = tmp_path / "reconstruction.json"
    
    save_reconstruction_json(recon, str(out_json))
    
    assert out_json.exists()
    out_nc = tmp_path / "reconstruction.nc"
    assert out_nc.exists()
    
    with open(out_json, 'r') as f:
        data = json.load(f)
        assert data['trajectory_file_path'] == 'reconstruction.nc'
        assert data['contract_version'] == "1.0"
        
    loaded_recon = load_reconstruction_json(str(out_json))
    
    assert loaded_recon.search_window_hours == 6.0
    assert loaded_recon.n_particles_seeded == 1
    assert np.allclose(loaded_recon.origin_positions['lon'], [72.4])
    assert loaded_recon.uncertainty.semi_major_m == 1200.0
    assert loaded_recon.coverage_warnings == ["Test warning"]
    assert loaded_recon.trajectories.lon.shape == (1, 2)
    print("Reconstruction contract test passed.")


def test_attribution_contract(tmp_path):
    dc = DataConfidence(coverage_fraction=0.9, overlapping_gaps=[], adjacent_gaps=[], note="Good")
    mc = MetadataContext(vessel_type="Cargo", length_m=200.0, cargo="Oil")
    feat = EvidenceFeature(
        name="spatial_proximity", value=0.95, raw_value=2.0, raw_unit="km",
        weight=0.3, contribution=0.285, explanation="Close", limitation=None
    )
    
    ca = CandidateAssessment(
        mmsi="123456789", vessel_name="TEST_VESSEL", score=0.95, category="High-Priority Candidate",
        features=[feat], excluded_features=[], data_confidence=dc, metadata_context=mc,
        timeline=[], n_observations_considered=50, data_completeness_notes=[]
    )
    
    attr = AttributionResult(
        reconstruction_summary={'centroid': (72.4, 18.8)},
        search_radius_km=15.0,
        search_time_window=(datetime(2026, 8, 31, 8, 0), datetime(2026, 8, 31, 14, 0)),
        candidates=[ca],
        n_vessels_considered=1,
        caveats=["Test caveat"]
    )
    
    out_json = tmp_path / "attribution.json"
    save_attribution_json(attr, str(out_json))
    
    assert out_json.exists()
    with open(out_json, 'r') as f:
        data = json.load(f)
        assert data['contract_version'] == "1.0"
        assert data['analysis_metadata']['search_radius_km'] == 15.0
        assert data['analysis_metadata']['n_vessels_considered'] == 1
        assert len(data['candidates']) == 1
        assert data['candidates'][0]['mmsi'] == "123456789"
    assert data['candidates'][0]['features'][0]['name'] == "spatial_proximity"
    print("Attribution contract test passed.")


def test_reconstruction_contract_with_invalid_netcdf_metadata(tmp_path):
    # Mock trajectories dataset with an invalid type attribute (e.g., numpy.int32 type object)
    times = [np.datetime64('2026-08-31T14:00:00'), np.datetime64('2026-08-31T08:00:00')]
    ds = xr.Dataset({
        'lon': (['trajectory', 'time'], np.array([[72.5, 72.4]])),
        'lat': (['trajectory', 'time'], np.array([[18.9, 18.8]]))
    }, coords={'time': times, 'trajectory': [0]})
    
    # Inject the invalid attribute (type object, not instance) that OpenDrift sometimes leaves behind
    ds.attrs['invalid_dtype_attr'] = np.int32
    ds['lon'].attrs['another_invalid_attr'] = np.float64
    
    unc = Uncertainty(
        semi_major_m=1200.0, semi_minor_m=800.0, orientation_deg=45.0,
        radius_68_m=1000.0, radius_95_m=1500.0, method="covariance_ellipse"
    )
    
    recon = SourceReconstruction(
        observation_time=datetime(2026, 8, 31, 14, 0, 0),
        origin_time=datetime(2026, 8, 31, 8, 0, 0),
        search_window_hours=6.0,
        origin_centroid=(72.4, 18.8),
        origin_positions={'lon': np.array([72.4]), 'lat': np.array([18.8])},
        uncertainty=unc,
        trajectories=ds,
        n_particles_seeded=1,
        n_particles_valid=1,
        env_mode='netcdf',
        env_kwargs={'paths': 'dummy.nc'},
        coverage_warnings=[]
    )
    
    out_json = tmp_path / "reconstruction_invalid.json"
    
    # This should not raise a TypeError now
    save_reconstruction_json(recon, str(out_json))
    
    assert out_json.exists()
    out_nc = tmp_path / "reconstruction_invalid.nc"
    assert out_nc.exists()
    
    loaded_recon = load_reconstruction_json(str(out_json))
    assert str(np.int32) in loaded_recon.trajectories.attrs['invalid_dtype_attr']
    assert str(np.float64) in loaded_recon.trajectories['lon'].attrs['another_invalid_attr']
    print("Reconstruction contract invalid NetCDF metadata regression test passed.")


if __name__ == "__main__":
    # Small test wrapper since we don't have pytest available in this isolated context easily
    # We will use tmp_path logic by providing a temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        test_detection_contract(tdp)
        test_reconstruction_contract(tdp)
        test_reconstruction_contract_with_invalid_netcdf_metadata(tdp)
        test_attribution_contract(tdp)
    print("ALL CONTRACT TESTS PASSED.")
