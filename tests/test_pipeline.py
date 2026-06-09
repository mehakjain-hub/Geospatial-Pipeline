import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import pytest
from pipeline import (
    haversine_distance,
    detect_null_island,
    validate_coordinates,
    detect_speed_anomalies,
    detect_missing_fields,
    run_pipeline,
)

def make_row(
    rlat=18.55, rlon=73.92,
    dlat=18.65, dlon=74.01,
    time_min=20,
    age=25, rating=4.5,
    time_orderd="10:00", city="Metropolitian",
    weather="Clear",
    driver_id="DRV_001",
):
    return {
        "ID": "0x001",
        "Delivery_person_ID": driver_id,
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Restaurant_latitude": rlat,
        "Restaurant_longitude": rlon,
        "Delivery_location_latitude": dlat,
        "Delivery_location_longitude": dlon,
        "Time_Orderd": time_orderd,
        "Time_Order_picked": "10:10",
        "Order_Date": "01-06-2024",
        "Weather_conditions": weather,
        "Road_traffic_density": "Medium",
        "Vehicle_condition": 1,
        "Type_of_order": "Meal",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 1,
        "Festival": "No",
        "City": city,
        "Time_taken (min)": time_min,
    }

@pytest.fixture
def clean_df():
    return pd.DataFrame([make_row(), make_row(rlat=12.97, rlon=77.59, dlat=13.05, dlon=77.65, driver_id="DRV_002")])

@pytest.fixture
def full_anomaly_df():
    return pd.DataFrame([
        make_row(),                                          
        make_row(rlat=0.001, rlon=0.001, dlat=0.002, dlon=0.002),  
        make_row(rlat=-18.5, rlon=73.9),                    
        make_row(rlat=18.55, rlon=73.92, dlat=25.0, dlon=80.0, time_min=5),  
        make_row(age=None, city=None),                       
    ])
    
# ── Haversine ─────────────────────────────────────────────────────────────────

def test_haversine_zero_distance():
    d = haversine_distance(
        np.array([48.8566]), np.array([2.3522]),
        np.array([48.8566]), np.array([2.3522])
    )
    assert d[0] == pytest.approx(0.0, abs=1e-6)

def test_haversine_known_distance():
    # Mumbai to Pune straight-line ≈ 120 km
    d = haversine_distance(
        np.array([19.0760]), np.array([72.8777]),
        np.array([18.5204]), np.array([73.8567])
    )
    assert 110 < d[0] < 130

def test_haversine_symmetry():
    d1 = haversine_distance(np.array([18.5]), np.array([73.9]), np.array([19.1]), np.array([72.9]))
    d2 = haversine_distance(np.array([19.1]), np.array([72.9]), np.array([18.5]), np.array([73.9]))
    assert d1[0] == pytest.approx(d2[0], rel=1e-6)

def test_null_island_not_flagged_on_clean(clean_df):
    result = detect_null_island(clean_df)
    assert not result["null_island"].any()

def test_null_island_flagged():
    df = pd.DataFrame([make_row(rlat=0.001, rlon=0.001, dlat=0.002, dlon=0.002)])
    result = detect_null_island(df)
    assert result["null_island"].iloc[0]

def test_null_island_delivery_side():
    df = pd.DataFrame([make_row(dlat=0.005, dlon=0.003)])
    result = detect_null_island(df)
    assert result["null_island"].iloc[0]

# ── Coordinate Validation ─────────────────────────────────────────────────────

def test_valid_coords_not_flagged(clean_df):
    result = validate_coordinates(clean_df)
    assert not result["invalid_coords"].any()

def test_negative_lat_flagged():
    df = pd.DataFrame([make_row(rlat=-18.5)])
    result = validate_coordinates(df)
    assert result["invalid_coords"].iloc[0]

def test_out_of_range_lat_flagged():
    df = pd.DataFrame([make_row(rlat=95.0)])
    result = validate_coordinates(df)
    assert result["invalid_coords"].iloc[0]

# ── Speed Anomaly ─────────────────────────────────────────────────────────────

def test_no_speed_anomaly_on_clean(clean_df):
    df = validate_coordinates(detect_null_island(clean_df))
    result = detect_speed_anomalies(df, max_speed_kmh=80)
    assert not result["speed_anomaly"].any()

def test_speed_anomaly_flagged():
    df = pd.DataFrame([make_row(rlat=18.52, rlon=73.86, dlat=28.61, dlon=77.21, time_min=5)])
    df = validate_coordinates(detect_null_island(df))
    result = detect_speed_anomalies(df, max_speed_kmh=80)
    assert result["speed_anomaly"].iloc[0]

def test_speed_distance_computed():
    df = pd.DataFrame([make_row()])
    df = validate_coordinates(detect_null_island(df))
    result = detect_speed_anomalies(df)
    assert result["delivery_distance_km"].iloc[0] > 0
    assert result["implied_speed_kmh"].iloc[0] > 0

# ── Missing Fields ────────────────────────────────────────────────────────────

def test_no_missing_on_clean(clean_df):
    result = detect_missing_fields(clean_df)
    assert not result["missing_critical_fields"].any()

def test_missing_age_flagged():
    df = pd.DataFrame([make_row(age=None)])
    result = detect_missing_fields(df)
    assert result["missing_critical_fields"].iloc[0]
    assert result["missing_field_count"].iloc[0] == 1

# ── Full Pipeline ─────────────────────────────────────────────────────────────

def test_pipeline_on_clean_df(clean_df, tmp_path):
    _, report = run_pipeline(clean_df, output_dir=str(tmp_path))
    assert report["total_records"] == len(clean_df)
    assert "data_health_score" in report
    assert float(report["data_health_score"].strip("%")) > 90

def test_pipeline_catches_all_anomaly_types(full_anomaly_df, tmp_path):
    _, report = run_pipeline(full_anomaly_df, output_dir=str(tmp_path))
    assert report["null_island_records"] >= 1
    assert report["speed_anomaly_records"] >= 1
    assert report["missing_critical_fields"] >= 1
    assert report["total_flagged_records"] >= 1

def test_health_report_json_written(clean_df, tmp_path):
    run_pipeline(clean_df, output_dir=str(tmp_path))
    report_path = tmp_path / "health_report.json"
    assert report_path.exists()
