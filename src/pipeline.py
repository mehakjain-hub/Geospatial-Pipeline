"""
Detects GPS data quality issues in food delivery telemetry:
  • Null Island coordinates (GPS hardware failure returning 0,0)
  • Impossible delivery speeds (teleportation anomalies)
  • Negative / out-of-range coordinate signs (data corruption)
  • Missing critical fields
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

# 1. HAVERSINE DISTANCE

def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Vectorised great-circle distance via the Haversine formula.
    Parameters
    ----------
    lat1, lon1 : origin coordinates (degrees)
    lat2, lon2 : destination coordinates (degrees)
    Returns
    -------
    np.ndarray : distances in kilometres
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# 2. NULL ISLAND DETECTOR

def detect_null_island(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Flag rows where GPS coordinates are at or near (0, 0) — 'Null Island'.
    Parameters
    ----------
    df        : DataFrame with Restaurant_latitude/longitude and
                Delivery_location_latitude/longitude columns
    threshold : coordinate absolute value below which a ping is flagged
                (default 0.01° ≈ 1.1 km from origin)
    """
    df = df.copy()
    rest_null = (df["Restaurant_latitude"].abs() < threshold) | \
                (df["Restaurant_longitude"].abs() < threshold)
    dlv_null  = (df["Delivery_location_latitude"].abs() < threshold) | \
                (df["Delivery_location_longitude"].abs() < threshold)
    df["null_island"] = rest_null | dlv_null
    return df

# 3. COORDINATE RANGE VALIDATOR

def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows with:
      • Null/NaN coordinates
      • Latitude outside ±90° or longitude outside ±180°
      • Negative coordinates for Indian cities
        (all major Indian cities have lat ∈ [8, 37], lon ∈ [68, 97])
    """
    df = df.copy()
    def _invalid(lat_col, lon_col):
        null_mask = df[lat_col].isna() | df[lon_col].isna()
        range_mask = (df[lat_col].abs() > 90) | (df[lon_col].abs() > 180)
        neg_mask = (df[lat_col] < 0) | (df[lon_col] < 0)
        return null_mask | range_mask | neg_mask

    df["invalid_restaurant_coords"]  = _invalid("Restaurant_latitude", "Restaurant_longitude")
    df["invalid_delivery_coords"]    = _invalid("Delivery_location_latitude", "Delivery_location_longitude")
    df["invalid_coords"]             = df["invalid_restaurant_coords"] | df["invalid_delivery_coords"]
    return df

# 4. TELEPORTATION / SPEED ANOMALY DETECTOR

def detect_speed_anomalies(df: pd.DataFrame, max_speed_kmh: float = 80) -> pd.DataFrame:
    """
    Flag deliveries where the implied average speed between restaurant
    and delivery location exceeds a physical threshold.
    Parameters
    ----------
    df            : DataFrame with coordinate and time columns
    max_speed_kmh : speed ceiling (default 80 km/h)
    """
    df = df.copy()
    valid_mask = (
        ~df["invalid_coords"] &
        (df["Time_taken (min)"] > 0)
    )

    df["delivery_distance_km"] = np.nan
    df["implied_speed_kmh"]    = np.nan
    df["speed_anomaly"]        = False

    if valid_mask.any():
        v = df[valid_mask]
        dist = haversine_distance(
            v["Restaurant_latitude"].values,
            v["Restaurant_longitude"].values,
            v["Delivery_location_latitude"].values,
            v["Delivery_location_longitude"].values,
        )
        speed = dist / (v["Time_taken (min)"].values / 60.0)

        df.loc[valid_mask, "delivery_distance_km"] = np.round(dist, 3)
        df.loc[valid_mask, "implied_speed_kmh"]    = np.round(speed, 2)
        df.loc[valid_mask, "speed_anomaly"]        = speed > max_speed_kmh

    return df
  
# 5. MISSING CRITICAL FIELDS

def detect_missing_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows missing fields that are critical for operational analytics:
    delivery person age, ratings, order time, city, or festival flag.
    """
    df = df.copy()
    critical = ["Delivery_person_Age", "Delivery_person_Ratings",
                "Time_Orderd", "City", "Weather_conditions"]
    df["missing_critical_fields"] = df[critical].isnull().any(axis=1)
    df["missing_field_count"]     = df[critical].isnull().sum(axis=1)
    return df

# 6. PREP / LOADER

def load_and_prep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "ID", "Delivery_person_ID",
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
        "Time_taken (min)"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    return df
  
# 7. HEALTH REPORT

def generate_health_report(df: pd.DataFrame, output_dir: str = "reports") -> dict:
    """
    Compute a data health summary and write it as a JSON audit report.
    """
    os.makedirs(output_dir, exist_ok=True)

    total          = len(df)
    null_island    = int(df.get("null_island",               pd.Series(False)).sum())
    invalid_coords = int(df.get("invalid_coords",            pd.Series(False)).sum())
    speed_anom     = int(df.get("speed_anomaly",             pd.Series(False)).sum())
    missing_fields = int(df.get("missing_critical_fields",   pd.Series(False)).sum())

    # Union of all flagged rows (a row may have multiple issues)
    flag_cols = ["null_island", "invalid_coords", "speed_anomaly", "missing_critical_fields"]
    present   = [c for c in flag_cols if c in df.columns]
    any_flag  = df[present].any(axis=1).sum() if present else 0
    health_pct = round(100 * (1 - any_flag / max(total, 1)), 2)

    # Per-vehicle and per-city breakdown
    vehicle_anomalies = {}
    if "Type_of_vehicle" in df.columns and "speed_anomaly" in df.columns:
        vehicle_anomalies = (
            df[df["speed_anomaly"]]
            .groupby("Type_of_vehicle")
            .size()
            .to_dict()
        )
    city_health = {}
    if "City" in df.columns:
        city_health = (
            df.groupby("City")
            .apply(lambda g: round(100 * (1 - g[present].any(axis=1).sum() / max(len(g), 1)), 2))
            .to_dict()
        )

    report = {
        "generated_at":              datetime.utcnow().isoformat() + "Z",
        "dataset":                   "Zomato Delivery Operations Analytics",
        "total_records":             total,
        "unique_delivery_persons":   df["Delivery_person_ID"].nunique(),
        "null_island_records":       null_island,
        "invalid_coordinate_records": invalid_coords,
        "speed_anomaly_records":     speed_anom,
        "missing_critical_fields":   missing_fields,
        "total_flagged_records":     int(any_flag),
        "data_health_score":         f"{health_pct}%",
        "speed_anomalies_by_vehicle": vehicle_anomalies,
        "health_score_by_city":      city_health,
    }
    out_path = os.path.join(output_dir, "health_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nHealth Report → {out_path}")
    return report

# 8. FULL PIPELINE RUNNER

def run_pipeline(
    df: pd.DataFrame,
    max_speed_kmh: float = 80,
    output_dir: str = "reports",
) -> tuple:

    print("Step 1 — Detecting Null Island coordinates.")
    df = detect_null_island(df)
    print("Step 2 — Validating coordinate ranges.")
    df = validate_coordinates(df)
    print("Step 3 — Computing delivery speeds & flagging anomalies.")
    df = detect_speed_anomalies(df, max_speed_kmh=max_speed_kmh)
    print("Step 4 — Scanning for missing critical fields.")
    df = detect_missing_fields(df)
    print("Generating health report.")
    report = generate_health_report(df, output_dir=output_dir)

    return df, report
