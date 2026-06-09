# 🛵 Zomato Delivery Geospatial Data Sanitation Pipeline

> Automated GPS anomaly detection for food delivery telemetry — built around the Zomato Delivery Operations Analytics Dataset.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)](#running-tests)
[![Dataset](https://img.shields.io/badge/dataset-Zomato%20Delivery%20Ops-orange)](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

## The Problem

Every food delivery platform processes millions of GPS records daily — and a meaningful fraction of those records are corrupt. On **45,584 Zomato delivery records**, this pipeline found:

| Issue | Count | % of records |
|---|---|---|
| Null Island coordinates (GPS returning 0,0) | 3,640 | 7.98% |
| Impossible delivery speeds (>80 km/h) | 331 | 0.73% |
| Invalid / negative coordinates | 431 | 0.95% |
| Missing critical operational fields | 3,134 | 6.87% |
| **Total flagged records** | **6,953** | **15.25%** |
| **Data health score** | | **84.75%** |

Without automated sanitation, these corrupt records flow directly into ETA models, dynamic pricing engines, delivery partner ratings, and SLA dashboards.

---

## What This Pipeline Detects

### 1. Null Island (GPS Hardware Failure)
When a GPS device loses satellite fix, it often defaults to coordinates `(0.0, 0.0)` — a point in the Atlantic Ocean off the coast of West Africa, nicknamed "Null Island." This is the single largest source of corruption in the dataset (7.98% of records).

### 2. Impossible Speed Anomalies
Using the **Haversine formula**, the pipeline computes the straight-line distance between restaurant and delivery location, then derives the implied average speed from `Time_taken (min)`. Any speed above 80 km/h for a motorcycle/scooter fleet is flagged as physically implausible.

```
d = 2R · arcsin( √[ sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2) ] )
```

### 3. Coordinate Sign Corruption
All Indian cities have latitude ∈ [8°N, 37°N] and longitude ∈ [68°E, 97°E]. Records with negative coordinates indicate a sign-flip corruption in the data pipeline.

### 4. Missing Critical Fields
Flags records where delivery person age, ratings, order time, city, or weather conditions are absent — fields required for operational analytics and ML model training.

---

## Results by City and Vehicle

| City | Health Score |
|---|---|
| Metropolitian | 87.35% |
| Urban | 86.03% |
| Semi-Urban | 84.15% |

| Vehicle | Speed Anomalies |
|---|---|
| Scooter | 155 |
| Motorcycle | 134 |
| Electric Scooter | 42 |
| Bicycle | 0 |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/mehakjain-hub/geospatial-pipeline.git
cd geospatial-pipeline

# 2. Install
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place it at:
#    data/Zomato_Dataset.csv

# 4. Run
python main.py

# Custom speed threshold
python main.py --speed 60
```

---

## Output

```
reports/
├── annotated_deliveries.csv    ← original data + flag columns
└── health_report.json          ← summary metrics
```

**Annotation columns added:**

| Column | Description |
|---|---|
| `null_island` | True if either coordinate pair is at/near (0, 0) |
| `invalid_coords` | True if coordinates are null, out-of-range, or negative |
| `delivery_distance_km` | Haversine distance restaurant → delivery |
| `implied_speed_kmh` | Average speed derived from distance and time |
| `speed_anomaly` | True if implied speed exceeds threshold |
| `missing_critical_fields` | True if any critical operational field is null |
| `missing_field_count` | Count of missing critical fields per record |

---

## Project Structure

```
geospatial-pipeline/
├── src/
│   └── pipeline.py              ← all detection logic
├── tests/
│   └── test_pipeline.py         ← 17 unit tests (pytest)
├── data/
│   └── Zomato_Dataset.csv       ← source dataset (add manually)
├── reports/                     ← generated output
├── main.py                      ← CLI entrypoint
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
# 17 passed
```

Tests cover: Haversine correctness, symmetry, known distances (Mumbai–Pune); Null Island detection on both coordinate pairs; negative/out-of-range coordinate flagging; speed anomaly detection; missing field detection; full pipeline smoke tests with anomaly assertions.

---

## Author

**Mehak Jain** — M.Sc. Mathematics & Statistics, IIT Tirupati  
[LinkedIn](https://www.linkedin.com/in/mehak-jain-901b7a229/) · [GitHub](https://github.com/mehakjain-hub)

---

## License

MIT
