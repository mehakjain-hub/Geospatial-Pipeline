import argparse
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import load_and_prep, run_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="Zomato Delivery Geospatial Sanitation Pipeline")
    p.add_argument("--input",  default="data/Zomato_Dataset.csv",
                   help="Path to Zomato delivery CSV")
    p.add_argument("--speed",  type=float, default=80,
                   help="Max plausible delivery speed km/h (default: 80)")
    p.add_argument("--output", default="reports",
                   help="Directory for output files (default: reports/)")
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        print("Download from: https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset")
        print("Then place it at: data/Zomato_Dataset.csv")
        sys.exit(1)

    print(f"\nLoading: {args.input}")
    df = load_and_prep(args.input)
    print(f"{len(df):,} records | {df['Delivery_person_ID'].nunique():,} delivery persons")
    print(f"Cities: {sorted(df['City'].dropna().unique().tolist())}")
    print(f"Vehicles: {sorted(df['Type_of_vehicle'].unique().tolist())}\n")

    annotated_df, report = run_pipeline(
        df,
        max_speed_kmh=args.speed,
        output_dir=args.output,
    )

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE — HEALTH SUMMARY")
    print("=" * 55)
    skip = {"speed_anomalies_by_vehicle", "health_score_by_city"}
    for k, v in report.items():
        if k not in skip:
            print(f"  {k.replace('_', ' ').title():<35} {v}")

    print("\nSpeed Anomalies by Vehicle:")
    for v, n in report.get("speed_anomalies_by_vehicle", {}).items():
        print(f"{v:<22} {n:>5} flagged")

    print("\nHealth Score by City:")
    for city, score in report.get("health_score_by_city", {}).items():
        print(f"{str(city):<18} {score}%")
    print("=" * 55)

    # ── Save annotated output ─────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    out_csv = os.path.join(args.output, "annotated_deliveries.csv")
    annotated_df.to_csv(out_csv, index=False)
    print(f"\nAnnotated data → {out_csv}")
    print(f"Health report  → {args.output}/health_report.json\n")

if __name__ == "__main__":
    main()
