"""
build_features.py
-----------------
Adds derived calendar and lag features to every document in the pjm_hourly
MongoDB collection.  Run this AFTER ingest_pjm.py and ingest_weather.py have
finished loading the raw load and temperature data.

Features added per document
---------------------------
  hour_of_day     int   0–23   (derived from datetime)
  day_of_week     int   0–6    (0 = Monday)
  month           int   1–12
  is_holiday      bool         US federal holiday flag
  demand_lag_24h  float        demand_mw from exactly 24 h prior (None if missing)
  demand_lag_168h float        demand_mw from exactly 168 h prior (None if missing)

Usage:
    python build_features.py

Requirements:
    pip install pymongo[srv] holidays tqdm
"""

from datetime import datetime, timedelta, timezone

import holidays
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI  = "YOUR_MONGODB_ATLAS_URI"
DB_NAME    = "energy_forecast"
COLLECTION = "pjm_hourly"

US_HOLIDAYS = holidays.US()   # federal holiday calendar


# ---------------------------------------------------------------------------
# Helper: parse datetime string to UTC datetime object
# ---------------------------------------------------------------------------
def parse_utc(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper: format datetime back to the document key string
# ---------------------------------------------------------------------------
def fmt_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Main feature-engineering pass
# ---------------------------------------------------------------------------
def build_features():
    client     = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]

    # Load all datetime -> demand_mw pairs into memory for O(1) lag lookups.
    # ~87,600 documents × ~50 bytes each ≈ 4 MB — easily fits in RAM.
    print("Loading demand index from MongoDB ...", flush=True)
    demand_index: dict[str, float | None] = {
        doc["datetime"]: doc.get("demand_mw")
        for doc in collection.find({}, {"datetime": 1, "demand_mw": 1, "_id": 0})
    }
    print(f"  {len(demand_index):,} documents indexed")

    # Stream all documents, compute features, batch-write updates
    cursor     = collection.find({}, {"datetime": 1, "_id": 1})
    batch      = []
    BATCH_SIZE = 500
    total_updated = 0

    for doc in tqdm(cursor, total=len(demand_index), desc="Computing features"):
        dt_str = doc.get("datetime")
        if not dt_str:
            continue

        dt = parse_utc(dt_str)

        # Calendar features
        hour_of_day = dt.hour
        day_of_week = dt.weekday()          # 0 = Monday, 6 = Sunday
        month       = dt.month
        is_holiday  = dt.date() in US_HOLIDAYS

        # Lag features (24 h and 168 h = 1 week)
        lag_24h_key  = fmt_utc(dt - timedelta(hours=24))
        lag_168h_key = fmt_utc(dt - timedelta(hours=168))

        demand_lag_24h  = demand_index.get(lag_24h_key)    # None if not in index
        demand_lag_168h = demand_index.get(lag_168h_key)

        update_fields = {
            "hour_of_day":     hour_of_day,
            "day_of_week":     day_of_week,
            "month":           month,
            "is_holiday":      is_holiday,
            "demand_lag_24h":  demand_lag_24h,
            "demand_lag_168h": demand_lag_168h,
        }

        batch.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": update_fields})
        )

        if len(batch) >= BATCH_SIZE:
            result = collection.bulk_write(batch, ordered=False)
            total_updated += result.modified_count
            batch = []

    # Flush remaining
    if batch:
        result = collection.bulk_write(batch, ordered=False)
        total_updated += result.modified_count

    print(f"\nDone. Features added to {total_updated:,} documents.")
    client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_features()
