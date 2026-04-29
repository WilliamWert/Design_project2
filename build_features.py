"""
build_features.py
-----------------
Adds derived calendar and lag features to every document in the pjm_hourly
MongoDB collection. Run this AFTER ingest_pjm.py has finished.

Features added per document
---------------------------
  hour_of_day     int   0-23
  day_of_week     int   0-6  (0 = Monday, 6 = Sunday)
  month           int   1-12
  year            int   e.g. 2024
  season          str   "winter" | "spring" | "summer" | "fall"
  is_holiday      bool  True if US federal holiday
  demand_lag_24h  float demand_mw from 24 hours prior  (None if unavailable)
  demand_lag_168h float demand_mw from 168 hours prior (None if unavailable)

Usage:
    python build_features.py

Requirements:
    pip install pymongo[srv] holidays tqdm
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import holidays
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging configuration — writes to logs/build_features.log
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/build_features.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI   = os.getenv("MONGO_URI", "YOUR_MONGODB_ATLAS_URI")
DB_NAME     = "energy_forecast"
COLLECTION  = "pjm_hourly"
BATCH_SIZE  = 500   # documents per bulk_write call

US_HOLIDAYS = holidays.US()


# ---------------------------------------------------------------------------
# Helper: parse ISO-8601 UTC string to datetime
# ---------------------------------------------------------------------------
def parse_utc(dt_str: str) -> datetime:
    """Parse a UTC datetime string of the form 'YYYY-MM-DDTHH:MM:SSZ'."""
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper: format a datetime back to the document key string
# ---------------------------------------------------------------------------
def fmt_utc(dt: datetime) -> str:
    """Format a UTC datetime as the canonical document key string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Helper: derive meteorological season from month number
# ---------------------------------------------------------------------------
def get_season(month: int) -> str:
    """Return the meteorological season name for a given month (1-12)."""
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"


# ---------------------------------------------------------------------------
# Main feature-engineering pass
# ---------------------------------------------------------------------------
def build_features() -> None:
    """
    Read all documents from pjm_hourly, compute calendar and lag features,
    and write them back to MongoDB via batched bulk updates.

    The function builds an in-memory demand index (datetime -> demand_mw) to
    allow O(1) lag lookups without additional DB queries.
    """
    log.info("Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
        client.admin.command("ping")
        log.info("Connected to MongoDB Atlas successfully.")
    except Exception as e:
        log.error("Failed to connect to MongoDB: %s", e)
        raise

    collection = client[DB_NAME][COLLECTION]

    # Build an in-memory index of datetime -> demand_mw for lag lookups.
    # ~87,600 documents × ~60 bytes ≈ 5 MB — safely fits in RAM.
    log.info("Loading demand index from MongoDB for lag feature computation...")
    demand_index: Dict[str, Optional[float]] = {
        doc["datetime"]: doc.get("demand_mw")
        for doc in collection.find({}, {"datetime": 1, "demand_mw": 1, "_id": 0})
    }
    total_docs = len(demand_index)
    log.info("Demand index loaded: %d documents.", total_docs)

    # Stream all documents, compute features, batch-write updates
    cursor = collection.find({}, {"datetime": 1, "_id": 1})
    batch = []
    total_updated = 0

    for doc in tqdm(cursor, total=total_docs, desc="Computing features"):
        dt_str = doc.get("datetime")
        if not dt_str:
            log.warning("Document %s has no 'datetime' field — skipping.", doc["_id"])
            continue

        try:
            dt = parse_utc(dt_str)
        except ValueError as e:
            log.warning("Could not parse datetime '%s': %s — skipping.", dt_str, e)
            continue

        # Calendar features
        hour_of_day = dt.hour
        day_of_week = dt.weekday()       # 0 = Monday
        month       = dt.month
        year        = dt.year
        season      = get_season(month)
        is_holiday  = dt.date() in US_HOLIDAYS

        # Lag features
        demand_lag_24h  = demand_index.get(fmt_utc(dt - timedelta(hours=24)))
        demand_lag_168h = demand_index.get(fmt_utc(dt - timedelta(hours=168)))

        batch.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {
                    "hour_of_day":     hour_of_day,
                    "day_of_week":     day_of_week,
                    "month":           month,
                    "year":            year,
                    "season":          season,
                    "is_holiday":      is_holiday,
                    "demand_lag_24h":  demand_lag_24h,
                    "demand_lag_168h": demand_lag_168h,
                }},
            )
        )

        # Flush batch when it reaches BATCH_SIZE
        if len(batch) >= BATCH_SIZE:
            try:
                result = collection.bulk_write(batch, ordered=False)
                total_updated += result.modified_count
            except BulkWriteError as e:
                log.error("Bulk write error: %s", e.details)
            batch = []

    # Flush any remaining operations
    if batch:
        try:
            result = collection.bulk_write(batch, ordered=False)
            total_updated += result.modified_count
        except BulkWriteError as e:
            log.error("Bulk write error (final flush): %s", e.details)

    log.info("Feature engineering complete. Documents updated: %d / %d", total_updated, total_docs)
    client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting build_features.py")
    build_features()
    log.info("Done.")
