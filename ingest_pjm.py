"""
ingest_pjm.py
-------------
Downloads hourly actual metered load for the PJM East (PJM_E) zone from the
PJM Data Miner 2 REST API and upserts each hour as a document in MongoDB Atlas.

Usage:
    python ingest_pjm.py --start 2014-01-01 --end 2024-12-31

Requirements:
    pip install pymongo[srv] requests python-dateutil tqdm
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import requests
from dateutil.parser import parse as parse_dt
from pymongo import MongoClient, UpdateOne

# ---------------------------------------------------------------------------
# Configuration — set via environment variables or edit defaults below
# ---------------------------------------------------------------------------
MONGO_URI   = "mongodb+srv://williamcwert_db_user:KsjMkB8lcwSg9NcM@cluster0.v51ynam.mongodb.net/"
DB_NAME     = "energy_forecast"
COLLECTION  = "pjm_hourly"

PJM_API_BASE = "https://dataminer2.pjm.com/feed/hrl_load_metered/"
ZONE         = "PJM_E"
MAX_ROWS     = 50_000        # PJM DM2 page size limit
RATE_LIMIT_S = 1.0           # seconds to wait between API pages

# ---------------------------------------------------------------------------
# Helper: fetch one page of PJM hourly load data
# ---------------------------------------------------------------------------
def fetch_pjm_page(start_ept: str, end_ept: str, start_row: int = 1) -> dict:
    """
    Call the PJM Data Miner 2 hrl_load_metered feed.

    Parameters
    ----------
    start_ept : str  e.g. "2024-01-01T00:00"
    end_ept   : str  e.g. "2024-01-08T00:00"
    start_row : int  1-based row offset for pagination

    Returns
    -------
    dict with keys 'totalRows', 'data' (list of row dicts)
    """
    params = {
        "startRow":                 start_row,
        "rowCount":                 MAX_ROWS,
        "datetime_beginning_ept":   start_ept,
        "datetime_ending_ept":      end_ept,
        "fields":                   "datetime_beginning_utc,area_load_mw",
        "area":                     ZONE,
        "format":                   "json",
    }
    resp = requests.get(PJM_API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helper: build a MongoDB document from one PJM row
# ---------------------------------------------------------------------------
def build_doc(row: dict) -> dict:
    """
    Convert a raw PJM API row into the project document schema.
    Only the fields managed by this script are set here; calendar/lag
    features are added later by build_features.py.
    """
    dt_utc = parse_dt(row["datetime_beginning_utc"]).replace(tzinfo=timezone.utc)
    return {
        "datetime":     dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "region":       ZONE,
        "demand_mw":    float(row["area_load_mw"]) if row.get("area_load_mw") is not None else None,
        "ingested_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------
def ingest(start_date: str, end_date: str):
    client     = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]

    # Unique index on datetime so upserts are idempotent
    collection.create_index("datetime", unique=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    # Chunk into 30-day windows to stay well within PJM's row-count limit
    chunk_days = 30
    current    = start
    total_upserted = 0

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        start_ept = current.strftime("%Y-%m-%dT00:00")
        end_ept   = chunk_end.strftime("%Y-%m-%dT00:00")

        print(f"  Fetching {start_ept} → {end_ept} ...", end=" ", flush=True)

        start_row = 1
        chunk_docs = []

        while True:
            data = fetch_pjm_page(start_ept, end_ept, start_row)
            rows = data.get("data", [])
            chunk_docs.extend([build_doc(r) for r in rows])

            total_rows = data.get("totalRows", 0)
            if start_row + MAX_ROWS - 1 >= total_rows:
                break
            start_row += MAX_ROWS
            time.sleep(RATE_LIMIT_S)

        # Bulk upsert — match on datetime, replace full document
        if chunk_docs:
            ops = [
                UpdateOne({"datetime": d["datetime"]}, {"$set": d}, upsert=True)
                for d in chunk_docs
            ]
            result = collection.bulk_write(ops, ordered=False)
            total_upserted += result.upserted_count + result.modified_count
            print(f"{len(chunk_docs)} rows  ({result.upserted_count} new, {result.modified_count} updated)")
        else:
            print("0 rows")

        current = chunk_end
        time.sleep(RATE_LIMIT_S)

    print(f"\nDone. Total documents upserted/updated: {total_upserted:,}")
    client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PJM hourly load into MongoDB")
    parser.add_argument("--start", default="2014-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-12-31", help="End date   YYYY-MM-DD")
    args = parser.parse_args()
    ingest(args.start, args.end)
