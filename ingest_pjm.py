"""
ingest_pjm.py
-------------
Downloads hourly actual metered load for the PJM East (PJM_E) zone from the
PJM Data Miner 2 REST API and upserts each hour as a document in MongoDB Atlas.

Each document stored has the schema:
    {
        "datetime":    "2024-01-15T14:00:00Z",   # UTC ISO-8601 timestamp
        "region":      "PJM_E",
        "demand_mw":   31842.7,
        "ingested_at": "2026-04-22T03:00:00Z"
    }

Calendar and lag features are added separately by build_features.py.

Usage:
    python ingest_pjm.py --start 2014-01-01 --end 2024-12-31

Requirements:
    pip install pymongo[srv] requests python-dateutil tqdm
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests
from dateutil.parser import parse as parse_dt
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ---------------------------------------------------------------------------
# Logging configuration — writes to logs/ingest_pjm.log
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ingest_pjm.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — replace with your values or set as environment variables
# ---------------------------------------------------------------------------
MONGO_URI   = os.getenv("MONGO_URI", "YOUR_MONGODB_ATLAS_URI")
DB_NAME     = "energy_forecast"
COLLECTION  = "pjm_hourly"

PJM_API_BASE = "https://dataminer2.pjm.com/feed/hrl_load_metered/"
ZONE         = "PJM_E"
MAX_ROWS     = 50_000        # PJM DM2 page-size limit
RATE_LIMIT_S = 1.0           # seconds between API requests to respect rate limits
CHUNK_DAYS   = 30            # days per API request window


# ---------------------------------------------------------------------------
# Helper: fetch one page of PJM hourly load data
# ---------------------------------------------------------------------------
def fetch_pjm_page(start_ept: str, end_ept: str, start_row: int = 1) -> dict:
    """
    Call the PJM Data Miner 2 hrl_load_metered feed for one page of results.

    Parameters
    ----------
    start_ept : str   Start of window in EPT, e.g. "2024-01-01T00:00"
    end_ept   : str   End of window in EPT,   e.g. "2024-02-01T00:00"
    start_row : int   1-based row offset for pagination

    Returns
    -------
    dict  PJM API JSON response with keys 'totalRows' and 'data'
    """
    params = {
        "startRow":              start_row,
        "rowCount":              MAX_ROWS,
        "datetime_beginning_ept": start_ept,
        "datetime_ending_ept":   end_ept,
        "fields":                "datetime_beginning_utc,area_load_mw",
        "area":                  ZONE,
        "format":                "json",
    }
    try:
        resp = requests.get(PJM_API_BASE, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        log.error("HTTP error fetching PJM data: %s", e)
        raise
    except requests.exceptions.ConnectionError as e:
        log.error("Connection error fetching PJM data: %s", e)
        raise
    except requests.exceptions.Timeout:
        log.error("Timeout fetching PJM data for window %s -> %s", start_ept, end_ept)
        raise


# ---------------------------------------------------------------------------
# Helper: convert one raw PJM API row into a MongoDB document
# ---------------------------------------------------------------------------
def build_doc(row: dict) -> dict:
    """
    Convert a raw PJM API row dict into the project's document schema.

    Parameters
    ----------
    row : dict   One row from PJM API 'data' array

    Returns
    -------
    dict   MongoDB document ready for upsert
    """
    dt_utc = parse_dt(row["datetime_beginning_utc"]).replace(tzinfo=timezone.utc)
    demand = row.get("area_load_mw")
    return {
        "datetime":     dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "region":       ZONE,
        "demand_mw":    float(demand) if demand is not None else None,
        "data_quality": None if demand is not None else "missing",
        "ingested_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------
def ingest(start_date: str, end_date: str) -> None:
    """
    Ingest hourly PJM load data for the given date range into MongoDB Atlas.

    Processes data in CHUNK_DAYS-day windows to stay within PJM's row limit.
    Each document is upserted so the script is safe to re-run (idempotent).

    Parameters
    ----------
    start_date : str  "YYYY-MM-DD" inclusive start
    end_date   : str  "YYYY-MM-DD" inclusive end
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

    # Create a unique index on datetime so upserts are idempotent
    collection.create_index("datetime", unique=True)
    log.info("Unique index on 'datetime' confirmed.")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    current = start
    total_upserted = 0

    while current < end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end)
        start_ept = current.strftime("%Y-%m-%dT00:00")
        end_ept   = chunk_end.strftime("%Y-%m-%dT00:00")

        log.info("Fetching window %s → %s ...", start_ept, end_ept)

        start_row  = 1
        chunk_docs = []

        # Paginate through all rows for this window
        while True:
            try:
                data = fetch_pjm_page(start_ept, end_ept, start_row)
            except Exception:
                log.warning("Skipping window %s → %s due to API error.", start_ept, end_ept)
                break

            rows = data.get("data", [])
            if not rows:
                break

            chunk_docs.extend([build_doc(r) for r in rows])

            total_rows = data.get("totalRows", 0)
            if start_row + MAX_ROWS - 1 >= total_rows:
                break
            start_row += MAX_ROWS
            time.sleep(RATE_LIMIT_S)

        # Bulk upsert — match on datetime, replace document
        if chunk_docs:
            ops = [
                UpdateOne({"datetime": d["datetime"]}, {"$set": d}, upsert=True)
                for d in chunk_docs
            ]
            try:
                result = collection.bulk_write(ops, ordered=False)
                n = result.upserted_count + result.modified_count
                total_upserted += n
                log.info(
                    "  %d rows processed (%d new, %d updated)",
                    len(chunk_docs), result.upserted_count, result.modified_count,
                )
            except BulkWriteError as e:
                log.error("Bulk write error: %s", e.details)
        else:
            log.warning("  No rows returned for window %s → %s.", start_ept, end_ept)

        current = chunk_end
        time.sleep(RATE_LIMIT_S)

    log.info("Ingestion complete. Total documents upserted/updated: %d", total_upserted)
    client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PJM hourly metered load data into MongoDB Atlas"
    )
    parser.add_argument(
        "--start", default="2014-01-01",
        help="Start date inclusive (YYYY-MM-DD). Default: 2014-01-01"
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="End date inclusive (YYYY-MM-DD). Default: 2024-12-31"
    )
    args = parser.parse_args()

    log.info("Starting PJM ingestion: %s → %s", args.start, args.end)
    ingest(args.start, args.end)
    log.info("Done.")
