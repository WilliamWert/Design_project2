"""
ingest_pjm.py
-------------
Downloads hourly electricity demand for the PJM interconnection from the
U.S. Energy Information Administration (EIA) Open Data API v2 and upserts
each hour as a document in MongoDB Atlas.

Data source:
    EIA API v2 — Hourly Electric Grid Monitor
    https://www.eia.gov/opendata/

Free API key registration (instant, no credit card):
    https://www.eia.gov/opendata/register.php

Each MongoDB document has the schema:
    {
        "datetime":    "2024-01-15T14:00:00Z",  # UTC ISO-8601
        "region":      "PJM",
        "demand_mw":   89231.0,                 # megawatt-hours ≈ MW for hourly data
        "ingested_at": "2026-04-29T03:00:00Z"
    }

Calendar and lag features are added by build_features.py.

Usage:
    python ingest_pjm.py --start 2016-01-01 --end 2024-12-31

    # Or set the EIA key and Mongo URI as environment variables:
    export EIA_API_KEY="your_key_here"
    export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/"
    python ingest_pjm.py --start 2016-01-01 --end 2024-12-31

Requirements:
    pip install pymongo[srv] requests certifi tqdm
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import certifi
import requests
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ---------------------------------------------------------------------------
# Logging — stdout and logs/ingest_pjm.log
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
# Configuration — set as env vars or edit defaults below
# ---------------------------------------------------------------------------
EIA_API_KEY = os.getenv("EIA_API_KEY", "YOUR_EIA_API_KEY")   # <-- paste your key here
MONGO_URI   = os.getenv("MONGO_URI",   "YOUR_MONGODB_ATLAS_URI")
DB_NAME     = "energy_forecast"
COLLECTION  = "pjm_hourly"

# EIA API v2 — Hourly Electric Grid Monitor, regional demand
EIA_URL     = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
RESPONDENT  = "PJM"       # Full PJM Interconnection
DATA_TYPE   = "D"         # D = Demand
PAGE_SIZE   = 5000        # EIA max rows per request
CHUNK_DAYS  = 90          # date window per request (EIA handles larger windows fine)
RATE_LIMIT  = 0.5         # seconds between API pages


# ---------------------------------------------------------------------------
# Helper: fetch one page of EIA hourly demand data
# ---------------------------------------------------------------------------
def fetch_eia_page(start: str, end: str, offset: int = 0) -> dict:
    """
    Call the EIA API v2 for one page of hourly PJM demand data.

    Parameters
    ----------
    start  : str  Start in EIA format "YYYY-MM-DDTHH", e.g. "2024-01-01T00"
    end    : str  End   in EIA format "YYYY-MM-DDTHH", e.g. "2024-03-31T23"
    offset : int  Pagination offset (0-based)

    Returns
    -------
    dict  Full EIA API JSON response
    """
    params = {
        "api_key":               EIA_API_KEY,
        "frequency":             "hourly",
        "data[0]":               "value",
        "facets[respondent][]":  RESPONDENT,
        "facets[type][]":        DATA_TYPE,
        "start":                 start,
        "end":                   end,
        "length":                PAGE_SIZE,
        "offset":                offset,
        "sort[0][column]":       "period",
        "sort[0][direction]":    "asc",
    }

    resp = requests.get(EIA_URL, params=params, timeout=60, verify=certifi.where())

    if resp.status_code != 200:
        log.error("HTTP %s from EIA. Body: %s", resp.status_code, resp.text[:400])
        resp.raise_for_status()

    payload = resp.json()

    # Surface any EIA-level error messages
    if "error" in payload:
        log.error("EIA API error: %s", payload["error"])
        raise RuntimeError(f"EIA API error: {payload['error']}")

    return payload


# ---------------------------------------------------------------------------
# Helper: convert one EIA row into a MongoDB document
# ---------------------------------------------------------------------------
def build_doc(row: dict) -> dict:
    """
    Convert a raw EIA API row into the project document schema.

    EIA hourly periods are formatted as "YYYY-MM-DDTHH" in local time (ET).
    We store as UTC (EIA Eastern times are UTC-5/UTC-4 depending on DST,
    but for this project we treat the period label as the hour identifier
    and store it as-is with a Z suffix, consistent across the full dataset).

    Parameters
    ----------
    row : dict  One item from the EIA response data array

    Returns
    -------
    dict  MongoDB document ready for upsert
    """
    period = row.get("period", "")           # e.g. "2024-01-15T14"
    demand = row.get("value")

    # Normalise to full ISO-8601 UTC string
    dt_str = period + ":00:00Z" if len(period) == 13 else period + "Z"

    return {
        "datetime":     dt_str,
        "region":       RESPONDENT,
        "demand_mw":    float(demand) if demand is not None else None,
        "data_quality": None if demand is not None else "missing",
        "ingested_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------
def ingest(start_date: str, end_date: str) -> None:
    """
    Ingest hourly PJM demand data for the given date range into MongoDB Atlas.

    Parameters
    ----------
    start_date : str  "YYYY-MM-DD" inclusive start
    end_date   : str  "YYYY-MM-DD" inclusive end
    """
    if EIA_API_KEY == "YOUR_EIA_API_KEY":
        log.error(
            "EIA API key not set. Register free at https://www.eia.gov/opendata/register.php "
            "then set EIA_API_KEY env var or paste your key into this script."
        )
        return

    # Connect to MongoDB
    log.info("Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000,
                             tlsCAFile=certifi.where())
        client.admin.command("ping")
        log.info("Connected successfully.")
    except Exception as e:
        log.error("MongoDB connection failed: %s: %s", type(e).__name__, e)
        raise

    collection = client[DB_NAME][COLLECTION]
    collection.create_index("datetime", unique=True)
    log.info("Unique index on 'datetime' confirmed.")

    start   = datetime.strptime(start_date, "%Y-%m-%d")
    end     = datetime.strptime(end_date,   "%Y-%m-%d")
    current = start
    total_upserted = 0

    while current < end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end)

        # EIA format: "YYYY-MM-DDTHH"
        eia_start = current.strftime("%Y-%m-%dT00")
        eia_end   = (chunk_end - timedelta(hours=1)).strftime("%Y-%m-%dT23")

        log.info("Fetching %s → %s ...", eia_start, eia_end)

        offset     = 0
        chunk_docs = []

        while True:
            try:
                payload = fetch_eia_page(eia_start, eia_end, offset)
            except Exception as e:
                log.error(
                    "Error fetching %s → %s (offset %d): %s: %s",
                    eia_start, eia_end, offset, type(e).__name__, e,
                )
                break

            response_body = payload.get("response", {})
            rows          = response_body.get("data", [])
            total_rows    = int(response_body.get("total", 0))

            if not rows:
                log.warning("No rows returned. Response body: %s", str(response_body)[:300])
                break

            for r in rows:
                try:
                    chunk_docs.append(build_doc(r))
                except Exception as e:
                    log.warning("Skipping malformed row %s: %s", r, e)

            log.info("  Page offset=%d | got %d rows | total=%d", offset, len(rows), total_rows)

            offset += PAGE_SIZE
            if offset >= total_rows:
                break
            time.sleep(RATE_LIMIT)

        # Bulk upsert
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
                    "  Upserted %d docs (%d new, %d updated)",
                    n, result.upserted_count, result.modified_count,
                )
            except BulkWriteError as e:
                log.error("Bulk write error: %s", e.details)
        else:
            log.warning("  No documents loaded for this window.")

        current = chunk_end
        time.sleep(RATE_LIMIT)

    log.info("Ingestion complete. Total documents upserted/updated: %d", total_upserted)
    client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest EIA hourly PJM demand into MongoDB Atlas"
    )
    parser.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-12-31", help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    log.info("Starting EIA/PJM ingestion: %s → %s", args.start, args.end)
    ingest(args.start, args.end)
    log.info("Done.")
