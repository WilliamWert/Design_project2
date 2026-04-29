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
    pip install pymongo[srv] requests python-dateutil tqdm certifi
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import certifi
import requests
from dateutil.parser import parse as parse_dt
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ---------------------------------------------------------------------------
# Logging — writes to logs/ingest_pjm.log AND stdout
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
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI    = os.getenv("MONGO_URI", "YOUR_MONGODB_ATLAS_URI")
DB_NAME      = "energy_forecast"
COLLECTION   = "pjm_hourly"

# PJM Data Miner 2 — hourly metered load feed
PJM_API_BASE = "https://dataminer2.pjm.com/feed/hrl_load_metered/"
ZONE         = "PJM_E"
MAX_ROWS     = 50_000
RATE_LIMIT_S = 1.2    # seconds between requests
CHUNK_DAYS   = 30


# ---------------------------------------------------------------------------
# Helper: probe the API once and log the raw response for debugging
# ---------------------------------------------------------------------------
def probe_api() -> bool:
    """
    Make a single minimal request and log the raw response so we can verify
    the API is working and the response format matches what we expect.

    Returns True if the response looks usable, False otherwise.
    """
    log.info("Probing PJM DM2 API with a 1-day test window...")
    params = {
        "startRow":               1,
        "rowCount":               5,
        "datetime_beginning_ept": "2024-01-01T00:00",
        "datetime_ending_ept":    "2024-01-02T00:00",
    }
    try:
        resp = requests.get(
            PJM_API_BASE, params=params,
            timeout=30, verify=certifi.where(),
        )
        log.info("Probe HTTP status: %s", resp.status_code)
        log.info("Probe response (first 800 chars): %s", resp.text[:800])

        if resp.status_code != 200:
            log.error("API returned non-200 status. Cannot proceed.")
            return False

        payload = resp.json()
        log.info("Probe JSON keys: %s", list(payload.keys()) if isinstance(payload, dict) else type(payload))

        # Try to locate the data rows regardless of key name
        rows = _extract_rows(payload)
        log.info("Probe extracted %d rows. First row: %s", len(rows), rows[0] if rows else "none")
        return True

    except Exception as e:
        log.error("Probe failed with exception: %s: %s", type(e).__name__, e)
        return False


# ---------------------------------------------------------------------------
# Helper: extract the list of data rows from whatever the API returns
# ---------------------------------------------------------------------------
def _extract_rows(payload) -> list:
    """
    PJM's API may wrap data under different keys depending on version.
    Try common key names before giving up.
    """
    if isinstance(payload, list):
        return payload
    for key in ("data", "items", "results", "Data", "Records", "records"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    return []


# ---------------------------------------------------------------------------
# Helper: extract total row count from the API response
# ---------------------------------------------------------------------------
def _extract_total_rows(payload) -> int:
    """Return the total row count advertised by the API."""
    for key in ("totalRows", "total_rows", "total", "count", "TotalRows"):
        if key in payload:
            try:
                return int(payload[key])
            except (ValueError, TypeError):
                pass
    return 0


# ---------------------------------------------------------------------------
# Helper: fetch one page of PJM hourly load data
# ---------------------------------------------------------------------------
def fetch_pjm_page(start_ept: str, end_ept: str, start_row: int = 1) -> dict:
    """
    Call the PJM DM2 hrl_load_metered feed for one page of data.

    Parameters
    ----------
    start_ept  : str  Window start in EPT, e.g. "2024-01-01T00:00"
    end_ept    : str  Window end   in EPT, e.g. "2024-02-01T00:00"
    start_row  : int  1-based pagination offset

    Returns
    -------
    dict  Raw API JSON payload
    """
    params = {
        "startRow":               start_row,
        "rowCount":               MAX_ROWS,
        "datetime_beginning_ept": start_ept,
        "datetime_ending_ept":    end_ept,
    }
    resp = requests.get(
        PJM_API_BASE, params=params,
        timeout=60, verify=certifi.where(),
    )

    if resp.status_code != 200:
        log.error(
            "HTTP %s for window %s→%s. Body: %s",
            resp.status_code, start_ept, end_ept, resp.text[:400],
        )
        resp.raise_for_status()

    try:
        return resp.json()
    except json.JSONDecodeError as e:
        log.error(
            "JSON parse error for window %s→%s: %s | Raw (first 400): %s",
            start_ept, end_ept, e, resp.text[:400],
        )
        raise


# ---------------------------------------------------------------------------
# Helper: convert one raw API row into a MongoDB document
# ---------------------------------------------------------------------------
def build_doc(row: dict) -> dict:
    """
    Convert a PJM API row into the project document schema.

    Tries multiple field names in case the API response format varies.

    Parameters
    ----------
    row : dict  One row from the API data array

    Returns
    -------
    dict  MongoDB document ready for upsert
    """
    # Locate the UTC datetime field
    dt_str = (
        row.get("datetime_beginning_utc")
        or row.get("datetime_beginning_ept")
        or row.get("datetime_beginning")
        or row.get("DateTime")
    )
    if not dt_str:
        raise ValueError(f"No datetime field found in row: {list(row.keys())}")

    dt_utc = parse_dt(dt_str)
    if dt_utc.tzinfo is None:
        # EPT = UTC-5 (EST) or UTC-4 (EDT); store as-is UTC approximation
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    # Locate the load field
    demand = (
        row.get("area_load_mw")
        or row.get("instantaneous_load")
        or row.get("MW")
        or row.get("load_mw")
        or row.get("mw")
    )

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

    Parameters
    ----------
    start_date : str  "YYYY-MM-DD" inclusive start
    end_date   : str  "YYYY-MM-DD" inclusive end
    """
    # Probe first so we fail fast with clear diagnostics
    if not probe_api():
        log.error("API probe failed — aborting. Check logs above for details.")
        return

    log.info("Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000,
                             tlsCAFile=certifi.where())
        client.admin.command("ping")
        log.info("Connected to MongoDB Atlas successfully.")
    except Exception as e:
        log.error("Failed to connect to MongoDB: %s: %s", type(e).__name__, e)
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
        start_ept = current.strftime("%Y-%m-%dT00:00")
        end_ept   = chunk_end.strftime("%Y-%m-%dT00:00")

        log.info("Fetching window %s → %s ...", start_ept, end_ept)

        start_row  = 1
        chunk_docs = []

        while True:
            try:
                payload = fetch_pjm_page(start_ept, end_ept, start_row)
            except Exception as e:
                log.error(
                    "Skipping window %s → %s. Error: %s: %s",
                    start_ept, end_ept, type(e).__name__, e,
                )
                break

            rows = _extract_rows(payload)

            if not rows:
                log.warning(
                    "No rows in response. Keys: %s | Sample: %s",
                    list(payload.keys()) if isinstance(payload, dict) else type(payload),
                    str(payload)[:300],
                )
                break

            for r in rows:
                try:
                    chunk_docs.append(build_doc(r))
                except Exception as e:
                    log.warning("Skipping malformed row %s: %s", r, e)

            total_rows = _extract_total_rows(payload)
            if total_rows == 0 or start_row + MAX_ROWS - 1 >= total_rows:
                break
            start_row += MAX_ROWS
            time.sleep(RATE_LIMIT_S)

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
                    "  %d rows  (%d new, %d updated)",
                    len(chunk_docs), result.upserted_count, result.modified_count,
                )
            except BulkWriteError as e:
                log.error("Bulk write error: %s", e.details)
        else:
            log.warning("  No data loaded for window %s → %s.", start_ept, end_ept)

        current = chunk_end
        time.sleep(RATE_LIMIT_S)

    log.info("Ingestion complete. Total documents upserted/updated: %d", total_upserted)
    client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PJM hourly metered load into MongoDB Atlas"
    )
    parser.add_argument("--start", default="2014-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-12-31", help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    log.info("Starting PJM ingestion: %s → %s", args.start, args.end)
    ingest(args.start, args.end)
    log.info("Done.")
