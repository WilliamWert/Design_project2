"""
ingest_weather.py
-----------------
Fetches hourly dry-bulb temperature observations (°F) from four ASOS weather
stations across the PJM East footprint via the NOAA Climate Data Online (CDO)
API, averages them to a single grid-representative value per UTC hour, and
writes the result into the existing MongoDB documents created by ingest_pjm.py.

Stations used (GHCND IDs):
    GHCND:USW00013739  Philadelphia International Airport (PHL)
    GHCND:USW00093721  Dulles International Airport (IAD)
    GHCND:USW00094823  Pittsburgh International Airport (PIT)
    GHCND:USW00094846  Chicago O'Hare International Airport (ORD)

Usage:
    python ingest_weather.py --start 2014-01-01 --end 2024-12-31

Requirements:
    pip install pymongo[srv] requests python-dateutil tqdm
    A free NOAA CDO API token is required — register at:
    https://www.ncdc.noaa.gov/cdo-web/token
"""

import argparse
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from dateutil.parser import parse as parse_dt
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI  = "mongodb+srv://williamcwert_db_user:KsjMkB8lcwSg9NcM@cluster0.v51ynam.mongodb.net/"
DB_NAME    = "energy_forecast"
COLLECTION = "pjm_hourly"

NOAA_TOKEN   = "YOUR_NOAA_CDO_TOKEN"   # free token from ncdc.noaa.gov/cdo-web/token
NOAA_API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

# The four ASOS stations spanning the PJM East footprint
STATIONS = [
    "GHCND:USW00013739",   # PHL — Philadelphia
    "GHCND:USW00093721",   # IAD — Dulles / Washington DC
    "GHCND:USW00094823",   # PIT — Pittsburgh
    "GHCND:USW00094846",   # ORD — Chicago O'Hare
]

DATATYPE     = "HLY-TEMP-NORMAL"   # hourly temperature (°F) from NOAA ISD
RATE_LIMIT_S = 0.25                # NOAA CDO allows ~5 req/s on free tier
MAX_ROWS     = 1_000               # CDO page size limit


# ---------------------------------------------------------------------------
# Helper: fetch one page of NOAA CDO hourly temperature data
# ---------------------------------------------------------------------------
def fetch_noaa_page(station: str, start: str, end: str, offset: int = 1) -> dict:
    """
    Call the NOAA CDO /data endpoint for hourly temperatures.

    Parameters
    ----------
    station : str   GHCND station ID
    start   : str   ISO date "YYYY-MM-DD"
    end     : str   ISO date "YYYY-MM-DD"
    offset  : int   1-based pagination offset

    Returns
    -------
    dict with keys 'metadata' and 'results'
    """
    headers = {"token": NOAA_TOKEN}
    params  = {
        "datasetid":  "NORMAL_HLY",
        "stationid":  station,
        "datatypeid": DATATYPE,
        "startdate":  start,
        "enddate":    end,
        "units":      "standard",   # °F
        "limit":      MAX_ROWS,
        "offset":     offset,
    }
    resp = requests.get(NOAA_API_BASE, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Core: fetch all observations for one station over a date window
# ---------------------------------------------------------------------------
def fetch_station_hourly(station: str, start: str, end: str) -> dict:
    """
    Return a dict mapping UTC-hour strings -> temperature_f (float)
    for the given station and date range.
    """
    hour_temps = {}
    offset = 1

    while True:
        data    = fetch_noaa_page(station, start, end, offset)
        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            try:
                dt_utc = parse_dt(obs["date"]).astimezone(timezone.utc)
                key    = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                hour_temps[key] = float(obs["value"])
            except (KeyError, ValueError):
                pass

        total = data.get("metadata", {}).get("resultset", {}).get("count", 0)
        if offset + MAX_ROWS - 1 >= total:
            break
        offset += MAX_ROWS
        time.sleep(RATE_LIMIT_S)

    return hour_temps


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------
def ingest(start_date: str, end_date: str):
    client     = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]

    # Chunk into 1-month windows (CDO date-range limit is 1 year, but shorter
    # chunks keep memory usage low and allow easier resume on failure)
    chunk_days = 30
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    total_updated = 0
    current = start

    while current < end:
        chunk_end  = min(current + timedelta(days=chunk_days), end)
        start_str  = current.strftime("%Y-%m-%d")
        end_str    = chunk_end.strftime("%Y-%m-%d")

        print(f"\nChunk {start_str} → {end_str}")

        # Gather readings from all stations, keyed by UTC hour
        combined: dict[str, list[float]] = defaultdict(list)

        for station in STATIONS:
            print(f"  Station {station} ...", end=" ", flush=True)
            temps = fetch_station_hourly(station, start_str, end_str)
            for hour_key, temp_f in temps.items():
                combined[hour_key].append(temp_f)
            print(f"{len(temps)} obs")
            time.sleep(RATE_LIMIT_S)

        # Average across stations; update matching MongoDB documents
        bulk_ops = []
        for hour_key, readings in combined.items():
            if not readings:
                continue
            avg_temp = sum(readings) / len(readings)
            bulk_ops.append({
                "filter": {"datetime": hour_key},
                "update": {
                    "$set": {
                        "temperature_f": round(avg_temp, 2),
                        "ingested_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                },
            })

        if bulk_ops:
            from pymongo import UpdateOne
            ops    = [UpdateOne(op["filter"], op["update"]) for op in bulk_ops]
            result = collection.bulk_write(ops, ordered=False)
            total_updated += result.modified_count
            print(f"  Updated {result.modified_count:,} documents with temperature data")

        current = chunk_end

    print(f"\nDone. Total documents updated with temperature: {total_updated:,}")
    client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NOAA hourly temperature into MongoDB")
    parser.add_argument("--start", default="2014-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-12-31", help="End date   YYYY-MM-DD")
    args = parser.parse_args()
    ingest(args.start, args.end)
