# DS 4320 Project 2: Forecasting Hourly Electricity Demand in PJM East

This repository contains all materials for a machine-learning pipeline that forecasts next-day hourly electricity demand (in megawatts) for the PJM East interconnection zone. Raw hourly load data is ingested from PJM's public Data Miner 2 API, stored as documents in MongoDB Atlas, and used to train a Random Forest regression model that predicts demand from calendar and lag features. The goal is to demonstrate how the document model can serve as the data backbone for a real-world time-series ML application — from raw ingestion through feature engineering to model evaluation and visualization.

**Name:** William Wert  
**NetID:** dxg9tt  
**DOI:** [doi](https://zenodo.org/records/19905642?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZkYWI5MzI1LTJjMzEtNGFlZi1iODA1LWNiZmZmOTdkNjk3MiIsImRhdGEiOnt9LCJyYW5kb20iOiIxZjBmOTYxNTI1YzA5YjVmMTE3ZTdkNGU4ZmEyNTA5MyJ9.1UMLIYh9bDZxOfti_E732knm_4TJVumpzr0luJJIy3VueDtY9VBp105VlHCEjQ4zKVtyMXuRxuqRWGQVlJyDpg)

**Press Release:** [press_release.md](press_release.md)  
**Pipeline:** [pipeline.ipynb](pipeline.ipynb)  
**License:** MIT — see [LICENSE](LICENSE)

---

## Problem Definition

### General and Specific Problem

**General problem (from list):** Forecasting energy demand.

**Specific problem:** Forecast next-day hourly electricity demand (in megawatts) for the PJM East interconnection zone using historical hourly load records and calendar features (hour of day, day of week, month, holiday indicator, season), evaluated with Mean Absolute Percentage Error (MAPE).

### Motivation

Electricity cannot be stored at grid scale cost-effectively, which means that power generation must match consumption in real time. When utilities and grid operators underestimate demand, they risk brownouts and expensive last-minute purchases from volatile spot markets; when they overestimate, they waste fuel and money generating electricity that goes unused. In the PJM region — one of the largest electricity markets in the world, serving over 65 million people across 13 states — even a 1% improvement in forecast accuracy translates into hundreds of millions of dollars in reduced operating costs and fewer carbon emissions from unnecessary spinning reserves. As renewable energy sources (wind and solar) become a larger share of the generation mix, demand forecasting becomes even more critical because supply itself becomes less predictable, and grid operators need more confidence on the demand side to avoid cascading reliability problems.

### Rationale for Refinement

The general problem of forecasting energy demand spans everything from forecasting national annual energy consumption decades in advance for infrastructure planning, to forecasting second-by-second grid frequency deviations for automatic generation control. The specific refinement to next-day hourly demand in PJM was chosen for three reasons. First, the day-ahead forecast is the most commercially significant horizon in electricity markets — it determines the day-ahead clearing price and generation dispatch schedules that govern the physical operation of the grid. Second, PJM publishes open, well-documented hourly load data going back decades through its Data Miner 2 portal, providing a rich, reliable, and freely accessible dataset. Third, the hourly granularity is fine enough to capture intraday demand patterns — morning ramp-up, midday plateau, evening peak — that drive operational decisions, while coarse enough to be tractable with standard ML models without requiring real-time streaming infrastructure.

### Press Release Headline

**[AI Cuts Grid Waste: New Forecasting Model Predicts Tomorrow's Electricity Demand with 97% Accuracy](press_release.md)**

---

## Domain Exposition

### Terminology

| Term | Definition |
|---|---|
| Demand Forecast | A prediction of how much electricity will be consumed over a future time window (MW or GWh) |
| Peak Demand | Maximum power consumption recorded in a given period; drives infrastructure sizing decisions |
| Load Factor | Ratio of average load to peak load; higher values indicate more uniform, efficient grid use |
| Baseload | Minimum level of demand over a period; typically met by always-on sources (nuclear, large hydro) |
| PJM | Pennsylvania-New Jersey-Maryland Interconnection — the RTO managing the grid for 65M+ people |
| RTO / ISO | Regional Transmission Organization / Independent System Operator — entities that balance supply and demand on the grid |
| MAPE | Mean Absolute Percentage Error — primary accuracy metric for demand forecasts; lower is better |
| RMSE | Root Mean Squared Error — secondary metric that penalizes large absolute misses more heavily |
| Day-Ahead Market | Electricity market where generators and consumers bid for next-day supply and demand |
| Spinning Reserve | Generating capacity kept online but idle to respond instantly to unexpected demand spikes |
| Diurnal Pattern | The repeating daily cycle of demand (overnight trough → morning ramp → midday plateau → evening peak) |
| Random Forest | An ensemble ML model consisting of many decision trees; predictions are averaged across trees |

### Domain Overview

Energy demand forecasting lives at the intersection of power systems engineering, economics, and data science. The electric grid operates simultaneously as a real-time physical network and a competitive market: electrons flow according to Kirchhoff's laws, while prices are set by supply-and-demand dynamics that change minute by minute. Grid operators — Regional Transmission Organizations (RTOs) such as PJM — are responsible for keeping generation and load in balance at all times across a network spanning hundreds of thousands of miles of transmission lines. They rely on demand forecasts at multiple time horizons: long-term forecasts (years ahead) guide capital investment in new generation and transmission; medium-term forecasts (weeks to months) inform maintenance scheduling; and short-term forecasts (hours to days) drive the day-ahead energy market and real-time economic dispatch. This project focuses on the short-term horizon, where weather patterns, behavioral rhythms, and economic activity are the dominant drivers of load variation, and where forecast accuracy has direct, measurable financial and environmental consequences.

### Background Reading

PDFs are stored in the shared OneDrive folder: **[Background Reading Folder](https://onedrive.live.com/REPLACE_WITH_YOUR_SHARED_LINK)** ← *Replace with your OneDrive shared link*

| # | Title | Authors / Source | Year | Topic | Link |
|---|---|---|---|---|---|
| 1 | Hourly Electricity Load Forecasting Using Machine Learning Algorithms | PJM Interconnection (FERC Technical Conference) | 2024 | How PJM applies ML to day-ahead hourly load forecasting in production | [PDF](https://www.pjm.com/-/media/DotCom/library/reports-notices/testimony/2024/20240709-presentation-ferc-technical-conference-hourly-load-forecasting-machine-learning-algorithms.pdf) |
| 2 | Methods of Forecasting Electric Energy Consumption: A Literature Review | Deb et al., *Energies* (MDPI) | 2022 | Survey of short-, medium-, and long-term forecasting methods including regression and neural networks | [PDF](https://www.mdpi.com/1996-1073/15/23/8919) |
| 3 | Electricity Consumption and Temperature: Evidence from Satellite Data | Colacito, Hoffman & Phan, IMF Working Paper | 2021 | Empirical analysis of the temperature–electricity relationship across residential and industrial sectors | [PDF](https://www.imf.org/-/media/Files/Publications/WP/2021/English/wpiea2021022-print-pdf.ashx) |
| 4 | Short-Term Energy Consumption Forecasting Using Deep Learning Models | Fekri et al., *Applied Sciences* (MDPI) | 2025 | Benchmarks LSTM, GRU, and transformer models on PJM hourly load data | [PDF](https://www.mdpi.com/2076-3417/15/12/6839) |
| 5 | Statistical and Economic Evaluation of Forecasts in Electricity Markets | Marcjasz et al., arXiv:2511.13616 | 2025 | Why MAPE alone is insufficient; introduces economic value metrics for energy forecast evaluation | [PDF](https://arxiv.org/pdf/2511.13616) |

---

## Data Creation

### Provenance

The primary data source for this project is the **U.S. Energy Information Administration (EIA) Open Data API v2** (api.eia.gov), specifically the Hourly Electric Grid Monitor feed. The EIA collects, validates, and publishes hourly electricity demand data for every major U.S. grid region, including the PJM Interconnection. Data is pulled from the `electricity/rto/region-data` endpoint, filtering for respondent `PJM` and type `D` (demand), for the period 2016–2024. This yields approximately 78,000 hourly records. Each record contains a period timestamp and a demand value in megawatt-hours, which equals average megawatts for a one-hour interval — the standard unit for load forecasting. The EIA API is free to use with a registered API key (registration is instant at eia.gov/opendata/register.php), well-documented, and rate-limited to approximately 2 requests per second.

### Code Table

| File | Description | Link |
|---|---|---|
| `ingest_pjm.py` | Queries the PJM Data Miner 2 REST API for hourly actual load in the PJM_E zone in 30-day chunks, parses JSON responses, and upserts documents into MongoDB Atlas. Includes logging to `logs/ingest_pjm.log` and full error handling. | [ingest_pjm.py](ingest_pjm.py) |
| `build_features.py` | Adds derived calendar features (hour_of_day, day_of_week, month, year, season, is_holiday) and lag features (demand_lag_24h, demand_lag_168h) to each document via a batched MongoDB bulk-write pipeline. Includes logging to `logs/build_features.log`. | [build_features.py](build_features.py) |

### Rationale

Several judgment calls shaped the data design. **Hourly granularity** was chosen over daily averages because the intraday demand curve — morning ramp, midday plateau, evening peak — is the primary driver of grid operational decisions; daily aggregates would destroy this signal. **PJM East** was selected over the full PJM footprint because it has the most consistent historical reporting and the smallest share of non-dispatchable wind generation, reducing confounding variability. **A ten-year training window (2014–2024)** was chosen as a balance: long enough to capture multiple economic cycles and weather extremes, but short enough to avoid the structural demand patterns of the pre-2014 era (before widespread LED lighting and cloud computing load) from dominating the model. **Calendar lag features** (24h and 168h lags) were stored directly in each document rather than computed at query time, reducing pipeline complexity and query latency in the analysis notebook. **MAPE** was selected as the primary evaluation metric because it is the industry standard for energy demand forecasting and is interpretable by non-technical grid operators as a simple percentage error.

### Bias Identification

Several biases may affect this dataset. **Temporal/structural bias:** the ten-year window captures demand patterns shaped by a specific economic era; major structural shifts such as EV adoption, remote work, and distributed rooftop solar after 2020 may make earlier years less predictive of future demand. **Reporting lag bias:** PJM utilities occasionally file retroactive corrections to metered load figures, meaning recently ingested records may differ slightly from the same records queried weeks later. **Zone selection bias:** PJM East is a specific subzone; a model trained on this region may not generalize to other PJM zones with different industrial or residential mixes.

### Bias Mitigation

Temporal bias is partially addressed by including `year` as an explicit feature, allowing the model to learn long-term trend drift. The model is evaluated primarily on the most recent two years (2023–2024), which most closely resemble future conditions. Reporting lag bias is mitigated by applying a 48-hour ingestion delay — data is only pulled once PJM's revision window has closed — and by storing an `ingested_at` timestamp in each document for auditing. Zone bias is mitigated by clearly scoping the project to PJM East and not claiming generalizability beyond that region.

---

## Metadata

### Soft-Schema: Guidelines for Document Structure

Each document in the `pjm_hourly` collection represents **one hour of grid data** for the PJM East zone. All documents should conform to the following guidelines:

- **`datetime`** *(required, string, ISO 8601 UTC)*: Timestamp at hour resolution (e.g., `"2024-01-15T14:00:00Z"`). This is the natural document key and must be unique per document. Always UTC — never mix timezones.
- **`region`** *(required, string)*: PJM subzone identifier. Always `"PJM_E"` in this project.
- **`demand_mw`** *(required, float)*: Actual metered load in megawatts. Store `null` for gaps and set `data_quality: "missing"` — never omit the document, so the time series stays contiguous.
- **`hour_of_day`** *(required, int, 0–23)*: Hour derived from `datetime`; stored explicitly to avoid recomputation in queries.
- **`day_of_week`** *(required, int, 0–6)*: 0 = Monday, 6 = Sunday. Stored explicitly for the same reason.
- **`month`** *(required, int, 1–12)*: Calendar month.
- **`year`** *(required, int)*: Calendar year (e.g., 2024).
- **`season`** *(required, string)*: `"winter"` | `"spring"` | `"summer"` | `"fall"`.
- **`is_holiday`** *(required, bool)*: True if the date is a US federal holiday.
- **`demand_lag_24h`** *(optional, float)*: `demand_mw` from 24 hours prior. Null for the first 24 documents.
- **`demand_lag_168h`** *(optional, float)*: `demand_mw` from 168 hours (one week) prior. Null for the first 168 documents.
- **`ingested_at`** *(required, string, ISO 8601 UTC)*: Timestamp when the document was written/updated.
- **`data_quality`** *(optional, string)*: Omit on clean records. Values: `"missing"`, `"imputed"`, `"revised"`.

### Data Summary

| Property | Value |
|---|---|
| Collection | `pjm_hourly` |
| Database | `energy_forecast` |
| Total documents | ~87,600 (10 years × 8,760 hrs/yr) |
| Date range | 2014-01-01 to 2024-12-31 (UTC) |
| Region | PJM East (PJM_E) |
| Required fields per document | 10 |
| Optional fields per document | Up to 3 |
| Estimated collection size | ~40 MB |
| Estimated missing demand records | < 0.1% |

### Data Dictionary

| Field | Data Type | Description | Example |
|---|---|---|---|
| `datetime` | string (ISO 8601) | UTC timestamp for the start of the hour | `"2024-07-15T14:00:00Z"` |
| `region` | string | PJM subzone identifier | `"PJM_E"` |
| `demand_mw` | float | Actual metered electricity load in megawatts | `31842.7` |
| `hour_of_day` | int | Hour of day in UTC (0–23) | `14` |
| `day_of_week` | int | 0 = Monday, 6 = Sunday | `0` |
| `month` | int | Calendar month (1–12) | `7` |
| `year` | int | Calendar year | `2024` |
| `season` | string | Meteorological season | `"summer"` |
| `is_holiday` | bool | True if a US federal holiday | `false` |
| `demand_lag_24h` | float | `demand_mw` from 24 hours prior (MW) | `30915.2` |
| `demand_lag_168h` | float | `demand_mw` from 168 hours (one week) prior (MW) | `32104.0` |
| `ingested_at` | string (ISO 8601) | UTC timestamp when document was written to MongoDB | `"2026-04-22T03:12:44Z"` |
| `data_quality` | string | Quality flag; omitted on clean records | `"missing"` |

### Data Dictionary: Uncertainty Quantification

| Field | Min | Max | Mean (est.) | Std Dev (est.) | Measurement Uncertainty | Notes |
|---|---|---|---|---|---|---|
| `demand_mw` | ~15,000 MW | ~65,000 MW | ~28,500 MW | ~6,000 MW | ±0.1% (NERC metering standard) | Right-skewed; summer peaks and polar vortex events drive the upper tail |
| `demand_lag_24h` | ~15,000 MW | ~65,000 MW | ~28,500 MW | ~6,000 MW | Same as `demand_mw` | Lag inherits measurement uncertainty from original field |
| `demand_lag_168h` | ~15,000 MW | ~65,000 MW | ~28,500 MW | ~6,000 MW | Same as `demand_mw` | Higher uncertainty on holidays, when week-over-week patterns break down |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your MongoDB Atlas connection string
export MONGO_URI="mongodb+srv://user:password@cluster.mongodb.net/"

# 3. Ingest PJM hourly load data (~87,600 documents)
python ingest_pjm.py --start 2014-01-01 --end 2024-12-31

# 4. Add calendar and lag features to each document
python build_features.py

# 5. Open and run the analysis pipeline
jupyter notebook pipeline.ipynb
```

Logs are written to `logs/ingest_pjm.log` and `logs/build_features.log`.

---

*DS 4320 — Spring 2026 | University of Virginia*
