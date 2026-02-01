# Architecture Overview

## Core Data Flow

1. **Google Drive → Supabase**
   - `dvolley/services/gdrive.py` lists and reads `.dvw` files.
   - `dvolley/services/data_loader.update_database` (rally-level) and `update_database_full` (touch-level) parse files and upload into Supabase tables.
   - Dates are normalized to `YYYY-MM-DD` via `dvolley/data/normalization.py`.

2. **Supabase → App**
   - `app.py` starts a `BackgroundLoader` and pulls rally data using `dvolley/services/data_loader.load_data_from_db`.
   - Data is cached in `st.session_state` for the UI.

3. **Model Fit → Parameters**
   - The Streamlit UI fits parameters on demand using `dvolley/domain/analysis_regr.py`.
   - Parameters live in memory (`st.session_state["fitted_params_df"]`); no local CSV is required.

4. **Simulator → UI**
   - The rotation simulator in `dvolley/domain/simulator.py` runs from UI inputs and fitted params.
   - Results are rendered in the Streamlit pages under `dvolley/ui/pages/`.

## Modules and Responsibilities

- **Domain (`dvolley/domain/`)**
  - `analysis_regr.py`: logistic regression model fitting.
  - `models.py`: alternative models for backtesting/comparison.
  - `simulator.py`: deterministic rotation simulator.
  - `backtest_engine.py`: evaluation and calibration utilities.

- **Data (`dvolley/data/`)**
  - `dvw_parser.py`: rally-level parsing.
  - `full_parser.py`: touch-level parsing.
  - `normalization.py`: strict date normalization.

- **Services (`dvolley/services/`)**
  - `db.py`: Supabase CRUD and pagination.
  - `gdrive.py`: Google Drive I/O wrapper.
  - `data_loader.py`: orchestration for ingestion and DB reads.
  - `maintenance.py`: DB date normalization.
  - `database_connection.py`: Supabase client.

- **UI (`dvolley/ui/`)**
  - `pages/rotation.py`, `teams_summary.py`, `model_analysis.py`, `load_data.py`: Streamlit pages.

- **CLI (`dvolley/cli/main.py`)**
  - `reset-db`: wipe and reload from Google Drive.
  - `normalize-dates`: fix `match_date` and `match_alternative_id`.
  - `fit-model`: fit and export parameters to CSV.

## Operational Notes

- **Source of truth**: Supabase. Local CSVs are not required for the app.
- **Date handling**: strict `DD/MM/YYYY` and ISO only to prevent day/month swaps.
- **Testing**: unit tests live in `tests/` and exercise parsers, normalization, and service layers.
