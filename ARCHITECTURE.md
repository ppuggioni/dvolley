# Architecture Overview

## Runtime layers

1. **Ingestion layer**
   - Reads `.dvw` files from Google Drive.
   - Parses rally-level rows (`dvw_parser.py`) and touch-level rows (`full_parser.py`).
   - Uploads into Supabase tables:
     - `rally_level_data`
     - `touch_level_data`

2. **Application layer (Streamlit)**
   - `app.py` starts a background loader for rally data only.
   - Pages in `dvolley/ui/pages/` consume loader data and on-demand DB queries.

3. **Domain layer**
   - `analysis_regr.py` and `models.py` fit model parameters.
   - `simulator.py` computes point/set/match probabilities.
   - `breakpoint_touch_analysis.py` and `sideout_touch_analysis.py` build touch-based tables.

4. **Service layer**
   - `db.py`: low-level Supabase calls (paged reads, writes, filtered match fetch).
   - `data_loader.py`: normalized, app-facing loaders and ingestion orchestration.

## End-to-end flows

### A) Google Drive -> Supabase

1. `update_database(...)` ingests rally-level rows.
2. `update_database_full(...)` ingests touch-level rows.
3. Dates are normalized to `YYYY-MM-DD`.
4. `match_alternative_id` is rebuilt from normalized date plus team IDs.

### B) Streamlit startup -> model state

1. `app.py` loads rally data with `load_data_from_db()`.
2. Rally dataframe is stored in `loader.data` and session state.
3. Auto-fit runs once per new load id, using the selected model option.

### C) Detailed Analysis page (team-first, filtered touch load)

1. Team catalog and match list are built from `loader.data` (rally dataset).
2. User selects team and match scope.
3. App loads touch rows only for selected matches via:
   - `load_matches_data_from_db(match_ids)`
   - `db.fetch_touches_by_match_ids(match_ids)`
4. Page routes to breakpoint or sideout analysis with the filtered touch dataframe.

This is the key performance behavior: **no full touch-table load on page entry**.

## Module map

- `dvolley/ui/pages/setup.py`: DB sync, reload, model fitting, downloads.
- `dvolley/ui/pages/detailed_analysis.py`: shared selectors plus phase radio.
- `dvolley/ui/pages/breakpoint_touch.py`: breakpoint touch tables.
- `dvolley/ui/pages/sideout_touch.py`: sideout touch tables.
- `dvolley/ui/pages/rotation.py`: rotation simulation.
- `dvolley/ui/pages/teams_summary.py`, `dvolley/ui/pages/model_analysis.py`: parameter summaries.

## Operational constraints

- Supabase is the source of truth for app data.
- Dates must remain `YYYY-MM-DD` across upload/read paths.
- Keep fit outputs in memory for app usage (`st.session_state`) unless exporting via CLI.
- Unit tests (`tests/unit`) should pass before merge.
