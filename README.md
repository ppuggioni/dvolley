# dvolley

`dvolley` ingests Data Volley `.dvw` files into Supabase, fits rally-level breakpoint/sideout models, and exposes match analytics and simulation pages through Streamlit.

See `ARCHITECTURE.md` for the full data and runtime flow.

## Main features

- Sync rally-level and touch-level data from Google Drive into Supabase.
- Fit logistic and empirical models from DB rally data.
- Run rotation simulations from fitted parameters.
- Run touch-by-touch **Detailed Analysis** (Breakpoint/Sideout) with team-first filtering.
- Run **Descriptive Statistics** for sideout/breakpoint event and attack-quality summaries.
- Run **Player Analysis** for player-level sideout pass/attack quality breakdowns.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entrypoint and page routing. |
| `dvolley/ui/pages/` | App pages: Setup, Detailed Analysis, Descriptive Statistics, Player Analysis, Conditional Breakpoint Probability, Rotation, Teams Summary, Model Analysis. |
| `dvolley/domain/` | Core logic: modeling, simulation, breakpoint/sideout touch analysis. |
| `dvolley/services/` | DB/GDrive integration and data-loading orchestration. |
| `dvolley/data/` | DVW parsers and date normalization helpers. |
| `dvolley/cli/main.py` | CLI commands (`reset-db`, `normalize-dates`, `fit-model`). |
| `scripts/` | Thin wrappers for manual ingestion runs. |
| `tests/unit/` | Unit tests for parsers and services. |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Configure credentials in `st.secrets` for:
- Supabase (`dvolley/services/database_connection.py`)
- Google Drive (`dvolley/services/gdrive_utils.py`)

## Run

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

## Typical app workflow

1. Open **Setup & Status**.
2. Sync Google Drive to DB (rally + touch tables).
3. Load DB into app and fit a model.
4. Use:
   - **Detailed Analysis** for touch-by-touch breakpoint/sideout tables.
   - **Descriptive Statistics** for event-level and attack-quality point-won summaries.
   - **Player Analysis** for sideout player pass/attack-quality splits by rotation.
   - **Rotation Simulator** for 6x6 win-probability grids.
   - **Teams Summary** and **Model Analysis** for fitted-parameter diagnostics.

## Page guide

### Detailed Analysis

- Touch-by-touch scouting page with two submodes:
  - **Breakpoint**: selected team is serving.
  - **Sideout**: selected team is receiving.
- Useful to inspect reception/serve quality classes, rotation splits, and player summaries.
- Team and match selectors are driven by rally data; touch rows are loaded only for selected matches.

### Descriptive Statistics

- Team-first, match-filtered touch analysis with a phase selector:
  - **Sideout**: selected team is receiving.
  - **Breakpoint**: selected team is serving.
- Event table reports:
  - `Actions`, `% share`, `Successful`, `% successful`.
  - Optional segment columns by rotation (`Total`, `P1`..`P6`), where `% share` is computed within each segment.
- Sideout mode includes a flag to exclude opponent serve errors from stats.
- Event drilldown table breaks selected event rows (e.g. `+`) by first attack quality.

### Conditional Breakpoint Probability

- Estimates `P(point won by selected team | first receiving attack quality)`.
- Two interpretations:
  - **Team sideout**: selected team is receiving.
  - **Team breakpoint**: selected team is serving.
- Excludes rallies with no first receiving attack.
- Includes share columns:
  - `Condition_share_of_first_attacks`: frequency of each quality over all analyzed first attacks.
  - `Condition_share_within_rotation`: quality mix inside each rotation.
  - `Condition_share_within_player`: quality mix inside each player (sideout mode).
- Outcome columns are shown as selected-team point outcomes:
  - `Point_won_count`
  - `Point_won_probability`

### Player Analysis

- Sideout-only, team-first and player-second workflow.
- Team and match lists come from rally data; touch rows are loaded only for selected matches.
- Provides three table groups with aggregate and optional rotation splits (`Total`, `P1`..`P6`):
  - First-attack quality after serve reception, including per-pass-quality tabs.
  - Non-first attack quality (attacks not immediately after the first pass).
  - First-pass quality for rallies where selected player is the first receiver.
- Tables include actions, shares, success counts, success rates, and Bayesian 95% intervals.

## Important behavior: touch-analysis loading

- Team and match lists come from the already-loaded rally dataset (`loader.data`) for:
  - Detailed Analysis
  - Descriptive Statistics
  - Conditional Breakpoint Probability
- Touch rows are fetched from Supabase **only after** selecting team and matches.
- DB query is limited to selected `match_alternative_id` values (`load_matches_data_from_db`).
- This avoids loading the full touch table for every visit.

## CLI commands

```powershell
.\.venv\Scripts\python -m dvolley.cli.main reset-db --confirm
.\.venv\Scripts\python -m dvolley.cli.main normalize-dates --only-match-alt
.\.venv\Scripts\python -m dvolley.cli.main fit-model --alpha 0.001 --out params_out_break_sideout.csv
```

Notes:
- `fit-model` CLI default alpha comes from `dvolley/config.py` (`DEFAULT_ALPHA=0.001`).
- In-app model selection defaults to `logistic_rotation_alpha_0.005`.

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests/unit -q
```

## Data and date rules

- Supabase is the source of truth for both rally and touch data.
- Dates are normalized to `YYYY-MM-DD` on upload and read paths.
- `match_alternative_id` is always derived from normalized date + team IDs.
