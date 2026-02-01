# dvolley

`dvolley` collects tools to ingest Data Volley `.dvw` match logs into Supabase, fit rally-level breakpoint/sideout models, and inspect rotation scenarios through a Streamlit UI or the simulation APIs.

## Highlights

- ingest raw Data Volley exports into Supabase (rally and touch tables)
- fit logistic-regression and Bayesian breakpoint/sideout models with time decay and constrained parameters
- run fast parameter-driven point-by-point simulators (CLI or Python API) that mirror the models
- explore 6x6 rotation grids interactively in Streamlit to answer practical coaching questions
- See `ARCHITECTURE.md` for the end-to-end data/model/app flow.

## Repository layout

| Path | Description |
| --- | --- |
| `app.py` | Streamlit rotation simulator that exposes sliders for global/team/rotation parameters and renders win-probability matrices. |
| `dvolley/domain/` | Core modeling + simulation (`analysis_regr.py`, `simulator.py`, `models.py`, `backtest_engine.py`). |
| `dvolley/services/` | Supabase and Google Drive integration (`db.py`, `database_connection.py`, `gdrive_utils.py`). |
| `scripts/` | Thin wrappers around the service layer (`load_data.py`, `load_full_data.py`). |
| `data/` | Drop raw `.dvw` exports here (sample files are included). |
| `dvolley/cli/main.py` | CLI entrypoint (reset DB, normalize dates, fit model). |
| `requirements.txt` | Minimal dependencies for the Streamlit app and regression workflow. |

## Getting started

1. Install Python 3.10+ and a recent `pip`.
2. (Recommended) create a virtual environment and install dependencies:

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
    pip install -r requirements.txt
    pip install pymc arviz pytensor datavolley  # optional: needed for Bayesian analysis / load_full_data.py
    ```

3. Configure Supabase and Google Drive credentials in `st.secrets` (see `dvolley/services/database_connection.py` and `dvolley/services/gdrive_utils.py`).

## Typical workflow

1. Use the Streamlit **Data Management** page to sync new `.dvw` files from Google Drive into Supabase.
2. Fit the logistic model inside the app (alpha=0.001) to populate in-memory parameters.
3. Launch the Streamlit app (`streamlit run app.py`) to explore rotations with the latest parameters.
4. Optionally use the simulator classes directly for batch analysis.

## Data preparation

### `load_data.py`

- `scripts/load_data.py` wraps `dvolley.services.data_loader.update_database(...)` and uploads rally rows into Supabase; dates are normalized to `YYYY-MM-DD`.

Key columns include `match_type`, `match_date`, `team_id_h`, `team_id_a`, `team_h`, `team_a`, `set_number`, `pre/post_set_won_*`, `pre/post_point_won_*`, `p_h`, `p_a`, `point_won_team`, `serve_team`, and `serve_h/serve_a`.

### `load_full_data.py`

When you need the entire scout (skills, video time, etc.), use `scripts/load_full_data.py`, which wraps `dvolley.services.data_loader.update_database_full(...)`.

## Modeling

### Logistic baseline (`dvolley/domain/analysis_regr.py`)

`VolleyballBreakpointSideoutRegModelNoHome` ingests rally-level data from Supabase, applies exponential time decay, enforces sum-to-zero constraints for teams/rotations, and fits a constrained logistic regression through `SGDClassifier`. The app exposes a one-click refit (alpha=0.001) and stores parameters in memory.

### Bayesian serve-receive model

The Bayesian serve-receive workflow is optional and not part of the default repo scripts. If you need it, add a dedicated script in a `scripts/` folder and install `pymc`, `arviz`, and `pytensor`.

## Database utilities

- `.\.venv\Scripts\python -m dvolley.cli.main reset-db --confirm` wipes `rally_level_data` and `touch_level_data`, then reloads from Google Drive.
- `.\.venv\Scripts\python -m dvolley.cli.main normalize-dates --only-match-alt` fixes `match_alternative_id` from normalized dates and team IDs.
- `.\.venv\Scripts\python -m dvolley.cli.main fit-model --alpha 0.001` fits parameters from Supabase and writes a local CSV.

## Configuration

Central defaults live in `dvolley/config.py`:

- `DEFAULT_ALPHA`: model refit alpha used by the Streamlit UI.
- `DATE_FORMATS`: allowed input formats (DD/MM is enforced to prevent month/day swaps).
- `TEAM_NAME_TO_FIX`: manual ID consolidation for a known team.
- `SQL_ORDERED_COLS`: column ordering for touch-level exports.

## Streamlit rotation simulator

Launch the UI with:

```powershell
streamlit run app.py
```

The sidebar (see `rotation_simulator_controls_in_sidebar`) lets you:

- load base parameters for two teams (after fitting) and tweak global/team/rotation adjustments with sliders,
- set the initial score, select the serving team, and toggle a tiebreak,
- click **APPLY** to recompute the full 6x6 grid by calling `compute_rotation_probability_matrix`, which stitches together new `global_df`, `team_home_df`, and `team_away_df` snapshots before passing them to the simulator.

The main panel displays:

- a styled home-vs-away rotation probability matrix, plus the full table including average rows,
- expandable sections that echo the exact inputs sent to the simulator (useful for sharing scenarios),
- metadata about the starting score/serve/tiebreak configuration currently stored in `st.session_state`.

Use this page to compare rotations, test hypothetical parameter tweaks, or sanity check the model output before sharing it with coaches.

## Command-line rotation grid & simulator API

For scripted analysis, import `dvolley/domain/simulator.py` directly from notebooks:

```python
from dvolley.domain.simulator import VolleyballPointByPointSimulator, VolleyballProbabilitySimulator

base = VolleyballPointByPointSimulator(best_of=5)
base.load_parameters(global_df, team_home_df, team_away_df)
base.set_initial_conditions(p_h=1, p_a=1, serve_team="h")
prob = VolleyballProbabilitySimulator(base)
print(prob.home_win_analytical_calculations())
```

That API makes it easy to simulate arbitrary game states, reseed rotations mid-set, or run Monte Carlo experiments beyond the canned grid.

## Data dictionary (rally-level schema)

- `match_type`, `match_date`: metadata pulled from the DVW header.
- `team_id_h`, `team_id_a`, `team_h`, `team_a`: Data Volley identifiers and labels for the home/away teams.
- `set_number`, `pre_set_won_h`, `pre_set_won_a`, `post_set_won_h`, `post_set_won_a`: running set counters before/after the rally.
- `pre_point_won_h`, `pre_point_won_a`, `post_point_won_h`, `post_point_won_a`: rally-level scoreboard.
- `p_h`, `p_a`: setter rotations (1-6) inferred from `*z`/`az` tags right before the rally.
- `serve_team`, `serve_h`, `serve_a`: serve indicators for the upcoming rally (`serve_team` is `"h"` or `"a"`).
- `point_won_h`, `point_won_a`, `point_won_team`: rally winner expressed as home, away, or `"h"/"a"` strings.
- Additional helper columns (e.g., `current_set`, `serve_sequence`) can be extended downstream; keep the original schema so the modeling scripts continue to work.

## Troubleshooting & tips

- If the app says no parameters are fitted, load data from the DB and click **Fit model (alpha=0.001)**.
- The loaders assume UTF-8/CP1252 DVW text files. If Data Volley exports a new format, adjust `dvw_rallies_to_df` accordingly.
- `dvolley/domain/analysis_regr.py` expects every row to have `serve_team` and rotation columns; drop rallies with missing setters before fitting.
- Bayesian runs can take a while; start with fewer draws or a subset of the data if you only need sanity checks.
- Keep raw data under `data/` and generated artefacts (e.g., `rotation_win_probs.csv`) out of version control if files are large or sensitive.
