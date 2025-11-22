# dvolley

`dvolley` collects tools to parse Data Volley `.dvw` match logs, fit rally-level breakpoint/sideout models, and inspect rotation scenarios through a Streamlit UI or the simulation APIs.

## Highlights

- ingest raw Data Volley exports into consistent rally-level CSV datasets
- fit logistic-regression and Bayesian breakpoint/sideout models with time decay and constrained parameters
- run fast parameter-driven point-by-point simulators (CLI or Python API) that mirror the models
- explore 6x6 rotation grids interactively in Streamlit to answer practical coaching questions

## Repository layout

| Path | Description |
| --- | --- |
| `app.py` | Streamlit rotation simulator that exposes sliders for global/team/rotation parameters and renders win-probability matrices. |
| `simulator.py` | `VolleyballPointByPointSimulator` and `VolleyballProbabilitySimulator`, i.e. the deterministic simulation core shared by the app and scripts. |
| `run_simulations.py` | Example CLI utility that sweeps all 6x6 starting rotations and writes `rotation_win_probs.csv`. |
| `analysis_regr.py` | Logistic-regression (`scikit-learn`) model that produces breakpoint/sideout parameters. |
| `analysis.py` | Bayesian (`pymc`) serve-receive model for deeper analyses and posterior visualisation. |
| `load_data.py` | Lightweight DVW parser that converts every rally into tabular rows and saves `clean_data/clean_data.csv`. |
| `load_full_data.py` | Alternative parser built on `datavolley` when you need the complete scout with richer metadata. |
| `data/` | Drop raw `.dvw` exports here (sample files are included). |
| `clean_data/` | Cached rally-level CSVs produced by the loaders. |
| `params/` | Model parameter files consumed by the simulators (`params_out_break_sideout.csv`). |
| `requirements.txt` | Minimal dependencies for the Streamlit app and regression workflow. |

## Getting started

1. Install Python 3.10+ and a recent `pip`.
2. (Recommended) create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pip install pymc arviz pytensor datavolley  # optional: needed for analysis.py / load_full_data.py
```

3. Place your `.dvw` matches under `data/` and follow the workflow below.

## Typical workflow

1. Parse new DVW files into `clean_data/clean_data.csv` with `load_data.py`.
2. Fit the logistic breakpoint/sideout model via `analysis_regr.py` to refresh `params/params_out_break_sideout.csv`.
3. Launch the Streamlit app (`streamlit run app.py`) to explore rotations with the latest parameters.
4. Optionally use `run_simulations.py` or the simulator classes directly for batch analysis or what-if comparisons.

## Data preparation

### `load_data.py`

- `dvw_rallies_to_df` reads CP1252 DVW text exports, tracks setter rotations, and emits one row per rally with scoreboard pre/post states.
- Running the script walks every file in `./data` and concatenates the results into `./clean_data/clean_data.csv`.

```powershell
python load_data.py
```

Key columns include `match_type`, `match_date`, `team_id_h`, `team_id_a`, `team_h`, `team_a`, `set_number`, `pre/post_set_won_*`, `pre/post_point_won_*`, `p_h`, `p_a`, `point_won_team`, `serve_team`, and `serve_h/serve_a`.

### `load_full_data.py`

When you need the entire scout (skills, video time, etc.), use the richer parser that leverages `datavolley`:

```powershell
python load_full_data.py
```

It emits `clean_full_data.csv` with per-action metadata that you can align or merge later via `concat_align_and_save`.

## Modeling

### Logistic baseline (`analysis_regr.py`)

`VolleyballBreakpointSideoutRegModelNoHome` ingests `clean_data/clean_data.csv`, applies exponential time decay, enforces sum-to-zero constraints for teams/rotations, and fits a constrained logistic regression through `SGDClassifier`. The end product is a tidy parameter table compatible with the simulator stack.

```powershell
python analysis_regr.py
```

Adjust the paths near the bottom of the script if your data/params live elsewhere. The default run writes `./params/params_out_break_sideout.csv`.

### Bayesian serve-receive model (`analysis.py`)

`VolleyballServeReceiveModel` mirrors the same breakpoint/sideout structure but expresses it in PyMC, producing posterior samples and visualisations (via `arviz`). Because it is heavier, install `pymc`, `arviz`, and `pytensor`, then execute:

```powershell
python analysis.py
```

Set `csv_path` inside the `__main__` section if you want to point at a different cleaned dataset.

## Parameter files

The Streamlit app and simulators expect CSV files with the schema used in `params/params_out_break_sideout.csv`:

- One `global` row with `par_name == "global_breakpoint"`.
- For every team: `breakpoint_team_adjustment`, `sideout_team_adjustment`, and `breakpoint_pos_1`..`breakpoint_pos_6` / `sideout_pos_1`..`sideout_pos_6`.
- Optional helper columns such as `impact_on_baseline_probability` or `empirical_probability` can be present but are ignored by the simulator.

`app.py` reads the file defined by `PARAMS_FILE` (defaults to `./params/params_out_break_sideout.csv`), so update that constant if you save multiple parameter exports.

## Streamlit rotation simulator

Launch the UI with:

```powershell
streamlit run app.py
```

The sidebar (see `rotation_simulator_controls_in_sidebar`) lets you:

- load base parameters for two teams from the dropdowns and tweak global/team/rotation adjustments with sliders,
- set the initial score, select the serving team, and toggle a tiebreak,
- click **APPLY** to recompute the full 6x6 grid by calling `compute_rotation_probability_matrix`, which stitches together new `global_df`, `team_home_df`, and `team_away_df` snapshots before passing them to the simulator.

The main panel displays:

- a styled home-vs-away rotation probability matrix, plus the full table including average rows,
- expandable sections that echo the exact inputs sent to the simulator (useful for sharing scenarios),
- metadata about the starting score/serve/tiebreak configuration currently stored in `st.session_state`.

Use this page to compare rotations, test hypothetical parameter tweaks, or sanity check the model output before sharing it with coaches.

## Command-line rotation grid & simulator API

For scripted analysis, edit the IDs in `run_simulations.py` and run:

```powershell
python run_simulations.py
```

It sweeps every home/away starting rotation pair, calls into `VolleyballPointByPointSimulator`/`VolleyballProbabilitySimulator`, prints the matrix, and writes `rotation_win_probs.csv`.

You can also import `simulator.py` directly from notebooks:

```python
from simulator import VolleyballPointByPointSimulator, VolleyballProbabilitySimulator

base = VolleyballPointByPointSimulator(best_of=5)
base.load_parameters(global_df, team_home_df, team_away_df)
base.set_initial_conditions(p_h=1, p_a=1, serve_team="h")
prob = VolleyballProbabilitySimulator(base)
print(prob.home_win_analytical_calculations())
```

That API makes it easy to simulate arbitrary game states, reseed rotations mid-set, or run Monte Carlo experiments beyond the canned grid.

## Data dictionary (clean_data/clean_data.csv)

- `match_type`, `match_date`: metadata pulled from the DVW header.
- `team_id_h`, `team_id_a`, `team_h`, `team_a`: Data Volley identifiers and labels for the home/away teams.
- `set_number`, `pre_set_won_h`, `pre_set_won_a`, `post_set_won_h`, `post_set_won_a`: running set counters before/after the rally.
- `pre_point_won_h`, `pre_point_won_a`, `post_point_won_h`, `post_point_won_a`: rally-level scoreboard.
- `p_h`, `p_a`: setter rotations (1-6) inferred from `*z`/`az` tags right before the rally.
- `serve_team`, `serve_h`, `serve_a`: serve indicators for the upcoming rally (`serve_team` is `"h"` or `"a"`).
- `point_won_h`, `point_won_a`, `point_won_team`: rally winner expressed as home, away, or `"h"/"a"` strings.
- Additional helper columns (e.g., `current_set`, `serve_sequence`) can be extended downstream; keep the original schema so the modeling scripts continue to work.

## Troubleshooting & tips

- If Streamlit cannot find your parameter file, confirm `PARAMS_FILE` points to the export you generated with `analysis_regr.py`.
- The loaders assume UTF-8/CP1252 DVW text files. If Data Volley exports a new format, adjust `dvw_rallies_to_df` accordingly.
- `analysis_regr.py` expects every row to have `serve_team` and rotation columns; drop rallies with missing setters before fitting.
- Bayesian runs (`analysis.py`) can take a while; start with fewer draws or a subset of the data if you only need sanity checks.
- Keep raw data under `data/` and generated artefacts (`clean_data/`, `params/`, `rotation_win_probs.csv`) out of version control if the files are large or sensitive.
