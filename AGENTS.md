# Repository Guidelines

## Project Structure & Responsibilities

- `app.py`: Streamlit entrypoint and page routing.
- `dvolley/ui/pages/`: user-facing pages (`setup.py`, `detailed_analysis.py`, `descriptive_stats.py`, `conditional_breakpoint.py`, `rotation.py`, `teams_summary.py`, `model_analysis.py`).
- `dvolley/domain/`: model fitting, simulation, and touch analysis logic.
- `dvolley/services/`: Supabase/GDrive integration and DB loaders.
- `dvolley/data/`: DVW parsers and normalization helpers.
- `dvolley/cli/main.py`: operational commands (`reset-db`, `normalize-dates`, `fit-model`).

## Required Command Style (Windows)

- Always run Python commands with the project venv:
  - `.\.venv\Scripts\python ...`
- Streamlit run command:
  - `.\.venv\Scripts\python -m streamlit run app.py`
- Unit tests:
  - `.\.venv\Scripts\python -m pytest tests/unit -q`

## Data Loading Rules

- Supabase is the source of truth; do not add local CSV dependencies for app runtime.
- Keep date handling consistent as `YYYY-MM-DD` in both uploads and reads.
- For touch-analysis pages (`Detailed Analysis`, `Descriptive Statistics`, `Conditional Breakpoint Probability`), do not preload full touch data:
  - build team/match options from rally data (`loader.data`)
  - fetch touch rows only for selected match IDs (`load_matches_data_from_db`)

## Coding & Refactoring Standards

- Python style: PEP 8, 4-space indentation, `snake_case` modules/functions, `CapWords` classes.
- Keep domain logic out of UI files when adding new calculations.
- Prefer small composable functions with explicit inputs/outputs.
- Preserve backward-compatible behavior for existing pages unless requested.

## Validation Checklist Before Finishing

- Run unit tests and report pass/fail.
- Compile changed Python files when adding new modules:
  - `.\.venv\Scripts\python -m py_compile <files...>`
- If UI behavior changed, verify key flows in Streamlit:
  - Setup sync/load/fit
  - Detailed Analysis (team -> matches -> filtered touch load)
  - Descriptive Statistics (phase/team/match filters, rotation toggle, drilldown)
  - Conditional Breakpoint Probability (point-won labels and metrics)
  - Rotation and Teams Summary still working

## PR/Commit Expectations

- Keep commit messages short and imperative.
- Summarize user-visible behavior changes and DB-impacting changes.
- Include screenshots for UI changes when relevant.
