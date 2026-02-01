# Repository Guidelines

## Project Structure & Module Organization

- Top-level scripts handle ingestion via wrappers (`scripts/load_data.py`, `scripts/load_full_data.py`) and the app (`app.py`); domain logic lives under `dvolley/domain/`.
- Raw `.dvw` exports live in `data/`, but the app uses Supabase as the source of truth (no local CSV params).
- DB utilities live in `dvolley/services/db.py`, with Supabase wiring in `dvolley/services/database_connection.py`.
- Maintenance scripts: `dvolley/cli/main.py` provides `reset-db`, `normalize-dates`, and `fit-model` commands.
- Architecture overview lives in `ARCHITECTURE.md`.

## Build, Test, and Development Commands

- Use the virtualenv Python explicitly: `.\.venv\Scripts\python`.
- `.\.venv\Scripts\python -m streamlit run app.py`: launch the Streamlit UI.
- `.\.venv\Scripts\python -m dvolley.cli.main reset-db --confirm`: wipe `rally_level_data` and `touch_level_data`, then reload from Google Drive.
- `.\.venv\Scripts\python -m dvolley.cli.main normalize-dates --only-match-alt`: rebuild `match_alternative_id` using normalized dates.
- `.\.venv\Scripts\python -m dvolley.cli.main fit-model --alpha 0.001 --out params_out_break_sideout.csv`: fit parameters from DB and save a CSV.

## Coding Style & Naming Conventions

- Use 4-space indentation and standard Python style (PEP 8). Prefer clear, descriptive function names.
- File and module names follow `snake_case.py`. Classes use `CapWords`.

## Testing Guidelines

- Tests live under `tests/` and run with `.\.venv\Scripts\python -m pytest tests/unit -q`.
- Before PRs, run the Streamlit app paths you changed and confirm DB reads still work.

## Commit & Pull Request Guidelines

- Recent commits mix terse subjects (e.g., `fix`, `new`) and conventional-style messages (`feat: ...`). Keep subjects short and imperative.
- PRs should include: a clear description, any DB migration/cleanup steps, and screenshots for Streamlit UI changes.
- Link related issues or datasets and note new dependencies in `requirements.txt`.

## Security & Configuration Tips

- Supabase and Google Drive credentials are provided via `st.secrets`; do not hardcode them.
- Keep large outputs (e.g., `rotation_win_probs.csv`) out of version control if files are large or sensitive.
- Defaults live in `dvolley/config.py` (`DEFAULT_ALPHA`, `DATE_FORMATS`, `TEAM_NAME_TO_FIX`, `SQL_ORDERED_COLS`).
