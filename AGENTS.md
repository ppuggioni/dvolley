# Repository Guidelines

## Project Structure & Module Organization

- Top-level scripts handle ingestion, modeling, and simulation (`load_data.py`, `load_full_data.py`, `analysis_regr.py`, `simulator.py`, `app.py`).
- Raw `.dvw` exports live in `data/`, but the app uses Supabase as the source of truth (no local CSV params).
- DB utilities live in `db_utils.py`, with Supabase wiring in `database_connection.py`.
- Maintenance scripts: `reset_and_reload_db.py` (wipe + reload from Google Drive) and `normalize_db_dates.py` (fix date fields and `match_alternative_id`).

## Build, Test, and Development Commands

- Use the virtualenv Python explicitly: `.\.venv\Scripts\python`.
- `.\.venv\Scripts\python -m streamlit run app.py`: launch the Streamlit UI.
- `.\.venv\Scripts\python reset_and_reload_db.py --confirm`: wipe `rally_level_data` and `touch_level_data`, then reload from Google Drive.
- `.\.venv\Scripts\python normalize_db_dates.py --only-match-alt`: rebuild `match_alternative_id` using normalized dates.
- `.\.venv\Scripts\python run_simulations.py`: sweep all rotation pairs and write `rotation_win_probs.csv`.

## Coding Style & Naming Conventions

- Use 4-space indentation and standard Python style (PEP 8). Prefer clear, descriptive function names.
- File and module names follow `snake_case.py`. Classes use `CapWords`.

## Testing Guidelines

- No dedicated test suite is present. When adding tests, prefer `pytest` under `tests/` with names like `test_simulator.py`.
- Before PRs, run the Streamlit app paths you changed and confirm DB reads still work.

## Commit & Pull Request Guidelines

- Recent commits mix terse subjects (e.g., `fix`, `new`) and conventional-style messages (`feat: ...`). Keep subjects short and imperative.
- PRs should include: a clear description, any DB migration/cleanup steps, and screenshots for Streamlit UI changes.
- Link related issues or datasets and note new dependencies in `requirements.txt`.

## Security & Configuration Tips

- Supabase and Google Drive credentials are provided via `st.secrets`; do not hardcode them.
- Keep large outputs (e.g., `rotation_win_probs.csv`) out of version control if files are large or sensitive.
