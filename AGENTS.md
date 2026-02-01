# Repository Guidelines

## Project Structure & Module Organization

- Top-level Python scripts handle data ingestion, modeling, and simulation (e.g., `load_data.py`, `analysis_regr.py`, `simulator.py`, `app.py`).
- Data inputs live in `data/` (raw `.dvw` exports). Generated datasets go to `clean_data/` and model outputs to `params/`.
- Analysis artifacts and results are stored in folders like `backtest_results/`, `backtest_loocv_results/`, and `mantova_loocv_results/`.
- The Streamlit UI is in `app.py` and consumes parameters from `params/params_out_break_sideout.csv`.

## Build, Test, and Development Commands

- `python -m venv .venv` and `pip install -r requirements.txt`: create a local Python environment.
- `python load_data.py`: parse DVW files under `data/` into `clean_data/clean_data.csv`.
- `python analysis_regr.py`: fit logistic regression parameters and refresh `params/params_out_break_sideout.csv`.
- `python analysis.py`: run the Bayesian model (requires extra dependencies: `pymc`, `arviz`, `pytensor`).
- `streamlit run app.py`: launch the rotation simulator UI.
- `python run_simulations.py`: sweep all rotation pairs and write `rotation_win_probs.csv`.

## Coding Style & Naming Conventions

- Use 4-space indentation and standard Python style (PEP 8). Prefer clear, descriptive function names.
- File and module names follow `snake_case.py`. Classes use `CapWords`.
- If you add formatting, use a single tool consistently (e.g., `black`) and document it here.

## Testing Guidelines

- No dedicated test suite is present. When adding tests, prefer `pytest` and place them under `tests/` with names like `test_simulator.py`.
- Before PRs, run the core scripts you touched (e.g., `python load_data.py`, `python analysis_regr.py`).

## Commit & Pull Request Guidelines

- Recent commits mix terse subjects (e.g., `fix`, `new`) and conventional-style messages (`feat: ?`). Keep subjects short and imperative.
- PRs should include: a clear description of the change, any updated data/parameter outputs, and screenshots for Streamlit UI changes.
- Link related issues or datasets and note any new dependencies in `requirements.txt`.

## Security & Configuration Tips

- Keep large or sensitive data outputs out of version control (especially `clean_data/` and `params/`).
- Parameter file paths are controlled by `PARAMS_FILE` in `app.py`; document new exports if you change it.
