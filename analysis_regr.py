import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from typing import Optional


class VolleyballBreakpointSideoutRegModelNoHome:
    """
    Logistic-regression volleyball model WITHOUT home effect.

    Target: y = 1 if *serving team* wins the rally (i.e. breakpoint success),
            y = 0 otherwise (sideout success by receiver).

    We encode parameters so that the final *reported* model can be read as:

        logit P(server wins) =
            global_breakpoint
          + breakpoint_team_adjustment[serving_team]
          + breakpoint_position_adjustment[serving_team, rotation]
          - sideout_team_adjustment[receiving_team]
          - sideout_position_adjustment[receiving_team, rotation]

    i.e.:
    - breakpoint params are ADDED
    - sideout params are SUBTRACTED
    - hence bigger sideout params = better sideout team.

    We still fit a standard logistic regression on a constrained design:
    - sum of breakpoint-team effects = 0
    - sum of sideout-team (pre-flip) effects = 0
    - for every team, sum of 6 breakpoint-rotation effects = 0
    - for every team, sum of 6 sideout-rotation (pre-flip) effects = 0
    """

    REQUIRED_COLS = [
        "match_type",
        "match_date",
        "team_id_h",
        "team_id_a",
        "team_h",
        "team_a",
        "p_h",
        "p_a",
        "point_won_team",
        "serve_team",
    ]

    def __init__(
        self,
        half_life_days: float = 90.0,
        alpha: float = 1e-2,
        max_iter: int = 5000,
        random_state: int = 42,
    ):
        self.half_life_days = half_life_days
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        # data containers
        self.df: Optional[pd.DataFrame] = None
        self.all_team_ids = None
        self.team_id_to_idx = None
        self.team_id_to_name = None
        self.n_teams = None

        # per-row encoded info
        self.server_idx = None
        self.receiver_idx = None
        self.server_pos_idx = None
        self.receiver_pos_idx = None
        self.y = None
        self.weights = None

        # design matrix + metadata about each column
        self.X = None
        self.feature_meta = None

        # empirical probabilities
        self.empirical_breakpoint = None  # overall P(server wins)
        self.emp_break_team = None        # t -> P(server wins | team t serving)
        self.emp_sideout_team = None      # t -> P(server wins | team t receiving)
        self.emp_break_pos = None         # t -> array(6) P(server wins | team t serving in pos)
        self.emp_sideout_pos = None       # t -> array(6) P(server wins | team t receiving in pos)

        # fitted model
        self.model: Optional[SGDClassifier] = None

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _impact_add(self, baseline_logit: float, coef: float) -> float:
        """impact on P when the coef is ADDED to the logit."""
        p0 = self._sigmoid(baseline_logit)
        p1 = self._sigmoid(baseline_logit + coef)
        return p1 - p0

    def _impact_subtract(self, baseline_logit: float, coef: float) -> float:
        """
        impact on P when the coef is SUBTRACTED from the logit
        (this is what we want for sideout params).
        """
        p0 = self._sigmoid(baseline_logit)
        p1 = self._sigmoid(baseline_logit - coef)
        return p1 - p0

    # ------------------------------------------------------------------
    # load data
    # ------------------------------------------------------------------
    def load_data(self, csv_path: str, encoding: str = "cp1252"):
        df = pd.read_csv(
            csv_path,
            encoding=encoding,
            parse_dates=["match_date"],
            dayfirst=True,
        )
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input CSV: {missing}")

        # build team index space
        all_team_ids = pd.Index(
            pd.concat([df["team_id_h"], df["team_id_a"]]).unique()
        ).sort_values()
        team_id_to_idx = {tid: i for i, tid in enumerate(all_team_ids)}
        n_teams = len(all_team_ids)

        # id -> name
        id_to_name = {}
        for _, row in df.iterrows():
            id_to_name.setdefault(row["team_id_h"], row["team_h"])
            id_to_name.setdefault(row["team_id_a"], row["team_a"])

        home_idx = df["team_id_h"].map(team_id_to_idx).to_numpy()
        away_idx = df["team_id_a"].map(team_id_to_idx).to_numpy()

        serve_is_home = (df["serve_team"] == "h").to_numpy()
        serve_is_away = ~serve_is_home

        # serving / receiving team index per row
        server_idx = np.where(serve_is_home, home_idx, away_idx)
        receiver_idx = np.where(serve_is_home, away_idx, home_idx)

        # rotations 1..6 → 0..5
        server_pos = np.where(
            serve_is_home, df["p_h"].to_numpy(), df["p_a"].to_numpy()
        ).astype(int)
        receiver_pos = np.where(
            serve_is_home, df["p_a"].to_numpy(), df["p_h"].to_numpy()
        ).astype(int)
        server_pos_idx = server_pos - 1
        receiver_pos_idx = receiver_pos - 1

        # outcome: serving team won? (i.e. breakpoint scored?)
        point_won_team = df["point_won_team"].to_numpy()
        y = np.where(
            (serve_is_home & (point_won_team == "h"))
            | (serve_is_away & (point_won_team == "a")),
            1,
            0,
        ).astype(int)

        # empirical overall breakpoint probability
        empirical_breakpoint = y.mean()

        # time weights
        tmax = df["match_date"].max()
        age_days = (tmax - df["match_date"]).dt.days.to_numpy()
        weights = 0.5 ** (age_days / self.half_life_days)

        # store
        self.df = df
        self.all_team_ids = all_team_ids
        self.team_id_to_idx = team_id_to_idx
        self.team_id_to_name = id_to_name
        self.n_teams = n_teams

        self.server_idx = server_idx
        self.receiver_idx = receiver_idx
        self.server_pos_idx = server_pos_idx
        self.receiver_pos_idx = receiver_pos_idx
        self.y = y
        self.weights = weights
        self.empirical_breakpoint = empirical_breakpoint

        # compute empirical buckets (still based on "server wins")
        self._compute_empirical_buckets()

        # build constrained design matrix
        self._build_design_matrix_effects()

    def _compute_empirical_buckets(self):
        """
        Compute raw probabilities from the data for every team and team+rotation.
        Still in terms of "prob(server wins)" because that's what we modeled.
        """
        n_teams = self.n_teams
        y = self.y

        emp_break_team = {}
        emp_sideout_team = {}
        emp_break_pos = {}
        emp_sideout_pos = {}

        for t in range(n_teams):
            # breakpoint team bucket (team t serving)
            mask_s = (self.server_idx == t)
            emp_break_team[t] = y[mask_s].mean() if mask_s.any() else np.nan

            # sideout team bucket (team t receiving)
            mask_r = (self.receiver_idx == t)
            emp_sideout_team[t] = y[mask_r].mean() if mask_r.any() else np.nan

            # breakpoint positions
            break_pos_arr = np.full(6, np.nan)
            for p in range(6):
                m = mask_s & (self.server_pos_idx == p)
                break_pos_arr[p] = y[m].mean() if m.any() else np.nan
            emp_break_pos[t] = break_pos_arr

            # sideout positions
            sideout_pos_arr = np.full(6, np.nan)
            for p in range(6):
                m = mask_r & (self.receiver_pos_idx == p)
                sideout_pos_arr[p] = y[m].mean() if m.any() else np.nan
            emp_sideout_pos[t] = sideout_pos_arr

        self.emp_break_team = emp_break_team
        self.emp_sideout_team = emp_sideout_team
        self.emp_break_pos = emp_break_pos
        self.emp_sideout_pos = emp_sideout_pos

    def _build_design_matrix_effects(self):
        """
        Build X with sum-to-zero encoding.

        We keep exactly the same strategy as before, just rename the groups:

        - breakpoint team (formerly serve): (T-1) cols
        - sideout team (formerly receive): (T-1) cols
        - per-team breakpoint rotation: 5 cols (6th is reference)
        - per-team sideout rotation: 5 cols (6th is reference)
        """
        n = len(self.df)
        T = self.n_teams
        feature_cols = []
        feature_meta = []

        # breakpoint team (T-1)
        for t in range(T - 1):
            col = np.zeros(n)
            col[self.server_idx == t] = 1.0
            col[self.server_idx == (T - 1)] = -1.0
            feature_cols.append(col)
            feature_meta.append({"group": "break_team_adjustment", "team_idx": t})

        # sideout team (T-1)  (still coded in +1/-1, we'll flip sign at output)
        for t in range(T - 1):
            col = np.zeros(n)
            col[self.receiver_idx == t] = 1.0
            col[self.receiver_idx == (T - 1)] = -1.0
            feature_cols.append(col)
            feature_meta.append({"group": "sideout_team_adjustment_raw", "team_idx": t})

        # per-team breakpoint rotation
        for t in range(T):
            for p in range(5):
                col = np.zeros(n)
                col[(self.server_idx == t) & (self.server_pos_idx == p)] = 1.0
                col[(self.server_idx == t) & (self.server_pos_idx == 5)] = -1.0
                feature_cols.append(col)
                feature_meta.append(
                    {
                        "group": "break_position_adjustment",
                        "team_idx": t,
                        "pos_idx": p,
                    }
                )

        # per-team sideout rotation (raw, sign flipped later)
        for t in range(T):
            for p in range(5):
                col = np.zeros(n)
                col[(self.receiver_idx == t) & (self.receiver_pos_idx == p)] = 1.0
                col[(self.receiver_idx == t) & (self.receiver_pos_idx == 5)] = -1.0
                feature_cols.append(col)
                feature_meta.append(
                    {
                        "group": "sideout_position_adjustment_raw",
                        "team_idx": t,
                        "pos_idx": p,
                    }
                )

        self.X = np.column_stack(feature_cols)
        self.feature_meta = feature_meta

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self):
        if self.X is None:
            raise RuntimeError("Call load_data(...) first.")

        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            fit_intercept=True,
        )
        clf.fit(self.X, self.y, sample_weight=self.weights)
        self.model = clf

    # ------------------------------------------------------------------
    # viz / export
    # ------------------------------------------------------------------
    def viz_parameters(self, save_path: Optional[str] = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() first.")

        coef = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        baseline_logit = intercept
        baseline_prob = self._sigmoid(baseline_logit)

        rows = [{
            "par_type": "global",
            "team_id": "global",
            "team_name": "global",
            "par_name": "global_breakpoint",
            "par_value": baseline_logit,
            "impact_on_baseline_probability": baseline_prob,
            "empirical_probability": self.empirical_breakpoint,
        }]

        T = self.n_teams

        # reconstruct constrained params
        break_team_vals = np.zeros(T)
        sideout_team_vals_raw = np.zeros(T)
        break_pos_vals = {t: np.zeros(6) for t in range(T)}
        sideout_pos_vals_raw = {t: np.zeros(6) for t in range(T)}

        for meta, val in zip(self.feature_meta, coef):
            g = meta["group"]
            if g == "break_team_adjustment":
                break_team_vals[meta["team_idx"]] = val
            elif g == "sideout_team_adjustment_raw":
                sideout_team_vals_raw[meta["team_idx"]] = val
            elif g == "break_position_adjustment":
                break_pos_vals[meta["team_idx"]][meta["pos_idx"]] = val
            elif g == "sideout_position_adjustment_raw":
                sideout_pos_vals_raw[meta["team_idx"]][meta["pos_idx"]] = val

        # close constraints (last team / last pos is negative sum of the others)
        break_team_vals[T - 1] = -break_team_vals[: T - 1].sum()
        sideout_team_vals_raw[T - 1] = -sideout_team_vals_raw[: T - 1].sum()
        for t in range(T):
            break_pos_vals[t][5] = -break_pos_vals[t][:5].sum()
            sideout_pos_vals_raw[t][5] = -sideout_pos_vals_raw[t][:5].sum()

        # NOW flip the "raw" sideout vals so that positive = better sideout
        # Remember: model logit = baseline + break_parts + sideout_raw_parts
        # We want:  logit = baseline + break_parts - sideout_parts
        # => sideout_parts = -sideout_raw_parts
        sideout_team_vals = -sideout_team_vals_raw
        sideout_pos_vals = {t: -sideout_pos_vals_raw[t] for t in range(T)}

        # output team-by-team
        for t in range(T):
            tid = self.all_team_ids[t]
            tname = self.team_id_to_name.get(tid, str(tid))

            # breakpoint team effect
            v_break_team = break_team_vals[t]
            rows.append({
                "par_type": "team",
                "team_id": tid,
                "team_name": tname,
                "par_name": "breakpoint_team_adjustment",
                "par_value": v_break_team,
                "impact_on_baseline_probability": self._impact_add(baseline_logit, v_break_team),
                "empirical_probability": self.emp_break_team[t],
            })

            # sideout team effect (already flipped -> positive = good sideout)
            v_side_team = sideout_team_vals[t]
            rows.append({
                "par_type": "team",
                "team_id": tid,
                "team_name": tname,
                "par_name": "sideout_team_adjustment",
                "par_value": v_side_team,
                "impact_on_baseline_probability": self._impact_subtract(baseline_logit, v_side_team),
                "empirical_probability": self.emp_sideout_team[t],
            })

            # breakpoint positions
            for p in range(6):
                v_bp = break_pos_vals[t][p]
                rows.append({
                    "par_type": "team",
                    "team_id": tid,
                    "team_name": tname,
                    "par_name": f"breakpoint_pos_{p+1}",
                    "par_value": v_bp,
                    "impact_on_baseline_probability": self._impact_add(baseline_logit, v_bp),
                    "empirical_probability": self.emp_break_pos[t][p],
                })

            # sideout positions (flipped)
            for p in range(6):
                v_so = sideout_pos_vals[t][p]
                rows.append({
                    "par_type": "team",
                    "team_id": tid,
                    "team_name": tname,
                    "par_name": f"sideout_pos_{p+1}",
                    "par_value": v_so,
                    "impact_on_baseline_probability": self._impact_subtract(baseline_logit, v_so),
                    "empirical_probability": self.emp_sideout_pos[t][p],
                })

        df_params = pd.DataFrame(rows)

        # order: global first, then by team, with team-level first then rotations
        df_params["__pt"] = np.where(df_params["par_type"] == "global", 0, 1)
        df_params["__tid"] = df_params["team_id"].apply(lambda x: -1 if x == "global" else int(x))

        def _ord(name: str) -> int:
            if name == "breakpoint_team_adjustment":
                return 0
            if name == "sideout_team_adjustment":
                return 1
            if name.startswith("breakpoint_pos_"):
                return 10 + int(name.split("_")[-1])
            if name.startswith("sideout_pos_"):
                return 20 + int(name.split("_")[-1])
            return 999

        df_params["__po"] = df_params["par_name"].map(_ord)
        df_params = df_params.sort_values(by=["__pt", "__tid", "__po"]).drop(
            columns=["__pt", "__tid", "__po"]
        )

        if save_path:
            df_params.to_csv(save_path, index=False, encoding="utf-8")

        return df_params


# example usage
if __name__ == "__main__":
    m = VolleyballBreakpointSideoutRegModelNoHome()
    m.load_data("clean_data_2.csv")  # put your csv here
    m.fit()
    params_df = m.viz_parameters(save_path="params_out_break_sideout.csv")
    print(params_df.head(40))
