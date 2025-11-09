import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from typing import Optional


class VolleyballServeReceiveRegModelNoHome:
    """
    Logistic-regression version (no home effect).
    - sum of serve-team effects = 0
    - sum of receive-team effects = 0
    - for every team, sum of 6 serve-rotation effects = 0
    - for every team, sum of 6 receive-rotation effects = 0
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
        alpha: float = 1e-2,      # slightly stronger regularization
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

        self.server_idx = None
        self.receiver_idx = None
        self.server_pos_idx = None
        self.receiver_pos_idx = None
        self.y = None
        self.weights = None

        # design matrix
        self.X = None
        self.feature_meta = None

        # empirical bucket probs
        self.empirical_serve_win = None
        self.emp_serve_team = None           # dict: t -> prob
        self.emp_receive_team = None         # dict: t -> prob
        self.emp_serve_pos = None            # dict: t -> np.array(6,)
        self.emp_receive_pos = None          # dict: t -> np.array(6,)

        # fitted model
        self.model: Optional[SGDClassifier] = None

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _impact_on_baseline(self, baseline_logit: float, coef: float) -> float:
        p0 = self._sigmoid(baseline_logit)
        p1 = self._sigmoid(baseline_logit + coef)
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

        # team index space
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

        # outcome: serving team won?
        point_won_team = df["point_won_team"].to_numpy()
        y = np.where(
            (serve_is_home & (point_won_team == "h"))
            | (serve_is_away & (point_won_team == "a")),
            1,
            0,
        ).astype(int)

        # empirical overall
        empirical_serve_win = y.mean()

        # time weights
        tmax = df["match_date"].max()
        age_days = (tmax - df["match_date"]).dt.days.to_numpy()
        weights = 0.5 ** (age_days / self.half_life_days)

        # store main stuff
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
        self.empirical_serve_win = empirical_serve_win

        # precompute empirical probs per bucket
        self._compute_empirical_buckets()

        # build design matrix with constraints
        self._build_design_matrix_effects()

    def _compute_empirical_buckets(self):
        """Compute raw probabilities from the data for every team and team+rotation."""
        n_teams = self.n_teams
        y = self.y

        emp_serve_team = {}
        emp_receive_team = {}
        emp_serve_pos = {}
        emp_receive_pos = {}

        for t in range(n_teams):
            # serve team bucket
            mask_s = (self.server_idx == t)
            emp_serve_team[t] = y[mask_s].mean() if mask_s.any() else np.nan

            # receive team bucket
            mask_r = (self.receiver_idx == t)
            emp_receive_team[t] = y[mask_r].mean() if mask_r.any() else np.nan

            # serve positions
            serve_pos_arr = np.full(6, np.nan)
            for p in range(6):
                m = mask_s & (self.server_pos_idx == p)
                serve_pos_arr[p] = y[m].mean() if m.any() else np.nan
            emp_serve_pos[t] = serve_pos_arr

            # receive positions
            receive_pos_arr = np.full(6, np.nan)
            for p in range(6):
                m = mask_r & (self.receiver_pos_idx == p)
                receive_pos_arr[p] = y[m].mean() if m.any() else np.nan
            emp_receive_pos[t] = receive_pos_arr

        self.emp_serve_team = emp_serve_team
        self.emp_receive_team = emp_receive_team
        self.emp_serve_pos = emp_serve_pos
        self.emp_receive_pos = emp_receive_pos

    def _build_design_matrix_effects(self):
        """
        Build X with sum-to-zero encoding.

        - Serve team: (T-1) cols, last team is reference → encoded as -1 on all
        - Receive team: (T-1) cols
        - Serve pos: for each team, 5 cols, 6th is reference
        - Receive pos: same
        """
        n = len(self.df)
        T = self.n_teams
        feature_cols = []
        feature_meta = []

        # serve team (T-1)
        for t in range(T - 1):
            col = np.zeros(n)
            col[self.server_idx == t] = 1.0
            col[self.server_idx == (T - 1)] = -1.0
            feature_cols.append(col)
            feature_meta.append({"group": "serve_team_adjustment", "team_idx": t})

        # receive team (T-1)
        for t in range(T - 1):
            col = np.zeros(n)
            col[self.receiver_idx == t] = 1.0
            col[self.receiver_idx == (T - 1)] = -1.0
            feature_cols.append(col)
            feature_meta.append({"group": "receive_team_adjustment", "team_idx": t})

        # per-team serve rotation (5 cols → 6th is reference)
        for t in range(T):
            for p in range(5):
                col = np.zeros(n)
                col[(self.server_idx == t) & (self.server_pos_idx == p)] = 1.0
                col[(self.server_idx == t) & (self.server_pos_idx == 5)] = -1.0
                feature_cols.append(col)
                feature_meta.append(
                    {
                        "group": "serve_team_position_adjustment",
                        "team_idx": t,
                        "pos_idx": p,
                    }
                )

        # per-team receive rotation
        for t in range(T):
            for p in range(5):
                col = np.zeros(n)
                col[(self.receiver_idx == t) & (self.receiver_pos_idx == p)] = 1.0
                col[(self.receiver_idx == t) & (self.receiver_pos_idx == 5)] = -1.0
                feature_cols.append(col)
                feature_meta.append(
                    {
                        "group": "receive_team_position_adjustment",
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
            "par_name": "global_serve",
            "par_value": baseline_logit,
            "impact_on_baseline_probability": baseline_prob,
            "empirical_probability": self.empirical_serve_win,
        }]

        T = self.n_teams

        # reconstruct constrained params
        serve_team_vals = np.zeros(T)
        receive_team_vals = np.zeros(T)
        serve_pos_vals = {t: np.zeros(6) for t in range(T)}
        receive_pos_vals = {t: np.zeros(6) for t in range(T)}

        for meta, val in zip(self.feature_meta, coef):
            g = meta["group"]
            if g == "serve_team_adjustment":
                serve_team_vals[meta["team_idx"]] = val
            elif g == "receive_team_adjustment":
                receive_team_vals[meta["team_idx"]] = val
            elif g == "serve_team_position_adjustment":
                serve_pos_vals[meta["team_idx"]][meta["pos_idx"]] = val
            elif g == "receive_team_position_adjustment":
                receive_pos_vals[meta["team_idx"]][meta["pos_idx"]] = val

        # close the constraints
        serve_team_vals[T - 1] = -serve_team_vals[: T - 1].sum()
        receive_team_vals[T - 1] = -receive_team_vals[: T - 1].sum()
        for t in range(T):
            serve_pos_vals[t][5] = -serve_pos_vals[t][:5].sum()
            receive_pos_vals[t][5] = -receive_pos_vals[t][:5].sum()

        # now output team-by-team
        for t in range(T):
            tid = self.all_team_ids[t]
            tname = self.team_id_to_name.get(tid, str(tid))

            # serve team
            v = serve_team_vals[t]
            rows.append({
                "par_type": "team",
                "team_id": tid,
                "team_name": tname,
                "par_name": "serve_team_adjustment",
                "par_value": v,
                "impact_on_baseline_probability": self._impact_on_baseline(baseline_logit, v),
                "empirical_probability": self.emp_serve_team[t],
            })

            # receive team
            v = receive_team_vals[t]
            rows.append({
                "par_type": "team",
                "team_id": tid,
                "team_name": tname,
                "par_name": "receive_team_adjustment",
                "par_value": v,
                "impact_on_baseline_probability": self._impact_on_baseline(baseline_logit, v),
                "empirical_probability": self.emp_receive_team[t],
            })

            # serve positions
            for p in range(6):
                v = serve_pos_vals[t][p]
                rows.append({
                    "par_type": "team",
                    "team_id": tid,
                    "team_name": tname,
                    "par_name": f"serve_pos_{p+1}",
                    "par_value": v,
                    "impact_on_baseline_probability": self._impact_on_baseline(baseline_logit, v),
                    "empirical_probability": self.emp_serve_pos[t][p],
                })

            # receive positions
            for p in range(6):
                v = receive_pos_vals[t][p]
                rows.append({
                    "par_type": "team",
                    "team_id": tid,
                    "team_name": tname,
                    "par_name": f"receive_pos_{p+1}",
                    "par_value": v,
                    "impact_on_baseline_probability": self._impact_on_baseline(baseline_logit, v),
                    "empirical_probability": self.emp_receive_pos[t][p],
                })

        df_params = pd.DataFrame(rows)

        # order: global first, then by team, with team-level first then rotations
        df_params["__pt"] = np.where(df_params["par_type"] == "global", 0, 1)
        df_params["__tid"] = df_params["team_id"].apply(lambda x: -1 if x == "global" else int(x))

        def _ord(name):
            if name == "serve_team_adjustment":
                return 0
            if name == "receive_team_adjustment":
                return 1
            if name.startswith("serve_pos_"):
                return 10 + int(name.split("_")[-1])
            if name.startswith("receive_pos_"):
                return 20 + int(name.split("_")[-1])
            return 999

        df_params["__po"] = df_params["par_name"].map(_ord)
        df_params = df_params.sort_values(by=["__pt", "__tid", "__po"]).drop(columns=["__pt", "__tid", "__po"])

        if save_path:
            df_params.to_csv(save_path, index=False, encoding="utf-8")

        return df_params


# example usage
if __name__ == "__main__":
    m = VolleyballServeReceiveRegModelNoHome()
    m.load_data("clean_data.csv")
    m.fit()
    params_df = m.viz_parameters(save_path="params_out.csv")
    print(params_df.head(40))
