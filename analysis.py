import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor
pytensor.config.cxx = ""

class VolleyballServeReceiveModel:
    """
    Bayesian rally-level model for volleyball serve vs receive.

    1. load_data(csv_path)
    2. fit()
    3. viz_parameters()

    Model:
        logit(p_serve_wins) =
            global_serve
          + 1{serve_is_home} * (global_home_effect + team_home_effect[home_team])
          + serve_team_adjustment[serving_team]
          + receive_team_adjustment[receiving_team]
          + serve_team_position_adjustment[serving_team, serving_rotation]
          + receive_team_position_adjustment[receiving_team, receiving_rotation]

    Likelihood is time-weighted with exponential decay (half-life set in __init__).
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
        draws: int = 1500,
        tune: int = 1500,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ):
        self.half_life_days = half_life_days
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed

        # will be filled after load_data
        self.df = None
        self.all_team_ids = None
        self.team_id_to_idx = None
        self.n_teams = None

        self.home_idx = None
        self.away_idx = None
        self.serve_is_home = None
        self.server_idx = None
        self.receiver_idx = None
        self.server_pos_idx = None
        self.receiver_pos_idx = None
        self.y = None
        self.weights = None

        self.model = None
        self.trace = None

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    def load_data(self, csv_path: str, encoding: str = "cp1252"):
        df = pd.read_csv(
            csv_path,
            encoding=encoding,
            parse_dates=["match_date"],
            dayfirst=True,
        )

        # check columns
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input CSV: {missing}")

        # build team index space
        all_team_ids = pd.Index(
            pd.concat([df["team_id_h"], df["team_id_a"]]).unique()
        ).sort_values()
        team_id_to_idx = {tid: i for i, tid in enumerate(all_team_ids)}
        n_teams = len(all_team_ids)

        home_idx = df["team_id_h"].map(team_id_to_idx).to_numpy()
        away_idx = df["team_id_a"].map(team_id_to_idx).to_numpy()

        # serving / receiving
        serve_is_home = (df["serve_team"] == "h").to_numpy()
        serve_is_away = ~serve_is_home

        server_idx = np.where(serve_is_home, home_idx, away_idx)
        receiver_idx = np.where(serve_is_home, away_idx, home_idx)

        # setter positions (1..6) -> (0..5)
        server_pos = np.where(serve_is_home, df["p_h"].to_numpy(), df["p_a"].to_numpy()).astype(int)
        receiver_pos = np.where(serve_is_home, df["p_a"].to_numpy(), df["p_h"].to_numpy()).astype(int)
        server_pos_idx = server_pos - 1
        receiver_pos_idx = receiver_pos - 1

        # outcome: serving team wins?
        point_won_team = df["point_won_team"].to_numpy()
        y = np.where(
            (serve_is_home & (point_won_team == "h"))
            | (serve_is_away & (point_won_team == "a")),
            1,
            0,
        ).astype("int8")

        # time weights
        tmax = df["match_date"].max()
        age_days = (tmax - df["match_date"]).dt.days.to_numpy()
        weights = 0.5 ** (age_days / self.half_life_days)

        # store
        self.df = df
        self.all_team_ids = all_team_ids
        self.team_id_to_idx = team_id_to_idx
        self.n_teams = n_teams

        self.home_idx = home_idx
        self.away_idx = away_idx
        self.serve_is_home = serve_is_home
        self.server_idx = server_idx
        self.receiver_idx = receiver_idx
        self.server_pos_idx = server_pos_idx
        self.receiver_pos_idx = receiver_pos_idx
        self.y = y
        self.weights = weights

    # ------------------------------------------------------------------
    # 2. FIT MODEL
    # ------------------------------------------------------------------
    def fit(self):
        if self.df is None:
            raise RuntimeError("Call load_data(...) before fit().")

        n_teams = self.n_teams

        with pm.Model() as model:
            # ----------------------------
            # global parameters
            # ----------------------------
            global_serve = pm.Normal("global_serve", mu=0.0, sigma=1.0)
            global_home_effect = pm.Normal("global_home_effect", mu=0.0, sigma=0.5)

            # ----------------------------
            # team home effect (sum-to-zero)
            # ----------------------------
            raw_team_home = pm.Normal("raw_team_home", mu=0.0, sigma=1.0, shape=n_teams)
            sigma_team_home = pm.HalfNormal("sigma_team_home", sigma=0.5)
            team_home_effect = pm.Deterministic(
                "team_home_effect",
                (raw_team_home - pm.math.mean(raw_team_home)) * sigma_team_home,
            )

            # ----------------------------
            # serve team adjustment (sum-to-zero)
            # ----------------------------
            raw_serve_team_adj = pm.Normal(
                "raw_serve_team_adj", mu=0.0, sigma=1.0, shape=n_teams
            )
            sigma_serve_team_adj = pm.HalfNormal("sigma_serve_team_adj", sigma=0.5)
            serve_team_adjustment = pm.Deterministic(
                "serve_team_adjustment",
                (raw_serve_team_adj - pm.math.mean(raw_serve_team_adj))
                * sigma_serve_team_adj,
            )

            # ----------------------------
            # receive team adjustment (sum-to-zero)
            # ----------------------------
            raw_receive_team_adj = pm.Normal(
                "raw_receive_team_adj", mu=0.0, sigma=1.0, shape=n_teams
            )
            sigma_receive_team_adj = pm.HalfNormal("sigma_receive_team_adj", sigma=0.5)
            receive_team_adjustment = pm.Deterministic(
                "receive_team_adjustment",
                (raw_receive_team_adj - pm.math.mean(raw_receive_team_adj))
                * sigma_receive_team_adj,
            )

            # ----------------------------
            # rotation / position adjustments
            # ----------------------------
            # serve
            raw_serve_pos = pm.Normal(
                "raw_serve_pos", mu=0.0, sigma=1.0, shape=(n_teams, 6)
            )
            sigma_serve_pos = pm.HalfNormal("sigma_serve_pos", sigma=0.3)
            serve_pos_centered = raw_serve_pos - pm.math.mean(
                raw_serve_pos, axis=1, keepdims=True
            )
            serve_team_position_adjustment = pm.Deterministic(
                "serve_team_position_adjustment",
                serve_pos_centered * sigma_serve_pos,
            )

            # receive
            raw_recv_pos = pm.Normal(
                "raw_recv_pos", mu=0.0, sigma=1.0, shape=(n_teams, 6)
            )
            sigma_recv_pos = pm.HalfNormal("sigma_recv_pos", sigma=0.3)
            recv_pos_centered = raw_recv_pos - pm.math.mean(
                raw_recv_pos, axis=1, keepdims=True
            )
            receive_team_position_adjustment = pm.Deterministic(
                "receive_team_position_adjustment",
                recv_pos_centered * sigma_recv_pos,
            )

            # ----------------------------
            # DATA (use pm.Data for your PyMC version)
            # ----------------------------
            serve_is_home_dt = pm.Data("serve_is_home", self.serve_is_home.astype("int8"))
            server_idx_dt = pm.Data("server_idx", self.server_idx)
            receiver_idx_dt = pm.Data("receiver_idx", self.receiver_idx)
            server_pos_dt = pm.Data("server_pos_idx", self.server_pos_idx)
            receiver_pos_dt = pm.Data("receiver_pos_idx", self.receiver_pos_idx)
            y_dt = pm.Data("y", self.y)
            weights_dt = pm.Data("weights", self.weights)
            home_idx_dt = pm.Data("home_idx", self.home_idx)

            # ----------------------------
            # linear predictor
            # ----------------------------
            eta = global_serve

            # home bonus when server is home
            eta = eta + serve_is_home_dt * (
                global_home_effect + team_home_effect[home_idx_dt]
            )

            # serving / receiving strength
            eta = eta + serve_team_adjustment[server_idx_dt]
            eta = eta + receive_team_adjustment[receiver_idx_dt]

            # rotation-specific
            eta = eta + serve_team_position_adjustment[server_idx_dt, server_pos_dt]
            eta = eta + receive_team_position_adjustment[receiver_idx_dt, receiver_pos_dt]

            p = pm.Deterministic("p_rally_win", pm.math.sigmoid(eta))

            # ----------------------------
            # weighted likelihood
            # ----------------------------
            bernoulli_dist = pm.Bernoulli.dist(p=p)
            logp = pm.logp(bernoulli_dist, y_dt)
            pm.Potential("weighted_likelihood", weights_dt * logp)

            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
            )

        self.model = model
        self.trace = trace

    # ------------------------------------------------------------------
    # 3. VISUALIZATION
    # ------------------------------------------------------------------
    def viz_parameters(self, show_rotation_for_team: int | None = 0):
        if self.trace is None:
            raise RuntimeError("Call fit() before viz_parameters().")

        # global + sigma summary
        print(
            az.summary(
                self.trace,
                var_names=[
                    "global_serve",
                    "global_home_effect",
                    "sigma_team_home",
                    "sigma_serve_team_adj",
                    "sigma_receive_team_adj",
                    "sigma_serve_pos",
                    "sigma_recv_pos",
                ],
            )
        )

        # team-level forest
        az.plot_forest(
            self.trace,
            var_names=[
                "team_home_effect",
                "serve_team_adjustment",
                "receive_team_adjustment",
            ],
            combined=True,
            figsize=(10, 6),
        )

        # rotation-level for one team
        if show_rotation_for_team is not None:
            team_idx = int(show_rotation_for_team)
            if team_idx < 0 or team_idx >= self.n_teams:
                raise ValueError(
                    f"team index {team_idx} out of range (0..{self.n_teams-1})"
                )

            team_id = self.all_team_ids[team_idx]
            print(f"\nRotation effects for team index {team_idx} (team_id={team_id}):")

            az.plot_forest(
                self.trace,
                var_names=[
                    "serve_team_position_adjustment",
                    "receive_team_position_adjustment",
                ],
                coords={
                    "serve_team_position_adjustment_dim_0": [team_idx],
                    "receive_team_position_adjustment_dim_0": [team_idx],
                },
                figsize=(10, 6),
            )


# example
if __name__ == "__main__":
    csv_path = "clean_data.csv"
    m = VolleyballServeReceiveModel()
    m.load_data(csv_path)
    m.fit()
    m.viz_parameters(show_rotation_for_team=0)
