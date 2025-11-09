import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional
import copy


class VolleyballPointByPointSimulator:
    """
    Point-by-point volleyball simulator based on fitted parameters.

    Usage:
    - load_parameters(global_df, team_home_df, team_away_df)
    - set_initial_conditions(...)
    - set_end_point(set_n=0, point_n=0)  # 0,0 = full match
    - run_simulation() -> pd.DataFrame
    """

    def __init__(self, seed: Optional[int] = 123, best_of: int = 5):
        self.rng = np.random.default_rng(seed)
        self.best_of = best_of  # e.g. 5 -> first to 3

        # parameters
        self.global_logit = 0.0
        self.home_params = {}
        self.away_params = {}
        self.team_id_h = None
        self.team_id_a = None
        self.team_name_h = None
        self.team_name_a = None

        # match metadata (optional)
        self.match_type = ""
        self.match_date = ""

        # current match state
        self.cur_set = 1
        self.set_won_h = 0
        self.set_won_a = 0
        self.point_h = 0
        self.point_a = 0
        self.p_h = 1
        self.p_a = 1
        self.serve_team = "a"  # 'h' or 'a'

        # remember initial rotations so we can reset at new set
        self._start_p_h = 1
        self._start_p_a = 1

        # stop condition (0,0 means: play full match)
        self.end_set = 0
        self.end_rally_in_set = 0

    # ------------------------------------------------------------------
    # LOADING PARAMETERS
    # ------------------------------------------------------------------
    def load_parameters(
        self,
        global_df: pd.DataFrame,
        team_home_df: pd.DataFrame,
        team_away_df: pd.DataFrame,
        match_type: str = "",
        match_date: str = "",
    ):
        """Parse the three dataframes and store them."""

        # 1) global
        g = global_df.loc[global_df["par_name"] == "global_serve"]
        if g.empty:
            raise ValueError("global_df must contain a row with par_name == 'global_serve'")
        self.global_logit = float(g["par_value"].iloc[0])

        self.match_type = match_type
        self.match_date = match_date

        def _parse_team(df: pd.DataFrame):
            team_id = df["team_id"].iloc[0]
            team_name = df["team_name"].iloc[0]
            d = {
                "team_id": team_id,
                "team_name": team_name,
                "serve_team_adjustment": 0.0,
                "receive_team_adjustment": 0.0,
                "serve_pos": {i: 0.0 for i in range(1, 7)},
                "receive_pos": {i: 0.0 for i in range(1, 7)},
            }
            for _, row in df.iterrows():
                par_name = row["par_name"]
                val = float(row["par_value"])
                if par_name == "serve_team_adjustment":
                    d["serve_team_adjustment"] = val
                elif par_name == "receive_team_adjustment":
                    d["receive_team_adjustment"] = val
                elif par_name.startswith("serve_pos_"):
                    pos = int(par_name.split("_")[-1])
                    d["serve_pos"][pos] = val
                elif par_name.startswith("receive_pos_"):
                    pos = int(par_name.split("_")[-1])
                    d["receive_pos"][pos] = val
            return d

        self.home_params = _parse_team(team_home_df)
        self.away_params = _parse_team(team_away_df)

        self.team_id_h = self.home_params["team_id"]
        self.team_id_a = self.away_params["team_id"]
        self.team_name_h = self.home_params["team_name"]
        self.team_name_a = self.away_params["team_name"]

    # ------------------------------------------------------------------
    # INITIAL CONDITIONS
    # ------------------------------------------------------------------
    def set_initial_conditions(
        self,
        set_won_h: int = 0,
        set_won_a: int = 0,
        point_won_h: int = 0,
        point_won_a: int = 0,
        p_h: int = 1,
        p_a: int = 1,
        serve_team: str = "a",
        current_set: int = 1,
    ):
        """
        Set starting situation (like starting from mid-match).
        """
        self.set_won_h = set_won_h
        self.set_won_a = set_won_a
        self.point_h = point_won_h
        self.point_a = point_won_a
        self.p_h = p_h
        self.p_a = p_a
        self.serve_team = serve_team
        self.cur_set = current_set

        self._start_p_h = p_h
        self._start_p_a = p_a

    # ------------------------------------------------------------------
    # END POINT
    # ------------------------------------------------------------------
    def set_end_point(self, set_n: int = 0, point_n: int = 0):
        """
        - set_n = 0  AND point_n = 0  -> simulate the whole match (until someone wins)
        - set_n = K  AND point_n = 0  -> simulate to the END of set K
        - set_n = K  AND point_n = M  -> simulate to rally M of set K
        """
        self.end_set = set_n
        self.end_rally_in_set = point_n

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _rotate_on_sideout(pos: int) -> int:
        """volleyball rotation: 6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 6"""
        return pos - 1 if pos > 1 else 6

    def _rally_win_prob(self) -> float:
        """p(server wins) given current state."""
        logit = self.global_logit

        if self.serve_team == "h":
            logit += self.home_params["serve_team_adjustment"]
            logit += self.away_params["receive_team_adjustment"]
            logit += self.home_params["serve_pos"].get(self.p_h, 0.0)
            logit += self.away_params["receive_pos"].get(self.p_a, 0.0)
        else:
            logit += self.away_params["serve_team_adjustment"]
            logit += self.home_params["receive_team_adjustment"]
            logit += self.away_params["serve_pos"].get(self.p_a, 0.0)
            logit += self.home_params["receive_pos"].get(self.p_h, 0.0)

        return self._sigmoid(logit)

    # ------------------------------------------------------------------
    # SIMULATION
    # ------------------------------------------------------------------
    def run_simulation(self) -> pd.DataFrame:
        rows = []
        rallies_in_this_set = 0

        while True:
            # ----- STOP CONDITIONS (top of loop) -----
            if self.end_set == 0:
                # full match, we will check match end after rally
                pass
            else:
                if self.cur_set > self.end_set:
                    break
                if self.cur_set == self.end_set and self.end_rally_in_set > 0:
                    if rallies_in_this_set >= self.end_rally_in_set:
                        break

            # ----- record pre state -----
            pre_set_won_h = self.set_won_h
            pre_set_won_a = self.set_won_a
            pre_point_won_h = self.point_h
            pre_point_won_a = self.point_a
            pre_p_h = self.p_h
            pre_p_a = self.p_a
            pre_serve_team = self.serve_team

            # probability + outcome
            p_server = self._rally_win_prob()
            server_wins = self.rng.random() < p_server

            # ----- update according to outcome -----
            if pre_serve_team == "h":
                if server_wins:
                    self.point_h += 1
                    point_won_team = "h"
                    point_won_h = 1
                    point_won_a = 0
                    self.serve_team = "h"
                else:
                    self.point_a += 1
                    point_won_team = "a"
                    point_won_h = 0
                    point_won_a = 1
                    self.serve_team = "a"
                    self.p_a = self._rotate_on_sideout(self.p_a)
            else:  # away serving
                if server_wins:
                    self.point_a += 1
                    point_won_team = "a"
                    point_won_h = 0
                    point_won_a = 1
                    self.serve_team = "a"
                else:
                    self.point_h += 1
                    point_won_team = "h"
                    point_won_h = 1
                    point_won_a = 0
                    self.serve_team = "h"
                    self.p_h = self._rotate_on_sideout(self.p_h)

            # ----- check set end -----
            set_finished = False
            target_points = 25 if self.cur_set < self.best_of else 15
            if (self.point_h >= target_points or self.point_a >= target_points) and \
               abs(self.point_h - self.point_a) >= 2:
                set_finished = True

            post_set_won_h = self.set_won_h
            post_set_won_a = self.set_won_a

            if set_finished:
                if self.point_h > self.point_a:
                    self.set_won_h += 1
                else:
                    self.set_won_a += 1

                post_set_won_h = self.set_won_h
                post_set_won_a = self.set_won_a

                # prepare next set
                self.cur_set += 1
                self.point_h = 0
                self.point_a = 0
                rallies_in_this_set = 0
                self.p_h = self._start_p_h
                self.p_a = self._start_p_a
                self.serve_team = "a" if self.cur_set % 2 == 1 else "h"
            else:
                rallies_in_this_set += 1

            # serve flags for row
            serve_h = 1 if pre_serve_team == "h" else 0
            serve_a = 1 if pre_serve_team == "a" else 0

            rows.append(
                {
                    "match_type": self.match_type,
                    "match_date": self.match_date,
                    "team_id_h": self.team_id_h,
                    "team_id_a": self.team_id_a,
                    "team_h": self.team_name_h,
                    "team_a": self.team_name_a,
                    "set_number": self.cur_set if set_finished else self.cur_set,
                    "pre_set_won_h": pre_set_won_h,
                    "pre_set_won_a": pre_set_won_a,
                    "pre_point_won_h": pre_point_won_h,
                    "pre_point_won_a": pre_point_won_a,
                    "p_h": pre_p_h,
                    "p_a": pre_p_a,
                    "post_set_won_h": post_set_won_h,
                    "post_set_won_a": post_set_won_a,
                    "post_point_won_h": self.point_h,
                    "post_point_won_a": self.point_a,
                    "point_won_h": point_won_h,
                    "point_won_a": point_won_a,
                    "point_won_team": point_won_team,
                    "serve_h": serve_h,
                    "serve_a": serve_a,
                    "serve_team": pre_serve_team,
                }
            )

            # ----- match end? -----
            match_over = (
                self.set_won_h > self.best_of // 2
                or self.set_won_a > self.best_of // 2
            )
            if match_over:
                if self.end_set == 0:
                    break
                else:
                    if self.cur_set > self.end_set:
                        break

        return pd.DataFrame(rows)


# ======================================================================
# NEW: Probability wrapper
# ======================================================================
import pandas as pd
import numpy as np
from functools import lru_cache


class VolleyballProbabilitySimulator:
    """
    Wraps a VolleyballPointByPointSimulator and gives you:
    - run_simulations(n, save_path)  -> big df with sim_n
    - home_win_prob()                -> MC estimate from last run
    - home_win_analytical_calculations(max_extra=10) -> DP / exact-ish set win prob
      (bounded to avoid infinite recursion on long deuces)
    """

    def __init__(self, base_simulator):
        self.base_sim = base_simulator
        self._last_sim_dfs = []
        self._last_home_win_rate = None

    # ----------------------------------------------------------
    # MONTE CARLO
    # ----------------------------------------------------------
    def run_simulations(self, n: int = 1000, save_path: str | None = None) -> pd.DataFrame:
        all_rows = []
        home_wins = 0

        for i in range(n):
            sim = self._clone_base_sim()
            df_i = sim.run_simulation()
            df_i["sim_n"] = i
            all_rows.append(df_i)

            last_row = df_i.iloc[-1]
            if last_row["post_set_won_h"] > last_row["post_set_won_a"]:
                home_wins += 1

        big_df = pd.concat(all_rows, ignore_index=True)
        self._last_sim_dfs = [big_df]
        self._last_home_win_rate = home_wins / n

        if save_path is not None:
            big_df.to_csv(save_path, index=False, encoding="utf-8")

        return big_df

    def home_win_prob(self) -> float:
        if self._last_home_win_rate is None:
            raise RuntimeError("Run run_simulations(...) first.")
        return self._last_home_win_rate

    # ----------------------------------------------------------
    # ANALYTICAL / DP
    # ----------------------------------------------------------
    def home_win_analytical_calculations(self, max_extra: int = 20) -> float:
        """
        Return P(home wins THIS set) starting from the current state
        of the base simulator, using a bounded DP to avoid infinite recursion.

        max_extra: how many points beyond normal target (25 or 15) we allow
                   before we force a decision.
        """
        bs = self.base_sim

        # current state
        s_h0 = bs.point_h
        s_a0 = bs.point_a
        p_h0 = bs.p_h
        p_a0 = bs.p_a
        srv0 = bs.serve_team
        cur_set = bs.cur_set

        target_points = 25 if cur_set < bs.best_of else 15
        cap = target_points + max_extra  # upper bound on score

        global_logit = bs.global_logit
        home_params = bs.home_params
        away_params = bs.away_params

        def rally_win_prob_from_model(p_h: int, p_a: int, srv: str) -> float:
            """Exactly the same formula as the simulator uses."""
            logit = global_logit
            if srv == "h":
                logit += home_params["serve_team_adjustment"]
                logit += away_params["receive_team_adjustment"]
                logit += home_params["serve_pos"].get(p_h, 0.0)
                logit += away_params["receive_pos"].get(p_a, 0.0)
            else:
                logit += away_params["serve_team_adjustment"]
                logit += home_params["receive_team_adjustment"]
                logit += away_params["serve_pos"].get(p_a, 0.0)
                logit += home_params["receive_pos"].get(p_h, 0.0)
            return 1.0 / (1.0 + np.exp(-logit))

        def rotate(pos: int) -> int:
            return pos - 1 if pos > 1 else 6

        @lru_cache(maxsize=None)
        def V(s_h: int, s_a: int, p_h: int, p_a: int, srv: str) -> float:
            # 1) normal terminal condition
            if s_h >= target_points or s_a >= target_points:
                if abs(s_h - s_a) >= 2:
                    return 1.0 if s_h > s_a else 0.0
                # else: it's deuce-ish, fall through to bounded logic

            # 2) bounded terminal to avoid infinite recursion
            if s_h >= cap or s_a >= cap:
                if s_h > s_a:
                    return 1.0
                elif s_a > s_h:
                    return 0.0
                else:
                    # super rare tie at cap, split
                    return 0.5

            q = rally_win_prob_from_model(p_h, p_a, srv)

            if srv == "h":
                # home serves
                win_val = V(s_h + 1, s_a, p_h, p_a, "h")
                lose_val = V(s_h, s_a + 1, p_h, rotate(p_a), "a")
                return q * win_val + (1 - q) * lose_val
            else:
                # away serves
                win_val = V(s_h, s_a + 1, p_h, p_a, "a")
                lose_val = V(s_h + 1, s_a, rotate(p_h), p_a, "h")
                return q * win_val + (1 - q) * lose_val

        return V(s_h0, s_a0, p_h0, p_a0, srv0)

    # ----------------------------------------------------------
    # internal helper
    # ----------------------------------------------------------
    def _clone_base_sim(self):
        bs = self.base_sim

        new_sim = bs.__class__(seed=None, best_of=bs.best_of)

        # copy params
        new_sim.global_logit = bs.global_logit
        new_sim.home_params = {
            "team_id": bs.home_params["team_id"],
            "team_name": bs.home_params["team_name"],
            "serve_team_adjustment": bs.home_params["serve_team_adjustment"],
            "receive_team_adjustment": bs.home_params["receive_team_adjustment"],
            "serve_pos": dict(bs.home_params["serve_pos"]),
            "receive_pos": dict(bs.home_params["receive_pos"]),
        }
        new_sim.away_params = {
            "team_id": bs.away_params["team_id"],
            "team_name": bs.away_params["team_name"],
            "serve_team_adjustment": bs.away_params["serve_team_adjustment"],
            "receive_team_adjustment": bs.away_params["receive_team_adjustment"],
            "serve_pos": dict(bs.away_params["serve_pos"]),
            "receive_pos": dict(bs.away_params["receive_pos"]),
        }
        new_sim.team_id_h = bs.team_id_h
        new_sim.team_id_a = bs.team_id_a
        new_sim.team_name_h = bs.team_name_h
        new_sim.team_name_a = bs.team_name_a
        new_sim.match_type = bs.match_type
        new_sim.match_date = bs.match_date

        # copy state
        new_sim.cur_set = bs.cur_set
        new_sim.set_won_h = bs.set_won_h
        new_sim.set_won_a = bs.set_won_a
        new_sim.point_h = bs.point_h
        new_sim.point_a = bs.point_a
        new_sim.p_h = bs.p_h
        new_sim.p_a = bs.p_a
        new_sim.serve_team = bs.serve_team
        new_sim._start_p_h = bs._start_p_h
        new_sim._start_p_a = bs._start_p_a

        # end conditions
        new_sim.end_set = bs.end_set
        new_sim.end_rally_in_set = bs.end_rally_in_set

        return new_sim



# ======================================================================
# MAIN EXAMPLE
# ======================================================================
if __name__ == "__main__":
    # your params file
    params_path = "params_out.csv"
    team_h_id = "6727"
    team_a_id = "6747"

    all_params = pd.read_csv(params_path)

    global_df = all_params.loc[all_params["par_type"] == "global"]
    team_home_df = all_params.loc[
        (all_params["par_type"] == "team") & (all_params["team_id"] == team_h_id)
    ]
    team_away_df = all_params.loc[
        (all_params["par_type"] == "team") & (all_params["team_id"] == team_a_id)
    ]

    # 1) make base simulator
    base_sim = VolleyballPointByPointSimulator(seed=None)
    base_sim.load_parameters(
        global_df,
        team_home_df,
        team_away_df,
        match_type="Amichevole",
        match_date="08/10/2025",
    )
    # start exactly like your DV snippet
    base_sim.set_initial_conditions(
        set_won_h=2,
        set_won_a=2,
        point_won_h=15,
        point_won_a=15,
        p_h=1,   # home setter in 1
        p_a=1,   # away setter in 1
        serve_team="h",
        current_set=5,
    )
    # full match
    base_sim.set_end_point(set_n=5, point_n=0)

    # 2) probability simulator
    prob_sim = VolleyballProbabilitySimulator(base_sim)

    # run, save
    big_df = prob_sim.run_simulations(n=10000, save_path=None)
    print("Home win probability:", prob_sim.home_win_prob())
    p_exact = prob_sim.home_win_analytical_calculations()
    print("Exact home set-win prob:", p_exact)

    # if you want to peek:
    print(big_df.head())
