import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional


class VolleyballPointByPointSimulator:
    """
    Point-by-point volleyball simulator based on fitted *breakpoint/sideout* parameters.

    Expected parameter file rows (like params_out_break_sideout.csv):
      - par_name == "global_breakpoint" (single global row)
      - per-team rows with:
            "breakpoint_team_adjustment"
            "sideout_team_adjustment"
            "breakpoint_pos_1" .. "breakpoint_pos_6"
            "sideout_pos_1"    .. "sideout_pos_6"

    Model for each rally (server tries to score a break point):

        logit P(server wins) =
            global_breakpoint
          + server.breakpoint_team_adjustment
          + server.breakpoint_pos[current_rotation]
          - receiver.sideout_team_adjustment
          - receiver.sideout_pos[current_rotation]

    So: **bigger sideout** makes it harder for the server to score → we subtract it.
    """

    def __init__(self, seed: Optional[int] = 123, best_of: int = 5):
        self.rng = np.random.default_rng(seed)
        self.best_of = best_of  # e.g. 5 -> first to 3

        # parameters (filled by load_parameters)
        self.global_logit: float = 0.0
        self.home_params: dict = {}
        self.away_params: dict = {}
        self.team_id_h: Optional[str] = None
        self.team_id_a: Optional[str] = None
        self.team_name_h: Optional[str] = None
        self.team_name_a: Optional[str] = None

        # optional metadata
        self.match_type: str = ""
        self.match_date: str = ""

        # current match state
        self.cur_set: int = 1
        self.set_won_h: int = 0
        self.set_won_a: int = 0
        self.point_h: int = 0
        self.point_a: int = 0
        self.p_h: int = 1   # home setter/rotation 1..6
        self.p_a: int = 1   # away rotation
        self.serve_team: str = "a"  # 'h' or 'a'

        # remember initial rotations to reset at new set
        self._start_p_h: int = 1
        self._start_p_a: int = 1

        # stop condition (0,0 => full match)
        self.end_set: int = 0
        self.end_rally_in_set: int = 0

    # ------------------------------------------------------------------
    # LOAD PARAMETERS
    # ------------------------------------------------------------------
    def load_parameters(
        self,
        global_df: pd.DataFrame,
        team_home_df: pd.DataFrame,
        team_away_df: pd.DataFrame,
        match_type: str = "",
        match_date: str = "",
    ):
        """
        Parse the 3 dataframes coming from params_out_break_sideout.csv.
        """
        g = global_df.loc[global_df["par_name"] == "global_breakpoint"]
        if g.empty:
            raise ValueError("global_df must contain par_name == 'global_breakpoint'")
        self.global_logit = float(g["par_value"].iloc[0])

        self.match_type = match_type
        self.match_date = match_date

        def _parse_team(df: pd.DataFrame) -> dict:
            # assume df is filtered to one team
            team_id = str(df["team_id"].iloc[0])
            team_name = df["team_name"].iloc[0]

            team_dict = {
                "team_id": team_id,
                "team_name": team_name,
                "breakpoint_team_adjustment": 0.0,
                "sideout_team_adjustment": 0.0,
                "break_pos": {i: 0.0 for i in range(1, 7)},
                "sideout_pos": {i: 0.0 for i in range(1, 7)},
            }

            for _, row in df.iterrows():
                par_name = row["par_name"]
                val = float(row["par_value"])

                if par_name == "breakpoint_team_adjustment":
                    team_dict["breakpoint_team_adjustment"] = val
                elif par_name == "sideout_team_adjustment":
                    team_dict["sideout_team_adjustment"] = val
                elif par_name.startswith("breakpoint_pos_"):
                    pos = int(par_name.split("_")[-1])
                    team_dict["break_pos"][pos] = val
                elif par_name.startswith("sideout_pos_"):
                    pos = int(par_name.split("_")[-1])
                    team_dict["sideout_pos"][pos] = val

            return team_dict

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
        Set starting situation for the match/set.
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
        - (0,0): play full match
        - (K,0): play to END of set K
        - (K,M): play to rally M of set K
        """
        self.end_set = set_n
        self.end_rally_in_set = point_n

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _rotate_on_sideout(pos: int) -> int:
        """volleyball rotation order, 6 → 5 → 4 → 3 → 2 → 1 → 6"""
        return pos - 1 if pos > 1 else 6

    def _rally_win_prob(self) -> float:
        """
        p(server wins) = σ(global_breakpoint + server_break + server_break_pos
                           - recv_sideout - recv_sideout_pos)
        """
        logit = self.global_logit

        if self.serve_team == "h":
            # server = home, receiver = away
            logit += self.home_params["breakpoint_team_adjustment"]
            logit += self.home_params["break_pos"].get(self.p_h, 0.0)
            logit -= self.away_params["sideout_team_adjustment"]
            logit -= self.away_params["sideout_pos"].get(self.p_a, 0.0)
        else:
            # server = away, receiver = home
            logit += self.away_params["breakpoint_team_adjustment"]
            logit += self.away_params["break_pos"].get(self.p_a, 0.0)
            logit -= self.home_params["sideout_team_adjustment"]
            logit -= self.home_params["sideout_pos"].get(self.p_h, 0.0)

        return self._sigmoid(logit)

    # ------------------------------------------------------------------
    # SIMULATION
    # ------------------------------------------------------------------
    def run_simulation(self) -> pd.DataFrame:
        """
        Simulate rally-by-rally until the stop condition is met.
        """
        rows = []
        rallies_in_this_set = 0

        while True:
            # top-of-loop stop for partial sims
            if self.end_set != 0:
                if self.cur_set > self.end_set:
                    break
                if self.cur_set == self.end_set and self.end_rally_in_set > 0:
                    if rallies_in_this_set >= self.end_rally_in_set:
                        break

            # pre state
            pre_set_won_h = self.set_won_h
            pre_set_won_a = self.set_won_a
            pre_point_won_h = self.point_h
            pre_point_won_a = self.point_a
            pre_p_h = self.p_h
            pre_p_a = self.p_a
            pre_serve_team = self.serve_team

            # sample rally
            p_server = self._rally_win_prob()
            server_wins = self.rng.random() < p_server

            if pre_serve_team == "h":
                if server_wins:
                    # home scored on serve (breakpoint)
                    self.point_h += 1
                    point_won_team = "h"
                    point_won_h = 1
                    point_won_a = 0
                    # home keeps serving, no rotation
                    self.serve_team = "h"
                else:
                    # away sided out
                    self.point_a += 1
                    point_won_team = "a"
                    point_won_h = 0
                    point_won_a = 1
                    # away will serve next, away rotates
                    self.serve_team = "a"
                    self.p_a = self._rotate_on_sideout(self.p_a)
            else:
                # away was serving
                if server_wins:
                    # away scored on serve
                    self.point_a += 1
                    point_won_team = "a"
                    point_won_h = 0
                    point_won_a = 1
                    self.serve_team = "a"
                else:
                    # home sided out
                    self.point_h += 1
                    point_won_team = "h"
                    point_won_h = 1
                    point_won_a = 0
                    self.serve_team = "h"
                    self.p_h = self._rotate_on_sideout(self.p_h)

            # check set end
            set_finished = False
            target_points = 25 if self.cur_set < self.best_of else 15
            if (self.point_h >= target_points or self.point_a >= target_points) and \
               abs(self.point_h - self.point_a) >= 2:
                set_finished = True

            post_set_won_h = self.set_won_h
            post_set_won_a = self.set_won_a

            if set_finished:
                # assign set
                if self.point_h > self.point_a:
                    self.set_won_h += 1
                else:
                    self.set_won_a += 1

                post_set_won_h = self.set_won_h
                post_set_won_a = self.set_won_a

                # go to next set
                self.cur_set += 1
                self.point_h = 0
                self.point_a = 0
                rallies_in_this_set = 0
                # reset rotations
                self.p_h = self._start_p_h
                self.p_a = self._start_p_a
                # pick starting server of next set
                self.serve_team = "a" if self.cur_set % 2 == 1 else "h"
            else:
                rallies_in_this_set += 1

            # row
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
                    "serve_h": 1 if pre_serve_team == "h" else 0,
                    "serve_a": 1 if pre_serve_team == "a" else 0,
                    "serve_team": pre_serve_team,
                }
            )

            # match end?
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
# PROBABILITY WRAPPER
# ======================================================================
class VolleyballProbabilitySimulator:
    """
    Wraps a VolleyballPointByPointSimulator and provides:
      - run_simulations(...)
      - home_win_prob()
      - home_win_analytical_calculations(...)
    All using the SAME breakpoint/sideout formula.
    """

    def __init__(self, base_simulator: VolleyballPointByPointSimulator):
        self.base_sim = base_simulator
        self._last_home_win_rate = None

    def _clone_base_sim(self) -> VolleyballPointByPointSimulator:
        bs = self.base_sim
        new_sim = VolleyballPointByPointSimulator(seed=None, best_of=bs.best_of)

        # copy parameters
        new_sim.global_logit = bs.global_logit
        new_sim.home_params = {
            "team_id": bs.home_params["team_id"],
            "team_name": bs.home_params["team_name"],
            "breakpoint_team_adjustment": bs.home_params["breakpoint_team_adjustment"],
            "sideout_team_adjustment": bs.home_params["sideout_team_adjustment"],
            "break_pos": dict(bs.home_params["break_pos"]),
            "sideout_pos": dict(bs.home_params["sideout_pos"]),
        }
        new_sim.away_params = {
            "team_id": bs.away_params["team_id"],
            "team_name": bs.away_params["team_name"],
            "breakpoint_team_adjustment": bs.away_params["breakpoint_team_adjustment"],
            "sideout_team_adjustment": bs.away_params["sideout_team_adjustment"],
            "break_pos": dict(bs.away_params["break_pos"]),
            "sideout_pos": dict(bs.away_params["sideout_pos"]),
        }

        # copy meta
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

        # copy end condition
        new_sim.end_set = bs.end_set
        new_sim.end_rally_in_set = bs.end_rally_in_set

        return new_sim

    # ----------------------------------------------------------
    # Monte Carlo
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
        self._last_home_win_rate = home_wins / n

        if save_path:
            big_df.to_csv(save_path, index=False, encoding="utf-8")

        return big_df

    def home_win_prob(self) -> float:
        if self._last_home_win_rate is None:
            raise RuntimeError("Run run_simulations(...) first.")
        return self._last_home_win_rate

    # ----------------------------------------------------------
    # Analytical set win
    # ----------------------------------------------------------
    def home_win_analytical_calculations(self, max_extra: int = 20) -> float:
        """
        Compute P(home wins THIS set) from current state, using the same
        breakpoint/sideout logit. max_extra bounds very long deuces.
        """
        bs = self.base_sim

        s_h0 = bs.point_h
        s_a0 = bs.point_a
        p_h0 = bs.p_h
        p_a0 = bs.p_a
        srv0 = bs.serve_team
        cur_set = bs.cur_set

        target_points = 25 if cur_set < bs.best_of else 15
        cap = target_points + max_extra

        gl = bs.global_logit
        hp = bs.home_params
        ap = bs.away_params

        def rally_prob(p_h: int, p_a: int, srv: str) -> float:
            logit = gl
            if srv == "h":
                logit += hp["breakpoint_team_adjustment"]
                logit += hp["break_pos"].get(p_h, 0.0)
                logit -= ap["sideout_team_adjustment"]
                logit -= ap["sideout_pos"].get(p_a, 0.0)
            else:
                logit += ap["breakpoint_team_adjustment"]
                logit += ap["break_pos"].get(p_a, 0.0)
                logit -= hp["sideout_team_adjustment"]
                logit -= hp["sideout_pos"].get(p_h, 0.0)
            return 1.0 / (1.0 + np.exp(-logit))

        def rotate(pos: int) -> int:
            return pos - 1 if pos > 1 else 6

        @lru_cache(maxsize=None)
        def V(s_h: int, s_a: int, p_h: int, p_a: int, srv: str) -> float:
            # normal terminal (win by 2)
            if (s_h >= target_points or s_a >= target_points) and abs(s_h - s_a) >= 2:
                return 1.0 if s_h > s_a else 0.0

            # bounded terminal
            if s_h >= cap or s_a >= cap:
                if s_h > s_a:
                    return 1.0
                if s_a > s_h:
                    return 0.0
                return 0.5

            q = rally_prob(p_h, p_a, srv)

            if srv == "h":
                win_val = V(s_h + 1, s_a, p_h, p_a, "h")
                lose_val = V(s_h, s_a + 1, p_h, rotate(p_a), "a")
                return q * win_val + (1 - q) * lose_val
            else:
                win_val = V(s_h, s_a + 1, p_h, p_a, "a")
                lose_val = V(s_h + 1, s_a, rotate(p_h), p_a, "h")
                return q * win_val + (1 - q) * lose_val

        return V(s_h0, s_a0, p_h0, p_a0, srv0)


# quick manual test
if __name__ == "__main__":
    params_path = "params_out_break_sideout.csv"
    team_h_id = "6727"
    team_a_id = "6736"

    all_params = pd.read_csv(params_path, dtype={"team_id": str})
    global_df = all_params[all_params["par_type"] == "global"]
    team_home_df = all_params[(all_params["par_type"] == "team") & (all_params["team_id"] == team_h_id)]
    team_away_df = all_params[(all_params["par_type"] == "team") & (all_params["team_id"] == team_a_id)]

    base = VolleyballPointByPointSimulator(seed=42)
    base.load_parameters(global_df, team_home_df, team_away_df,
                         match_type="Amichevole",
                         match_date="08/10/2025")
    base.set_initial_conditions(
        set_won_h=0, set_won_a=0,
        point_won_h=0, point_won_a=0,
        p_h=6, p_a=5,
        serve_team="a",
        current_set=1,
    )
    base.set_end_point(set_n=1, point_n=0)

    prob = VolleyballProbabilitySimulator(base)
    df = prob.run_simulations(n=10000)
    print("MC home win prob:", prob.home_win_prob())
    print("Analytical home set-win prob:", prob.home_win_analytical_calculations())
