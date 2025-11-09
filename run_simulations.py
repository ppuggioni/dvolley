# run_rotation_grid.py

import pandas as pd
from simulator import (
    VolleyballPointByPointSimulator,
    VolleyballProbabilitySimulator,
)


def main():
    params_path = "params_out.csv"
    team_h_id = "6727"
    team_a_id = "6747"

    # load fitted parameters
    all_params = pd.read_csv(params_path)

    global_df = all_params.loc[all_params["par_type"] == "global"]

    # if team_id in CSV is int, cast to int; if it's str, drop int()
    team_home_df = all_params.loc[
        (all_params["par_type"] == "team") & (all_params["team_id"] == team_h_id)
    ]
    team_away_df = all_params.loc[
        (all_params["par_type"] == "team") & (all_params["team_id"] == team_a_id)
    ]

    results = []

    # loop over all 6x6 rotations
    for rot_h in range(1, 7):      # home setter starting pos
        for rot_a in range(1, 7):  # away setter starting pos
            # fresh base simulator
            base_sim = VolleyballPointByPointSimulator(seed=None)
            base_sim.load_parameters(
                global_df,
                team_home_df,
                team_away_df,
                match_type="Amichevole",
                match_date="08/10/2025",
            )

            # set this pair of starting rotations
            base_sim.set_initial_conditions(
                set_won_h=2,
                set_won_a=2,
                point_won_h=0,
                point_won_a=0,
                p_h=rot_h,
                p_a=rot_a,
                serve_team="h",  # consistent with your DV example
                current_set=5,
            )
            # simulate full set
            base_sim.set_end_point(set_n=5, point_n=0)

            prob_sim = VolleyballProbabilitySimulator(base_sim)

            # run several simulations for this pair
            # _ = prob_sim.run_simulations(n=10000, save_path=None)
            # print(_.loc[(_['p_h'] == 1) & (_['p_a'] == 1) & _['serve_h'] == 1]['point_won_h'].mean())
            win_prob_h = prob_sim.home_win_analytical_calculations()

            results.append(
                {
                    "starting_rotation_h": rot_h,
                    "starting_rotation_a": rot_a,
                    "win_prob_h": win_prob_h,
                }
            )

    df_res = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # averages by home rotation (across ALL away rotations)
    # -> opponent rotation becomes 0
    # ------------------------------------------------------------------
    home_avgs = (
        df_res.groupby("starting_rotation_h")["win_prob_h"]
        .mean()
        .reset_index()
    )
    home_avgs["starting_rotation_a"] = 0  # meaning: avg over all away rotations
    # order columns
    home_avgs = home_avgs[["starting_rotation_h", "starting_rotation_a", "win_prob_h"]]

    # ------------------------------------------------------------------
    # averages by away rotation (across ALL home rotations)
    # -> opponent rotation becomes 0
    # ------------------------------------------------------------------
    away_avgs = (
        df_res.groupby("starting_rotation_a")["win_prob_h"]
        .mean()
        .reset_index()
    )
    away_avgs["starting_rotation_h"] = 0  # meaning: avg over all home rotations
    away_avgs = away_avgs[["starting_rotation_h", "starting_rotation_a", "win_prob_h"]]

    total_averages = pd.DataFrame(index=[0], data=[[0, 0, away_avgs['win_prob_h'].mean()]], columns=["starting_rotation_h", "starting_rotation_a", "win_prob_h"])

    # append both kinds of averages
    df_res_all = pd.concat([df_res, home_avgs, away_avgs, total_averages], ignore_index=True)

    print("All combinations + averaged rows:")
    print(df_res_all.sort_values(
        by=["starting_rotation_h", "starting_rotation_a"]
    ).reset_index(drop=True))

    # matrix view for the raw 6x6 (without the 0 rows)
    pivot = df_res_all.pivot(
        index="starting_rotation_h",
        columns="starting_rotation_a",
        values="win_prob_h",
    )
    print("\nMatrix view (home rows, away cols):")
    print(pivot.round(3))

    # save if needed
    df_res_all.to_csv("rotation_win_probs.csv", index=False)


if __name__ == "__main__":
    main()
