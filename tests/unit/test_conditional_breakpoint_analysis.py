import pandas as pd

from dvolley.domain.conditional_breakpoint_analysis import build_conditional_breakpoint_analysis


def _make_rally_rows(
    *,
    match_id: str,
    rally_number: int,
    serving_team: str,
    point_won_by: str,
    home_setter: int,
    away_setter: int,
    attack_team: str | None,
    attack_eval: str | None,
    attack_player: str | None,
    start_unique_id: int,
):
    rows = [
        {
            "unique_row_id": start_unique_id,
            "match_alternative_id": match_id,
            "match_date": "2025-02-11",
            "home_team_id": "1",
            "home_team": "A",
            "visiting_team_id": "2",
            "visiting_team": "B",
            "set_number": 1,
            "rally_number": rally_number,
            "serving_team": serving_team,
            "point_won_by": point_won_by,
            "skill": "Serve",
            "team": serving_team,
            "evaluation_code": "+",
            "home_setter_position": home_setter,
            "visiting_setter_position": away_setter,
            "player_name": "Server",
        }
    ]

    next_id = start_unique_id + 1
    if attack_team is not None:
        rows.append(
            {
                "unique_row_id": next_id,
                "match_alternative_id": match_id,
                "match_date": "2025-02-11",
                "home_team_id": "1",
                "home_team": "A",
                "visiting_team_id": "2",
                "visiting_team": "B",
                "set_number": 1,
                "rally_number": rally_number,
                "serving_team": serving_team,
                "point_won_by": point_won_by,
                "skill": "Attack",
                "team": attack_team,
                "evaluation_code": attack_eval,
                "home_setter_position": home_setter,
                "visiting_setter_position": away_setter,
                "player_name": attack_player,
            }
        )
        next_id += 1

    rows.append(
        {
            "unique_row_id": next_id,
            "match_alternative_id": match_id,
            "match_date": "2025-02-11",
            "home_team_id": "1",
            "home_team": "A",
            "visiting_team_id": "2",
            "visiting_team": "B",
            "set_number": 1,
            "rally_number": rally_number,
            "serving_team": serving_team,
            "point_won_by": point_won_by,
            "skill": "Point",
            "team": point_won_by,
            "evaluation_code": None,
            "home_setter_position": home_setter,
            "visiting_setter_position": away_setter,
            "player_name": None,
        }
    )
    return rows


def test_sideout_mode_uses_receiving_attack_and_excludes_no_first_attack():
    rows = []
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=1,
        serving_team="B",
        point_won_by="A",
        home_setter=1,
        away_setter=4,
        attack_team="A",
        attack_eval="#",
        attack_player="P1",
        start_unique_id=1,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=2,
        serving_team="B",
        point_won_by="B",
        home_setter=2,
        away_setter=5,
        attack_team="A",
        attack_eval="=",
        attack_player="P1",
        start_unique_id=10,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=3,
        serving_team="B",
        point_won_by="B",
        home_setter=3,
        away_setter=6,
        attack_team="A",
        attack_eval="!",
        attack_player="P2",
        start_unique_id=20,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=4,
        serving_team="B",
        point_won_by="B",
        home_setter=4,
        away_setter=1,
        attack_team=None,  # must be excluded
        attack_eval=None,
        attack_player=None,
        start_unique_id=30,
    )

    df = pd.DataFrame(rows)
    result = build_conditional_breakpoint_analysis(df, team_id="1", mode="sideout")

    assert len(result.rally_df) == 3
    assert result.diagnostics["skipped_no_first_attack"] == 1

    quality_probs = {
        row["Attack_quality"]: row["Point_won_probability"]
        for _, row in result.quality_summary.iterrows()
        if row["Attack_quality"] != "Total"
    }
    assert "Condition_share_of_first_attacks 95% CI low" in result.quality_summary.columns
    assert "Condition_share_of_first_attacks 95% CI high" in result.quality_summary.columns
    assert "Point_won_probability 95% CI low" in result.quality_summary.columns
    assert "Point_won_probability 95% CI high" in result.quality_summary.columns
    quality_shares = {
        row["Attack_quality"]: row["Condition_share_of_first_attacks"]
        for _, row in result.quality_summary.iterrows()
        if row["Attack_quality"] != "Total"
    }
    assert quality_probs["#"] == 1.0
    assert quality_probs["="] == 0.0
    assert quality_probs["!"] == 0.0
    assert quality_shares["#"] == 1 / 3
    assert quality_shares["="] == 1 / 3
    assert quality_shares["!"] == 1 / 3

    by_player = result.player_summary.set_index("Player")
    assert int(by_player.loc["P1", "Attempts"]) == 2
    assert int(by_player.loc["P2", "Attempts"]) == 1


def test_breakpoint_mode_rotation_is_selected_server_rotation():
    rows = []
    rows += _make_rally_rows(
        match_id="m2",
        rally_number=1,
        serving_team="A",
        point_won_by="A",
        home_setter=5,  # selected server rotation
        away_setter=2,  # opponent receiving rotation (must not be used)
        attack_team="B",
        attack_eval="-",
        attack_player="Opp1",
        start_unique_id=100,
    )
    rows += _make_rally_rows(
        match_id="m2",
        rally_number=2,
        serving_team="A",
        point_won_by="B",
        home_setter=6,
        away_setter=1,
        attack_team="B",
        attack_eval="#",
        attack_player="Opp2",
        start_unique_id=110,
    )

    df = pd.DataFrame(rows)
    result = build_conditional_breakpoint_analysis(df, team_id="1", mode="breakpoint")

    assert sorted(result.rally_df["selected_rotation"].tolist()) == [5, 6]
    assert result.rotation_axis_label == "Serving rotation (selected team)"

    row = result.rotation_quality_summary[
        (result.rotation_quality_summary["Rotation"] == 5)
        & (result.rotation_quality_summary["Attack_quality"] == "-")
    ].iloc[0]
    assert row["Point_won_probability"] == 1.0
    assert "Condition_share_within_rotation 95% CI low" in result.rotation_quality_summary.columns
    assert "Condition_share_within_rotation 95% CI high" in result.rotation_quality_summary.columns
    assert 0.0 <= row["Point_won_probability 95% CI low"] <= 1.0
    assert 0.0 <= row["Point_won_probability 95% CI high"] <= 1.0
    assert row["Point_won_probability 95% CI low"] <= row["Point_won_probability 95% CI high"]
    assert row["Condition_share_of_first_attacks"] == 0.5
    assert row["Condition_share_within_rotation"] == 1.0

    quality_probs = {
        row["Attack_quality"]: row["Point_won_probability"]
        for _, row in result.quality_summary.iterrows()
        if row["Attack_quality"] != "Total"
    }
    assert quality_probs["#"] == 0.0
    assert result.player_summary.empty
    assert result.player_quality_summary.empty
