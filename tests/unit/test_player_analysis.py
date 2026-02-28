import pandas as pd
import pytest

from dvolley.domain.player_analysis import (
    build_player_analysis_tables,
    build_player_sideout_dataset,
)


def _make_rally_rows(
    *,
    match_id: str,
    rally_number: int,
    serving_team: str,
    point_won_by: str,
    home_setter: int,
    away_setter: int,
    actions: list[dict[str, str | None]],
    start_unique_id: int,
) -> list[dict[str, object]]:
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
    for action in actions:
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
                "skill": action["skill"],
                "team": action["team"],
                "evaluation_code": action.get("eval"),
                "home_setter_position": home_setter,
                "visiting_setter_position": away_setter,
                "player_name": action.get("player"),
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


def _make_touch_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # Sideout rallies (B serves, A receives).
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=1,
        serving_team="B",
        point_won_by="A",
        home_setter=1,
        away_setter=4,
        actions=[
            {"skill": "Reception", "team": "A", "eval": "+", "player": "Receiver1"},
            {"skill": "Attack", "team": "A", "eval": "!", "player": "Alice"},
            {"skill": "Attack", "team": "A", "eval": "#", "player": "Alice"},
        ],
        start_unique_id=1,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=2,
        serving_team="B",
        point_won_by="A",
        home_setter=2,
        away_setter=5,
        actions=[
            {"skill": "Reception", "team": "A", "eval": "#", "player": "Alice"},
            {"skill": "Attack", "team": "A", "eval": "#", "player": "Alice"},
        ],
        start_unique_id=20,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=3,
        serving_team="B",
        point_won_by="B",
        home_setter=1,
        away_setter=6,
        actions=[
            {"skill": "Reception", "team": "A", "eval": "-", "player": "Receiver2"},
            {"skill": "Attack", "team": "A", "eval": "=", "player": "Bob"},
            {"skill": "Attack", "team": "A", "eval": "/", "player": "Alice"},
        ],
        start_unique_id=40,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=4,
        serving_team="B",
        point_won_by="B",
        home_setter=3,
        away_setter=1,
        actions=[
            {"skill": "Reception", "team": "A", "eval": "/", "player": "Alice"},
            {"skill": "Attack", "team": "A", "eval": "+", "player": "Carol"},
        ],
        start_unique_id=60,
    )

    # Breakpoint rally (A serves): ignored by sideout-rally tables but included in all-attack tables.
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=5,
        serving_team="A",
        point_won_by="A",
        home_setter=4,
        away_setter=2,
        actions=[
            {"skill": "Reception", "team": "B", "eval": "+", "player": "OppR"},
            {"skill": "Attack", "team": "A", "eval": "#", "player": "Alice"},
        ],
        start_unique_id=80,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=6,
        serving_team="A",
        point_won_by="A",
        home_setter=5,
        away_setter=3,
        actions=[
            {"skill": "Reception", "team": "B", "eval": "!", "player": "OppR2"},
            {"skill": "Attack", "team": "A", "eval": "+", "player": "Alice"},
        ],
        start_unique_id=100,
    )

    return pd.DataFrame(rows)


def test_build_player_sideout_dataset_extracts_first_and_non_first_attacks():
    df = _make_touch_df()
    dataset = build_player_sideout_dataset(df, team_id="1")

    assert len(dataset.sideout_rallies) == 4
    assert "Alice" in dataset.players
    assert "Bob" in dataset.players
    assert "Carol" in dataset.players

    alice_attacks = dataset.team_attacks[dataset.team_attacks["player_name"] == "Alice"]
    assert len(alice_attacks) == 6
    assert int(alice_attacks["is_first_attack"].sum()) == 2
    assert int((alice_attacks["is_first_attack"] == 0).sum()) == 4
    assert dataset.diagnostics["sideout_rallies"] == 4
    assert dataset.diagnostics["attack_rows"] == len(dataset.team_attacks)


def test_build_player_analysis_tables_returns_expected_breakdowns():
    df = _make_touch_df()
    dataset = build_player_sideout_dataset(df, team_id="1")
    result = build_player_analysis_tables(dataset, "Alice", include_by_rotation=True)

    assert result.first_attack_attempts == 2
    assert result.total_attack_attempts == 6
    assert result.non_first_attack_attempts == 4
    assert result.pass_attempts == 2

    first_attack = result.first_attack_overall
    assert first_attack.loc["#", ("Total", "Actions")] == 1
    assert first_attack.loc["!", ("Total", "Actions")] == 1
    assert first_attack.loc["#", ("P2", "Actions")] == 1
    assert first_attack.loc["!", ("P1", "Actions")] == 1
    assert first_attack.loc["#", ("Total", "Rallies won")] == 1
    assert first_attack.loc["!", ("Total", "Rallies won")] == 1
    assert first_attack.loc["#", ("Total", "% rally won")] == pytest.approx(1.0)
    assert first_attack.loc["!", ("Total", "% rally won")] == pytest.approx(1.0)
    assert ("Total", "% successful") not in first_attack.columns

    by_pass = result.first_attack_by_pass_quality
    assert set(by_pass.keys()) == {"#", "+"}
    assert by_pass["#"].loc["#", ("Total", "Actions")] == 1
    assert by_pass["+"].loc["!", ("Total", "Actions")] == 1
    assert by_pass["#"].loc["#", ("Total", "% rally won")] == pytest.approx(1.0)
    assert by_pass["+"].loc["!", ("Total", "% rally won")] == pytest.approx(1.0)
    assert ("Total", "% successful") not in by_pass["#"].columns

    total_attack = result.total_attack_table
    assert total_attack.loc["#", ("Total", "Actions")] == 3
    assert total_attack.loc["!", ("Total", "Actions")] == 1
    assert total_attack.loc["+", ("Total", "Actions")] == 1
    assert total_attack.loc["/", ("Total", "Actions")] == 1
    assert total_attack.loc["#", ("Total", "% rally won")] == pytest.approx(1.0)
    assert total_attack.loc["!", ("Total", "% rally won")] == pytest.approx(1.0)
    assert total_attack.loc["+", ("Total", "% rally won")] == pytest.approx(1.0)
    assert total_attack.loc["/", ("Total", "% rally won")] == pytest.approx(0.0)
    assert ("Total", "% successful") not in total_attack.columns

    non_first = result.non_first_attack_table
    assert non_first.loc["#", ("Total", "Actions")] == 2
    assert non_first.loc["+", ("Total", "Actions")] == 1
    assert non_first.loc["/", ("Total", "Actions")] == 1
    assert non_first.loc["#", ("Total", "% rally won")] == pytest.approx(1.0)
    assert non_first.loc["+", ("Total", "% rally won")] == pytest.approx(1.0)
    assert non_first.loc["/", ("Total", "% rally won")] == pytest.approx(0.0)
    assert ("Total", "% successful") not in non_first.columns

    pass_table = result.pass_quality_table
    assert pass_table.loc["#", ("Total", "Actions")] == 1
    assert pass_table.loc["/", ("Total", "Actions")] == 1
    assert pass_table.loc["#", ("Total", "% rally won")] == pytest.approx(1.0)
    assert pass_table.loc["/", ("Total", "% rally won")] == pytest.approx(0.0)
    assert ("Total", "% successful") not in pass_table.columns
