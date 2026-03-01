import pandas as pd
import pytest

from dvolley.domain.player_analysis import build_player_sideout_dataset
from dvolley.domain.team_player_comparison import build_team_player_comparison


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


def test_total_attack_table_has_players_total_and_efficiency():
    dataset = build_player_sideout_dataset(_make_touch_df(), team_id="1")
    result = build_team_player_comparison(dataset)

    table = result.total_attack_table
    assert "TOTAL" in table.index
    assert ("Efficiency", "Score") in table.columns

    assert table.loc["Alice", ("#", "Count")] == 3
    assert table.loc["Alice", ("+", "Count")] == 1
    assert table.loc["Alice", ("/", "Count")] == 1
    assert table.loc["Alice", ("!", "Count")] == 1
    assert table.loc["Alice", ("Efficiency", "Score")] == pytest.approx(2.0 / 3.0)

    assert table.loc["TOTAL", ("+", "Count")] == 2
    assert table.loc["TOTAL", ("+", "% rally won")] == pytest.approx(0.5)
    assert table.loc["TOTAL", ("Efficiency", "Score")] == pytest.approx(0.25)


def test_first_attack_by_pass_quality_tables_exist_and_are_filtered():
    dataset = build_player_sideout_dataset(_make_touch_df(), team_id="1")
    result = build_team_player_comparison(dataset)

    tables = result.first_attack_by_pass_quality
    assert set(tables.keys()) == {"#", "+", "-", "/"}

    hash_table = tables["#"]
    assert hash_table.loc["Alice", ("#", "Count")] == 1
    assert hash_table.loc["TOTAL", ("#", "% rally won")] == pytest.approx(1.0)

    plus_table = tables["+"]
    assert plus_table.loc["Alice", ("!", "Count")] == 1
    assert plus_table.loc["TOTAL", ("!", "% rally won")] == pytest.approx(1.0)

    minus_table = tables["-"]
    assert minus_table.loc["Bob", ("=", "Count")] == 1
    assert minus_table.loc["TOTAL", ("=", "% rally won")] == pytest.approx(0.0)

    slash_table = tables["/"]
    assert slash_table.loc["Carol", ("+", "Count")] == 1
    assert slash_table.loc["TOTAL", ("+", "% rally won")] == pytest.approx(0.0)


def test_pass_quality_table_has_player_rows_and_total():
    dataset = build_player_sideout_dataset(_make_touch_df(), team_id="1")
    result = build_team_player_comparison(dataset)

    table = result.pass_quality_table
    assert "TOTAL" in table.index
    assert table.loc["Alice", ("#", "Count")] == 1
    assert table.loc["Alice", ("/", "Count")] == 1
    assert table.loc["Alice", ("Efficiency", "Score")] == pytest.approx(0.0)
    assert table.loc["Receiver1", ("+", "Count")] == 1
    assert table.loc["Receiver1", ("Efficiency", "Score")] == pytest.approx(1.0)
    assert table.loc["Receiver2", ("-", "Count")] == 1
    assert table.loc["Receiver2", ("Efficiency", "Score")] == pytest.approx(-1.0)
    assert table.loc["TOTAL", ("#", "% rally won")] == pytest.approx(1.0)
    assert table.loc["TOTAL", ("/", "% rally won")] == pytest.approx(0.0)
