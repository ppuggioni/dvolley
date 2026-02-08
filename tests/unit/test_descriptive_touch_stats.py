import pandas as pd
import pytest

from dvolley.domain.descriptive_touch_stats import (
    build_attack_quality_drilldown_table,
    build_descriptive_touch_stats,
)


def _make_rally_rows(
    *,
    match_id: str,
    rally_number: int,
    serving_team: str,
    point_won_by: str,
    home_setter: int,
    away_setter: int,
    serve_eval: str,
    reception_team: str | None = None,
    reception_eval: str | None = None,
    attack_team: str | None = None,
    attack_eval: str | None = None,
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
            "evaluation_code": serve_eval,
            "home_setter_position": home_setter,
            "visiting_setter_position": away_setter,
            "player_name": "Server",
        }
    ]

    next_id = start_unique_id + 1
    if reception_team is not None:
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
                "skill": "Reception",
                "team": reception_team,
                "evaluation_code": reception_eval,
                "home_setter_position": home_setter,
                "visiting_setter_position": away_setter,
                "player_name": "Receiver",
            }
        )
        next_id += 1

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
                "player_name": "Attacker",
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
    rows = []
    # Sideout rallies for team A (team_id=1): B serves.
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=1,
        serving_team="B",
        point_won_by="A",
        home_setter=1,
        away_setter=4,
        serve_eval="=",
        start_unique_id=1,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=2,
        serving_team="B",
        point_won_by="A",
        home_setter=1,
        away_setter=5,
        serve_eval="+",
        reception_team="A",
        reception_eval="+",
        attack_team="A",
        attack_eval="#",
        start_unique_id=10,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=3,
        serving_team="B",
        point_won_by="B",
        home_setter=2,
        away_setter=6,
        serve_eval="+",
        reception_team="A",
        reception_eval="+",
        attack_team="A",
        attack_eval="=",
        start_unique_id=20,
    )

    # Breakpoint rallies for team A (team_id=1): A serves.
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=4,
        serving_team="A",
        point_won_by="A",
        home_setter=1,
        away_setter=1,
        serve_eval="+",
        reception_team="B",
        reception_eval="+",
        attack_team="B",
        attack_eval="!",
        start_unique_id=30,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=5,
        serving_team="A",
        point_won_by="B",
        home_setter=2,
        away_setter=2,
        serve_eval="+",
        reception_team="B",
        reception_eval="+",
        attack_team="B",
        attack_eval="#",
        start_unique_id=40,
    )
    rows += _make_rally_rows(
        match_id="m1",
        rally_number=6,
        serving_team="A",
        point_won_by="B",
        home_setter=1,
        away_setter=3,
        serve_eval="=",
        start_unique_id=50,
    )
    return pd.DataFrame(rows)


def test_sideout_summary_counts_serve_errors_and_uses_within_rotation_share():
    df = _make_touch_df()
    result = build_descriptive_touch_stats(
        touches_df=df,
        team_id="1",
        mode="sideout",
        include_by_rotation=True,
        exclude_sideout_serve_errors=False,
    )
    summary = result.summary_table

    assert summary.loc["SERVE_ERROR", ("Total", "Actions")] == 1
    assert summary.loc["SERVE_ERROR", ("Total", "Successful")] == 1
    assert summary.loc["SERVE_ERROR", ("Total", "% successful")] == pytest.approx(1.0)
    assert ("Total", "% share 95% CI low") in summary.columns
    assert ("Total", "% share 95% CI high") in summary.columns
    assert 0.0 <= summary.loc["SERVE_ERROR", ("Total", "% successful 95% CI low")] <= 1.0
    assert 0.0 <= summary.loc["SERVE_ERROR", ("Total", "% successful 95% CI high")] <= 1.0

    assert summary.loc["+", ("P1", "Actions")] == 1
    assert summary.loc["+", ("P1", "% share")] == pytest.approx(0.5)


def test_sideout_summary_can_exclude_serve_errors():
    df = _make_touch_df()
    result = build_descriptive_touch_stats(
        touches_df=df,
        team_id="1",
        mode="sideout",
        include_by_rotation=False,
        exclude_sideout_serve_errors=True,
    )
    summary = result.summary_table

    assert "SERVE_ERROR" not in summary.index
    assert summary.loc["+", ("Total", "Actions")] == 2
    assert summary.loc["+", ("Total", "% share")] == pytest.approx(1.0)


def test_sideout_attack_quality_drilldown_for_plus():
    df = _make_touch_df()
    result = build_descriptive_touch_stats(
        touches_df=df,
        team_id="1",
        mode="sideout",
        include_by_rotation=False,
    )
    drilldown = build_attack_quality_drilldown_table(
        rallies_df=result.rallies_df,
        event_key="+",
        include_by_rotation=False,
    )

    assert drilldown.loc["#", ("Total", "Actions")] == 1
    assert drilldown.loc["#", ("Total", "% successful")] == pytest.approx(1.0)
    assert drilldown.loc["=", ("Total", "Actions")] == 1
    assert drilldown.loc["=", ("Total", "% successful")] == pytest.approx(0.0)


def test_breakpoint_summary_and_drilldown_for_plus():
    df = _make_touch_df()
    result = build_descriptive_touch_stats(
        touches_df=df,
        team_id="1",
        mode="breakpoint",
        include_by_rotation=False,
    )
    summary = result.summary_table

    assert summary.loc["+", ("Total", "Actions")] == 2
    assert summary.loc["+", ("Total", "Successful")] == 1
    assert summary.loc["+", ("Total", "% successful")] == pytest.approx(0.5)
    assert (
        summary.loc["+", ("Total", "% successful 95% CI low")]
        <= summary.loc["+", ("Total", "% successful")]
        <= summary.loc["+", ("Total", "% successful 95% CI high")]
    )

    drilldown = build_attack_quality_drilldown_table(
        rallies_df=result.rallies_df,
        event_key="+",
        include_by_rotation=False,
    )
    assert drilldown.loc["!", ("Total", "Actions")] == 1
    assert drilldown.loc["!", ("Total", "% successful")] == pytest.approx(1.0)
    assert drilldown.loc["#", ("Total", "Actions")] == 1
    assert drilldown.loc["#", ("Total", "% successful")] == pytest.approx(0.0)

    no_attack = build_attack_quality_drilldown_table(
        rallies_df=result.rallies_df,
        event_key="SERVE_ERROR",
        include_by_rotation=False,
    )
    assert no_attack.empty
