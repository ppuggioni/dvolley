from dvolley.data.dvw_parser import dvw_rallies_to_df


def test_dvw_rallies_to_df_parses_basic_metadata():
    content = "\n".join(
        [
            "[3MATCH]",
            "03/01/2016;20.30.00;Season;Competition;Amichevole",
            "[3TEAMS]",
            "1001;Team A",
            "1002;Team B",
            "[3SCOUT]",
            "*z1",
            "az2",
            "*06S",
            "*p1:0",
        ]
    )

    df = dvw_rallies_to_df(content)
    assert not df.empty
    row = df.iloc[0]
    assert row["match_date"] == "2016-01-03"
    assert row["match_type"] == "Amichevole"
    assert row["team_id_h"] == "1001"
    assert row["team_id_a"] == "1002"
    assert row["team_h"] == "Team A"
    assert row["team_a"] == "Team B"
