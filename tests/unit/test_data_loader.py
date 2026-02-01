from unittest.mock import MagicMock

import pandas as pd

import dvolley.services.data_loader as loader


def test_load_data_from_db_normalizes_dates(monkeypatch):
    df = pd.DataFrame(
        {
            "match_date": ["03/01/2016"],
            "file_id": ["f1"],
            "rally_idx": [0],
            "team_h": ["A"],
            "team_a": ["B"],
            "team_id_h": ["1"],
            "team_id_a": ["2"],
        }
    )
    monkeypatch.setattr(loader.db, "fetch_all_rallies", lambda: df)

    out = loader.load_data_from_db()
    assert out.loc[0, "match_date"] == "2016-01-03"


def test_update_database_no_new_files(monkeypatch):
    monkeypatch.setattr(loader.db, "get_existing_file_ids", lambda: {"1"})
    monkeypatch.setattr(loader, "list_files_in_folder", lambda _fid: [{"id": "1", "name": "a.dvw"}])

    result = loader.update_database(["folder"])
    assert result == []
