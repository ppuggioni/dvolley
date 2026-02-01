from unittest.mock import MagicMock

import pandas as pd

import dvolley.services.db as db


def test_fetch_all_rallies_paginates(monkeypatch):
    calls = []

    def fake_select(_):
        return fake_table

    def fake_range(start, end):
        calls.append((start, end))
        return fake_table

    def fake_execute():
        if len(calls) == 1:
            return MagicMock(data=[{"file_id": "a"}])
        return MagicMock(data=[])

    fake_table = MagicMock()
    fake_table.select.side_effect = fake_select
    fake_table.range.side_effect = fake_range
    fake_table.execute.side_effect = fake_execute

    monkeypatch.setattr(db, "supabase", MagicMock(table=lambda _: fake_table))

    df = db.fetch_all_rallies()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert calls == [(0, 999)]


def test_get_existing_file_ids_handles_empty(monkeypatch):
    fake_table = MagicMock()
    fake_table.select.return_value = fake_table
    fake_table.execute.return_value = MagicMock(data=[])

    monkeypatch.setattr(db, "supabase", MagicMock(table=lambda _: fake_table))

    ids = db.get_existing_file_ids()
    assert ids == set()
