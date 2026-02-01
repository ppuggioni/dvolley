import pytest

from dvolley.data.normalization import normalize_date_str


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2016-01-03", "2016-01-03"),
        ("03/01/2016", "2016-01-03"),
        ("2016/01/03", "2016-01-03"),
        ("", None),
        (None, None),
        ("03-01-2016", None),
        ("2016.01.03", None),
    ],
)
def test_normalize_date_str(raw, expected):
    assert normalize_date_str(raw) == expected
