from dvolley.data.full_parser import sanitize_dv_content


def test_sanitize_dv_content_comments_short_codes():
    content = "\n".join(
        [
            "[3SCOUT]",
            "ABCDE;foo",
            "ABCDEF;bar",
        ]
    )

    sanitized = sanitize_dv_content(content)
    lines = sanitized.splitlines()
    assert lines[1].startswith("*")  # short code (len=5) is commented out
    assert not lines[2].startswith("*")
