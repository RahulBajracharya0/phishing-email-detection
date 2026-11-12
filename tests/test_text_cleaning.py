from phishing.text_cleaning import MAX_CHARS, clean_text


def test_lowercases_text():
    assert clean_text("HELLO World") == "hello world"


def test_replaces_links_with_url_token():
    out = clean_text("Click http://bit.ly/abc or www.example.com now")
    assert "http" not in out
    assert "bit ly" not in out
    assert out.count("URL") == 2


def test_replaces_email_addresses_with_email_token():
    out = clean_text("Contact support@bank.com today")
    assert "@" not in out
    assert "EMAIL" in out


def test_removes_html_tags():
    assert clean_text("<b>Urgent</b> <a>reply</a>") == "urgent reply"


def test_removes_numbers_and_punctuation():
    assert clean_text("Win $1,000,000!!! Now.") == "win now"


def test_collapses_whitespace():
    assert clean_text("a   b\n\n\tc") == "a b c"


def test_non_string_input_returns_empty_string():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
    assert clean_text(float("nan")) == ""


def test_empty_string_returns_empty_string():
    assert clean_text("") == ""


def test_very_long_input_is_truncated():
    out = clean_text("word " * 1_000_000)
    assert len(out) <= MAX_CHARS


def test_cleaning_is_repeatable():
    text = "Hello <b>World</b> http://x.co a@b.com"
    assert clean_text(text) == clean_text(text)
