from src.preprocessing import normalize_text

def test_normalize_basic():
    s = "Check THIS out! https://example.com #Launch @brand"
    out = normalize_text(s)
    assert out == 'check this out launch'
