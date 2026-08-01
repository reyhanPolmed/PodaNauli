import pandas as pd

from src.preprocessing import (
    clean_text_basic,
    parse_coordinate,
    parse_price,
    parse_rating,
    standardize_columns,
)


def test_standardize_columns_to_snake_case():
    df = pd.DataFrame(columns=["Place Name", "review-text", "RATING BPODT"])
    assert list(standardize_columns(df).columns) == ["place_name", "review_text", "rating_bpodt"]


def test_parse_rating_comma_and_dot():
    assert parse_rating("4,5") == 4.5
    assert parse_rating("4.2") == 4.2
    assert parse_rating("2025") is None


def test_clean_text_preserves_negation_words():
    text = clean_text_basic("  toilet tidak bersih dan parkir kurang luas  ")
    assert text == "toilet tidak bersih dan parkir kurang luas"
    assert "tidak" in text
    assert "kurang" in text


def test_parse_coordinate_validates_range():
    parsed = parse_coordinate("2.3492596002020694, 99.07327785959252")
    assert parsed["parsing_success"] is True
    assert parsed["latitude"] == 2.3492596002020694
    assert parsed["longitude"] == 99.07327785959252
    assert parse_coordinate("200, 99")["parsing_success"] is False


def test_parse_price_range_and_free():
    paid = parse_price("5.000 - 10.000")
    assert paid["min_price"] == 5000.0
    assert paid["max_price"] == 10000.0
    assert paid["is_free"] is False

    free = parse_price("Gratis")
    assert free["min_price"] == 0.0
    assert free["max_price"] == 0.0
    assert free["is_free"] is True
