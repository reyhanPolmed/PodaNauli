"""Small preprocessing helpers shared by exploration and later pipeline stages."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any

import pandas as pd


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD_FOR_COLUMNS = re.compile(r"[^0-9a-zA-Z]+")
_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_PRICE_NUMBER = re.compile(r"\d[\d\.,]*")
_PLACEHOLDERS = {"", "-", "--", "nan", "NaN", "None", "none", "null", "NULL", "n/a", "N/A"}


def normalize_column_name(name: Any) -> str:
    """Normalize a workbook column name to snake_case without assuming language."""
    text = "" if name is None else str(name)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = _NON_WORD_FOR_COLUMNS.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def make_unique_column_names(columns: list[Any]) -> list[str]:
    """Normalize columns to snake_case and make duplicate names explicit."""
    counts: dict[str, int] = {}
    normalized_columns = []
    for column in columns:
        normalized = normalize_column_name(column)
        counts[normalized] = counts.get(normalized, 0) + 1
        if counts[normalized] == 1:
            normalized_columns.append(normalized)
        else:
            normalized_columns.append(f"{normalized}_{counts[normalized]}")
    return normalized_columns


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized snake_case columns."""
    result = df.copy()
    result.columns = make_unique_column_names(list(result.columns))
    return result


def is_missing_like(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in _PLACEHOLDERS:
        return True
    return False


def clean_text_basic(value: Any) -> str | None:
    """Clean review text lightly while preserving Indonesian negation words."""
    if is_missing_like(value):
        return None
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if text in _PLACEHOLDERS:
        return None
    return text


def normalize_url_spacing(value: Any) -> str | None:
    """Normalize URLs and repeated spaces without removing text content."""
    text = clean_text_basic(value)
    if text is None:
        return None
    text = _URL.sub(lambda match: match.group(0).strip(), text)
    return _WHITESPACE.sub(" ", text).strip()


def parse_rating(value: Any) -> float | None:
    """Parse ratings like 4, 4.5, or 4,5 into a float."""
    if is_missing_like(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rating = float(value)
        return rating if 1.0 <= rating <= 5.0 else None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+(?:[\.,]\d+)?", text)
    if not match:
        return None
    rating = float(match.group(0).replace(",", "."))
    return rating if 1.0 <= rating <= 5.0 else None


def text_length(value: Any) -> int:
    cleaned = clean_text_basic(value)
    return len(cleaned) if cleaned else 0


def normalize_place_name(value: Any) -> str | None:
    """Conservative place-name normalization for profiling comparisons."""
    cleaned = clean_text_basic(value)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    lowered = re.sub(r"\bbul[\s\-]+bul\b", "bul bul", lowered)
    lowered = re.sub(r"\blumban\s+bul\s+bul\b", "lumban bulbul", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    lowered = _WHITESPACE.sub(" ", lowered).strip()
    return lowered or None


def parse_coordinate(value: Any) -> dict[str, Any]:
    """Parse coordinate text into latitude/longitude with range validation."""
    original = None if pd.isna(value) else str(value)
    if is_missing_like(value):
        return {
            "latitude": None,
            "longitude": None,
            "parsing_success": False,
            "is_missing": True,
            "is_inferred": False,
            "original": original,
            "error": "missing",
        }
    text = str(value).strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(matches) < 2:
        return {
            "latitude": None,
            "longitude": None,
            "parsing_success": False,
            "is_missing": False,
            "is_inferred": False,
            "original": original,
            "error": "could_not_find_two_numbers",
        }
    lat = float(matches[0])
    lon = float(matches[1])
    valid = -90 <= lat <= 90 and -180 <= lon <= 180
    return {
        "latitude": lat if valid else None,
        "longitude": lon if valid else None,
        "parsing_success": valid,
        "is_missing": False,
        "is_inferred": False,
        "original": original,
        "error": None if valid else "coordinate_out_of_range",
    }


def _parse_price_number(token: str) -> float | None:
    cleaned = token.strip().replace("Rp", "").replace("rp", "").replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "." in cleaned:
        parts = cleaned.split(".")
        if all(len(part) == 3 for part in parts[1:]):
            cleaned = "".join(parts)
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_price(value: Any) -> dict[str, Any]:
    """Parse Indonesian price strings while preserving the original text."""
    original = None if pd.isna(value) else str(value)
    if is_missing_like(value):
        return {
            "min_price": None,
            "max_price": None,
            "is_free": False,
            "price_text_original": original,
            "parsing_success": False,
            "is_missing": True,
            "is_inferred": False,
        }
    text = str(value).strip()
    lowered = text.lower()
    if "gratis" in lowered or "free" in lowered:
        return {
            "min_price": 0.0,
            "max_price": 0.0,
            "is_free": True,
            "price_text_original": original,
            "parsing_success": True,
            "is_missing": False,
            "is_inferred": False,
        }
    prices = [_parse_price_number(match.group(0)) for match in _PRICE_NUMBER.finditer(text)]
    prices = [price for price in prices if price is not None]
    if not prices:
        return {
            "min_price": None,
            "max_price": None,
            "is_free": False,
            "price_text_original": original,
            "parsing_success": False,
            "is_missing": False,
            "is_inferred": False,
        }
    return {
        "min_price": min(prices),
        "max_price": max(prices),
        "is_free": False,
        "price_text_original": original,
        "parsing_success": True,
        "is_missing": False,
        "is_inferred": len(prices) == 1,
    }


def parse_review_date(published_at: Any, collect_date: Any = None) -> dict[str, Any]:
    """Parse absolute, Excel, or relative review dates without inventing values."""
    collect_ts = pd.to_datetime(collect_date, errors="coerce")
    published_clean = clean_text_basic(published_at)
    if published_clean is None:
        if pd.notna(collect_ts):
            return {
                "review_date": collect_ts.normalize(),
                "parsing_success": True,
                "is_missing": False,
                "is_inferred": True,
                "source": "collect_date_fallback",
            }
        return {
            "review_date": pd.NaT,
            "parsing_success": False,
            "is_missing": True,
            "is_inferred": False,
            "source": None,
        }

    absolute = pd.to_datetime(published_clean, errors="coerce")
    if pd.notna(absolute):
        return {
            "review_date": absolute.normalize(),
            "parsing_success": True,
            "is_missing": False,
            "is_inferred": False,
            "source": "published_at_absolute",
        }

    if pd.notna(collect_ts):
        text = published_clean.lower()
        number_words = {"a": 1, "an": 1, "one": 1}
        number_match = re.search(r"(\d+|a|an|one)\s+(day|week|month|year|hour|minute)s?\s+ago", text)
        if number_match:
            raw_number = number_match.group(1)
            amount = number_words.get(raw_number, int(raw_number) if raw_number.isdigit() else 1)
            unit = number_match.group(2)
            if unit == "day":
                delta = timedelta(days=amount)
            elif unit == "week":
                delta = timedelta(weeks=amount)
            elif unit == "month":
                delta = timedelta(days=30 * amount)
            elif unit == "year":
                delta = timedelta(days=365 * amount)
            elif unit == "hour":
                delta = timedelta(hours=amount)
            else:
                delta = timedelta(minutes=amount)
            return {
                "review_date": (collect_ts - delta).normalize(),
                "parsing_success": True,
                "is_missing": False,
                "is_inferred": True,
                "source": "published_at_relative_to_collect_date",
            }
        if "yesterday" in text:
            return {
                "review_date": (collect_ts - timedelta(days=1)).normalize(),
                "parsing_success": True,
                "is_missing": False,
                "is_inferred": True,
                "source": "published_at_relative_to_collect_date",
            }
    return {
        "review_date": pd.NaT,
        "parsing_success": False,
        "is_missing": False,
        "is_inferred": False,
        "source": "unparsed",
    }


def weak_sentiment_from_rating(rating: float | None) -> str | None:
    """Create a weak sentiment label from rating; this is not ground truth."""
    if rating is None or pd.isna(rating):
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    if rating >= 4:
        return "positive"
    return None
