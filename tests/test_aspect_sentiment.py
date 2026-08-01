import pandas as pd

from src.aspect_sentiment import split_clauses
from src.gap_scoring import build_review_aspect_rows


def test_split_clauses_separates_contrast():
    clauses = split_clauses("Pemandangannya bagus tetapi toilet kotor dan parkir sempit.")
    assert clauses == ["Pemandangannya bagus", "toilet kotor dan parkir sempit"]


def test_gap_scoring_prefers_clause_level_aspect_sentiment_schema():
    clause_predictions = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "canonical_place_id": "p1",
                "aspect": "pemandangan",
                "is_negative": False,
                "sentiment_label": "non_negative",
            },
            {
                "review_id": "r1",
                "canonical_place_id": "p1",
                "aspect": "toilet",
                "is_negative": True,
                "sentiment_label": "negative",
            },
        ]
    )
    mentions = build_review_aspect_rows(clause_predictions, ["pemandangan", "toilet"])
    assert not mentions.loc[mentions["aspect"] == "pemandangan", "is_negative"].iloc[0]
    assert mentions.loc[mentions["aspect"] == "toilet", "is_negative"].iloc[0]
