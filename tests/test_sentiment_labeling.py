import pandas as pd

from src.sentiment_labeling import (
    create_annotation_sample,
    preserve_completed_annotation_sample,
    rating_class,
    text_length_bucket,
)


def test_rating_class_and_text_bucket():
    assert rating_class(5.0) == "rating_5"
    assert rating_class(None) == "rating_missing"
    assert text_length_bucket(0) == "empty"
    assert text_length_bucket(20) == "very_short"
    assert text_length_bucket(80) == "short"
    assert text_length_bucket(200) == "medium"
    assert text_length_bucket(400) == "long"


def test_create_annotation_sample_columns_and_blank_manual_fields():
    rows = []
    for idx in range(30):
        rating = [1.0, 3.0, 5.0][idx % 3]
        label = ["negative", "neutral", "positive"][idx % 3]
        rows.append(
            {
                "review_id": f"r{idx}",
                "canonical_place_id": f"p{idx % 10}",
                "place_name": f"Place {idx % 10}",
                "place_category": ["wisata", "hotel"][idx % 2],
                "reviewer_rating": rating,
                "review_text_raw": f"review text {idx}",
                "weak_sentiment_label": label,
                "text_length": 20 + idx,
                "is_duplicate": False,
            }
        )
    sample, report = create_annotation_sample(pd.DataFrame(rows), sample_size=12, random_seed=42)
    assert len(sample) == 12
    assert set(sample.columns) == {
        "review_id",
        "place_name",
        "reviewer_rating",
        "review_text_raw",
        "weak_sentiment_label",
        "manual_sentiment_label",
        "annotation_notes",
    }
    assert sample["manual_sentiment_label"].eq("").all()
    assert report["sample_size"] == 12


def test_existing_manual_sample_is_not_overwritten(tmp_path):
    sample = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "place_name": "Place",
                "reviewer_rating": 1.0,
                "review_text_raw": "buruk",
                "weak_sentiment_label": "negative",
                "manual_sentiment_label": "",
                "annotation_notes": "",
            }
        ]
    )
    existing = sample.copy()
    existing["manual_sentiment_label"] = "negative"
    existing["annotation_notes"] = "checked"
    path = tmp_path / "annotations.csv"
    existing.to_csv(path, index=False, encoding="utf-8-sig")
    merged = preserve_completed_annotation_sample(sample, path)
    assert merged.loc[0, "manual_sentiment_label"] == "negative"
    assert merged.loc[0, "annotation_notes"] == "checked"
