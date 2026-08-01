import pandas as pd
import joblib
from pathlib import Path

from src.train_sentiment import (
    LABEL_ORDER,
    prepare_trainable_reviews,
    split_train_test,
    split_train_validation_test,
)


def test_prepare_trainable_reviews_filters_duplicates_and_empty_text():
    reviews = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "review_text_clean": "bagus",
                "weak_sentiment_label": "positive",
                "canonical_place_id": "p1",
                "is_duplicate": False,
                "text_length": 5,
            },
            {
                "review_id": "r2",
                "review_text_clean": "buruk",
                "weak_sentiment_label": "negative",
                "canonical_place_id": "p2",
                "is_duplicate": True,
                "text_length": 5,
            },
            {
                "review_id": "r3",
                "review_text_clean": "",
                "weak_sentiment_label": "neutral",
                "canonical_place_id": "p3",
                "is_duplicate": False,
                "text_length": 0,
            },
        ]
    )
    trainable = prepare_trainable_reviews(reviews)
    assert trainable["review_id"].tolist() == ["r1"]


def test_split_train_test_has_no_group_overlap():
    rows = []
    for label_index, label in enumerate(LABEL_ORDER):
        for group_number in range(8):
            for item in range(2):
                rows.append(
                    {
                        "review_id": f"{label}_{group_number}_{item}",
                        "review_text_clean": f"{label} text {group_number} {item}",
                        "weak_sentiment_label": label,
                        "canonical_place_id": f"{label}_place_{group_number}",
                        "is_duplicate": False,
                        "text_length": 20,
                    }
                )
    data = prepare_trainable_reviews(pd.DataFrame(rows))
    train_idx, test_idx, report = split_train_test(data, random_seed=42)
    train_groups = set(data.iloc[train_idx]["canonical_place_id"])
    test_groups = set(data.iloc[test_idx]["canonical_place_id"])
    assert not (train_groups & test_groups)
    assert report["group_overlap_count"] == 0


def test_three_way_sentiment_split_has_no_group_overlap():
    rows = []
    for label in LABEL_ORDER:
        for group_number in range(10):
            for item in range(2):
                rows.append(
                    {
                        "review_id": f"{label}_{group_number}_{item}",
                        "review_text_clean": f"{label} text {group_number} {item}",
                        "weak_sentiment_label": label,
                        "canonical_place_id": f"{label}_place_{group_number}",
                        "is_duplicate": False,
                        "text_length": 20,
                    }
                )
    data = prepare_trainable_reviews(pd.DataFrame(rows))
    train, validation, test, report = split_train_validation_test(data, random_seed=42)
    assert report["group_overlap_count"] == 0
    assert not set(train["canonical_place_id"]) & set(validation["canonical_place_id"])
    assert not set(train["canonical_place_id"]) & set(test["canonical_place_id"])


def test_sentiment_artifacts_can_be_loaded_after_training():
    champion_path = Path("models/sentiment_champion.joblib")
    encoder_path = Path("models/sentiment_label_encoder.joblib")
    assert champion_path.exists()
    assert encoder_path.exists()
    model = joblib.load(champion_path)
    encoder = joblib.load(encoder_path)
    assert set(encoder.classes_) == set(LABEL_ORDER)
    prediction = model.predict(["Pemandangannya bagus tetapi toilet kotor dan parkir sempit."])
    assert prediction[0] in LABEL_ORDER
