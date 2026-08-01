import numpy as np
import pandas as pd

from src.sentiment_annotation import (
    assert_gold_not_regressed,
    create_balanced_annotation_queue,
    derive_ai_suggestion,
    generate_ai_suggestions,
    preserve_existing_manual_values,
    reuse_or_create_annotation_queue,
    validate_annotation_frame,
)


def _reviews() -> pd.DataFrame:
    rows = []
    for label_index, label in enumerate(["negative", "neutral", "positive"]):
        for index in range(20):
            rows.append(
                {
                    "review_id": f"{label}_{index}",
                    "canonical_place_id": f"p_{label_index}_{index % 8}",
                    "place_name": f"Place {index % 8}",
                    "place_category": "wisata",
                    "reviewer_rating": [1.0, 3.0, 5.0][label_index],
                    "review_text_raw": f"{label} review {index}",
                    "weak_sentiment_label": label,
                    "text_length": 20 + index,
                    "is_duplicate": False,
                }
            )
    return pd.DataFrame(rows)


def test_balanced_annotation_queue_and_manual_preservation():
    queue, report = create_balanced_annotation_queue(
        _reviews(),
        target_per_class=8,
        max_per_place=3,
        random_seed=42,
        error_review_ids={"negative_1", "neutral_1"},
    )
    assert report["class_counts"] == {"negative": 8, "neutral": 8, "positive": 8}
    existing = queue.head(1)[["review_id"]].copy()
    existing["manual_sentiment_label"] = "negative"
    for column in [
        "second_annotator_label",
        "annotation_status",
        "annotator_id",
        "second_annotator_id",
        "annotation_confidence",
        "annotation_notes",
    ]:
        existing[column] = ""
    preserved = preserve_existing_manual_values(queue, existing)
    assert preserved.loc[preserved["review_id"] == existing.iloc[0]["review_id"], "manual_sentiment_label"].iloc[0] == "negative"


def test_existing_annotation_queue_is_not_resampled():
    existing, _report = create_balanced_annotation_queue(
        _reviews(),
        target_per_class=2,
        max_per_place=2,
        random_seed=42,
    )
    existing["manual_sentiment_label"] = "negative"
    existing["human_approved"] = "yes"
    reused, report = reuse_or_create_annotation_queue(
        _reviews(),
        existing_queue=existing,
        target_per_class=3,
        max_per_place=1,
        random_seed=7,
        error_review_ids={"positive_19"},
    )
    assert reused["review_id"].tolist() == existing["review_id"].tolist()
    assert reused["manual_sentiment_label"].tolist() == existing["manual_sentiment_label"].tolist()
    assert report["reused_existing_queue"] is True


def test_automatic_preparation_rejects_gold_row_loss():
    previous = pd.DataFrame({"review_id": ["r1", "r2"]})
    reduced = pd.DataFrame({"review_id": ["r1"]})
    try:
        assert_gold_not_regressed(previous, reduced)
    except ValueError as error:
        assert "remove 1 existing gold rows" in str(error)
    else:
        raise AssertionError("Expected gold-regression protection to raise ValueError.")


def test_annotation_validation_excludes_conflicts_and_invalid_labels():
    queue, _report = create_balanced_annotation_queue(
        _reviews(),
        target_per_class=2,
        max_per_place=2,
        random_seed=42,
    )
    queue.loc[0, "manual_sentiment_label"] = "negative"
    queue.loc[0, "human_approved"] = "yes"
    queue.loc[1, "manual_sentiment_label"] = "positive"
    queue.loc[1, "second_annotator_label"] = "negative"
    queue.loc[1, "human_approved"] = "yes"
    queue.loc[2, "manual_sentiment_label"] = "unknown"
    queue.loc[2, "human_approved"] = "yes"
    gold, report = validate_annotation_frame(queue)
    assert len(gold) == 1
    assert report["adjudication_required_rows"] == 1
    assert report["invalid_rows"] == 1


def test_ai_suggestion_is_silver_and_exposes_disagreement():
    suggestion = derive_ai_suggestion(
        sentiment_prediction="positive",
        sentiment_probability=0.80,
        complaint_decision="negative",
        complaint_probability=0.70,
        weak_label="positive",
        high_probability=0.75,
        medium_probability=0.55,
    )
    assert suggestion["ai_suggested_sentiment_label"] == "negative"
    assert suggestion["ai_model_agreement"] == "no"
    assert suggestion["ai_weak_label_agreement"] == "no"
    assert suggestion["ai_confidence"] == "low"


def test_manual_label_without_human_approval_does_not_enter_gold():
    queue, _report = create_balanced_annotation_queue(
        _reviews(),
        target_per_class=1,
        max_per_place=1,
        random_seed=42,
    )
    queue.loc[0, "manual_sentiment_label"] = "negative"
    gold, report = validate_annotation_frame(queue)
    assert gold.empty
    assert report["awaiting_human_approval_rows"] == 1


def test_ai_suggestions_preserve_approved_rows_with_string_probability_columns(monkeypatch):
    class FakeSentimentModel:
        classes_ = np.array(["negative", "neutral", "positive"])

        def predict_proba(self, texts):
            return np.tile([0.8, 0.1, 0.1], (len(texts), 1))

    queue, _report = create_balanced_annotation_queue(
        _reviews(),
        target_per_class=1,
        max_per_place=1,
        random_seed=42,
    )
    queue["human_approved"] = ["yes", "", ""]
    queue["ai_sentiment_probability"] = pd.Series(["0.91", "", ""], dtype="str")
    queue["ai_complaint_probability"] = pd.Series(["0.08", "", ""], dtype="str")
    queue.loc[0, "ai_suggested_sentiment_label"] = "positive"

    def fake_complaint_predictions(_bundle, texts):
        return pd.DataFrame(
            {
                "complaint_decision": ["negative"] * len(texts),
                "complaint_probability": [0.9] * len(texts),
            }
        )

    monkeypatch.setattr(
        "src.sentiment_annotation.predict_with_abstention",
        fake_complaint_predictions,
    )
    result = generate_ai_suggestions(
        queue=queue,
        texts=queue["review_text_raw"],
        sentiment_model=FakeSentimentModel(),
        complaint_bundle={"version": "test"},
        config={"sentiment_improvement": {"ai_annotation": {}}},
    )

    assert result.loc[0, "ai_suggested_sentiment_label"] == "positive"
    assert result.loc[0, "ai_sentiment_probability"] == "0.91"
    assert result.loc[1, "ai_sentiment_probability"] == 0.8
    assert result.loc[1, "ai_complaint_probability"] == 0.9
