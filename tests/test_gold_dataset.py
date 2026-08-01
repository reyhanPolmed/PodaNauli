import pandas as pd

from src.gold_dataset import (
    apply_gold_split,
    create_grouped_gold_splits,
    semantic_gold_hash,
    validate_gold_frame,
)


def _gold_rows() -> pd.DataFrame:
    rows = []
    for label in ["negative", "neutral", "positive"]:
        for group in range(10):
            rows.append(
                {
                    "review_id": f"{label}_{group}",
                    "canonical_place_id": f"{label}_place_{group}",
                    "manual_sentiment_label": label,
                    "annotator_id": "A01",
                    "human_approved": "yes",
                }
            )
    return pd.DataFrame(rows)


def test_gold_validation_and_hash_are_deterministic():
    gold = _gold_rows()
    report = validate_gold_frame(gold)
    assert report["valid"]
    assert report["rows"] == 30
    assert semantic_gold_hash(gold) == semantic_gold_hash(gold.sample(frac=1.0, random_state=7))


def test_gold_split_has_no_group_overlap_and_can_be_applied(tmp_path):
    gold = _gold_rows()
    assignments = create_grouped_gold_splits(gold, random_seed=42)
    groups = {
        split: set(assignments.loc[assignments["split"] == split, "canonical_place_id"])
        for split in ["train", "validation", "test"]
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    split_path = tmp_path / "split.parquet"
    assignments.to_parquet(split_path, index=False)
    data = gold.assign(review_text_clean="text", weak_sentiment_label=gold["manual_sentiment_label"])
    train, validation, test, report = apply_gold_split(data, split_path)
    assert len(train) + len(validation) + len(test) == len(gold)
    assert report["group_overlap_count"] == 0
