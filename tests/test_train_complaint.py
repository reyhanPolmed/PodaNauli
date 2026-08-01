import numpy as np
import pandas as pd

from src.train_complaint import (
    build_complaint_model,
    choose_negative_threshold,
    predict_with_abstention,
    split_train_validation_test,
)


def test_threshold_selection_respects_precision_floor_when_possible():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.45, 0.40, 0.70, 0.90])
    threshold, metrics = choose_negative_threshold(
        y_true,
        probabilities,
        minimum_precision=0.75,
        beta=2.0,
    )
    assert 0.05 <= threshold <= 0.95
    assert metrics["precision"] >= 0.75


def test_grouped_complaint_split_has_no_place_overlap():
    rows = []
    for target in [0, 1]:
        for group in range(12):
            for item in range(2):
                rows.append(
                    {
                        "review_text_clean": f"target {target} group {group} item {item}",
                        "canonical_place_id": f"target_{target}_place_{group}",
                        "complaint_target": target,
                    }
                )
    train, validation, test, report = split_train_validation_test(pd.DataFrame(rows), random_seed=42)
    assert report["group_overlap_count"] == 0
    assert not set(train["canonical_place_id"]) & set(validation["canonical_place_id"])
    assert not set(train["canonical_place_id"]) & set(test["canonical_place_id"])


def test_complaint_bundle_can_abstain():
    texts = pd.Series(["tempat sangat bagus", "pelayanan sangat buruk", "biasa saja"])
    targets = np.array([0, 1, 0])
    model = build_complaint_model("word", c_value=1.0, negative_class_weight=2.0, random_seed=42)
    model.set_params(features__min_df=1)
    model.fit(texts, targets)
    bundle = {
        "model": model,
        "negative_threshold": 0.5,
        "uncertainty_margin": 0.49,
        "version": "test",
        "label_source": "test",
    }
    predictions = predict_with_abstention(bundle, texts)
    assert set(predictions["complaint_decision"]).issubset({"negative", "non_negative", "uncertain"})
    assert len(predictions) == len(texts)
