"""Compare fair weak-label baselines with gold-trained models on one gold test set."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.evaluate import classification_metrics
from src.gold_dataset import SPLIT_PATH, apply_gold_split
from src.train_complaint import (
    binary_evaluation,
    build_complaint_model,
    choose_negative_threshold,
    predict_complaint_probabilities,
    prepare_complaint_dataset,
    split_train_validation_test as split_complaint_data,
)
from src.train_sentiment import (
    LABEL_ORDER,
    build_candidate_models,
    prepare_sentiment_training_data,
    prepare_trainable_reviews,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
JSON_PATH = REPORT_DIR / "gold_vs_weak_model_comparison.json"
CSV_PATH = REPORT_DIR / "gold_vs_weak_model_comparison.csv"
ARCHIVE_REPORT_DIR = REPORT_DIR / "archive"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare_gold_and_weak_models() -> dict[str, Any]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing locked gold split: {SPLIT_PATH}")
    config = load_config()
    random_seed = int(config.get("random_seed", 42))
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    gold_sentiment, label_source = prepare_sentiment_training_data(reviews, config)
    if label_source != "human_gold":
        raise ValueError("Gold comparison requires a ready human gold dataset.")
    _gold_train, _gold_validation, gold_test, split_report = apply_gold_split(gold_sentiment)
    test_groups = set(gold_test["canonical_place_id"])

    weak_data = prepare_trainable_reviews(reviews)
    weak_train = weak_data[~weak_data["canonical_place_id"].isin(test_groups)].copy()
    weak_sentiment_model = build_candidate_models(random_seed)["sentiment_logistic"]
    weak_sentiment_model.fit(
        weak_train["review_text_clean"],
        weak_train["weak_sentiment_label"],
    )
    weak_sentiment_predictions = weak_sentiment_model.predict(gold_test["review_text_clean"])
    weak_sentiment_metrics = classification_metrics(
        gold_test["weak_sentiment_label"].to_numpy(),
        weak_sentiment_predictions,
        LABEL_ORDER,
    )

    gold_sentiment_model = joblib.load(MODEL_DIR / "sentiment_champion.joblib")
    gold_sentiment_predictions = gold_sentiment_model.predict(gold_test["review_text_clean"])
    gold_sentiment_metrics = classification_metrics(
        gold_test["weak_sentiment_label"].to_numpy(),
        gold_sentiment_predictions,
        LABEL_ORDER,
    )

    complaint_data, complaint_source = prepare_complaint_dataset(reviews, config)
    if complaint_source != "human_gold":
        raise ValueError("Gold complaint comparison requires human gold labels.")
    _complaint_train, _complaint_validation, complaint_test, _complaint_split = apply_gold_split(
        complaint_data
    )
    weak_complaint_data = weak_data[~weak_data["canonical_place_id"].isin(test_groups)].copy()
    weak_complaint_data["sentiment_label"] = weak_complaint_data["weak_sentiment_label"]
    weak_complaint_data["complaint_target"] = (
        weak_complaint_data["weak_sentiment_label"] == "negative"
    ).astype(int)
    weak_train_part, weak_validation_part, _weak_unused_test, _weak_split_report = (
        split_complaint_data(weak_complaint_data, random_seed)
    )
    archived_metrics = _load_json(
        ARCHIVE_REPORT_DIR / "complaint_metrics_v2_weak.json"
    )
    archived_parameters = archived_metrics.get("selection", {}).get("parameters", {})
    feature_set = str(archived_parameters.get("feature_set", "word_char"))
    c_value = float(archived_parameters.get("c_value", 1.0))
    class_weight = float(archived_parameters.get("negative_class_weight", 2.0))
    weak_complaint_model = build_complaint_model(
        feature_set,
        c_value,
        class_weight,
        random_seed,
    )
    weak_complaint_model.fit(
        weak_train_part["review_text_clean"],
        weak_train_part["complaint_target"],
    )
    validation_probabilities = predict_complaint_probabilities(
        weak_complaint_model,
        weak_validation_part["review_text_clean"],
    )
    complaint_config = config.get("sentiment_improvement", {}).get("complaint_model", {})
    weak_threshold, _weak_threshold_metrics = choose_negative_threshold(
        weak_validation_part["complaint_target"].to_numpy(),
        validation_probabilities,
        minimum_precision=float(complaint_config.get("threshold_min_precision", 0.60)),
        beta=float(complaint_config.get("threshold_beta", 2.0)),
    )
    weak_complaint_model.fit(
        weak_complaint_data["review_text_clean"],
        weak_complaint_data["complaint_target"],
    )
    weak_complaint_probabilities = predict_complaint_probabilities(
        weak_complaint_model,
        complaint_test["review_text_clean"],
    )
    weak_complaint_metrics = binary_evaluation(
        complaint_test["complaint_target"].to_numpy(),
        weak_complaint_probabilities,
        weak_threshold,
    )

    gold_complaint_bundle = joblib.load(MODEL_DIR / "complaint_detector.joblib")
    gold_complaint_probabilities = predict_complaint_probabilities(
        gold_complaint_bundle["model"],
        complaint_test["review_text_clean"],
    )
    gold_complaint_metrics = binary_evaluation(
        complaint_test["complaint_target"].to_numpy(),
        gold_complaint_probabilities,
        float(gold_complaint_bundle["negative_threshold"]),
    )

    comparison_rows = [
        {
            "task": "sentiment_3class",
            "model": "weak_label_retrained_baseline",
            "label_source": "weak_rating",
            "macro_f1": weak_sentiment_metrics["macro_f1"],
            "balanced_accuracy": weak_sentiment_metrics["balanced_accuracy"],
            "negative_recall": weak_sentiment_metrics["negative_recall"],
            "negative_precision": weak_sentiment_metrics["classification_report"]["negative"]["precision"],
            "neutral_f1": weak_sentiment_metrics["classification_report"]["neutral"]["f1-score"],
        },
        {
            "task": "sentiment_3class",
            "model": "gold_trained_champion",
            "label_source": "human_gold",
            "macro_f1": gold_sentiment_metrics["macro_f1"],
            "balanced_accuracy": gold_sentiment_metrics["balanced_accuracy"],
            "negative_recall": gold_sentiment_metrics["negative_recall"],
            "negative_precision": gold_sentiment_metrics["classification_report"]["negative"]["precision"],
            "neutral_f1": gold_sentiment_metrics["classification_report"]["neutral"]["f1-score"],
        },
        {
            "task": "complaint_binary",
            "model": "weak_label_retrained_baseline",
            "label_source": "weak_rating",
            "macro_f1": weak_complaint_metrics["macro_f1"],
            "balanced_accuracy": weak_complaint_metrics["balanced_accuracy"],
            "negative_recall": weak_complaint_metrics["negative_recall"],
            "negative_precision": weak_complaint_metrics["negative_precision"],
            "neutral_f1": None,
        },
        {
            "task": "complaint_binary",
            "model": "gold_trained_champion",
            "label_source": "human_gold",
            "macro_f1": gold_complaint_metrics["macro_f1"],
            "balanced_accuracy": gold_complaint_metrics["balanced_accuracy"],
            "negative_recall": gold_complaint_metrics["negative_recall"],
            "negative_precision": gold_complaint_metrics["negative_precision"],
            "neutral_f1": None,
        },
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(CSV_PATH, index=False, encoding="utf-8")
    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "evaluation_dataset": "locked human gold test split",
        "test_rows": int(len(gold_test)),
        "test_groups": int(gold_test["canonical_place_id"].nunique()),
        "split": split_report,
        "sentiment": {
            "weak_label_retrained_baseline": weak_sentiment_metrics,
            "gold_trained_champion": gold_sentiment_metrics,
        },
        "complaint": {
            "weak_label_retrained_baseline": {
                **weak_complaint_metrics,
                "threshold": weak_threshold,
            },
            "gold_trained_champion": gold_complaint_metrics,
        },
        "outputs": {
            "json": str(JSON_PATH),
            "csv": str(CSV_PATH),
        },
        "limitations": [
            "The gold dataset has one annotator and no inter-annotator agreement estimate.",
            "Annotators could see AI suggestions, which may introduce confirmation bias.",
            "The weak baselines exclude every place present in the locked gold test split.",
        ],
    }
    JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = compare_gold_and_weak_models()
    print(
        json.dumps(
            {
                "test_rows": result["test_rows"],
                "test_groups": result["test_groups"],
                "comparison": result["outputs"]["csv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
