"""Train a complaint detector optimized for negative-review recall."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from src.evaluate import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
)
from src.gold_dataset import SPLIT_PATH, apply_gold_split
from src.sentiment_labeling import text_length_bucket
from src.train_sentiment import prepare_trainable_reviews


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
ANNOTATION_DIR = ROOT / "data" / "annotations"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"
GOLD_PATH = ANNOTATION_DIR / "sentiment_gold.csv"

MODEL_PATH = MODEL_DIR / "complaint_detector.joblib"
METRICS_PATH = REPORT_DIR / "complaint_metrics.json"
COMPARISON_PATH = REPORT_DIR / "complaint_model_comparison.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "review_complaint_predictions.parquet"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"random_seed": 42}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def build_complaint_model(
    feature_set: str,
    c_value: float,
    negative_class_weight: float,
    random_seed: int,
) -> Pipeline:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        C=float(c_value),
        class_weight={0: 1.0, 1: float(negative_class_weight)},
        max_iter=1200,
        random_state=random_seed,
    )
    if feature_set == "word":
        return Pipeline([("features", word), ("classifier", classifier)])
    if feature_set == "word_char":
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True,
        )
        return Pipeline(
            [
                ("features", FeatureUnion([("word", word), ("char", char)])),
                ("classifier", classifier),
            ]
        )
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _split_grouped(
    data: pd.DataFrame,
    label_column: str,
    random_seed: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    label_counts = data[label_column].value_counts()
    group_count = data["canonical_place_id"].nunique()
    effective_splits = min(n_splits, group_count, int(label_counts.min()))
    if effective_splits < 2:
        raise ValueError("Not enough labels or place groups for a leakage-safe split.")
    splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_seed)
    return next(
        splitter.split(
            data["review_text_clean"],
            data[label_column],
            data["canonical_place_id"],
        )
    )


def split_train_validation_test(
    data: pd.DataFrame,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create place-disjoint train, validation, and final test partitions."""
    development_idx, test_idx = _split_grouped(data, "complaint_target", random_seed, n_splits=5)
    development = data.iloc[development_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)
    train_idx, validation_idx = _split_grouped(development, "complaint_target", random_seed + 1, n_splits=4)
    train = development.iloc[train_idx].reset_index(drop=True)
    validation = development.iloc[validation_idx].reset_index(drop=True)
    train_groups = set(train["canonical_place_id"])
    validation_groups = set(validation["canonical_place_id"])
    test_groups = set(test["canonical_place_id"])
    overlap = (train_groups & validation_groups) | (train_groups & test_groups) | (validation_groups & test_groups)
    if overlap:
        raise ValueError(f"Place leakage detected: {sorted(overlap)[:5]}")
    report = {
        "method": "nested StratifiedGroupKFold first folds",
        "random_seed": int(random_seed),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_groups": int(len(train_groups)),
        "validation_groups": int(len(validation_groups)),
        "test_groups": int(len(test_groups)),
        "group_overlap_count": 0,
        "train_positive_complaints": int(train["complaint_target"].sum()),
        "validation_positive_complaints": int(validation["complaint_target"].sum()),
        "test_positive_complaints": int(test["complaint_target"].sum()),
    }
    return train, validation, test, report


def threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    beta: float = 2.0,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f_beta": float(fbeta_score(y_true, predictions, beta=beta, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def choose_negative_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_precision: float,
    beta: float,
) -> tuple[float, dict[str, float]]:
    """Choose a validation-only threshold, prioritizing complaint F-beta."""
    rows = [
        threshold_metrics(y_true, probabilities, float(threshold), beta=beta)
        for threshold in np.linspace(0.05, 0.95, 181)
    ]
    eligible = [row for row in rows if row["precision"] >= minimum_precision]
    pool = eligible or rows
    best = max(pool, key=lambda row: (row["f_beta"], row["recall"], row["precision"], -row["threshold"]))
    return float(best["threshold"]), best


def binary_evaluation(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "negative_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "negative_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "negative_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": ["non_negative", "negative"],
        "support": {
            "non_negative": int((y_true == 0).sum()),
            "negative": int((y_true == 1).sum()),
        },
    }


def predict_complaint_probabilities(model: Pipeline, texts: pd.Series | list[str]) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    complaint_index = classes.index(1)
    return np.asarray(model.predict_proba(texts))[:, complaint_index]


def predict_with_abstention(bundle: dict[str, Any], texts: pd.Series | list[str]) -> pd.DataFrame:
    probabilities = predict_complaint_probabilities(bundle["model"], texts)
    threshold = float(bundle["negative_threshold"])
    margin = float(bundle["uncertainty_margin"])
    lower = max(0.0, threshold - margin)
    upper = min(1.0, threshold + margin)
    decisions = np.where(probabilities >= upper, "negative", np.where(probabilities <= lower, "non_negative", "uncertain"))
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    return pd.DataFrame(
        {
            "complaint_probability": probabilities.astype(float),
            "complaint_decision": decisions,
            "is_negative": decisions == "negative",
            "prediction_confidence": confidence.astype(float),
        }
    )


def _load_gold_reviews(reviews: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    annotation_config = config.get("sentiment_improvement", {}).get("annotation", {})
    minimum_rows = int(annotation_config.get("minimum_gold_rows", 300))
    minimum_per_class = int(annotation_config.get("minimum_gold_rows_per_class", 50))
    if not GOLD_PATH.exists():
        return pd.DataFrame(), False
    gold = pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "manual_sentiment_label" not in gold.columns:
        return pd.DataFrame(), False
    gold["manual_sentiment_label"] = gold["manual_sentiment_label"].str.strip().str.lower()
    gold = gold[gold["manual_sentiment_label"].isin({"negative", "neutral", "positive"})]
    counts = gold["manual_sentiment_label"].value_counts()
    ready = len(gold) >= minimum_rows and all(
        int(counts.get(label, 0)) >= minimum_per_class
        for label in ["negative", "neutral", "positive"]
    )
    if not ready:
        return pd.DataFrame(), False
    base = prepare_trainable_reviews(reviews).drop(columns=["weak_sentiment_label"], errors="ignore")
    merged = gold[["review_id", "manual_sentiment_label"]].merge(base, on="review_id", how="inner")
    merged["sentiment_label"] = merged["manual_sentiment_label"]
    return merged, True


def prepare_complaint_dataset(reviews: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    gold, gold_ready = _load_gold_reviews(reviews, config)
    if gold_ready:
        data = gold
        label_source = "human_gold"
    else:
        data = prepare_trainable_reviews(reviews).copy()
        data["sentiment_label"] = data["weak_sentiment_label"]
        label_source = "weak_rating"
    data["complaint_target"] = (data["sentiment_label"] == "negative").astype(int)
    if data["complaint_target"].nunique() != 2:
        raise ValueError("Complaint training requires both negative and non-negative examples.")
    return data.reset_index(drop=True), label_source


def language_bucket(text: Any) -> str:
    value = str(text).lower()
    english = any(token in value.split() for token in ["the", "and", "was", "good", "bad", "room", "food"])
    indonesian = any(token in value.split() for token in ["yang", "dan", "tidak", "bagus", "buruk", "kamar"])
    if english and indonesian:
        return "mixed_id_en"
    if english:
        return "english_signal"
    return "indonesian_or_local"


def evaluation_slices(test: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    frame = test.copy()
    frame["probability"] = probabilities
    frame["text_length_bucket"] = frame["text_length"].map(text_length_bucket)
    frame["language_bucket"] = frame["review_text_clean"].map(language_bucket)
    rows = []
    for slice_column in ["place_category", "text_length_bucket", "language_bucket"]:
        for value, group in frame.groupby(slice_column, dropna=False):
            if len(group) < 10 or group["complaint_target"].nunique() < 2:
                continue
            metrics = binary_evaluation(
                group["complaint_target"].to_numpy(),
                group["probability"].to_numpy(),
                threshold,
            )
            rows.append(
                {
                    "slice_column": slice_column,
                    "slice_value": str(value),
                    "rows": int(len(group)),
                    "negative_support": metrics["support"]["negative"],
                    "negative_precision": metrics["negative_precision"],
                    "negative_recall": metrics["negative_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )
    return rows


def train_complaint_detector() -> dict[str, Any]:
    config = load_config()
    random_seed = int(config.get("random_seed", 42))
    model_config = config.get("sentiment_improvement", {}).get("complaint_model", {})
    acceptance = config.get("sentiment_improvement", {}).get("acceptance_gates", {})
    minimum_precision = float(model_config.get("threshold_min_precision", 0.60))
    beta = float(model_config.get("threshold_beta", 2.0))
    uncertainty_margin = float(model_config.get("uncertainty_margin", 0.08))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    dataset, label_source = prepare_complaint_dataset(reviews, config)
    if label_source == "human_gold" and SPLIT_PATH.exists():
        train, validation, test, split_report = apply_gold_split(dataset)
        split_report["source"] = str(SPLIT_PATH)
    else:
        train, validation, test, split_report = split_train_validation_test(dataset, random_seed)
    x_train = train["review_text_clean"]
    y_train = train["complaint_target"].to_numpy()
    x_validation = validation["review_text_clean"]
    y_validation = validation["complaint_target"].to_numpy()

    comparison_rows = []
    best: dict[str, Any] | None = None
    for feature_set in model_config.get("feature_sets", ["word", "word_char"]):
        for c_value in model_config.get("c_values", [0.3, 1.0, 3.0]):
            for class_weight in model_config.get("negative_class_weights", [2.0, 3.0, 4.0]):
                model = build_complaint_model(feature_set, c_value, class_weight, random_seed)
                started = time.perf_counter()
                model.fit(x_train, y_train)
                training_seconds = time.perf_counter() - started
                probabilities = predict_complaint_probabilities(model, x_validation)
                threshold, validation_metrics = choose_negative_threshold(
                    y_validation,
                    probabilities,
                    minimum_precision=minimum_precision,
                    beta=beta,
                )
                row = {
                    "feature_set": feature_set,
                    "c_value": float(c_value),
                    "negative_class_weight": float(class_weight),
                    "threshold": threshold,
                    "validation_precision": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_f1": validation_metrics["f1"],
                    "validation_f_beta": validation_metrics["f_beta"],
                    "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                    "validation_average_precision": float(average_precision_score(y_validation, probabilities)),
                    "training_seconds": float(training_seconds),
                }
                comparison_rows.append(row)
                selection_key = (
                    row["validation_f_beta"],
                    row["validation_recall"],
                    row["validation_average_precision"],
                    row["validation_precision"],
                )
                if best is None or selection_key > best["selection_key"]:
                    best = {
                        "selection_key": selection_key,
                        "parameters": row,
                    }
    if best is None:
        raise RuntimeError("No complaint detector candidate was trained.")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_f_beta", "validation_recall", "validation_average_precision"],
        ascending=False,
    )
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8")
    best_parameters = best["parameters"]
    development = pd.concat([train, validation], ignore_index=True)
    champion = build_complaint_model(
        str(best_parameters["feature_set"]),
        float(best_parameters["c_value"]),
        float(best_parameters["negative_class_weight"]),
        random_seed,
    )
    champion.fit(development["review_text_clean"], development["complaint_target"].to_numpy())
    threshold = float(best_parameters["threshold"])
    test_probabilities = predict_complaint_probabilities(champion, test["review_text_clean"])
    test_metrics = binary_evaluation(test["complaint_target"].to_numpy(), test_probabilities, threshold)

    uncertain = np.abs(test_probabilities - threshold) < uncertainty_margin
    covered = ~uncertain
    operational_threshold = min(1.0, threshold + uncertainty_margin)
    operational_metrics = binary_evaluation(
        test["complaint_target"].to_numpy(),
        test_probabilities,
        operational_threshold,
    )
    covered_metrics = (
        binary_evaluation(
            test.loc[covered, "complaint_target"].to_numpy(),
            test_probabilities[covered],
            threshold,
        )
        if covered.any() and test.loc[covered, "complaint_target"].nunique() == 2
        else None
    )
    coverage = float(covered.mean())
    version = "v3-complaint-gold" if label_source == "human_gold" else "v2-complaint-weak"
    bundle = {
        "model": champion,
        "negative_threshold": threshold,
        "uncertainty_margin": uncertainty_margin,
        "label_source": label_source,
        "version": version,
        "trained_at": datetime.now().astimezone().isoformat(),
        "feature_set": best_parameters["feature_set"],
        "c_value": float(best_parameters["c_value"]),
        "negative_class_weight": float(best_parameters["negative_class_weight"]),
    }
    joblib.dump(bundle, MODEL_PATH)

    all_reviews = reviews[
        reviews["review_text_clean"].notna()
        & (reviews["text_length"] > 0)
        & (~reviews["is_duplicate"])
    ].copy().reset_index(drop=True)
    all_predictions = predict_with_abstention(bundle, all_reviews["review_text_clean"])
    prediction_output = pd.concat(
        [
            all_reviews[
                [
                    "review_id",
                    "canonical_place_id",
                    "place_name",
                    "place_category",
                    "review_text_clean",
                    "weak_sentiment_label",
                ]
            ],
            all_predictions,
        ],
        axis=1,
    )
    prediction_output.to_parquet(PREDICTIONS_PATH, index=False)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        test_metrics["labels"],
        FIGURE_DIR / "complaint_confusion_matrix_normalized.png",
        normalize=True,
        title="Complaint Detector Confusion Matrix (Row Normalized)",
    )
    plot_precision_recall_curve(
        test["complaint_target"].to_numpy(),
        test_probabilities,
        FIGURE_DIR / "complaint_precision_recall_curve.png",
        threshold,
    )
    plot_calibration_curve(
        test["complaint_target"].to_numpy(),
        test_probabilities,
        FIGURE_DIR / "complaint_calibration_curve.png",
    )

    gate_results = {
        "negative_recall": operational_metrics["negative_recall"] >= float(acceptance.get("negative_recall", 0.80)),
        "negative_precision": operational_metrics["negative_precision"] >= float(acceptance.get("negative_precision", 0.60)),
        "macro_f1": operational_metrics["macro_f1"] >= float(acceptance.get("macro_f1", 0.70)),
        "balanced_accuracy": operational_metrics["balanced_accuracy"] >= float(acceptance.get("balanced_accuracy", 0.70)),
        "prediction_coverage": coverage >= float(acceptance.get("minimum_prediction_coverage", 0.75)),
    }
    deployment_ready = label_source == "human_gold" and all(gate_results.values())
    gold_manifest_path = REPORT_DIR / "gold_dataset_manifest.json"
    gold_manifest = (
        json.loads(gold_manifest_path.read_text(encoding="utf-8"))
        if gold_manifest_path.exists()
        else {}
    )
    metrics = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "task": "negative complaint vs non-negative",
        "label_source": label_source,
        "evaluation_mode": "gold_holdout" if label_source == "human_gold" else "weak_label_holdout",
        "gold_evaluation_available": label_source == "human_gold",
        "gold_semantic_sha256": gold_manifest.get("semantic_sha256"),
        "training_rows": int(len(dataset)),
        "split": split_report,
        "selection": {
            "objective": f"validation F{beta:g} with precision floor {minimum_precision:.2f}",
            "parameters": best_parameters,
            "comparison_rows": int(len(comparison)),
        },
        "test_metrics": test_metrics,
        "abstention": {
            "uncertainty_margin": uncertainty_margin,
            "negative_decision_threshold": operational_threshold,
            "uncertain_rows": int(uncertain.sum()),
            "coverage": coverage,
            "operational_metrics_with_uncertain_as_non_negative": operational_metrics,
            "covered_metrics": covered_metrics,
        },
        "slice_metrics": evaluation_slices(test, test_probabilities, threshold),
        "acceptance_gates": acceptance,
        "gate_results": gate_results,
        "deployment_ready": deployment_ready,
        "outputs": {
            "model": str(MODEL_PATH),
            "metrics": str(METRICS_PATH),
            "comparison": str(COMPARISON_PATH),
            "predictions": str(PREDICTIONS_PATH),
        },
        "limitations": [
            "Weak-label evaluation is not production evidence." if label_source != "human_gold" else "",
            "The detector predicts review-level complaint probability; aspect-level assignment is handled separately.",
            "Thresholds must be re-tuned whenever human gold labels or the training distribution change.",
            "The current human gold dataset was reviewed by one annotator."
            if label_source == "human_gold"
            else "",
            "Annotators could see AI suggestions, so confirmation bias remains possible."
            if label_source == "human_gold"
            else "",
        ],
    }
    metrics["limitations"] = [item for item in metrics["limitations"] if item]
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_complaint_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"model", "negative_threshold", "uncertainty_margin", "version", "label_source"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Complaint model bundle is missing fields: {sorted(missing)}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TobaPulse complaint detector.")
    parser.parse_args()
    result = train_complaint_detector()
    print(
        json.dumps(
            {
                "label_source": result["label_source"],
                "test_metrics": result["test_metrics"],
                "deployment_ready": result["deployment_ready"],
                "model": result["outputs"]["model"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
