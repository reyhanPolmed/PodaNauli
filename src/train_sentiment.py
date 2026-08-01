"""Tahap 4 pipeline: train CPU sentiment baseline models."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluate import class_metrics_frame, classification_metrics, plot_class_metrics, plot_confusion_matrix
from src.gold_dataset import SPLIT_PATH, apply_gold_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"
GOLD_PATH = ROOT / "data" / "annotations" / "sentiment_gold.csv"

LABEL_ORDER = ["negative", "neutral", "positive"]


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        return {"random_seed": 42}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def prepare_trainable_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Filter to deduplicated weak-labeled reviews with usable text."""
    required = {"review_text_clean", "weak_sentiment_label", "canonical_place_id", "is_duplicate", "text_length"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    trainable = reviews[
        reviews["review_text_clean"].notna()
        & reviews["weak_sentiment_label"].isin(LABEL_ORDER)
        & reviews["canonical_place_id"].notna()
        & (reviews["text_length"] > 0)
        & (~reviews["is_duplicate"])
    ].copy()
    trainable["review_text_clean"] = trainable["review_text_clean"].astype(str)
    trainable = trainable.sort_values("review_id").reset_index(drop=True)
    return trainable


def prepare_sentiment_training_data(
    reviews: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Prefer human gold labels once configured minimum coverage is satisfied."""
    weak_data = prepare_trainable_reviews(reviews)
    annotation_config = config.get("sentiment_improvement", {}).get("annotation", {})
    minimum_rows = int(annotation_config.get("minimum_gold_rows", 300))
    minimum_per_class = int(annotation_config.get("minimum_gold_rows_per_class", 50))
    if not GOLD_PATH.exists():
        return weak_data, "weak_rating"
    gold = pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not {"review_id", "manual_sentiment_label"}.issubset(gold.columns):
        return weak_data, "weak_rating"
    gold["manual_sentiment_label"] = gold["manual_sentiment_label"].str.strip().str.lower()
    gold = gold[gold["manual_sentiment_label"].isin(LABEL_ORDER)].drop_duplicates("review_id", keep="last")
    counts = gold["manual_sentiment_label"].value_counts()
    ready = len(gold) >= minimum_rows and all(
        int(counts.get(label, 0)) >= minimum_per_class
        for label in LABEL_ORDER
    )
    if not ready:
        return weak_data, "weak_rating"
    merged = weak_data.drop(columns=["weak_sentiment_label"]).merge(
        gold[["review_id", "manual_sentiment_label"]],
        on="review_id",
        how="inner",
    )
    merged["weak_sentiment_label"] = merged.pop("manual_sentiment_label")
    return merged.sort_values("review_id").reset_index(drop=True), "human_gold"


def split_train_test(data: pd.DataFrame, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use StratifiedGroupKFold to avoid place leakage between train and test."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
    y = data["weak_sentiment_label"].to_numpy()
    groups = data["canonical_place_id"].to_numpy()
    train_idx, test_idx = next(splitter.split(data["review_text_clean"], y, groups))
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    split_report = {
        "split_method": "StratifiedGroupKFold first fold",
        "n_splits": 5,
        "random_seed": random_seed,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_groups": int(len(train_groups)),
        "test_groups": int(len(test_groups)),
        "group_overlap_count": int(len(overlap)),
        "train_label_counts": data.iloc[train_idx]["weak_sentiment_label"].value_counts().to_dict(),
        "test_label_counts": data.iloc[test_idx]["weak_sentiment_label"].value_counts().to_dict(),
    }
    if overlap:
        raise ValueError(f"Group leakage detected across train/test: {sorted(list(overlap))[:5]}")
    return train_idx, test_idx, split_report


def split_train_validation_test(
    data: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create place-disjoint splits when a persistent gold manifest is unavailable."""
    development_idx, test_idx, outer_report = split_train_test(data, random_seed=random_seed)
    development = data.iloc[development_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)
    train_idx, validation_idx, _inner_report = split_train_test(development, random_seed=random_seed + 1)
    train = development.iloc[train_idx].reset_index(drop=True)
    validation = development.iloc[validation_idx].reset_index(drop=True)
    groups = {
        "train": set(train["canonical_place_id"]),
        "validation": set(validation["canonical_place_id"]),
        "test": set(test["canonical_place_id"]),
    }
    overlap = (
        (groups["train"] & groups["validation"])
        | (groups["train"] & groups["test"])
        | (groups["validation"] & groups["test"])
    )
    if overlap:
        raise ValueError(f"Place leakage detected in three-way split: {sorted(overlap)[:5]}")
    report = {
        "method": "Nested StratifiedGroupKFold first folds",
        "random_seed": int(random_seed),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_groups": int(len(groups["train"])),
        "validation_groups": int(len(groups["validation"])),
        "test_groups": int(len(groups["test"])),
        "group_overlap_count": 0,
        "train_label_counts": train["weak_sentiment_label"].value_counts().to_dict(),
        "validation_label_counts": validation["weak_sentiment_label"].value_counts().to_dict(),
        "test_label_counts": test["weak_sentiment_label"].value_counts().to_dict(),
        "outer_split": outer_report,
    }
    return train, validation, test, report


def build_candidate_models(random_seed: int = 42) -> dict[str, Pipeline]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
    )
    return {
        "sentiment_logistic": Pipeline(
            [
                ("tfidf_word", word_vectorizer),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "sentiment_svm": Pipeline(
            [
                ("tfidf_word", clone(word_vectorizer)),
                ("classifier", LinearSVC(class_weight="balanced", random_state=random_seed)),
            ]
        ),
        "sentiment_word_char_logistic": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", clone(word_vectorizer)),
                            ("char", char_vectorizer),
                        ]
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }


def cross_validate_macro_f1(
    model: Pipeline,
    x: pd.Series,
    y: np.ndarray,
    groups: np.ndarray,
    random_seed: int,
    max_splits: int = 3,
) -> dict[str, Any]:
    group_count = len(set(groups))
    label_counts = pd.Series(y).value_counts()
    n_splits = min(max_splits, group_count, int(label_counts.min()))
    if n_splits < 2:
        return {"fold_macro_f1": [], "mean_macro_f1": None, "std_macro_f1": None, "n_splits": 0}
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        fold_model = clone(model)
        fold_model.fit(x.iloc[train_idx], y[train_idx])
        predictions = fold_model.predict(x.iloc[valid_idx])
        metrics = classification_metrics(y[valid_idx], predictions, LABEL_ORDER)
        fold_scores.append(metrics["macro_f1"])
    return {
        "fold_macro_f1": [float(score) for score in fold_scores],
        "mean_macro_f1": float(np.mean(fold_scores)),
        "std_macro_f1": float(np.std(fold_scores)),
        "n_splits": n_splits,
    }


def model_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def normalize_scores(values: dict[str, float], higher_is_better: bool = True) -> dict[str, float]:
    if not values:
        return {}
    raw = values.copy()
    if not higher_is_better:
        raw = {key: -value for key, value in raw.items()}
    min_value = min(raw.values())
    max_value = max(raw.values())
    if np.isclose(max_value, min_value):
        return {key: 1.0 for key in raw}
    return {key: float((value - min_value) / (max_value - min_value)) for key, value in raw.items()}


def choose_champion(comparison: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Choose champion with documented weighted normalized components."""
    rows = comparison.copy()
    component_values = {
        "macro_f1_component": normalize_scores(rows.set_index("model_name")["macro_f1"].to_dict(), True),
        "negative_recall_component": normalize_scores(rows.set_index("model_name")["negative_recall"].to_dict(), True),
        "cv_stability_component": normalize_scores(rows.set_index("model_name")["cv_stability_raw"].to_dict(), True),
        "inference_speed_component": normalize_scores(rows.set_index("model_name")["inference_samples_per_second"].to_dict(), True),
        "deployment_size_component": normalize_scores(rows.set_index("model_name")["model_size_efficiency_raw"].to_dict(), True),
    }
    for column, values in component_values.items():
        rows[column] = rows["model_name"].map(values)
    rows["champion_score"] = (
        0.45 * rows["macro_f1_component"]
        + 0.20 * rows["negative_recall_component"]
        + 0.15 * rows["cv_stability_component"]
        + 0.10 * rows["inference_speed_component"]
        + 0.10 * rows["deployment_size_component"]
    )
    rows = rows.sort_values(["champion_score", "macro_f1", "negative_recall"], ascending=False).reset_index(drop=True)
    return str(rows.iloc[0]["model_name"]), rows


def train_and_evaluate(sample_limit: int | None = None) -> dict[str, Any]:
    config = load_config()
    random_seed = int(config.get("random_seed", 42))
    acceptance = config.get("sentiment_improvement", {}).get("acceptance_gates", {})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path = PROCESSED_DIR / "reviews_clean.parquet"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing processed reviews: {reviews_path}")
    reviews = pd.read_parquet(reviews_path)
    trainable, label_source = prepare_sentiment_training_data(reviews, config)
    if sample_limit is not None:
        trainable = trainable.sample(n=min(sample_limit, len(trainable)), random_state=random_seed).reset_index(drop=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ORDER)
    joblib.dump(label_encoder, MODEL_DIR / "sentiment_label_encoder.joblib")

    if label_source == "human_gold" and SPLIT_PATH.exists() and sample_limit is None:
        train_data, validation_data, test_data, split_report = apply_gold_split(trainable)
        split_report["source"] = str(SPLIT_PATH)
    else:
        train_data, validation_data, test_data, split_report = split_train_validation_test(
            trainable,
            random_seed=random_seed,
        )
    x_train = train_data["review_text_clean"]
    y_train = train_data["weak_sentiment_label"].to_numpy()
    x_validation = validation_data["review_text_clean"]
    y_validation = validation_data["weak_sentiment_label"].to_numpy()
    x_test = test_data["review_text_clean"]
    y_test = test_data["weak_sentiment_label"].to_numpy()
    groups_train = train_data["canonical_place_id"].to_numpy()

    candidates = build_candidate_models(random_seed=random_seed)
    comparison_rows = []
    model_reports: dict[str, Any] = {}

    for model_name, model in candidates.items():
        cv = cross_validate_macro_f1(model, x_train, y_train, groups_train, random_seed=random_seed)

        fitted_model = clone(model)
        train_start = time.perf_counter()
        fitted_model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        y_pred = fitted_model.predict(x_validation)
        inference_seconds = time.perf_counter() - predict_start
        inference_samples_per_second = (
            float(len(x_validation) / inference_seconds)
            if inference_seconds > 0
            else 0.0
        )

        metrics = classification_metrics(y_validation, y_pred, LABEL_ORDER)
        class_metrics = class_metrics_frame(y_validation, y_pred, LABEL_ORDER)

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(fitted_model, model_path)
        size_mb = model_file_size_mb(model_path)
        cv_std = cv["std_macro_f1"] if cv["std_macro_f1"] is not None else 1.0
        cv_stability_raw = max(0.0, 1.0 - float(cv_std))
        size_efficiency = 1.0 / max(size_mb, 1e-9)

        comparison_rows.append(
            {
                "model_name": model_name,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "negative_recall": metrics["negative_recall"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "cv_macro_f1_mean": cv["mean_macro_f1"],
                "cv_macro_f1_std": cv["std_macro_f1"],
                "cv_stability_raw": cv_stability_raw,
                "training_seconds": float(training_seconds),
                "inference_seconds": float(inference_seconds),
                "inference_samples_per_second": inference_samples_per_second,
                "model_size_mb": size_mb,
                "model_size_efficiency_raw": size_efficiency,
                "artifact_path": str(model_path),
                "selection_split": "validation",
            }
        )
        model_reports[model_name] = {
            "evaluation_split": "validation",
            "metrics": metrics,
            "class_metrics": class_metrics.to_dict(orient="records"),
            "cross_validation": cv,
            "training_seconds": float(training_seconds),
            "inference_seconds": float(inference_seconds),
            "inference_samples_per_second": inference_samples_per_second,
            "model_size_mb": size_mb,
            "artifact_path": str(model_path),
        }

    comparison = pd.DataFrame(comparison_rows)
    champion_name, scored_comparison = choose_champion(comparison)
    champion_path = MODEL_DIR / "sentiment_champion.joblib"
    scored_comparison.to_csv(REPORT_DIR / "sentiment_model_comparison.csv", index=False)

    development_data = pd.concat([train_data, validation_data], ignore_index=True)
    final_model = clone(candidates[champion_name])
    final_training_start = time.perf_counter()
    final_model.fit(
        development_data["review_text_clean"],
        development_data["weak_sentiment_label"].to_numpy(),
    )
    final_training_seconds = time.perf_counter() - final_training_start
    final_inference_start = time.perf_counter()
    final_predictions = final_model.predict(x_test)
    final_inference_seconds = time.perf_counter() - final_inference_start
    champion_metrics = classification_metrics(y_test, final_predictions, LABEL_ORDER)
    champion_class_metrics = class_metrics_frame(y_test, final_predictions, LABEL_ORDER)
    joblib.dump(final_model, MODEL_DIR / f"{champion_name}.joblib")
    joblib.dump(final_model, champion_path)

    plot_confusion_matrix(
        champion_metrics["confusion_matrix"],
        LABEL_ORDER,
        FIGURE_DIR / "sentiment_confusion_matrix.png",
    )
    plot_confusion_matrix(
        champion_metrics["confusion_matrix"],
        LABEL_ORDER,
        FIGURE_DIR / "sentiment_confusion_matrix_normalized.png",
        normalize=True,
        title="Sentiment Confusion Matrix (Row Normalized)",
    )
    plot_class_metrics(champion_class_metrics, FIGURE_DIR / "sentiment_class_metrics.png")

    neutral_f1 = float(
        champion_metrics["classification_report"].get("neutral", {}).get("f1-score", 0.0)
    )
    gate_results = {
        "macro_f1": champion_metrics["macro_f1"] >= float(acceptance.get("macro_f1", 0.70)),
        "negative_recall": champion_metrics["negative_recall"]
        >= float(acceptance.get("negative_recall", 0.80)),
        "neutral_f1": neutral_f1 >= float(acceptance.get("neutral_f1", 0.50)),
        "balanced_accuracy": champion_metrics["balanced_accuracy"]
        >= float(acceptance.get("balanced_accuracy", 0.70)),
    }
    deployment_ready = label_source == "human_gold" and all(gate_results.values())
    gold_manifest_path = REPORT_DIR / "gold_dataset_manifest.json"
    gold_manifest = (
        json.loads(gold_manifest_path.read_text(encoding="utf-8"))
        if gold_manifest_path.exists()
        else {}
    )
    metrics_summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": {
            "reviews_path": str(reviews_path),
            "total_reviews": int(len(reviews)),
            "trainable_reviews_after_dedup": int(len(trainable)),
            "deduplication_rule": "Use rows with is_duplicate == False before split.",
            "label_source": (
                "manual_sentiment_label from validated human gold dataset."
                if label_source == "human_gold"
                else "weak_sentiment_label derived from reviewer_rating."
            ),
            "manual_annotation_file": str(GOLD_PATH),
            "manual_annotation_used": label_source == "human_gold",
            "gold_semantic_sha256": gold_manifest.get("semantic_sha256"),
        },
        "split": split_report,
        "models": model_reports,
        "champion": {
            "model_name": champion_name,
            "artifact_path": str(champion_path),
            "selection_formula": "0.45*macro_f1_norm + 0.20*negative_recall_norm + 0.15*cv_stability_norm + 0.10*inference_speed_norm + 0.10*deployment_size_norm",
            "selection_split": "validation",
            "selection_table_path": str(REPORT_DIR / "sentiment_model_comparison.csv"),
            "validation_metrics": model_reports[champion_name]["metrics"],
            "final_test_split": "test",
            "metrics": champion_metrics,
            "final_training_seconds": float(final_training_seconds),
            "final_inference_seconds": float(final_inference_seconds),
            "final_inference_samples_per_second": (
                float(len(x_test) / final_inference_seconds)
                if final_inference_seconds > 0
                else 0.0
            ),
        },
        "acceptance_gates": acceptance,
        "gate_results": gate_results,
        "deployment_ready": deployment_ready,
        "limitations": [
            *(
                []
                if label_source == "human_gold"
                else [
                    "Sentiment labels are weak labels from rating, not manually verified ground truth.",
                    "Manual gold labels have not reached the configured readiness threshold.",
                ]
            ),
            "Reviews from the same canonical place are isolated across train, validation, and test.",
            "The current human gold dataset was reviewed by one annotator, so inter-annotator agreement is unavailable."
            if label_source == "human_gold"
            else "",
            "Annotators could see AI suggestions, so confirmation bias remains possible."
            if label_source == "human_gold"
            else "",
        ],
    }
    metrics_summary["limitations"] = [item for item in metrics_summary["limitations"] if item]
    (REPORT_DIR / "sentiment_metrics.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment baseline models.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--model", default="cpu-baselines", help="Reserved for future transformer option.")
    parser.add_argument("--enable-transformers", action="store_true", help="Transformer training is intentionally not implemented in CPU baseline.")
    args = parser.parse_args()
    if args.enable_transformers:
        raise SystemExit("Transformer training is optional and not part of Tahap 4 CPU baseline.")
    summary = train_and_evaluate(sample_limit=args.sample_limit)
    print(
        json.dumps(
            {
                "champion": summary["champion"]["model_name"],
                "macro_f1": summary["champion"]["metrics"]["macro_f1"],
                "negative_recall": summary["champion"]["metrics"]["negative_recall"],
                "comparison_path": str(REPORT_DIR / "sentiment_model_comparison.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
