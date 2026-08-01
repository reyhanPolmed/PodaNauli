"""Tahap 9 pipeline: model, data, ranking, and error evaluation."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluate import classification_metrics
from src.gold_dataset import SPLIT_PATH, apply_gold_split
from src.train_sentiment import (
    LABEL_ORDER,
    load_config as load_sentiment_config,
    prepare_sentiment_training_data,
    prepare_trainable_reviews,
    split_train_test,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"


POSITIVE_HINTS = {"bagus", "indah", "enak", "ramah", "bersih", "nyaman", "mantap", "keren", "recommended", "good", "nice", "beautiful"}
NEGATIVE_HINTS = {"tidak", "kurang", "buruk", "kotor", "mahal", "sempit", "jelek", "lambat", "rusak", "bad", "dirty", "worst"}
LOCAL_HINTS = {"toba", "balige", "samosir", "parapat", "batak", "bulbul", "lumban", "sipiso", "pangururan"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def has_mixed_language(text: str) -> bool:
    lower = text.lower()
    has_english = bool(re.search(r"\b(the|and|is|was|food|room|hotel|view|good|bad|service)\b", lower))
    has_indonesian = bool(re.search(r"\b(dan|yang|tidak|bagus|makanan|kamar|pelayanan|tempat)\b", lower))
    return has_english and has_indonesian


def error_tags(text: str, rating: float | None, actual: str, predicted: str, place_review_count: int) -> list[str]:
    lower = text.lower()
    tags = []
    if len(text) <= 30:
        tags.append("VERY_SHORT_TEXT")
    if POSITIVE_HINTS & set(re.findall(r"\w+", lower)) and NEGATIVE_HINTS & set(re.findall(r"\w+", lower)):
        tags.append("MIXED_POSITIVE_NEGATIVE")
    if any(marker in lower for marker in ["mantap sekali", "bagus sekali"]) and actual == "negative":
        tags.append("POSSIBLE_SARCASM_OR_RATING_TEXT_MISMATCH")
    if has_mixed_language(text):
        tags.append("MIXED_ID_EN")
    if LOCAL_HINTS & set(re.findall(r"\w+", lower)):
        tags.append("LOCAL_NAME_OR_TERM")
    if re.search(r"\b(yg|tdk|gk|ga|bgt|tmpt|sdh|krn|dgn)\b", lower):
        tags.append("TYPO_OR_INFORMAL_ABBREVIATION")
    if rating is not None and not pd.isna(rating):
        if rating >= 4 and NEGATIVE_HINTS & set(re.findall(r"\w+", lower)):
            tags.append("HIGH_RATING_WITH_NEGATIVE_TEXT")
        if rating <= 2 and POSITIVE_HINTS & set(re.findall(r"\w+", lower)):
            tags.append("LOW_RATING_WITH_POSITIVE_TEXT")
    if place_review_count < 5:
        tags.append("LOW_PLACE_REVIEW_COUNT")
    if not tags:
        tags.append("GENERAL_MODEL_ERROR")
    return tags


def sentiment_error_analysis(sample_size: int = 500) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the sentiment test split and export misclassified examples."""
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    config = load_sentiment_config()
    trainable, label_source = prepare_sentiment_training_data(reviews, config)
    if label_source == "human_gold" and SPLIT_PATH.exists():
        _train_data, _validation_data, test_data, split_report = apply_gold_split(trainable)
        split_report["source"] = str(SPLIT_PATH)
    else:
        _train_idx, test_idx, split_report = split_train_test(trainable, random_seed=42)
        test_data = trainable.iloc[test_idx].copy().reset_index(drop=True)
    model = joblib.load(MODEL_DIR / "sentiment_champion.joblib")

    start = time.perf_counter()
    predictions = model.predict(test_data["review_text_clean"])
    inference_seconds = time.perf_counter() - start
    actual = test_data["weak_sentiment_label"].to_numpy()
    metrics = classification_metrics(actual, predictions, LABEL_ORDER)

    place_counts = trainable.groupby("canonical_place_id")["review_id"].count().to_dict()
    errors = test_data[predictions != actual].copy()
    errors["actual_label"] = actual[predictions != actual]
    errors["predicted_label"] = predictions[predictions != actual]
    errors["error_type"] = np.where(
        errors["actual_label"] == "negative",
        "false_negative_negative_class",
        np.where(errors["predicted_label"] == "negative", "false_positive_negative_class", "other_misclassification"),
    )
    errors["analysis_tags"] = errors.apply(
        lambda row: "|".join(
            error_tags(
                str(row["review_text_clean"]),
                row.get("reviewer_rating"),
                row["actual_label"],
                row["predicted_label"],
                int(place_counts.get(row["canonical_place_id"], 0)),
            )
        ),
        axis=1,
    )
    error_columns = [
        "review_id",
        "canonical_place_id",
        "place_name",
        "place_category",
        "reviewer_rating",
        "review_text_raw",
        "review_text_clean",
        "actual_label",
        "predicted_label",
        "error_type",
        "analysis_tags",
    ]
    errors = errors[error_columns].sort_values(["error_type", "review_id"]).head(sample_size)
    errors.to_csv(REPORT_DIR / "sentiment_error_analysis.csv", index=False, encoding="utf-8")

    report = {
        "test_rows": int(len(test_data)),
        "error_rows": int((predictions != actual).sum()),
        "sampled_error_rows": int(len(errors)),
        "inference_seconds": float(inference_seconds),
        "inference_samples_per_second": float(len(test_data) / inference_seconds) if inference_seconds > 0 else 0.0,
        "metrics": metrics,
        "label_source": label_source,
        "split": split_report,
        "error_type_counts": errors["error_type"].value_counts().to_dict(),
        "tag_counts": errors["analysis_tags"].str.split("|").explode().value_counts().to_dict(),
    }
    return errors, report


def entity_resolution_audit() -> dict[str, Any]:
    mapping_path = PROCESSED_DIR / "entity_mapping.parquet"
    review_path = REPORT_DIR / "entity_matches_for_review.csv"
    mapping = pd.read_parquet(mapping_path)
    needs_review = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame()
    return {
        "mapping_rows": int(len(mapping)),
        "canonical_places": int(mapping["canonical_place_id"].nunique()),
        "auto_matches": int((mapping["match_status"] == "auto").sum()),
        "new_canonical_rows": int(mapping["match_status"].astype(str).str.startswith("new").sum()),
        "needs_manual_review_rows": int(mapping["needs_manual_review"].sum()),
        "manual_review_file": str(review_path),
        "manual_review_file_rows": int(len(needs_review)),
    }


def service_gap_stability() -> dict[str, Any]:
    scores = pd.read_parquet(PROCESSED_DIR / "service_gap_scores.parquet")
    top_scores = scores.sort_values("service_gap_score", ascending=False).head(50).copy()
    by_category = (
        scores.groupby("place_category")["service_gap_score"]
        .agg(["count", "mean", "max", "std"])
        .reset_index()
        .fillna(0)
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(scores)),
        "places": int(scores["canonical_place_id"].nunique()),
        "aspects": int(scores["aspect"].nunique()),
        "score_min": float(scores["service_gap_score"].min()),
        "score_max": float(scores["service_gap_score"].max()),
        "top_50_unique_places": int(top_scores["canonical_place_id"].nunique()),
        "top_50_unique_aspects": int(top_scores["aspect"].nunique()),
        "category_summary": by_category,
    }


def data_completeness_summary() -> dict[str, Any]:
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    places = pd.read_parquet(PROCESSED_DIR / "places_master.parquet")
    return {
        "reviews": {
            "rows": int(len(reviews)),
            "with_text": int((reviews["text_length"] > 0).sum()),
            "duplicates": int(reviews["is_duplicate"].sum()),
            "weak_label_missing": int(reviews["weak_sentiment_label"].isna().sum()),
            "review_date_parse_success": int(reviews["review_date_parsing_success"].sum()),
        },
        "places": {
            "rows": int(len(places)),
            "with_valid_coordinates": int((places["latitude"].notna() & places["longitude"].notna()).sum()),
            "with_place_rating": int(places["place_rating"].notna().sum()),
            "with_price": int(places["price_text_original"].notna().sum()),
            "with_address": int(places["address"].notna().sum()),
        },
    }


def create_model_registry() -> dict[str, Any]:
    sentiment_metrics = load_json(REPORT_DIR / "sentiment_metrics.json")
    complaint_metrics = load_json(REPORT_DIR / "complaint_metrics.json")
    aspect_metrics = load_json(REPORT_DIR / "aspect_metrics.json")
    aspect_uses_gold = aspect_metrics.get("label_source") == "human_gold"
    dataset_hash = load_json(REPORT_DIR / "raw_data_manifest.json").get("sha256")
    gold_hash = load_json(REPORT_DIR / "gold_dataset_manifest.json").get("semantic_sha256")
    sentiment_uses_gold = bool(
        sentiment_metrics.get("dataset", {}).get("manual_annotation_used", False)
    )
    now = datetime.now().astimezone().isoformat()
    registry = {
        "generated_at": now,
        "dataset_hash": dataset_hash,
        "models": [
            {
                "model_name": "sentiment_champion",
                "task": "sentiment_classification",
                "version": "v2-human-gold" if sentiment_uses_gold else "v1-weak-label-baseline",
                "trained_at": sentiment_metrics.get("generated_at"),
                "dataset_hash": dataset_hash,
                "training_label_hash": gold_hash if sentiment_uses_gold else None,
                "feature_configuration": sentiment_metrics.get("champion", {}).get("model_name"),
                "metrics": sentiment_metrics.get("champion", {}).get("metrics"),
                "artifact_path": str(MODEL_DIR / "sentiment_champion.joblib"),
                "limitations": sentiment_metrics.get("limitations", []),
                "deployment_ready": bool(sentiment_metrics.get("deployment_ready", False)),
                "is_champion": True,
            },
            {
                "model_name": "aspect_champion",
                "task": "aspect_multilabel_classification",
                "version": aspect_metrics.get("version", "v1-rule-label-baseline"),
                "trained_at": aspect_metrics.get("generated_at"),
                "dataset_hash": dataset_hash,
                "training_label_hash": (
                    aspect_metrics.get("training_label_hash") if aspect_uses_gold else None
                ),
                "feature_configuration": (
                    aspect_metrics.get("selection")
                    if aspect_uses_gold
                    else "word+char TF-IDF; OneVsRest LogisticRegression trained on rule-based weak labels"
                ),
                "metrics": aspect_metrics.get("test_metrics"),
                "artifact_path": str(MODEL_DIR / "aspect_champion.joblib"),
                "limitations": aspect_metrics.get("limitations", []),
                "deployment_ready": bool(aspect_metrics.get("deployment_ready", False)),
                "is_champion": True,
            },
            {
                "model_name": "complaint_detector",
                "task": "negative_complaint_detection",
                "version": (
                    "v3-complaint-gold"
                    if complaint_metrics.get("label_source") == "human_gold"
                    else "v2-complaint-weak"
                ),
                "trained_at": complaint_metrics.get("generated_at"),
                "dataset_hash": dataset_hash,
                "training_label_hash": gold_hash
                if complaint_metrics.get("label_source") == "human_gold"
                else None,
                "feature_configuration": complaint_metrics.get("selection", {}).get("parameters"),
                "metrics": complaint_metrics.get("test_metrics"),
                "artifact_path": str(MODEL_DIR / "complaint_detector.joblib"),
                "limitations": complaint_metrics.get("limitations", []),
                "deployment_ready": bool(complaint_metrics.get("deployment_ready", False)),
                "is_champion": True,
            },
        ],
    }
    (MODEL_DIR / "model_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


def calculate_project_readiness(
    registry: dict[str, Any],
    service_gap_methodology: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Separate validated model-pipeline readiness from application deployment."""
    model_status = {
        str(model.get("model_name")): bool(model.get("deployment_ready", False))
        for model in registry.get("models", [])
    }
    validation = service_gap_methodology.get("human_ranking_validation", {})
    threshold = float(
        config.get("sentiment_improvement", {})
        .get("acceptance_gates", {})
        .get("top_20_service_gap_validity", 0.80)
    )
    required_models = [
        "sentiment_champion",
        "complaint_detector",
        "aspect_champion",
    ]
    checks = {
        "required_models_deployment_ready": all(
            model_status.get(name, False) for name in required_models
        ),
        "top20_fully_reviewed": bool(validation.get("fully_reviewed", False)),
        "top20_evidence_validity": (
            validation.get("evidence_validity_rate") is not None
            and float(validation["evidence_validity_rate"]) >= threshold
        ),
        "top20_priority_validity": (
            validation.get("priority_validity_rate") is not None
            and float(validation["priority_validity_rate"]) >= threshold
        ),
        "top20_overall_validity": (
            validation.get("validity_rate") is not None
            and float(validation["validity_rate"]) >= threshold
        ),
    }
    return {
        "acceptance_threshold": threshold,
        "model_status": model_status,
        "service_gap_validation": validation,
        "checks": checks,
        "model_and_ranking_pipeline_ready": all(checks.values()),
        "production_application_ready": False,
        "production_application_blockers": [
            "FastAPI stage is not implemented.",
            "Dashboard stage is not implemented.",
            "Production monitoring, drift alerts, and rollback operations are not implemented.",
        ],
        "limitations": [
            "Human gold labels currently come from one annotator.",
            "Visible AI suggestions may introduce confirmation bias.",
        ],
    }


def run_model_evaluation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _errors, sentiment_error_report = sentiment_error_analysis()
    registry = create_model_registry()
    service_gap_methodology = load_json(REPORT_DIR / "service_gap_methodology.json")
    readiness = calculate_project_readiness(
        registry,
        service_gap_methodology,
        load_sentiment_config(),
    )
    readiness_path = REPORT_DIR / "project_readiness.json"
    readiness_path.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "sentiment_classification": sentiment_error_report,
        "complaint_detection": load_json(REPORT_DIR / "complaint_metrics.json"),
        "aspect_classification": {
            "mode": (
                "human_gold_locked_test"
                if load_json(REPORT_DIR / "aspect_metrics.json").get("label_source")
                == "human_gold"
                else "weak_label_coverage_only"
            ),
            "metrics": load_json(REPORT_DIR / "aspect_metrics.json"),
        },
        "aspect_sentiment": load_json(REPORT_DIR / "aspect_sentiment_summary.json"),
        "entity_resolution": entity_resolution_audit(),
        "service_gap_ranking_stability": service_gap_stability(),
        "service_gap_human_validation": service_gap_methodology.get(
            "human_ranking_validation",
            {},
        ),
        "project_readiness": readiness,
        "data_completeness": data_completeness_summary(),
        "inference_performance": {
            "sentiment_samples_per_second": sentiment_error_report["inference_samples_per_second"],
            "sentiment_inference_seconds_for_test_split": sentiment_error_report["inference_seconds"],
        },
        "model_registry_path": str(MODEL_DIR / "model_registry.json"),
        "model_registry_model_count": len(registry["models"]),
        "outputs": {
            "model_evaluation_summary": str(REPORT_DIR / "model_evaluation_summary.json"),
            "sentiment_error_analysis": str(REPORT_DIR / "sentiment_error_analysis.csv"),
            "model_registry": str(MODEL_DIR / "model_registry.json"),
            "project_readiness": str(readiness_path),
        },
    }
    (REPORT_DIR / "model_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TobaPulse model and pipeline evaluation.")
    parser.parse_args()
    summary = run_model_evaluation()
    print(
        json.dumps(
            {
                "sentiment_error_rows": summary["sentiment_classification"]["error_rows"],
                "model_registry": summary["model_registry_path"],
                "summary": summary["outputs"]["model_evaluation_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
