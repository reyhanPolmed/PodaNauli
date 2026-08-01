"""Clause-level aspect sentiment inference for service-gap evidence."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.train_aspect import (
    detect_weak_aspects,
    load_aspect_taxonomy,
    predict_aspect_probabilities,
)
from src.train_complaint import load_complaint_bundle, predict_with_abstention


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
OUTPUT_PATH = PROCESSED_DIR / "review_aspect_sentiment.parquet"
SUMMARY_PATH = REPORT_DIR / "aspect_sentiment_summary.json"
ASPECT_METADATA_PATH = MODEL_DIR / "aspect_metadata.json"

CONTRAST_PATTERN = re.compile(
    r"\s+(?:tetapi|tapi|namun|sedangkan|akan tetapi|walaupun|meskipun|but|however|although)\s+",
    flags=re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?;])\s+|\s*\|\s*")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"random_seed": 42}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def split_clauses(text: Any) -> list[str]:
    """Split review text at sentence and contrast boundaries."""
    if pd.isna(text) or not str(text).strip():
        return []
    clauses = []
    for sentence in SENTENCE_PATTERN.split(str(text)):
        for clause in CONTRAST_PATTERN.split(sentence):
            cleaned = re.sub(r"\s+", " ", clause).strip(" ,.-")
            if cleaned:
                clauses.append(cleaned)
    return clauses


def build_clause_frame(reviews: pd.DataFrame) -> pd.DataFrame:
    records = []
    usable = reviews[
        reviews["review_text_clean"].notna()
        & (reviews["text_length"] > 0)
        & (~reviews["is_duplicate"])
    ]
    for _, row in usable.iterrows():
        clauses = split_clauses(row["review_text_clean"])
        for clause_index, clause in enumerate(clauses):
            records.append(
                {
                    "review_id": row["review_id"],
                    "canonical_place_id": row["canonical_place_id"],
                    "place_name": row["place_name"],
                    "place_category": row["place_category"],
                    "clause_index": clause_index,
                    "clause_text": clause,
                }
            )
    return pd.DataFrame(records)


def infer_aspect_sentiment() -> dict[str, Any]:
    config = load_config()
    model_config = config.get("sentiment_improvement", {}).get("complaint_model", {})
    aspect_metadata = (
        json.loads(ASPECT_METADATA_PATH.read_text(encoding="utf-8"))
        if ASPECT_METADATA_PATH.exists()
        else {}
    )
    aspect_threshold = float(
        aspect_metadata.get(
            "threshold",
            model_config.get("aspect_probability_threshold", 0.50),
        )
    )
    aspect_uses_gold = aspect_metadata.get("label_source") == "human_gold"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    clauses = build_clause_frame(reviews)
    if clauses.empty:
        raise ValueError("No usable review clauses were found.")

    complaint_bundle = load_complaint_bundle()
    complaint_predictions = predict_with_abstention(complaint_bundle, clauses["clause_text"])
    aspect_model = joblib.load(MODEL_DIR / "aspect_champion.joblib")
    multilabel_binarizer = joblib.load(MODEL_DIR / "aspect_multilabel_binarizer.joblib")
    aspect_ids = [str(item) for item in multilabel_binarizer.classes_]
    aspect_probabilities = predict_aspect_probabilities(aspect_model, clauses["clause_text"], aspect_ids)
    taxonomy = load_aspect_taxonomy()

    records = []
    for position, row in clauses.iterrows():
        detection = detect_weak_aspects(row["clause_text"], taxonomy)
        rule_aspects = [aspect for aspect in detection["weak_aspects"] if aspect != "lainnya"]
        negative_rule_aspects = set(detection["negative_aspects"])
        probability_row = aspect_probabilities.iloc[position]
        candidates = sorted(
            [
                (aspect_id, float(probability_row[f"aspect_probability_{aspect_id}"]))
                for aspect_id in aspect_ids
                if aspect_id != "lainnya"
                and float(probability_row[f"aspect_probability_{aspect_id}"]) >= aspect_threshold
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        if aspect_uses_gold:
            selected_aspects = [aspect_id for aspect_id, _probability in candidates[:3]]
            aspect_source = "human_gold_aspect_model"
        else:
            selected_aspects = rule_aspects.copy()
            aspect_source = "taxonomy_rule"
            if not selected_aspects:
                selected_aspects = [aspect_id for aspect_id, _probability in candidates[:3]]
                aspect_source = "aspect_model_fallback"
        if not selected_aspects:
            continue

        complaint = complaint_predictions.iloc[position]
        for aspect_id in sorted(set(selected_aspects)):
            explicit_negative = aspect_id in negative_rule_aspects
            if explicit_negative or complaint["complaint_decision"] == "negative":
                sentiment_label = "negative"
                is_negative = True
            elif complaint["complaint_decision"] == "uncertain":
                sentiment_label = "uncertain"
                is_negative = False
            else:
                sentiment_label = "non_negative"
                is_negative = False
            if explicit_negative and complaint["complaint_decision"] == "negative":
                sentiment_source = "taxonomy_and_complaint_model"
            elif explicit_negative:
                sentiment_source = "taxonomy_negative_evidence"
            else:
                sentiment_source = "complaint_model"
            confidence = float(complaint["prediction_confidence"])
            if explicit_negative:
                confidence = max(confidence, 0.90)
            records.append(
                {
                    "review_id": row["review_id"],
                    "canonical_place_id": row["canonical_place_id"],
                    "place_name": row["place_name"],
                    "place_category": row["place_category"],
                    "clause_index": int(row["clause_index"]),
                    "clause_text": row["clause_text"],
                    "aspect": aspect_id,
                    "aspect_probability": float(probability_row[f"aspect_probability_{aspect_id}"]),
                    "aspect_source": aspect_source,
                    "complaint_probability": float(complaint["complaint_probability"]),
                    "sentiment_label": sentiment_label,
                    "is_negative": bool(is_negative),
                    "prediction_confidence": confidence,
                    "sentiment_source": sentiment_source,
                    "label_source": complaint_bundle["label_source"],
                    "model_version": complaint_bundle["version"],
                }
            )
    output = pd.DataFrame(records)
    if output.empty:
        raise ValueError("Aspect-sentiment inference produced no aspect mentions.")
    output = output.sort_values(["review_id", "clause_index", "aspect"]).reset_index(drop=True)
    output.to_parquet(OUTPUT_PATH, index=False)

    review_sentiment_counts = output.groupby("review_id")["sentiment_label"].nunique()
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "unit_of_analysis": "review clause paired with one aspect",
        "label_source": complaint_bundle["label_source"],
        "model_version": complaint_bundle["version"],
        "aspect_label_source": aspect_metadata.get("label_source", "rule_based_weak"),
        "aspect_model_version": aspect_metadata.get("version", "v1-rule-label-baseline"),
        "aspect_probability_threshold": aspect_threshold,
        "review_rows_scanned": int(reviews["review_text_clean"].notna().sum()),
        "clause_rows": int(len(clauses)),
        "aspect_sentiment_rows": int(len(output)),
        "reviews_with_aspect_sentiment": int(output["review_id"].nunique()),
        "unique_aspects": int(output["aspect"].nunique()),
        "sentiment_counts": output["sentiment_label"].value_counts().to_dict(),
        "negative_rows": int(output["is_negative"].sum()),
        "mixed_sentiment_reviews": int((review_sentiment_counts > 1).sum()),
        "aspect_source_counts": output["aspect_source"].value_counts().to_dict(),
        "sentiment_source_counts": output["sentiment_source"].value_counts().to_dict(),
        "output": str(OUTPUT_PATH),
        "limitations": [
            "Clause segmentation is deterministic and may not resolve every complex sentence.",
            "Aspect labels remain weakly supervised until manual aspect annotations are available."
            if not aspect_uses_gold
            else "",
            "Complaint sentiment remains weak-label based until the human gold dataset meets readiness criteria."
            if complaint_bundle["label_source"] != "human_gold"
            else "",
        ],
    }
    summary["limitations"] = [item for item in summary["limitations"] if item]
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer clause-level aspect sentiment.")
    parser.parse_args()
    result = infer_aspect_sentiment()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
