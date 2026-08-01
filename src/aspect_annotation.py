"""Clause-level multi-label aspect annotation workflow."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.aspect_sentiment import build_clause_frame
from src.train_aspect import (
    detect_weak_aspects,
    load_aspect_taxonomy,
    predict_aspect_probabilities,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
ANNOTATION_DIR = ROOT / "data" / "annotations"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
QUEUE_PATH = ANNOTATION_DIR / "aspect_clause_annotation_queue.csv"
QUEUE_PENDING_PATH = ANNOTATION_DIR / "aspect_clause_annotation_queue.pending.csv"
GOLD_PATH = ANNOTATION_DIR / "aspect_gold.csv"
SUMMARY_PATH = REPORT_DIR / "aspect_annotation_workflow.json"
AI_SUMMARY_PATH = REPORT_DIR / "aspect_ai_annotation_summary.json"
MODEL_METADATA_PATH = MODEL_DIR / "aspect_metadata.json"

SPECIAL_NONE = "none"
MANUAL_COLUMNS = [
    "manual_aspects",
    "annotation_status",
    "annotator_id",
    "human_approved",
    "annotation_notes",
]
AI_COLUMNS = [
    "ai_suggested_aspects",
    "ai_aspect_probabilities",
    "ai_model_version",
]
QUEUE_COLUMNS = [
    "clause_id",
    "review_id",
    "canonical_place_id",
    "place_name",
    "place_category",
    "clause_index",
    "clause_text",
    "weak_aspects",
    "selection_reason",
    *AI_COLUMNS,
    *MANUAL_COLUMNS,
]
GOLD_COLUMNS = [
    "clause_id",
    "review_id",
    "canonical_place_id",
    "place_name",
    "place_category",
    "clause_index",
    "clause_text",
    "weak_aspects",
    "manual_aspects",
    "annotator_id",
    "human_approved",
    "annotation_notes",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"random_seed": 42}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def aspect_ids() -> list[str]:
    return [str(item["id"]) for item in load_aspect_taxonomy()]


def normalize_aspect_labels(value: Any, valid_aspects: set[str]) -> tuple[list[str], str | None]:
    """Normalize a pipe-separated multi-label value and enforce exclusive none."""
    if pd.isna(value) or not str(value).strip():
        return [], None
    raw = str(value).strip().lower().replace(",", "|").replace(";", "|")
    labels = list(dict.fromkeys(part.strip() for part in raw.split("|") if part.strip()))
    invalid = sorted(set(labels) - (valid_aspects | {SPECIAL_NONE}))
    if invalid:
        return [], f"invalid labels: {', '.join(invalid)}"
    if SPECIAL_NONE in labels and len(labels) > 1:
        return [], "none must be the only label"
    if labels == [SPECIAL_NONE]:
        return labels, None
    order = {label: index for index, label in enumerate(aspect_ids())}
    return sorted(labels, key=lambda label: order[label]), None


def build_clause_candidates(reviews: pd.DataFrame) -> pd.DataFrame:
    """Create stable clause IDs and rule-based silver aspect labels."""
    taxonomy = load_aspect_taxonomy()
    clauses = build_clause_frame(reviews)
    clauses["clause_id"] = clauses.apply(
        lambda row: f"{row['review_id']}::c{int(row['clause_index']):03d}",
        axis=1,
    )
    detections = clauses["clause_text"].map(lambda text: detect_weak_aspects(text, taxonomy))
    clauses["weak_aspect_list"] = detections.map(lambda item: item["weak_aspects"])
    clauses["weak_aspects"] = clauses["weak_aspect_list"].map(lambda values: "|".join(values))
    return clauses


def create_clause_annotation_queue(
    candidates: pd.DataFrame,
    target_rows: int,
    max_per_place: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a deterministic aspect-aware clause sample with a place cap."""
    required = {
        "clause_id",
        "review_id",
        "canonical_place_id",
        "place_name",
        "place_category",
        "clause_index",
        "clause_text",
        "weak_aspects",
        "weak_aspect_list",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing clause candidate columns: {sorted(missing)}")
    target_rows = min(int(target_rows), len(candidates))
    shuffled = candidates.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    shuffled["_selection_reason"] = ""
    labels = [label for label in aspect_ids() if label != "lainnya"]
    quota = max(1, target_rows // (len(labels) + 1))
    selected: list[int] = []
    selected_set: set[int] = set()
    place_counts: dict[str, int] = {}

    def take_rows(mask: pd.Series, reason: str, target: int) -> None:
        taken = 0
        for index, row in shuffled[mask].iterrows():
            if index in selected_set:
                continue
            place_id = str(row["canonical_place_id"])
            if place_counts.get(place_id, 0) >= max_per_place:
                continue
            selected.append(index)
            selected_set.add(index)
            place_counts[place_id] = place_counts.get(place_id, 0) + 1
            shuffled.at[index, "_selection_reason"] = reason
            taken += 1
            if taken >= target or len(selected) >= target_rows:
                break

    for label in labels:
        take_rows(
            shuffled["weak_aspect_list"].map(lambda values, item=label: item in values),
            f"weak_aspect:{label}",
            quota,
        )
    take_rows(
        shuffled["weak_aspect_list"].map(lambda values: values == ["lainnya"]),
        "taxonomy_unmatched",
        quota,
    )
    take_rows(pd.Series(True, index=shuffled.index), "stratified_fill", target_rows)

    queue = shuffled.loc[selected[:target_rows]].copy()
    queue["selection_reason"] = queue["_selection_reason"]
    queue["ai_suggested_aspects"] = ""
    queue["ai_aspect_probabilities"] = ""
    queue["ai_model_version"] = ""
    queue["manual_aspects"] = ""
    queue["annotation_status"] = "pending"
    queue["annotator_id"] = ""
    queue["human_approved"] = ""
    queue["annotation_notes"] = ""
    queue = queue[QUEUE_COLUMNS].sort_values(
        ["selection_reason", "canonical_place_id", "clause_id"]
    ).reset_index(drop=True)
    report = {
        "candidate_clauses": int(len(candidates)),
        "queue_rows": int(len(queue)),
        "target_rows": int(target_rows),
        "unique_reviews": int(queue["review_id"].nunique()),
        "unique_places": int(queue["canonical_place_id"].nunique()),
        "selection_reason_counts": queue["selection_reason"].value_counts().to_dict(),
        "random_seed": int(random_seed),
    }
    return queue, report


def validate_aspect_annotation_frame(
    queue: pd.DataFrame,
    approval_values: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate approved human labels and export only valid gold rows."""
    output = queue.copy()
    valid_aspects = set(aspect_ids())
    approval_values = approval_values or {"yes", "true", "1", "approved"}
    if "human_approved" not in output:
        output["human_approved"] = ""
    normalized_values: list[str] = []
    errors: list[str] = []
    for value in output["manual_aspects"]:
        labels, error = normalize_aspect_labels(value, valid_aspects)
        normalized_values.append("|".join(labels))
        errors.append(error or "")
    output["manual_aspects"] = normalized_values
    output["_validation_error"] = errors
    approved = output["human_approved"].astype(str).str.strip().str.lower().isin(approval_values)
    has_label = output["manual_aspects"].astype(str).str.strip().ne("")
    valid = output["_validation_error"].eq("")
    output["annotation_status"] = "pending"
    output.loc[has_label & ~approved & valid, "annotation_status"] = "awaiting_human_approval"
    output.loc[~valid, "annotation_status"] = "invalid"
    output.loc[has_label & approved & valid, "annotation_status"] = "completed"
    gold = output[has_label & approved & valid].copy()
    gold = gold[GOLD_COLUMNS].sort_values("clause_id").reset_index(drop=True)

    label_counts: dict[str, int] = {label: 0 for label in [*aspect_ids(), SPECIAL_NONE]}
    for value in gold["manual_aspects"]:
        for label in str(value).split("|"):
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
    report = {
        "queue_rows": int(len(output)),
        "completed_gold_rows": int(len(gold)),
        "pending_rows": int((output["annotation_status"] == "pending").sum()),
        "awaiting_human_approval_rows": int(
            (output["annotation_status"] == "awaiting_human_approval").sum()
        ),
        "invalid_rows": int((output["annotation_status"] == "invalid").sum()),
        "gold_label_counts": label_counts,
        "gold_unique_places": int(gold["canonical_place_id"].nunique()) if not gold.empty else 0,
    }
    return gold, report


def _assert_gold_not_regressed(previous_gold: pd.DataFrame, next_gold: pd.DataFrame) -> None:
    if previous_gold.empty:
        return
    missing = set(previous_gold["clause_id"].astype(str)) - set(next_gold["clause_id"].astype(str))
    if missing:
        raise ValueError(
            "Automatic aspect queue preparation would remove "
            f"{len(missing)} existing gold clauses."
        )


def _write_queue(queue: pd.DataFrame) -> tuple[str, str]:
    try:
        queue.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")
        return "updated", str(QUEUE_PATH)
    except PermissionError:
        queue.to_csv(QUEUE_PENDING_PATH, index=False, encoding="utf-8-sig")
        return "primary_file_locked_pending_written", str(QUEUE_PENDING_PATH)


def prepare_aspect_annotation_queue() -> dict[str, Any]:
    config = load_config()
    annotation_config = config.get("aspect_improvement", {}).get("annotation", {})
    random_seed = int(config.get("random_seed", 42))
    target_rows = int(annotation_config.get("target_rows", 1200))
    max_per_place = int(annotation_config.get("max_per_place", 12))
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    previous_gold = (
        pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if GOLD_PATH.exists()
        else pd.DataFrame()
    )
    if QUEUE_PATH.exists():
        queue = pd.read_csv(
            QUEUE_PATH,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        for column in QUEUE_COLUMNS:
            if column not in queue:
                queue[column] = ""
        queue = queue[QUEUE_COLUMNS].drop_duplicates("clause_id", keep="last")
        queue_report = {
            "queue_rows": int(len(queue)),
            "unique_reviews": int(queue["review_id"].nunique()),
            "unique_places": int(queue["canonical_place_id"].nunique()),
            "reused_existing_queue": True,
            "random_seed": random_seed,
        }
    else:
        candidates = build_clause_candidates(reviews)
        queue, queue_report = create_clause_annotation_queue(
            candidates,
            target_rows=target_rows,
            max_per_place=max_per_place,
            random_seed=random_seed,
        )
    approval_values = {
        str(value).strip().lower()
        for value in config.get("sentiment_improvement", {})
        .get("ai_annotation", {})
        .get("human_approval_values", ["yes", "true", "1", "approved"])
    }
    gold, validation = validate_aspect_annotation_frame(queue, approval_values)
    _assert_gold_not_regressed(previous_gold, gold)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    queue_write_status, queue_output = _write_queue(queue)
    gold.to_csv(GOLD_PATH, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "unit_of_annotation": "review clause",
        "label_policy": (
            "manual_aspects is pipe-separated multi-label; none is exclusive; "
            "lainnya means an aspect outside the taxonomy."
        ),
        "queue": queue_report,
        "validation": validation,
        "queue_write_status": queue_write_status,
        "outputs": {
            "annotation_queue": queue_output,
            "gold_labels": str(GOLD_PATH),
            "workflow_summary": str(SUMMARY_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def suggest_aspect_annotation_queue() -> dict[str, Any]:
    if not QUEUE_PATH.exists():
        prepare_aspect_annotation_queue()
    queue = pd.read_csv(QUEUE_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    model = joblib.load(MODEL_DIR / "aspect_champion.joblib")
    mlb = joblib.load(MODEL_DIR / "aspect_multilabel_binarizer.joblib")
    metadata = (
        json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        if MODEL_METADATA_PATH.exists()
        else {}
    )
    threshold = float(metadata.get("threshold", 0.50))
    probabilities = predict_aspect_probabilities(model, queue["clause_text"], list(mlb.classes_))
    approval_values = {"yes", "true", "1", "approved"}
    approved = queue["human_approved"].astype(str).str.strip().str.lower().isin(approval_values)
    pending = ~approved
    if pending.any():
        for index in queue.index[pending]:
            scores = {
                str(label): float(probabilities.loc[index, f"aspect_probability_{label}"])
                for label in mlb.classes_
            }
            selected = [
                label
                for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
                if score >= threshold
            ][:3]
            queue.at[index, "ai_suggested_aspects"] = "|".join(selected or [SPECIAL_NONE])
            queue.at[index, "ai_aspect_probabilities"] = json.dumps(
                dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]),
                ensure_ascii=False,
            )
            queue.at[index, "ai_model_version"] = str(
                metadata.get("version", "v1-rule-label-baseline")
            )
    if pending.any():
        queue_write_status, queue_output = _write_queue(queue)
    else:
        queue_write_status = "unchanged_all_rows_human_approved"
        queue_output = str(QUEUE_PATH)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "label_type": "AI-assisted silver labels; not human gold.",
        "queue_rows": int(len(queue)),
        "suggested_rows": int(queue["ai_suggested_aspects"].astype(str).str.strip().ne("").sum()),
        "skipped_human_approved_rows": int(approved.sum()),
        "model_version": metadata.get("version", "v1-rule-label-baseline"),
        "threshold": threshold,
        "queue_write_status": queue_write_status,
        "output": queue_output,
    }
    AI_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_aspect_annotation_queue() -> dict[str, Any]:
    if not QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing aspect annotation queue: {QUEUE_PATH}")
    queue = pd.read_csv(QUEUE_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    gold, validation = validate_aspect_annotation_frame(queue)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    queue_write_status, queue_output = _write_queue(queue)
    gold.to_csv(GOLD_PATH, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "validation": validation,
        "queue_write_status": queue_write_status,
        "outputs": {
            "annotation_queue": queue_output,
            "gold_labels": str(GOLD_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage clause-level aspect annotations.")
    parser.add_argument(
        "action",
        choices=["prepare", "suggest", "validate"],
        nargs="?",
        default="prepare",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_aspect_annotation_queue()
    elif args.action == "suggest":
        result = suggest_aspect_annotation_queue()
    else:
        result = validate_aspect_annotation_queue()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
