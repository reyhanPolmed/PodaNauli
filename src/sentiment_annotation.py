"""Safe manual-annotation workflow for sentiment gold labels."""

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
from sklearn.metrics import cohen_kappa_score

from src.train_complaint import load_complaint_bundle, predict_with_abstention


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
ANNOTATION_DIR = ROOT / "data" / "annotations"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
QUEUE_PATH = ANNOTATION_DIR / "sentiment_annotation_queue.csv"
GOLD_PATH = ANNOTATION_DIR / "sentiment_gold.csv"
SUMMARY_PATH = REPORT_DIR / "sentiment_annotation_workflow.json"
AI_SUMMARY_PATH = REPORT_DIR / "ai_annotation_summary.json"
ERROR_PATH = REPORT_DIR / "sentiment_error_analysis.csv"
LEGACY_SAMPLE_PATH = REPORT_DIR / "sentiment_annotation_sample.csv"

MANUAL_COLUMNS = [
    "manual_sentiment_label",
    "second_annotator_label",
    "annotation_status",
    "annotator_id",
    "second_annotator_id",
    "annotation_confidence",
    "annotation_notes",
]
AI_COLUMNS = [
    "ai_suggested_sentiment_label",
    "ai_sentiment_model_prediction",
    "ai_sentiment_probability",
    "ai_complaint_probability",
    "ai_confidence",
    "ai_model_agreement",
    "ai_weak_label_agreement",
    "ai_rationale",
    "ai_model_version",
    "human_approved",
]
PRESERVED_COLUMNS = [*MANUAL_COLUMNS, *AI_COLUMNS]
VALID_LABELS = {"negative", "neutral", "positive"}
QUEUE_COLUMNS = [
    "review_id",
    "canonical_place_id",
    "place_name",
    "place_category",
    "reviewer_rating",
    "review_text_raw",
    "weak_sentiment_label",
    "text_length",
    "selection_reason",
    *AI_COLUMNS[:-1],
    *MANUAL_COLUMNS,
    "human_approved",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def normalize_manual_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _annotation_values_by_review(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if "review_id" not in frame.columns:
            continue
        for column in PRESERVED_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
        frames.append(frame[["review_id", *PRESERVED_COLUMNS]])
    if not frames:
        return pd.DataFrame(columns=["review_id", *PRESERVED_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("review_id", keep="last")
    return combined


def _select_with_place_cap(
    candidates: pd.DataFrame,
    target: int,
    max_per_place: int,
    existing_place_counts: dict[str, int],
) -> pd.DataFrame:
    selected_indexes: list[int] = []
    deferred_indexes: list[int] = []
    for index, row in candidates.iterrows():
        place_id = str(row["canonical_place_id"])
        if existing_place_counts.get(place_id, 0) < max_per_place:
            selected_indexes.append(index)
            existing_place_counts[place_id] = existing_place_counts.get(place_id, 0) + 1
        else:
            deferred_indexes.append(index)
        if len(selected_indexes) >= target:
            break
    if len(selected_indexes) < target:
        selected_indexes.extend(deferred_indexes[: target - len(selected_indexes)])
    return candidates.loc[selected_indexes[:target]]


def create_balanced_annotation_queue(
    reviews: pd.DataFrame,
    target_per_class: int,
    max_per_place: int,
    random_seed: int,
    error_review_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a deterministic, class-balanced queue with hard cases first."""
    required = {
        "review_id",
        "canonical_place_id",
        "place_name",
        "place_category",
        "reviewer_rating",
        "review_text_raw",
        "weak_sentiment_label",
        "text_length",
        "is_duplicate",
    }
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Missing review columns for annotation: {sorted(missing)}")

    candidates = reviews[
        reviews["review_text_raw"].notna()
        & reviews["weak_sentiment_label"].isin(VALID_LABELS)
        & (~reviews["is_duplicate"])
        & (reviews["text_length"] > 0)
    ].copy()
    candidates["selection_reason"] = "stratified_random"
    error_review_ids = error_review_ids or set()
    candidates.loc[candidates["review_id"].isin(error_review_ids), "selection_reason"] = "prior_model_error"
    candidates["is_hard_case"] = candidates["review_id"].isin(error_review_ids)
    candidates = candidates.sample(frac=1.0, random_state=random_seed)
    candidates = candidates.sort_values(
        ["weak_sentiment_label", "is_hard_case", "text_length"],
        ascending=[True, False, True],
        kind="stable",
    )

    selected = []
    place_counts: dict[str, int] = {}
    for label in ["negative", "neutral", "positive"]:
        label_candidates = candidates[candidates["weak_sentiment_label"] == label]
        take = min(target_per_class, len(label_candidates))
        selected.append(_select_with_place_cap(label_candidates, take, max_per_place, place_counts))
    queue = pd.concat(selected, ignore_index=True).drop_duplicates("review_id")
    queue = queue.sort_values(["weak_sentiment_label", "selection_reason", "review_id"]).reset_index(drop=True)
    queue["manual_sentiment_label"] = ""
    queue["second_annotator_label"] = ""
    queue["annotation_status"] = "pending"
    queue["annotator_id"] = ""
    queue["second_annotator_id"] = ""
    queue["annotation_confidence"] = ""
    queue["annotation_notes"] = ""
    for column in AI_COLUMNS:
        queue[column] = ""

    report = {
        "candidate_rows": int(len(candidates)),
        "queue_rows": int(len(queue)),
        "target_per_class": int(target_per_class),
        "class_counts": queue["weak_sentiment_label"].value_counts().to_dict(),
        "hard_case_rows": int((queue["selection_reason"] == "prior_model_error").sum()),
        "unique_places": int(queue["canonical_place_id"].nunique()),
        "random_seed": int(random_seed),
    }
    return queue[QUEUE_COLUMNS], report


def reuse_or_create_annotation_queue(
    reviews: pd.DataFrame,
    existing_queue: pd.DataFrame,
    target_per_class: int,
    max_per_place: int,
    random_seed: int,
    error_review_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep an existing queue stable once annotation has started."""
    if existing_queue.empty:
        return create_balanced_annotation_queue(
            reviews,
            target_per_class=target_per_class,
            max_per_place=max_per_place,
            random_seed=random_seed,
            error_review_ids=error_review_ids,
        )
    queue = existing_queue.copy()
    for column in QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    queue = queue[QUEUE_COLUMNS].drop_duplicates("review_id", keep="last").reset_index(drop=True)
    report = {
        "candidate_rows": int(len(reviews)),
        "queue_rows": int(len(queue)),
        "target_per_class": int(target_per_class),
        "class_counts": queue["weak_sentiment_label"].value_counts().to_dict(),
        "hard_case_rows": int((queue["selection_reason"] == "prior_model_error").sum()),
        "unique_places": int(queue["canonical_place_id"].nunique()),
        "random_seed": int(random_seed),
        "reused_existing_queue": True,
    }
    return queue, report


def preserve_existing_manual_values(queue: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Merge annotation fields by review_id so regenerating a queue never erases work."""
    if existing.empty:
        return queue
    for column in PRESERVED_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
        if column not in existing.columns:
            existing[column] = ""
    renamed = existing.rename(columns={column: f"{column}__existing" for column in PRESERVED_COLUMNS})
    merged = queue.merge(renamed, on="review_id", how="left")
    for column in PRESERVED_COLUMNS:
        existing_column = f"{column}__existing"
        existing_values = merged[existing_column].fillna("").astype(str)
        use_existing = existing_values.str.strip().ne("")
        merged.loc[use_existing, column] = existing_values[use_existing]
        merged = merged.drop(columns=[existing_column])
    return merged


def assert_gold_not_regressed(previous_gold: pd.DataFrame, next_gold: pd.DataFrame) -> None:
    """Prevent automatic queue preparation from silently discarding human gold."""
    if previous_gold.empty:
        return
    previous_ids = set(previous_gold["review_id"].astype(str))
    next_ids = set(next_gold["review_id"].astype(str))
    missing_ids = previous_ids - next_ids
    if missing_ids:
        raise ValueError(
            "Automatic annotation preparation would remove "
            f"{len(missing_ids)} existing gold rows. Use explicit validation or snapshot recovery."
        )


def derive_ai_suggestion(
    sentiment_prediction: str,
    sentiment_probability: float,
    complaint_decision: str,
    complaint_probability: float,
    weak_label: str,
    high_probability: float,
    medium_probability: float,
) -> dict[str, Any]:
    """Combine two text models while retaining explicit disagreement signals."""
    suggested = "negative" if complaint_decision == "negative" else sentiment_prediction
    binary_sentiment_prediction = "negative" if sentiment_prediction == "negative" else "non_negative"
    model_agreement = (
        complaint_decision != "uncertain"
        and binary_sentiment_prediction == complaint_decision
    )
    weak_agreement = suggested == weak_label
    if model_agreement and weak_agreement and sentiment_probability >= high_probability:
        confidence = "high"
    elif model_agreement and sentiment_probability >= medium_probability:
        confidence = "medium"
    else:
        confidence = "low"
    rationale = (
        f"sentiment_model={sentiment_prediction} (p={sentiment_probability:.3f}); "
        f"complaint_model={complaint_decision} (p={complaint_probability:.3f}); "
        f"model_agreement={'yes' if model_agreement else 'no'}; "
        f"weak_label={weak_label}; "
        f"weak_agreement={'yes' if weak_agreement else 'no'}"
    )
    return {
        "ai_suggested_sentiment_label": suggested,
        "ai_confidence": confidence,
        "ai_model_agreement": "yes" if model_agreement else "no",
        "ai_weak_label_agreement": "yes" if weak_agreement else "no",
        "ai_rationale": rationale,
    }


def generate_ai_suggestions(
    queue: pd.DataFrame,
    texts: pd.Series,
    sentiment_model: Any,
    complaint_bundle: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Populate silver-label suggestions without modifying human label fields."""
    if len(queue) != len(texts):
        raise ValueError("Annotation queue and text input must have the same row count.")
    ai_config = config.get("sentiment_improvement", {}).get("ai_annotation", {})
    high_probability = float(ai_config.get("high_confidence_probability", 0.75))
    medium_probability = float(ai_config.get("medium_confidence_probability", 0.55))

    probabilities = np.asarray(sentiment_model.predict_proba(texts))
    classes = [str(label) for label in sentiment_model.classes_]
    best_indexes = probabilities.argmax(axis=1)
    sentiment_predictions = [classes[index] for index in best_indexes]
    best_probabilities = probabilities[np.arange(len(probabilities)), best_indexes]
    complaint_predictions = predict_with_abstention(complaint_bundle, texts).reset_index(drop=True)

    output = queue.copy().reset_index(drop=True)
    approval_values = {
        str(value).strip().lower()
        for value in config.get("sentiment_improvement", {})
        .get("ai_annotation", {})
        .get("human_approval_values", ["yes", "true", "1", "approved"])
    }
    approved_mask = (
        output["human_approved"].astype(str).str.strip().str.lower().isin(approval_values)
        if "human_approved" in output.columns
        else pd.Series(False, index=output.index)
    )
    suggestions = []
    for index, row in output.iterrows():
        suggestions.append(
            derive_ai_suggestion(
                sentiment_prediction=sentiment_predictions[index],
                sentiment_probability=float(best_probabilities[index]),
                complaint_decision=str(complaint_predictions.loc[index, "complaint_decision"]),
                complaint_probability=float(complaint_predictions.loc[index, "complaint_probability"]),
                weak_label=str(row["weak_sentiment_label"]),
                high_probability=high_probability,
                medium_probability=medium_probability,
            )
        )
    pending_mask = ~approved_mask
    if pending_mask.any():
        suggestion_frame = pd.DataFrame(suggestions)
        for column in suggestion_frame.columns:
            output.loc[pending_mask, column] = suggestion_frame.loc[pending_mask, column]
        output.loc[pending_mask, "ai_sentiment_model_prediction"] = pd.Series(sentiment_predictions)[
            pending_mask
        ]
        for column in ("ai_sentiment_probability", "ai_complaint_probability"):
            if column not in output.columns:
                output[column] = np.nan
            output[column] = output[column].astype(object)
        output.loc[pending_mask, "ai_sentiment_probability"] = best_probabilities[pending_mask]
        output.loc[pending_mask, "ai_complaint_probability"] = complaint_predictions.loc[
            pending_mask,
            "complaint_probability",
        ]
        output.loc[pending_mask, "ai_model_version"] = f"sentiment_champion+{complaint_bundle['version']}"
    if "human_approved" not in output.columns:
        output["human_approved"] = ""
    return output


def suggest_annotation_queue() -> dict[str, Any]:
    """Run both current text models and write AI-assisted silver suggestions."""
    if not QUEUE_PATH.exists():
        prepare_annotation_queue()
    queue = pd.read_csv(QUEUE_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    reviews = pd.read_parquet(
        PROCESSED_DIR / "reviews_clean.parquet",
        columns=["review_id", "review_text_clean"],
    )
    merged = queue.merge(reviews, on="review_id", how="left", validate="one_to_one")
    texts = merged["review_text_clean"].fillna(merged["review_text_raw"]).astype(str)
    sentiment_model = joblib.load(MODEL_DIR / "sentiment_champion.joblib")
    complaint_bundle = load_complaint_bundle()
    config = load_config()
    suggested = generate_ai_suggestions(
        queue=queue,
        texts=texts,
        sentiment_model=sentiment_model,
        complaint_bundle=complaint_bundle,
        config=config,
    )
    suggested.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "label_type": "AI-assisted silver labels; not human gold.",
        "queue_rows": int(len(suggested)),
        "suggested_rows": int(suggested["ai_suggested_sentiment_label"].astype(str).str.strip().ne("").sum()),
        "skipped_human_approved_rows": int(
            suggested["human_approved"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"yes", "true", "1", "approved"})
            .sum()
        ),
        "suggestion_counts": suggested["ai_suggested_sentiment_label"].value_counts().to_dict(),
        "confidence_counts": suggested["ai_confidence"].value_counts().to_dict(),
        "model_agreement_counts": suggested["ai_model_agreement"].value_counts().to_dict(),
        "weak_label_agreement_counts": suggested["ai_weak_label_agreement"].value_counts().to_dict(),
        "human_approved_rows": int(
            suggested["human_approved"].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "approved"}).sum()
        ),
        "manual_label_rows": int(suggested["manual_sentiment_label"].astype(str).str.strip().ne("").sum()),
        "gold_rows": int(len(pd.read_csv(GOLD_PATH, encoding="utf-8-sig"))) if GOLD_PATH.exists() else 0,
        "model_version": f"sentiment_champion+{complaint_bundle['version']}",
        "output": str(QUEUE_PATH),
    }
    AI_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_annotation_frame(
    queue: pd.DataFrame,
    approval_values: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate labels and export only non-conflicting human annotations as gold."""
    validated = queue.copy()
    for column in MANUAL_COLUMNS:
        if column not in validated.columns:
            validated[column] = ""
    approval_values = approval_values or {"yes", "true", "1", "approved"}
    approval_required = "human_approved" in validated.columns
    if "human_approved" not in validated.columns:
        validated["human_approved"] = ""
    primary = validated["manual_sentiment_label"].map(normalize_manual_label)
    secondary = validated["second_annotator_label"].map(normalize_manual_label)
    approved = validated["human_approved"].astype(str).str.strip().str.lower().isin(approval_values)
    if not approval_required:
        approved = pd.Series(True, index=validated.index)
    primary_valid = primary.isin(VALID_LABELS)
    secondary_filled = secondary.ne("")
    secondary_valid = secondary.isin(VALID_LABELS)
    invalid_mask = (primary.ne("") & ~primary_valid) | (secondary_filled & ~secondary_valid)
    conflict_mask = primary_valid & secondary_valid & primary.ne(secondary)
    completed_mask = primary_valid & approved & ~conflict_mask & ~invalid_mask
    awaiting_approval_mask = primary_valid & ~approved & ~conflict_mask & ~invalid_mask

    validated["manual_sentiment_label"] = primary
    validated["second_annotator_label"] = secondary
    validated.loc[invalid_mask, "annotation_status"] = "invalid_label"
    validated.loc[conflict_mask, "annotation_status"] = "adjudication_required"
    validated.loc[completed_mask, "annotation_status"] = "completed"
    validated.loc[awaiting_approval_mask, "annotation_status"] = "awaiting_human_approval"
    validated.loc[~primary_valid & ~invalid_mask, "annotation_status"] = "pending"

    gold_columns = [
        "review_id",
        "canonical_place_id",
        "place_name",
        "place_category",
        "reviewer_rating",
        "review_text_raw",
        "weak_sentiment_label",
        "manual_sentiment_label",
        "annotation_confidence",
        "annotation_notes",
        "annotator_id",
        "human_approved",
    ]
    gold = validated.loc[completed_mask, gold_columns].copy()
    overlap = primary_valid & secondary_valid
    kappa = None
    if int(overlap.sum()) >= 2 and primary[overlap].nunique() > 1 and secondary[overlap].nunique() > 1:
        kappa = float(cohen_kappa_score(primary[overlap], secondary[overlap], labels=sorted(VALID_LABELS)))
    report = {
        "queue_rows": int(len(validated)),
        "completed_gold_rows": int(len(gold)),
        "pending_rows": int((validated["annotation_status"] == "pending").sum()),
        "awaiting_human_approval_rows": int(
            (validated["annotation_status"] == "awaiting_human_approval").sum()
        ),
        "invalid_rows": int(invalid_mask.sum()),
        "adjudication_required_rows": int(conflict_mask.sum()),
        "double_annotated_rows": int(overlap.sum()),
        "cohen_kappa": kappa,
        "gold_class_counts": gold["manual_sentiment_label"].value_counts().to_dict(),
    }
    return gold, report


def prepare_annotation_queue() -> dict[str, Any]:
    config = load_config()
    random_seed = int(config.get("random_seed", 42))
    annotation_config = config.get("sentiment_improvement", {}).get("annotation", {})
    target_per_class = int(annotation_config.get("target_per_class", 300))
    max_per_place = int(annotation_config.get("max_per_place", 12))

    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    previous_gold = (
        pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if GOLD_PATH.exists()
        else pd.DataFrame()
    )
    error_ids: set[str] = set()
    if ERROR_PATH.exists():
        errors = pd.read_csv(ERROR_PATH, usecols=["review_id"], encoding="utf-8-sig")
        error_ids = set(errors["review_id"].astype(str))
    existing_queue = (
        pd.read_csv(QUEUE_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if QUEUE_PATH.exists()
        else pd.DataFrame()
    )
    queue, queue_report = reuse_or_create_annotation_queue(
        reviews,
        existing_queue=existing_queue,
        target_per_class=target_per_class,
        max_per_place=max_per_place,
        random_seed=random_seed,
        error_review_ids=error_ids,
    )
    existing = _annotation_values_by_review([LEGACY_SAMPLE_PATH])
    queue = preserve_existing_manual_values(queue, existing)
    approval_values = {
        str(value).strip().lower()
        for value in config.get("sentiment_improvement", {})
        .get("ai_annotation", {})
        .get("human_approval_values", ["yes", "true", "1", "approved"])
    }
    gold, validation_report = validate_annotation_frame(queue, approval_values=approval_values)
    assert_gold_not_regressed(previous_gold, gold)

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")
    gold.to_csv(GOLD_PATH, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "label_policy": "Human labels only become gold; weak rating labels remain audit context.",
        "queue": queue_report,
        "validation": validation_report,
        "readiness": {
            "minimum_gold_rows": int(annotation_config.get("minimum_gold_rows", 300)),
            "minimum_gold_rows_per_class": int(annotation_config.get("minimum_gold_rows_per_class", 50)),
            "gold_training_ready": False,
        },
        "outputs": {
            "annotation_queue": str(QUEUE_PATH),
            "gold_labels": str(GOLD_PATH),
            "workflow_summary": str(SUMMARY_PATH),
        },
    }
    class_counts = validation_report["gold_class_counts"]
    summary["readiness"]["gold_training_ready"] = bool(
        validation_report["completed_gold_rows"] >= summary["readiness"]["minimum_gold_rows"]
        and all(
            int(class_counts.get(label, 0)) >= summary["readiness"]["minimum_gold_rows_per_class"]
            for label in VALID_LABELS
        )
    )
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_annotation_queue() -> dict[str, Any]:
    if not QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing annotation queue: {QUEUE_PATH}")
    queue = pd.read_csv(QUEUE_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    config = load_config()
    approval_values = {
        str(value).strip().lower()
        for value in config.get("sentiment_improvement", {})
        .get("ai_annotation", {})
        .get("human_approval_values", ["yes", "true", "1", "approved"])
    }
    gold, validation_report = validate_annotation_frame(queue, approval_values=approval_values)
    try:
        queue.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")
    except PermissionError:
        with open(QUEUE_PATH, "r+", encoding="utf-8-sig") as f:
            f.seek(0)
            f.write(queue.to_csv(index=False, encoding="utf-8-sig"))
            f.truncate()
    try:
        gold.to_csv(GOLD_PATH, index=False, encoding="utf-8-sig")
    except PermissionError:
        with open(GOLD_PATH, "r+", encoding="utf-8-sig") as f:
            f.seek(0)
            f.write(gold.to_csv(index=False, encoding="utf-8-sig"))
            f.truncate()
    annotation_config = config.get("sentiment_improvement", {}).get("annotation", {})
    minimum_rows = int(annotation_config.get("minimum_gold_rows", 300))
    minimum_per_class = int(annotation_config.get("minimum_gold_rows_per_class", 50))
    class_counts = validation_report["gold_class_counts"]
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "validation": validation_report,
        "readiness": {
            "minimum_gold_rows": minimum_rows,
            "minimum_gold_rows_per_class": minimum_per_class,
            "gold_training_ready": bool(
                validation_report["completed_gold_rows"] >= minimum_rows
                and all(int(class_counts.get(label, 0)) >= minimum_per_class for label in VALID_LABELS)
            ),
        },
        "outputs": {
            "annotation_queue": str(QUEUE_PATH),
            "gold_labels": str(GOLD_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def restore_annotation_queue_from_gold_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """Recover a queue from an immutable gold snapshot and archived weak models."""
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing gold snapshot: {snapshot_path}")
    weak_sentiment_path = MODEL_DIR / "archive" / "sentiment_champion_v1_weak.joblib"
    weak_complaint_path = MODEL_DIR / "archive" / "complaint_detector_v2_weak.joblib"
    if not weak_sentiment_path.exists() or not weak_complaint_path.exists():
        raise FileNotFoundError("Archived weak sentiment and complaint models are required for recovery.")

    snapshot = pd.read_csv(
        snapshot_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    required = {
        "review_id",
        "canonical_place_id",
        "place_name",
        "place_category",
        "reviewer_rating",
        "review_text_raw",
        "weak_sentiment_label",
        "manual_sentiment_label",
        "annotator_id",
        "human_approved",
    }
    missing = required - set(snapshot.columns)
    if missing:
        raise ValueError(f"Gold snapshot is missing required columns: {sorted(missing)}")
    if snapshot["review_id"].duplicated().any():
        raise ValueError("Gold snapshot contains duplicate review_id values.")

    reviews = pd.read_parquet(
        PROCESSED_DIR / "reviews_clean.parquet",
        columns=["review_id", "text_length"],
    )
    queue = snapshot.merge(reviews, on="review_id", how="left", validate="one_to_one")
    if queue["text_length"].isna().any():
        raise ValueError("Gold snapshot contains review IDs missing from reviews_clean.parquet.")
    queue["selection_reason"] = "restored_human_gold"
    queue["second_annotator_label"] = ""
    queue["annotation_status"] = "completed"
    queue["second_annotator_id"] = ""
    for column in AI_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    for column in MANUAL_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    queue = queue[QUEUE_COLUMNS]

    manual_values = queue[["review_id", *PRESERVED_COLUMNS]].copy()
    queue["human_approved"] = ""
    config = load_config()
    queue = generate_ai_suggestions(
        queue=queue,
        texts=queue["review_text_raw"].astype(str),
        sentiment_model=joblib.load(weak_sentiment_path),
        complaint_bundle=load_complaint_bundle(weak_complaint_path),
        config=config,
    )
    queue = preserve_existing_manual_values(queue, manual_values)
    approval_values = {
        str(value).strip().lower()
        for value in config.get("sentiment_improvement", {})
        .get("ai_annotation", {})
        .get("human_approval_values", ["yes", "true", "1", "approved"])
    }
    gold, validation_report = validate_annotation_frame(queue, approval_values=approval_values)
    if len(gold) != len(snapshot):
        raise ValueError(
            f"Recovery produced {len(gold)} gold rows from a {len(snapshot)}-row snapshot."
        )

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")
    gold.to_csv(GOLD_PATH, index=False, encoding="utf-8-sig")
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_snapshot": str(snapshot_path),
        "recovered_queue_rows": int(len(queue)),
        "recovered_gold_rows": int(len(gold)),
        "gold_class_counts": gold["manual_sentiment_label"].value_counts().to_dict(),
        "ai_model_version": queue["ai_model_version"].value_counts().to_dict(),
        "validation": validation_report,
        "outputs": {
            "annotation_queue": str(QUEUE_PATH),
            "gold_labels": str(GOLD_PATH),
        },
    }
    (REPORT_DIR / "annotation_recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, suggest, validate, or restore sentiment annotations."
    )
    parser.add_argument(
        "action",
        choices=["prepare", "suggest", "validate", "restore"],
        nargs="?",
        default="prepare",
    )
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_annotation_queue()
    elif args.action == "suggest":
        result = suggest_annotation_queue()
    elif args.action == "validate":
        result = validate_annotation_queue()
    else:
        if args.snapshot is None:
            parser.error("--snapshot is required for restore")
        result = restore_annotation_queue_from_gold_snapshot(args.snapshot)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
