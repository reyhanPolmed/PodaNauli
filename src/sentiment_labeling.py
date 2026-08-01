"""Tahap 3 pipeline: weak sentiment labels and annotation audit sample."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.preprocessing import weak_sentiment_from_rating


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

ANNOTATION_COLUMNS = [
    "review_id",
    "place_name",
    "reviewer_rating",
    "review_text_raw",
    "weak_sentiment_label",
    "manual_sentiment_label",
    "annotation_notes",
]


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        return {"random_seed": 42}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def rating_class(rating: Any) -> str:
    if pd.isna(rating):
        return "rating_missing"
    return f"rating_{int(round(float(rating)))}"


def text_length_bucket(length: Any) -> str:
    if pd.isna(length) or int(length) <= 0:
        return "empty"
    length = int(length)
    if length <= 30:
        return "very_short"
    if length <= 100:
        return "short"
    if length <= 250:
        return "medium"
    return "long"


def ensure_weak_sentiment_labels(reviews: pd.DataFrame) -> pd.DataFrame:
    """Recompute weak labels from rating so Tahap 3 is independently auditable."""
    labeled = reviews.copy()
    labeled["weak_sentiment_label"] = labeled["reviewer_rating"].map(weak_sentiment_from_rating)
    return labeled


def _target_group_sizes(group_sizes: pd.Series, sample_size: int) -> dict[str, int]:
    proportions = group_sizes / group_sizes.sum()
    raw_targets = (proportions * sample_size).round().astype(int)
    raw_targets[raw_targets == 0] = 1

    while raw_targets.sum() > sample_size:
        largest = raw_targets.sort_values(ascending=False).index[0]
        if raw_targets[largest] <= 1:
            break
        raw_targets[largest] -= 1

    while raw_targets.sum() < sample_size:
        remaining_capacity = group_sizes - raw_targets
        eligible = remaining_capacity[remaining_capacity > 0]
        if eligible.empty:
            break
        raw_targets[eligible.sort_values(ascending=False).index[0]] += 1

    return {str(key): int(value) for key, value in raw_targets.items()}


def create_annotation_sample(
    reviews: pd.DataFrame,
    sample_size: int = 300,
    random_seed: int = 42,
    max_per_place: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a deterministic annotation sample with practical stratification."""
    required = {"review_id", "place_name", "reviewer_rating", "review_text_raw", "weak_sentiment_label"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Missing required review columns: {sorted(missing)}")

    candidates = reviews[
        reviews["weak_sentiment_label"].notna()
        & reviews["review_text_raw"].notna()
        & (reviews["text_length"] > 0)
        & (~reviews["is_duplicate"])
    ].copy()
    if candidates.empty:
        candidates = reviews[
            reviews["weak_sentiment_label"].notna()
            & reviews["review_text_raw"].notna()
            & (reviews["text_length"] > 0)
        ].copy()

    target_size = min(sample_size, len(candidates))
    candidates["rating_class"] = candidates["reviewer_rating"].map(rating_class)
    candidates["text_length_bucket"] = candidates["text_length"].map(text_length_bucket)
    candidates["stratum"] = (
        candidates["weak_sentiment_label"].astype(str)
        + "|"
        + candidates["rating_class"].astype(str)
        + "|"
        + candidates["place_category"].astype(str)
        + "|"
        + candidates["text_length_bucket"].astype(str)
    )
    candidates = candidates.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    group_sizes = candidates["stratum"].value_counts()
    targets = _target_group_sizes(group_sizes, target_size)
    selected_indexes: list[int] = []
    place_counts: dict[str, int] = {}

    for stratum, target in targets.items():
        group = candidates[candidates["stratum"] == stratum]
        taken = 0
        deferred = []
        for idx, row in group.iterrows():
            place_id = str(row.get("canonical_place_id") or row["place_name"])
            if place_counts.get(place_id, 0) < max_per_place:
                selected_indexes.append(idx)
                place_counts[place_id] = place_counts.get(place_id, 0) + 1
                taken += 1
            else:
                deferred.append(idx)
            if taken >= target:
                break
        if taken < target:
            for idx in deferred[: target - taken]:
                selected_indexes.append(idx)

    if len(selected_indexes) < target_size:
        selected_set = set(selected_indexes)
        for idx, row in candidates.iterrows():
            if idx in selected_set:
                continue
            place_id = str(row.get("canonical_place_id") or row["place_name"])
            if place_counts.get(place_id, 0) >= max_per_place:
                continue
            selected_indexes.append(idx)
            place_counts[place_id] = place_counts.get(place_id, 0) + 1
            if len(selected_indexes) >= target_size:
                break

    if len(selected_indexes) < target_size:
        selected_set = set(selected_indexes)
        for idx in candidates.index:
            if idx not in selected_set:
                selected_indexes.append(idx)
            if len(selected_indexes) >= target_size:
                break

    sample = candidates.loc[selected_indexes[:target_size]].copy()
    sample = sample.sort_values(
        ["weak_sentiment_label", "place_category", "rating_class", "text_length_bucket", "place_name", "review_id"]
    ).reset_index(drop=True)
    sample["manual_sentiment_label"] = ""
    sample["annotation_notes"] = ""

    report = {
        "candidate_rows": int(len(candidates)),
        "sample_size": int(len(sample)),
        "sample_size_requested": int(sample_size),
        "random_seed": int(random_seed),
        "max_per_place": int(max_per_place),
        "stratification_columns": [
            "weak_sentiment_label",
            "rating_class",
            "place_category",
            "text_length_bucket",
            "canonical_place_id",
        ],
        "weak_sentiment_distribution": sample["weak_sentiment_label"].value_counts().to_dict(),
        "rating_class_distribution": sample["rating_class"].value_counts().to_dict(),
        "place_category_distribution": sample["place_category"].value_counts().to_dict(),
        "text_length_bucket_distribution": sample["text_length_bucket"].value_counts().to_dict(),
        "unique_places_in_sample": int(sample["canonical_place_id"].nunique(dropna=True)),
    }
    return sample[ANNOTATION_COLUMNS], report


def preserve_completed_annotation_sample(sample: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Preserve human-entered fields when the deterministic sample is regenerated."""
    if not existing_path.exists():
        return sample
    existing = pd.read_csv(existing_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "review_id" not in existing.columns:
        return sample
    manual_columns = ["manual_sentiment_label", "annotation_notes"]
    available = ["review_id", *[column for column in manual_columns if column in existing.columns]]
    existing = existing[available].drop_duplicates("review_id", keep="last")
    renamed = existing.rename(columns={column: f"{column}__existing" for column in manual_columns if column in existing})
    merged = sample.merge(renamed, on="review_id", how="left")
    for column in manual_columns:
        existing_column = f"{column}__existing"
        if existing_column not in merged.columns:
            continue
        values = merged[existing_column].fillna("").astype(str)
        keep = values.str.strip().ne("")
        merged.loc[keep, column] = values[keep]
        merged = merged.drop(columns=[existing_column])
    return merged[ANNOTATION_COLUMNS]


def run_sentiment_labeling(sample_size: int = 300) -> dict[str, Any]:
    config = load_config()
    random_seed = int(config.get("random_seed", 42))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path = PROCESSED_DIR / "reviews_clean.parquet"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing processed reviews: {reviews_path}")

    reviews = pd.read_parquet(reviews_path)
    labeled_reviews = ensure_weak_sentiment_labels(reviews)
    labeled_reviews.to_parquet(reviews_path, index=False)

    sample, sample_report = create_annotation_sample(
        labeled_reviews,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    sample_path = REPORT_DIR / "sentiment_annotation_sample.csv"
    sample = preserve_completed_annotation_sample(sample, sample_path)
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "reviews_path": str(reviews_path),
        "total_reviews": int(len(labeled_reviews)),
        "weak_label_counts": labeled_reviews["weak_sentiment_label"].value_counts(dropna=False).astype(int).to_dict(),
        "labeled_reviews": int(labeled_reviews["weak_sentiment_label"].notna().sum()),
        "unlabeled_reviews_due_to_missing_or_invalid_rating": int(labeled_reviews["weak_sentiment_label"].isna().sum()),
        "annotation_sample_path": str(sample_path),
        "annotation_sample": sample_report,
        "limitations": [
            "Weak sentiment labels are derived from reviewer_rating, not manual ground truth.",
            "Rating can disagree with review text, so model evaluation must not treat weak labels as perfect human labels.",
            "Manual sentiment labels are intentionally blank for human annotation.",
        ],
    }
    summary_path = REPORT_DIR / "sentiment_labeling_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weak sentiment labels and annotation sample.")
    parser.add_argument("--sample-size", type=int, default=300)
    args = parser.parse_args()
    summary = run_sentiment_labeling(sample_size=args.sample_size)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
