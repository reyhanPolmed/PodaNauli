"""Version, validate, and split the human sentiment gold dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
GOLD_PATH = ROOT / "data" / "annotations" / "sentiment_gold.csv"
SNAPSHOT_DIR = ROOT / "data" / "annotations" / "snapshots"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
ARCHIVE_MODEL_DIR = MODEL_DIR / "archive"
ARCHIVE_REPORT_DIR = REPORT_DIR / "archive"
SPLIT_PATH = PROCESSED_DIR / "sentiment_gold_split.parquet"
SPLIT_CSV_PATH = REPORT_DIR / "sentiment_gold_split.csv"
REPORT_PATH = REPORT_DIR / "gold_dataset_manifest.json"
LABELS = ["negative", "neutral", "positive"]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"random_seed": 42}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_gold_hash(gold: pd.DataFrame) -> str:
    columns = [
        "review_id",
        "canonical_place_id",
        "manual_sentiment_label",
        "annotator_id",
        "human_approved",
    ]
    stable = gold[columns].sort_values("review_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def validate_gold_frame(gold: pd.DataFrame) -> dict[str, Any]:
    required = {
        "review_id",
        "canonical_place_id",
        "manual_sentiment_label",
        "annotator_id",
        "human_approved",
    }
    missing = required - set(gold.columns)
    if missing:
        raise ValueError(f"Gold dataset is missing columns: {sorted(missing)}")
    labels = gold["manual_sentiment_label"].astype(str).str.strip().str.lower()
    approvals = gold["human_approved"].astype(str).str.strip().str.lower()
    invalid_labels = ~labels.isin(LABELS)
    unapproved = ~approvals.isin({"yes", "true", "1", "approved"})
    duplicate_ids = gold["review_id"].duplicated(keep=False)
    missing_groups = gold["canonical_place_id"].astype(str).str.strip().eq("")
    report = {
        "rows": int(len(gold)),
        "label_counts": labels.value_counts().to_dict(),
        "unique_places": int(gold["canonical_place_id"].nunique()),
        "annotator_counts": gold["annotator_id"].value_counts().to_dict(),
        "invalid_label_rows": int(invalid_labels.sum()),
        "unapproved_rows": int(unapproved.sum()),
        "duplicate_review_rows": int(duplicate_ids.sum()),
        "missing_group_rows": int(missing_groups.sum()),
        "valid": bool(
            len(gold) > 0
            and not invalid_labels.any()
            and not unapproved.any()
            and not duplicate_ids.any()
            and not missing_groups.any()
        ),
        "limitations": [
            "All current labels were produced by one annotator; inter-annotator agreement is unavailable."
            if gold["annotator_id"].nunique() < 2
            else "",
            "Annotators could see AI suggestions, so confirmation bias remains possible.",
        ],
    }
    report["limitations"] = [item for item in report["limitations"] if item]
    return report


def create_grouped_gold_splits(gold: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """Create deterministic 60/20/20-like place-disjoint partitions."""
    frame = gold.copy().reset_index(drop=True)
    frame["manual_sentiment_label"] = frame["manual_sentiment_label"].astype(str).str.strip().str.lower()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
    development_idx, test_idx = next(
        outer.split(
            frame["review_id"],
            frame["manual_sentiment_label"],
            frame["canonical_place_id"],
        )
    )
    development = frame.iloc[development_idx].copy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_seed + 1)
    train_local_idx, validation_local_idx = next(
        inner.split(
            development["review_id"],
            development["manual_sentiment_label"],
            development["canonical_place_id"],
        )
    )
    assignments = pd.DataFrame(
        {
            "review_id": frame["review_id"],
            "canonical_place_id": frame["canonical_place_id"],
            "manual_sentiment_label": frame["manual_sentiment_label"],
            "split": "",
        }
    )
    assignments.loc[test_idx, "split"] = "test"
    assignments.loc[development.index[train_local_idx], "split"] = "train"
    assignments.loc[development.index[validation_local_idx], "split"] = "validation"
    if assignments["split"].eq("").any():
        raise ValueError("Some gold rows were not assigned to a split.")

    groups = {
        split: set(assignments.loc[assignments["split"] == split, "canonical_place_id"])
        for split in ["train", "validation", "test"]
    }
    overlap = (
        (groups["train"] & groups["validation"])
        | (groups["train"] & groups["test"])
        | (groups["validation"] & groups["test"])
    )
    if overlap:
        raise ValueError(f"Gold split has place leakage: {sorted(overlap)[:5]}")
    return assignments.sort_values("review_id").reset_index(drop=True)


def split_report(assignments: pd.DataFrame) -> dict[str, Any]:
    groups = {
        split: set(assignments.loc[assignments["split"] == split, "canonical_place_id"])
        for split in ["train", "validation", "test"]
    }
    overlap = (
        (groups["train"] & groups["validation"])
        | (groups["train"] & groups["test"])
        | (groups["validation"] & groups["test"])
    )
    return {
        "method": "Nested StratifiedGroupKFold with canonical_place_id groups",
        "rows": assignments["split"].value_counts().to_dict(),
        "groups": {key: len(value) for key, value in groups.items()},
        "label_counts": pd.crosstab(
            assignments["split"],
            assignments["manual_sentiment_label"],
        ).to_dict(orient="index"),
        "group_overlap_count": int(len(overlap)),
    }


def apply_gold_split(
    data: pd.DataFrame,
    split_path: Path = SPLIT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing gold split manifest: {split_path}")
    assignments = pd.read_parquet(split_path)
    merged = data.merge(assignments[["review_id", "split"]], on="review_id", how="inner", validate="one_to_one")
    if len(merged) != len(assignments):
        raise ValueError(
            f"Gold training rows ({len(merged)}) do not match split rows ({len(assignments)})."
        )
    train = merged[merged["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    validation = merged[merged["split"] == "validation"].drop(columns=["split"]).reset_index(drop=True)
    test = merged[merged["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    report = split_report(assignments)
    return train, validation, test, report


def archive_weak_baselines() -> dict[str, str]:
    """Archive weak artifacts once before gold training overwrites champion paths."""
    ARCHIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = {
        MODEL_DIR / "sentiment_champion.joblib": ARCHIVE_MODEL_DIR / "sentiment_champion_v1_weak.joblib",
        MODEL_DIR / "complaint_detector.joblib": ARCHIVE_MODEL_DIR / "complaint_detector_v2_weak.joblib",
        REPORT_DIR / "sentiment_metrics.json": ARCHIVE_REPORT_DIR / "sentiment_metrics_v1_weak.json",
        REPORT_DIR / "complaint_metrics.json": ARCHIVE_REPORT_DIR / "complaint_metrics_v2_weak.json",
    }
    archived = {}
    for source, destination in candidates.items():
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
        if destination.exists():
            archived[source.name] = str(destination)
    return archived


def prepare_gold_dataset() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    random_seed = int(config.get("random_seed", 42))
    if not GOLD_PATH.exists():
        report = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "gold_ready": False,
            "reason": f"Missing {GOLD_PATH}",
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    gold = pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    validation = validate_gold_frame(gold)
    semantic_hash = semantic_gold_hash(gold) if validation["valid"] else None
    file_hash = sha256_file(GOLD_PATH)
    if not validation["valid"]:
        report = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "gold_ready": False,
            "file_sha256": file_hash,
            "validation": validation,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    snapshot_path = SNAPSHOT_DIR / f"sentiment_gold_{semantic_hash[:12]}.csv"
    if not snapshot_path.exists():
        shutil.copy2(GOLD_PATH, snapshot_path)
    assignments = create_grouped_gold_splits(gold, random_seed=random_seed)
    assignments.to_parquet(SPLIT_PATH, index=False)
    assignments.to_csv(SPLIT_CSV_PATH, index=False, encoding="utf-8-sig")
    archived = archive_weak_baselines()
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "gold_ready": True,
        "file_sha256": file_hash,
        "semantic_sha256": semantic_hash,
        "snapshot_path": str(snapshot_path),
        "validation": validation,
        "split": split_report(assignments),
        "archived_weak_artifacts": archived,
        "outputs": {
            "split_parquet": str(SPLIT_PATH),
            "split_csv": str(SPLIT_CSV_PATH),
            "manifest": str(REPORT_PATH),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    result = prepare_gold_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
