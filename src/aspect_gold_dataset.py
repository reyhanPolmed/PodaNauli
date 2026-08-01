"""Validation, snapshotting, and locked splitting for clause-level aspect gold."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.aspect_annotation import GOLD_PATH, SPECIAL_NONE, aspect_ids, normalize_aspect_labels


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
SNAPSHOT_DIR = ROOT / "data" / "annotations" / "snapshots"
SPLIT_PATH = PROCESSED_DIR / "aspect_gold_split.parquet"
SPLIT_CSV_PATH = REPORT_DIR / "aspect_gold_split.csv"
MANIFEST_PATH = REPORT_DIR / "aspect_gold_manifest.json"
SPLIT_VERSION = "v2-multilabel-balance"
ARCHIVE_MODEL_DIR = MODEL_DIR / "archive"
ARCHIVE_REPORT_DIR = REPORT_DIR / "archive"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"random_seed": 42}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"random_seed": 42}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(gold: pd.DataFrame) -> str:
    columns = [
        "clause_id",
        "review_id",
        "canonical_place_id",
        "clause_text",
        "manual_aspects",
        "annotator_id",
        "human_approved",
    ]
    payload = (
        gold[columns]
        .fillna("")
        .astype(str)
        .sort_values("clause_id")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def label_matrix(gold: pd.DataFrame) -> pd.DataFrame:
    labels = aspect_ids()
    matrix = pd.DataFrame(0, index=gold.index, columns=labels, dtype=int)
    for index, value in gold["manual_aspects"].items():
        for label in str(value).split("|"):
            if label in matrix.columns:
                matrix.at[index, label] = 1
    return matrix


def _split_score(
    gold: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.Series,
) -> float:
    expected = {"train": 0.60, "validation": 0.20, "test": 0.20}
    score = 0.0
    overall = labels.mean()
    for split, target_rate in expected.items():
        mask = assignments.eq(split)
        score += abs(float(mask.mean()) - target_rate) * 4.0
        if not mask.any():
            return float("inf")
        prevalence = labels.loc[mask].mean()
        score += float((prevalence - overall).abs().mean())
    for label in labels.columns:
        total_positive = int(labels[label].sum())
        positive_groups = gold.loc[labels[label].eq(1), "canonical_place_id"].nunique()
        if total_positive:
            distribution = {
                split: float(labels.loc[assignments.eq(split), label].sum() / total_positive)
                for split in expected
            }
            score += sum(
                abs(distribution[split] - expected[split])
                for split in expected
            ) / len(expected)
        if positive_groups >= 3:
            missing_splits = sum(
                int(labels.loc[assignments.eq(split), label].sum() == 0)
                for split in expected
            )
            score += missing_splits * 2.0
            if total_positive >= 20:
                score += max(0.0, 0.45 - distribution["train"]) * 4.0
                score += max(0.0, 0.10 - distribution["validation"]) * 4.0
                score += max(0.0, 0.10 - distribution["test"]) * 4.0
    return score


def create_locked_aspect_split(
    gold: pd.DataFrame,
    random_seed: int = 42,
    attempts: int = 200,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose a deterministic place-disjoint split with balanced label prevalence."""
    if gold["canonical_place_id"].nunique() < 5:
        raise ValueError("At least five unique places are required for an aspect gold split.")
    labels = label_matrix(gold)
    best_assignments: pd.Series | None = None
    best_score = float("inf")
    for attempt in range(attempts):
        seed = random_seed + attempt
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        remaining_index, test_index = next(
            outer.split(gold, groups=gold["canonical_place_id"])
        )
        remaining = gold.iloc[remaining_index]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 10000)
        train_relative, validation_relative = next(
            inner.split(remaining, groups=remaining["canonical_place_id"])
        )
        assignments = pd.Series("", index=gold.index, dtype=object)
        assignments.iloc[test_index] = "test"
        assignments.iloc[remaining_index[train_relative]] = "train"
        assignments.iloc[remaining_index[validation_relative]] = "validation"
        score = _split_score(gold, labels, assignments)
        if score < best_score:
            best_score = score
            best_assignments = assignments
    if best_assignments is None:
        raise RuntimeError("Unable to create aspect gold split.")
    split = gold[
        ["clause_id", "review_id", "canonical_place_id", "manual_aspects"]
    ].copy()
    split["split"] = best_assignments
    split["split_version"] = SPLIT_VERSION
    split = split.sort_values(["split", "canonical_place_id", "clause_id"]).reset_index(drop=True)
    report = aspect_split_report(split)
    report["selection_score"] = float(best_score)
    report["attempts"] = int(attempts)
    return split, report


def aspect_split_report(split: pd.DataFrame) -> dict[str, Any]:
    label_counts: dict[str, dict[str, int]] = {}
    groups: dict[str, set[str]] = {}
    for split_name in ["train", "validation", "test"]:
        current = split[split["split"] == split_name]
        groups[split_name] = set(current["canonical_place_id"].astype(str))
        counts = {label: 0 for label in [*aspect_ids(), SPECIAL_NONE]}
        for value in current["manual_aspects"]:
            for label in str(value).split("|"):
                if label:
                    counts[label] = counts.get(label, 0) + 1
        label_counts[split_name] = counts
    overlap = (
        (groups["train"] & groups["validation"])
        | (groups["train"] & groups["test"])
        | (groups["validation"] & groups["test"])
    )
    return {
        "method": (
            "best-of-200 nested GroupShuffleSplit using canonical_place_id "
            "with multilabel prevalence balancing"
        ),
        "split_version": SPLIT_VERSION,
        "rows": split["split"].value_counts().reindex(
            ["train", "validation", "test"], fill_value=0
        ).to_dict(),
        "groups": {
            name: len(values)
            for name, values in groups.items()
        },
        "label_counts": label_counts,
        "group_overlap_count": int(len(overlap)),
    }


def archive_weak_aspect_artifacts() -> dict[str, str]:
    ARCHIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = {
        MODEL_DIR / "aspect_champion.joblib": ARCHIVE_MODEL_DIR / "aspect_champion_v1_weak.joblib",
        MODEL_DIR / "aspect_multilabel_binarizer.joblib": (
            ARCHIVE_MODEL_DIR / "aspect_multilabel_binarizer_v1_weak.joblib"
        ),
        REPORT_DIR / "aspect_metrics.json": ARCHIVE_REPORT_DIR / "aspect_metrics_v1_weak.json",
    }
    archived: dict[str, str] = {}
    for source, destination in candidates.items():
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
        if destination.exists():
            archived[source.name] = str(destination)
    return archived


def prepare_aspect_gold_dataset() -> dict[str, Any]:
    config = load_config()
    annotation_config = config.get("aspect_improvement", {}).get("annotation", {})
    minimum_rows = int(annotation_config.get("minimum_gold_rows", 600))
    minimum_key = int(annotation_config.get("minimum_positive_per_key_aspect", 20))
    key_aspects = [
        str(label)
        for label in annotation_config.get(
            "key_aspects",
            ["harga", "pelayanan", "kebersihan", "akomodasi"],
        )
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if not GOLD_PATH.exists():
        gold = pd.DataFrame()
    else:
        gold = pd.read_csv(
            GOLD_PATH,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

    errors: list[str] = []
    normalized: list[str] = []
    if not gold.empty:
        valid_aspects = set(aspect_ids())
        for value in gold["manual_aspects"]:
            labels, error = normalize_aspect_labels(value, valid_aspects)
            normalized.append("|".join(labels))
            errors.append(error or "")
        gold["manual_aspects"] = normalized
    duplicate_rows = int(gold["clause_id"].duplicated().sum()) if not gold.empty else 0
    invalid_rows = int(sum(bool(error) for error in errors))
    label_counts = {label: 0 for label in [*aspect_ids(), SPECIAL_NONE]}
    if not gold.empty:
        for value in gold["manual_aspects"]:
            for label in str(value).split("|"):
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
    ready = bool(
        len(gold) >= minimum_rows
        and duplicate_rows == 0
        and invalid_rows == 0
        and all(label_counts.get(label, 0) >= minimum_key for label in key_aspects)
    )
    semantic_hash = semantic_sha256(gold) if not gold.empty else None
    snapshot_path: Path | None = None
    split_report: dict[str, Any] | None = None
    archived: dict[str, str] = {}
    if not gold.empty:
        snapshot_path = SNAPSHOT_DIR / f"aspect_gold_{semantic_hash[:12]}.csv"
        if not snapshot_path.exists():
            gold.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    if ready:
        reuse_split = False
        if SPLIT_PATH.exists():
            existing = pd.read_parquet(SPLIT_PATH)
            reuse_split = bool(
                "gold_semantic_sha256" in existing
                and existing["gold_semantic_sha256"].eq(semantic_hash).all()
                and "split_version" in existing
                and existing["split_version"].eq(SPLIT_VERSION).all()
                and set(existing["clause_id"].astype(str)) == set(gold["clause_id"].astype(str))
            )
        if reuse_split:
            split = existing
            split_report = aspect_split_report(split)
        else:
            split, split_report = create_locked_aspect_split(
                gold,
                random_seed=int(config.get("random_seed", 42)),
            )
            split["gold_semantic_sha256"] = semantic_hash
            split.to_parquet(SPLIT_PATH, index=False)
            split.to_csv(SPLIT_CSV_PATH, index=False, encoding="utf-8-sig")
        archived = archive_weak_aspect_artifacts()

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "gold_ready": ready,
        "file_sha256": file_sha256(GOLD_PATH) if GOLD_PATH.exists() else None,
        "semantic_sha256": semantic_hash,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "validation": {
            "rows": int(len(gold)),
            "label_counts": label_counts,
            "unique_places": int(gold["canonical_place_id"].nunique()) if not gold.empty else 0,
            "annotator_counts": (
                gold["annotator_id"].value_counts().to_dict() if not gold.empty else {}
            ),
            "duplicate_clause_rows": duplicate_rows,
            "invalid_rows": invalid_rows,
            "minimum_gold_rows": minimum_rows,
            "minimum_positive_per_key_aspect": minimum_key,
            "key_aspects": key_aspects,
            "limitations": [
                "A single annotator cannot provide inter-annotator agreement.",
                "AI suggestions must remain silver and may introduce confirmation bias if visible.",
            ],
        },
        "split": split_report,
        "archived_weak_artifacts": archived,
        "outputs": {
            "split_parquet": str(SPLIT_PATH) if ready else None,
            "split_csv": str(SPLIT_CSV_PATH) if ready else None,
            "manifest": str(MANIFEST_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_aspect_gold_splits(
    gold_path: Path = GOLD_PATH,
    split_path: Path = SPLIT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not gold_path.exists() or not split_path.exists():
        raise FileNotFoundError("Aspect gold labels and locked split are required.")
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assignments = pd.read_parquet(split_path)
    merged = gold.merge(
        assignments[["clause_id", "split"]],
        on="clause_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(gold):
        raise ValueError("Aspect gold rows do not match locked split assignments.")
    train = merged[merged["split"] == "train"].drop(columns="split").reset_index(drop=True)
    validation = (
        merged[merged["split"] == "validation"].drop(columns="split").reset_index(drop=True)
    )
    test = merged[merged["split"] == "test"].drop(columns="split").reset_index(drop=True)
    return train, validation, test, aspect_split_report(assignments)
