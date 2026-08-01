"""Tahap 6 pipeline: weak aspect labeling and multi-label baseline model."""

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
ASPECT_TAXONOMY_PATH = ROOT / "configs" / "aspect_taxonomy.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_METADATA_PATH = MODEL_DIR / "aspect_metadata.json"
ASPECT_GOLD_MANIFEST_PATH = REPORT_DIR / "aspect_gold_manifest.json"

NEGATIVE_CONTEXT_WORDS = {
    "tidak",
    "kurang",
    "buruk",
    "jelek",
    "kotor",
    "mahal",
    "sulit",
    "susah",
    "lambat",
    "rusak",
    "sempit",
    "bau",
    "ramai",
    "penuh",
    "overpriced",
    "bad",
    "dirty",
}

ANNOTATION_COLUMNS = [
    "review_id",
    "place_name",
    "review_text_raw",
    "weak_aspects",
    "manual_aspects",
    "annotation_notes",
]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config() -> dict[str, Any]:
    config = load_yaml(CONFIG_PATH)
    config.setdefault("random_seed", 42)
    return config


def load_aspect_taxonomy(path: Path = ASPECT_TAXONOMY_PATH) -> list[dict[str, Any]]:
    data = load_yaml(path)
    aspects = data.get("aspects", [])
    if not aspects:
        raise ValueError(f"No aspects found in {path}")
    return aspects


def normalize_for_matching(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^0-9a-zA-Z_\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def keyword_regex(keyword: str) -> re.Pattern[str]:
    normalized = normalize_for_matching(keyword)
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def compile_taxonomy_patterns(aspects: list[dict[str, Any]]) -> dict[str, dict[str, list[re.Pattern[str]]]]:
    compiled: dict[str, dict[str, list[re.Pattern[str]]]] = {}
    for aspect in aspects:
        aspect_id = aspect["id"]
        compiled[aspect_id] = {
            "positive": [keyword_regex(str(keyword)) for keyword in aspect.get("keywords_positive", []) if str(keyword).strip()],
            "negative": [keyword_regex(str(keyword)) for keyword in aspect.get("keywords_negative", []) if str(keyword).strip()],
            "related": [keyword_regex(str(keyword)) for keyword in aspect.get("related_facility_types", []) if str(keyword).strip()],
        }
    return compiled


def _matches(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    found = []
    for pattern in patterns:
        if pattern.search(text):
            found.append(pattern.pattern)
    return found


def detect_weak_aspects(text: Any, aspects: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Detect weak aspect labels from taxonomy keyword/rule evidence."""
    if pd.isna(text) or not str(text).strip():
        return {"weak_aspects": [], "negative_aspects": [], "evidence": {}}
    aspects = aspects or load_aspect_taxonomy()
    patterns = compile_taxonomy_patterns(aspects)
    normalized_text = normalize_for_matching(str(text))
    weak_aspects = []
    negative_aspects = []
    evidence: dict[str, Any] = {}

    has_negative_context = any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized_text) for word in NEGATIVE_CONTEXT_WORDS)
    for aspect in aspects:
        aspect_id = aspect["id"]
        if aspect_id == "lainnya":
            continue
        positive_hits = _matches(patterns[aspect_id]["positive"], normalized_text)
        negative_hits = _matches(patterns[aspect_id]["negative"], normalized_text)
        related_hits = _matches(patterns[aspect_id]["related"], normalized_text)
        related_hits_with_context = related_hits if has_negative_context else []
        if positive_hits or negative_hits or related_hits_with_context:
            weak_aspects.append(aspect_id)
            if negative_hits or related_hits_with_context:
                negative_aspects.append(aspect_id)
            evidence[aspect_id] = {
                "positive_hits": len(positive_hits),
                "negative_hits": len(negative_hits),
                "related_hits": len(related_hits_with_context),
                "negative_context": bool(has_negative_context),
            }
    if not weak_aspects:
        weak_aspects = ["lainnya"]
        evidence["lainnya"] = {
            "positive_hits": 0,
            "negative_hits": 0,
            "related_hits": 0,
            "negative_context": bool(has_negative_context),
        }
    return {
        "weak_aspects": sorted(set(weak_aspects)),
        "negative_aspects": sorted(set(negative_aspects)),
        "evidence": evidence,
    }


def prepare_aspect_dataset(reviews: pd.DataFrame, aspects: list[dict[str, Any]]) -> pd.DataFrame:
    """Create weak aspect columns for non-empty, non-duplicate reviews."""
    dataset = reviews[
        reviews["review_text_clean"].notna()
        & (reviews["text_length"] > 0)
        & (~reviews["is_duplicate"])
    ].copy()
    detections = dataset["review_text_clean"].map(lambda text: detect_weak_aspects(text, aspects))
    dataset["weak_aspects"] = detections.map(lambda item: item["weak_aspects"])
    dataset["weak_negative_aspects"] = detections.map(lambda item: item["negative_aspects"])
    dataset["weak_aspect_evidence"] = detections.map(lambda item: json.dumps(item["evidence"], ensure_ascii=False))
    dataset["weak_aspects_text"] = dataset["weak_aspects"].map(lambda values: "|".join(values))
    dataset["weak_negative_aspects_text"] = dataset["weak_negative_aspects"].map(lambda values: "|".join(values))
    return dataset.reset_index(drop=True)


def create_aspect_annotation_sample(dataset: pd.DataFrame, sample_size: int, random_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a deterministic sample for manual multi-label aspect annotation."""
    candidates = dataset.copy()
    target_size = min(sample_size, len(candidates))
    aspect_rows = []
    for aspect_id in sorted({aspect for values in candidates["weak_aspects"] for aspect in values}):
        aspect_candidates = candidates[candidates["weak_aspects"].map(lambda values: aspect_id in values)]
        if aspect_candidates.empty:
            continue
        take = min(max(3, target_size // 30), len(aspect_candidates))
        aspect_rows.append(aspect_candidates.sample(n=take, random_state=random_seed))
    seeded = pd.concat(aspect_rows, ignore_index=False) if aspect_rows else candidates.head(0)
    remaining = candidates.drop(index=seeded.index, errors="ignore")
    if len(seeded) < target_size and not remaining.empty:
        seeded = pd.concat(
            [
                seeded,
                remaining.sample(n=min(target_size - len(seeded), len(remaining)), random_state=random_seed),
            ],
            ignore_index=False,
        )
    sample = seeded.drop_duplicates("review_id").head(target_size).copy()
    if len(sample) < target_size:
        remaining = candidates[~candidates["review_id"].isin(sample["review_id"])]
        sample = pd.concat([sample, remaining.head(target_size - len(sample))], ignore_index=False)
    sample = sample.sort_values(["weak_aspects_text", "place_category", "review_id"]).reset_index(drop=True)
    sample["manual_aspects"] = ""
    sample["annotation_notes"] = ""
    output = sample.assign(weak_aspects=sample["weak_aspects_text"])[ANNOTATION_COLUMNS]
    report = {
        "sample_size": int(len(output)),
        "sample_size_requested": int(sample_size),
        "unique_places": int(sample["canonical_place_id"].nunique(dropna=True)),
        "weak_aspect_rows_in_sample": sample["weak_aspects_text"].value_counts().head(20).to_dict(),
    }
    return output, report


def preserve_aspect_annotation_sample(sample: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Preserve manual multi-label annotations across idempotent pipeline runs."""
    if not existing_path.exists():
        return sample
    existing = pd.read_csv(existing_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "review_id" not in existing.columns:
        return sample
    manual_columns = ["manual_aspects", "annotation_notes"]
    available = ["review_id", *[column for column in manual_columns if column in existing.columns]]
    existing = existing[available].drop_duplicates("review_id", keep="last")
    renamed = existing.rename(columns={column: f"{column}__existing" for column in manual_columns if column in existing})
    merged = sample.merge(renamed, on="review_id", how="left")
    for column in manual_columns:
        existing_column = f"{column}__existing"
        if existing_column not in merged:
            continue
        values = merged[existing_column].fillna("").astype(str)
        keep = values.str.strip().ne("")
        merged.loc[keep, column] = values[keep]
        merged = merged.drop(columns=[existing_column])
    return merged[ANNOTATION_COLUMNS]


def build_aspect_model(random_seed: int = 42, c_value: float = 1.0) -> Pipeline:
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
    classifier = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            C=c_value,
            max_iter=1000,
            random_state=random_seed,
        )
    )
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])),
            ("classifier", classifier),
        ]
    )


def predict_aspect_probabilities(model: Pipeline, texts: pd.Series, aspect_ids: list[str]) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
    else:
        decision = model.decision_function(texts)
        probabilities = 1.0 / (1.0 + np.exp(-decision))
    return pd.DataFrame(probabilities, columns=[f"aspect_probability_{aspect_id}" for aspect_id in aspect_ids])


def coverage_report(dataset: pd.DataFrame, aspect_ids: list[str]) -> pd.DataFrame:
    rows = []
    total = len(dataset)
    for aspect_id in aspect_ids:
        aspect_mask = dataset["weak_aspects"].map(lambda values: aspect_id in values)
        negative_mask = dataset["weak_negative_aspects"].map(lambda values: aspect_id in values)
        rows.append(
            {
                "aspect_id": aspect_id,
                "weak_label_count": int(aspect_mask.sum()),
                "weak_label_rate": float(aspect_mask.mean()) if total else 0.0,
                "weak_negative_count": int(negative_mask.sum()),
                "weak_negative_rate": float(negative_mask.mean()) if total else 0.0,
                "positive_review_count": int((aspect_mask & (dataset["weak_sentiment_label"] == "positive")).sum()),
                "neutral_review_count": int((aspect_mask & (dataset["weak_sentiment_label"] == "neutral")).sum()),
                "negative_review_count": int((aspect_mask & (dataset["weak_sentiment_label"] == "negative")).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("weak_label_count", ascending=False)


def manual_aspect_lists(frame: pd.DataFrame, valid_aspects: set[str]) -> list[list[str]]:
    labels = []
    for value in frame["manual_aspects"]:
        labels.append(
            [
                item
                for item in str(value).split("|")
                if item and item != "none" and item in valid_aspects
            ]
        )
    return labels


def evaluate_multilabel_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    aspect_ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )
    per_aspect = {}
    average_precisions = []
    for index, aspect_id in enumerate(aspect_ids):
        if 0 < int(y_true[:, index].sum()) < len(y_true):
            average_precision = float(
                average_precision_score(y_true[:, index], probabilities[:, index])
            )
            average_precisions.append(average_precision)
        else:
            average_precision = None
        per_aspect[aspect_id] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "average_precision": average_precision,
        }
    true_none = y_true.sum(axis=1) == 0
    predicted_none = predictions.sum(axis=1) == 0
    none_precision, none_recall, none_f1, _support = precision_recall_fscore_support(
        true_none.astype(int),
        predicted_none.astype(int),
        average="binary",
        zero_division=0,
    )
    return {
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "subset_accuracy": float(accuracy_score(y_true, predictions)),
        "macro_average_precision": (
            float(np.mean(average_precisions)) if average_precisions else None
        ),
        "none_precision": float(none_precision),
        "none_recall": float(none_recall),
        "none_f1": float(none_f1),
        "threshold": float(threshold),
        "per_aspect": per_aspect,
    }


def train_gold_aspect_model(
    config: dict[str, Any],
    aspect_ids: list[str],
) -> tuple[Pipeline, MultiLabelBinarizer, dict[str, Any], pd.DataFrame]:
    """Select a supervised model on validation and evaluate locked test once."""
    from src.aspect_gold_dataset import load_aspect_gold_splits

    train, validation, test, split_report = load_aspect_gold_splits()
    random_seed = int(config.get("random_seed", 42))
    model_config = config.get("aspect_improvement", {}).get("model", {})
    c_values = [float(value) for value in model_config.get("c_values", [0.3, 1.0, 3.0])]
    thresholds = [
        float(value)
        for value in model_config.get(
            "threshold_values",
            [0.25, 0.35, 0.45, 0.55, 0.65],
        )
    ]
    mlb = MultiLabelBinarizer(classes=aspect_ids)
    y_train = mlb.fit_transform(manual_aspect_lists(train, set(aspect_ids)))
    y_validation = mlb.transform(manual_aspect_lists(validation, set(aspect_ids)))
    y_test = mlb.transform(manual_aspect_lists(test, set(aspect_ids)))
    candidates = []
    best: tuple[float, float, dict[str, Any], Pipeline] | None = None
    for c_value in c_values:
        model = build_aspect_model(random_seed=random_seed, c_value=c_value)
        model.fit(train["clause_text"], y_train)
        validation_probabilities = np.asarray(model.predict_proba(validation["clause_text"]))
        for threshold in thresholds:
            metrics = evaluate_multilabel_predictions(
                y_validation,
                validation_probabilities,
                aspect_ids,
                threshold,
            )
            row = {
                "c_value": c_value,
                "threshold": threshold,
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "hamming_loss": metrics["hamming_loss"],
                "subset_accuracy": metrics["subset_accuracy"],
            }
            candidates.append(row)
            selection_key = (
                metrics["macro_f1"],
                metrics["micro_f1"],
                -metrics["hamming_loss"],
            )
            if best is None or selection_key > (
                best[2]["macro_f1"],
                best[2]["micro_f1"],
                -best[2]["hamming_loss"],
            ):
                best = (c_value, threshold, metrics, model)
    if best is None:
        raise RuntimeError("No supervised aspect candidate was trained.")
    selected_c, selected_threshold, validation_metrics, _validation_model = best
    combined = pd.concat([train, validation], ignore_index=True)
    y_combined = mlb.transform(manual_aspect_lists(combined, set(aspect_ids)))
    final_model = build_aspect_model(random_seed=random_seed, c_value=selected_c)
    final_model.fit(combined["clause_text"], y_combined)
    test_probabilities = np.asarray(final_model.predict_proba(test["clause_text"]))
    test_metrics = evaluate_multilabel_predictions(
        y_test,
        test_probabilities,
        aspect_ids,
        selected_threshold,
    )
    gates = config.get("aspect_improvement", {}).get("acceptance_gates", {})
    key_aspects = (
        config.get("aspect_improvement", {})
        .get("annotation", {})
        .get("key_aspects", [])
    )
    key_f1 = {
        label: test_metrics["per_aspect"].get(label, {}).get("f1", 0.0)
        for label in key_aspects
    }
    gate_results = {
        "micro_f1": test_metrics["micro_f1"] >= float(gates.get("micro_f1", 0.70)),
        "macro_f1": test_metrics["macro_f1"] >= float(gates.get("macro_f1", 0.55)),
        "key_aspect_minimum_f1": bool(key_f1)
        and min(key_f1.values()) >= float(gates.get("key_aspect_minimum_f1", 0.50)),
    }
    comparison = pd.DataFrame(candidates).sort_values(
        ["macro_f1", "micro_f1", "hamming_loss"],
        ascending=[False, False, True],
    )
    report = {
        "split": split_report,
        "selection_split": "validation",
        "selected_parameters": {
            "c_value": selected_c,
            "threshold": selected_threshold,
        },
        "validation_metrics": validation_metrics,
        "final_test_split": "test",
        "test_metrics": test_metrics,
        "key_aspect_test_f1": key_f1,
        "acceptance_gates": gates,
        "gate_results": gate_results,
        "deployment_ready": all(gate_results.values()),
    }
    return final_model, mlb, report, comparison


def run_aspect_training(sample_size: int = 300) -> dict[str, Any]:
    config = load_config()
    random_seed = int(config.get("random_seed", 42))
    aspects = load_aspect_taxonomy()
    aspect_ids = [aspect["id"] for aspect in aspects]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path = PROCESSED_DIR / "reviews_clean.parquet"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing processed reviews: {reviews_path}")
    reviews = pd.read_parquet(reviews_path)
    dataset = prepare_aspect_dataset(reviews, aspects)

    gold_manifest = (
        json.loads(ASPECT_GOLD_MANIFEST_PATH.read_text(encoding="utf-8"))
        if ASPECT_GOLD_MANIFEST_PATH.exists()
        else {}
    )
    uses_gold = bool(gold_manifest.get("gold_ready", False))
    supervised_report: dict[str, Any] | None = None
    if uses_gold:
        model, mlb, supervised_report, comparison = train_gold_aspect_model(
            config,
            aspect_ids,
        )
        comparison.to_csv(
            REPORT_DIR / "aspect_model_comparison.csv",
            index=False,
            encoding="utf-8",
        )
        threshold = float(supervised_report["selected_parameters"]["threshold"])
        label_source = "human_gold"
        version = "v2-clause-human-gold"
    else:
        mlb = MultiLabelBinarizer(classes=aspect_ids)
        y = mlb.fit_transform(dataset["weak_aspects"])
        model = build_aspect_model(random_seed=random_seed)
        model.fit(dataset["review_text_clean"], y)
        threshold = float(
            config.get("sentiment_improvement", {})
            .get("complaint_model", {})
            .get("aspect_probability_threshold", 0.50)
        )
        label_source = "rule_based_weak"
        version = "v1-rule-label-baseline"

    joblib.dump(model, MODEL_DIR / "aspect_champion.joblib")
    joblib.dump(mlb, MODEL_DIR / "aspect_multilabel_binarizer.joblib")
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "version": version,
        "label_source": label_source,
        "threshold": threshold,
        "training_label_hash": gold_manifest.get("semantic_sha256") if uses_gold else None,
    }
    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    probabilities = predict_aspect_probabilities(model, dataset["review_text_clean"], list(mlb.classes_))
    predictions = dataset[
        [
            "review_id",
            "canonical_place_id",
            "place_name",
            "place_category",
            "weak_sentiment_label",
            "review_text_clean",
            "weak_aspects_text",
            "weak_negative_aspects_text",
        ]
    ].copy()
    predictions = pd.concat([predictions, probabilities], axis=1)
    predictions.to_parquet(PROCESSED_DIR / "review_aspect_predictions.parquet", index=False)

    coverage = coverage_report(dataset, list(mlb.classes_))
    coverage.to_csv(REPORT_DIR / "aspect_coverage.csv", index=False, encoding="utf-8")

    annotation_path = REPORT_DIR / "aspect_annotation_sample.csv"
    annotation_sample, annotation_report = create_aspect_annotation_sample(dataset, sample_size, random_seed)
    annotation_sample = preserve_aspect_annotation_sample(annotation_sample, annotation_path)
    annotation_sample.to_csv(annotation_path, index=False, encoding="utf-8-sig")

    metrics = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "reviews_path": str(reviews_path),
        "training_rows": int(len(dataset)),
        "aspect_count": int(len(aspect_ids)),
        "aspect_ids": aspect_ids,
        "version": version,
        "label_source": label_source,
        "training_label_hash": gold_manifest.get("semantic_sha256") if uses_gold else None,
        "supervised_evaluation_available": uses_gold,
        "supervised_metrics": (
            supervised_report["test_metrics"] if supervised_report else None
        ),
        "validation_metrics": (
            supervised_report["validation_metrics"] if supervised_report else None
        ),
        "test_metrics": supervised_report["test_metrics"] if supervised_report else None,
        "selection": (
            supervised_report["selected_parameters"] if supervised_report else None
        ),
        "split": supervised_report["split"] if supervised_report else None,
        "acceptance_gates": (
            supervised_report["acceptance_gates"] if supervised_report else {}
        ),
        "gate_results": supervised_report["gate_results"] if supervised_report else {},
        "deployment_ready": (
            bool(supervised_report["deployment_ready"]) if supervised_report else False
        ),
        "coverage_summary": {
            "rows_with_lainnya_only": int(dataset["weak_aspects"].map(lambda values: values == ["lainnya"]).sum()),
            "rows_with_any_taxonomy_aspect": int(dataset["weak_aspects"].map(lambda values: values != ["lainnya"]).sum()),
            "top_aspects": coverage.head(10).to_dict(orient="records"),
        },
        "annotation_sample": annotation_report,
        "outputs": {
            "aspect_champion": str(MODEL_DIR / "aspect_champion.joblib"),
            "aspect_multilabel_binarizer": str(MODEL_DIR / "aspect_multilabel_binarizer.joblib"),
            "aspect_coverage": str(REPORT_DIR / "aspect_coverage.csv"),
            "aspect_annotation_sample": str(REPORT_DIR / "aspect_annotation_sample.csv"),
            "review_aspect_predictions": str(PROCESSED_DIR / "review_aspect_predictions.parquet"),
            "aspect_metadata": str(MODEL_METADATA_PATH),
            "aspect_model_comparison": (
                str(REPORT_DIR / "aspect_model_comparison.csv") if uses_gold else None
            ),
        },
        "limitations": (
            [
                "The current human gold dataset has one annotator.",
                "AI suggestions may introduce confirmation bias if visible during annotation.",
                "Clause segmentation remains deterministic.",
            ]
            if uses_gold
            else [
                "Aspect labels are weak rule-based labels, not manually verified labels.",
                "The trained aspect model imitates the weak labeling rules and should not be interpreted as supervised ground-truth performance.",
                "No supervised metrics are reported because clause-level human gold is not ready.",
            ]
        ),
    }
    (REPORT_DIR / "aspect_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weakly supervised aspect model.")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--model", default="tfidf-logistic", help="Reserved for future alternatives.")
    parser.add_argument("--enable-transformers", action="store_true", help="Transformer training is optional and not part of this CPU stage.")
    args = parser.parse_args()
    if args.enable_transformers:
        raise SystemExit("Transformer aspect training is optional and not part of the CPU baseline.")
    metrics = run_aspect_training(sample_size=args.sample_size)
    print(
        json.dumps(
            {
                "training_rows": metrics["training_rows"],
                "aspect_count": metrics["aspect_count"],
                "supervised_evaluation_available": metrics["supervised_evaluation_available"],
                "aspect_coverage": metrics["outputs"]["aspect_coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
