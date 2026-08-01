"""Small, offline inference adapter used by the PodaNauli demo notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


SENTIMENT_LABELS = {
    "negative": "Negatif",
    "neutral": "Netral",
    "positive": "Positif",
}
COMPLAINT_LABELS = {
    "negative": "Terdeteksi",
    "non_negative": "Tidak terdeteksi",
    "uncertain": "Perlu tinjau",
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository without exposing an absolute path in notebook output."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "models" / "model_registry.json").exists() and (
            candidate / "outputs" / "predictions" / "service_gap_rankings.csv"
        ).exists():
            return candidate
    raise FileNotFoundError("Root proyek tidak ditemukan. Jalankan notebook dari root repository.")


def _required_artifacts(root: Path) -> dict[str, Path]:
    return {
        "sentiment": root / "models" / "sentiment_champion.joblib",
        "complaint": root / "models" / "complaint_detector.joblib",
        "aspect": root / "models" / "aspect_champion.joblib",
        "aspect_labels": root / "models" / "aspect_multilabel_binarizer.joblib",
        "aspect_metadata": root / "models" / "aspect_metadata.json",
        "service_gap": root / "outputs" / "predictions" / "service_gap_rankings.csv",
    }


def load_demo_bundle(print_status: bool = True) -> dict[str, Any]:
    """Load champion models and the ranking artifact without training or network access."""
    root = find_project_root()
    artifacts = _required_artifacts(root)
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Artefak demo tidak lengkap: " + ", ".join(missing))

    metadata = json.loads(artifacts["aspect_metadata"].read_text(encoding="utf-8"))
    bundle = {
        "root": root,
        "sentiment_model": joblib.load(artifacts["sentiment"]),
        "complaint_bundle": joblib.load(artifacts["complaint"]),
        "aspect_model": joblib.load(artifacts["aspect"]),
        "aspect_labels": joblib.load(artifacts["aspect_labels"]),
        "aspect_threshold": float(metadata["threshold"]),
        "service_gap_path": artifacts["service_gap"],
    }

    if print_status:
        print("[OK] Model sentimen berhasil dimuat")
        print("[OK] Model complaint berhasil dimuat")
        print("[OK] Model aspek berhasil dimuat")
        print("[OK] Artefak Service Gap berhasil dimuat")
    return bundle


def _complaint_predictions(bundle: dict[str, Any], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    complaint_bundle = bundle["complaint_bundle"]
    model = complaint_bundle["model"]
    classes = list(model.named_steps["classifier"].classes_)
    complaint_index = classes.index(1)
    probabilities = np.asarray(model.predict_proba(texts))[:, complaint_index]
    threshold = float(complaint_bundle["negative_threshold"])
    margin = float(complaint_bundle["uncertainty_margin"])
    lower = max(0.0, threshold - margin)
    upper = min(1.0, threshold + margin)
    decisions = np.where(
        probabilities >= upper,
        "negative",
        np.where(probabilities <= lower, "non_negative", "uncertain"),
    )
    return probabilities, decisions


def _aspect_predictions(bundle: dict[str, Any], texts: list[str]) -> list[list[str]]:
    probabilities = np.asarray(bundle["aspect_model"].predict_proba(texts))
    classes = [str(label) for label in bundle["aspect_labels"].classes_]
    threshold = float(bundle["aspect_threshold"])
    results: list[list[str]] = []
    for row in probabilities:
        ranked = sorted(zip(classes, row), key=lambda item: float(item[1]), reverse=True)
        selected = [label for label, score in ranked if label != "lainnya" and float(score) >= threshold][:4]
        if not selected:
            selected = [label for label, score in ranked if label == "lainnya" and float(score) >= threshold]
        results.append(selected)
    return results


def _aspect_text(labels: Iterable[str]) -> str:
    values = [str(label).replace("_", " ").title() for label in labels]
    return ", ".join(values) if values else "Tidak terdeteksi"


def predict_reviews(texts: Iterable[str], bundle: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run all three champion models and return presentation-safe labels."""
    values = [str(text).strip() for text in texts]
    if not values or any(not value for value in values):
        raise ValueError("Semua input ulasan harus berisi teks.")
    active_bundle = bundle or load_demo_bundle(print_status=False)

    sentiment_raw = [str(label) for label in active_bundle["sentiment_model"].predict(values)]
    _, complaint_raw = _complaint_predictions(active_bundle, values)
    aspect_raw = _aspect_predictions(active_bundle, values)

    notes: list[str] = []
    for sentiment, complaint in zip(sentiment_raw, complaint_raw):
        if sentiment == "positive" and complaint == "negative":
            notes.append("Sinyal berbeda; perlu verifikasi manusia")
        elif complaint == "uncertain":
            notes.append("Complaint berada pada area abu-abu")
        else:
            notes.append("Output model; bukan keputusan otomatis")

    return pd.DataFrame(
        {
            "Review": values,
            "Sentimen": [SENTIMENT_LABELS.get(label, label) for label in sentiment_raw],
            "Complaint": [COMPLAINT_LABELS.get(str(label), str(label)) for label in complaint_raw],
            "Aspek": [_aspect_text(labels) for labels in aspect_raw],
            "Catatan": notes,
        }
    )


def display_prediction_result(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Review", "Sentimen", "Complaint", "Aspek", "Catatan"]
    return frame.loc[:, [column for column in columns if column in frame.columns]]


def display_metrics_summary(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Model": "Sentimen", "Metrik": "Macro F1", "Nilai": metrics["sentiment"]["macro_f1"]},
            {
                "Model": "Sentimen",
                "Metrik": "Recall negatif",
                "Nilai": metrics["sentiment"]["negative_recall"],
            },
            {"Model": "Complaint", "Metrik": "Macro F1", "Nilai": metrics["complaint"]["macro_f1"]},
            {
                "Model": "Complaint",
                "Metrik": "Recall complaint",
                "Nilai": metrics["complaint"]["negative_recall"],
            },
            {"Model": "Aspek", "Metrik": "Micro F1", "Nilai": metrics["aspect"]["micro_f1"]},
            {"Model": "Aspek", "Metrik": "Macro F1", "Nilai": metrics["aspect"]["macro_f1"]},
        ]
    ).assign(Nilai=lambda frame: frame["Nilai"].map(lambda value: f"{value:.4f}"))


def display_service_gap_ranking(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    columns = ["rank", "place_name", "category", "aspect", "service_gap_score", "confidence", "evidence_count", "main_reason"]
    output = frame.loc[:, columns].head(limit).copy()
    output["service_gap_score"] = output["service_gap_score"].map(lambda value: f"{float(value):.2f}")
    return output.rename(
        columns={
            "rank": "Peringkat",
            "place_name": "Tempat",
            "category": "Kategori",
            "aspect": "Aspek",
            "service_gap_score": "Skor",
            "confidence": "Confidence",
            "evidence_count": "Bukti",
            "main_reason": "Alasan utama",
        }
    )


def display_place_detail(detail: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Informasi": "Tempat", "Nilai": detail["place_name"]},
            {"Informasi": "Aspek", "Nilai": detail["aspect"]},
            {"Informasi": "Service Gap Score", "Nilai": f"{detail['service_gap_score']:.2f}"},
            {"Informasi": "Jumlah bukti", "Nilai": detail["evidence_count"]},
            {"Informasi": "Sentimen dominan", "Nilai": detail["dominant_sentiment"]},
            {"Informasi": "Confidence", "Nilai": detail["confidence"]},
            {"Informasi": "Validasi manusia", "Nilai": detail["human_validation"]},
        ]
    )


def display_data_quality_summary(data_quality: dict[str, Any]) -> pd.DataFrame:
    summary = data_quality["summary"]
    rows = [
        ("Sheet sumber", summary["sheet_count"]),
        ("Ulasan mentah", summary["raw_review_rows"]),
        ("Ulasan dengan teks valid", summary["reviews_with_text"]),
        ("Korpus NLP setelah deduplikasi", summary["nlp_corpus_rows"]),
        ("Tempat kanonis", summary["canonical_places"]),
        ("Gold sentiment", summary["sentiment_gold_rows"]),
        ("Gold aspek", summary["aspect_gold_rows"]),
        ("Teks kosong", summary["missing_review_text_rows"]),
        ("Duplikat terdeteksi", summary["duplicate_review_rows"]),
    ]
    return pd.DataFrame(rows, columns=["Indikator", "Jumlah"])
