"""Prepare a deterministic, privacy-safe PodaNauli video demo package."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "scratch" / "matplotlib_demo"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyBboxPatch

from demo.demo_runtime import SENTIMENT_LABELS, load_demo_bundle, predict_reviews


DEMO_DIR = ROOT / "demo"
FIGURE_DIR = DEMO_DIR / "figures"
OUTPUT_DIR = DEMO_DIR / "outputs"
CONFIG_PATH = DEMO_DIR / "demo_config.yaml"

NAVY = "#17324D"
TEAL = "#239B8F"
BLUE = "#3977B9"
ORANGE = "#F0A23B"
RED = "#C94C4C"
GREEN = "#5A9A52"
GRAY = "#66737F"
LIGHT = "#F4F7F9"

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?62|0)[\s.-]?(?:\d[\s.-]?){8,13}(?!\d)")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_video_safe(value: Any, max_characters: int = 180) -> bool:
    text = clean_text(value)
    if not 20 <= len(text) <= max_characters:
        return False
    return not (EMAIL_RE.search(text) or PHONE_RE.search(text) or URL_RE.search(text))


def redact_sensitive(value: Any) -> str:
    text = clean_text(value)
    text = EMAIL_RE.sub("[email disamarkan]", text)
    text = PHONE_RE.sub("[nomor disamarkan]", text)
    return text


def relative_source_map(config: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value).replace("\\", "/") for key, value in config["sources"].items()}


def select_demo_reviews(
    reviews: pd.DataFrame,
    split: pd.DataFrame,
    bundle: dict[str, Any],
    max_examples: int,
    max_characters: int,
) -> pd.DataFrame:
    locked = split[split["split"].eq("test")].merge(reviews, on=["review_id", "canonical_place_id"], how="inner")
    locked["review_text"] = locked["review_text_clean"].map(clean_text)
    locked = locked[
        locked["review_text"].map(lambda text: is_video_safe(text, max_characters))
        & locked["place_name"].notna()
    ].copy()
    locked["predicted_raw"] = bundle["sentiment_model"].predict(locked["review_text"].tolist())
    locked["is_error"] = locked["predicted_raw"].astype(str).ne(locked["manual_sentiment_label"].astype(str))

    selected_indexes: list[int] = []
    correct = locked[~locked["is_error"]]
    for label in ["negative", "neutral", "positive"]:
        candidates = correct[correct["manual_sentiment_label"].eq(label)].sample(frac=1, random_state=42)
        selected_indexes.extend(candidates.head(2).index.tolist())
    error_candidates = locked[locked["is_error"]].sample(frac=1, random_state=42)
    if not error_candidates.empty:
        selected_indexes.append(int(error_candidates.index[0]))

    if len(selected_indexes) < max_examples:
        remaining = locked.drop(index=list(dict.fromkeys(selected_indexes))).sample(frac=1, random_state=42)
        selected_indexes.extend(remaining.head(max_examples - len(selected_indexes)).index.tolist())
    selected_indexes = list(dict.fromkeys(selected_indexes))[:max_examples]
    selected = locked.loc[selected_indexes].copy().reset_index(drop=True)

    predictions = predict_reviews(selected["review_text"].tolist(), bundle)
    selected["predicted_sentiment"] = predictions["Sentimen"].values
    selected["predicted_complaint"] = predictions["Complaint"].values
    selected["predicted_aspects"] = predictions["Aspek"].values
    selected["expected_sentiment"] = selected["manual_sentiment_label"].map(SENTIMENT_LABELS)
    selected["is_error_example"] = selected["predicted_sentiment"].ne(selected["expected_sentiment"])
    selected["safe_for_video"] = True
    selected.insert(0, "demo_id", [f"D{i:02d}" for i in range(1, len(selected) + 1)])
    return selected[
        [
            "demo_id",
            "place_name",
            "review_text",
            "expected_sentiment",
            "predicted_sentiment",
            "predicted_complaint",
            "predicted_aspects",
            "is_error_example",
            "safe_for_video",
        ]
    ]


def select_raw_examples(reviews: pd.DataFrame, max_characters: int) -> list[dict[str, Any]]:
    candidates = reviews[
        reviews["review_text_raw"].map(lambda text: is_video_safe(text, max_characters))
        & reviews["place_name"].notna()
        & reviews["reviewer_rating"].notna()
        & ~reviews["is_duplicate"].fillna(False)
    ].copy()
    candidates["sentiment_group"] = candidates["weak_sentiment_label"].fillna("unknown")
    records: list[dict[str, Any]] = []
    for label in ["negative", "neutral", "positive"]:
        group = candidates[candidates["sentiment_group"].eq(label)].sample(frac=1, random_state=42).head(1)
        for _, row in group.iterrows():
            records.append(
                {
                    "source_sheet": str(row["source_sheet"]),
                    "place_name": str(row["place_name"]),
                    "reviewer_rating": float(row["reviewer_rating"]),
                    "review_text": redact_sensitive(row["review_text_raw"]),
                }
            )
    fallback = candidates.sample(frac=1, random_state=43)
    used_texts = {item["review_text"] for item in records}
    for _, row in fallback.iterrows():
        text = redact_sensitive(row["review_text_raw"])
        if text in used_texts:
            continue
        records.append(
            {
                "source_sheet": str(row["source_sheet"]),
                "place_name": str(row["place_name"]),
                "reviewer_rating": float(row["reviewer_rating"]),
                "review_text": text,
            }
        )
        if len(records) == 5:
            break
    return records[:5]


def select_cleaning_example(reviews: pd.DataFrame, max_characters: int) -> dict[str, str]:
    candidates = reviews[
        reviews["review_text_raw"].notna()
        & reviews["review_text_clean"].notna()
        & reviews["review_text_raw"].astype(str).ne(reviews["review_text_clean"].astype(str))
    ]
    for _, row in candidates.iterrows():
        before = redact_sensitive(row["review_text_raw"])
        after = redact_sensitive(row["review_text_clean"])
        if before != after and is_video_safe(after, max_characters) and len(before) <= max_characters:
            return {"before": before, "after": after, "source": "contoh aktual dari hasil cleaning"}
    return {
        "before": "  Tempatnya   bagus, tapi toiletnya kurang bersih  ",
        "after": "Tempatnya bagus, tapi toiletnya kurang bersih",
        "source": "contoh deterministik aturan whitespace",
    }


def prepare_metrics(config: dict[str, Any]) -> dict[str, Any]:
    sentiment = load_json(config["sources"]["sentiment_metrics"])
    complaint = load_json(config["sources"]["complaint_metrics"])
    aspect = load_json(config["sources"]["aspect_metrics"])
    readiness = load_json(config["sources"]["readiness"])
    sentiment_test = sentiment["champion"]["metrics"]
    complaint_test = complaint["test_metrics"]
    aspect_test = aspect["test_metrics"]
    validation = readiness["service_gap_validation"]
    aspect_total_counts = load_json("outputs/reports/aspect_gold_manifest.json")["validation"]["label_counts"]

    return {
        "product": "PodaNauli",
        "metrics_scope": "locked human-gold test dan validasi manusia top-20",
        "sentiment": {
            "gold_rows": int(sentiment["dataset"]["trainable_reviews_after_dedup"]),
            "macro_f1": float(sentiment_test["macro_f1"]),
            "negative_recall": float(sentiment_test["negative_recall"]),
            "balanced_accuracy": float(sentiment_test["balanced_accuracy"]),
            "classification_report": sentiment_test["classification_report"],
            "confusion_matrix": sentiment_test["confusion_matrix"],
            "labels": sentiment_test["labels"],
            "split_method": sentiment["split"]["method"],
            "split_rows": sentiment["split"]["rows"],
            "split_places": sentiment["split"]["groups"],
            "group_overlap_count": int(sentiment["split"]["group_overlap_count"]),
            "model_name": sentiment["champion"]["model_name"],
        },
        "complaint": {
            "macro_f1": float(complaint_test["macro_f1"]),
            "negative_recall": float(complaint_test["negative_recall"]),
            "negative_precision": float(complaint_test["negative_precision"]),
            "confusion_matrix": complaint_test["confusion_matrix"],
            "labels": complaint_test["labels"],
            "support": complaint_test["support"],
            "model_name": "complaint_detector",
        },
        "aspect": {
            "gold_rows": int(aspect["training_rows"]),
            "micro_f1": float(aspect_test["micro_f1"]),
            "macro_f1": float(aspect_test["macro_f1"]),
            "hamming_loss": float(aspect_test["hamming_loss"]),
            "subset_accuracy": float(aspect_test["subset_accuracy"]),
            "per_aspect": aspect_test["per_aspect"],
            "split_method": aspect["split"]["method"],
            "split_rows": aspect["split"]["rows"],
            "split_places": aspect["split"]["groups"],
            "group_overlap_count": int(aspect["split"]["group_overlap_count"]),
            "model_name": "aspect_champion",
            "lainnya_total_gold": int(aspect_total_counts["lainnya"]),
            "lainnya_locked_test_support": int(aspect_test["per_aspect"]["lainnya"]["support"]),
        },
        "service_gap_validation": {
            "reviewed_rows": int(validation["reviewed_rows"]),
            "evidence_validity": float(validation["evidence_validity_rate"]),
            "priority_validity": float(validation["priority_validity_rate"]),
            "overall_validity": float(validation["validity_rate"]),
            "validator_count": 1,
        },
        "readiness": {
            "model_and_ranking_pipeline_ready": bool(readiness["model_and_ranking_pipeline_ready"]),
            "production_application_ready": bool(readiness["production_application_ready"]),
            "production_blockers": readiness["production_application_blockers"],
        },
        "sources": relative_source_map(config),
    }


def prepare_data_quality(config: dict[str, Any], reviews: pd.DataFrame) -> dict[str, Any]:
    profile = load_json(config["sources"]["data_profile"])
    cleaning = load_json(config["sources"]["cleaning_summary"])
    gold = load_json("outputs/reports/gold_dataset_manifest.json")
    aspect_gold = load_json("outputs/reports/aspect_gold_manifest.json")
    places = pd.read_parquet(ROOT / "data" / "processed" / "places_master.parquet")
    entity = pd.read_parquet(ROOT / "data" / "processed" / "entity_mapping.parquet")
    max_characters = int(config["max_review_characters"])
    valid_text = reviews["review_text_clean"].fillna("").astype(str).str.strip().ne("")
    nlp_rows = int((valid_text & ~reviews["is_duplicate"].fillna(False)).sum())
    valid_coordinates = int((places["latitude"].notna() & places["longitude"].notna()).sum())

    entity_candidates = entity[
        entity["match_method"].eq("normalized_exact")
        & entity["source_place_name"].astype(str).ne(entity["canonical_place_name"].astype(str))
    ]
    entity_row = entity_candidates.iloc[0]
    transformations = [
        {"Masalah Data": "Rating koma/titik", "Perlakuan": "Parsing numerik konsisten", "Dampak": "Rating dapat dianalisis"},
        {"Masalah Data": "Teks kosong", "Perlakuan": "Dikeluarkan dari korpus NLP", "Dampak": "Model hanya menerima teks valid"},
        {"Masalah Data": "Duplikasi", "Perlakuan": "Flag dan deduplikasi", "Dampak": "Mengurangi bias pengulangan"},
        {"Masalah Data": "Variasi nama tempat", "Perlakuan": "Normalisasi dan entity resolution", "Dampak": "Satu ID tempat kanonis"},
        {"Masalah Data": "Koordinat gabungan", "Perlakuan": "Parsing lintang dan bujur", "Dampak": "Analisis geospasial"},
        {"Masalah Data": "Metadata tersebar", "Perlakuan": "Integrasi lintas sheet", "Dampak": "Profil tempat lebih lengkap"},
        {"Masalah Data": "Kelas tidak seimbang", "Perlakuan": "Macro F1 dan group split", "Dampak": "Evaluasi lebih adil"},
    ]
    return {
        "summary": {
            "sheet_count": int(profile["sheet_count"]),
            "raw_review_rows": int(cleaning["review_rows"]),
            "reviews_with_text": int(cleaning["review_rows_with_text"]),
            "nlp_corpus_rows": nlp_rows,
            "canonical_places": int(cleaning["canonical_place_count"]),
            "sentiment_gold_rows": int(gold["validation"]["rows"]),
            "aspect_gold_rows": int(aspect_gold["validation"]["rows"]),
            "missing_review_text_rows": int(cleaning["review_rows"] - cleaning["review_rows_with_text"]),
            "duplicate_review_rows": int(cleaning["duplicate_review_rows"]),
            "entity_matches_needing_review": int(cleaning["entity_matches_needing_manual_review"]),
            "unresolved_excel_cells": int(cleaning["unresolved_excel_cells"]),
            "valid_coordinate_places": valid_coordinates,
            "missing_coordinate_places": int(len(places) - valid_coordinates),
        },
        "raw_examples": select_raw_examples(reviews, max_characters),
        "transformations": transformations,
        "cleaning_example": select_cleaning_example(reviews, max_characters),
        "entity_resolution_example": {
            "source_place_name": str(entity_row["source_place_name"]),
            "normalized_place_name": str(entity_row["normalized_place_name"]),
            "canonical_place_name": str(entity_row["canonical_place_name"]),
            "match_method": str(entity_row["match_method"]),
            "match_score": float(entity_row["match_score"]),
        },
        "important_notes": [
            "Negasi seperti tidak, kurang, dan belum dipertahankan.",
            "Metadata kosong tidak berarti fasilitas pasti tidak tersedia.",
            "Sebanyak 65 tempat belum memiliki koordinat valid dan tidak masuk seluruh analisis spasial.",
        ],
    }


REASON_LABELS = {
    "HIGH_NEGATIVE_RATE": "Rasio negatif tinggi",
    "FREQUENT_COMPLAINT": "Keluhan sering muncul",
    "LOW_NEARBY_SERVICE_DENSITY": "Layanan sekitar terbatas",
    "HIGH_REVIEW_CONFIDENCE": "Dukungan ulasan kuat",
    "LOW_DATA_RELIABILITY": "Reliabilitas data rendah",
}


def prepare_service_gap(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranking = pd.read_csv(ROOT / config["sources"]["service_gap_ranking"])
    validation = pd.read_csv(ROOT / config["sources"]["service_gap_validation"])
    top = ranking.sort_values("rank").head(10).copy()
    top["category"] = top["place_category"].astype(str).str.title()
    top["confidence"] = top["confidence_level"].astype(str).str.title()
    top["evidence_count"] = top["aspect_mention_count"].astype(int)
    top["main_reason"] = top["reason_codes_text"].fillna("").map(
        lambda value: REASON_LABELS.get(str(value).split("|")[0], str(value).split("|")[0].replace("_", " ").title())
    )
    output = top[
        ["rank", "place_name", "category", "aspect", "service_gap_score", "confidence", "evidence_count", "main_reason"]
    ].copy()
    output["aspect"] = output["aspect"].astype(str).str.replace("_", " ", regex=False).str.title()
    output["service_gap_score"] = output["service_gap_score"].astype(float).round(2)

    first = top.iloc[0]
    matched = validation[validation["rank"].astype(int).eq(int(first["rank"]))]
    if matched.empty:
        raise ValueError("Detail ranking teratas tidak ditemukan pada validasi manusia.")
    checked = matched.iloc[0]
    evidence = [
        redact_sensitive(checked[column])
        for column in ["evidence_clause_1", "evidence_clause_2", "evidence_clause_3"]
        if clean_text(checked.get(column, ""))
    ]
    detail = {
        "rank": int(first["rank"]),
        "place_name": str(first["place_name"]),
        "category": str(first["place_category"]).title(),
        "aspect": str(first["aspect"]).replace("_", " ").title(),
        "service_gap_score": round(float(first["service_gap_score"]), 2),
        "evidence_count": int(first["aspect_mention_count"]),
        "negative_evidence_count": int(first["negative_mention_count"]),
        "dominant_sentiment": f"Negatif ({int(first['negative_mention_count'])} dari {int(first['aspect_mention_count'])} bukti)",
        "complaint_evidence": int(first["negative_mention_count"]),
        "reason_codes": [str(value) for value in str(first["reason_codes_text"]).split("|") if value],
        "reason_labels": [REASON_LABELS.get(value, value.replace("_", " ").title()) for value in str(first["reason_codes_text"]).split("|") if value],
        "confidence": str(first["confidence_level"]).title(),
        "service_scarcity": round(float(first["service_scarcity"]), 4),
        "explanation": (
            f"{first['place_name']} berada pada prioritas analisis aspek {first['aspect']} karena "
            f"{int(first['negative_mention_count'])} dari {int(first['aspect_mention_count'])} bukti aspek "
            f"terindikasi negatif, didukung {int(first['review_count'])} ulasan, dan indikator kelangkaan "
            f"layanan sekitar bernilai {float(first['service_scarcity']):.2f}."
        ),
        "evidence_snippets": evidence,
        "human_validation": (
            "Bukti valid dan prioritas valid"
            if str(checked["manual_evidence_valid"]).strip().lower() == "yes"
            and str(checked["manual_priority_valid"]).strip().lower() == "yes"
            else "Memerlukan tinjauan"
        ),
        "validator_id": "A01",
        "disclaimer": "Ranking adalah prioritas analisis, bukan jaminan peluang bisnis.",
    }
    return output, detail


def prepare_error_examples(max_characters: int) -> list[dict[str, str]]:
    errors = pd.read_csv(ROOT / "outputs" / "reports" / "sentiment_error_analysis.csv")
    errors["text"] = errors["review_text_clean"].map(clean_text)
    errors = errors[errors["text"].map(lambda text: is_video_safe(text, max_characters))].copy()
    errors = errors.sample(frac=1, random_state=42).head(4)
    cause_labels = {
        "SHORT_TEXT": "Teks sangat pendek memberi konteks terbatas",
        "MIXED_SENTIMENT": "Sinyal positif dan negatif muncul bersamaan",
        "NEGATION": "Pola negasi masih sulit dibedakan",
        "RATING_TEXT_MISMATCH": "Rating dan isi teks tidak konsisten",
        "TYPO_OR_INFORMAL": "Bahasa informal atau typo",
        "GENERAL_MODEL_ERROR": "Pola bahasa belum terwakili dengan baik",
    }
    records = []
    for _, row in errors.iterrows():
        tags = [tag for tag in str(row["analysis_tags"]).split("|") if tag]
        cause = next((cause_labels[tag] for tag in tags if tag in cause_labels), "Perlu analisis data tambahan")
        records.append(
            {
                "Teks": redact_sensitive(row["text"]),
                "Gold": SENTIMENT_LABELS.get(str(row["actual_label"]), str(row["actual_label"])),
                "Prediksi": SENTIMENT_LABELS.get(str(row["predicted_label"]), str(row["predicted_label"])),
                "Tipe error": str(row["error_type"]).replace("_", " ").title(),
                "Dugaan penyebab": cause,
            }
        )
    return records


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 24,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str, tight: bool = True) -> None:
    options: dict[str, Any] = {"dpi": 120, "facecolor": "white"}
    if tight:
        options["bbox_inches"] = "tight"
    fig.savefig(FIGURE_DIR / name, **options)
    plt.close(fig)


def figure_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.6, 8.2, "Alur Analisis PodaNauli", color=NAVY, fontsize=28, fontweight="bold")
    boxes = [
        (0.7, 4.8, 2.2, 1.5, "Dataset\npanitia", BLUE),
        (3.25, 4.8, 2.4, 1.5, "Cleaning &\nentity resolution", TEAL),
        (6.1, 5.9, 2.2, 1.25, "Sentimen", ORANGE),
        (6.1, 4.35, 2.2, 1.25, "Complaint", RED),
        (6.1, 2.8, 2.2, 1.25, "Aspek\nmulti-label", GREEN),
        (8.9, 4.8, 2.4, 1.5, "Bukti\ntempat-aspek", BLUE),
        (11.85, 4.8, 2.2, 1.5, "Service Gap\nRanking", TEAL),
        (11.85, 2.2, 2.2, 1.3, "Validasi\nmanusia", NAVY),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05,rounding_size=0.12", facecolor=color, edgecolor="none")
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color="white", fontsize=16, fontweight="bold")
    arrows = [
        ((2.9, 5.55), (3.25, 5.55)),
        ((5.65, 5.55), (6.1, 6.5)),
        ((5.65, 5.55), (6.1, 5.0)),
        ((5.65, 5.55), (6.1, 3.4)),
        ((8.3, 6.5), (8.9, 5.8)),
        ((8.3, 5.0), (8.9, 5.55)),
        ((8.3, 3.4), (8.9, 5.2)),
        ((11.3, 5.55), (11.85, 5.55)),
        ((12.95, 4.8), (12.95, 3.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 2.5})
    ax.text(0.7, 0.75, "Output: prioritas analisis yang dapat ditelusuri ke bukti, bukan keputusan otomatis.", fontsize=17, color=GRAY)
    save_figure(fig, "pipeline_overview.png")


def figure_data_transformation(data_quality: dict[str, Any]) -> None:
    summary = data_quality["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.05, 1]})
    stages = ["Ulasan mentah", "Teks valid", "Korpus NLP"]
    values = [summary["raw_review_rows"], summary["reviews_with_text"], summary["nlp_corpus_rows"]]
    colors = [BLUE, ORANGE, TEAL]
    axes[0].bar(stages, values, color=colors, width=0.68)
    axes[0].set_title("Funnel Pengolahan Ulasan", color=NAVY, pad=20)
    axes[0].set_ylabel("Jumlah record")
    axes[0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        axes[0].text(index, value + 450, f"{value:,}".replace(",", "."), ha="center", fontweight="bold", color=NAVY)

    issues = ["Teks kosong", "Duplikat", "Entity review", "Koordinat kosong"]
    issue_values = [
        summary["missing_review_text_rows"],
        summary["duplicate_review_rows"],
        summary["entity_matches_needing_review"],
        summary["missing_coordinate_places"],
    ]
    axes[1].barh(issues, issue_values, color=[RED, ORANGE, BLUE, GRAY])
    axes[1].set_title("Isu Kualitas yang Dicatat", color=NAVY, pad=20)
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].invert_yaxis()
    for index, value in enumerate(issue_values):
        axes[1].text(value + max(issue_values) * 0.02, index, f"{value:,}".replace(",", "."), va="center", fontweight="bold", color=NAVY)
    fig.suptitle("Transformasi Data PodaNauli", color=NAVY, fontsize=28, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "data_transformation.png")


def figure_model_metrics(metrics: dict[str, Any]) -> None:
    labels = ["Sentimen\nMacro F1", "Sentimen\nRecall negatif", "Complaint\nMacro F1", "Complaint\nRecall", "Aspek\nMicro F1", "Aspek\nMacro F1"]
    values = [
        metrics["sentiment"]["macro_f1"], metrics["sentiment"]["negative_recall"],
        metrics["complaint"]["macro_f1"], metrics["complaint"]["negative_recall"],
        metrics["aspect"]["micro_f1"], metrics["aspect"]["macro_f1"],
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(labels, values, color=[BLUE, BLUE, RED, RED, TEAL, TEAL], width=0.68)
    ax.set_ylim(0, 1.05)
    ax.set_title("Evaluasi Locked Test Model PodaNauli", color=NAVY, pad=22)
    ax.set_ylabel("Nilai metrik")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.4f}", ha="center", fontweight="bold", color=NAVY)
    fig.text(0.06, 0.025, "Macro F1 dipakai untuk memperhatikan kelas yang tidak seimbang.", color=GRAY, fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, "model_metrics.png")


def figure_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str, name: str) -> None:
    values = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(16, 9))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Gold label")
    ax.set_title(title, color=NAVY, pad=22)
    threshold = values.max() / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=24, fontweight="bold", color="white" if values[i, j] > threshold else NAVY)
    fig.subplots_adjust(left=0.24, right=0.80, top=0.86, bottom=0.16)
    save_figure(fig, name, tight=False)


def figure_aspect_metrics(metrics: dict[str, Any]) -> None:
    rows = [(key, value["f1"], value["support"]) for key, value in metrics["aspect"]["per_aspect"].items()]
    rows.sort(key=lambda item: item[1])
    labels = [label.replace("_", " ").title() for label, _, _ in rows]
    values = [float(value) for _, value, _ in rows]
    supports = [int(support) for _, _, support in rows]
    colors = [RED if support <= 3 else TEAL if value >= 0.7 else ORANGE for value, support in zip(values, supports)]
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1.08)
    ax.set_title("F1 per Aspek pada Locked Test", color=NAVY, pad=18)
    ax.set_xlabel("F1")
    ax.grid(axis="x", alpha=0.2)
    for bar, value, support in zip(bars, values, supports):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}  (n={support})", va="center", fontsize=11, color=NAVY)
    fig.tight_layout()
    save_figure(fig, "aspect_metrics.png")


def figure_service_gap_validation(metrics: dict[str, Any]) -> None:
    labels = ["Validitas bukti", "Validitas prioritas", "Validitas keseluruhan"]
    values = [
        metrics["service_gap_validation"]["evidence_validity"],
        metrics["service_gap_validation"]["priority_validity"],
        metrics["service_gap_validation"]["overall_validity"],
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(labels, values, color=[BLUE, TEAL, ORANGE], width=0.62)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color=RED, linestyle="--", linewidth=2, label="Gate minimum 0,80")
    ax.set_title("Validasi Manusia Service Gap Top-20", color=NAVY, pad=22)
    ax.set_ylabel("Proporsi valid")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontweight="bold", fontsize=18, color=NAVY)
    ax.text(0.01, 0.02, "Cakupan validasi terbatas pada 20 peringkat teratas dan satu validator.", transform=ax.transAxes, color=GRAY)
    fig.tight_layout()
    save_figure(fig, "service_gap_validation.png")


def figure_service_gap_top10(service_gap: pd.DataFrame) -> None:
    frame = service_gap.sort_values("rank", ascending=False)
    labels = [f"#{int(row['rank'])} {row['place_name']} - {row['aspect']}" for _, row in frame.iterrows()]
    values = frame["service_gap_score"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh(labels, values, color=[TEAL if rank <= 3 else BLUE for rank in frame["rank"]])
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_title("Service Gap Ranking Top-10", color=NAVY, pad=20)
    ax.set_xlabel("Service Gap Score (0-100)")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontweight="bold", color=NAVY)
    ax.text(0.01, -0.10, "Prioritas analisis berdasarkan bukti dalam dataset; bukan prediksi keuntungan.", transform=ax.transAxes, color=GRAY)
    fig.tight_layout()
    save_figure(fig, "service_gap_top10.png")


def markdown_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {}, "outputs": [], "source": source.strip().splitlines(keepends=True)}


def build_notebook() -> None:
    cells = [
        markdown_cell(
            """
# PodaNauli
## Demonstrasi Analisis Service Gap Pariwisata Danau Toba

**Masalah:** ulasan wisata tersebar dan sulit diterjemahkan menjadi prioritas layanan yang dapat ditindaklanjuti.

**Solusi:** PodaNauli menggabungkan sentimen, complaint, aspek multi-label, metadata tempat, dan bukti ulasan menjadi Service Gap Ranking yang transparan.

**Pengguna:** pengelola destinasi, pemerintah daerah, pelaku UMKM, dan analis pariwisata.

> Hasil merupakan sistem pendukung analisis. Ranking bukan prediksi keuntungan, keputusan otomatis, atau pengganti validasi lapangan.
""",
            "title-purpose",
        ),
        code_cell(
            """
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = next(
    (
        candidate
        for candidate in (Path.cwd(), *Path.cwd().parents)
        if (candidate / "demo" / "demo_runtime.py").exists()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Root proyek tidak ditemukan. Buka notebook dari folder repository atau folder demo.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.demo_runtime import (
    display_data_quality_summary,
    display_metrics_summary,
    display_place_detail,
    display_prediction_result,
    display_service_gap_ranking,
    load_demo_bundle,
    predict_reviews,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 160)

DEMO_DIR = PROJECT_ROOT / "demo"
metrics = json.loads((DEMO_DIR / "demo_metrics.json").read_text(encoding="utf-8"))
quality = json.loads((DEMO_DIR / "demo_data_quality.json").read_text(encoding="utf-8"))
place_detail = json.loads((DEMO_DIR / "demo_place_detail.json").read_text(encoding="utf-8"))
demo_reviews = pd.read_csv(DEMO_DIR / "demo_reviews.csv")
service_gap = pd.read_csv(DEMO_DIR / "demo_service_gap.csv")
print("[OK] Paket demo lokal siap digunakan")
""",
            "setup",
        ),
        markdown_cell("## 1. Karakter Dataset\n\nAngka berikut dibaca dari artefak aktual, bukan ditulis ulang secara manual.", "dataset-heading"),
        code_cell(
            """
display(display_data_quality_summary(quality))
display(pd.DataFrame(quality["raw_examples"]))
""",
            "dataset-summary",
        ),
        markdown_cell("## 2. Transformasi Data\n\nTeks kosong dan duplikat dipisahkan dari korpus NLP, variasi nama tempat disatukan, dan negasi seperti *tidak*, *kurang*, serta *belum* tetap dipertahankan.", "transform-heading"),
        code_cell(
            """
display(pd.DataFrame(quality["transformations"]))
display(pd.DataFrame([quality["cleaning_example"]]).rename(columns={"before": "Sebelum", "after": "Sesudah", "source": "Sumber"}))
display(pd.DataFrame([quality["entity_resolution_example"]]))
display(Image(filename=str(DEMO_DIR / "figures" / "data_transformation.png"), width=1050))
""",
            "transform-output",
        ),
        markdown_cell("## 3. Cara Kerja Tiga Model\n\nSentimen menentukan polaritas umum, complaint menyaring indikasi keluhan, dan aspek menentukan bidang layanan yang dibahas. Satu ulasan dapat memiliki beberapa aspek.", "models-heading"),
        code_cell(
            """
model_roles = pd.DataFrame([
    {"Model": "Sentimen", "Fungsi": "Polaritas umum", "Output": "Negatif / Netral / Positif", "Peran": "Memberi konteks umum"},
    {"Model": "Complaint", "Fungsi": "Deteksi keluhan", "Output": "Terdeteksi / Tidak / Tinjau", "Peran": "Menyaring bukti negatif"},
    {"Model": "Aspek", "Fungsi": "Klasifikasi multi-label", "Output": "Satu atau beberapa aspek", "Peran": "Menentukan bidang layanan"},
])
display(model_roles)
display(Image(filename=str(DEMO_DIR / "figures" / "pipeline_overview.png"), width=1050))
""",
            "model-roles",
        ),
        markdown_cell("## 4. Muat Model Champion\n\nSel ini hanya memuat model tersimpan pada CPU. Tidak ada training atau akses internet.", "load-heading"),
        code_cell("bundle = load_demo_bundle(print_status=True)", "load-models"),
        markdown_cell("## 5. Inferensi Tiga Input Manual\n\nInput A bersifat campuran, input B positif, dan input C memuat complaint akses.", "manual-heading"),
        code_cell(
            """
manual_inputs = [
    "Pemandangannya sangat indah, tetapi toilet kurang bersih dan area parkir sempit.",
    "Pelayanannya ramah, tempatnya nyaman, dan makanan disajikan dengan cepat.",
    "Akses jalan rusak, petunjuk arah kurang jelas, dan kendaraan sulit mencapai lokasi.",
]
manual_predictions = predict_reviews(manual_inputs, bundle)
display(display_prediction_result(manual_predictions))
""",
            "manual-inference",
        ),
        markdown_cell("## 6. Contoh Dataset Aktual\n\nContoh diambil dari locked test secara deterministik, telah dianonimkan, dan sengaja mencakup minimal satu kesalahan model.", "actual-heading"),
        code_cell(
            """
actual_view = demo_reviews[["place_name", "review_text", "expected_sentiment", "predicted_sentiment", "predicted_complaint", "predicted_aspects", "is_error_example"]].copy()
actual_view.columns = ["Tempat", "Ulasan", "Gold", "Prediksi", "Complaint", "Aspek", "Contoh error"]
display(actual_view)
""",
            "actual-examples",
        ),
        markdown_cell("## 7. Evaluasi Sentimen\n\nMacro F1 digunakan karena distribusi kelas tidak seimbang. Recall negatif penting untuk mengurangi risiko keluhan yang terlewat.", "sentiment-heading"),
        code_cell(
            """
display(display_metrics_summary(metrics).query("Model == 'Sentimen'"))
display(Image(filename=str(DEMO_DIR / "figures" / "sentiment_confusion_matrix.png"), width=900))
""",
            "sentiment-eval",
        ),
        markdown_cell("## 8. Evaluasi Complaint\n\nSentimen mengukur polaritas umum, sedangkan complaint berfokus pada indikasi keluhan yang dapat ditindaklanjuti.", "complaint-heading"),
        code_cell(
            """
display(display_metrics_summary(metrics).query("Model == 'Complaint'"))
display(pd.DataFrame([metrics["complaint"]["support"]]).rename(columns={"non_negative": "Non-complaint", "negative": "Complaint"}))
display(Image(filename=str(DEMO_DIR / "figures" / "complaint_confusion_matrix.png"), width=900))
""",
            "complaint-eval",
        ),
        markdown_cell("## 9. Evaluasi Aspek Multi-label\n\nMicro F1 menunjukkan performa agregat, Macro F1 pemerataan antaraspek, Hamming loss proporsi keputusan label yang salah, dan subset accuracy menuntut seluruh label tepat sekaligus.", "aspect-heading"),
        code_cell(
            """
aspect_summary = pd.DataFrame([
    {"Metrik": "Micro F1", "Nilai": f"{metrics['aspect']['micro_f1']:.4f}"},
    {"Metrik": "Macro F1", "Nilai": f"{metrics['aspect']['macro_f1']:.4f}"},
    {"Metrik": "Hamming loss", "Nilai": f"{metrics['aspect']['hamming_loss']:.4f}"},
    {"Metrik": "Subset accuracy", "Nilai": f"{metrics['aspect']['subset_accuracy']:.4f}"},
])
display(aspect_summary)
display(Image(filename=str(DEMO_DIR / "figures" / "aspect_metrics.png"), width=1100))
print(f"Label 'lainnya': {metrics['aspect']['lainnya_total_gold']} gold total; support locked test {metrics['aspect']['lainnya_locked_test_support']}.")
""",
            "aspect-eval",
        ),
        markdown_cell("## 10. Locked Test dan Pencegahan Leakage\n\nSplit dilakukan berdasarkan tempat. Tempat yang sama tidak tersebar bebas antar-split, sehingga model diuji pada kelompok tempat yang tidak digunakan saat training.", "split-heading"),
        code_cell(
            """
split_rows = []
for model_name, model_key in [("Sentimen", "sentiment"), ("Aspek", "aspect")]:
    for split_name in ["train", "validation", "test"]:
        split_rows.append({"Model": model_name, "Split": split_name.title(), "Baris": metrics[model_key]["split_rows"][split_name], "Tempat": metrics[model_key]["split_places"][split_name]})
display(pd.DataFrame(split_rows))
print("Overlap tempat antarsplit:", metrics["sentiment"]["group_overlap_count"], "(sentimen),", metrics["aspect"]["group_overlap_count"], "(aspek)")
""",
            "split-output",
        ),
        markdown_cell("## 11. Service Gap Ranking\n\nRanking menunjukkan prioritas analisis berdasarkan bukti yang tersedia dalam dataset.", "ranking-heading"),
        code_cell(
            """
display(display_service_gap_ranking(service_gap))
display(Image(filename=str(DEMO_DIR / "figures" / "service_gap_top10.png"), width=1100))
""",
            "ranking-output",
        ),
        markdown_cell("## 12. Detail Peringkat Teratas\n\nSetiap ranking membawa jumlah bukti, reason code, confidence, dan potongan evidence yang telah dianonimkan.", "detail-heading"),
        code_cell(
            """
display(display_place_detail(place_detail))
display(pd.DataFrame({"Reason code": place_detail["reason_codes"], "Makna": place_detail["reason_labels"]}))
display(pd.DataFrame({"Potongan bukti": place_detail["evidence_snippets"]}))
print(place_detail["disclaimer"])
""",
            "detail-output",
        ),
        markdown_cell("## 13. Validasi Manusia Top-20\n\nValidasi terbatas pada 20 peringkat teratas dan satu validator. Validitas bukti tepat berada pada gate minimum 0,80.", "validation-heading"),
        code_cell(
            """
validation = metrics["service_gap_validation"]
display(pd.DataFrame([
    {"Metrik": "Evidence validity", "Nilai": f"{validation['evidence_validity']:.2f}"},
    {"Metrik": "Priority validity", "Nilai": f"{validation['priority_validity']:.2f}"},
    {"Metrik": "Overall validity", "Nilai": f"{validation['overall_validity']:.2f}"},
]))
display(Image(filename=str(DEMO_DIR / "figures" / "service_gap_validation.png"), width=1000))
""",
            "validation-output",
        ),
        markdown_cell("## 14. Error Analysis\n\nKesalahan aktual tetap ditampilkan agar batas penggunaan model dapat dijelaskan secara jujur.", "error-heading"),
        code_cell("display(pd.DataFrame(metrics[\"sentiment_error_examples\"]))", "error-output"),
        markdown_cell("## 15. Nilai Manfaat", "benefit-heading"),
        code_cell(
            """
benefits = pd.DataFrame([
    {"Pengguna": "Pengelola", "Informasi": "Keluhan, aspek, dan bukti", "Keputusan yang didukung": "Prioritas perbaikan layanan"},
    {"Pengguna": "Pemerintah", "Informasi": "Pola lintas tempat dan lokasi", "Keputusan yang didukung": "Agenda verifikasi lapangan"},
    {"Pengguna": "UMKM", "Informasi": "Sinyal kebutuhan layanan", "Keputusan yang didukung": "Hipotesis layanan, bukan jaminan keuntungan"},
    {"Pengguna": "Analis", "Informasi": "Data terintegrasi dan metrik", "Keputusan yang didukung": "Analisis yang dapat ditelusuri"},
])
display(benefits)
""",
            "benefit-output",
        ),
        markdown_cell("## 16. Keterbatasan", "limitations-heading"),
        code_cell(
            """
limitations = [
    "Human gold dibuat oleh satu anotator A01; inter-annotator agreement belum tersedia.",
    "Saran AI terlihat saat anotasi sehingga confirmation bias masih mungkin terjadi.",
    "Validasi ranking baru mencakup top-20 dan evidence validity tepat 0,80.",
    "Label aspek 'lainnya' hanya memiliki tiga gold secara keseluruhan.",
    "Metadata kosong bukan bukti bahwa fasilitas tidak tersedia.",
    "Hasil terbatas pada distribusi dataset ini dan tetap memerlukan validasi manusia.",
    "Sistem belum dinyatakan siap produksi.",
]
display(pd.DataFrame({"Keterbatasan yang perlu diperhatikan": limitations}))
""",
            "limitations-output",
        ),
        markdown_cell(
            """
## 17. Penutup

**PodaNauli mengubah suara wisatawan menjadi petunjuk berbasis data untuk membantu memahami prioritas layanan pariwisata Danau Toba.**

Model dan ranking pipeline memenuhi acceptance gate untuk analisis pada dataset ini, tetapi belum dinyatakan siap produksi.
""",
            "closing",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(DEMO_DIR / "01_podanauli_video_demo.ipynb", notebook)


def main() -> int:
    started_at = datetime.now().astimezone()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "scratch" / "matplotlib_demo").mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    bundle = load_demo_bundle(print_status=False)
    reviews = pd.read_parquet(ROOT / "data" / "processed" / "reviews_clean.parquet")
    sentiment_split = pd.read_parquet(ROOT / "data" / "processed" / "sentiment_gold_split.parquet")
    demo_reviews = select_demo_reviews(
        reviews,
        sentiment_split,
        bundle,
        max_examples=int(config["max_actual_examples"]),
        max_characters=int(config["max_review_characters"]),
    )
    demo_reviews.to_csv(DEMO_DIR / "demo_reviews.csv", index=False, encoding="utf-8-sig")

    metrics = prepare_metrics(config)
    metrics["sentiment_error_examples"] = prepare_error_examples(int(config["max_review_characters"]))
    write_json(DEMO_DIR / "demo_metrics.json", metrics)

    data_quality = prepare_data_quality(config, reviews)
    write_json(DEMO_DIR / "demo_data_quality.json", data_quality)

    service_gap, place_detail = prepare_service_gap(config)
    service_gap.to_csv(DEMO_DIR / "demo_service_gap.csv", index=False, encoding="utf-8-sig")
    write_json(DEMO_DIR / "demo_place_detail.json", place_detail)

    manual_texts = [item["text"] for item in config["manual_inputs"]]
    manual_predictions = predict_reviews(manual_texts, bundle).rename(
        columns={"Review": "review_text", "Sentimen": "predicted_sentiment", "Complaint": "predicted_complaint", "Aspek": "predicted_aspects", "Catatan": "notes"}
    )
    manual_predictions.insert(0, "demo_id", [f"MANUAL_{item['id']}" for item in config["manual_inputs"]])
    actual_predictions = demo_reviews.copy()
    actual_predictions.insert(1, "source_type", "locked_test")
    manual_predictions.insert(1, "source_type", "manual")
    combined_columns = sorted(set(actual_predictions.columns) | set(manual_predictions.columns))
    pd.concat(
        [manual_predictions.reindex(columns=combined_columns), actual_predictions.reindex(columns=combined_columns)],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "demo_predictions.csv", index=False, encoding="utf-8-sig")

    configure_plot()
    figure_pipeline_overview()
    figure_data_transformation(data_quality)
    figure_model_metrics(metrics)
    figure_confusion_matrix(metrics["sentiment"]["confusion_matrix"], ["Negatif", "Netral", "Positif"], "Confusion Matrix Sentimen - Locked Test", "sentiment_confusion_matrix.png")
    figure_confusion_matrix(metrics["complaint"]["confusion_matrix"], ["Non-complaint", "Complaint"], "Confusion Matrix Complaint - Locked Test", "complaint_confusion_matrix.png")
    figure_aspect_metrics(metrics)
    figure_service_gap_validation(metrics)
    figure_service_gap_top10(service_gap)
    build_notebook()

    elapsed = (datetime.now().astimezone() - started_at).total_seconds()
    log_lines = [
        f"started_at={started_at.isoformat(timespec='seconds')}",
        "mode=offline_cpu_no_training",
        f"demo_reviews={len(demo_reviews)}",
        "manual_inputs=3",
        "figures=8",
        f"preparation_seconds={elapsed:.3f}",
        "status=PREPARED",
    ]
    (OUTPUT_DIR / "demo_execution_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("[OK] Data demo anonim dibuat")
    print("[OK] Delapan figure demo dibuat")
    print("[OK] Notebook demo dibuat")
    print(f"[OK] Persiapan selesai dalam {elapsed:.2f} detik")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
