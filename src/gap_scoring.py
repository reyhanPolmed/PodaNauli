"""Tahap 8 pipeline: transparent Service Gap Score calculation."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"
PREDICTION_DIR = ROOT / "outputs" / "predictions"
CONFIG_PATH = ROOT / "configs" / "gap_score_weights.yaml"
ASPECT_TAXONOMY_PATH = ROOT / "configs" / "aspect_taxonomy.yaml"
RANKING_VALIDATION_PATH = REPORT_DIR / "service_gap_top20_validation.csv"
RANKING_VALIDATION_PENDING_PATH = REPORT_DIR / "service_gap_top20_validation.pending.csv"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_gap_config() -> dict[str, Any]:
    config = load_yaml(CONFIG_PATH)
    if not config:
        raise FileNotFoundError(f"Missing gap score config: {CONFIG_PATH}")
    return config


def load_aspect_ids() -> list[str]:
    data = load_yaml(ASPECT_TAXONOMY_PATH)
    return [aspect["id"] for aspect in data.get("aspects", []) if aspect["id"] != "lainnya"]


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def bayesian_smoothed_rate(negative_count: int, total_count: int, alpha: float, beta: float) -> float:
    """Bayesian smoothing for negative rate to avoid tiny-sample extremes."""
    return float((negative_count + alpha) / (total_count + alpha + beta))


def review_volume_confidence(review_count: int, confidence_review_count: int) -> float:
    """Saturating confidence from review volume."""
    if confidence_review_count <= 0:
        return 1.0
    return clamp(math.log1p(review_count) / math.log1p(confidence_review_count))


def rating_gap(place_rating: Any) -> float:
    """Normalize rating gap where rating below 5 implies more room for improvement."""
    if pd.isna(place_rating):
        return 0.5
    return clamp((5.0 - float(place_rating)) / 4.0)


def metadata_completeness(row: pd.Series) -> float:
    """Estimate metadata completeness from fields useful for service intelligence."""
    fields = [
        "place_category",
        "place_type",
        "address",
        "latitude",
        "longitude",
        "place_rating",
        "status",
        "price_text_original",
        "facility_text",
    ]
    present = 0
    total = 0
    for field in fields:
        if field not in row.index:
            continue
        total += 1
        value = row[field]
        if isinstance(value, (list, tuple, np.ndarray)):
            present += int(len(value) > 0)
        elif pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
            present += 1
    return float(present / total) if total else 0.0


def haversine_distance_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6371.0088
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return pd.Series(2 * radius * np.arcsin(np.sqrt(a)), index=lat2.index)


def compute_service_scarcity(places: pd.DataFrame, radius_km: float) -> pd.Series:
    """Estimate scarcity as inverse nearby place density in the same broad category."""
    valid = (
        places["latitude"].notna()
        & places["longitude"].notna()
        & places["latitude"].between(-90, 90)
        & places["longitude"].between(-180, 180)
    )
    densities: dict[str, int] = {}
    for idx, row in places.iterrows():
        if not valid.loc[idx]:
            densities[row["canonical_place_id"]] = 0
            continue
        category_mask = places["place_category"].fillna("") == row.get("place_category")
        valid_neighbors = places[valid & category_mask].copy()
        distances = haversine_distance_km(float(row["latitude"]), float(row["longitude"]), valid_neighbors["latitude"], valid_neighbors["longitude"])
        densities[row["canonical_place_id"]] = int((distances <= radius_km).sum() - 1)

    density_series = pd.Series(densities)
    max_density = max(int(density_series.max()), 1)
    scarcity = 1.0 - (density_series / max_density)
    return scarcity.clip(0.0, 1.0)


def split_aspects(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def build_review_aspect_rows(aspect_predictions: pd.DataFrame, aspect_ids: list[str]) -> pd.DataFrame:
    """Explode weak aspect prediction rows into one row per review-aspect mention."""
    if {"review_id", "canonical_place_id", "aspect", "is_negative"}.issubset(aspect_predictions.columns):
        clause_mentions = aspect_predictions[aspect_predictions["aspect"].isin(aspect_ids)].copy()
        clause_mentions["is_negative"] = clause_mentions["is_negative"].fillna(False).astype(bool)
        if "sentiment_label" not in clause_mentions.columns:
            clause_mentions["sentiment_label"] = np.where(
                clause_mentions["is_negative"],
                "negative",
                "non_negative",
            )
        return clause_mentions[
            [
                "review_id",
                "canonical_place_id",
                "aspect",
                "is_negative",
                "sentiment_label",
            ]
        ].reset_index(drop=True)

    records = []
    for _, row in aspect_predictions.iterrows():
        weak_aspects = set(split_aspects(row.get("weak_aspects_text")))
        negative_aspects = set(split_aspects(row.get("weak_negative_aspects_text")))
        if "lainnya" in weak_aspects:
            weak_aspects.remove("lainnya")
        for aspect_id in sorted(weak_aspects):
            if aspect_id not in aspect_ids:
                continue
            is_negative = aspect_id in negative_aspects or row.get("weak_sentiment_label") == "negative"
            records.append(
                {
                    "review_id": row["review_id"],
                    "canonical_place_id": row["canonical_place_id"],
                    "aspect": aspect_id,
                    "is_negative": bool(is_negative),
                    "weak_sentiment_label": row.get("weak_sentiment_label"),
                }
            )
    return pd.DataFrame(records)


def latest_review_recency_factor(reviews: pd.DataFrame, canonical_place_id: str, recency_days: int) -> float:
    if "review_date" not in reviews.columns:
        return 0.5
    place_dates = pd.to_datetime(
        reviews.loc[reviews["canonical_place_id"] == canonical_place_id, "review_date"],
        errors="coerce",
    ).dropna()
    if place_dates.empty:
        return 0.5
    latest = place_dates.max()
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    age_days = max((now - latest).days, 0)
    return clamp(1.0 - (age_days / max(recency_days, 1)))


def priority_level(score: float) -> str:
    if score >= 70:
        return "tinggi"
    if score >= 40:
        return "menengah"
    return "rendah"


def confidence_level(review_confidence: float, data_reliability: float) -> str:
    value = (review_confidence + data_reliability) / 2
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def reason_codes_for_row(row: pd.Series, thresholds: dict[str, Any]) -> list[str]:
    codes = []
    if row["negative_rate_smoothed"] >= float(thresholds["high_negative_rate"]):
        codes.append("HIGH_NEGATIVE_RATE")
    if row["aspect_mention_frequency"] >= float(thresholds["frequent_complaint"]):
        codes.append("FREQUENT_COMPLAINT")
    if row["service_scarcity"] >= float(thresholds["low_nearby_service_density"]):
        codes.append("LOW_NEARBY_SERVICE_DENSITY")
    if row["review_volume_confidence"] >= float(thresholds["high_review_confidence"]):
        codes.append("HIGH_REVIEW_CONFIDENCE")
    if row["data_reliability"] <= float(thresholds["low_data_reliability"]):
        codes.append("LOW_DATA_RELIABILITY")
    return codes or ["BASELINE_SIGNAL"]


def explanation_for_row(row: pd.Series) -> str:
    negative_percentage = round(float(row["negative_rate_smoothed"]) * 100, 1)
    scarcity_level = "tinggi" if row["service_scarcity"] >= 0.7 else "menengah" if row["service_scarcity"] >= 0.35 else "rendah"
    return (
        f"{row['place_name']} memiliki prioritas {row['priority_level']} pada aspek {row['aspect']} "
        f"karena {negative_percentage}% penyebutan aspek tersebut bersentimen negatif setelah smoothing, "
        f"berdasarkan {int(row['review_count'])} ulasan, dan ketersediaan layanan terkait di sekitar lokasi tergolong {scarcity_level}."
    )


def compute_service_gap_scores(
    places: pd.DataFrame,
    reviews: pd.DataFrame,
    aspect_predictions: pd.DataFrame,
    config: dict[str, Any],
    aspect_ids: list[str],
) -> pd.DataFrame:
    smoothing = config["bayesian_smoothing"]
    weights = config["weights"]
    thresholds = config["thresholds"]
    scarcity_config = config["service_scarcity"]

    alpha = float(smoothing["alpha"])
    beta = float(smoothing["beta"])
    confidence_review_count = int(thresholds["volume_confidence_review_count"])
    recency_days = int(thresholds["recency_days"])

    mentions = build_review_aspect_rows(aspect_predictions, aspect_ids)
    review_counts = reviews.groupby("canonical_place_id")["review_id"].nunique().to_dict()
    place_scarcity = compute_service_scarcity(places, float(scarcity_config["category_density_radius_km"]))
    max_aspect_mentions = max(int(mentions.groupby(["canonical_place_id", "aspect"]).size().max()) if not mentions.empty else 0, 1)

    rows = []
    for _, place in places.iterrows():
        place_id = place["canonical_place_id"]
        place_mentions = mentions[mentions["canonical_place_id"] == place_id] if not mentions.empty else pd.DataFrame()
        total_place_reviews = int(review_counts.get(place_id, 0))
        metadata_complete = metadata_completeness(place)
        data_reliability = clamp((metadata_complete + review_volume_confidence(total_place_reviews, confidence_review_count)) / 2)
        for aspect_id in aspect_ids:
            aspect_rows = place_mentions[place_mentions["aspect"] == aspect_id] if not place_mentions.empty else pd.DataFrame()
            mention_count = int(len(aspect_rows))
            if mention_count == 0 and total_place_reviews < int(scarcity_config["scarcity_review_count_floor"]):
                continue
            negative_count = int(aspect_rows["is_negative"].sum()) if mention_count else 0
            negative_rate_smoothed = bayesian_smoothed_rate(negative_count, mention_count, alpha, beta)
            mention_frequency = clamp(mention_count / max(total_place_reviews, 1))
            mention_frequency_normalized = clamp(mention_count / max_aspect_mentions)
            volume_confidence = review_volume_confidence(total_place_reviews, confidence_review_count)
            scarcity = float(place_scarcity.get(place_id, 0.5))
            rating_gap_value = rating_gap(place.get("place_rating"))
            recency = latest_review_recency_factor(reviews, place_id, recency_days)
            metadata_gap = 1.0 - metadata_complete

            base_score = (
                float(weights["negative_sentiment_rate_smoothed"]) * negative_rate_smoothed
                + float(weights["aspect_mention_frequency"]) * mention_frequency_normalized
                + float(weights["service_scarcity"]) * scarcity
                + float(weights["review_volume_confidence"]) * volume_confidence
                + float(weights["rating_gap"]) * rating_gap_value
                + float(weights["recency_factor"]) * recency
                + float(weights["metadata_completeness_gap"]) * metadata_gap
            )
            final_score = clamp(base_score) * 100.0 * data_reliability
            rows.append(
                {
                    "canonical_place_id": place_id,
                    "place_name": place["canonical_place_name"],
                    "place_category": place.get("place_category"),
                    "aspect": aspect_id,
                    "service_gap_score": round(float(final_score), 4),
                    "confidence_level": confidence_level(volume_confidence, data_reliability),
                    "review_count": total_place_reviews,
                    "aspect_mention_count": mention_count,
                    "negative_mention_count": negative_count,
                    "negative_rate_smoothed": round(float(negative_rate_smoothed), 6),
                    "aspect_mention_frequency": round(float(mention_frequency), 6),
                    "aspect_mention_frequency_normalized": round(float(mention_frequency_normalized), 6),
                    "service_scarcity": round(float(scarcity), 6),
                    "review_volume_confidence": round(float(volume_confidence), 6),
                    "metadata_completeness": round(float(metadata_complete), 6),
                    "data_reliability": round(float(data_reliability), 6),
                    "rating_gap_normalized": round(float(rating_gap_value), 6),
                    "recency_factor": round(float(recency), 6),
                    "base_score": round(float(base_score), 6),
                }
            )

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    scores["priority_level"] = scores["service_gap_score"].map(priority_level)
    scores["reason_codes"] = scores.apply(lambda row: reason_codes_for_row(row, thresholds), axis=1)
    scores["reason_codes_text"] = scores["reason_codes"].map(lambda values: "|".join(values))
    scores["explanation"] = scores.apply(explanation_for_row, axis=1)
    scores = scores.sort_values(["service_gap_score", "negative_mention_count", "aspect_mention_count"], ascending=False).reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1)
    return scores


def update_cluster_average_gap(scores: pd.DataFrame) -> None:
    clusters_path = PROCESSED_DIR / "place_clusters.parquet"
    if not clusters_path.exists() or scores.empty:
        return
    clusters = pd.read_parquet(clusters_path)
    top_scores = scores.groupby("canonical_place_id")["service_gap_score"].max().reset_index()
    merged = clusters.drop(columns=["average_gap_score"], errors="ignore").merge(top_scores, on="canonical_place_id", how="left")
    cluster_avg = merged.groupby("geo_cluster_id")["service_gap_score"].mean().rename("average_gap_score").reset_index()
    merged = merged.merge(cluster_avg, on="geo_cluster_id", how="left")
    merged = merged.drop(columns=["service_gap_score"], errors="ignore")
    merged.to_parquet(clusters_path, index=False)

    summary_path = REPORT_DIR / "geospatial_cluster_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary.drop(columns=["average_gap_score"], errors="ignore").merge(cluster_avg, on="geo_cluster_id", how="left")
        summary.to_csv(summary_path, index=False, encoding="utf-8")

    geojson_path = ROOT / "outputs" / "maps" / "place_clusters.geojson"
    if geojson_path.exists():
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        avg_lookup = cluster_avg.set_index("geo_cluster_id")["average_gap_score"].to_dict()
        for feature in geojson.get("features", []):
            properties = feature.get("properties", {})
            cluster_id = properties.get("geo_cluster_id")
            if cluster_id in avg_lookup and pd.notna(avg_lookup[cluster_id]):
                properties["average_gap_score"] = round(float(avg_lookup[cluster_id]), 4)
        geojson_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")


def export_service_gap_validation_sample(
    scores: pd.DataFrame,
    top_n: int = 20,
    aspect_sentiment: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Export a stable human-audit sheet for the highest-ranked service gaps."""
    columns = [
        "rank",
        "canonical_place_id",
        "place_name",
        "place_category",
        "aspect",
        "service_gap_score",
        "negative_mention_count",
        "aspect_mention_count",
        "explanation",
    ]
    sample = scores.head(top_n)[columns].copy()
    for evidence_index in range(1, 4):
        sample[f"evidence_clause_{evidence_index}"] = ""
    if aspect_sentiment is not None and {
        "canonical_place_id",
        "aspect",
        "clause_text",
        "is_negative",
    }.issubset(aspect_sentiment.columns):
        negative_evidence = aspect_sentiment[aspect_sentiment["is_negative"].fillna(False)].copy()
        if "prediction_confidence" in negative_evidence.columns:
            negative_evidence = negative_evidence.sort_values(
                "prediction_confidence",
                ascending=False,
            )
        evidence_lookup = (
            negative_evidence.drop_duplicates(
                ["canonical_place_id", "aspect", "clause_text"]
            )
            .groupby(["canonical_place_id", "aspect"])["clause_text"]
            .apply(lambda values: [str(value) for value in values.head(3)])
            .to_dict()
        )
        for index, row in sample.iterrows():
            evidence = evidence_lookup.get(
                (row["canonical_place_id"], row["aspect"]),
                [],
            )
            for evidence_index, clause in enumerate(evidence, start=1):
                sample.loc[index, f"evidence_clause_{evidence_index}"] = clause
    manual_columns = [
        "manual_evidence_valid",
        "manual_priority_valid",
        "validator_id",
        "validation_notes",
    ]
    for column in manual_columns:
        sample[column] = ""
    if RANKING_VALIDATION_PATH.exists():
        existing = pd.read_csv(
            RANKING_VALIDATION_PATH,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        keys = ["canonical_place_id", "aspect"]
        available = [*keys, *[column for column in manual_columns if column in existing.columns]]
        existing = existing[available].drop_duplicates(keys, keep="last")
        renamed = existing.rename(
            columns={column: f"{column}__existing" for column in manual_columns if column in existing}
        )
        sample = sample.merge(renamed, on=keys, how="left")
        for column in manual_columns:
            existing_column = f"{column}__existing"
            if existing_column not in sample:
                continue
            values = sample[existing_column].fillna("").astype(str)
            keep = values.str.strip().ne("")
            sample.loc[keep, column] = values[keep]
            sample = sample.drop(columns=[existing_column])
    output_path = RANKING_VALIDATION_PATH
    write_status = "updated"
    try:
        sample.to_csv(RANKING_VALIDATION_PATH, index=False, encoding="utf-8-sig")
    except PermissionError:
        output_path = RANKING_VALIDATION_PENDING_PATH
        write_status = "primary_file_locked_pending_written"
        sample.to_csv(output_path, index=False, encoding="utf-8-sig")

    valid_values = {"yes", "true", "1", "valid"}
    evidence = sample["manual_evidence_valid"].astype(str).str.strip().str.lower()
    priority = sample["manual_priority_valid"].astype(str).str.strip().str.lower()
    evidence_reviewed = evidence.ne("")
    priority_reviewed = priority.ne("")
    fully_reviewed_mask = evidence_reviewed & priority_reviewed
    evidence_valid = evidence.isin(valid_values)
    priority_valid = priority.isin(valid_values)
    overall_valid = evidence_valid & priority_valid
    reviewed_count = int(fully_reviewed_mask.sum())
    evidence_valid_count = int(evidence_valid[evidence_reviewed].sum())
    priority_valid_count = int(priority_valid[priority_reviewed].sum())
    overall_valid_count = int(overall_valid[fully_reviewed_mask].sum())
    all_rows_reviewed = reviewed_count == len(sample)
    return {
        "sample_rows": int(len(sample)),
        "reviewed_rows": reviewed_count,
        "evidence_reviewed_rows": int(evidence_reviewed.sum()),
        "priority_reviewed_rows": int(priority_reviewed.sum()),
        "evidence_valid_rows": evidence_valid_count,
        "priority_valid_rows": priority_valid_count,
        "valid_rows": overall_valid_count,
        "evidence_validity_rate": (
            float(evidence_valid_count / evidence_reviewed.sum())
            if evidence_reviewed.any()
            else None
        ),
        "priority_validity_rate": (
            float(priority_valid_count / priority_reviewed.sum())
            if priority_reviewed.any()
            else None
        ),
        "validity_rate": (
            float(overall_valid_count / len(sample)) if all_rows_reviewed else None
        ),
        "fully_reviewed": all_rows_reviewed,
        "write_status": write_status,
        "output": str(output_path),
    }


def run_gap_scoring() -> dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    config = load_gap_config()
    aspect_ids = load_aspect_ids()
    places = pd.read_parquet(PROCESSED_DIR / "places_master.parquet")
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    clause_aspect_path = PROCESSED_DIR / "review_aspect_sentiment.parquet"
    legacy_aspect_path = PROCESSED_DIR / "review_aspect_predictions.parquet"
    if clause_aspect_path.exists():
        aspect_predictions = pd.read_parquet(clause_aspect_path)
        sentiment_signal_source = "clause_level_aspect_sentiment"
        aspect_prediction_path = clause_aspect_path
    else:
        aspect_predictions = pd.read_parquet(legacy_aspect_path)
        sentiment_signal_source = "legacy_review_level_weak_sentiment"
        aspect_prediction_path = legacy_aspect_path

    scores = compute_service_gap_scores(places, reviews, aspect_predictions, config, aspect_ids)
    scores.to_parquet(PROCESSED_DIR / "service_gap_scores.parquet", index=False)
    scores.to_csv(PREDICTION_DIR / "service_gap_rankings.csv", index=False, encoding="utf-8")
    update_cluster_average_gap(scores)
    ranking_validation = export_service_gap_validation_sample(
        scores,
        aspect_sentiment=aspect_predictions
        if sentiment_signal_source == "clause_level_aspect_sentiment"
        else None,
    )

    methodology = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "config_path": str(CONFIG_PATH),
        "aspect_taxonomy_path": str(ASPECT_TAXONOMY_PATH),
        "score_range": "0-100",
        "sentiment_signal_source": sentiment_signal_source,
        "aspect_prediction_path": str(aspect_prediction_path),
        "formula": {
            "smoothed_negative_rate": "(negative_mentions + alpha) / (total_aspect_mentions + alpha + beta)",
            "base_score": "weighted sum of smoothed negative rate, aspect mention frequency, service scarcity, review volume confidence, rating gap, recency, and metadata completeness gap",
            "final_score": "100 * base_score * data_reliability_factor",
        },
        "config": config,
        "rows": int(len(scores)),
        "places_scored": int(scores["canonical_place_id"].nunique()) if not scores.empty else 0,
        "aspects_scored": int(scores["aspect"].nunique()) if not scores.empty else 0,
        "top_10": scores.head(10)[
            [
                "rank",
                "canonical_place_id",
                "place_name",
                "aspect",
                "service_gap_score",
                "confidence_level",
                "reason_codes_text",
            ]
        ].to_dict(orient="records") if not scores.empty else [],
        "human_ranking_validation": ranking_validation,
        "limitations": [
            "Weights are prototype configuration values and require stakeholder validation.",
            "Aspect and sentiment signals are not manually verified ground truth."
            if sentiment_signal_source == "clause_level_aspect_sentiment"
            else "Aspect and sentiment signals are weak review-level labels, not manually verified ground truth.",
            "Service scarcity is estimated from nearby same-category place density, not from official service inventory.",
            "The score is opportunity prioritization, not a prediction of revenue or visitor volume.",
        ],
        "outputs": {
            "service_gap_scores": str(PROCESSED_DIR / "service_gap_scores.parquet"),
            "service_gap_rankings": str(PREDICTION_DIR / "service_gap_rankings.csv"),
            "service_gap_methodology": str(REPORT_DIR / "service_gap_methodology.json"),
        },
    }
    (REPORT_DIR / "service_gap_methodology.json").write_text(json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8")
    return methodology


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute transparent Service Gap Score.")
    parser.parse_args()
    methodology = run_gap_scoring()
    print(
        json.dumps(
            {
                "rows": methodology["rows"],
                "places_scored": methodology["places_scored"],
                "aspects_scored": methodology["aspects_scored"],
                "rankings": methodology["outputs"]["service_gap_rankings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
