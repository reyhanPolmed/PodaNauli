"""Artifact-backed query and inference service used by FastAPI."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import pandas as pd

from api.import_service import BatchImportService
from src.gap_scoring import compute_service_gap_scores


def _none_if_missing(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [_none_if_missing(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _float(value: Any) -> float | None:
    cleaned = _none_if_missing(value)
    return None if cleaned is None else float(cleaned)


def _int(value: Any) -> int | None:
    cleaned = _none_if_missing(value)
    return None if cleaned is None else int(cleaned)


def _text(value: Any) -> str | None:
    cleaned = _none_if_missing(value)
    if cleaned is None:
        return None
    text = str(cleaned).strip()
    return text or None


class ArtifactStore:
    """Load immutable model and dashboard artifacts once per API process."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_registry = self._json("models/model_registry.json")
        self.metrics = self._json("demo/demo_metrics.json")
        self.data_quality = self._json("demo/demo_data_quality.json")
        self.readiness = self._json("outputs/reports/project_readiness.json")
        self.sentiment_model = joblib.load(root / "models/sentiment_champion.joblib")
        self.complaint_bundle = joblib.load(root / "models/complaint_detector.joblib")
        self.aspect_model = joblib.load(root / "models/aspect_champion.joblib")
        self.aspect_labels = joblib.load(root / "models/aspect_multilabel_binarizer.joblib")
        self.aspect_metadata = self._json("models/aspect_metadata.json")
        self.rankings = pd.read_csv(root / "outputs/predictions/service_gap_rankings.csv")
        self.places = pd.read_parquet(root / "data/processed/places_master.parquet")
        self.reviews = pd.read_parquet(root / "data/processed/reviews_clean.parquet")
        self.aspect_evidence = pd.read_parquet(root / "data/processed/review_aspect_sentiment.parquet")
        self.clusters = pd.read_parquet(root / "data/processed/place_clusters.parquet")
        self.geojson = self._json("outputs/maps/place_clusters.geojson")
        self.sheet_quality = pd.read_csv(root / "outputs/reports/data_quality_summary.csv")
        self.model_versions = {
            str(item["task"]): str(item["version"])
            for item in self.model_registry.get("models", [])
            if item.get("is_champion")
        }
        self._baseline_rankings = self.rankings.copy(deep=True)
        self._baseline_reviews = self.reviews.copy(deep=True)
        self._baseline_aspect_evidence = self.aspect_evidence.copy(deep=True)
        self._baseline_geojson = deepcopy(self.geojson)
        self._refresh_lock = Lock()
        runtime_dir = Path(os.getenv("PODANAULI_RUNTIME_DIR", str(root / "data" / "runtime")))
        self.imports = BatchImportService(
            root=root,
            runtime_dir=runtime_dir,
            sentiment_model=self.sentiment_model,
            complaint_bundle=self.complaint_bundle,
            aspect_model=self.aspect_model,
            aspect_labels=self.aspect_labels,
            aspect_metadata=self.aspect_metadata,
            model_versions=self.model_versions,
            baseline_places=self.places,
        )
        self.reload_published_data()

    def reload_published_data(self) -> None:
        review_frames, evidence_frames = self.imports.load_published_frames()
        self.reviews = self._baseline_reviews.copy(deep=True)
        self.aspect_evidence = self._baseline_aspect_evidence.copy(deep=True)
        self.rankings = self._baseline_rankings.copy(deep=True)
        if review_frames:
            self.reviews = pd.concat([self.reviews, *review_frames], ignore_index=True, sort=False)
            self.reviews = self.reviews.drop_duplicates("review_id", keep="first").reset_index(drop=True)
        if evidence_frames:
            self.aspect_evidence = pd.concat(
                [self.aspect_evidence, *evidence_frames], ignore_index=True, sort=False
            )
            self.aspect_evidence = self.aspect_evidence.drop_duplicates(
                ["review_id", "clause_index", "aspect"], keep="first"
            ).reset_index(drop=True)
        if review_frames or evidence_frames:
            self.rankings = compute_service_gap_scores(
                self.places,
                self.reviews,
                self.aspect_evidence,
                self.imports.gap_config,
                self.imports.aspect_ids,
            )
        self.geojson = deepcopy(self._baseline_geojson)
        self._prepare_views()

    def publish_import(self, import_id: str) -> dict[str, Any]:
        with self._refresh_lock:
            summary = self.imports.publish(import_id)
            try:
                self.reload_published_data()
            except Exception:
                self.imports.unpublish(import_id)
                self.reload_published_data()
                raise
            return summary

    def unpublish_import(self, import_id: str) -> dict[str, Any]:
        with self._refresh_lock:
            summary = self.imports.unpublish(import_id)
            self.reload_published_data()
            return summary

    def _json(self, relative_path: str) -> dict[str, Any]:
        return json.loads((self.root / relative_path).read_text(encoding="utf-8-sig"))

    def _prepare_views(self) -> None:
        self.rankings = self.rankings.sort_values("rank").reset_index(drop=True)
        self.valid_reviews = self.reviews.loc[
            self.reviews["review_text_clean"].notna() & ~self.reviews["is_duplicate"].fillna(False)
        ].copy()
        review_counts = self.valid_reviews.groupby("canonical_place_id")["review_id"].nunique().to_dict()
        top_gap = self.rankings.drop_duplicates("canonical_place_id").set_index("canonical_place_id")
        cluster_lookup = self.clusters.set_index("canonical_place_id")
        self.place_rows: list[dict[str, Any]] = []
        for row in self.places.itertuples(index=False):
            place_id = str(row.canonical_place_id)
            gap = top_gap.loc[place_id] if place_id in top_gap.index else None
            cluster = cluster_lookup.loc[place_id] if place_id in cluster_lookup.index else None
            self.place_rows.append(
                {
                    "place_id": place_id,
                    "name": str(row.canonical_place_name),
                    "category": str(row.place_category),
                    "place_type": _text(row.place_type),
                    "rating": _float(row.place_rating),
                    "review_count": int(review_counts.get(place_id, 0)),
                    "latitude": _float(row.latitude),
                    "longitude": _float(row.longitude),
                    "cluster_id": _int(cluster.geo_cluster_id) if cluster is not None else None,
                    "top_aspect": str(gap.aspect) if gap is not None else None,
                    "service_gap_score": _float(gap.service_gap_score) if gap is not None else None,
                    "confidence": str(gap.confidence_level) if gap is not None else None,
                }
            )
        self._enrich_geojson(top_gap)

    def _enrich_geojson(self, top_gap: pd.DataFrame) -> None:
        for feature in self.geojson.get("features", []):
            properties = feature.setdefault("properties", {})
            place_id = str(properties.get("canonical_place_id", ""))
            if place_id and place_id in top_gap.index:
                gap = top_gap.loc[place_id]
                properties.update(
                    {
                        "top_aspect": str(gap.aspect),
                        "service_gap_score": round(float(gap.service_gap_score), 4),
                        "confidence": str(gap.confidence_level),
                        "priority": str(gap.priority_level),
                        "review_count": int(gap.review_count),
                        "evidence_count": int(gap.aspect_mention_count),
                    }
                )

    @property
    def loaded(self) -> bool:
        return all(
            item is not None
            for item in (self.sentiment_model, self.complaint_bundle, self.aspect_model, self.aspect_labels)
        ) and not self.places.empty

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.loaded else "degraded",
            "model_loaded": self.loaded,
            "dataset_loaded": not self.places.empty and not self.rankings.empty,
            "model_version": self.model_versions,
        }

    def model_metric_items(self) -> list[dict[str, Any]]:
        return [
            {"model": "sentiment", "metric": "macro_f1", "value": self.metrics["sentiment"]["macro_f1"]},
            {
                "model": "sentiment",
                "metric": "negative_recall",
                "value": self.metrics["sentiment"]["negative_recall"],
            },
            {"model": "complaint", "metric": "macro_f1", "value": self.metrics["complaint"]["macro_f1"]},
            {
                "model": "complaint",
                "metric": "complaint_recall",
                "value": self.metrics["complaint"]["negative_recall"],
            },
            {"model": "aspect", "metric": "micro_f1", "value": self.metrics["aspect"]["micro_f1"]},
            {"model": "aspect", "metric": "macro_f1", "value": self.metrics["aspect"]["macro_f1"]},
        ]

    def summary(self) -> dict[str, Any]:
        aspect_counts = (
            self.rankings.groupby("aspect")["negative_mention_count"].sum().sort_values(ascending=False).head(6)
        )
        sentiment_values = self.valid_reviews["weak_sentiment_label"].copy()
        if "model_sentiment_label" in self.valid_reviews.columns:
            sentiment_values = self.valid_reviews["model_sentiment_label"].combine_first(sentiment_values)
        sentiment_counts = sentiment_values.value_counts().to_dict()
        ratings = pd.to_numeric(self.places["place_rating"], errors="coerce").dropna()
        return {
            "total_reviews": int(len(self.valid_reviews)),
            "total_places": int(len(self.places)),
            "total_service_gaps": int(len(self.rankings)),
            "average_rating": round(float(ratings.mean()), 2) if not ratings.empty else None,
            "places_with_coordinates": int(self.places["coordinate_parsing_success"].fillna(False).sum()),
            "category_distribution": {
                str(key): int(value) for key, value in self.places["place_category"].value_counts().items()
            },
            "sentiment_distribution": {str(key): int(value) for key, value in sentiment_counts.items()},
            "top_negative_aspects": [
                {"aspect": str(key), "negative_mentions": int(value)} for key, value in aspect_counts.items()
            ],
            "model_metrics_summary": self.model_metric_items(),
        }

    def list_places(
        self,
        *,
        category: str | None,
        aspect: str | None,
        min_gap_score: float,
        cluster_id: int | None,
        search: str | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        rows = self.place_rows
        if aspect:
            candidates = self.rankings.loc[self.rankings["aspect"].str.lower() == aspect.lower()]
            aspect_gap = candidates.drop_duplicates("canonical_place_id").set_index("canonical_place_id")
            rows = []
            for item in self.place_rows:
                if item["place_id"] not in aspect_gap.index:
                    continue
                gap = aspect_gap.loc[item["place_id"]]
                rows.append(
                    {
                        **item,
                        "top_aspect": str(gap.aspect),
                        "service_gap_score": float(gap.service_gap_score),
                        "confidence": str(gap.confidence_level),
                    }
                )
        filtered = [item for item in rows if (item["service_gap_score"] or 0) >= min_gap_score]
        if category:
            filtered = [item for item in filtered if item["category"].lower() == category.lower()]
        if cluster_id is not None:
            filtered = [item for item in filtered if item["cluster_id"] == cluster_id]
        if search:
            needle = search.casefold()
            filtered = [item for item in filtered if needle in item["name"].casefold()]
        filtered.sort(key=lambda item: (-(item["service_gap_score"] or 0), item["name"]))
        return {"total": len(filtered), "limit": limit, "offset": offset, "items": filtered[offset : offset + limit]}

    @staticmethod
    def _gap_item(row: Any) -> dict[str, Any]:
        return {
            "rank": int(row["rank"]),
            "place_id": str(row["canonical_place_id"]),
            "place_name": str(row["place_name"]),
            "category": str(row["place_category"]),
            "aspect": str(row["aspect"]),
            "score": round(float(row["service_gap_score"]), 4),
            "confidence": str(row["confidence_level"]),
            "priority": str(row["priority_level"]),
            "review_count": int(row["review_count"]),
            "evidence_count": int(row["aspect_mention_count"]),
            "negative_mentions": int(row["negative_mention_count"]),
            "negative_rate": round(float(row["negative_rate_smoothed"]), 6),
            "data_reliability": round(float(row["data_reliability"]), 6),
            "reason_codes": [code for code in str(row["reason_codes_text"]).split("|") if code],
            "explanation": str(row["explanation"]),
        }

    def list_service_gaps(
        self,
        *,
        aspect: str | None,
        min_score: float,
        category: str | None,
        cluster_id: int | None,
        search: str | None,
        confidence: str | None,
        min_reviews: int | None,
        sort: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        frame = self.rankings.loc[self.rankings["service_gap_score"] >= min_score].copy()
        if aspect:
            frame = frame.loc[frame["aspect"].str.lower() == aspect.lower()]
        if category:
            frame = frame.loc[frame["place_category"].str.lower() == category.lower()]
        if search:
            frame = frame.loc[frame["place_name"].str.contains(search, case=False, regex=False, na=False)]
        if confidence:
            frame = frame.loc[frame["confidence_level"].str.lower() == confidence.lower()]
        if min_reviews is not None:
            frame = frame.loc[frame["review_count"] >= min_reviews]
        if cluster_id is not None:
            ids = set(
                self.clusters.loc[self.clusters["geo_cluster_id"] == cluster_id, "canonical_place_id"].astype(str)
            )
            frame = frame.loc[frame["canonical_place_id"].astype(str).isin(ids)]
        frame = frame.sort_values(
            ["service_gap_score", "rank"],
            ascending=[sort == "score_asc", True],
            kind="mergesort",
        )
        total = int(len(frame))
        items = [self._gap_item(row) for _, row in frame.iloc[offset : offset + limit].iterrows()]
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    def list_place_evidence(
        self,
        place_id: str,
        *,
        aspect: str | None,
        search: str | None,
        min_complaint_probability: float,
        min_confidence: float,
        sort: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any] | None:
        matching = self.places.loc[self.places["canonical_place_id"].astype(str) == place_id]
        if matching.empty:
            return None

        frame = self.aspect_evidence.loc[
            (self.aspect_evidence["canonical_place_id"].astype(str) == place_id)
            & self.aspect_evidence["is_negative"].fillna(False)
        ].copy()
        if frame.empty:
            return {
                "total": 0,
                "total_all": 0,
                "limit": limit,
                "offset": offset,
                "aspect_counts": {},
                "items": [],
            }

        frame["clause_text"] = frame["clause_text"].fillna("").astype(str).str.strip()
        frame["aspect"] = frame["aspect"].fillna("lainnya").astype(str)
        frame["complaint_probability"] = pd.to_numeric(
            frame["complaint_probability"], errors="coerce"
        ).fillna(0.0)
        frame["prediction_confidence"] = pd.to_numeric(
            frame["prediction_confidence"], errors="coerce"
        ).fillna(0.0)
        frame = frame.loc[frame["clause_text"].ne("")]
        frame = frame.sort_values(
            ["complaint_probability", "prediction_confidence"],
            ascending=False,
            kind="mergesort",
        ).drop_duplicates(subset=["clause_text", "aspect"], keep="first")

        aspect_counts_series = frame["aspect"].value_counts()
        aspect_counts = {
            str(key): int(value)
            for key, value in aspect_counts_series.sort_values(ascending=False).items()
        }
        total_all = int(len(frame))

        if aspect:
            frame = frame.loc[frame["aspect"].str.lower() == aspect.lower()]
        if search:
            frame = frame.loc[
                frame["clause_text"].str.contains(search.strip(), case=False, regex=False, na=False)
            ]
        frame = frame.loc[
            (frame["complaint_probability"] >= min_complaint_probability)
            & (frame["prediction_confidence"] >= min_confidence)
        ]

        primary_sort = "prediction_confidence" if sort == "confidence_desc" else "complaint_probability"
        secondary_sort = "complaint_probability" if sort == "confidence_desc" else "prediction_confidence"
        frame = frame.sort_values(
            [primary_sort, secondary_sort, "aspect", "clause_text"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        total = int(len(frame))

        items = []
        has_sentiment_source = "sentiment_source" in frame.columns
        for item in frame.iloc[offset : offset + limit].itertuples(index=False):
            items.append(
                {
                    "text": str(item.clause_text),
                    "aspect": str(item.aspect),
                    "complaint_probability": round(float(item.complaint_probability), 6),
                    "confidence": round(float(item.prediction_confidence), 6),
                    "sentiment_source": (
                        str(item.sentiment_source) if has_sentiment_source else "model"
                    ),
                }
            )
        return {
            "total": total,
            "total_all": total_all,
            "limit": limit,
            "offset": offset,
            "aspect_counts": aspect_counts,
            "items": items,
        }

    def place_detail(self, place_id: str) -> dict[str, Any] | None:
        matching = self.places.loc[self.places["canonical_place_id"].astype(str) == place_id]
        if matching.empty:
            return None
        row = matching.iloc[0]
        place_reviews = self.valid_reviews.loc[self.valid_reviews["canonical_place_id"].astype(str) == place_id]
        sentiment_values = place_reviews["weak_sentiment_label"].copy()
        if "model_sentiment_label" in place_reviews.columns:
            sentiment_values = place_reviews["model_sentiment_label"].combine_first(sentiment_values)
        sentiment = {str(key): int(value) for key, value in sentiment_values.value_counts().items()}
        gaps = self.rankings.loc[self.rankings["canonical_place_id"].astype(str) == place_id]
        gap_items = [self._gap_item(item) for _, item in gaps.iterrows()]
        cluster = self.clusters.loc[self.clusters["canonical_place_id"].astype(str) == place_id]
        evidence_payload = self.list_place_evidence(
            place_id,
            aspect=None,
            search=None,
            min_complaint_probability=0,
            min_confidence=0,
            sort="complaint_desc",
            limit=8,
            offset=0,
        )
        evidence = evidence_payload["items"] if evidence_payload is not None else []
        top_aspects = [
            {
                "aspect": item["aspect"],
                "score": item["score"],
                "evidence_count": item["evidence_count"],
                "negative_mentions": item["negative_mentions"],
            }
            for item in gap_items[:6]
        ]
        return {
            "place_id": place_id,
            "name": str(row["canonical_place_name"]),
            "category": str(row["place_category"]),
            "place_type": _text(row["place_type"]),
            "address": _text(row["address"]),
            "status": _text(row["status"]),
            "rating": _float(row["place_rating"]),
            "review_count": int(place_reviews["review_id"].nunique()),
            "latitude": _float(row["latitude"]),
            "longitude": _float(row["longitude"]),
            "cluster_id": _int(cluster.iloc[0]["geo_cluster_id"]) if not cluster.empty else None,
            "facilities": _text(row["facility_text"]),
            "min_price": _float(row["min_price"]),
            "max_price": _float(row["max_price"]),
            "sentiment_distribution": sentiment,
            "top_aspects": top_aspects,
            "service_gaps": gap_items,
            "evidence": evidence,
            "limitations": [
                "Evidence berasal dari model dan harus diperiksa manusia.",
                "Metadata kosong tidak berarti fasilitas tidak tersedia.",
            ],
        }

    def clusters_geojson(self, aspect: str | None = None, cluster_id: int | None = None) -> dict[str, Any]:
        features = []
        for feature in self.geojson.get("features", []):
            properties = feature.get("properties", {})
            if properties.get("feature_type") != "place":
                continue
            if aspect and str(properties.get("top_aspect", "")).lower() != aspect.lower():
                continue
            if cluster_id is not None and _int(properties.get("geo_cluster_id")) != cluster_id:
                continue
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    def model_metrics(self) -> dict[str, Any]:
        aspect = self.metrics["aspect"]
        validation = self.readiness.get("service_gap_validation", {})
        return {
            "scope": self.metrics["metrics_scope"],
            "sentiment": self.metrics["sentiment"],
            "complaint": self.metrics["complaint"],
            "aspect": {
                **aspect,
                "per_aspect": aspect.get("per_aspect", {}),
            },
            "service_gap_validation": {
                key: _none_if_missing(value)
                for key, value in validation.items()
                if key
                in {
                    "status",
                    "reviewed_rows",
                    "total_rows",
                    "minimum_validity",
                    "overall_validity",
                    "evidence_validity",
                    "priority_validity",
                    "recommendation_validity",
                }
            },
            "limitations": list(self.readiness.get("limitations", []))
            + ["Sistem belum dinyatakan siap produksi."],
        }

    def data_quality_payload(self) -> dict[str, Any]:
        sheets = []
        for _, row in self.sheet_quality.iterrows():
            sheets.append(
                {
                    str(key): _none_if_missing(value)
                    for key, value in row.to_dict().items()
                    if key
                    in {
                        "sheet_name",
                        "rows",
                        "duplicate_rows",
                        "missing_cells",
                        "formula_cell_count",
                        "unique_places",
                        "reviews_with_text",
                        "reviews_without_text",
                    }
                }
            )
        return {
            "summary": self.data_quality["summary"],
            "transformations": self.data_quality["transformations"],
            "sheets": sheets,
            "limitations": list(self.data_quality.get("important_notes", [])),
        }

    def analyze_review(self, text: str) -> dict[str, Any]:
        cleaned = " ".join(text.split())
        sentiment_label = str(self.sentiment_model.predict([cleaned])[0])
        sentiment_probabilities = np.asarray(self.sentiment_model.predict_proba([cleaned]))[0]
        sentiment_classes = [str(value) for value in self.sentiment_model.classes_]
        sentiment_scores = {
            label: round(float(score), 6) for label, score in zip(sentiment_classes, sentiment_probabilities)
        }

        complaint_model = self.complaint_bundle["model"]
        complaint_classes = list(complaint_model.named_steps["classifier"].classes_)
        complaint_index = complaint_classes.index(1)
        complaint_probability = float(complaint_model.predict_proba([cleaned])[0][complaint_index])
        threshold = float(self.complaint_bundle["negative_threshold"])
        margin = float(self.complaint_bundle["uncertainty_margin"])
        if complaint_probability >= min(1.0, threshold + margin):
            complaint = "detected"
        elif complaint_probability <= max(0.0, threshold - margin):
            complaint = "not_detected"
        else:
            complaint = "review_required"

        aspect_probabilities = np.asarray(self.aspect_model.predict_proba([cleaned]))[0]
        aspect_classes = [str(label) for label in self.aspect_labels.classes_]
        aspect_threshold = float(self.aspect_metadata["threshold"])
        aspects = sorted(
            [
                {"label": label, "probability": round(float(score), 6)}
                for label, score in zip(aspect_classes, aspect_probabilities)
                if label != "lainnya" and float(score) >= aspect_threshold
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )[:6]
        if not aspects:
            fallback = max(zip(aspect_classes, aspect_probabilities), key=lambda item: float(item[1]))
            if fallback[0] == "lainnya" and float(fallback[1]) >= aspect_threshold:
                aspects = [{"label": "lainnya", "probability": round(float(fallback[1]), 6)}]

        warnings = ["Hasil model merupakan dukungan analisis dan bukan keputusan otomatis."]
        if complaint == "review_required":
            warnings.append("Sinyal keluhan berada pada area abu-abu dan perlu pemeriksaan manusia.")
        if sentiment_label == "positive" and complaint == "detected":
            warnings.append("Sentimen umum dan sinyal keluhan berbeda; periksa konteks ulasan.")
        if not aspects:
            warnings.append("Tidak ada aspek yang melewati ambang model.")
        return {
            "sentiment": sentiment_label,
            "sentiment_scores": sentiment_scores,
            "complaint": complaint,
            "complaint_probability": round(complaint_probability, 6),
            "aspects": aspects,
            "warnings": warnings,
            "model_version": self.model_versions,
        }
