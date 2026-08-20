"""Validated batch import, inference, scoring, and geospatial projection."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.gap_scoring import compute_service_gap_scores, load_aspect_ids, load_gap_config
from src.geospatial import run_dbscan_haversine, valid_coordinate_mask
from src.preprocessing import clean_text_basic, normalize_column_name, weak_sentiment_from_rating
from src.train_aspect import detect_weak_aspects, load_aspect_taxonomy


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 5_000
SUPPORTED_SUFFIXES = {".csv", ".xlsx"}
REQUIRED_COLUMNS = {"place_name", "place_category", "review_text"}
CONTRAST_PATTERN = re.compile(
    r"\s+(?:tetapi|tapi|namun|sedangkan|akan tetapi|walaupun|meskipun|but|however|although)\s+",
    flags=re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?;])\s+|\s*\|\s*")

TEMPLATE_COLUMNS = [
    "place_id",
    "place_name",
    "place_category",
    "place_type",
    "address",
    "latitude",
    "longitude",
    "place_rating",
    "status",
    "facility_text",
    "min_price",
    "max_price",
    "review_id",
    "review_text",
    "review_rating",
    "review_date",
]

REVIEW_TEMPLATE_COLUMNS = ["review_id", "review_text", "review_rating", "review_date"]

COLUMN_ALIASES = {
    "id_tempat": "place_id",
    "destination_id": "place_id",
    "nama_tempat": "place_name",
    "nama_destinasi": "place_name",
    "destination_name": "place_name",
    "kategori": "place_category",
    "kategori_tempat": "place_category",
    "jenis_tempat": "place_type",
    "tipe_tempat": "place_type",
    "alamat": "address",
    "lat": "latitude",
    "lng": "longitude",
    "lon": "longitude",
    "rating_tempat": "place_rating",
    "fasilitas": "facility_text",
    "harga_minimum": "min_price",
    "harga_maksimum": "max_price",
    "id_ulasan": "review_id",
    "ulasan": "review_text",
    "teks_ulasan": "review_text",
    "review_text_raw": "review_text",
    "rating_ulasan": "review_rating",
    "reviewer_rating": "review_rating",
    "tanggal_ulasan": "review_date",
}


class ImportDataError(ValueError):
    """Raised when an uploaded dataset cannot be processed safely."""

    def __init__(self, message: str, issues: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.issues = issues or []


def _native(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _slug(value: str) -> str:
    normalized = normalize_column_name(value).replace("_", "-")
    return normalized[:48].strip("-") or "destinasi"


def _number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if pd.isna(value):
        return None
    cleaned = str(value).strip().replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def _first_present(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    values = values.loc[values.astype(str).str.strip().ne("")]
    return values.iloc[0] if not values.empty else None


def _split_clauses(text: Any) -> list[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    clauses = []
    for sentence in SENTENCE_PATTERN.split(str(text)):
        for clause in CONTRAST_PATTERN.split(sentence):
            cleaned = re.sub(r"\s+", " ", clause).strip(" ,.-")
            if cleaned:
                clauses.append(cleaned)
    return clauses


class BatchImportService:
    """Process new review datasets without mutating training or baseline artifacts."""

    def __init__(
        self,
        *,
        root: Path,
        runtime_dir: Path,
        sentiment_model: Any,
        complaint_bundle: dict[str, Any],
        aspect_model: Any,
        aspect_labels: Any,
        aspect_metadata: dict[str, Any],
        model_versions: dict[str, str],
        baseline_places: pd.DataFrame,
    ) -> None:
        self.root = root
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.sentiment_model = sentiment_model
        self.complaint_bundle = complaint_bundle
        self.aspect_model = aspect_model
        self.aspect_labels = [str(value) for value in aspect_labels.classes_]
        self.aspect_threshold = float(aspect_metadata["threshold"])
        self.model_versions = model_versions
        self.baseline_places = baseline_places.copy()
        self.gap_config = load_gap_config()
        self.aspect_ids = load_aspect_ids()
        self.taxonomy = load_aspect_taxonomy()
        self.lock = Lock()

    @staticmethod
    def template_csv() -> str:
        sample = {
            "review_id": "contoh-001",
            "review_text": "Pemandangan indah, tetapi toilet kurang bersih.",
            "review_rating": 3,
            "review_date": "2026-08-01",
        }
        return pd.DataFrame([sample], columns=REVIEW_TEMPLATE_COLUMNS).to_csv(index=False, lineterminator="\n")

    def process_for_place(
        self,
        *,
        place: pd.Series,
        filename: str,
        content: bytes,
        existing_review_texts: set[str],
    ) -> dict[str, Any]:
        if not filename:
            raise ImportDataError("Nama file wajib disertakan.")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ImportDataError("Format file harus CSV atau XLSX.")
        if not content:
            raise ImportDataError("File kosong tidak dapat diproses.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ImportDataError("Ukuran file melebihi batas 10 MB.")

        frame = self._read_file(suffix, content)
        if frame.empty:
            raise ImportDataError("File tidak memiliki baris data.")
        if len(frame) > MAX_UPLOAD_ROWS:
            raise ImportDataError(f"Jumlah baris melebihi batas {MAX_UPLOAD_ROWS:,}.")

        import_id = f"imp-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
        normalized, issues = self._normalize_review_rows(
            frame=frame,
            import_id=import_id,
            place=place,
            existing_review_texts=existing_review_texts,
        )
        if normalized.empty:
            raise ImportDataError("Tidak ada ulasan baru yang valid untuk dianalisis.", issues)

        places, reviews, metadata_warnings = self._build_entities(normalized, import_id)
        issues.extend(metadata_warnings)
        reviews = self._predict_reviews(reviews)
        evidence = self._predict_clause_evidence(reviews)
        rankings = compute_service_gap_scores(
            places,
            reviews,
            evidence,
            self.gap_config,
            self.aspect_ids,
        )
        clusters = self._cluster_places(places)
        geojson = self._geojson(places, clusters, rankings)
        summary = self._summary(
            import_id=import_id,
            filename=Path(filename).name,
            rows_received=len(frame),
            normalized=normalized,
            places=places,
            reviews=reviews,
            evidence=evidence,
            rankings=rankings,
            geojson=geojson,
            issues=issues,
            scope="published_existing_place",
            target_place_id=str(place["canonical_place_id"]),
        )
        self._persist(import_id, normalized, places, reviews, evidence, rankings, clusters, geojson, summary)
        return summary

    def process(self, filename: str, content: bytes) -> dict[str, Any]:
        if not filename:
            raise ImportDataError("Nama file wajib disertakan.")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ImportDataError("Format file harus CSV atau XLSX.")
        if not content:
            raise ImportDataError("File kosong tidak dapat diproses.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ImportDataError("Ukuran file melebihi batas 10 MB.")

        frame = self._read_file(suffix, content)
        if frame.empty:
            raise ImportDataError("File tidak memiliki baris data.")
        if len(frame) > MAX_UPLOAD_ROWS:
            raise ImportDataError(f"Jumlah baris melebihi batas {MAX_UPLOAD_ROWS:,}.")

        import_id = f"imp-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
        normalized, issues = self._normalize_rows(frame, import_id)
        if normalized.empty:
            raise ImportDataError("Tidak ada baris valid yang dapat dianalisis.", issues)

        places, reviews, metadata_warnings = self._build_entities(normalized, import_id)
        issues.extend(metadata_warnings)
        reviews = self._predict_reviews(reviews)
        evidence = self._predict_clause_evidence(reviews)
        rankings = compute_service_gap_scores(
            places,
            reviews,
            evidence,
            self.gap_config,
            self.aspect_ids,
        )
        clusters = self._cluster_places(places)
        geojson = self._geojson(places, clusters, rankings)
        summary = self._summary(
            import_id=import_id,
            filename=Path(filename).name,
            rows_received=len(frame),
            normalized=normalized,
            places=places,
            reviews=reviews,
            evidence=evidence,
            rankings=rankings,
            geojson=geojson,
            issues=issues,
        )
        self._persist(import_id, normalized, places, reviews, evidence, rankings, clusters, geojson, summary)
        return summary

    @staticmethod
    def _read_file(suffix: str, content: bytes) -> pd.DataFrame:
        try:
            if suffix == ".xlsx":
                return pd.read_excel(BytesIO(content), dtype=object)
            decoded = None
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    decoded = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if decoded is None:
                raise ImportDataError("Encoding CSV tidak didukung. Gunakan UTF-8.")
            return pd.read_csv(StringIO(decoded), dtype=object, sep=None, engine="python")
        except ImportDataError:
            raise
        except Exception as exc:
            raise ImportDataError(f"File tidak dapat dibaca: {exc}") from exc

    def _normalize_rows(self, frame: pd.DataFrame, import_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        renamed: dict[Any, str] = {}
        seen: set[str] = set()
        for original in frame.columns:
            normalized = COLUMN_ALIASES.get(normalize_column_name(original), normalize_column_name(original))
            if normalized in seen:
                raise ImportDataError(f"Kolom '{normalized}' muncul lebih dari satu kali setelah normalisasi.")
            seen.add(normalized)
            renamed[original] = normalized
        frame = frame.rename(columns=renamed)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ImportDataError("Kolom wajib belum lengkap: " + ", ".join(missing))

        issues: list[dict[str, Any]] = []
        unknown = sorted(set(frame.columns) - set(TEMPLATE_COLUMNS) - {"coordinate"})
        if unknown:
            issues.append(
                {
                    "row": None,
                    "field": "columns",
                    "severity": "warning",
                    "code": "ignored_columns",
                    "message": "Kolom diabaikan: " + ", ".join(unknown),
                }
            )

        records: list[dict[str, Any]] = []
        for position, (_, source) in enumerate(frame.iterrows(), start=2):
            place_name = _text(source.get("place_name"))
            category = _text(source.get("place_category"))
            review_text = clean_text_basic(source.get("review_text"))
            row_errors = []
            if not place_name:
                row_errors.append(("place_name", "Nama tempat kosong."))
            if not category:
                row_errors.append(("place_category", "Kategori tempat kosong."))
            if not review_text or len(review_text) < 3:
                row_errors.append(("review_text", "Teks ulasan minimal 3 karakter."))
            if review_text and len(review_text) > 2_000:
                row_errors.append(("review_text", "Teks ulasan melebihi 2.000 karakter."))

            latitude = _number(source.get("latitude"))
            longitude = _number(source.get("longitude"))
            if (latitude is None) != (longitude is None):
                issues.append(self._issue(position, "coordinates", "warning", "incomplete_coordinate", "Latitude dan longitude harus diisi berpasangan; koordinat dikosongkan."))
                latitude, longitude = None, None
            elif latitude is not None and longitude is not None:
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    issues.append(self._issue(position, "coordinates", "warning", "invalid_coordinate", "Koordinat di luar rentang valid; titik tidak dipetakan."))
                    latitude, longitude = None, None

            place_rating = _number(source.get("place_rating"))
            review_rating = _number(source.get("review_rating"))
            for field, value in (("place_rating", place_rating), ("review_rating", review_rating)):
                if value is not None and not 1 <= value <= 5:
                    row_errors.append((field, "Rating harus berada pada rentang 1 sampai 5."))
            for field, message in row_errors:
                issues.append(self._issue(position, field, "error", "invalid_value", message))
            if row_errors:
                continue

            supplied_place_id = _text(source.get("place_id"))
            identity = supplied_place_id or f"{place_name}|{category}"
            digest = sha1(identity.casefold().encode("utf-8")).hexdigest()[:10]
            place_id = f"upload-{_slug(identity)}-{digest}"
            supplied_review_id = _text(source.get("review_id"))
            review_id = supplied_review_id or f"{import_id}-r{position - 1:05d}"
            review_date = pd.to_datetime(source.get("review_date"), errors="coerce")
            if _text(source.get("review_date")) and pd.isna(review_date):
                issues.append(self._issue(position, "review_date", "warning", "invalid_date", "Tanggal tidak dikenali dan dikosongkan. Gunakan format YYYY-MM-DD."))

            records.append(
                {
                    "source_row": position,
                    "canonical_place_id": place_id,
                    "canonical_place_name": place_name,
                    "place_category": category.casefold(),
                    "place_type": _text(source.get("place_type")),
                    "address": _text(source.get("address")),
                    "latitude": latitude,
                    "longitude": longitude,
                    "place_rating": place_rating,
                    "status": _text(source.get("status")),
                    "facility_text": _text(source.get("facility_text")),
                    "min_price": _number(source.get("min_price")),
                    "max_price": _number(source.get("max_price")),
                    "review_id": review_id,
                    "review_text_raw": _text(source.get("review_text")),
                    "review_text_clean": review_text,
                    "reviewer_rating": review_rating,
                    "review_date": review_date if pd.notna(review_date) else pd.NaT,
                }
            )

        normalized = pd.DataFrame(records)
        if normalized.empty:
            return normalized, issues
        duplicate_mask = normalized.duplicated(
            subset=["canonical_place_id", "review_text_clean"], keep="first"
        )
        for row_number in normalized.loc[duplicate_mask, "source_row"].tolist():
            issues.append(self._issue(int(row_number), "review_text", "warning", "duplicate_review", "Ulasan duplikat diabaikan."))
        normalized = normalized.loc[~duplicate_mask].reset_index(drop=True)
        return normalized, issues

    def _normalize_review_rows(
        self,
        *,
        frame: pd.DataFrame,
        import_id: str,
        place: pd.Series,
        existing_review_texts: set[str],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        renamed: dict[Any, str] = {}
        seen: set[str] = set()
        for original in frame.columns:
            normalized = COLUMN_ALIASES.get(normalize_column_name(original), normalize_column_name(original))
            if normalized in seen:
                raise ImportDataError(f"Kolom '{normalized}' muncul lebih dari satu kali setelah normalisasi.")
            seen.add(normalized)
            renamed[original] = normalized
        frame = frame.rename(columns=renamed)
        if "review_text" not in frame.columns:
            raise ImportDataError("Kolom wajib belum lengkap: review_text")

        issues: list[dict[str, Any]] = []
        unknown = sorted(set(frame.columns) - set(REVIEW_TEMPLATE_COLUMNS))
        if unknown:
            issues.append(self._issue(None, "columns", "warning", "ignored_columns", "Kolom diabaikan: " + ", ".join(unknown)))

        records: list[dict[str, Any]] = []
        seen_texts = {text.casefold() for text in existing_review_texts if text}
        seen_ids: set[str] = set()
        for position, (_, source) in enumerate(frame.iterrows(), start=2):
            review_text_raw = _text(source.get("review_text"))
            review_text = clean_text_basic(source.get("review_text"))
            row_errors = []
            if not review_text or len(review_text) < 3:
                row_errors.append(("review_text", "Teks ulasan minimal 3 karakter."))
            if review_text and len(review_text) > 2_000:
                row_errors.append(("review_text", "Teks ulasan melebihi 2.000 karakter."))
            review_rating = _number(source.get("review_rating"))
            if review_rating is not None and not 1 <= review_rating <= 5:
                row_errors.append(("review_rating", "Rating ulasan harus berada pada rentang 1 sampai 5."))
            for field, message in row_errors:
                issues.append(self._issue(position, field, "error", "invalid_value", message))
            if row_errors:
                continue

            text_key = str(review_text).casefold()
            if text_key in seen_texts:
                issues.append(self._issue(position, "review_text", "warning", "duplicate_review", "Ulasan sudah tersedia untuk destinasi ini dan diabaikan."))
                continue
            seen_texts.add(text_key)

            supplied_review_id = _text(source.get("review_id")) or f"r{position - 1:05d}"
            review_id = f"{import_id}-{_slug(supplied_review_id)}"
            if review_id in seen_ids:
                review_id = f"{review_id}-{position - 1}"
            seen_ids.add(review_id)
            review_date = pd.to_datetime(source.get("review_date"), errors="coerce")
            if _text(source.get("review_date")) and pd.isna(review_date):
                issues.append(self._issue(position, "review_date", "warning", "invalid_date", "Tanggal tidak dikenali dan dikosongkan. Gunakan format YYYY-MM-DD."))

            records.append(
                {
                    "source_row": position,
                    "canonical_place_id": str(place["canonical_place_id"]),
                    "canonical_place_name": str(place["canonical_place_name"]),
                    "place_category": str(place["place_category"]),
                    "place_type": place.get("place_type"),
                    "address": place.get("address"),
                    "latitude": place.get("latitude"),
                    "longitude": place.get("longitude"),
                    "place_rating": place.get("place_rating"),
                    "status": place.get("status"),
                    "facility_text": place.get("facility_text"),
                    "min_price": place.get("min_price"),
                    "max_price": place.get("max_price"),
                    "review_id": review_id,
                    "review_text_raw": review_text_raw,
                    "review_text_clean": review_text,
                    "reviewer_rating": review_rating,
                    "review_date": review_date if pd.notna(review_date) else pd.NaT,
                }
            )
        return pd.DataFrame(records), issues

    @staticmethod
    def _issue(row: int | None, field: str, severity: str, code: str, message: str) -> dict[str, Any]:
        return {"row": row, "field": field, "severity": severity, "code": code, "message": message}

    def _build_entities(
        self, normalized: pd.DataFrame, import_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
        warnings: list[dict[str, Any]] = []
        places = []
        metadata_fields = [
            "canonical_place_name", "place_category", "place_type", "address", "latitude", "longitude",
            "place_rating", "status", "facility_text", "min_price", "max_price",
        ]
        for place_id, group in normalized.groupby("canonical_place_id", sort=False):
            for field in metadata_fields:
                values = group[field].dropna().astype(str).str.strip().unique()
                if len(values) > 1:
                    warnings.append(self._issue(None, field, "warning", "metadata_conflict", f"Metadata berbeda untuk {group['canonical_place_name'].iloc[0]}; nilai pertama digunakan."))
            place_rating = _first_present(group, "place_rating")
            if place_rating is None:
                review_ratings = pd.to_numeric(group["reviewer_rating"], errors="coerce").dropna()
                place_rating = float(review_ratings.mean()) if not review_ratings.empty else None
            places.append(
                {
                    "canonical_place_id": place_id,
                    "canonical_place_name": _first_present(group, "canonical_place_name"),
                    "normalized_place_name": normalize_column_name(_first_present(group, "canonical_place_name")),
                    "place_category": _first_present(group, "place_category"),
                    "place_type": _first_present(group, "place_type"),
                    "address": _first_present(group, "address"),
                    "latitude": _first_present(group, "latitude"),
                    "longitude": _first_present(group, "longitude"),
                    "place_rating": place_rating,
                    "status": _first_present(group, "status"),
                    "facility_text": _first_present(group, "facility_text"),
                    "min_price": _first_present(group, "min_price"),
                    "max_price": _first_present(group, "max_price"),
                    "price_text_original": None,
                    "coordinate_parsing_success": bool(
                        _first_present(group, "latitude") is not None
                        and _first_present(group, "longitude") is not None
                    ),
                    "source_sheets": "runtime_upload",
                    "import_id": import_id,
                }
            )
        place_frame = pd.DataFrame(places)
        reviews = pd.DataFrame(
            {
                "review_id": normalized["review_id"].astype(str),
                "canonical_place_id": normalized["canonical_place_id"].astype(str),
                "place_name": normalized["canonical_place_name"].astype(str),
                "place_category": normalized["place_category"].astype(str),
                "reviewer_rating": normalized["reviewer_rating"],
                "review_text_raw": normalized["review_text_raw"],
                "review_text_clean": normalized["review_text_clean"],
                "review_date": normalized["review_date"],
                "source_sheet": "runtime_upload",
                "latitude": normalized["latitude"],
                "longitude": normalized["longitude"],
                "is_duplicate": False,
                "text_length": normalized["review_text_clean"].str.len().astype(int),
                "weak_sentiment_label": normalized["reviewer_rating"].map(weak_sentiment_from_rating),
                "import_id": import_id,
            }
        )
        return place_frame, reviews, warnings

    def _predict_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        texts = reviews["review_text_clean"].tolist()
        sentiment_labels = self.sentiment_model.predict(texts)
        sentiment_probabilities = np.asarray(self.sentiment_model.predict_proba(texts))
        reviews["model_sentiment_label"] = [str(value) for value in sentiment_labels]
        reviews["sentiment_confidence"] = sentiment_probabilities.max(axis=1).astype(float)

        complaint_model = self.complaint_bundle["model"]
        classes = list(complaint_model.named_steps["classifier"].classes_)
        complaint_index = classes.index(1)
        probabilities = np.asarray(complaint_model.predict_proba(texts))[:, complaint_index]
        threshold = float(self.complaint_bundle["negative_threshold"])
        margin = float(self.complaint_bundle["uncertainty_margin"])
        lower, upper = max(0, threshold - margin), min(1, threshold + margin)
        reviews["complaint_probability"] = probabilities.astype(float)
        reviews["complaint_decision"] = np.where(
            probabilities >= upper,
            "detected",
            np.where(probabilities <= lower, "not_detected", "review_required"),
        )
        return reviews

    def _predict_clause_evidence(self, reviews: pd.DataFrame) -> pd.DataFrame:
        clause_rows: list[dict[str, Any]] = []
        for review in reviews.itertuples(index=False):
            for clause_index, clause in enumerate(_split_clauses(review.review_text_clean)):
                clause_rows.append(
                    {
                        "review_id": review.review_id,
                        "canonical_place_id": review.canonical_place_id,
                        "place_name": review.place_name,
                        "place_category": review.place_category,
                        "clause_index": clause_index,
                        "clause_text": clause,
                    }
                )
        if not clause_rows:
            return pd.DataFrame(columns=["review_id", "canonical_place_id", "aspect", "is_negative"])
        clauses = pd.DataFrame(clause_rows)
        texts = clauses["clause_text"].tolist()
        aspect_probabilities = np.asarray(self.aspect_model.predict_proba(texts))
        complaint_model = self.complaint_bundle["model"]
        classes = list(complaint_model.named_steps["classifier"].classes_)
        complaint_index = classes.index(1)
        complaint_probabilities = np.asarray(complaint_model.predict_proba(texts))[:, complaint_index]
        threshold = float(self.complaint_bundle["negative_threshold"])
        margin = float(self.complaint_bundle["uncertainty_margin"])
        lower, upper = max(0, threshold - margin), min(1, threshold + margin)

        records: list[dict[str, Any]] = []
        for position, clause in clauses.iterrows():
            probability_row = aspect_probabilities[position]
            candidates = sorted(
                [
                    (label, float(score))
                    for label, score in zip(self.aspect_labels, probability_row)
                    if label != "lainnya" and float(score) >= self.aspect_threshold
                ],
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            if not candidates:
                continue
            detection = detect_weak_aspects(clause["clause_text"], self.taxonomy)
            negative_rule_aspects = set(detection["negative_aspects"])
            complaint_probability = float(complaint_probabilities[position])
            complaint_decision = (
                "negative" if complaint_probability >= upper else
                "non_negative" if complaint_probability <= lower else
                "uncertain"
            )
            for aspect, aspect_probability in candidates:
                explicit_negative = aspect in negative_rule_aspects
                is_negative = explicit_negative or complaint_decision == "negative"
                sentiment_label = "negative" if is_negative else complaint_decision
                sentiment_source = "taxonomy_and_complaint_model" if explicit_negative else "complaint_model"
                confidence = max(complaint_probability, 1 - complaint_probability)
                if explicit_negative:
                    confidence = max(confidence, 0.9)
                records.append(
                    {
                        **clause.to_dict(),
                        "aspect": aspect,
                        "aspect_probability": aspect_probability,
                        "aspect_source": "human_gold_aspect_model",
                        "complaint_probability": complaint_probability,
                        "sentiment_label": sentiment_label,
                        "is_negative": bool(is_negative),
                        "prediction_confidence": float(confidence),
                        "sentiment_source": sentiment_source,
                        "label_source": self.complaint_bundle["label_source"],
                        "model_version": self.complaint_bundle["version"],
                    }
                )
        return pd.DataFrame(records)

    def _cluster_places(self, places: pd.DataFrame) -> pd.DataFrame:
        combined_columns = [
            "canonical_place_id", "canonical_place_name", "place_category", "place_type",
            "address", "latitude", "longitude",
        ]
        baseline = self.baseline_places[combined_columns].copy()
        uploaded_ids = set(places["canonical_place_id"].astype(str))
        baseline = baseline.loc[~baseline["canonical_place_id"].astype(str).isin(uploaded_ids)]
        combined = pd.concat([baseline, places[combined_columns]], ignore_index=True)
        valid = combined.loc[valid_coordinate_mask(combined)].copy().reset_index(drop=True)
        if valid.empty:
            return pd.DataFrame(columns=combined_columns + ["geo_cluster_id"])
        labels = run_dbscan_haversine(valid, eps_km=5.0, min_samples=4)
        valid["geo_cluster_id"] = labels.astype(int)
        return valid.loc[valid["canonical_place_id"].astype(str).isin(uploaded_ids)].reset_index(drop=True)

    @staticmethod
    def _geojson(places: pd.DataFrame, clusters: pd.DataFrame, rankings: pd.DataFrame) -> dict[str, Any]:
        cluster_lookup = clusters.set_index("canonical_place_id") if not clusters.empty else pd.DataFrame()
        top_gap = rankings.drop_duplicates("canonical_place_id").set_index("canonical_place_id") if not rankings.empty else pd.DataFrame()
        features = []
        for place in places.itertuples(index=False):
            if pd.isna(place.latitude) or pd.isna(place.longitude):
                continue
            gap = top_gap.loc[place.canonical_place_id] if not top_gap.empty and place.canonical_place_id in top_gap.index else None
            cluster_id = None
            if not clusters.empty and place.canonical_place_id in cluster_lookup.index:
                cluster_id = int(cluster_lookup.loc[place.canonical_place_id]["geo_cluster_id"])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(place.longitude), float(place.latitude)]},
                    "properties": {
                        "feature_type": "place",
                        "canonical_place_id": str(place.canonical_place_id),
                        "place_name": str(place.canonical_place_name),
                        "place_category": str(place.place_category),
                        "geo_cluster_id": cluster_id,
                        "top_aspect": str(gap["aspect"]) if gap is not None else None,
                        "service_gap_score": round(float(gap["service_gap_score"]), 4) if gap is not None else None,
                        "confidence": str(gap["confidence_level"]) if gap is not None else None,
                        "priority": str(gap["priority_level"]) if gap is not None else None,
                        "review_count": int(gap["review_count"]) if gap is not None else 0,
                        "evidence_count": int(gap["aspect_mention_count"]) if gap is not None else 0,
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def _summary(
        self,
        *,
        import_id: str,
        filename: str,
        rows_received: int,
        normalized: pd.DataFrame,
        places: pd.DataFrame,
        reviews: pd.DataFrame,
        evidence: pd.DataFrame,
        rankings: pd.DataFrame,
        geojson: dict[str, Any],
        issues: list[dict[str, Any]],
        scope: str = "runtime_upload",
        target_place_id: str | None = None,
    ) -> dict[str, Any]:
        errors = [issue for issue in issues if issue["severity"] == "error"]
        warnings = [issue for issue in issues if issue["severity"] == "warning"]
        top_priorities = []
        if not rankings.empty:
            top_priorities = [self._ranking_item(row) for _, row in rankings.head(10).iterrows()]
        return {
            "import_id": import_id,
            "status": "completed_with_warnings" if issues else "completed",
            "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows_received": int(rows_received),
            "rows_accepted": int(len(normalized)),
            "rows_rejected": int(rows_received - len(normalized)),
            "place_count": int(len(places)),
            "review_count": int(len(reviews)),
            "places_with_coordinates": int(len(geojson["features"])),
            "evidence_count": int(len(evidence)),
            "ranking_count": int(len(rankings)),
            "sentiment_distribution": {
                str(key): int(value) for key, value in reviews["model_sentiment_label"].value_counts().items()
            },
            "complaint_distribution": {
                str(key): int(value) for key, value in reviews["complaint_decision"].value_counts().items()
            },
            "top_priorities": top_priorities,
            "warnings": warnings[:100],
            "errors": errors[:100],
            "model_version": self.model_versions,
            "training_performed": False,
            "scope": scope,
            "target_place_id": target_place_id,
            "published": False,
            "published_at": None,
        }

    @staticmethod
    def _ranking_item(row: pd.Series) -> dict[str, Any]:
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
            "reason_codes": [item for item in str(row["reason_codes_text"]).split("|") if item],
            "explanation": str(row["explanation"]),
        }

    def _persist(
        self,
        import_id: str,
        normalized: pd.DataFrame,
        places: pd.DataFrame,
        reviews: pd.DataFrame,
        evidence: pd.DataFrame,
        rankings: pd.DataFrame,
        clusters: pd.DataFrame,
        geojson: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        destination = self.runtime_dir / import_id
        with self.lock:
            destination.mkdir(parents=False, exist_ok=False)
            normalized.to_parquet(destination / "normalized_input.parquet", index=False)
            places.to_parquet(destination / "places.parquet", index=False)
            reviews.to_parquet(destination / "reviews.parquet", index=False)
            evidence.to_parquet(destination / "evidence.parquet", index=False)
            rankings.to_csv(destination / "service_gap_rankings.csv", index=False, encoding="utf-8")
            clusters.to_parquet(destination / "clusters.parquet", index=False)
            (destination / "geojson.json").write_text(
                json.dumps(_native(geojson), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            (destination / "summary.json").write_text(
                json.dumps(_native(summary), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

    def list_imports(self, limit: int = 20) -> list[dict[str, Any]]:
        summaries = []
        for path in self.runtime_dir.glob("imp-*/summary.json"):
            try:
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        summaries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return summaries[:limit]

    def get_summary(self, import_id: str) -> dict[str, Any] | None:
        path = self._safe_import_path(import_id) / "summary.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_rankings(self, import_id: str, limit: int, offset: int) -> dict[str, Any] | None:
        path = self._safe_import_path(import_id) / "service_gap_rankings.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        total = int(len(frame))
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [self._ranking_item(row) for _, row in frame.iloc[offset: offset + limit].iterrows()],
        }

    def get_geojson(self, import_id: str) -> dict[str, Any] | None:
        path = self._safe_import_path(import_id) / "geojson.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_evidence(
        self,
        import_id: str,
        *,
        place_id: str | None,
        aspect: str | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any] | None:
        path = self._safe_import_path(import_id) / "evidence.parquet"
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        if frame.empty:
            return {"total": 0, "total_all": 0, "limit": limit, "offset": offset, "aspect_counts": {}, "items": []}
        frame = frame.loc[frame["is_negative"].fillna(False)].copy()
        if place_id:
            frame = frame.loc[frame["canonical_place_id"].astype(str) == place_id]
        total_all = int(len(frame))
        aspect_counts = {str(key): int(value) for key, value in frame["aspect"].value_counts().items()}
        if aspect:
            frame = frame.loc[frame["aspect"].astype(str).str.casefold() == aspect.casefold()]
        frame = frame.sort_values(["complaint_probability", "prediction_confidence"], ascending=False)
        frame = frame.drop_duplicates(["clause_text", "aspect"])
        total = int(len(frame))
        items = [
            {
                "text": str(row["clause_text"]),
                "aspect": str(row["aspect"]),
                "complaint_probability": round(float(row["complaint_probability"]), 6),
                "confidence": round(float(row["prediction_confidence"]), 6),
                "sentiment_source": str(row["sentiment_source"]),
            }
            for _, row in frame.iloc[offset: offset + limit].iterrows()
        ]
        return {"total": total, "total_all": total_all, "limit": limit, "offset": offset, "aspect_counts": aspect_counts, "items": items}

    def published_import_ids(self) -> list[str]:
        path = self.runtime_dir / "published_imports.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [str(value) for value in payload.get("import_ids", []) if isinstance(value, str)]

    def publish(self, import_id: str) -> dict[str, Any]:
        destination = self._safe_import_path(import_id)
        summary_path = destination / "summary.json"
        if not summary_path.exists():
            raise ImportDataError("Hasil impor tidak ditemukan.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("target_place_id"):
            raise ImportDataError("Hanya impor ulasan untuk destinasi terpilih yang dapat diterbitkan.")
        with self.lock:
            import_ids = self.published_import_ids()
            if import_id not in import_ids:
                import_ids.append(import_id)
            manifest = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "import_ids": import_ids,
            }
            (self.runtime_dir / "published_imports.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary["published"] = True
            summary["published_at"] = datetime.now(timezone.utc).isoformat()
            summary_path.write_text(
                json.dumps(_native(summary), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        return summary

    def unpublish(self, import_id: str) -> dict[str, Any]:
        destination = self._safe_import_path(import_id)
        summary_path = destination / "summary.json"
        if not summary_path.exists():
            raise ImportDataError("Hasil impor tidak ditemukan.")
        with self.lock:
            import_ids = [value for value in self.published_import_ids() if value != import_id]
            manifest = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "import_ids": import_ids,
            }
            (self.runtime_dir / "published_imports.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["published"] = False
            summary["published_at"] = None
            summary_path.write_text(
                json.dumps(_native(summary), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        return summary

    def load_published_frames(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        review_frames: list[pd.DataFrame] = []
        evidence_frames: list[pd.DataFrame] = []
        for import_id in self.published_import_ids():
            destination = self._safe_import_path(import_id)
            reviews_path = destination / "reviews.parquet"
            evidence_path = destination / "evidence.parquet"
            if reviews_path.exists():
                review_frames.append(pd.read_parquet(reviews_path))
            if evidence_path.exists():
                evidence_frames.append(pd.read_parquet(evidence_path))
        return review_frames, evidence_frames

    def _safe_import_path(self, import_id: str) -> Path:
        if not re.fullmatch(r"imp-[0-9]{14}-[a-f0-9]{8}", import_id):
            raise ImportDataError("ID impor tidak valid.")
        return self.runtime_dir / import_id
