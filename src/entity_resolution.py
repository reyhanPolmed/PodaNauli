"""Entity resolution utilities for matching place names across sheets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalize_place_name


@dataclass(frozen=True)
class MatchThresholds:
    fuzzy_auto: float = 94.0
    fuzzy_review: float = 88.0
    tfidf_auto: float = 0.94
    tfidf_review: float = 0.86


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _build_tfidf_lookup(canonical_names: list[str]) -> tuple[TfidfVectorizer | None, Any]:
    if not canonical_names:
        return None, None
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(canonical_names)
    return vectorizer, matrix


def resolve_entities(
    source_places: pd.DataFrame,
    thresholds: MatchThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve source place rows into canonical place ids.

    Expected columns are source_sheet, source_place_name, normalized_place_name,
    place_category, address, latitude, longitude, and place_type.
    """
    thresholds = thresholds or MatchThresholds()
    rows = source_places.copy()
    rows = rows.dropna(subset=["normalized_place_name"]).copy()
    rows = rows.sort_values(["source_priority", "normalized_place_name", "source_sheet"]).reset_index(drop=True)

    canonical_records: list[dict[str, Any]] = []
    mapping_records: list[dict[str, Any]] = []

    for _, row in rows.iterrows():
        normalized = _safe_text(row["normalized_place_name"])
        source_name = _safe_text(row["source_place_name"])
        category = _safe_text(row.get("place_category"))
        address = _safe_text(row.get("address"))
        lat = row.get("latitude")
        lon = row.get("longitude")
        place_type = _safe_text(row.get("place_type"))

        exact_candidates = [
            record for record in canonical_records
            if record["normalized_place_name"] == normalized
        ]
        if exact_candidates:
            chosen = exact_candidates[0]
            match_score = 100.0
            match_method = "normalized_exact"
            match_status = "auto"
            needs_manual_review = False
            candidate_canonical_place_id = None
            candidate_canonical_place_name = None
        else:
            chosen = None
            match_score = 0.0
            match_method = "new_canonical"
            match_status = "new"
            needs_manual_review = False
            candidate_canonical_place_id = None
            candidate_canonical_place_name = None

            canonical_names = [record["normalized_place_name"] for record in canonical_records]
            best_fuzzy_idx = None
            best_fuzzy_score = 0.0
            for idx, candidate_name in enumerate(canonical_names):
                score = float(fuzz.WRatio(normalized, candidate_name))
                if score > best_fuzzy_score:
                    best_fuzzy_score = score
                    best_fuzzy_idx = idx

            best_tfidf_idx = None
            best_tfidf_score = 0.0
            if canonical_names:
                vectorizer, matrix = _build_tfidf_lookup(canonical_names)
                if vectorizer is not None and matrix is not None:
                    query = vectorizer.transform([normalized])
                    similarities = cosine_similarity(query, matrix)[0]
                    best_tfidf_idx = int(similarities.argmax())
                    best_tfidf_score = float(similarities[best_tfidf_idx])

            evidence_bonus = 0.0
            candidate_idx = best_fuzzy_idx
            if best_tfidf_idx is not None and best_tfidf_score * 100 > best_fuzzy_score:
                candidate_idx = best_tfidf_idx
            if candidate_idx is not None:
                candidate = canonical_records[candidate_idx]
                if category and candidate.get("place_category") == category:
                    evidence_bonus += 2.0
                if place_type and candidate.get("place_type") == place_type:
                    evidence_bonus += 1.0
                if pd.notna(lat) and pd.notna(lon) and pd.notna(candidate.get("latitude")) and pd.notna(candidate.get("longitude")):
                    if abs(float(lat) - float(candidate["latitude"])) <= 0.002 and abs(float(lon) - float(candidate["longitude"])) <= 0.002:
                        evidence_bonus += 3.0

                combined_score = max(best_fuzzy_score, best_tfidf_score * 100.0) + evidence_bonus
                if combined_score >= thresholds.fuzzy_auto:
                    chosen = candidate
                    match_score = min(combined_score, 100.0)
                    match_method = "fuzzy_tfidf_auto"
                    match_status = "auto"
                    needs_manual_review = False
                elif combined_score >= thresholds.fuzzy_review:
                    match_score = min(combined_score, 100.0)
                    match_method = "fuzzy_tfidf_candidate"
                    match_status = "new_needs_manual_review"
                    needs_manual_review = True
                    candidate_canonical_place_id = candidate["canonical_place_id"]
                    candidate_canonical_place_name = candidate["canonical_place_name"]

            if chosen is None:
                canonical_id = f"place_{len(canonical_records) + 1:05d}"
                chosen = {
                    "canonical_place_id": canonical_id,
                    "canonical_place_name": source_name,
                    "normalized_place_name": normalized,
                    "place_category": category or None,
                    "place_type": place_type or None,
                    "address": address or None,
                    "latitude": lat,
                    "longitude": lon,
                    "source_sheets": set(),
                }
                canonical_records.append(chosen)

        chosen["source_sheets"].add(row["source_sheet"])
        if not chosen.get("place_category") and category:
            chosen["place_category"] = category
        if not chosen.get("place_type") and place_type:
            chosen["place_type"] = place_type
        if not chosen.get("address") and address:
            chosen["address"] = address
        if pd.isna(chosen.get("latitude")) and pd.notna(lat):
            chosen["latitude"] = lat
            chosen["longitude"] = lon

        mapping_records.append(
            {
                "canonical_place_id": chosen["canonical_place_id"],
                "canonical_place_name": chosen["canonical_place_name"],
                "source_sheet": row["source_sheet"],
                "source_place_name": source_name,
                "normalized_place_name": normalized,
                "match_score": round(float(match_score), 4),
                "match_method": match_method,
                "match_status": match_status,
                "needs_manual_review": bool(needs_manual_review),
                "candidate_canonical_place_id": candidate_canonical_place_id,
                "candidate_canonical_place_name": candidate_canonical_place_name,
            }
        )

    canonical_output = []
    for record in canonical_records:
        output = {key: value for key, value in record.items() if key != "source_sheets"}
        output["source_sheets"] = sorted(record["source_sheets"])
        canonical_output.append(output)

    return pd.DataFrame(canonical_output), pd.DataFrame(mapping_records)
