import pandas as pd

from src.gap_scoring import (
    bayesian_smoothed_rate,
    compute_service_gap_scores,
    export_service_gap_validation_sample,
    metadata_completeness,
    rating_gap,
    review_volume_confidence,
)


def test_bayesian_smoothing_avoids_extreme_for_tiny_sample():
    assert bayesian_smoothed_rate(0, 1, alpha=1, beta=4) > 0
    assert bayesian_smoothed_rate(1, 1, alpha=1, beta=4) < 1


def test_review_volume_confidence_bounds():
    assert 0 <= review_volume_confidence(0, 30) <= 1
    assert review_volume_confidence(100, 30) == 1.0


def test_rating_gap_bounds():
    assert rating_gap(5.0) == 0.0
    assert rating_gap(1.0) == 1.0
    assert 0 <= rating_gap(None) <= 1


def test_metadata_completeness_bounds():
    row = pd.Series({"place_category": "wisata", "latitude": 2.0, "longitude": 99.0})
    assert 0 <= metadata_completeness(row) <= 1


def test_service_gap_scores_are_in_range():
    places = pd.DataFrame(
        [
            {
                "canonical_place_id": "p1",
                "canonical_place_name": "Place 1",
                "place_category": "wisata",
                "place_type": "Wisata Alam",
                "address": "Addr",
                "latitude": 2.0,
                "longitude": 99.0,
                "place_rating": 4.0,
                "status": "beroperasi",
                "price_text_original": "Gratis",
                "facility_text": "toilet",
            },
            {
                "canonical_place_id": "p2",
                "canonical_place_name": "Place 2",
                "place_category": "wisata",
                "place_type": "Wisata Alam",
                "address": "Addr",
                "latitude": 2.01,
                "longitude": 99.01,
                "place_rating": 5.0,
                "status": "beroperasi",
                "price_text_original": "Gratis",
                "facility_text": "parkir",
            },
        ]
    )
    reviews = pd.DataFrame(
        [
            {"review_id": "r1", "canonical_place_id": "p1", "review_date": "2025-01-01"},
            {"review_id": "r2", "canonical_place_id": "p1", "review_date": "2025-01-02"},
        ]
    )
    aspects = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "canonical_place_id": "p1",
                "weak_aspects_text": "toilet",
                "weak_negative_aspects_text": "toilet",
                "weak_sentiment_label": "negative",
            }
        ]
    )
    config = {
        "bayesian_smoothing": {"alpha": 1.0, "beta": 4.0},
        "weights": {
            "negative_sentiment_rate_smoothed": 0.30,
            "aspect_mention_frequency": 0.20,
            "service_scarcity": 0.15,
            "review_volume_confidence": 0.15,
            "rating_gap": 0.10,
            "recency_factor": 0.05,
            "metadata_completeness_gap": 0.05,
        },
        "thresholds": {
            "high_negative_rate": 0.35,
            "frequent_complaint": 0.15,
            "low_nearby_service_density": 0.70,
            "high_review_confidence": 0.70,
            "low_data_reliability": 0.55,
            "volume_confidence_review_count": 30,
            "recency_days": 365,
        },
        "service_scarcity": {
            "category_density_radius_km": 5.0,
            "scarcity_review_count_floor": 1,
        },
    }
    scores = compute_service_gap_scores(places, reviews, aspects, config, ["toilet"])
    assert not scores.empty
    assert scores["service_gap_score"].between(0, 100).all()


def test_validation_sample_includes_negative_clause_evidence(tmp_path, monkeypatch):
    scores = pd.DataFrame(
        [
            {
                "rank": 1,
                "canonical_place_id": "p1",
                "place_name": "Place 1",
                "place_category": "wisata",
                "aspect": "toilet",
                "service_gap_score": 80.0,
                "negative_mention_count": 2,
                "aspect_mention_count": 3,
                "explanation": "Toilet perlu diperbaiki.",
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "canonical_place_id": "p1",
                "aspect": "toilet",
                "clause_text": "toilet kotor",
                "is_negative": True,
                "prediction_confidence": 0.9,
            }
        ]
    )
    output = tmp_path / "validation.csv"
    monkeypatch.setattr("src.gap_scoring.RANKING_VALIDATION_PATH", output)
    report = export_service_gap_validation_sample(scores, aspect_sentiment=evidence)
    saved = pd.read_csv(
        output,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    assert saved.loc[0, "evidence_clause_1"] == "toilet kotor"
    assert report["sample_rows"] == 1
    assert report["fully_reviewed"] is False
    saved.loc[0, "manual_evidence_valid"] = "yes"
    saved.loc[0, "manual_priority_valid"] = "yes"
    saved.loc[0, "validator_id"] = "A01"
    saved.to_csv(output, index=False, encoding="utf-8-sig")
    reviewed = export_service_gap_validation_sample(scores, aspect_sentiment=evidence)
    assert reviewed["fully_reviewed"] is True
    assert reviewed["evidence_validity_rate"] == 1.0
    assert reviewed["priority_validity_rate"] == 1.0
    assert reviewed["validity_rate"] == 1.0
