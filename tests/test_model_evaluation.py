import pandas as pd

from src.model_evaluation import (
    calculate_project_readiness,
    error_tags,
    has_mixed_language,
    service_gap_stability,
)


def test_has_mixed_language_detects_basic_mix():
    assert has_mixed_language("Pelayanan bagus and the food is good")
    assert not has_mixed_language("Pelayanan bagus dan makanan enak")


def test_error_tags_detects_rating_text_mismatch_and_typo():
    tags = error_tags(
        "tmpt bagus tapi toilet kotor",
        rating=5.0,
        actual="positive",
        predicted="negative",
        place_review_count=2,
    )
    assert "MIXED_POSITIVE_NEGATIVE" in tags
    assert "TYPO_OR_INFORMAL_ABBREVIATION" in tags
    assert "HIGH_RATING_WITH_NEGATIVE_TEXT" in tags
    assert "LOW_PLACE_REVIEW_COUNT" in tags


def test_service_gap_stability_reads_existing_scores():
    summary = service_gap_stability()
    assert summary["rows"] > 0
    assert 0 <= summary["score_min"] <= summary["score_max"] <= 100


def test_project_readiness_requires_models_and_complete_top20():
    registry = {
        "models": [
            {"model_name": "sentiment_champion", "deployment_ready": True},
            {"model_name": "complaint_detector", "deployment_ready": True},
            {"model_name": "aspect_champion", "deployment_ready": True},
        ]
    }
    methodology = {
        "human_ranking_validation": {
            "fully_reviewed": True,
            "evidence_validity_rate": 0.80,
            "priority_validity_rate": 0.95,
            "validity_rate": 0.80,
        }
    }
    config = {
        "sentiment_improvement": {
            "acceptance_gates": {"top_20_service_gap_validity": 0.80}
        }
    }
    readiness = calculate_project_readiness(registry, methodology, config)
    assert readiness["model_and_ranking_pipeline_ready"] is True
    assert readiness["production_application_ready"] is False
