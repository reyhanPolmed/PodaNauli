import pandas as pd

from src.train_aspect import (
    create_aspect_annotation_sample,
    detect_weak_aspects,
    evaluate_multilabel_predictions,
    preserve_aspect_annotation_sample,
    prepare_aspect_dataset,
    train_gold_aspect_model,
)
import numpy as np


ASPECTS = [
    {
        "id": "toilet",
        "keywords_positive": ["toilet bersih"],
        "keywords_negative": ["toilet kotor"],
        "related_facility_types": ["toilet"],
    },
    {
        "id": "parkir",
        "keywords_positive": ["parkir luas"],
        "keywords_negative": ["parkir sempit"],
        "related_facility_types": ["parkir"],
    },
    {
        "id": "lainnya",
        "keywords_positive": [],
        "keywords_negative": [],
        "related_facility_types": [],
    },
]


def test_detect_weak_aspects_positive_and_negative_hits():
    detected = detect_weak_aspects("Toilet kotor dan parkir sempit", ASPECTS)
    assert detected["weak_aspects"] == ["parkir", "toilet"]
    assert detected["negative_aspects"] == ["parkir", "toilet"]


def test_detect_weak_aspects_falls_back_to_lainnya():
    detected = detect_weak_aspects("Pemandangannya sangat indah", ASPECTS)
    assert detected["weak_aspects"] == ["lainnya"]


def test_prepare_aspect_dataset_filters_duplicate_rows():
    reviews = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "review_text_clean": "toilet kotor",
                "review_text_raw": "toilet kotor",
                "text_length": 12,
                "is_duplicate": False,
                "place_name": "Place",
                "place_category": "wisata",
                "canonical_place_id": "p1",
                "weak_sentiment_label": "negative",
            },
            {
                "review_id": "r2",
                "review_text_clean": "parkir luas",
                "review_text_raw": "parkir luas",
                "text_length": 11,
                "is_duplicate": True,
                "place_name": "Place",
                "place_category": "wisata",
                "canonical_place_id": "p1",
                "weak_sentiment_label": "positive",
            },
        ]
    )
    dataset = prepare_aspect_dataset(reviews, ASPECTS)
    assert dataset["review_id"].tolist() == ["r1"]
    assert dataset.iloc[0]["weak_aspects"] == ["toilet"]


def test_create_aspect_annotation_sample_has_blank_manual_fields():
    reviews = pd.DataFrame(
        [
            {
                "review_id": f"r{i}",
                "review_text_raw": "toilet kotor",
                "review_text_clean": "toilet kotor",
                "text_length": 12,
                "is_duplicate": False,
                "place_name": "Place",
                "place_category": "wisata",
                "canonical_place_id": f"p{i}",
                "weak_sentiment_label": "negative",
            }
            for i in range(10)
        ]
    )
    dataset = prepare_aspect_dataset(reviews, ASPECTS)
    sample, report = create_aspect_annotation_sample(dataset, sample_size=5, random_seed=42)
    assert len(sample) == 5
    assert sample["manual_aspects"].fillna("").eq("").all()
    assert report["sample_size"] == 5


def test_aspect_annotation_is_preserved(tmp_path):
    sample = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "place_name": "Place",
                "review_text_raw": "toilet kotor",
                "weak_aspects": "toilet",
                "manual_aspects": "",
                "annotation_notes": "",
            }
        ]
    )
    existing = sample.copy()
    existing["manual_aspects"] = "toilet"
    existing["annotation_notes"] = "checked"
    path = tmp_path / "aspect.csv"
    existing.to_csv(path, index=False, encoding="utf-8-sig")
    merged = preserve_aspect_annotation_sample(sample, path)
    assert merged.loc[0, "manual_aspects"] == "toilet"
    assert merged.loc[0, "annotation_notes"] == "checked"


def test_multilabel_metrics_include_none_and_per_aspect():
    y_true = np.array([[1, 0], [0, 1], [0, 0]])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.2]])
    metrics = evaluate_multilabel_predictions(
        y_true,
        probabilities,
        ["toilet", "parkir"],
        threshold=0.5,
    )
    assert metrics["micro_f1"] == 1.0
    assert metrics["none_f1"] == 1.0
    assert metrics["per_aspect"]["toilet"]["recall"] == 1.0


def test_gold_training_uses_validation_then_locked_test(monkeypatch):
    aspect_labels = [item["id"] for item in ASPECTS]

    def frame(prefix, rows):
        records = []
        for index in range(rows):
            label = aspect_labels[index % len(aspect_labels)]
            records.append(
                {
                    "clause_id": f"{prefix}_{index}",
                    "review_id": f"{prefix}_r{index}",
                    "canonical_place_id": f"{prefix}_p{index // 2}",
                    "clause_text": f"{label} {label} quality {index % 3}",
                    "manual_aspects": label,
                    "annotator_id": "A01",
                    "human_approved": "yes",
                }
            )
        return pd.DataFrame(records)

    monkeypatch.setattr(
        "src.aspect_gold_dataset.load_aspect_gold_splits",
        lambda: (
            frame("train", 30),
            frame("validation", 15),
            frame("test", 15),
            {"group_overlap_count": 0},
        ),
    )
    config = {
        "random_seed": 42,
        "aspect_improvement": {
            "model": {"c_values": [1.0], "threshold_values": [0.3, 0.5]},
            "annotation": {"key_aspects": ["toilet", "parkir"]},
            "acceptance_gates": {
                "micro_f1": 0.0,
                "macro_f1": 0.0,
                "key_aspect_minimum_f1": 0.0,
            },
        },
    }
    _model, mlb, report, comparison = train_gold_aspect_model(config, aspect_labels)
    assert list(mlb.classes_) == aspect_labels
    assert report["selection_split"] == "validation"
    assert report["final_test_split"] == "test"
    assert len(comparison) == 2
