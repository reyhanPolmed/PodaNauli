import pandas as pd

from src.aspect_annotation import (
    create_clause_annotation_queue,
    normalize_aspect_labels,
    validate_aspect_annotation_frame,
)


def test_normalize_multilabel_and_exclusive_none():
    valid = {"harga", "pelayanan", "lainnya"}
    labels, error = normalize_aspect_labels("pelayanan|harga", valid)
    assert labels == ["harga", "pelayanan"]
    assert error is None
    labels, error = normalize_aspect_labels("none|harga", valid)
    assert labels == []
    assert error == "none must be the only label"


def test_clause_queue_is_multilabel_ready_and_place_capped():
    candidates = pd.DataFrame(
        [
            {
                "clause_id": f"r{i}::c000",
                "review_id": f"r{i}",
                "canonical_place_id": f"p{i % 4}",
                "place_name": f"Place {i % 4}",
                "place_category": "hotel",
                "clause_index": 0,
                "clause_text": "harga mahal dan pelayanan lambat",
                "weak_aspects": "harga|pelayanan",
                "weak_aspect_list": ["harga", "pelayanan"],
            }
            for i in range(20)
        ]
    )
    queue, report = create_clause_annotation_queue(
        candidates,
        target_rows=8,
        max_per_place=2,
        random_seed=42,
    )
    assert len(queue) == 8
    assert queue.groupby("canonical_place_id").size().max() <= 2
    assert queue["manual_aspects"].eq("").all()
    assert report["queue_rows"] == 8


def test_validated_gold_accepts_multilabel_and_none():
    queue = pd.DataFrame(
        [
            {
                "clause_id": "r1::c000",
                "review_id": "r1",
                "canonical_place_id": "p1",
                "place_name": "Place",
                "place_category": "hotel",
                "clause_index": "0",
                "clause_text": "harga mahal dan pelayanan lambat",
                "weak_aspects": "harga|pelayanan",
                "manual_aspects": "harga|pelayanan",
                "annotator_id": "A01",
                "human_approved": "yes",
                "annotation_notes": "",
            },
            {
                "clause_id": "r2::c000",
                "review_id": "r2",
                "canonical_place_id": "p2",
                "place_name": "Place 2",
                "place_category": "hotel",
                "clause_index": "0",
                "clause_text": "saya datang kemarin",
                "weak_aspects": "lainnya",
                "manual_aspects": "none",
                "annotator_id": "A01",
                "human_approved": "yes",
                "annotation_notes": "",
            },
        ]
    )
    gold, report = validate_aspect_annotation_frame(queue)
    assert len(gold) == 2
    assert report["gold_label_counts"]["harga"] == 1
    assert report["gold_label_counts"]["pelayanan"] == 1
    assert report["gold_label_counts"]["none"] == 1
