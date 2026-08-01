import pandas as pd

from src.aspect_gold_dataset import create_locked_aspect_split, semantic_sha256


def _gold_frame() -> pd.DataFrame:
    rows = []
    for group_index in range(18):
        for row_index in range(4):
            label = ["harga", "pelayanan", "none"][row_index % 3]
            rows.append(
                {
                    "clause_id": f"p{group_index}::c{row_index}",
                    "review_id": f"r{group_index}_{row_index}",
                    "canonical_place_id": f"p{group_index}",
                    "place_name": f"Place {group_index}",
                    "place_category": "hotel",
                    "clause_index": str(row_index),
                    "clause_text": f"clause {group_index} {row_index}",
                    "weak_aspects": label,
                    "manual_aspects": label,
                    "annotator_id": "A01",
                    "human_approved": "yes",
                    "annotation_notes": "",
                }
            )
    return pd.DataFrame(rows)


def test_aspect_split_is_place_disjoint_and_complete():
    gold = _gold_frame()
    split, report = create_locked_aspect_split(gold, random_seed=42, attempts=20)
    assert len(split) == len(gold)
    assert report["group_overlap_count"] == 0
    assert set(split["split"]) == {"train", "validation", "test"}


def test_aspect_semantic_hash_is_order_independent():
    gold = _gold_frame()
    shuffled = gold.sample(frac=1.0, random_state=7)
    assert semantic_sha256(gold) == semantic_sha256(shuffled)
