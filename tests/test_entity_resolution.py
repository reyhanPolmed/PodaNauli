import pandas as pd

from src.entity_resolution import MatchThresholds, resolve_entities


def test_entity_resolution_exact_match():
    source_places = pd.DataFrame(
        [
            {
                "source_sheet": "wisata-metadata",
                "source_priority": 0,
                "source_place_name": "Pantai Bulbul",
                "normalized_place_name": "pantai bulbul",
                "place_category": "wisata",
                "address": "Balige",
                "latitude": 2.0,
                "longitude": 99.0,
                "place_type": "Wisata Alam",
            },
            {
                "source_sheet": "wisata-v2",
                "source_priority": 1,
                "source_place_name": "Pantai Bulbul",
                "normalized_place_name": "pantai bulbul",
                "place_category": "wisata",
                "address": None,
                "latitude": None,
                "longitude": None,
                "place_type": None,
            },
        ]
    )
    places, mapping = resolve_entities(source_places)
    assert len(places) == 1
    assert mapping.iloc[1]["match_method"] == "normalized_exact"


def test_entity_resolution_ambiguous_fuzzy_is_not_forced_merge():
    source_places = pd.DataFrame(
        [
            {
                "source_sheet": "wisata-metadata",
                "source_priority": 0,
                "source_place_name": "Air Terjun Binanga Bolon",
                "normalized_place_name": "air terjun binanga bolon",
                "place_category": "wisata",
                "address": None,
                "latitude": None,
                "longitude": None,
                "place_type": "Wisata Alam",
            },
            {
                "source_sheet": "wisata-metadata",
                "source_priority": 0,
                "source_place_name": "Air Terjun Janji",
                "normalized_place_name": "air terjun janji",
                "place_category": "wisata",
                "address": None,
                "latitude": None,
                "longitude": None,
                "place_type": "Wisata Alam",
            },
        ]
    )
    places, mapping = resolve_entities(source_places, MatchThresholds(fuzzy_auto=99.0, fuzzy_review=80.0))
    assert len(places) == 2
    assert bool(mapping.iloc[1]["needs_manual_review"]) is True
    assert mapping.iloc[1]["match_status"] == "new_needs_manual_review"
