import pandas as pd

from src.topic_modeling import infer_topic_aspect, prepare_topic_corpus, top_keywords


def test_prepare_topic_corpus_filters_duplicates_and_empty_text():
    reviews = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "review_text_clean": "toilet kotor",
                "text_length": 12,
                "is_duplicate": False,
                "place_category": "wisata",
                "weak_sentiment_label": "negative",
            },
            {
                "review_id": "r2",
                "review_text_clean": "bagus",
                "text_length": 5,
                "is_duplicate": True,
                "place_category": "wisata",
                "weak_sentiment_label": "positive",
            },
            {
                "review_id": "r3",
                "review_text_clean": None,
                "text_length": 0,
                "is_duplicate": False,
                "place_category": "hotel",
                "weak_sentiment_label": "neutral",
            },
        ]
    )
    corpus = prepare_topic_corpus(reviews)
    assert corpus["review_id"].tolist() == ["r1"]


def test_top_keywords_returns_ranked_terms():
    keywords = top_keywords(
        components=__import__("numpy").array([[0.1, 0.7, 0.2]]),
        feature_names=__import__("numpy").array(["akses", "toilet", "harga"]),
        n_top_words=2,
    )
    assert [item["keyword"] for item in keywords[0]] == ["toilet", "harga"]


def test_infer_topic_aspect_uses_taxonomy_overlap():
    aspects = [
        {
            "id": "toilet",
            "keywords_positive": ["toilet bersih"],
            "keywords_negative": ["toilet kotor"],
            "related_facility_types": ["toilet"],
        }
    ]
    aspect, hits = infer_topic_aspect("toilet kotor fasilitas", aspects)
    assert aspect == "toilet"
    assert hits >= 1
