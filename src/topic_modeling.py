"""Tahap 5 pipeline: CPU topic modeling and aspect taxonomy reporting."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


CONFIG_PATH = ROOT / "configs" / "config.yaml"
ASPECT_TAXONOMY_PATH = ROOT / "configs" / "aspect_taxonomy.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"

INDONESIAN_STOPWORDS = {
    "ada",
    "agar",
    "akan",
    "aku",
    "amat",
    "anda",
    "apa",
    "atau",
    "baik",
    "banyak",
    "baru",
    "begitu",
    "belum",
    "bisa",
    "buat",
    "dalam",
    "dan",
    "dapat",
    "dari",
    "dengan",
    "di",
    "dia",
    "ini",
    "itu",
    "jadi",
    "juga",
    "kalau",
    "kami",
    "karena",
    "ke",
    "lebih",
    "mereka",
    "nya",
    "oleh",
    "pada",
    "saat",
    "saja",
    "sangat",
    "saya",
    "sebagai",
    "sebuah",
    "semua",
    "seperti",
    "serta",
    "sudah",
    "supaya",
    "tapi",
    "telah",
    "tempat",
    "tempatnya",
    "tersebut",
    "untuk",
    "yang",
    "yg",
    "ya",
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_topic_config() -> dict[str, Any]:
    config = load_yaml(CONFIG_PATH)
    topic_config = config.get("topic_modeling", {})
    return {
        "n_topics": int(topic_config.get("n_topics", 12)),
        "max_features": int(topic_config.get("max_features", 8000)),
        "min_df": int(topic_config.get("min_df", 5)),
        "max_df": float(topic_config.get("max_df", 0.9)),
        "n_top_words": int(topic_config.get("n_top_words", 15)),
        "random_seed": int(topic_config.get("random_seed", config.get("random_seed", 42))),
    }


def load_aspect_taxonomy(path: Path = ASPECT_TAXONOMY_PATH) -> list[dict[str, Any]]:
    data = load_yaml(path)
    aspects = data.get("aspects", [])
    required = {"id", "display_name", "description", "keywords_positive", "keywords_negative", "related_facility_types"}
    for index, aspect in enumerate(aspects):
        missing = required - set(aspect)
        if missing:
            raise ValueError(f"Aspect taxonomy row {index} missing keys: {sorted(missing)}")
    return aspects


def prepare_topic_corpus(reviews: pd.DataFrame) -> pd.DataFrame:
    """Filter usable non-duplicate review text for topic modeling."""
    required = {"review_id", "review_text_clean", "text_length", "is_duplicate", "place_category", "weak_sentiment_label"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Missing required review columns: {sorted(missing)}")
    corpus = reviews[
        reviews["review_text_clean"].notna()
        & (reviews["text_length"] > 0)
        & (~reviews["is_duplicate"])
    ].copy()
    corpus["review_text_clean"] = corpus["review_text_clean"].astype(str)
    corpus = corpus.sort_values("review_id").reset_index(drop=True)
    return corpus


def top_keywords(components: np.ndarray, feature_names: np.ndarray, n_top_words: int) -> list[list[dict[str, Any]]]:
    """Extract top weighted keywords per NMF topic."""
    all_keywords: list[list[dict[str, Any]]] = []
    for component in components:
        top_indexes = component.argsort()[-n_top_words:][::-1]
        all_keywords.append(
            [
                {
                    "keyword": str(feature_names[index]),
                    "weight": float(component[index]),
                }
                for index in top_indexes
            ]
        )
    return all_keywords


def representative_reviews(corpus: pd.DataFrame, document_topic_matrix: np.ndarray, topic_id: int, limit: int = 3) -> list[str]:
    topic_scores = document_topic_matrix[:, topic_id]
    top_indexes = topic_scores.argsort()[-limit:][::-1]
    return [str(corpus.iloc[index]["review_text_clean"])[:300] for index in top_indexes if topic_scores[index] > 0]


def infer_topic_aspect(keyword_text: str, aspects: list[dict[str, Any]]) -> tuple[str | None, int]:
    """Map topic keywords to the taxonomy only when direct keyword evidence exists."""
    text = f" {re.sub(r'[^0-9a-zA-Z_ ]+', ' ', keyword_text.lower())} "
    text = re.sub(r"\s+", " ", text)
    best_aspect = None
    best_hits = 0
    for aspect in aspects:
        candidates = aspect.get("keywords_positive", []) + aspect.get("keywords_negative", []) + aspect.get("related_facility_types", [])
        hits = 0
        for keyword in candidates:
            keyword_text_clean = str(keyword).lower().strip()
            if not keyword_text_clean:
                continue
            if " " in keyword_text_clean:
                matched = keyword_text_clean in text
            else:
                matched = f" {keyword_text_clean} " in text
            if matched:
                hits += 1
        if hits > best_hits:
            best_aspect = aspect["id"]
            best_hits = hits
    return best_aspect, best_hits


def run_topic_modeling() -> dict[str, Any]:
    config = load_topic_config()
    aspects = load_aspect_taxonomy()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path = PROCESSED_DIR / "reviews_clean.parquet"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing processed reviews: {reviews_path}")
    reviews = pd.read_parquet(reviews_path)
    corpus = prepare_topic_corpus(reviews)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        token_pattern=r"(?u)\b[\w][\w\-]+\b",
        ngram_range=(1, 2),
        min_df=config["min_df"],
        max_df=config["max_df"],
        max_features=config["max_features"],
        stop_words=sorted(INDONESIAN_STOPWORDS | set(ENGLISH_STOP_WORDS)),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(corpus["review_text_clean"])
    n_topics = min(config["n_topics"], tfidf.shape[0], tfidf.shape[1])
    if n_topics < 2:
        raise ValueError("Not enough corpus/features for NMF topic modeling.")

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=config["random_seed"],
        max_iter=500,
    )
    document_topic_matrix = model.fit_transform(tfidf)
    topic_assignments = document_topic_matrix.argmax(axis=1)
    topic_strength = document_topic_matrix.max(axis=1)
    feature_names = vectorizer.get_feature_names_out()
    keywords_by_topic = top_keywords(model.components_, feature_names, config["n_top_words"])

    topic_rows = []
    keyword_rows = []
    for topic_id, keywords in enumerate(keywords_by_topic):
        topic_mask = topic_assignments == topic_id
        topic_docs = corpus.loc[topic_mask].copy()
        keyword_string = ", ".join(item["keyword"] for item in keywords)
        mapped_aspect, evidence_hits = infer_topic_aspect(keyword_string, aspects)
        dominant_category = topic_docs["place_category"].mode().iloc[0] if not topic_docs.empty else None
        dominant_sentiment = topic_docs["weak_sentiment_label"].mode().iloc[0] if not topic_docs.empty else None
        topic_rows.append(
            {
                "topic_id": topic_id,
                "review_count": int(topic_mask.sum()),
                "review_share": float(topic_mask.mean()),
                "dominant_place_category": dominant_category,
                "dominant_weak_sentiment": dominant_sentiment,
                "mean_topic_strength": float(topic_strength[topic_mask].mean()) if topic_mask.any() else 0.0,
                "top_keywords": keyword_string,
                "mapped_initial_aspect": mapped_aspect,
                "taxonomy_keyword_hits": evidence_hits,
                "representative_reviews": " || ".join(representative_reviews(corpus, document_topic_matrix, topic_id)),
            }
        )
        for rank, item in enumerate(keywords, start=1):
            keyword_rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "keyword": item["keyword"],
                    "weight": item["weight"],
                    "mapped_initial_aspect": mapped_aspect,
                }
            )

    topic_summary = pd.DataFrame(topic_rows).sort_values("review_count", ascending=False)
    topic_keywords = pd.DataFrame(keyword_rows)
    topic_summary.to_csv(REPORT_DIR / "topic_summary.csv", index=False, encoding="utf-8")
    topic_keywords.to_csv(REPORT_DIR / "topic_keywords.csv", index=False, encoding="utf-8")

    support_rows = []
    for aspect in aspects:
        matched_topics = topic_summary[topic_summary["mapped_initial_aspect"] == aspect["id"]]
        support_rows.append(
            {
                "aspect_id": aspect["id"],
                "display_name": aspect["display_name"],
                "matched_topic_count": int(len(matched_topics)),
                "matched_review_count": int(matched_topics["review_count"].sum()) if not matched_topics.empty else 0,
                "matched_topic_ids": ",".join(str(topic_id) for topic_id in matched_topics["topic_id"].tolist()),
                "support_status": "topic_supported" if not matched_topics.empty else "not_observed_in_nmf_topics",
            }
        )
    taxonomy_support = pd.DataFrame(support_rows)
    taxonomy_support.to_csv(REPORT_DIR / "aspect_taxonomy_topic_support.csv", index=False, encoding="utf-8")

    assignments = corpus[
        [
            "review_id",
            "canonical_place_id",
            "place_name",
            "place_category",
            "weak_sentiment_label",
            "review_text_clean",
        ]
    ].copy()
    assignments["topic_id"] = topic_assignments
    assignments["topic_strength"] = topic_strength
    assignments.to_parquet(PROCESSED_DIR / "review_topics.parquet", index=False)

    plt.figure(figsize=(10, 6))
    topic_summary.sort_values("topic_id").plot(
        x="topic_id",
        y="review_count",
        kind="bar",
        legend=False,
        ax=plt.gca(),
    )
    plt.title("Distribusi Topic Modeling NMF")
    plt.xlabel("Topic ID")
    plt.ylabel("Jumlah Review")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "topic_distribution.png", dpi=150)
    plt.close()

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "reviews_path": str(reviews_path),
        "corpus_rows": int(len(corpus)),
        "tfidf_shape": [int(tfidf.shape[0]), int(tfidf.shape[1])],
        "n_topics": int(n_topics),
        "config": config,
        "taxonomy_path": str(ASPECT_TAXONOMY_PATH),
        "taxonomy_aspect_count": int(len(aspects)),
        "outputs": {
            "topic_summary": str(REPORT_DIR / "topic_summary.csv"),
            "topic_keywords": str(REPORT_DIR / "topic_keywords.csv"),
            "topic_distribution": str(FIGURE_DIR / "topic_distribution.png"),
            "review_topics": str(PROCESSED_DIR / "review_topics.parquet"),
            "aspect_taxonomy_topic_support": str(REPORT_DIR / "aspect_taxonomy_topic_support.csv"),
        },
        "limitations": [
            "NMF topic modeling is exploratory and does not create final aspect labels.",
            "Mapped initial aspects use direct keyword overlap with the taxonomy and need manual review.",
            "The corpus excludes duplicate rows and empty review text.",
        ],
    }
    (REPORT_DIR / "topic_modeling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU topic modeling with TF-IDF + NMF.")
    parser.parse_args()
    summary = run_topic_modeling()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
