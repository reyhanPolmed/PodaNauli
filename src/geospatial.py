"""Tahap 7 pipeline: geospatial clustering for TobaPulse places."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"
MAP_DIR = ROOT / "outputs" / "maps"


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def valid_coordinate_mask(places: pd.DataFrame) -> pd.Series:
    """Return rows with valid latitude and longitude ranges."""
    return (
        places["latitude"].notna()
        & places["longitude"].notna()
        & places["latitude"].between(-90, 90)
        & places["longitude"].between(-180, 180)
    )


def run_dbscan_haversine(
    coordinates: pd.DataFrame,
    eps_km: float,
    min_samples: int,
    earth_radius_km: float = 6371.0088,
) -> np.ndarray:
    """Cluster latitude/longitude coordinates with DBSCAN Haversine."""
    radians = np.radians(coordinates[["latitude", "longitude"]].to_numpy(dtype=float))
    eps_radians = eps_km / earth_radius_km
    model = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
    return model.fit_predict(radians)


def _dominant_negative_aspects_for_place(aspect_predictions: pd.DataFrame | None) -> dict[str, list[str]]:
    if aspect_predictions is None or aspect_predictions.empty:
        return {}
    if "weak_negative_aspects_text" not in aspect_predictions.columns:
        return {}
    result: dict[str, list[str]] = {}
    for place_id, group in aspect_predictions.groupby("canonical_place_id"):
        counts: dict[str, int] = {}
        for text in group["weak_negative_aspects_text"].dropna():
            for aspect in str(text).split("|"):
                aspect = aspect.strip()
                if aspect:
                    counts[aspect] = counts.get(aspect, 0) + 1
        result[str(place_id)] = [aspect for aspect, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]]
    return result


def summarize_clusters(clustered_places: pd.DataFrame, aspect_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create cluster-level summaries including center and dominant negative aspects."""
    aspect_lookup = _dominant_negative_aspects_for_place(aspect_predictions)
    rows = []
    for cluster_id, group in clustered_places.groupby("geo_cluster_id"):
        cluster_label = int(cluster_id)
        aspect_counts: dict[str, int] = {}
        for place_id in group["canonical_place_id"]:
            for aspect in aspect_lookup.get(str(place_id), []):
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
        dominant_aspects = [aspect for aspect, _count in sorted(aspect_counts.items(), key=lambda item: (-item[1], item[0]))[:5]]
        rows.append(
            {
                "geo_cluster_id": cluster_label,
                "cluster_center_latitude": float(group["latitude"].mean()),
                "cluster_center_longitude": float(group["longitude"].mean()),
                "cluster_size": int(len(group)),
                "dominant_negative_aspects": dominant_aspects,
                "dominant_negative_aspects_text": "|".join(dominant_aspects),
                "place_category_counts": json.dumps(group["place_category"].value_counts(dropna=False).astype(int).to_dict(), ensure_ascii=False),
                "average_gap_score": None,
                "is_noise": bool(cluster_label == -1),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_noise", "geo_cluster_id"]).reset_index(drop=True)


def build_place_cluster_rows(places: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels to places with valid coordinates."""
    clustered = places.copy().reset_index(drop=True)
    clustered["geo_cluster_id"] = labels.astype(int)
    return clustered[
        [
            "canonical_place_id",
            "canonical_place_name",
            "place_category",
            "place_type",
            "address",
            "latitude",
            "longitude",
            "geo_cluster_id",
        ]
    ]


def clusters_to_geojson(place_clusters: pd.DataFrame, cluster_summary: pd.DataFrame) -> dict[str, Any]:
    """Export places and cluster centers as GeoJSON FeatureCollection."""
    summary_lookup = cluster_summary.set_index("geo_cluster_id").to_dict(orient="index")
    features = []
    for _, row in place_clusters.iterrows():
        cluster_id = int(row["geo_cluster_id"])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": {
                    "feature_type": "place",
                    "canonical_place_id": row["canonical_place_id"],
                    "place_name": row["canonical_place_name"],
                    "place_category": row["place_category"],
                    "geo_cluster_id": cluster_id,
                    "is_noise": bool(cluster_id == -1),
                },
            }
        )
    for _, row in cluster_summary.iterrows():
        cluster_id = int(row["geo_cluster_id"])
        if cluster_id == -1:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["cluster_center_longitude"]), float(row["cluster_center_latitude"])],
                },
                "properties": {
                    "feature_type": "cluster_center",
                    "geo_cluster_id": cluster_id,
                    "cluster_size": int(row["cluster_size"]),
                    "dominant_negative_aspects": summary_lookup[cluster_id]["dominant_negative_aspects"],
                    "average_gap_score": row["average_gap_score"],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def run_geospatial_clustering() -> dict[str, Any]:
    config = load_config()
    geo_config = config.get("geospatial", {})
    eps_km = float(geo_config.get("eps_km", 20.0))
    min_samples = int(geo_config.get("min_samples", 4))
    earth_radius_km = float(geo_config.get("earth_radius_km", 6371.0088))
    min_valid_places = int(geo_config.get("min_valid_places", 10))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    places_path = PROCESSED_DIR / "places_master.parquet"
    if not places_path.exists():
        raise FileNotFoundError(f"Missing places master: {places_path}")
    places = pd.read_parquet(places_path)
    valid_mask = valid_coordinate_mask(places)
    valid_places = places.loc[valid_mask].copy().reset_index(drop=True)

    if len(valid_places) < min_valid_places:
        empty_clusters = pd.DataFrame(
            columns=[
                "canonical_place_id",
                "canonical_place_name",
                "place_category",
                "place_type",
                "address",
                "latitude",
                "longitude",
                "geo_cluster_id",
            ]
        )
        empty_clusters.to_parquet(PROCESSED_DIR / "place_clusters.parquet", index=False)
        summary = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "skipped",
            "reason": "not_enough_valid_coordinates",
            "valid_coordinate_places": int(len(valid_places)),
            "min_valid_places": min_valid_places,
        }
        (REPORT_DIR / "geospatial_clustering_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    labels = run_dbscan_haversine(
        valid_places[["latitude", "longitude"]],
        eps_km=eps_km,
        min_samples=min_samples,
        earth_radius_km=earth_radius_km,
    )
    place_clusters = build_place_cluster_rows(valid_places, labels)

    aspect_path = PROCESSED_DIR / "review_aspect_predictions.parquet"
    aspect_predictions = pd.read_parquet(aspect_path) if aspect_path.exists() else None
    cluster_summary = summarize_clusters(place_clusters, aspect_predictions)

    cluster_summary.to_csv(REPORT_DIR / "geospatial_cluster_summary.csv", index=False, encoding="utf-8")
    place_clusters_export = place_clusters.merge(
        cluster_summary[
            [
                "geo_cluster_id",
                "cluster_center_latitude",
                "cluster_center_longitude",
                "cluster_size",
                "dominant_negative_aspects_text",
                "average_gap_score",
                "is_noise",
            ]
        ],
        on="geo_cluster_id",
        how="left",
    )
    place_clusters_export.to_parquet(PROCESSED_DIR / "place_clusters.parquet", index=False)
    geojson = clusters_to_geojson(place_clusters, cluster_summary)
    (MAP_DIR / "place_clusters.geojson").write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")

    non_noise = cluster_summary[~cluster_summary["is_noise"]]
    noise_count = int((place_clusters["geo_cluster_id"] == -1).sum())
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "completed",
        "places_path": str(places_path),
        "total_places": int(len(places)),
        "valid_coordinate_places": int(len(valid_places)),
        "invalid_or_missing_coordinate_places": int((~valid_mask).sum()),
        "algorithm": "DBSCAN(metric=haversine)",
        "config": {
            "eps_km": eps_km,
            "min_samples": min_samples,
            "earth_radius_km": earth_radius_km,
            "min_valid_places": min_valid_places,
        },
        "cluster_count_excluding_noise": int(len(non_noise)),
        "noise_place_count": noise_count,
        "largest_cluster_size": int(non_noise["cluster_size"].max()) if not non_noise.empty else 0,
        "outputs": {
            "place_clusters": str(PROCESSED_DIR / "place_clusters.parquet"),
            "cluster_summary": str(REPORT_DIR / "geospatial_cluster_summary.csv"),
            "place_clusters_geojson": str(MAP_DIR / "place_clusters.geojson"),
        },
        "limitations": [
            "Clusters are based only on available valid coordinates.",
            "DBSCAN parameters are prototype configuration values and should be tuned with stakeholder context.",
            "average_gap_score is left null until Service Gap Score is produced in the next stage.",
        ],
    }
    (REPORT_DIR / "geospatial_clustering_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DBSCAN Haversine geospatial clustering.")
    parser.parse_args()
    summary = run_geospatial_clustering()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
