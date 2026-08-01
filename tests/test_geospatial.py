import json

import pandas as pd

from src.geospatial import clusters_to_geojson, run_dbscan_haversine, valid_coordinate_mask


def test_valid_coordinate_mask_rejects_invalid_ranges():
    places = pd.DataFrame(
        {
            "latitude": [2.0, 91.0, None],
            "longitude": [99.0, 99.0, 99.0],
        }
    )
    assert valid_coordinate_mask(places).tolist() == [True, False, False]


def test_run_dbscan_haversine_clusters_near_points():
    coordinates = pd.DataFrame(
        {
            "latitude": [2.0, 2.001, 2.002, 3.0],
            "longitude": [99.0, 99.001, 99.002, 98.0],
        }
    )
    labels = run_dbscan_haversine(coordinates, eps_km=1.0, min_samples=2)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == -1


def test_clusters_to_geojson_outputs_feature_collection():
    place_clusters = pd.DataFrame(
        [
            {
                "canonical_place_id": "p1",
                "canonical_place_name": "Place",
                "place_category": "wisata",
                "place_type": "Wisata Alam",
                "address": "Addr",
                "latitude": 2.0,
                "longitude": 99.0,
                "geo_cluster_id": 0,
            }
        ]
    )
    cluster_summary = pd.DataFrame(
        [
            {
                "geo_cluster_id": 0,
                "cluster_center_latitude": 2.0,
                "cluster_center_longitude": 99.0,
                "cluster_size": 1,
                "dominant_negative_aspects": ["akses_jalan"],
                "average_gap_score": None,
                "is_noise": False,
            }
        ]
    )
    geojson = clusters_to_geojson(place_clusters, cluster_summary)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    json.dumps(geojson)
