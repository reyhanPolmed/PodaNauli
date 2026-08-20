from __future__ import annotations

from fastapi.testclient import TestClient

from api import main as api_main
from api.main import app


def test_health_and_summary_load_real_artifacts() -> None:
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        summary = client.get("/api/v1/summary")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert health.json()["dataset_loaded"] is True
    assert summary.status_code == 200
    assert summary.json()["total_places"] == 380
    assert summary.json()["total_service_gaps"] > 5000


def test_analyze_review_runs_three_champion_models() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze-review",
            json={"text": "Akses jalan rusak dan toilet kurang bersih."},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["sentiment"] in {"negative", "neutral", "positive"}
    assert payload["complaint"] in {"detected", "not_detected", "review_required"}
    assert isinstance(payload["sentiment_scores"], dict)
    assert isinstance(payload["aspects"], list)
    assert payload["model_version"]


def test_analyze_review_rejects_invalid_text() -> None:
    with TestClient(app) as client:
        empty = client.post("/api/v1/analyze-review", json={"text": "  "})
        too_long = client.post("/api/v1/analyze-review", json={"text": "x" * 2001})

    assert empty.status_code == 422
    assert too_long.status_code == 422


def test_rankings_places_detail_and_geojson_are_privacy_safe() -> None:
    with TestClient(app) as client:
        rankings = client.get("/api/v1/service-gaps", params={"min_score": 50, "limit": 5})
        places = client.get("/api/v1/places", params={"category": "hotel", "limit": 5})
        place_id = places.json()["items"][0]["place_id"]
        detail = client.get(f"/api/v1/places/{place_id}")
        evidence = client.get(f"/api/v1/places/{place_id}/evidence", params={"limit": 2})
        geojson = client.get("/api/v1/clusters")

    assert rankings.status_code == 200
    assert len(rankings.json()["items"]) == 5
    assert places.status_code == 200
    assert detail.status_code == 200
    assert "reviewer_id" not in detail.text
    assert "reviewer_name" not in detail.text
    assert evidence.status_code == 200
    assert evidence.json()["total_all"] >= evidence.json()["total"]
    assert len(evidence.json()["items"]) <= 2
    assert isinstance(evidence.json()["aspect_counts"], dict)
    assert "reviewer_id" not in evidence.text
    assert "reviewer_name" not in evidence.text
    assert geojson.status_code == 200
    assert geojson.json()["type"] == "FeatureCollection"
    assert geojson.json()["features"]


def test_metrics_and_data_quality_endpoints() -> None:
    with TestClient(app) as client:
        metrics = client.get("/api/v1/model-metrics")
        quality = client.get("/api/v1/data-quality")

    assert metrics.status_code == 200
    assert metrics.json()["sentiment"]["macro_f1"] > 0
    assert "D:\\" not in metrics.text
    assert "output" not in metrics.json()["service_gap_validation"]
    assert quality.status_code == 200
    assert quality.json()["summary"]["sheet_count"] == 14


def test_fastapi_serves_react_build_and_keeps_api_404(tmp_path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text(
        '<!doctype html><div id="root">PodaNauli SPA</div>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("window.PODANAULI = true;", encoding="utf-8")
    monkeypatch.setattr(api_main, "FRONTEND_DIST", tmp_path)

    with TestClient(app) as client:
        root = client.get("/")
        react_route = client.get("/service-gap-ranking")
        asset = client.get("/assets/app.js")
        missing_api = client.get("/api/v1/tidak-ada")

    assert root.status_code == 200
    assert "PodaNauli SPA" in root.text
    assert react_route.status_code == 200
    assert "PodaNauli SPA" in react_route.text
    assert asset.status_code == 200
    assert "window.PODANAULI" in asset.text
    assert asset.headers["cache-control"] == "public, max-age=604800, immutable"
    assert missing_api.status_code == 404
    assert "PodaNauli SPA" not in missing_api.text
