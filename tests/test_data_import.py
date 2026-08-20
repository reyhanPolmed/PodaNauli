from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


ROOT = Path(__file__).resolve().parents[1]
DUMMY_REVIEWS = ROOT / "data" / "samples" / "dummy_new_reviews.csv"
TEST_PASSWORD = "TestStakeholder2026!"


def login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "stakeholder", "password": TEST_PASSWORD},
    )
    assert response.status_code == 200, response.text


def test_import_rejects_missing_required_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PODANAULI_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("PODANAULI_ADMIN_PASSWORD", TEST_PASSWORD)
    invalid = b"place_name,place_category\nContoh,wisata\n"

    with TestClient(app) as client:
        login(client)
        place_id = client.get("/api/v1/places", params={"limit": 1}).json()["items"][0]["place_id"]
        response = client.post(
            "/api/v1/imports",
            params={"filename": "invalid.csv", "place_id": place_id},
            content=invalid,
            headers={"Content-Type": "text/csv"},
        )

    assert response.status_code == 422
    assert "review_text" in response.text


def test_selected_destination_reviews_publish_to_main_dashboard(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PODANAULI_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("PODANAULI_ADMIN_PASSWORD", TEST_PASSWORD)

    with TestClient(app) as client:
        login(client)
        before_summary = client.get("/api/v1/summary").json()
        place = client.get("/api/v1/places", params={"limit": 1}).json()["items"][0]
        before_detail = client.get(f"/api/v1/places/{place['place_id']}").json()
        response = client.post(
            "/api/v1/imports",
            params={"filename": DUMMY_REVIEWS.name, "place_id": place["place_id"]},
            content=DUMMY_REVIEWS.read_bytes(),
            headers={"Content-Type": "text/csv"},
        )

        assert response.status_code == 201, response.text
        payload = response.json()
        after_summary = client.get("/api/v1/summary").json()
        after_detail = client.get(f"/api/v1/places/{place['place_id']}").json()
        rankings = client.get(
            "/api/v1/service-gaps",
            params={"search": place["name"], "limit": 100},
        ).json()
        geojson = client.get("/api/v1/clusters").json()

        assert payload["target_place_id"] == place["place_id"]
        assert payload["published"] is True
        assert payload["training_performed"] is False
        assert payload["rows_accepted"] == 6
        assert after_summary["total_reviews"] == before_summary["total_reviews"] + 6
        assert after_detail["review_count"] == before_detail["review_count"] + 6
        assert rankings["items"]
        assert any(
            str(feature["properties"].get("canonical_place_id")) == place["place_id"]
            for feature in geojson["features"]
        )

    with TestClient(app) as restarted_client:
        login(restarted_client)
        persisted_summary = restarted_client.get("/api/v1/summary").json()
        unpublish = restarted_client.post(f"/api/v1/imports/{payload['import_id']}/unpublish")
        restored_summary = restarted_client.get("/api/v1/summary").json()
        republish = restarted_client.post(f"/api/v1/imports/{payload['import_id']}/publish")
        republished_summary = restarted_client.get("/api/v1/summary").json()

    assert persisted_summary["total_reviews"] == before_summary["total_reviews"] + 6
    assert unpublish.status_code == 200
    assert unpublish.json()["published"] is False
    assert restored_summary["total_reviews"] == before_summary["total_reviews"]
    assert republish.status_code == 200
    assert republish.json()["published"] is True
    assert republished_summary["total_reviews"] == before_summary["total_reviews"] + 6
