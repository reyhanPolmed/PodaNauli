from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


TEST_PASSWORD = "TestStakeholder2026!"


def test_public_dashboard_and_protected_import_access(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PODANAULI_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("PODANAULI_ADMIN_PASSWORD", TEST_PASSWORD)

    with TestClient(app) as client:
        public_summary = client.get("/api/v1/summary")
        signed_out = client.get("/api/v1/auth/me")
        protected = client.get("/api/v1/imports")
        invalid_login = client.post(
            "/api/v1/auth/login",
            json={"username": "stakeholder", "password": "password-salah"},
        )
        valid_login = client.post(
            "/api/v1/auth/login",
            json={"username": "stakeholder", "password": TEST_PASSWORD},
        )
        signed_in = client.get("/api/v1/auth/me")
        imports = client.get("/api/v1/imports")
        logout = client.post("/api/v1/auth/logout")
        protected_after_logout = client.get("/api/v1/imports")

    assert public_summary.status_code == 200
    assert signed_out.status_code == 200
    assert signed_out.json() == {"authenticated": False, "user": None}
    assert protected.status_code == 401
    assert invalid_login.status_code == 401
    assert valid_login.status_code == 200
    assert valid_login.json()["user"] == {
        "username": "stakeholder",
        "display_name": "Stakeholder",
        "role": "admin",
    }
    assert signed_in.json()["authenticated"] is True
    assert imports.status_code == 200
    assert logout.status_code == 200
    assert protected_after_logout.status_code == 401


def test_login_is_disabled_without_configured_password(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PODANAULI_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("PODANAULI_ADMIN_PASSWORD", raising=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "stakeholder", "password": TEST_PASSWORD},
        )

    assert response.status_code == 503
