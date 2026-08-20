"""FastAPI entrypoint for the PodaNauli decision-support dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from api.schemas import (
    AnalyzeReviewRequest,
    AnalyzeReviewResponse,
    AuthStatus,
    DataQualityResponse,
    DataImportList,
    DataImportSummary,
    HealthResponse,
    LoginRequest,
    ModelMetricsResponse,
    PaginatedEvidence,
    PaginatedPlaces,
    PaginatedServiceGaps,
    PlaceDetail,
    SummaryResponse,
)
from api.auth import (
    SESSION_COOKIE,
    AuthenticatedUser,
    AuthenticationError,
    AuthenticationUnavailableError,
    AuthService,
    LoginRateLimitError,
)
from api.import_service import BatchImportService, ImportDataError
from api.services import ArtifactStore


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
FRONTEND_DIST = Path(
    os.getenv("PODANAULI_FRONTEND_DIST", str(ROOT / "frontend" / "dist"))
).resolve()
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("PODANAULI_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ArtifactStore(ROOT)
    app.state.auth = AuthService(app.state.store.imports.runtime_dir)
    yield


app = FastAPI(
    title="PodaNauli API",
    version="1.0.0",
    description=(
        "API pendukung keputusan pariwisata untuk ranking service gap, analisis tempat, "
        "data geospasial, metrik model, kualitas data, dan inferensi ulasan. "
        "Output model harus tetap diperiksa manusia."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


def get_store(request: Request) -> ArtifactStore:
    return request.app.state.store


Store = Annotated[ArtifactStore, Depends(get_store)]


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth


Auth = Annotated[AuthService, Depends(get_auth_service)]


def current_user(request: Request, auth: Auth) -> AuthenticatedUser | None:
    return auth.authenticate(request.cookies.get(SESSION_COOKIE))


CurrentUser = Annotated[AuthenticatedUser | None, Depends(current_user)]


def require_admin(user: CurrentUser) -> AuthenticatedUser:
    if user is None:
        raise HTTPException(status_code=401, detail="Login stakeholder diperlukan.")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Akses admin diperlukan.")
    return user


AdminUser = Annotated[AuthenticatedUser, Depends(require_admin)]


@app.get("/api/v1", include_in_schema=False)
def api_info() -> dict[str, str]:
    return {"name": "PodaNauli API", "docs": "/docs", "health": "/api/v1/health"}


@app.get("/api/v1/auth/me", response_model=AuthStatus, tags=["Authentication"])
def auth_me(user: CurrentUser) -> dict:
    return {
        "authenticated": user is not None,
        "user": user.payload() if user else None,
    }


@app.post("/api/v1/auth/login", response_model=AuthStatus, tags=["Authentication"])
def auth_login(payload: LoginRequest, request: Request, response: Response, auth: Auth) -> dict:
    client_address = request.client.host if request.client else "unknown"
    try:
        user, token = auth.login(
            payload.username,
            payload.password,
            client_address=client_address,
            user_agent=request.headers.get("user-agent", ""),
        )
    except AuthenticationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LoginRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=auth.session_hours * 60 * 60,
        httponly=True,
        secure=auth.cookie_secure,
        samesite="lax",
        path="/",
    )
    return {"authenticated": True, "user": user.payload()}


@app.post("/api/v1/auth/logout", response_model=AuthStatus, tags=["Authentication"])
def auth_logout(
    request: Request,
    response: Response,
    auth: Auth,
    user: CurrentUser,
) -> dict:
    auth.logout(request.cookies.get(SESSION_COOKIE), user)
    response.delete_cookie(
        key=SESSION_COOKIE,
        path="/",
        secure=auth.cookie_secure,
        httponly=True,
        samesite="lax",
    )
    return {"authenticated": False, "user": None}


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health(store: Store) -> dict:
    return store.health()


@app.get("/api/v1/summary", response_model=SummaryResponse, tags=["Dashboard"])
def summary(store: Store) -> dict:
    return store.summary()


@app.get("/api/v1/places", response_model=PaginatedPlaces, tags=["Places"])
def places(
    store: Store,
    category: str | None = None,
    aspect: str | None = None,
    min_gap_score: float = Query(default=0, ge=0, le=100),
    cluster_id: int | None = None,
    search: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    return store.list_places(
        category=category,
        aspect=aspect,
        min_gap_score=min_gap_score,
        cluster_id=cluster_id,
        search=search,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/places/{place_id}", response_model=PlaceDetail, tags=["Places"])
def place_detail(place_id: str, store: Store) -> dict:
    result = store.place_detail(place_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Tempat tidak ditemukan.")
    return result


@app.get("/api/v1/places/{place_id}/evidence", response_model=PaginatedEvidence, tags=["Places"])
def place_evidence(
    place_id: str,
    store: Store,
    aspect: str | None = Query(default=None, max_length=80),
    search: str | None = Query(default=None, max_length=100),
    min_complaint_probability: float = Query(default=0, ge=0, le=1),
    min_confidence: float = Query(default=0, ge=0, le=1),
    sort: Literal["complaint_desc", "confidence_desc"] = "complaint_desc",
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    result = store.list_place_evidence(
        place_id,
        aspect=aspect,
        search=search,
        min_complaint_probability=min_complaint_probability,
        min_confidence=min_confidence,
        sort=sort,
        limit=limit,
        offset=offset,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Tempat tidak ditemukan.")
    return result


@app.get("/api/v1/service-gaps", response_model=PaginatedServiceGaps, tags=["Service gaps"])
def service_gaps(
    store: Store,
    aspect: str | None = None,
    min_score: float = Query(default=0, ge=0, le=100),
    category: str | None = None,
    cluster_id: int | None = None,
    search: str | None = Query(default=None, max_length=100),
    confidence: str | None = Query(default=None, max_length=30),
    min_reviews: int | None = Query(default=None, ge=0),
    sort: Literal["score_desc", "score_asc"] = "score_desc",
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    return store.list_service_gaps(
        aspect=aspect,
        min_score=min_score,
        category=category,
        cluster_id=cluster_id,
        search=search,
        confidence=confidence,
        min_reviews=min_reviews,
        sort=sort,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/clusters", tags=["Map"])
def clusters(store: Store, aspect: str | None = None, cluster_id: int | None = None) -> dict:
    return store.clusters_geojson(aspect=aspect, cluster_id=cluster_id)


@app.get("/api/v1/model-metrics", response_model=ModelMetricsResponse, tags=["Evaluation"])
def model_metrics(store: Store) -> dict:
    return store.model_metrics()


@app.get("/api/v1/data-quality", response_model=DataQualityResponse, tags=["Evaluation"])
def data_quality(store: Store) -> dict:
    return store.data_quality_payload()


@app.post("/api/v1/analyze-review", response_model=AnalyzeReviewResponse, tags=["Inference"])
def analyze_review(payload: AnalyzeReviewRequest, store: Store) -> dict:
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Teks ulasan tidak boleh kosong.")
    return store.analyze_review(payload.text)


@app.get("/api/v1/imports/template", tags=["Data imports"])
def import_template(_: AdminUser) -> Response:
    return Response(
        content=BatchImportService.template_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="podanauli_review_template.csv"'},
    )


@app.get("/api/v1/imports", response_model=DataImportList, tags=["Data imports"])
def data_imports(
    store: Store,
    _: AdminUser,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    return {"items": store.imports.list_imports(limit=limit)}


@app.post("/api/v1/imports", response_model=DataImportSummary, status_code=201, tags=["Data imports"])
async def create_data_import(
    request: Request,
    store: Store,
    auth: Auth,
    admin: AdminUser,
    filename: str = Query(min_length=5, max_length=180),
    place_id: str = Query(min_length=1, max_length=100),
) -> dict:
    content = await request.body()
    try:
        matching = store.places.loc[store.places["canonical_place_id"].astype(str) == place_id]
        if matching.empty:
            raise HTTPException(status_code=404, detail="Destinasi yang dipilih tidak ditemukan.")
        existing_texts = set(
            store.reviews.loc[
                store.reviews["canonical_place_id"].astype(str) == place_id,
                "review_text_clean",
            ].dropna().astype(str)
        )
        result = store.imports.process_for_place(
            place=matching.iloc[0],
            filename=filename,
            content=content,
            existing_review_texts=existing_texts,
        )
        summary = store.publish_import(result["import_id"])
        auth.audit(
            "import_created",
            admin,
            target=f"{result['import_id']}:{place_id}",
            client_address=request.client.host if request.client else "",
        )
        return summary
    except ImportDataError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "issues": exc.issues[:100]},
        ) from exc


@app.get("/api/v1/imports/{import_id}", response_model=DataImportSummary, tags=["Data imports"])
def data_import_detail(import_id: str, store: Store, _: AdminUser) -> dict:
    try:
        result = store.imports.get_summary(import_id)
    except ImportDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Hasil impor tidak ditemukan.")
    return result


@app.post(
    "/api/v1/imports/{import_id}/publish",
    response_model=DataImportSummary,
    tags=["Data imports"],
)
def publish_data_import(
    import_id: str,
    request: Request,
    store: Store,
    auth: Auth,
    admin: AdminUser,
) -> dict:
    try:
        summary = store.publish_import(import_id)
        auth.audit(
            "import_published",
            admin,
            target=import_id,
            client_address=request.client.host if request.client else "",
        )
        return summary
    except ImportDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/v1/imports/{import_id}/unpublish",
    response_model=DataImportSummary,
    tags=["Data imports"],
)
def unpublish_data_import(
    import_id: str,
    request: Request,
    store: Store,
    auth: Auth,
    admin: AdminUser,
) -> dict:
    try:
        summary = store.unpublish_import(import_id)
        auth.audit(
            "import_unpublished",
            admin,
            target=import_id,
            client_address=request.client.host if request.client else "",
        )
        return summary
    except ImportDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/api/v1/imports/{import_id}/service-gaps",
    response_model=PaginatedServiceGaps,
    tags=["Data imports"],
)
def data_import_service_gaps(
    import_id: str,
    store: Store,
    _: AdminUser,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    try:
        result = store.imports.get_rankings(import_id, limit=limit, offset=offset)
    except ImportDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Hasil impor tidak ditemukan.")
    return result


@app.get("/api/v1/imports/{import_id}/geojson", tags=["Data imports"])
def data_import_geojson(import_id: str, store: Store, _: AdminUser) -> dict:
    try:
        result = store.imports.get_geojson(import_id)
    except ImportDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Hasil impor tidak ditemukan.")
    return result


@app.get(
    "/api/v1/imports/{import_id}/evidence",
    response_model=PaginatedEvidence,
    tags=["Data imports"],
)
def data_import_evidence(
    import_id: str,
    store: Store,
    _: AdminUser,
    place_id: str | None = Query(default=None, max_length=100),
    aspect: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    try:
        result = store.imports.get_evidence(
            import_id,
            place_id=place_id,
            aspect=aspect,
            limit=limit,
            offset=offset,
        )
    except ImportDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Hasil impor tidak ditemukan.")
    return result


def serve_frontend(frontend_path: str = "") -> Response:
    """Serve the Vite build and return index.html for client-side React routes."""
    if frontend_path == "api" or frontend_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Endpoint API tidak ditemukan.")

    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        if not frontend_path:
            return JSONResponse(
                {
                    "name": "PodaNauli API",
                    "docs": "/docs",
                    "health": "/api/v1/health",
                    "frontend": "belum dibangun; jalankan npm --prefix frontend run build",
                }
            )
        raise HTTPException(
            status_code=503,
            detail="Frontend belum dibangun. Jalankan npm --prefix frontend run build.",
        )

    if frontend_path:
        requested_file = (FRONTEND_DIST / frontend_path).resolve()
        if requested_file.is_relative_to(FRONTEND_DIST) and requested_file.is_file():
            cache_control = (
                "public, max-age=604800, immutable"
                if frontend_path.startswith("assets/")
                else "no-cache"
            )
            return FileResponse(requested_file, headers={"Cache-Control": cache_control})

    return FileResponse(index_file, headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False, response_model=None)
def frontend_root() -> Response:
    return serve_frontend()


@app.get("/{frontend_path:path}", include_in_schema=False, response_model=None)
def frontend_spa(frontend_path: str) -> Response:
    return serve_frontend(frontend_path)
