"""FastAPI entrypoint for the PodaNauli decision-support dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AnalyzeReviewRequest,
    AnalyzeReviewResponse,
    DataQualityResponse,
    HealthResponse,
    ModelMetricsResponse,
    PaginatedEvidence,
    PaginatedPlaces,
    PaginatedServiceGaps,
    PlaceDetail,
    SummaryResponse,
)
from api.services import ArtifactStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("PODANAULI_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ArtifactStore(ROOT)
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
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def get_store(request: Request) -> ArtifactStore:
    return request.app.state.store


Store = Annotated[ArtifactStore, Depends(get_store)]


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"name": "PodaNauli API", "docs": "/docs", "health": "/api/v1/health"}


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
