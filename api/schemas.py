"""Pydantic request and response contracts for the PodaNauli API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: str
    model_loaded: bool
    dataset_loaded: bool
    model_version: dict[str, str]


class LoginRequest(ApiModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class AuthUser(ApiModel):
    username: str
    display_name: str
    role: str


class AuthStatus(ApiModel):
    authenticated: bool
    user: AuthUser | None = None


class MetricItem(ApiModel):
    model: str
    metric: str
    value: float


class NegativeAspect(ApiModel):
    aspect: str
    negative_mentions: int


class SummaryResponse(ApiModel):
    total_reviews: int
    total_places: int
    total_service_gaps: int
    average_rating: float | None
    places_with_coordinates: int
    category_distribution: dict[str, int]
    sentiment_distribution: dict[str, int]
    top_negative_aspects: list[NegativeAspect]
    model_metrics_summary: list[MetricItem]


class PlaceListItem(ApiModel):
    place_id: str
    name: str
    category: str
    place_type: str | None = None
    rating: float | None = None
    review_count: int
    latitude: float | None = None
    longitude: float | None = None
    cluster_id: int | None = None
    top_aspect: str | None = None
    service_gap_score: float | None = None
    confidence: str | None = None


class PaginatedPlaces(ApiModel):
    total: int
    limit: int
    offset: int
    items: list[PlaceListItem]


class ServiceGapItem(ApiModel):
    rank: int
    place_id: str
    place_name: str
    category: str
    aspect: str
    score: float
    confidence: str
    priority: str
    review_count: int
    evidence_count: int
    negative_mentions: int
    negative_rate: float
    data_reliability: float
    reason_codes: list[str]
    explanation: str


class PaginatedServiceGaps(ApiModel):
    total: int
    limit: int
    offset: int
    items: list[ServiceGapItem]


class EvidenceItem(ApiModel):
    text: str
    aspect: str
    complaint_probability: float
    confidence: float
    sentiment_source: str


class PaginatedEvidence(ApiModel):
    total: int
    total_all: int
    limit: int
    offset: int
    aspect_counts: dict[str, int]
    items: list[EvidenceItem]


class PlaceDetail(ApiModel):
    place_id: str
    name: str
    category: str
    place_type: str | None = None
    address: str | None = None
    status: str | None = None
    rating: float | None = None
    review_count: int
    latitude: float | None = None
    longitude: float | None = None
    cluster_id: int | None = None
    facilities: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    sentiment_distribution: dict[str, int]
    top_aspects: list[dict[str, Any]]
    service_gaps: list[ServiceGapItem]
    evidence: list[EvidenceItem]
    limitations: list[str]


class AnalyzeReviewRequest(ApiModel):
    text: str = Field(min_length=3, max_length=2000)


class AspectPrediction(ApiModel):
    label: str
    probability: float


class AnalyzeReviewResponse(ApiModel):
    sentiment: str
    sentiment_scores: dict[str, float]
    complaint: str
    complaint_probability: float
    aspects: list[AspectPrediction]
    warnings: list[str]
    model_version: dict[str, str]


class ImportIssue(ApiModel):
    row: int | None = None
    field: str
    severity: str
    code: str
    message: str


class DataImportSummary(ApiModel):
    import_id: str
    status: str
    filename: str
    created_at: str
    rows_received: int
    rows_accepted: int
    rows_rejected: int
    place_count: int
    review_count: int
    places_with_coordinates: int
    evidence_count: int
    ranking_count: int
    sentiment_distribution: dict[str, int]
    complaint_distribution: dict[str, int]
    top_priorities: list[ServiceGapItem]
    warnings: list[ImportIssue]
    errors: list[ImportIssue]
    model_version: dict[str, str]
    training_performed: bool
    scope: str
    target_place_id: str | None = None
    published: bool = False
    published_at: str | None = None


class DataImportList(ApiModel):
    items: list[DataImportSummary]


class DataQualityResponse(ApiModel):
    summary: dict[str, int | float]
    transformations: list[dict[str, Any]]
    sheets: list[dict[str, Any]]
    limitations: list[str]


class ModelMetricsResponse(ApiModel):
    scope: str
    sentiment: dict[str, Any]
    complaint: dict[str, Any]
    aspect: dict[str, Any]
    service_gap_validation: dict[str, Any]
    limitations: list[str]
