import type {
  AnalyzeResult,
  AuthStatus,
  DataImportSummary,
  DataQuality,
  EvidencePage,
  GeoCollection,
  Health,
  ModelMetrics,
  Paginated,
  PlaceDetail,
  PlaceListItem,
  ServiceGap,
  Summary,
} from "../types/api";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (typeof init?.body === "string" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers,
    credentials: "include",
  });
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith("/auth/")) {
      window.dispatchEvent(new Event("podanauli:unauthorized"));
    }
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.message;
    throw new Error(message ?? `Permintaan gagal (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function queryString(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  });
  const result = search.toString();
  return result ? `?${result}` : "";
}

export const api = {
  health: () => request<Health>("/health"),
  authMe: () => request<AuthStatus>("/auth/me"),
  login: (username: string, password: string) =>
    request<AuthStatus>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  logout: () => request<AuthStatus>("/auth/logout", { method: "POST" }),
  summary: () => request<Summary>("/summary"),
  places: (params: Record<string, string | number | undefined>) =>
    request<Paginated<PlaceListItem>>(`/places${queryString(params)}`),
  place: (placeId: string) => request<PlaceDetail>(`/places/${encodeURIComponent(placeId)}`),
  placeEvidence: (placeId: string, params: Record<string, string | number | undefined>) =>
    request<EvidencePage>(`/places/${encodeURIComponent(placeId)}/evidence${queryString(params)}`),
  serviceGaps: (params: Record<string, string | number | undefined>) =>
    request<Paginated<ServiceGap>>(`/service-gaps${queryString(params)}`),
  clusters: (params: Record<string, string | number | undefined> = {}) =>
    request<GeoCollection>(`/clusters${queryString(params)}`),
  metrics: () => request<ModelMetrics>("/model-metrics"),
  dataQuality: () => request<DataQuality>("/data-quality"),
  analyze: (text: string) =>
    request<AnalyzeResult>("/analyze-review", { method: "POST", body: JSON.stringify({ text }) }),
  imports: () => request<{ items: DataImportSummary[] }>("/imports?limit=20"),
  importDetail: (importId: string) => request<DataImportSummary>(`/imports/${encodeURIComponent(importId)}`),
  importRankings: (importId: string) =>
    request<Paginated<ServiceGap>>(`/imports/${encodeURIComponent(importId)}/service-gaps?limit=100`),
  importGeojson: (importId: string) =>
    request<GeoCollection>(`/imports/${encodeURIComponent(importId)}/geojson`),
  importEvidence: (importId: string, params: Record<string, string | number | undefined>) =>
    request<EvidencePage>(`/imports/${encodeURIComponent(importId)}/evidence${queryString(params)}`),
  uploadReviews: (placeId: string, file: File) =>
    request<DataImportSummary>(`/imports?filename=${encodeURIComponent(file.name)}&place_id=${encodeURIComponent(placeId)}`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": file.type || "application/octet-stream" },
    }),
  publishImport: (importId: string) =>
    request<DataImportSummary>(`/imports/${encodeURIComponent(importId)}/publish`, { method: "POST" }),
  unpublishImport: (importId: string) =>
    request<DataImportSummary>(`/imports/${encodeURIComponent(importId)}/unpublish`, { method: "POST" }),
  importTemplateUrl: `${BASE_URL}/imports/template`,
};

export const formatLabel = (value: string) =>
  value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

export const formatNumber = (value: number) => new Intl.NumberFormat("id-ID").format(value);
export const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;
