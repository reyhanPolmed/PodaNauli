import type {
  AnalyzeResult,
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
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Permintaan gagal (${response.status})`);
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
};

export const formatLabel = (value: string) =>
  value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

export const formatNumber = (value: number) => new Intl.NumberFormat("id-ID").format(value);
export const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;
