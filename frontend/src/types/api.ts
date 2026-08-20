export interface Health {
  status: string;
  model_loaded: boolean;
  dataset_loaded: boolean;
  model_version: Record<string, string>;
}

export interface AuthUser {
  username: string;
  display_name: string;
  role: "admin";
}

export interface AuthStatus {
  authenticated: boolean;
  user: AuthUser | null;
}

export interface Summary {
  total_reviews: number;
  total_places: number;
  total_service_gaps: number;
  average_rating: number | null;
  places_with_coordinates: number;
  category_distribution: Record<string, number>;
  sentiment_distribution: Record<string, number>;
  top_negative_aspects: Array<{ aspect: string; negative_mentions: number }>;
  model_metrics_summary: Array<{ model: string; metric: string; value: number }>;
}

export interface PlaceListItem {
  place_id: string;
  name: string;
  category: string;
  place_type: string | null;
  rating: number | null;
  review_count: number;
  latitude: number | null;
  longitude: number | null;
  cluster_id: number | null;
  top_aspect: string | null;
  service_gap_score: number | null;
  confidence: string | null;
}

export interface ServiceGap {
  rank: number;
  place_id: string;
  place_name: string;
  category: string;
  aspect: string;
  score: number;
  confidence: string;
  priority: string;
  review_count: number;
  evidence_count: number;
  negative_mentions: number;
  negative_rate: number;
  data_reliability: number;
  reason_codes: string[];
  explanation: string;
}

export interface Paginated<T> {
  total: number;
  limit: number;
  offset: number;
  items: T[];
}

export interface Evidence {
  text: string;
  aspect: string;
  complaint_probability: number;
  confidence: number;
  sentiment_source: string;
}

export interface EvidencePage extends Paginated<Evidence> {
  total_all: number;
  aspect_counts: Record<string, number>;
}

export interface PlaceDetail {
  place_id: string;
  name: string;
  category: string;
  place_type: string | null;
  address: string | null;
  status: string | null;
  rating: number | null;
  review_count: number;
  latitude: number | null;
  longitude: number | null;
  cluster_id: number | null;
  facilities: string | null;
  min_price: number | null;
  max_price: number | null;
  sentiment_distribution: Record<string, number>;
  top_aspects: Array<{ aspect: string; score: number; evidence_count: number; negative_mentions: number }>;
  service_gaps: ServiceGap[];
  evidence: Evidence[];
  limitations: string[];
}

export interface AnalyzeResult {
  sentiment: string;
  sentiment_scores: Record<string, number>;
  complaint: string;
  complaint_probability: number;
  aspects: Array<{ label: string; probability: number }>;
  warnings: string[];
  model_version: Record<string, string>;
}

export interface ImportIssue {
  row: number | null;
  field: string;
  severity: string;
  code: string;
  message: string;
}

export interface DataImportSummary {
  import_id: string;
  status: string;
  filename: string;
  created_at: string;
  rows_received: number;
  rows_accepted: number;
  rows_rejected: number;
  place_count: number;
  review_count: number;
  places_with_coordinates: number;
  evidence_count: number;
  ranking_count: number;
  sentiment_distribution: Record<string, number>;
  complaint_distribution: Record<string, number>;
  top_priorities: ServiceGap[];
  warnings: ImportIssue[];
  errors: ImportIssue[];
  model_version: Record<string, string>;
  training_performed: boolean;
  scope: string;
  target_place_id: string | null;
  published: boolean;
  published_at: string | null;
}

export interface ModelMetrics {
  scope: string;
  sentiment: Record<string, unknown>;
  complaint: Record<string, unknown>;
  aspect: Record<string, unknown> & { per_aspect?: Record<string, AspectMetric> };
  service_gap_validation: Record<string, unknown>;
  limitations: string[];
}

export interface AspectMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
  average_precision: number;
}

export interface DataQuality {
  summary: Record<string, number>;
  transformations: Array<Record<string, string>>;
  sheets: Array<Record<string, string | number | null>>;
  limitations: string[];
}

export interface GeoFeature {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: Record<string, string | number | null>;
}

export interface GeoCollection {
  type: "FeatureCollection";
  features: GeoFeature[];
}
