export interface RankingFilters {
  search: string;
  clusterId: string;
  category: string;
  aspect: string;
  minReviews: string;
  confidence: string;
  period: string;
  handlingStatus: string;
}

export const defaultRankingFilters: RankingFilters = {
  search: "",
  clusterId: "",
  category: "",
  aspect: "",
  minReviews: "",
  confidence: "",
  period: "all",
  handlingStatus: "",
};
