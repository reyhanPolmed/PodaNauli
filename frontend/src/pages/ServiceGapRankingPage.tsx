import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ErrorState, LoadingState } from "../components/UI";
import { ServiceGapFilters, ServiceGapSearchHeader } from "../components/service-gap/ServiceGapFilters";
import { ServiceGapKpis } from "../components/service-gap/ServiceGapKpis";
import { ServiceGapTable } from "../components/service-gap/ServiceGapTable";
import { defaultRankingFilters, type RankingFilters } from "../components/service-gap/types";
import { api } from "../lib/api";

export function ServiceGapRankingPage() {
  const [searchParams] = useSearchParams();
  const initialSearch = searchParams.get("search") ?? "";
  const [filters, setFilters] = useState<RankingFilters>({
    ...defaultRankingFilters,
    search: initialSearch,
  });
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(8);

  const ranking = useQuery({
    queryKey: ["service-gaps", "new-ranking", filters.clusterId, filters.category, filters.aspect, filters.minReviews, filters.confidence, filters.search, page, pageSize],
    queryFn: () => api.serviceGaps({
      cluster_id: filters.clusterId || undefined,
      category: filters.category || undefined,
      aspect: filters.aspect || undefined,
      min_reviews: filters.minReviews || undefined,
      confidence: filters.confidence || undefined,
      search: filters.search || undefined,
      limit: pageSize,
      offset: page * pageSize,
    }),
    placeholderData: keepPreviousData,
  });
  const clusters = useQuery({ queryKey: ["clusters", "service-gap-filter"], queryFn: () => api.clusters() });

  const clusterIds = useMemo(() => Array.from(new Set(
    (clusters.data?.features ?? [])
      .map((feature) => Number(feature.properties.geo_cluster_id))
      .filter((value) => Number.isInteger(value) && value >= 0),
  )).sort((a, b) => a - b), [clusters.data]);

  function updateFilter<K extends keyof RankingFilters>(key: K, value: RankingFilters[K]) {
    setFilters((current) => ({ ...current, [key]: value }));
    setPage(0);
  }

  function resetFilters() {
    setFilters(defaultRankingFilters);
    setPage(0);
  }

  function applySearch(value: string) {
    updateFilter("search", value.trim());
  }

  if (ranking.isLoading) return <LoadingState label="Memuat prioritas penanganan" />;
  if (ranking.error || !ranking.data) return <ErrorState message={ranking.error?.message ?? "Prioritas penanganan tidak tersedia."} />;

  return (
    <div className="min-w-0">
      <header className="mb-7 flex flex-col gap-5 lg:mb-4 lg:flex-row lg:items-start lg:justify-between lg:gap-3 xl:mb-5 xl:gap-4 2xl:mb-7 2xl:gap-5">
        <div>
          <h1 className="text-[28px] font-bold leading-tight text-[#071A33] lg:text-[26px] xl:text-[28px] 2xl:text-4xl">Prioritas Penanganan</h1>
          <p className="mt-2 text-sm text-[#667085] lg:mt-1.5 lg:max-w-[310px] lg:text-xs xl:max-w-none xl:text-[11px] 2xl:mt-2 2xl:text-sm">Daftar prioritas layanan berdasarkan bukti keluhan dan confidence model.</p>
        </div>
        <div className="w-full lg:w-auto">
          <ServiceGapSearchHeader
            appliedSearch={filters.search}
            isSearching={ranking.isFetching}
            onSearch={applySearch}
          />
        </div>
      </header>

      <ServiceGapFilters filters={filters} clusterIds={clusterIds} onChange={updateFilter} onReset={resetFilters} />

      <div className="mt-4">
        <ServiceGapKpis total={ranking.data.total} items={ranking.data.items} />
      </div>

      <div className="mt-4 min-w-0">
        <ServiceGapTable
          items={ranking.data.items}
          total={ranking.data.total}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={(size) => { setPageSize(size); setPage(0); }}
        />
      </div>

      <div className="mt-4 flex items-start gap-3 rounded-xl bg-[#F8FAFC] px-4 py-3 text-xs leading-5 text-[#475467]">
        <Info aria-hidden="true" className="mt-0.5 shrink-0 text-[#667085]" size={16} />
        <p>Gunakan filter untuk menelusuri prioritas berdasarkan kawasan dan aspek masalah. Filter periode waktu dan status penanganan belum aktif karena kedua atribut tersebut belum tersedia pada data backend.</p>
      </div>
    </div>
  );
}
