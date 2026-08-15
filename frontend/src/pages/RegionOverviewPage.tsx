import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Filter, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import { ErrorState, LoadingState } from "../components/UI";
import { PriorityRegionMap, statusForScore } from "../components/region-summary/PriorityRegionMap";
import { RegionSummaryCards, type RegionalCondition, type RegionalStatus } from "../components/region-summary/RegionSummaryCards";
import { api, formatLabel } from "../lib/api";

const statusOrder: RegionalStatus[] = ["Baik", "Perlu Perhatian", "Kritis"];
const aspects = [
  "akses_jalan", "transportasi", "parkir", "kebersihan", "toilet", "harga", "pelayanan",
  "makanan", "akomodasi", "keamanan", "jam_operasional", "fasilitas_umum", "pemandangan",
  "keramaian", "aksesibilitas", "budaya",
];

export function RegionOverviewPage() {
  const [selectedId, setSelectedId] = useState<string | null | undefined>(undefined);
  const [aspect, setAspect] = useState("");
  const summary = useQuery({ queryKey: ["summary", "region"], queryFn: api.summary });
  const gaps = useQuery({
    queryKey: ["service-gaps", "region", aspect],
    queryFn: () => api.serviceGaps({ aspect: aspect || undefined, limit: 100 }),
    placeholderData: keepPreviousData,
  });
  const clusters = useQuery({
    queryKey: ["clusters", "region", aspect],
    queryFn: () => api.clusters({ aspect: aspect || undefined }),
    placeholderData: keepPreviousData,
  });

  const topPriorities = useMemo(() => {
    const seen = new Set<string>();
    const visiblePlaceIds = new Set((clusters.data?.features ?? []).map((feature) => String(feature.properties.canonical_place_id ?? "")));
    return (gaps.data?.items ?? []).filter((item) => {
      if (!visiblePlaceIds.has(item.place_id)) return false;
      if (seen.has(item.place_id)) return false;
      seen.add(item.place_id);
      return true;
    }).slice(0, 5);
  }, [clusters.data, gaps.data]);

  const conditions = useMemo<RegionalCondition[]>(() => {
    const features = clusters.data?.features ?? [];
    const counts: Record<RegionalStatus, number> = { Baik: 0, "Perlu Perhatian": 0, Kritis: 0 };
    features.forEach((feature) => counts[statusForScore(Number(feature.properties.service_gap_score ?? 0))] += 1);
    return statusOrder.map((label) => ({
      label,
      count: counts[label],
      percentage: features.length ? Math.round((counts[label] / features.length) * 100) : 0,
    }));
  }, [clusters.data]);

  if (summary.isLoading || gaps.isLoading || clusters.isLoading) return <LoadingState label="Memuat ikhtisar destinasi" />;
  if (summary.error || gaps.error || clusters.error || !summary.data || !clusters.data) {
    return <ErrorState message={(summary.error ?? gaps.error ?? clusters.error)?.message ?? "Ringkasan wilayah tidak tersedia."} />;
  }

  const activeId = selectedId === undefined ? topPriorities[0]?.place_id ?? null : selectedId;
  const isUpdating = gaps.isFetching || clusters.isFetching;
  const visibleDestinationCount = clusters.data.features.length;

  return (
    <div className="min-w-0">
      <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between lg:mb-4 xl:mb-5 2xl:mb-6">
        <div>
          <h1 className="text-[28px] font-bold leading-tight text-[#071A33] lg:text-[26px] xl:text-[28px] 2xl:text-4xl">Ikhtisar Destinasi</h1>
          <p className="mt-2 text-sm text-[#667085] lg:mt-1.5 lg:text-xs xl:text-[11px] 2xl:mt-2 2xl:text-sm">Gambaran umum kinerja layanan wisata di kawasan Danau Toba.</p>
        </div>
        <label className="w-full sm:w-60 lg:w-52 xl:w-60 2xl:w-64">
          <span className="mb-1.5 block text-[10px] font-semibold text-[#475467] 2xl:text-xs">Fokus aspek layanan</span>
          <span className="relative block">
            <Filter aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[#1666D8]" />
            <select
              value={aspect}
              onChange={(event) => {
                setAspect(event.target.value);
                setSelectedId(undefined);
              }}
              className="h-10 w-full rounded-lg border border-[#B2CCFF] bg-white pl-9 pr-3 text-xs font-semibold text-[#101828] outline-none transition-colors hover:border-[#84ADFF] focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] lg:h-9 lg:text-[10px] 2xl:h-10 2xl:text-xs"
            >
              <option value="">Semua aspek layanan</option>
              {aspects.map((item) => <option key={item} value={item}>{formatLabel(item)}</option>)}
            </select>
          </span>
          <span className="mt-1.5 flex min-h-4 items-center justify-end gap-1.5 text-[9px] text-[#667085] 2xl:text-[10px]">
            {isUpdating && <RefreshCw aria-hidden="true" className="size-3 animate-spin" />}
            {isUpdating ? "Memperbarui data" : `${visibleDestinationCount.toLocaleString("id-ID")} destinasi terpetakan`}
          </span>
        </label>
      </header>

      <RegionSummaryCards
        summary={summary.data}
        highestPriority={topPriorities[0]}
        conditions={conditions}
        selectedAspect={aspect}
        visiblePlaceCount={visibleDestinationCount}
        filteredGapCount={gaps.data?.total ?? 0}
      />

      <section aria-busy={isUpdating} className="mt-4 min-w-0 overflow-hidden rounded-xl border border-[#E4E7EC] bg-white shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex items-center justify-between gap-4 border-b border-[#E4E7EC] px-5 py-4 lg:px-4 lg:py-3 xl:px-5 xl:py-3.5 2xl:py-4">
          <div>
            <h2 className="text-base font-semibold text-[#101828] lg:text-sm xl:text-[13px] 2xl:text-base">Peta Sebaran Destinasi Prioritas</h2>
            <p className="mt-1 text-[10px] text-[#667085] 2xl:text-xs">{aspect ? `Prioritas untuk aspek ${formatLabel(aspect)}` : "Prioritas seluruh aspek layanan"}</p>
          </div>
          <span className="shrink-0 rounded-md bg-[#EAF2FC] px-2.5 py-1.5 text-[10px] font-semibold text-[#175CD3] 2xl:text-xs">{visibleDestinationCount.toLocaleString("id-ID")} destinasi</span>
        </div>
        <PriorityRegionMap
          features={clusters.data.features}
          topPriorities={topPriorities}
          conditions={conditions}
          selectedId={activeId}
          onSelect={setSelectedId}
          onClear={() => setSelectedId(null)}
        />
      </section>

      <div className="mt-4 flex items-center gap-2 text-xs text-[#667085]">
        <RefreshCw aria-hidden="true" size={15} strokeWidth={1.7} />
        <span>Terakhir diperbarui: 31 Mei 2024 10:30 WIB</span>
      </div>
    </div>
  );
}
