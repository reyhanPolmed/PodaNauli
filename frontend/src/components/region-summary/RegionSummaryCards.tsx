import { Info } from "lucide-react";
import type { ServiceGap, Summary } from "../../types/api";
import { formatLabel, formatNumber } from "../../lib/api";

export type RegionalStatus = "Kritis" | "Perlu Perhatian" | "Baik";

export interface RegionalCondition {
  label: RegionalStatus;
  count: number;
  percentage: number;
}

const statusColor: Record<RegionalStatus, string> = {
  Kritis: "bg-[#F04438]",
  "Perlu Perhatian": "bg-[#F79009]",
  Baik: "bg-[#12B76A]",
};

function MetricCard({ label, value, detail, detailTone = "muted", compactValue = false }: {
  label: string;
  value: string;
  detail: string;
  detailTone?: "muted" | "brand";
  compactValue?: boolean;
}) {
  return (
    <article className="flex h-[150px] min-w-0 flex-col overflow-hidden rounded-xl border border-[#E4E7EC] bg-white p-4 shadow-[0_1px_2px_rgba(16,24,40,0.04)] lg:h-32 lg:rounded-lg lg:p-3 xl:h-32 xl:rounded-xl xl:p-4 2xl:h-[150px] 2xl:p-5">
      <p className="line-clamp-2 text-[13px] font-semibold leading-snug text-[#344054] lg:text-[10px] xl:text-[10px] 2xl:text-[13px]">{label}</p>
      <p title={value} className={`mt-auto line-clamp-2 pt-3 font-semibold leading-tight text-[#071A33] lg:pt-2 xl:pt-2.5 2xl:pt-3 ${compactValue ? "text-xl lg:text-sm xl:text-base 2xl:text-xl" : "text-[28px] lg:text-[22px] xl:text-[22px] 2xl:text-[28px]"}`}>{value}</p>
      <p className={`mt-2 line-clamp-2 text-xs leading-4 lg:mt-1.5 lg:text-[10px] lg:leading-[14px] xl:text-[10px] xl:leading-4 2xl:mt-2 2xl:text-xs ${detailTone === "brand" ? "font-medium text-[#1666D8]" : "text-[#667085]"}`}>{detail}</p>
    </article>
  );
}

export function RegionSummaryCards({ summary, highestPriority, conditions, selectedAspect = "", visiblePlaceCount = 0, filteredGapCount = 0 }: {
  summary: Summary;
  highestPriority?: ServiceGap;
  conditions: RegionalCondition[];
  selectedAspect?: string;
  visiblePlaceCount?: number;
  filteredGapCount?: number;
}) {
  const topAspect = summary.top_negative_aspects[0];
  const negativeTotal = summary.top_negative_aspects.reduce((total, item) => total + item.negative_mentions, 0);
  const topAspectShare = topAspect && negativeTotal ? Math.round((topAspect.negative_mentions / negativeTotal) * 100) : 0;
  const hasAspectFilter = Boolean(selectedAspect);
  const selectedAspectLabel = selectedAspect ? formatLabel(selectedAspect) : "";

  return (
    <section aria-label="Ringkasan metrik wilayah" className="region-metric-grid gap-3 lg:gap-2 xl:gap-3 2xl:gap-4">
      <MetricCard
        label="Jumlah Destinasi"
        value={formatNumber(hasAspectFilter ? visiblePlaceCount : summary.total_places)}
        detail={hasAspectFilter ? `Aspek dominan: ${selectedAspectLabel}` : `${formatNumber(summary.places_with_coordinates)} memiliki koordinat valid`}
      />
      <MetricCard
        label="Jumlah Masukan"
        value={formatNumber(summary.total_reviews)}
        detail="Total ulasan valid setelah deduplikasi"
        detailTone="brand"
      />
      <MetricCard
        label="Jumlah Masalah"
        value={formatNumber(hasAspectFilter ? filteredGapCount : summary.total_service_gaps)}
        detail={hasAspectFilter ? `Gap layanan untuk ${selectedAspectLabel}` : "Kombinasi tempat dan aspek terindikasi"}
        detailTone="brand"
      />
      <MetricCard
        label="Prioritas Tertinggi"
        value={highestPriority?.place_name ?? "Belum tersedia"}
        detail={highestPriority ? `Skor ${highestPriority.score.toLocaleString("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "Belum ada prioritas"}
        compactValue
      />
      <MetricCard
        label="Aspek Terbanyak Bermasalah"
        value={hasAspectFilter ? selectedAspectLabel : topAspect ? formatLabel(topAspect.aspect) : "Belum tersedia"}
        detail={hasAspectFilter ? `${formatNumber(visiblePlaceCount)} destinasi terpetakan` : `${topAspectShare}% dari enam aspek teratas`}
        compactValue
      />
      <article className="h-[150px] min-w-0 overflow-hidden rounded-xl border border-[#E4E7EC] bg-white p-4 shadow-[0_1px_2px_rgba(16,24,40,0.04)] lg:h-32 lg:rounded-lg lg:p-3 xl:h-32 xl:rounded-xl xl:p-4 2xl:h-[150px] 2xl:p-5">
        <div className="flex items-center gap-2">
          <p className="line-clamp-2 text-xs font-semibold leading-4 text-[#344054] lg:text-[10px] lg:leading-[14px] xl:text-[10px] xl:leading-4 2xl:text-xs">Ringkasan Kondisi Wilayah</p>
          <button type="button" title="Status dihitung dari skor prioritas setiap destinasi" aria-label="Informasi perhitungan status wilayah" className="text-[#98A2B3] hover:text-[#1666D8]">
            <Info size={14} />
          </button>
        </div>
        <div className="mt-3 space-y-2 lg:mt-2 lg:space-y-1.5 xl:mt-2.5 xl:space-y-2 2xl:mt-3">
          {conditions.map((condition) => (
            <div key={condition.label} className="grid grid-cols-[8px_minmax(0,1fr)_auto] items-center gap-1.5 text-[11px] lg:grid-cols-[6px_minmax(0,1fr)_auto] lg:gap-1 lg:text-[9px] xl:grid-cols-[7px_minmax(0,1fr)_auto] xl:text-[9px] 2xl:grid-cols-[8px_minmax(0,1fr)_auto] 2xl:gap-1.5 2xl:text-[11px]">
              <span className={`size-2 rounded-full ${statusColor[condition.label]}`} />
              <span className="truncate text-[#475467]" title={condition.label}>{condition.label}</span>
              <strong className="whitespace-nowrap font-semibold text-[#101828]">{condition.percentage}% <span className="font-normal text-[#98A2B3]">({condition.count})</span></strong>
            </div>
          ))}
        </div>
      </article>
    </section>
  );
}
