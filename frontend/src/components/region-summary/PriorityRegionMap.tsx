import { ArrowRight, LocateFixed, Minus, Plus, X } from "lucide-react";
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap } from "react-leaflet";
import { Link } from "react-router-dom";
import { formatLabel } from "../../lib/api";
import type { GeoFeature, ServiceGap } from "../../types/api";
import type { RegionalCondition, RegionalStatus } from "./RegionSummaryCards";

const statusColor: Record<RegionalStatus, string> = {
  Kritis: "#F04438",
  "Perlu Perhatian": "#F79009",
  Baik: "#12B76A",
};

const statusBadge: Record<RegionalStatus, string> = {
  Kritis: "bg-red-50 text-[#D92D20]",
  "Perlu Perhatian": "bg-amber-50 text-[#B54708]",
  Baik: "bg-emerald-50 text-[#027A48]",
};

export function statusForScore(score: number): RegionalStatus {
  if (score >= 60) return "Kritis";
  if (score >= 40) return "Perlu Perhatian";
  return "Baik";
}

function MapControls() {
  const map = useMap();
  return (
    <div className="absolute bottom-4 right-4 z-[500] flex flex-col overflow-hidden rounded-lg border border-[#D0D5DD] bg-white shadow-[0_2px_8px_rgba(16,24,40,0.12)] lg:bottom-3 lg:right-3 xl:bottom-4 xl:right-4">
      <button type="button" title="Perbesar peta" aria-label="Perbesar peta" onClick={() => map.zoomIn()} className="grid size-10 place-items-center border-b border-[#E4E7EC] text-[#344054] hover:bg-[#F8FAFC] lg:size-8 lg:[&>svg]:size-4 xl:size-9 2xl:size-10 2xl:[&>svg]:size-[19px]"><Plus size={19} /></button>
      <button type="button" title="Perkecil peta" aria-label="Perkecil peta" onClick={() => map.zoomOut()} className="grid size-10 place-items-center border-b border-[#E4E7EC] text-[#344054] hover:bg-[#F8FAFC] lg:size-8 lg:[&>svg]:size-4 xl:size-9 2xl:size-10 2xl:[&>svg]:size-[19px]"><Minus size={19} /></button>
      <button type="button" title="Kembali ke Danau Toba" aria-label="Kembali ke Danau Toba" onClick={() => map.setView([2.62, 98.86], 9)} className="grid size-10 place-items-center text-[#344054] hover:bg-[#F8FAFC] lg:size-8 lg:[&>svg]:size-4 xl:size-9 2xl:size-10 2xl:[&>svg]:size-[18px]"><LocateFixed size={18} /></button>
    </div>
  );
}

function Legend({ conditions }: { conditions: RegionalCondition[] }) {
  return (
    <div className="rounded-xl border border-[#E4E7EC] bg-white p-4 shadow-[0_2px_8px_rgba(16,24,40,0.08)] lg:rounded-lg lg:p-3 xl:p-3.5 2xl:rounded-xl 2xl:p-4">
      <p className="text-xs font-semibold text-[#101828] lg:text-[10px] xl:text-[11px] 2xl:text-xs">Legenda Prioritas</p>
      <div className="mt-3 space-y-2.5 lg:mt-2 lg:space-y-1.5 xl:space-y-2 2xl:mt-3 2xl:space-y-2.5">
        {[...conditions].reverse().map((condition) => <div key={condition.label} className="flex items-center gap-2 text-xs text-[#344054] lg:gap-1.5 lg:text-[9px] xl:text-[10px] 2xl:gap-2 2xl:text-xs"><span className="size-2.5 shrink-0 rounded-full lg:size-2 2xl:size-2.5" style={{ backgroundColor: statusColor[condition.label] }} /><span>{condition.label}</span></div>)}
      </div>
    </div>
  );
}

function TopPriorityList({ items, onSelect }: { items: ServiceGap[]; onSelect: (placeId: string) => void }) {
  return (
    <div className="rounded-xl border border-[#E4E7EC] bg-white p-5 shadow-[0_2px_8px_rgba(16,24,40,0.08)] lg:rounded-lg lg:p-3 xl:p-4 2xl:rounded-xl 2xl:p-5">
      <p className="text-sm font-semibold text-[#101828] lg:text-[11px] xl:text-xs 2xl:text-sm">Top 5 Destinasi Prioritas</p>
      {items.length ? (
        <ol className="mt-4 space-y-3 lg:mt-2.5 lg:space-y-2 xl:mt-3 xl:space-y-2.5 2xl:mt-4 2xl:space-y-3">
          {items.map((item, index) => (
            <li key={item.place_id}>
              <button type="button" onClick={() => onSelect(item.place_id)} className="grid w-full grid-cols-[18px_minmax(0,1fr)_auto] items-center gap-2 text-left text-xs lg:grid-cols-[14px_minmax(0,1fr)_auto] lg:gap-1.5 lg:text-[9px] xl:grid-cols-[16px_minmax(0,1fr)_auto] xl:text-[10px] 2xl:grid-cols-[18px_minmax(0,1fr)_auto] 2xl:gap-2 2xl:text-xs">
                <span className="font-semibold text-[#344054]">{index + 1}</span>
                <span className="truncate text-[#344054]">{item.place_name}</span>
                <strong style={{ color: statusColor[statusForScore(item.score)] }} className="font-semibold">{item.score.toLocaleString("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong>
              </button>
            </li>
          ))}
        </ol>
      ) : (
        <p className="mt-3 text-xs leading-5 text-[#667085] lg:text-[9px] xl:text-[10px] 2xl:text-xs">Belum ada destinasi yang sesuai dengan aspek ini.</p>
      )}
      <Link to="/service-gap-ranking" className="mt-5 inline-flex items-center gap-2 text-xs font-semibold text-[#1666D8] hover:text-[#0B54B8] lg:mt-3 lg:gap-1 lg:text-[9px] xl:mt-4 xl:text-[10px] 2xl:mt-5 2xl:gap-2 2xl:text-xs">Lihat selengkapnya <ArrowRight className="lg:size-3 2xl:size-[14px]" size={14} /></Link>
    </div>
  );
}

function SelectedDestination({ feature, gap, onClose }: {
  feature?: GeoFeature;
  gap?: ServiceGap;
  onClose: () => void;
}) {
  if (!feature) return null;
  const props = feature.properties;
  const placeId = String(props.canonical_place_id ?? gap?.place_id ?? "");
  const name = String(props.place_name ?? gap?.place_name ?? "Destinasi");
  const score = Number(props.service_gap_score ?? gap?.score ?? 0);
  const status = statusForScore(score);
  const aspect = String(props.top_aspect ?? gap?.aspect ?? "Belum tersedia");

  return (
    <div className="rounded-xl border border-[#E4E7EC] bg-white p-5 shadow-[0_6px_20px_rgba(16,24,40,0.12)] lg:rounded-lg lg:p-3 xl:p-4 2xl:rounded-xl 2xl:p-5">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-[#101828] lg:text-[10px] xl:text-xs 2xl:text-sm">{name}</p>
          <span className={`mt-2 inline-flex rounded-md px-2 py-1 text-[11px] font-semibold lg:mt-1.5 lg:px-1.5 lg:py-0.5 lg:text-[9px] xl:text-[10px] 2xl:mt-2 2xl:px-2 2xl:py-1 2xl:text-[11px] ${statusBadge[status]}`}>{status}</span>
        </div>
        <button type="button" title="Tutup detail" aria-label="Tutup detail destinasi" onClick={onClose} className="grid size-8 shrink-0 place-items-center rounded-lg text-[#667085] hover:bg-[#F8FAFC] lg:size-6 lg:[&>svg]:size-3.5 xl:size-7 2xl:size-8 2xl:[&>svg]:size-[17px]"><X size={17} /></button>
      </div>
      <dl className="mt-5 space-y-4 lg:mt-3 lg:space-y-2.5 xl:mt-4 xl:space-y-3 2xl:mt-5 2xl:space-y-4">
        <div><dt className="text-xs text-[#667085] lg:text-[9px] xl:text-[10px] 2xl:text-xs">Skor Prioritas</dt><dd className="mt-1 text-xl font-semibold lg:text-base xl:text-lg 2xl:text-xl" style={{ color: statusColor[status] }}>{score.toLocaleString("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</dd></div>
        <div><dt className="text-xs text-[#667085] lg:text-[9px] xl:text-[10px] 2xl:text-xs">Aspek Terbanyak Bermasalah</dt><dd className="mt-1 text-sm font-medium text-[#344054] lg:text-[10px] xl:text-xs 2xl:text-sm">{formatLabel(aspect)}</dd></div>
        <div className="grid grid-cols-2 gap-3 text-xs lg:gap-2 lg:text-[9px] xl:text-[10px] 2xl:gap-3 2xl:text-xs"><div><dt className="text-[#667085]">Bukti</dt><dd className="mt-1 font-semibold text-[#101828]">{String(props.evidence_count ?? gap?.evidence_count ?? 0)}</dd></div><div><dt className="text-[#667085]">Confidence</dt><dd className="mt-1 truncate font-semibold text-[#101828]">{formatLabel(String(props.confidence ?? gap?.confidence ?? "-"))}</dd></div></div>
      </dl>
      {placeId && <Link to={`/destinasi/${placeId}`} className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-[#1666D8] hover:text-[#0B54B8] lg:mt-3 lg:gap-1 lg:text-[10px] xl:mt-4 xl:text-xs 2xl:mt-5 2xl:gap-2 2xl:text-sm">Lihat Detail <ArrowRight className="lg:size-3 2xl:size-[15px]" size={15} /></Link>}
    </div>
  );
}

export function PriorityRegionMap({ features, topPriorities, conditions, selectedId, onSelect, onClear }: {
  features: GeoFeature[];
  topPriorities: ServiceGap[];
  conditions: RegionalCondition[];
  selectedId: string | null;
  onSelect: (placeId: string) => void;
  onClear: () => void;
}) {
  const selectedFeature = features.find((feature) => String(feature.properties.canonical_place_id) === selectedId);
  const selectedGap = topPriorities.find((item) => item.place_id === selectedId);

  return (
    <>
      <div className="relative z-0 isolate h-[480px] w-full overflow-hidden md:h-[500px] lg:h-[500px] xl:h-[540px] 2xl:h-[580px]">
        <MapContainer center={[2.62, 98.86]} zoom={9} zoomControl={false} scrollWheelZoom className="h-full w-full">
          <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" opacity={0.72} />
          {features.map((feature) => {
            const [longitude, latitude] = feature.geometry.coordinates;
            const props = feature.properties;
            const placeId = String(props.canonical_place_id ?? "");
            const score = Number(props.service_gap_score ?? 0);
            const status = statusForScore(score);
            const selected = selectedId === placeId;
            return (
              <CircleMarker
                key={placeId}
                center={[latitude, longitude]}
                radius={Math.max(5, Math.min(selected ? 13 : 11, (selected ? 6 : 4) + score / 12))}
                pathOptions={{
                  color: "#ffffff",
                  weight: selected ? 3 : 1.5,
                  fillColor: statusColor[status],
                  fillOpacity: selected ? 1 : 0.88,
                }}
                eventHandlers={{ click: () => onSelect(placeId) }}
              >
                <Tooltip direction="top" offset={[0, -24]} opacity={0.96}><strong>{String(props.place_name ?? "Destinasi")}</strong><br /><span>{status} · Skor {score.toLocaleString("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span></Tooltip>
              </CircleMarker>
            );
          })}
          <MapControls />
        </MapContainer>

        {!features.length && (
          <div className="pointer-events-none absolute inset-0 z-[450] grid place-items-center p-6">
            <div className="max-w-xs rounded-lg border border-[#E4E7EC] bg-white px-5 py-4 text-center shadow-[0_4px_16px_rgba(16,24,40,0.12)]">
              <p className="text-sm font-semibold text-[#101828]">Destinasi belum tersedia</p>
              <p className="mt-1 text-xs leading-5 text-[#667085]">Tidak ada destinasi terpetakan untuk aspek yang dipilih.</p>
            </div>
          </div>
        )}

        <div className="pointer-events-none absolute inset-0 z-[400] hidden lg:block">
          <div className="pointer-events-auto absolute right-3 top-3 w-36 xl:right-4 xl:top-4 xl:w-40 2xl:w-44"><Legend conditions={conditions} /></div>
          <div className="pointer-events-auto absolute bottom-4 left-4 w-[220px] xl:bottom-5 xl:left-5 xl:w-[255px] 2xl:w-[285px]"><TopPriorityList items={topPriorities} onSelect={onSelect} /></div>
          {selectedFeature && <div className="pointer-events-auto absolute right-12 top-1/2 w-[220px] -translate-y-1/2 xl:right-16 xl:w-[270px] 2xl:right-20 2xl:w-[300px]"><SelectedDestination feature={selectedFeature} gap={selectedGap} onClose={onClear} /></div>}
        </div>
      </div>

      <div className="grid gap-4 border-t border-[#E4E7EC] p-4 lg:hidden">
        <Legend conditions={conditions} />
        <TopPriorityList items={topPriorities} onSelect={onSelect} />
        {selectedFeature && <SelectedDestination feature={selectedFeature} gap={selectedGap} onClose={onClear} />}
      </div>
    </>
  );
}
