import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, CircleAlert, MapPin, MessageSquareQuote, Star } from "lucide-react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ErrorState, LoadingState, MetricCard, PageHeader } from "../components/UI";
import { PlaceEvidencePanel } from "../components/PlaceEvidencePanel";
import { api, formatLabel, formatNumber } from "../lib/api";
import { clusterAreaName } from "../lib/clusterAreas";

export function PlacePage() {
  const { placeId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const aspectParam = searchParams.get("aspect")?.trim() ?? "";
  const focusedAspect = /^[a-z0-9_]{1,80}$/i.test(aspectParam) ? aspectParam.toLowerCase() : "";
  const query = useQuery({ queryKey: ["place", placeId], queryFn: () => api.place(placeId), enabled: Boolean(placeId) });

  if (query.isLoading) return <LoadingState label="Memuat profil destinasi" />;
  if (query.error || !query.data) return <ErrorState message={query.error?.message ?? "Destinasi tidak ditemukan."} />;
  const place = query.data;
  const focusedGap = focusedAspect
    ? place.service_gaps.find((item) => item.aspect.toLowerCase() === focusedAspect)
    : place.service_gaps[0];

  function updateFocusedAspect(value: string) {
    const next = new URLSearchParams(searchParams);
    if (value) next.set("aspect", value);
    else next.delete("aspect");
    setSearchParams(next, { replace: true });
  }

  return (
    <>
      <Link to="/service-gap-ranking" className="mb-4 inline-flex items-center gap-2 text-sm font-semibold text-[#475467] hover:text-[#1666D8]"><ArrowLeft size={17} /> Kembali ke prioritas</Link>
      <PageHeader
        title={place.name}
        description={[formatLabel(place.category), place.place_type, place.address].filter(Boolean).join(" / ")}
        action={focusedAspect ? (
          <div className="flex w-fit items-center gap-3 rounded-lg border border-[#B2CCFF] bg-[#F5F9FF] px-3 py-2">
            <span>
              <span className="block text-[9px] font-medium text-[#667085]">Fokus masalah</span>
              <strong className="mt-0.5 block text-xs font-semibold text-[#175CD3]">{formatLabel(focusedAspect)}</strong>
            </span>
            <button type="button" onClick={() => updateFocusedAspect("")} className="cursor-pointer border-l border-[#B2CCFF] pl-3 text-[10px] font-semibold text-[#1666D8] hover:text-[#0B54B8]">Semua aspek</button>
          </div>
        ) : undefined}
      />

      <div className="dashboard-metric-grid gap-3 xl:gap-4">
        <MetricCard label="Rating" value={place.rating?.toFixed(1) ?? "-"} detail="Metadata tempat" icon={Star} tone="amber" />
        <MetricCard label="Ulasan valid" value={formatNumber(place.review_count)} detail="Teks setelah deduplikasi" icon={MessageSquareQuote} />
        <MetricCard label="Kawasan lokasi" value={clusterAreaName(place.cluster_id)} detail={place.latitude !== null ? `${place.latitude.toFixed(4)}, ${place.longitude?.toFixed(4)}` : "Koordinat belum tersedia"} icon={MapPin} tone="blue" compactValue />
        <MetricCard label={focusedAspect ? "Service gap aspek" : "Service gap utama"} value={focusedGap?.score.toFixed(1) ?? "-"} detail={focusedGap ? formatLabel(focusedGap.aspect) : "Belum ada skor untuk aspek ini"} icon={CircleAlert} tone="rose" />
      </div>

      <div className="mt-5 lg:mt-4 2xl:mt-5">
        <PlaceEvidencePanel placeId={place.place_id} serviceGaps={place.service_gaps} metadata={place} initialAspect={focusedAspect} onAspectChange={updateFocusedAspect} />
      </div>
    </>
  );
}
