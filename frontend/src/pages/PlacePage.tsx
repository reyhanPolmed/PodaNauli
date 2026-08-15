import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, CircleAlert, MapPin, MessageSquareQuote, Star } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { ErrorState, LoadingState, MetricCard, PageHeader } from "../components/UI";
import { PlaceEvidencePanel } from "../components/PlaceEvidencePanel";
import { api, formatLabel, formatNumber } from "../lib/api";

export function PlacePage() {
  const { placeId = "" } = useParams();
  const query = useQuery({ queryKey: ["place", placeId], queryFn: () => api.place(placeId), enabled: Boolean(placeId) });

  if (query.isLoading) return <LoadingState label="Memuat profil destinasi" />;
  if (query.error || !query.data) return <ErrorState message={query.error?.message ?? "Destinasi tidak ditemukan."} />;
  const place = query.data;

  return (
    <>
      <Link to="/service-gap-ranking" className="mb-4 inline-flex items-center gap-2 text-sm font-semibold text-[#475467] hover:text-[#1666D8]"><ArrowLeft size={17} /> Kembali ke prioritas</Link>
      <PageHeader title={place.name} description={[formatLabel(place.category), place.place_type, place.address].filter(Boolean).join(" · ")} />

      <div className="dashboard-metric-grid gap-3 xl:gap-4">
        <MetricCard label="Rating" value={place.rating?.toFixed(1) ?? "-"} detail="Metadata tempat" icon={Star} tone="amber" />
        <MetricCard label="Ulasan valid" value={formatNumber(place.review_count)} detail="Teks setelah deduplikasi" icon={MessageSquareQuote} />
        <MetricCard label="Cluster wilayah" value={place.cluster_id === null ? "-" : `#${place.cluster_id}`} detail={place.latitude !== null ? `${place.latitude.toFixed(4)}, ${place.longitude?.toFixed(4)}` : "Koordinat belum tersedia"} icon={MapPin} tone="blue" />
        <MetricCard label="Service gap utama" value={place.service_gaps[0]?.score.toFixed(1) ?? "-"} detail={place.service_gaps[0] ? formatLabel(place.service_gaps[0].aspect) : "Belum ada skor"} icon={CircleAlert} tone="rose" />
      </div>

      <div className="mt-5 lg:mt-4 2xl:mt-5">
        <PlaceEvidencePanel placeId={place.place_id} serviceGaps={place.service_gaps} metadata={place} />
      </div>
    </>
  );
}
