import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  MapPin,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";
import { ChangeEvent, DragEvent, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";
import { ErrorState, LoadingState } from "../components/UI";
import { api, formatLabel, formatNumber, formatPercent } from "../lib/api";
import type { DataImportSummary, GeoFeature, PlaceListItem, ServiceGap } from "../types/api";

const priorityColor: Record<string, string> = {
  tinggi: "#F04438",
  menengah: "#F79009",
  rendah: "#12B76A",
};

const priorityBadge: Record<string, string> = {
  tinggi: "bg-red-50 text-[#B42318]",
  menengah: "bg-amber-50 text-[#B54708]",
  rendah: "bg-emerald-50 text-[#027A48]",
};

function Metric({ label, value, helper }: { label: string; value: string; helper: string }) {
  return (
    <div className="min-h-24 border-r border-[#E4E7EC] px-4 py-3 last:border-r-0 lg:min-h-20 lg:px-3 lg:py-2.5 2xl:min-h-24 2xl:px-5 2xl:py-4">
      <p className="text-[10px] font-medium text-[#667085] 2xl:text-xs">{label}</p>
      <p className="mt-1.5 text-xl font-semibold text-[#101828] lg:text-lg 2xl:text-2xl">{value}</p>
      <p className="mt-1 text-[9px] text-[#667085] 2xl:text-[11px]">{helper}</p>
    </div>
  );
}

function ImportMap({ features, selectedId, onSelect }: {
  features: GeoFeature[];
  selectedId: string | null;
  onSelect: (placeId: string) => void;
}) {
  return (
    <div className="relative z-0 isolate h-[360px] w-full overflow-hidden bg-[#EAF2FC] lg:h-[400px] 2xl:h-[460px]">
      <MapContainer center={[2.62, 98.86]} zoom={8} zoomControl scrollWheelZoom className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          opacity={0.72}
        />
        {features.map((feature) => {
          const [longitude, latitude] = feature.geometry.coordinates;
          const placeId = String(feature.properties.canonical_place_id ?? "");
          const score = Number(feature.properties.service_gap_score ?? 0);
          const priority = String(feature.properties.priority ?? "rendah");
          const selected = selectedId === placeId;
          return (
            <CircleMarker
              key={placeId}
              center={[latitude, longitude]}
              radius={selected ? 11 : 8}
              pathOptions={{
                color: "#ffffff",
                weight: selected ? 3 : 1.5,
                fillColor: priorityColor[priority] ?? priorityColor.rendah,
                fillOpacity: 0.94,
              }}
              eventHandlers={{ click: () => onSelect(placeId) }}
            >
              <Tooltip direction="top" offset={[0, -14]} opacity={0.96}>
                <strong>{String(feature.properties.place_name ?? "Destinasi")}</strong><br />
                {formatLabel(String(feature.properties.top_aspect ?? "belum tersedia"))} · {score.toLocaleString("id-ID", { maximumFractionDigits: 2 })}
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
      {!features.length && (
        <div className="pointer-events-none absolute inset-0 z-[400] grid place-items-center p-6">
          <p className="rounded-lg border border-[#E4E7EC] bg-white px-5 py-3 text-xs text-[#667085] shadow-sm">Tidak ada koordinat valid untuk dipetakan.</p>
        </div>
      )}
    </div>
  );
}

function UploadPanel({ destinations, selected, search, onSearch, onSelect, onUploaded }: {
  destinations: PlaceListItem[];
  selected: PlaceListItem | null;
  search: string;
  onSearch: (value: string) => void;
  onSelect: (place: PlaceListItem | null) => void;
  onUploaded: (summary: DataImportSummary) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mutation = useMutation({
    mutationFn: ({ placeId, selectedFile }: { placeId: string; selectedFile: File }) => api.uploadReviews(placeId, selectedFile),
    onSuccess: (summary) => {
      setFile(null);
      setDragging(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
      onUploaded(summary);
    },
  });

  function acceptFile(candidate?: File) {
    if (!candidate) return;
    const suffix = candidate.name.toLowerCase().split(".").pop();
    if (!suffix || !["csv", "xlsx"].includes(suffix)) {
      mutation.reset();
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    setFile(candidate);
    mutation.reset();
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  return (
    <section className="overflow-hidden rounded-xl border border-[#E4E7EC] bg-white">
      <div className="flex items-center justify-between gap-4 border-b border-[#E4E7EC] px-5 py-4 lg:px-4 lg:py-3 2xl:px-6 2xl:py-4">
        <div>
          <h2 className="text-sm font-semibold text-[#101828] 2xl:text-base">Pilih destinasi dan unggah ulasan</h2>
          <p className="mt-1 text-[10px] text-[#667085] 2xl:text-xs">Metadata dan koordinat diambil otomatis dari destinasi terpilih.</p>
        </div>
        <a href={api.importTemplateUrl} download className="inline-flex h-9 items-center gap-2 rounded-lg border border-[#D0D5DD] px-3 text-[10px] font-semibold text-[#344054] hover:bg-[#F8FAFC] 2xl:h-10 2xl:text-xs">
          <Download size={15} /> Template CSV
        </a>
      </div>
      <div className="p-5 lg:p-4 2xl:p-6">
        <div className="mb-4 grid gap-3 lg:grid-cols-2">
          <label>
            <span className="mb-1.5 block text-[10px] font-semibold text-[#475467] 2xl:text-xs">Cari destinasi</span>
            <span className="relative block">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[#667085]" />
              <input value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Ketik nama destinasi..." className="h-10 w-full rounded-lg border border-[#D0D5DD] pl-9 pr-3 text-[10px] text-[#101828] outline-none focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] 2xl:h-11 2xl:text-xs" />
            </span>
          </label>
          <label>
            <span className="mb-1.5 block text-[10px] font-semibold text-[#475467] 2xl:text-xs">Destinasi tujuan</span>
            <select value={selected?.place_id ?? ""} onChange={(event) => onSelect(destinations.find((item) => item.place_id === event.target.value) ?? null)} className="h-10 w-full rounded-lg border border-[#D0D5DD] bg-white px-3 text-[10px] text-[#101828] outline-none focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] 2xl:h-11 2xl:text-xs">
              <option value="">Pilih destinasi</option>
              {destinations.map((item) => <option key={item.place_id} value={item.place_id}>{item.name} · {formatLabel(item.category)}</option>)}
            </select>
          </label>
        </div>
        {selected && <div className="mb-4 flex items-center justify-between gap-3 rounded-lg border border-[#B2CCFF] bg-[#F5F9FF] px-3 py-2.5"><span><strong className="block text-[10px] text-[#101828] 2xl:text-xs">{selected.name}</strong><span className="mt-0.5 block text-[9px] text-[#667085] 2xl:text-[10px]">{formatLabel(selected.category)} · {formatNumber(selected.review_count)} ulasan saat ini</span></span><MapPin className="shrink-0 text-[#1666D8]" size={18} /></div>}
        <div
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
          className={`grid min-h-40 place-items-center rounded-lg border border-dashed px-5 py-6 text-center transition-colors ${dragging ? "border-[#1666D8] bg-[#F5F9FF]" : "border-[#B8C0CC] bg-[#F8FAFC]"}`}
        >
          <div>
            <span className="mx-auto grid size-10 place-items-center rounded-full bg-[#EAF2FC] text-[#1666D8]"><FileSpreadsheet size={19} /></span>
            <p className="mt-3 text-xs font-semibold text-[#101828] 2xl:text-sm">{file ? file.name : "Pilih atau tarik file ke area ini"}</p>
            <p className="mt-1 text-[10px] text-[#667085] 2xl:text-xs">Kolom wajib hanya review_text. Rating dan tanggal bersifat opsional.</p>
            <label htmlFor="dataset-file" className="mt-3 inline-flex h-9 cursor-pointer items-center rounded-lg border border-[#D0D5DD] bg-white px-3 text-[10px] font-semibold text-[#344054] hover:bg-slate-50 2xl:text-xs">Pilih file</label>
            <input ref={fileInputRef} id="dataset-file" type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" className="sr-only" onChange={(event: ChangeEvent<HTMLInputElement>) => acceptFile(event.target.files?.[0])} />
          </div>
        </div>
        {mutation.error && <p role="alert" className="mt-3 flex items-center gap-2 text-[10px] text-[#B42318] 2xl:text-xs"><AlertCircle size={15} /> {mutation.error.message}</p>}
        <div className="mt-4 flex items-center justify-between gap-4">
          <p className="text-[9px] leading-4 text-[#667085] 2xl:text-[11px]">Setelah valid, ulasan langsung ditambahkan ke analisis destinasi. Model tidak dilatih ulang.</p>
          <button
            type="button"
            disabled={!file || !selected || mutation.isPending}
            onClick={() => file && selected && mutation.mutate({ placeId: selected.place_id, selectedFile: file })}
            className="inline-flex h-10 shrink-0 items-center gap-2 rounded-lg bg-[#1666D8] px-4 text-[10px] font-semibold text-white hover:bg-[#0B54B8] disabled:cursor-not-allowed disabled:opacity-45 2xl:text-xs"
          >
            {mutation.isPending ? <RefreshCw className="animate-spin" size={15} /> : <UploadCloud size={16} />}
            {mutation.isPending ? "Menganalisis" : "Proses dan gabungkan"}
          </button>
        </div>
      </div>
    </section>
  );
}

export function DataImportPage() {
  const queryClient = useQueryClient();
  const [activeImportId, setActiveImportId] = useState<string | null>(null);
  const [selectedPlaceId, setSelectedPlaceId] = useState<string | null>(null);
  const [evidenceOffset, setEvidenceOffset] = useState(0);
  const [destinationSearch, setDestinationSearch] = useState("");
  const [selectedDestination, setSelectedDestination] = useState<PlaceListItem | null>(null);
  const deferredDestinationSearch = useDeferredValue(destinationSearch);
  const destinations = useQuery({
    queryKey: ["import-destinations", deferredDestinationSearch],
    queryFn: () => api.places({ search: deferredDestinationSearch || undefined, limit: 50 }),
  });
  const history = useQuery({ queryKey: ["data-imports"], queryFn: api.imports });
  const shownImportId = activeImportId ?? history.data?.items[0]?.import_id ?? null;
  const detail = useQuery({
    queryKey: ["data-import", shownImportId],
    queryFn: () => api.importDetail(shownImportId!),
    enabled: Boolean(shownImportId),
  });
  const rankings = useQuery({
    queryKey: ["data-import-rankings", shownImportId],
    queryFn: () => api.importRankings(shownImportId!),
    enabled: Boolean(shownImportId),
  });
  const geojson = useQuery({
    queryKey: ["data-import-geojson", shownImportId],
    queryFn: () => api.importGeojson(shownImportId!),
    enabled: Boolean(shownImportId),
  });

  const uniquePriorities = useMemo(() => {
    const seen = new Set<string>();
    return (rankings.data?.items ?? []).filter((item) => {
      if (seen.has(item.place_id)) return false;
      seen.add(item.place_id);
      return true;
    });
  }, [rankings.data]);
  const activePlaceId = selectedPlaceId ?? uniquePriorities[0]?.place_id ?? null;
  useEffect(() => setEvidenceOffset(0), [activePlaceId, shownImportId]);
  const evidence = useQuery({
    queryKey: ["data-import-evidence", shownImportId, activePlaceId, evidenceOffset],
    queryFn: () => api.importEvidence(shownImportId!, { place_id: activePlaceId!, limit: 50, offset: evidenceOffset }),
    enabled: Boolean(shownImportId && activePlaceId),
  });
  const unpublish = useMutation({
    mutationFn: api.unpublishImport,
    onSuccess: (summary) => {
      queryClient.setQueryData(["data-import", summary.import_id], summary);
      queryClient.invalidateQueries({ queryKey: ["data-imports"] });
      queryClient.invalidateQueries({ queryKey: ["summary"] });
      queryClient.invalidateQueries({ queryKey: ["service-gaps"] });
      queryClient.invalidateQueries({ queryKey: ["clusters"] });
      queryClient.invalidateQueries({ queryKey: ["place"] });
      queryClient.invalidateQueries({ queryKey: ["import-destinations"] });
    },
  });
  const publish = useMutation({
    mutationFn: api.publishImport,
    onSuccess: (summary) => {
      queryClient.setQueryData(["data-import", summary.import_id], summary);
      queryClient.invalidateQueries({ queryKey: ["data-imports"] });
      queryClient.invalidateQueries({ queryKey: ["summary"] });
      queryClient.invalidateQueries({ queryKey: ["service-gaps"] });
      queryClient.invalidateQueries({ queryKey: ["clusters"] });
      queryClient.invalidateQueries({ queryKey: ["place"] });
      queryClient.invalidateQueries({ queryKey: ["import-destinations"] });
    },
  });

  function confirmUnpublish(importId: string) {
    const confirmed = window.confirm(
      "Keluarkan ulasan pada unggahan ini dari Ikhtisar Destinasi dan perhitungan prioritas? Hasil analisis tetap tersimpan dan dapat digabungkan kembali.",
    );
    if (confirmed) unpublish.mutate(importId);
  }

  function uploaded(summary: DataImportSummary) {
    setActiveImportId(summary.import_id);
    setSelectedPlaceId(summary.top_priorities[0]?.place_id ?? null);
    setDestinationSearch("");
    setSelectedDestination(null);
    queryClient.invalidateQueries({ queryKey: ["data-imports"] });
    queryClient.setQueryData(["data-import", summary.import_id], summary);
    queryClient.invalidateQueries({ queryKey: ["summary"] });
    queryClient.invalidateQueries({ queryKey: ["service-gaps"] });
    queryClient.invalidateQueries({ queryKey: ["clusters"] });
    queryClient.invalidateQueries({ queryKey: ["place"] });
    queryClient.invalidateQueries({ queryKey: ["import-destinations"] });
  }

  const result = detail.data;
  const isResultLoading = Boolean(shownImportId && (detail.isLoading || rankings.isLoading || geojson.isLoading));

  return (
    <div className="min-w-0">
      <header className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between 2xl:mb-7">
        <div>
          <h1 className="text-[28px] font-bold leading-tight text-[#071A33] lg:text-[26px] xl:text-[28px] 2xl:text-4xl">Tambah Ulasan Destinasi</h1>
          <p className="mt-1.5 max-w-3xl text-xs leading-5 text-[#667085] 2xl:text-sm">Pilih destinasi yang sudah terdaftar, lalu unggah teks ulasan untuk memperbarui analisis prioritas.</p>
        </div>
        <span className="inline-flex w-fit items-center gap-2 rounded-md bg-[#ECFDF3] px-2.5 py-1.5 text-[10px] font-semibold text-[#027A48] 2xl:text-xs"><ShieldCheck size={15} /> Terhubung ke Ikhtisar</span>
      </header>

      <div className="mb-5 grid grid-cols-2 gap-2 rounded-xl border border-[#E4E7EC] bg-[#F8FAFC] p-3 lg:grid-cols-4 2xl:gap-3 2xl:p-4">
        {["Pilih destinasi terdaftar", "Validasi dan inferensi ulasan", "Perbarui Ikhtisar dan prioritas", "Gunakan koordinat destinasi"].map((label, index) => (
          <div key={label} className="flex items-center gap-2.5 rounded-lg bg-white px-3 py-2.5 text-[9px] font-medium text-[#344054] 2xl:text-[11px]"><span className="grid size-6 shrink-0 place-items-center rounded-full bg-[#EAF2FC] text-[10px] font-semibold text-[#1666D8]">{index + 1}</span>{label}</div>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px] 2xl:grid-cols-[minmax(0,1fr)_320px]">
        <UploadPanel destinations={destinations.data?.items ?? []} selected={selectedDestination} search={destinationSearch} onSearch={setDestinationSearch} onSelect={setSelectedDestination} onUploaded={uploaded} />
        <section className="rounded-xl border border-[#E4E7EC] bg-white">
          <div className="border-b border-[#E4E7EC] px-4 py-3"><h2 className="text-xs font-semibold text-[#101828] 2xl:text-sm">Riwayat analisis</h2></div>
          <div className="max-h-[330px] divide-y divide-[#E4E7EC] overflow-auto">
            {history.isLoading && <div className="p-4 text-[10px] text-[#667085]">Memuat riwayat...</div>}
            {!history.isLoading && !history.data?.items.length && <div className="p-4 text-[10px] leading-5 text-[#667085]">Belum ada dataset yang pernah diproses.</div>}
            {history.data?.items.map((item) => (
              <button key={item.import_id} type="button" onClick={() => { setActiveImportId(item.import_id); setSelectedPlaceId(null); }} className={`block w-full px-4 py-3 text-left hover:bg-[#F8FAFC] ${shownImportId === item.import_id ? "bg-[#F5F9FF]" : ""}`}>
                <span className="block truncate text-[10px] font-semibold text-[#101828] 2xl:text-xs">{item.filename}</span>
                <span className="mt-1 block text-[9px] text-[#667085] 2xl:text-[10px]">{new Date(item.created_at).toLocaleString("id-ID")} · {item.review_count} ulasan</span>
              </button>
            ))}
          </div>
        </section>
      </div>

      {isResultLoading && <div className="mt-5"><LoadingState label="Memuat hasil analisis unggahan" /></div>}
      {(detail.error || rankings.error || geojson.error) && <div className="mt-5"><ErrorState message={(detail.error ?? rankings.error ?? geojson.error)?.message ?? "Hasil impor tidak tersedia."} /></div>}

      {result && !isResultLoading && (
        <div className="mt-5 space-y-4">
          <section className="overflow-hidden rounded-xl border border-[#E4E7EC] bg-white">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#E4E7EC] px-4 py-3 2xl:px-5">
              <div><h2 className="text-sm font-semibold text-[#101828] 2xl:text-base">Hasil validasi dan pemrosesan</h2><p className="mt-1 text-[9px] text-[#667085] 2xl:text-[11px]">{result.filename} · model tidak dilatih ulang</p></div>
              <div className="flex items-center gap-3">
                <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold 2xl:text-xs ${result.published ? "text-[#027A48]" : "text-[#B54708]"}`}><CheckCircle2 size={15} /> {result.published ? "Sudah masuk Ikhtisar" : "Belum masuk Ikhtisar"}</span>
                {result.published ? (
                  <button type="button" disabled={unpublish.isPending} onClick={() => confirmUnpublish(result.import_id)} className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-[#D0D5DD] px-2.5 text-[9px] font-semibold text-[#475467] hover:bg-[#F8FAFC] disabled:opacity-50 2xl:text-[10px]"><RotateCcw size={13} /> Keluarkan dari Ikhtisar</button>
                ) : (
                  <button type="button" disabled={publish.isPending} onClick={() => publish.mutate(result.import_id)} className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-[#1666D8] px-2.5 text-[9px] font-semibold text-white hover:bg-[#0B54B8] disabled:opacity-50 2xl:text-[10px]"><Database size={13} /> Gabungkan kembali</button>
                )}
              </div>
            </div>
            <div className="grid grid-cols-2 divide-x divide-y divide-[#E4E7EC] sm:grid-cols-3 lg:grid-cols-6 lg:divide-y-0">
              <Metric label="Baris diterima" value={`${formatNumber(result.rows_accepted)}/${formatNumber(result.rows_received)}`} helper={`${formatNumber(result.rows_rejected)} ditolak`} />
              <Metric label="Destinasi" value={formatNumber(result.place_count)} helper="destinasi terpilih" />
              <Metric label="Ulasan" value={formatNumber(result.review_count)} helper="masuk inferensi" />
              <Metric label="Bukti aspek" value={formatNumber(result.evidence_count)} helper="klausa terdeteksi" />
              <Metric label="Titik peta" value={formatNumber(result.places_with_coordinates)} helper="koordinat valid" />
              <Metric label="Baris prioritas" value={formatNumber(result.ranking_count)} helper="destinasi-aspek" />
            </div>
            {(result.warnings.length > 0 || result.errors.length > 0) && (
              <div className="border-t border-[#E4E7EC] px-4 py-3">
                <p className="text-[10px] font-semibold text-[#344054] 2xl:text-xs">Catatan validasi</p>
                <ul className="mt-2 space-y-1.5 text-[9px] text-[#667085] 2xl:text-[11px]">
                  {[...result.errors, ...result.warnings].slice(0, 6).map((issue, index) => <li key={`${issue.code}-${issue.row}-${index}`}>Baris {issue.row ?? "umum"}: {issue.message}</li>)}
                </ul>
              </div>
            )}
          </section>

          <section className="grid overflow-hidden rounded-xl border border-[#E4E7EC] bg-white lg:grid-cols-[minmax(0,1fr)_280px] 2xl:grid-cols-[minmax(0,1fr)_340px]">
            <div className="min-w-0 border-b border-[#E4E7EC] lg:border-b-0 lg:border-r">
              <div className="border-b border-[#E4E7EC] px-4 py-3 2xl:px-5"><h2 className="text-sm font-semibold text-[#101828] 2xl:text-base">Lokasi destinasi terpilih</h2><p className="mt-1 text-[9px] text-[#667085] 2xl:text-[11px]">Koordinat berasal dari metadata destinasi yang sudah tersimpan.</p></div>
              <ImportMap features={geojson.data?.features ?? []} selectedId={activePlaceId} onSelect={setSelectedPlaceId} />
            </div>
            <div className="min-w-0">
              <div className="border-b border-[#E4E7EC] px-4 py-3"><h2 className="text-xs font-semibold text-[#101828] 2xl:text-sm">Prioritas per destinasi</h2></div>
              <div className="divide-y divide-[#E4E7EC]">
                {uniquePriorities.map((item) => (
                  <button key={item.place_id} type="button" onClick={() => setSelectedPlaceId(item.place_id)} className={`w-full px-4 py-3 text-left hover:bg-[#F8FAFC] ${activePlaceId === item.place_id ? "bg-[#F5F9FF]" : ""}`}>
                    <span className="flex items-start justify-between gap-3"><span className="min-w-0"><strong className="block truncate text-[10px] font-semibold text-[#101828] 2xl:text-xs">{item.place_name}</strong><span className="mt-1 block text-[9px] text-[#667085] 2xl:text-[10px]">{formatLabel(item.aspect)} · {item.evidence_count} bukti</span></span><strong className="text-sm" style={{ color: priorityColor[item.priority] ?? priorityColor.rendah }}>{item.score.toLocaleString("id-ID", { maximumFractionDigits: 2 })}</strong></span>
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="overflow-hidden rounded-xl border border-[#E4E7EC] bg-white">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#E4E7EC] px-4 py-3 2xl:px-5">
              <div><h2 className="text-sm font-semibold text-[#101828] 2xl:text-base">Ranking aspek prioritas</h2><p className="mt-1 text-[9px] text-[#667085] 2xl:text-[11px]">Diurutkan dari skor tertinggi dalam dataset unggahan.</p></div>
              <span className="text-[9px] text-[#667085] 2xl:text-[11px]">Skor bukan probabilitas dan tetap memerlukan validasi manusia.</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-[10px] 2xl:text-xs">
                <thead className="bg-[#F8FAFC] text-[#475467]"><tr><th className="px-4 py-3 font-semibold">Peringkat</th><th className="px-4 py-3 font-semibold">Destinasi</th><th className="px-4 py-3 font-semibold">Aspek</th><th className="px-4 py-3 font-semibold">Skor</th><th className="px-4 py-3 font-semibold">Bukti negatif</th><th className="px-4 py-3 font-semibold">Keandalan</th><th className="px-4 py-3 font-semibold">Prioritas</th></tr></thead>
                <tbody className="divide-y divide-[#E4E7EC]">
                  {(rankings.data?.items ?? []).slice(0, 20).map((item: ServiceGap) => (
                    <tr key={`${item.rank}-${item.place_id}-${item.aspect}`} className="cursor-pointer hover:bg-[#F8FAFC]" onClick={() => setSelectedPlaceId(item.place_id)}>
                      <td className="px-4 py-3 font-semibold text-[#475467]">#{item.rank}</td><td className="px-4 py-3 font-medium text-[#101828]">{item.place_name}</td><td className="px-4 py-3 text-[#344054]">{formatLabel(item.aspect)}</td><td className="px-4 py-3 font-semibold text-[#101828]">{item.score.toLocaleString("id-ID", { maximumFractionDigits: 2 })}</td><td className="px-4 py-3 text-[#344054]">{item.negative_mentions}/{item.evidence_count}</td><td className="px-4 py-3 text-[#344054]">{formatPercent(item.data_reliability)}</td><td className="px-4 py-3"><span className={`rounded-md px-2 py-1 font-semibold ${priorityBadge[item.priority] ?? priorityBadge.rendah}`}>{formatLabel(item.priority)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="overflow-hidden rounded-xl border border-[#E4E7EC] bg-white">
            <div className="flex items-center justify-between gap-4 border-b border-[#E4E7EC] px-4 py-3 2xl:px-5">
              <div><h2 className="text-sm font-semibold text-[#101828] 2xl:text-base">Bukti keluhan destinasi terpilih</h2><p className="mt-1 text-[9px] text-[#667085] 2xl:text-[11px]">Semua kutipan berasal dari ulasan pada file unggahan.</p></div>
              <Database size={18} className="text-[#1666D8]" />
            </div>
            {evidence.isLoading && <div className="p-5 text-[10px] text-[#667085]">Memuat bukti...</div>}
            {!evidence.isLoading && !evidence.data?.items.length && <div className="p-5 text-[10px] text-[#667085]">Tidak ada bukti negatif untuk destinasi ini.</div>}
            <div className="divide-y divide-[#E4E7EC]">
              {evidence.data?.items.map((item, index) => (
                <article key={`${item.aspect}-${index}`} className="grid gap-2 px-4 py-3 sm:grid-cols-[120px_minmax(0,1fr)_150px] sm:items-center 2xl:px-5 2xl:py-4">
                  <span className="w-fit rounded-md bg-[#EAF2FC] px-2 py-1 text-[9px] font-semibold text-[#175CD3] 2xl:text-[10px]">{formatLabel(item.aspect)}</span>
                  <p className="text-[10px] leading-5 text-[#344054] 2xl:text-xs">“{item.text}”</p>
                  <div className="text-[9px] text-[#667085] sm:text-right 2xl:text-[10px]">Keluhan {formatPercent(item.complaint_probability)}<br />Confidence {formatPercent(item.confidence)}</div>
                </article>
              ))}
            </div>
            {evidence.data && evidence.data.total > 50 && (
              <div className="flex items-center justify-between border-t border-[#E4E7EC] px-4 py-3 text-[9px] text-[#667085] 2xl:px-5 2xl:text-[10px]">
                <span>Menampilkan {evidenceOffset + 1}-{Math.min(evidenceOffset + 50, evidence.data.total)} dari {evidence.data.total} bukti</span>
                <div className="flex gap-2">
                  <button type="button" disabled={evidenceOffset === 0} onClick={() => setEvidenceOffset((value) => Math.max(0, value - 50))} className="h-8 rounded-lg border border-[#D0D5DD] px-3 font-semibold text-[#344054] disabled:opacity-40">Sebelumnya</button>
                  <button type="button" disabled={evidenceOffset + 50 >= evidence.data.total} onClick={() => setEvidenceOffset((value) => value + 50)} className="h-8 rounded-lg border border-[#D0D5DD] px-3 font-semibold text-[#344054] disabled:opacity-40">Berikutnya</button>
                </div>
              </div>
            )}
          </section>

          <p className="flex items-center gap-2 text-[9px] text-[#667085] 2xl:text-[11px]"><MapPin size={14} /> Koordinat hanya digunakan untuk pemetaan dan konteks kelangkaan layanan, bukan sebagai fitur model teks.</p>
        </div>
      )}
    </div>
  );
}
