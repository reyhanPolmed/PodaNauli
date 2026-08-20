import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, CircleAlert, Search } from "lucide-react";
import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { api, formatLabel, formatNumber, formatPercent } from "../lib/api";
import type { PlaceDetail, ServiceGap } from "../types/api";

const PAGE_SIZE = 10;
const filterLabelClass = "mb-1.5 block text-[10px] font-semibold text-[#475467]";
const filterControlClass = "h-9 w-full rounded-lg border border-[#D0D5DD] bg-white px-3 text-[10px] text-[#101828] outline-none transition-colors hover:border-[#98A2B3] focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] focus-visible:outline-none";
type PlaceMetadata = Pick<PlaceDetail, "status" | "min_price" | "max_price" | "facilities">;

const sourceLabels: Record<string, string> = {
  taxonomy_and_complaint_model: "Aturan dan model keluhan",
  taxonomy_negative_evidence: "Sinyal negatif eksplisit",
  complaint_model: "Model keluhan",
  model: "Model",
};

function sourceLabel(value: string) {
  return sourceLabels[value] ?? formatLabel(value);
}

function EvidenceMetric({ label, value, tone }: { label: string; value: number; tone: "complaint" | "confidence" }) {
  const width = `${Math.max(0, Math.min(100, value * 100))}%`;
  const barColor = tone === "complaint" ? "bg-[#F04438]" : "bg-[#1666D8]";
  return (
    <div>
      <div className="flex items-center justify-between gap-3 text-[10px] text-[#667085] 2xl:text-xs">
        <span>{label}</span>
        <strong className="font-semibold text-[#344054]">{formatPercent(value)}</strong>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[#EAECF0]">
        <span className={`block h-full rounded-full ${barColor}`} style={{ width }} />
      </div>
    </div>
  );
}

export function PlaceEvidencePanel({ placeId, serviceGaps, metadata, initialAspect = "", onAspectChange }: {
  placeId: string;
  serviceGaps: ServiceGap[];
  metadata: PlaceMetadata;
  initialAspect?: string;
  onAspectChange?: (aspect: string) => void;
}) {
  const [aspect, setAspect] = useState(initialAspect);
  const [search, setSearch] = useState("");
  const [minComplaint, setMinComplaint] = useState(0);
  const [minConfidence, setMinConfidence] = useState(0);
  const [sort, setSort] = useState("complaint_desc");
  const [page, setPage] = useState(0);
  const deferredSearch = useDeferredValue(search.trim());

  useEffect(() => {
    setAspect(initialAspect);
    setPage(0);
  }, [initialAspect, placeId]);

  const evidence = useQuery({
    queryKey: ["place-evidence", placeId, aspect, deferredSearch, minComplaint, minConfidence, sort, page],
    queryFn: () => api.placeEvidence(placeId, {
      aspect: aspect || undefined,
      search: deferredSearch || undefined,
      min_complaint_probability: minComplaint || undefined,
      min_confidence: minConfidence || undefined,
      sort,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }),
    enabled: Boolean(placeId),
  });

  const gapByAspect = useMemo(
    () => new Map(serviceGaps.map((item) => [item.aspect, item])),
    [serviceGaps],
  );
  const aspectOptions = useMemo(
    () => Object.entries(evidence.data?.aspect_counts ?? {}).sort(([aspectA, countA], [aspectB, countB]) => {
      const scoreDifference = (gapByAspect.get(aspectB)?.score ?? 0) - (gapByAspect.get(aspectA)?.score ?? 0);
      return scoreDifference || countB - countA || aspectA.localeCompare(aspectB);
    }),
    [evidence.data?.aspect_counts, gapByAspect],
  );

  const total = evidence.data?.total ?? 0;
  const totalAll = evidence.data?.total_all ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const start = total ? page * PAGE_SIZE + 1 : 0;
  const end = Math.min(total, (page + 1) * PAGE_SIZE);

  function updateFilter(callback: () => void) {
    callback();
    setPage(0);
  }

  function updateAspect(value: string) {
    updateFilter(() => setAspect(value));
    onAspectChange?.(value);
  }

  return (
    <section className="min-w-0 overflow-hidden rounded-xl bg-white shadow-[0_2px_10px_rgba(16,24,40,0.08)]">
      <header className="relative flex flex-col gap-3 border-b border-[#E4E7EC] px-5 py-4 sm:flex-row sm:items-start sm:justify-between 2xl:py-5">
        <span aria-hidden="true" className="absolute inset-y-0 left-0 w-1 bg-[#1666D8]" />
        <div className="pl-1">
          <h2 className="text-base font-semibold text-[#101828] 2xl:text-lg">Bukti Keluhan per Aspek</h2>
          <p className="mt-1 text-[10px] leading-5 text-[#667085] 2xl:text-xs">Seluruh klausa yang ditandai negatif oleh sistem, tanpa identitas pemberi ulasan.</p>
        </div>
        <div className="shrink-0 border-l border-[#D0D5DD] pl-4 text-left sm:text-right">
          <strong className="block text-2xl font-semibold leading-none text-[#1666D8]">{formatNumber(totalAll)}</strong>
          <span className="mt-1 block text-[10px] text-[#667085] 2xl:text-xs">total bukti tersedia</span>
        </div>
      </header>

      <div className="border-b border-[#E4E7EC] px-5 py-4">
        <p className="text-[10px] font-semibold text-[#475467] 2xl:text-xs">Metadata tempat</p>
        <dl className="mt-3 grid gap-4 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)_minmax(0,2.2fr)] md:gap-0">
          <div className="min-w-0 md:border-r md:border-[#E4E7EC] md:pr-4">
            <dt className="text-[9px] text-[#667085] 2xl:text-[10px]">Status</dt>
            <dd className="mt-1 break-words text-xs font-semibold text-[#101828]">{metadata.status ?? "Tidak tersedia"}</dd>
          </div>
          <div className="min-w-0 md:border-r md:border-[#E4E7EC] md:px-4">
            <dt className="text-[9px] text-[#667085] 2xl:text-[10px]">Kisaran harga</dt>
            <dd className="mt-1 break-words text-xs font-semibold text-[#101828]">
              {metadata.min_price !== null
                ? `${formatNumber(metadata.min_price)} - ${formatNumber(metadata.max_price ?? metadata.min_price)}`
                : "Tidak tersedia"}
            </dd>
          </div>
          <div className="min-w-0 md:pl-4">
            <dt className="text-[9px] text-[#667085] 2xl:text-[10px]">Fasilitas</dt>
            <dd className="mt-1 break-words text-[10px] leading-5 text-[#344054] 2xl:text-xs">{metadata.facilities ?? "Tidak tersedia"}</dd>
          </div>
        </dl>
      </div>

      <div className="grid grid-cols-3 divide-x divide-[#E4E7EC] border-b border-[#E4E7EC] bg-[#F8FAFC]">
        <div className="px-4 py-3.5"><span className="block text-[10px] text-[#667085]">Aspek tercakup</span><strong className="mt-1 block text-lg font-semibold text-[#071A33]">{formatNumber(aspectOptions.length)}</strong></div>
        <div className="px-4 py-3.5"><span className="block text-[10px] text-[#667085]">Hasil filter</span><strong className="mt-1 block text-lg font-semibold text-[#071A33]">{formatNumber(total)}</strong></div>
        <div className="min-w-0 px-4 py-3.5"><span className="block text-[10px] text-[#667085]">Aspek aktif</span><strong className="mt-1 block truncate text-sm font-semibold text-[#1666D8]">{aspect ? formatLabel(aspect) : "Semua aspek"}</strong></div>
      </div>

      {aspectOptions.length > 0 && (
        <div className="border-b border-[#E4E7EC] px-4 pt-3.5">
          <p className="text-[10px] font-semibold text-[#475467]">Aspek prioritas dengan bukti</p>
          <div className="mt-2 flex gap-2 overflow-x-auto">
            {aspectOptions.slice(0, 6).map(([itemAspect, count]) => {
              const gap = gapByAspect.get(itemAspect);
              const active = aspect === itemAspect;
              return (
                <button
                  key={itemAspect}
                  type="button"
                  onClick={() => updateAspect(active ? "" : itemAspect)}
                  className={`shrink-0 rounded-t-lg border-b-2 px-3 pb-3 pt-2 text-left text-[10px] outline-none transition-colors focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#84ADFF] ${active ? "border-[#1666D8] bg-[#EAF2FC] text-[#1666D8]" : "border-transparent text-[#475467] hover:bg-[#F8FAFC] hover:text-[#101828]"}`}
                >
                  <span className="block font-semibold">{formatLabel(itemAspect)}</span>
                  <span className="mt-0.5 block text-[9px] text-[#667085]">{formatNumber(count)} bukti{gap ? <> &middot; skor {gap.score.toLocaleString("id-ID", { maximumFractionDigits: 1 })}</> : null}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="border-b border-[#E4E7EC] bg-[#FCFCFD] p-4 xl:px-5">
        <div className="mb-3 flex items-center justify-between gap-4">
          <p className="text-xs font-semibold text-[#344054]">Filter bukti</p>
          <p className="text-[10px] text-[#667085]">{formatNumber(total)} hasil ditampilkan</p>
        </div>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-6">
          <label className="min-w-0 lg:col-span-2">
            <span className={filterLabelClass}>Cari isi bukti</span>
            <span className="relative block">
              <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[#667085]" />
              <input value={search} onChange={(event) => updateFilter(() => setSearch(event.target.value))} placeholder="Ketik kata kunci ulasan" className={`${filterControlClass} pl-9 placeholder:text-[#98A2B3]`} />
            </span>
          </label>
          <label className="min-w-0 lg:col-span-2">
            <span className={filterLabelClass}>Aspek keluhan</span>
            <select value={aspect} onChange={(event) => updateAspect(event.target.value)} className={filterControlClass}>
              <option value="">Semua aspek</option>
              {aspectOptions.map(([itemAspect, count]) => <option key={itemAspect} value={itemAspect}>{formatLabel(itemAspect)} ({count})</option>)}
            </select>
          </label>
          <label className="min-w-0 lg:col-span-2">
            <span className={filterLabelClass}>Urutkan bukti</span>
            <select value={sort} onChange={(event) => updateFilter(() => setSort(event.target.value))} className={filterControlClass}>
              <option value="complaint_desc">Keluhan terkuat</option>
              <option value="confidence_desc">Confidence tertinggi</option>
            </select>
          </label>
          <label className="min-w-0 lg:col-span-3">
            <span className={filterLabelClass}>Minimal probabilitas keluhan</span>
            <select value={minComplaint} onChange={(event) => updateFilter(() => setMinComplaint(Number(event.target.value)))} className={filterControlClass}>
              <option value={0}>Semua tingkat probabilitas</option>
              <option value={0.5}>Minimal 50%</option>
              <option value={0.7}>Minimal 70%</option>
              <option value={0.9}>Minimal 90%</option>
            </select>
          </label>
          <label className="min-w-0 lg:col-span-3">
            <span className={filterLabelClass}>Minimal confidence</span>
            <select value={minConfidence} onChange={(event) => updateFilter(() => setMinConfidence(Number(event.target.value)))} className={filterControlClass}>
              <option value={0}>Semua tingkat confidence</option>
              <option value={0.5}>Minimal 50%</option>
              <option value={0.7}>Minimal 70%</option>
              <option value={0.9}>Minimal 90%</option>
            </select>
          </label>
        </div>
      </div>

      <div className="flex items-start gap-2.5 border-b border-[#FEC84B] bg-[#FFFAEB] px-4 py-3 text-[10px] leading-5 text-[#7A2E0E]">
        <CircleAlert aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-[#F79009]" />
        <p><strong className="font-semibold">Perlu verifikasi manusia.</strong> Bukti ini merupakan hasil deteksi model atau aturan negatif dan digunakan sebagai dasar pemeriksaan, bukan keputusan final.</p>
      </div>

      {evidence.isLoading ? (
        <div className="grid min-h-48 place-items-center text-sm text-[#667085]">Memuat bukti keluhan...</div>
      ) : evidence.error ? (
        <div className="grid min-h-48 place-items-center px-6 text-center text-sm text-[#B42318]">{evidence.error.message}</div>
      ) : evidence.data?.items.length ? (
        <ol className="divide-y divide-[#E4E7EC]">
          {evidence.data.items.map((item, index) => {
            const gap = gapByAspect.get(item.aspect);
            return (
              <li key={`${item.aspect}-${item.text}-${index}`} className="grid gap-4 px-5 py-5 transition-colors hover:bg-[#F8FAFC] lg:grid-cols-[36px_minmax(0,1fr)_220px] lg:items-start">
                <span className="grid size-7 place-items-center rounded-full border border-[#B2CCFF] bg-[#EFF4FF] text-[10px] font-semibold text-[#175CD3]">{page * PAGE_SIZE + index + 1}</span>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[10px] text-[#667085]">
                    <span className="inline-flex rounded-md bg-[#EAF2FC] px-2 py-1 font-semibold text-[#175CD3]">{formatLabel(item.aspect)}</span>
                    {gap && <span>Skor service gap: <strong className="font-semibold text-[#344054]">{gap.score.toLocaleString("id-ID", { maximumFractionDigits: 2 })}</strong></span>}
                    <span>Sumber: <strong className="font-semibold text-[#344054]">{sourceLabel(item.sentiment_source)}</strong></span>
                  </div>
                  <blockquote className="mt-3 border-l-2 border-[#B2CCFF] pl-3 text-sm leading-6 text-[#101828]">&ldquo;{item.text}&rdquo;</blockquote>
                </div>
                <div className="grid gap-3 border-t border-[#EAECF0] pt-4 sm:grid-cols-2 lg:grid-cols-1 lg:border-l lg:border-t-0 lg:pl-5 lg:pt-0">
                  <p className="hidden text-[10px] font-semibold text-[#475467] lg:block">Indikator model</p>
                  <EvidenceMetric label="Probabilitas keluhan" value={item.complaint_probability} tone="complaint" />
                  <EvidenceMetric label="Confidence" value={item.confidence} tone="confidence" />
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="grid min-h-48 place-items-center px-6 text-center text-sm text-[#667085]">Tidak ada bukti yang sesuai dengan filter.</div>
      )}

      <footer className="flex flex-col gap-3 border-t border-[#E4E7EC] bg-[#FCFCFD] px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-[10px] text-[#667085]">Menampilkan {formatNumber(start)}-{formatNumber(end)} dari {formatNumber(total)} hasil filter</p>
        <div className="flex items-center gap-2">
          <button type="button" title="Halaman sebelumnya" aria-label="Halaman bukti sebelumnya" disabled={page === 0} onClick={() => setPage((value) => value - 1)} className="grid size-8 place-items-center rounded-lg border border-[#D0D5DD] bg-white text-[#475467] outline-none transition-colors hover:border-[#98A2B3] hover:bg-[#F8FAFC] focus-visible:ring-2 focus-visible:ring-[#84ADFF] disabled:opacity-40"><ChevronLeft size={15} /></button>
          <span className="min-w-20 text-center text-[10px] font-medium text-[#344054]">Halaman {page + 1} dari {pageCount}</span>
          <button type="button" title="Halaman berikutnya" aria-label="Halaman bukti berikutnya" disabled={page + 1 >= pageCount} onClick={() => setPage((value) => value + 1)} className="grid size-8 place-items-center rounded-lg border border-[#D0D5DD] bg-white text-[#475467] outline-none transition-colors hover:border-[#98A2B3] hover:bg-[#F8FAFC] focus-visible:ring-2 focus-visible:ring-[#84ADFF] disabled:opacity-40"><ChevronRight size={15} /></button>
        </div>
      </footer>
    </section>
  );
}
