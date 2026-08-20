import { ChevronLeft, ChevronRight } from "lucide-react";
import { Link } from "react-router-dom";
import { formatLabel, formatNumber } from "../../lib/api";
import type { ServiceGap } from "../../types/api";

const confidenceColor: Record<string, string> = {
  high: "bg-[#12B76A]",
  medium: "bg-[#F79009]",
  low: "bg-[#98A2B3]",
};

function rankTone(rank: number) {
  if (rank === 1) return "bg-red-50 text-[#D92D20]";
  if (rank <= 3) return "bg-amber-50 text-[#B54708]";
  return "bg-[#F2F4F7] text-[#475467]";
}

function scoreTone(score: number) {
  if (score >= 60) return "text-[#F04438]";
  if (score >= 40) return "text-[#F79009]";
  return "text-[#475467]";
}

function detailPath(item: ServiceGap) {
  return `/destinasi/${encodeURIComponent(item.place_id)}?aspect=${encodeURIComponent(item.aspect)}`;
}

export function ServiceGapTable({ items, total, page, pageSize, onPageChange, onPageSizeChange }: {
  items: ServiceGap[];
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const start = total ? page * pageSize + 1 : 0;
  const end = Math.min(total, (page + 1) * pageSize);
  const visiblePages = Array.from(new Set([0, page - 1, page, page + 1, pageCount - 1].filter((value) => value >= 0 && value < pageCount)));

  return (
    <section className="min-w-0 overflow-hidden rounded-xl border border-[#E4E7EC] bg-white shadow-[0_1px_2px_rgba(16,24,40,0.04)] lg:rounded-lg xl:rounded-xl">
      <div className="overflow-x-auto">
        <table className="service-gap-table w-full min-w-[800px] text-left text-sm">
          <thead className="border-b border-[#E4E7EC] bg-[#FCFCFD] text-xs font-semibold text-[#344054] lg:text-[9px] xl:text-[10px] 2xl:text-xs">
            <tr>
              <th className="ranking-col-rank px-5 py-4">Prioritas</th>
              <th className="ranking-col-place px-4 py-4">Tempat</th>
              <th className="ranking-col-aspect px-4 py-4">Masalah</th>
              <th className="ranking-col-score px-4 py-4">Skor</th>
              <th className="ranking-col-evidence px-4 py-4">Bukti</th>
              <th className="ranking-col-confidence px-4 py-4">Confidence</th>
              <th className="ranking-col-status px-4 py-4">Status</th>
              <th className="ranking-col-action px-4 py-4">Aksi</th>
            </tr>
          </thead>
          <tbody>
            {items.length ? items.map((item) => (
              <tr key={`${item.place_id}-${item.aspect}`} className="border-b border-[#EAECF0] last:border-0 hover:bg-[#F8FAFC]">
                <td className="ranking-col-rank px-5 py-3"><span className={`grid size-8 place-items-center rounded-lg text-xs font-semibold lg:size-6 lg:rounded-md lg:text-[10px] xl:size-7 xl:text-[11px] 2xl:size-8 2xl:rounded-lg 2xl:text-xs ${rankTone(item.rank)}`}>{item.rank}</span></td>
                <td className="ranking-col-place max-w-[230px] px-4 py-3"><Link to={detailPath(item)} className="block truncate font-semibold text-[#101828] hover:text-[#1666D8]">{item.place_name}</Link><span className="mt-1 block truncate text-xs text-[#667085] lg:mt-0.5 lg:text-[9px] xl:text-[10px] 2xl:mt-1 2xl:text-xs">{formatLabel(item.category)}</span></td>
                <td className="ranking-col-aspect max-w-[230px] px-4 py-3"><p className="truncate text-[#475467]" title={item.explanation}>{formatLabel(item.aspect)}</p></td>
                <td className={`ranking-col-score px-4 py-3 font-semibold ${scoreTone(item.score)}`}>{item.score.toLocaleString("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                <td className="ranking-col-evidence px-4 py-3"><strong className="block font-semibold text-[#101828]">{formatNumber(item.evidence_count)}</strong><span className="text-[11px] text-[#667085] lg:text-[9px] xl:text-[10px] 2xl:text-[11px]">bukti</span></td>
                <td className="ranking-col-confidence px-4 py-3"><span className="inline-flex min-w-0 items-center gap-2 text-xs font-medium text-[#344054] lg:gap-1 lg:text-[9px] xl:text-[10px] 2xl:gap-2 2xl:text-xs"><span className={`size-2 shrink-0 rounded-full lg:size-1.5 2xl:size-2 ${confidenceColor[item.confidence.toLowerCase()] ?? "bg-[#98A2B3]"}`} /><span className="truncate">{formatLabel(item.confidence)}</span></span></td>
                <td className="ranking-col-status px-4 py-3"><span title="Belum tersedia" className="inline-flex max-w-full rounded-md bg-[#F2F4F7] px-2 py-1 text-[11px] font-medium text-[#475467] lg:px-1.5 lg:py-0.5 lg:text-[9px] xl:text-[10px] 2xl:px-2 2xl:py-1 2xl:text-[11px]"><span className="ranking-full-label truncate">Belum tersedia</span><span className="ranking-compact-label">Belum</span></span></td>
                <td className="ranking-col-action px-4 py-3"><Link to={detailPath(item)} title={`Lihat detail ${item.place_name} untuk masalah ${formatLabel(item.aspect)}`} aria-label={`Lihat detail ${item.place_name} untuk masalah ${formatLabel(item.aspect)}`} className="inline-flex h-9 items-center justify-center gap-2 whitespace-nowrap rounded-lg border border-[#B2CCEE] px-3 text-xs font-semibold text-[#1666D8] hover:bg-[#EAF2FC] lg:size-7 lg:rounded-md lg:px-0 xl:h-8 xl:w-auto xl:px-2 xl:text-[10px] 2xl:h-9 2xl:gap-2 2xl:rounded-lg 2xl:px-3 2xl:text-xs"><span className="ranking-action-label">Lihat Detail</span><ChevronRight size={14} className="shrink-0 lg:size-3.5 2xl:size-[14px]" /></Link></td>
              </tr>
            )) : (
              <tr><td colSpan={8} className="px-6 py-16 text-center text-sm text-[#667085]">Tidak ada prioritas yang sesuai dengan filter.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex flex-col gap-3 border-t border-[#E4E7EC] px-5 py-4 sm:flex-row sm:items-center sm:justify-between lg:gap-2 lg:px-3 lg:py-2.5 xl:px-4 xl:py-3 2xl:px-5 2xl:py-4">
        <p className="text-xs text-[#475467] lg:text-[9px] xl:text-[10px] 2xl:text-xs">Menampilkan {formatNumber(start)} - {formatNumber(end)} dari {formatNumber(total)} prioritas</p>
        <div className="flex flex-wrap items-center gap-2 lg:gap-1 xl:gap-1.5 2xl:gap-2">
          <label className="sr-only" htmlFor="ranking-page-size">Jumlah baris per halaman</label>
          <select id="ranking-page-size" value={pageSize} onChange={(event) => onPageSizeChange(Number(event.target.value))} className="h-9 rounded-lg border border-[#D0D5DD] bg-white px-3 text-xs text-[#344054] lg:h-7 lg:rounded-md lg:px-1.5 lg:text-[9px] xl:h-8 xl:px-2 xl:text-[10px] 2xl:h-9 2xl:rounded-lg 2xl:px-3 2xl:text-xs">
            <option value={8}>8 per halaman</option><option value={10}>10 per halaman</option><option value={20}>20 per halaman</option>
          </select>
          <button type="button" title="Halaman sebelumnya" aria-label="Halaman sebelumnya" disabled={page === 0} onClick={() => onPageChange(page - 1)} className="grid size-9 place-items-center rounded-lg border border-[#D0D5DD] text-[#475467] disabled:opacity-40 lg:size-7 lg:rounded-md lg:[&>svg]:size-3.5 xl:size-8 2xl:size-9 2xl:rounded-lg 2xl:[&>svg]:size-4"><ChevronLeft size={16} /></button>
          {visiblePages.map((value, index) => (
            <span key={value} className="contents">
              {index > 0 && visiblePages[index - 1] < value - 1 && <span className="px-1 text-xs text-[#98A2B3]">...</span>}
              <button type="button" aria-label={`Halaman ${value + 1}`} onClick={() => onPageChange(value)} className={`grid size-9 place-items-center rounded-lg text-xs font-semibold lg:size-7 lg:rounded-md lg:text-[9px] xl:size-8 xl:text-[10px] 2xl:size-9 2xl:rounded-lg 2xl:text-xs ${value === page ? "bg-[#1666D8] text-white" : "border border-[#D0D5DD] text-[#344054] hover:bg-[#F8FAFC]"}`}>{value + 1}</button>
            </span>
          ))}
          <button type="button" title="Halaman berikutnya" aria-label="Halaman berikutnya" disabled={page + 1 >= pageCount} onClick={() => onPageChange(page + 1)} className="grid size-9 place-items-center rounded-lg border border-[#D0D5DD] text-[#475467] disabled:opacity-40 lg:size-7 lg:rounded-md lg:[&>svg]:size-3.5 xl:size-8 2xl:size-9 2xl:rounded-lg 2xl:[&>svg]:size-4"><ChevronRight size={16} /></button>
        </div>
      </div>
    </section>
  );
}
