import { ChevronDown, LoaderCircle, RotateCcw, Search, SlidersHorizontal, X } from "lucide-react";
import { useEffect, useState } from "react";
import { formatLabel } from "../../lib/api";
import { clusterAreaName } from "../../lib/clusterAreas";
import type { RankingFilters } from "./types";

const categories = ["wisata", "restoran", "hotel", "hotel_resto"];
const aspects = [
  "akses_jalan", "transportasi", "parkir", "kebersihan", "toilet", "harga", "pelayanan",
  "makanan", "akomodasi", "keamanan", "jam_operasional", "fasilitas_umum", "pemandangan",
  "keramaian", "aksesibilitas", "budaya",
];

function SelectFilter({ label, value, onChange, children, disabled = false }: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
  disabled?: boolean;
}) {
  const active = !disabled && value !== "" && value !== "all";

  return (
    <label className="min-w-0">
      <span className="mb-1.5 block truncate text-[10px] font-medium text-[#667085] lg:mb-1 lg:text-[9px] xl:text-[9px] 2xl:mb-1.5 2xl:text-[10px]">{label}</span>
      <span className="relative block">
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          className={`h-10 w-full appearance-none rounded-lg border py-0 pl-3 pr-8 text-[11px] font-medium outline-none transition-colors focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] disabled:cursor-not-allowed disabled:border-[#E4E7EC] disabled:bg-[#F2F4F7] disabled:text-[#98A2B3] lg:h-8 lg:pl-2.5 lg:pr-7 lg:text-[9px] xl:h-9 xl:text-[9px] 2xl:h-10 2xl:pl-3 2xl:pr-8 2xl:text-[10px] ${active ? "border-[#84ADFF] bg-[#F5F8FF] text-[#175CD3]" : "border-[#D0D5DD] bg-white text-[#344054] hover:border-[#98A2B3]"}`}
        >
          {children}
        </select>
        <ChevronDown aria-hidden="true" className="pointer-events-none absolute right-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[#667085] lg:right-2 lg:size-3 xl:size-3.5 2xl:right-2.5" />
      </span>
    </label>
  );
}

export function ServiceGapSearchHeader({ appliedSearch, isSearching, onSearch }: {
  appliedSearch: string;
  isSearching: boolean;
  onSearch: (value: string) => void;
}) {
  const [search, setSearch] = useState(appliedSearch);
  const hasChanges = search.trim() !== appliedSearch;

  useEffect(() => setSearch(appliedSearch), [appliedSearch]);

  return (
    <form
      role="search"
      onSubmit={(event) => {
        event.preventDefault();
        onSearch(search.trim());
      }}
      className="flex w-full items-center gap-2 lg:w-[390px] xl:w-[460px] 2xl:w-[540px]"
    >
      <label className="relative min-w-0 flex-1">
        <span className="sr-only">Cari destinasi, masalah, atau kata kunci</span>
        <Search aria-hidden="true" className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[#667085] lg:left-3 lg:size-4 xl:size-[17px] 2xl:left-4 2xl:size-[18px]" size={18} strokeWidth={1.7} />
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Cari destinasi atau masalah"
          className="h-12 w-full rounded-lg border border-[#D0D5DD] bg-white pl-11 pr-10 text-sm text-[#101828] outline-none transition-colors placeholder:text-[#98A2B3] hover:border-[#98A2B3] focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] lg:h-10 lg:pl-9 lg:text-[10px] xl:h-11 xl:text-[11px] 2xl:h-12 2xl:pl-11 2xl:text-sm"
        />
        {search && (
          <button type="button" onClick={() => { setSearch(""); onSearch(""); }} title="Bersihkan pencarian" aria-label="Bersihkan pencarian" className="absolute right-2 top-1/2 grid size-7 -translate-y-1/2 place-items-center rounded-md text-[#667085] outline-none hover:bg-[#F2F4F7] hover:text-[#344054] focus-visible:ring-2 focus-visible:ring-[#84ADFF]">
            <X size={15} />
          </button>
        )}
      </label>
      <button
        type="submit"
        disabled={!hasChanges || isSearching}
        className="inline-flex h-12 shrink-0 items-center justify-center gap-2 rounded-lg bg-[#1666D8] px-4 text-sm font-semibold text-white outline-none transition-colors hover:bg-[#0B54B8] focus-visible:ring-2 focus-visible:ring-[#84ADFF] focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:bg-[#B2CCFF] lg:h-10 lg:px-3 lg:text-[10px] xl:h-11 xl:px-4 xl:text-[11px] 2xl:h-12 2xl:text-sm"
      >
        {isSearching ? <LoaderCircle aria-hidden="true" className="size-4 animate-spin" /> : <Search aria-hidden="true" className="size-4" />}
        {isSearching ? "Mencari" : "Cari"}
      </button>
    </form>
  );
}

export function ServiceGapFilters({ filters, clusterIds, onChange, onReset }: {
  filters: RankingFilters;
  clusterIds: number[];
  onChange: <K extends keyof RankingFilters>(key: K, value: RankingFilters[K]) => void;
  onReset: () => void;
}) {
  const activeFilterCount = [filters.clusterId, filters.category, filters.aspect, filters.minReviews, filters.confidence, filters.search].filter(Boolean).length;

  return (
    <section aria-label="Filter prioritas penanganan" className="service-filter-grid gap-3 rounded-xl border border-[#DDE3EC] bg-[#FCFCFD] p-4 shadow-[0_2px_8px_rgba(16,24,40,0.05)] lg:gap-2 lg:rounded-lg lg:p-3 xl:rounded-xl xl:p-3.5 2xl:gap-3 2xl:p-4">
      <div className="col-span-full mb-1 flex items-center justify-between gap-3 border-b border-[#E4E7EC] pb-3 lg:pb-2 2xl:pb-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="grid size-7 shrink-0 place-items-center rounded-md bg-[#EAF2FC] text-[#1666D8] lg:size-6 2xl:size-7">
            <SlidersHorizontal aria-hidden="true" className="size-3.5 lg:size-3 2xl:size-3.5" />
          </span>
          <h2 className="text-xs font-semibold text-[#101828] lg:text-[10px] xl:text-[10px] 2xl:text-xs">Filter Prioritas</h2>
        </div>
        {activeFilterCount > 0 && <span className="shrink-0 rounded-md bg-[#EAF2FC] px-2 py-1 text-[9px] font-semibold text-[#175CD3] lg:text-[9px] 2xl:text-[10px]">{activeFilterCount} filter aktif</span>}
      </div>
      <SelectFilter label="Kawasan" value={filters.clusterId} onChange={(value) => onChange("clusterId", value)}>
        <option value="">Semua Kawasan</option>
        {clusterIds.map((clusterId) => <option key={clusterId} value={clusterId}>{clusterAreaName(clusterId)}</option>)}
      </SelectFilter>
      <SelectFilter label="Kategori" value={filters.category} onChange={(value) => onChange("category", value)}>
        <option value="">Semua Kategori</option>
        {categories.map((item) => <option key={item} value={item}>{formatLabel(item)}</option>)}
      </SelectFilter>
      <SelectFilter label="Aspek Masalah" value={filters.aspect} onChange={(value) => onChange("aspect", value)}>
        <option value="">Semua Aspek</option>
        {aspects.map((item) => <option key={item} value={item}>{formatLabel(item)}</option>)}
      </SelectFilter>
      <SelectFilter label="Jumlah Ulasan" value={filters.minReviews} onChange={(value) => onChange("minReviews", value)}>
        <option value="">Semua</option>
        <option value="10">Minimal 10</option>
        <option value="25">Minimal 25</option>
        <option value="50">Minimal 50</option>
      </SelectFilter>
      <SelectFilter label="Confidence" value={filters.confidence} onChange={(value) => onChange("confidence", value)}>
        <option value="">Semua</option>
        <option value="high">Tinggi</option>
        <option value="medium">Menengah</option>
        <option value="low">Rendah</option>
      </SelectFilter>
      <SelectFilter label="Periode Waktu" value={filters.period} onChange={(value) => onChange("period", value)} disabled>
        <option value="all">Seluruh Periode</option>
      </SelectFilter>
      <SelectFilter label="Status Penanganan" value={filters.handlingStatus} onChange={(value) => onChange("handlingStatus", value)} disabled>
        <option value="">Belum Tersedia</option>
      </SelectFilter>
      <div className="min-w-0">
        <span className="mb-1.5 block text-[10px] font-medium text-[#667085] lg:mb-1 lg:text-[9px] xl:text-[9px] 2xl:mb-1.5 2xl:text-[10px]">Aksi</span>
        <button type="button" onClick={onReset} disabled={activeFilterCount === 0} title="Reset filter" className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg border border-[#D0D5DD] bg-white px-3 text-[11px] font-semibold text-[#475467] outline-none transition-colors hover:border-[#98A2B3] hover:bg-[#F8FAFC] focus-visible:ring-2 focus-visible:ring-[#84ADFF] disabled:cursor-not-allowed disabled:bg-[#F2F4F7] disabled:text-[#98A2B3] lg:h-8 lg:gap-1 lg:px-2 lg:text-[9px] xl:h-9 xl:text-[9px] 2xl:h-10 2xl:gap-2 2xl:px-3 2xl:text-[10px]">
          <RotateCcw aria-hidden="true" className="size-4 lg:size-3.5 2xl:size-4" /> <span className="truncate">Reset Filter</span>
        </button>
      </div>
    </section>
  );
}
