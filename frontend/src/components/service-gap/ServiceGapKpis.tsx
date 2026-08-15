import { CircleAlert, ListChecks, MessageSquareWarning, TrendingUp } from "lucide-react";
import { formatNumber } from "../../lib/api";
import type { ServiceGap } from "../../types/api";

const kpiStyle = "min-w-0 rounded-xl border border-[#E4E7EC] bg-white p-5 shadow-[0_1px_2px_rgba(16,24,40,0.04)] lg:rounded-lg lg:p-3 xl:rounded-xl xl:p-4 2xl:p-5";

function KpiCard({ label, value, detail, icon, iconTone }: {
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
  iconTone: string;
}) {
  return (
    <article className={kpiStyle}>
      <div className="flex items-center gap-5 lg:gap-3 xl:gap-4 2xl:gap-5">
        <span className={`grid size-14 shrink-0 place-items-center rounded-xl lg:size-10 lg:rounded-lg lg:[&>svg]:size-5 xl:size-12 xl:rounded-xl xl:[&>svg]:size-[22px] 2xl:size-14 2xl:[&>svg]:size-6 ${iconTone}`}>{icon}</span>
        <div className="min-w-0">
          <p className="truncate text-sm text-[#475467] lg:text-[10px] xl:text-[10px] 2xl:text-sm">{label}</p>
          <p className="mt-1 text-[30px] font-semibold leading-none text-[#071A33] lg:text-[23px] xl:text-[23px] 2xl:text-[30px]">{value}</p>
          <p className="mt-2 truncate text-xs text-[#667085] lg:mt-1.5 lg:text-[9px] xl:text-[9px] 2xl:mt-2 2xl:text-xs">{detail}</p>
        </div>
      </div>
    </article>
  );
}

export function ServiceGapKpis({ total, items }: { total: number; items: ServiceGap[] }) {
  const highConfidence = items.filter((item) => item.confidence.toLowerCase() === "high").length;
  const evidence = items.reduce((sum, item) => sum + item.evidence_count, 0);
  const needsAttention = items.filter((item) => item.score >= 40).length;

  return (
    <section aria-label="Ringkasan service gap" className="service-kpi-grid gap-3 lg:gap-2 xl:gap-3 2xl:gap-4">
      <KpiCard label="Total Prioritas" value={formatNumber(total)} detail="Sesuai filter aktif" icon={<ListChecks size={25} />} iconTone="bg-[#EAF2FC] text-[#1666D8]" />
      <KpiCard label="Confidence Tinggi" value={formatNumber(highConfidence)} detail="Pada halaman ini" icon={<TrendingUp size={25} />} iconTone="bg-emerald-50 text-[#039855]" />
      <KpiCard label="Bukti Keluhan" value={formatNumber(evidence)} detail="Pada halaman ini" icon={<MessageSquareWarning size={24} />} iconTone="bg-amber-50 text-[#F79009]" />
      <KpiCard label="Perlu Perhatian" value={formatNumber(needsAttention)} detail="Skor minimal 40 pada halaman" icon={<CircleAlert size={25} />} iconTone="bg-red-50 text-[#F04438]" />
    </section>
  );
}
