import { AlertTriangle, LoaderCircle } from "lucide-react";
import type { ReactNode } from "react";

export function PageHeader({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-col justify-between gap-4 sm:flex-row sm:items-end lg:mb-4 lg:gap-3 xl:mb-5 2xl:mb-6 2xl:gap-4">
      <div>
        <h1 className="text-[28px] font-bold leading-tight text-[#071A33] lg:text-[26px] xl:text-[28px] 2xl:text-4xl">{title}</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-[#667085] lg:mt-1.5 lg:text-xs lg:leading-5 xl:text-[11px] 2xl:mt-2 2xl:text-sm 2xl:leading-6">{description}</p>
      </div>
      {action}
    </div>
  );
}

export function MetricCard({ label, value, detail, compactValue = false }: {
  label: string; value: string; detail?: string; icon?: unknown; tone?: "emerald" | "amber" | "blue" | "rose"; compactValue?: boolean;
}) {
  return (
    <div className="rounded-xl border border-[#E4E7EC] bg-white p-5 shadow-[0_1px_2px_rgba(16,24,40,0.04)] lg:rounded-lg lg:p-3 xl:rounded-xl xl:p-4 2xl:p-5">
      <p className="text-[13px] font-semibold text-[#475467] lg:text-[10px] xl:text-[10px] 2xl:text-[13px]">{label}</p>
      <p className={`mt-4 break-words font-semibold text-[#071A33] lg:mt-2.5 xl:mt-3 2xl:mt-4 ${compactValue ? "text-base leading-snug lg:text-[14px] xl:text-[15px] 2xl:text-lg" : "text-[30px] leading-none lg:text-[23px] xl:text-[23px] 2xl:text-[30px]"}`}>{value}</p>
      {detail && <p className="mt-3 text-xs leading-5 text-[#667085] lg:mt-2 lg:text-[10px] lg:leading-4 xl:text-[10px] 2xl:mt-3 2xl:text-xs 2xl:leading-5">{detail}</p>}
    </div>
  );
}

export function Panel({ title, subtitle, children, className = "" }: {
  title: string; subtitle?: string; children: ReactNode; className?: string;
}) {
  return (
    <section className={`min-w-0 rounded-xl border border-[#E4E7EC] bg-white shadow-[0_1px_2px_rgba(16,24,40,0.04)] ${className}`}>
      <div className="border-b border-[#E4E7EC] px-5 py-4 lg:px-3.5 lg:py-3 xl:px-4 xl:py-3.5 2xl:px-5 2xl:py-4">
        <h2 className="text-lg font-semibold text-[#101828] lg:text-sm xl:text-base 2xl:text-lg">{title}</h2>
        {subtitle && <p className="mt-1 text-xs leading-5 text-[#667085] lg:text-[10px] lg:leading-4 xl:text-[11px] 2xl:text-xs 2xl:leading-5">{subtitle}</p>}
      </div>
      <div className="p-5 lg:p-3.5 xl:p-4 2xl:p-5">{children}</div>
    </section>
  );
}

export function LoadingState({ label = "Memuat data" }: { label?: string }) {
  return <div className="grid min-h-64 place-items-center text-sm text-slate-500"><div className="flex items-center gap-2"><LoaderCircle className="animate-spin" size={18} />{label}</div></div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="flex min-h-52 items-center justify-center rounded-lg border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800"><AlertTriangle className="mr-2 shrink-0" size={19} />{message}</div>;
}

export function EmptyState({ message }: { message: string }) {
  return <div className="grid min-h-40 place-items-center text-center text-sm text-slate-500">{message}</div>;
}

export function ScoreBadge({ score }: { score: number }) {
  const tone = score >= 60 ? "bg-rose-100 text-rose-800" : score >= 40 ? "bg-amber-100 text-amber-800" : "bg-slate-100 text-slate-700";
  return <span className={`inline-flex min-w-14 justify-center rounded-md px-2 py-1 text-xs font-semibold ${tone}`}>{score.toLocaleString("id-ID", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}</span>;
}

export function PriorityBadge({ value }: { value: string }) {
  const normalized = value.toLowerCase();
  const tone = ["high", "tinggi", "critical", "kritis"].includes(normalized)
    ? "bg-red-50 text-[#D92D20]"
    : ["low", "rendah", "baik"].includes(normalized)
      ? "bg-emerald-50 text-[#027A48]"
      : "bg-amber-50 text-[#B54708]";
  return <span className={`inline-flex rounded-md px-2 py-1 text-xs font-semibold ${tone}`}>{value}</span>;
}
