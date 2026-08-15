import { FormEvent, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  BarChart3,
  ChevronDown,
  CircleHelp,
  Info,
  LayoutGrid,
  Menu,
  Search,
  X,
} from "lucide-react";

const primaryNavigation = [
  { to: "/ringkasan-wilayah", label: "Ikhtisar Destinasi", icon: LayoutGrid },
  { to: "/service-gap-ranking", label: "Prioritas Penanganan", icon: BarChart3 },
];

function NavigationItems({ items, onNavigate }: {
  items: typeof primaryNavigation;
  onNavigate?: () => void;
}) {
  return items.map(({ to, label, icon: Icon }) => (
    <NavLink
      key={to}
      to={to}
      end={to === "/"}
      onClick={onNavigate}
      className={({ isActive }) =>
        `relative flex min-h-12 items-center gap-3 rounded-lg px-4 text-sm font-medium transition-colors lg:min-h-10 lg:gap-2 lg:px-3 lg:text-[11px] xl:min-h-11 xl:gap-2.5 xl:text-xs 2xl:min-h-12 2xl:gap-3 2xl:px-4 2xl:text-sm ${
          isActive
            ? "bg-[#EAF2FC] text-[#071A33] before:absolute before:inset-y-2 before:left-0 before:w-[3px] before:rounded-r before:bg-[#1666D8]"
            : "text-[#475467] hover:bg-slate-50 hover:text-[#071A33]"
        }`
      }
    >
      <Icon aria-hidden="true" size={19} strokeWidth={1.7} className="size-[19px] shrink-0 lg:size-4 xl:size-[18px] 2xl:size-[19px]" />
      <span className="whitespace-nowrap">{label}</span>
    </NavLink>
  ));
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      <div className="px-6 pb-11 pt-8 lg:px-4 lg:pb-7 lg:pt-6 xl:px-5 xl:pb-9 xl:pt-7 2xl:px-6 2xl:pb-11 2xl:pt-8">
        <p className="text-[30px] font-bold leading-none text-[#071A33] lg:text-2xl xl:text-[25px] 2xl:text-[30px]">PodaNauli</p>
        <p className="mt-2 whitespace-nowrap text-xs text-[#475467] lg:mt-1.5 lg:text-[10px] xl:text-[11px] 2xl:mt-2 2xl:text-xs">Akselerator Pariwisata Toba</p>
      </div>

      <nav aria-label="Navigasi utama" className="flex-1 px-4 lg:px-3 2xl:px-4">
        <div className="space-y-1">
          <NavigationItems items={primaryNavigation} onNavigate={onNavigate} />
        </div>
      </nav>

      <div className="mx-6 border-t border-[#E4E7EC] py-5 lg:mx-4 lg:py-3 xl:mx-5 xl:py-4 2xl:mx-6 2xl:py-5">
        <a href="#tentang" className="flex h-11 items-center gap-3 text-sm text-[#475467] hover:text-[#071A33] lg:h-9 lg:gap-2 lg:text-[11px] xl:h-10 xl:text-xs 2xl:h-11 2xl:gap-3 2xl:text-sm">
          <Info aria-hidden="true" size={18} strokeWidth={1.7} />
          Tentang PodaNauli
        </a>
        <a href="#bantuan" className="flex h-11 items-center gap-3 text-sm text-[#475467] hover:text-[#071A33] lg:h-9 lg:gap-2 lg:text-[11px] xl:h-10 xl:text-xs 2xl:h-11 2xl:gap-3 2xl:text-sm">
          <CircleHelp aria-hidden="true" size={18} strokeWidth={1.7} />
          Bantuan
        </a>
      </div>
    </>
  );
}

function Topbar() {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = search.trim();
    navigate(value ? `/service-gap-ranking?search=${encodeURIComponent(value)}` : "/service-gap-ranking");
  }

  return (
    <header className="sticky top-0 z-40 hidden h-20 items-center border-b border-[#E4E7EC] bg-white px-8 lg:flex lg:h-16 lg:px-4 xl:h-[72px] xl:px-5 2xl:h-20 2xl:px-8">
      <form onSubmit={submitSearch} className="relative w-full max-w-[500px] lg:max-w-[300px] xl:max-w-[390px] 2xl:max-w-[500px]">
        <label htmlFor="global-search" className="sr-only">Cari destinasi, wilayah, atau aspek</label>
        <Search aria-hidden="true" className="absolute left-4 top-1/2 -translate-y-1/2 text-[#667085] lg:left-3 lg:size-4 xl:left-3.5 xl:size-[18px] 2xl:left-4 2xl:size-5" size={20} strokeWidth={1.7} />
        <input
          id="global-search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Cari destinasi, wilayah, atau aspek..."
          className="h-12 w-full rounded-xl border border-[#D0D5DD] bg-white pl-12 pr-4 text-sm text-[#101828] outline-none placeholder:text-[#667085] focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] lg:h-10 lg:rounded-lg lg:pl-9 lg:pr-3 lg:text-[11px] xl:h-11 xl:rounded-xl xl:pl-10 xl:text-[10px] 2xl:h-12 2xl:pl-12 2xl:pr-4 2xl:text-sm"
        />
      </form>

      <div className="ml-auto flex items-center gap-5 lg:gap-3 xl:gap-4 2xl:gap-5">
        <div className="flex items-center gap-3 text-left">
          <span className="grid size-11 place-items-center rounded-full bg-[#071A33] text-sm font-semibold text-white lg:size-9 lg:text-[11px] xl:size-10 xl:text-xs 2xl:size-11 2xl:text-sm">DP</span>
          <span className="hidden lg:block">
            <strong className="block text-sm font-semibold text-[#101828] lg:text-[11px] xl:text-[10px] 2xl:text-sm">Dinas Pariwisata</strong>
            <span className="mt-0.5 block text-xs text-[#667085] lg:text-[10px] xl:text-[11px] 2xl:text-xs">Kab. Samosir</span>
          </span>
          <ChevronDown aria-hidden="true" size={16} className="hidden text-[#667085] lg:block" />
        </div>
      </div>
    </header>
  );
}

export function Layout() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const hasPageLevelTopbar = location.pathname === "/service-gap-ranking";

  return (
    <div className="min-h-screen bg-white lg:grid lg:grid-cols-[208px_minmax(0,1fr)] xl:grid-cols-[232px_minmax(0,1fr)] 2xl:grid-cols-[256px_minmax(0,1fr)]">
      <aside className="sticky top-0 z-40 hidden h-screen min-h-0 flex-col border-r border-[#E4E7EC] bg-white lg:flex">
        <SidebarContent />
      </aside>

      <div className="min-w-0 bg-white">
        <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-[#E4E7EC] bg-white px-4 lg:hidden">
          <div>
            <p className="text-xl font-bold leading-none text-[#071A33]">PodaNauli</p>
            <p className="mt-1 text-[10px] text-[#667085]">Akselerator Pariwisata Toba</p>
          </div>
          <button
            type="button"
            aria-label="Buka navigasi"
            title="Buka navigasi"
            onClick={() => setOpen(true)}
            className="grid size-10 place-items-center rounded-lg border border-[#D0D5DD] text-[#475467]"
          >
            <Menu size={21} />
          </button>
        </header>

        {!hasPageLevelTopbar && <Topbar />}

        {open && (
          <div className="fixed inset-0 z-50 lg:hidden">
            <button type="button" className="absolute inset-0 bg-[#071A33]/35" aria-label="Tutup navigasi" onClick={() => setOpen(false)} />
            <aside className="relative flex h-full w-[min(320px,88vw)] flex-col bg-white shadow-xl">
              <button
                type="button"
                aria-label="Tutup navigasi"
                title="Tutup navigasi"
                onClick={() => setOpen(false)}
                className="absolute right-4 top-6 grid size-9 place-items-center rounded-lg text-[#475467] hover:bg-slate-50"
              >
                <X size={20} />
              </button>
              <SidebarContent onNavigate={() => setOpen(false)} />
            </aside>
          </div>
        )}

        <main className="min-w-0 px-4 pb-8 pt-6 sm:px-5 lg:px-4 lg:pt-5 xl:px-5 xl:pt-6 2xl:px-8 2xl:pt-7">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
