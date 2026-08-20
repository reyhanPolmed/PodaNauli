import { FormEvent, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  BarChart3,
  LayoutGrid,
  LogIn,
  LogOut,
  Menu,
  Search,
  UploadCloud,
  X,
} from "lucide-react";
import { useAuth } from "../auth/AuthContext";

const primaryNavigation = [
  { to: "/ringkasan-wilayah", label: "Ikhtisar Destinasi", icon: LayoutGrid, adminOnly: false },
  { to: "/service-gap-ranking", label: "Prioritas Penanganan", icon: BarChart3, adminOnly: false },
  { to: "/impor-data", label: "Tambah Ulasan", icon: UploadCloud, adminOnly: true },
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

function SidebarContent({ onNavigate, onRequestLogout }: {
  onNavigate?: () => void;
  onRequestLogout: () => void;
}) {
  const auth = useAuth();
  const navigate = useNavigate();
  const visibleNavigation = primaryNavigation.filter((item) => !item.adminOnly || auth.isAdmin);

  return (
    <>
      <div className="px-6 pb-11 pt-8 lg:px-4 lg:pb-7 lg:pt-6 xl:px-5 xl:pb-9 xl:pt-7 2xl:px-6 2xl:pb-11 2xl:pt-8">
        <p className="text-[30px] font-bold leading-none text-[#071A33] lg:text-2xl xl:text-[25px] 2xl:text-[30px]">PodaNauli</p>
        <p className="mt-2 whitespace-nowrap text-xs text-[#475467] lg:mt-1.5 lg:text-[10px] xl:text-[11px] 2xl:mt-2 2xl:text-xs">Akselerator Pariwisata Toba</p>
      </div>

      <nav aria-label="Navigasi utama" className="flex-1 px-4 lg:px-3 2xl:px-4">
        <div className="space-y-1">
          <NavigationItems items={visibleNavigation} onNavigate={onNavigate} />
        </div>
      </nav>

      <div className="mx-6 border-t border-[#E4E7EC] py-5 lg:mx-4 lg:py-3 xl:mx-5 xl:py-4 2xl:mx-6 2xl:py-5">
        <button
          type="button"
          onClick={() => {
            if (auth.isAdmin) onRequestLogout();
            else navigate("/login");
            onNavigate?.();
          }}
          className="flex h-11 w-full cursor-pointer items-center gap-3 text-left text-sm font-medium text-[#475467] hover:text-[#071A33] lg:h-10 lg:gap-2 lg:text-[11px] xl:text-xs 2xl:h-12 2xl:gap-3 2xl:text-sm"
        >
          {auth.isAdmin ? <LogOut aria-hidden="true" size={18} strokeWidth={1.7} /> : <LogIn aria-hidden="true" size={18} strokeWidth={1.7} />}
          {auth.isAdmin ? "Logout" : "Login"}
        </button>
      </div>
    </>
  );
}

function Topbar({ showSearch = true, onRequestLogout }: {
  showSearch?: boolean;
  onRequestLogout: () => void;
}) {
  const navigate = useNavigate();
  const auth = useAuth();
  const [search, setSearch] = useState("");

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = search.trim();
    navigate(value ? `/service-gap-ranking?search=${encodeURIComponent(value)}` : "/service-gap-ranking");
  }

  return (
    <header className="sticky top-0 z-40 hidden h-20 items-center border-b border-[#E4E7EC] bg-white px-8 lg:flex lg:h-16 lg:px-4 xl:h-[72px] xl:px-5 2xl:h-20 2xl:px-8">
      {showSearch && <form onSubmit={submitSearch} className="relative w-full max-w-[500px] lg:max-w-[300px] xl:max-w-[390px] 2xl:max-w-[500px]">
        <label htmlFor="global-search" className="sr-only">Cari destinasi, wilayah, atau aspek</label>
        <Search aria-hidden="true" className="absolute left-4 top-1/2 -translate-y-1/2 text-[#667085] lg:left-3 lg:size-4 xl:left-3.5 xl:size-[18px] 2xl:left-4 2xl:size-5" size={20} strokeWidth={1.7} />
        <input
          id="global-search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Cari destinasi, wilayah, atau aspek..."
          className="h-12 w-full rounded-xl border border-[#D0D5DD] bg-white pl-12 pr-4 text-sm text-[#101828] outline-none placeholder:text-[#667085] focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC] lg:h-10 lg:rounded-lg lg:pl-9 lg:pr-3 lg:text-[11px] xl:h-11 xl:rounded-xl xl:pl-10 xl:text-[10px] 2xl:h-12 2xl:pl-12 2xl:pr-4 2xl:text-sm"
        />
      </form>}

      {!showSearch && (
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid size-10 shrink-0 place-items-center rounded-lg bg-[#EAF2FC] text-[#1666D8] lg:size-9 xl:size-10 2xl:size-11">
            <LayoutGrid aria-hidden="true" className="size-[18px] lg:size-4 xl:size-[18px] 2xl:size-5" strokeWidth={1.7} />
          </span>
          <span className="min-w-0">
            <strong className="block truncate text-base font-semibold text-[#101828] lg:text-sm xl:text-base 2xl:text-lg">Ikhtisar Destinasi</strong>
            <span className="mt-0.5 block truncate text-[11px] text-[#667085] lg:text-[9px] xl:text-[10px] 2xl:text-xs">Gambaran umum kinerja layanan wisata di kawasan Danau Toba.</span>
          </span>
        </div>
      )}

      <div className="ml-auto flex items-center gap-3">
        {auth.isAdmin && auth.user ? (
          <>
            <span className="grid size-11 place-items-center rounded-full bg-[#071A33] text-sm font-semibold text-white lg:size-9 lg:text-[11px] xl:size-10 xl:text-xs 2xl:size-11 2xl:text-sm">ST</span>
            <span className="hidden lg:block">
              <strong className="block text-sm font-semibold text-[#101828] lg:text-[11px] xl:text-[10px] 2xl:text-sm">{auth.user.display_name}</strong>
              <span className="mt-0.5 block text-xs capitalize text-[#667085] lg:text-[10px] xl:text-[11px] 2xl:text-xs">{auth.user.role}</span>
            </span>
            <button type="button" aria-label="Logout" title="Logout" onClick={onRequestLogout} className="inline-flex h-9 cursor-pointer items-center gap-1.5 rounded-lg border border-[#D0D5DD] px-2.5 text-[10px] font-semibold text-[#475467] hover:bg-[#F8FAFC] hover:text-[#101828] 2xl:h-10 2xl:gap-2 2xl:px-3 2xl:text-xs"><LogOut size={16} /> Logout</button>
          </>
        ) : (
          <button type="button" onClick={() => navigate("/login")} className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-lg border border-[#D0D5DD] px-3 text-[11px] font-semibold text-[#344054] hover:bg-[#F8FAFC] 2xl:h-11 2xl:px-4 2xl:text-sm"><LogIn size={16} /> Login</button>
        )}
      </div>
    </header>
  );
}

export function Layout() {
  const [open, setOpen] = useState(false);
  const [logoutDialogOpen, setLogoutDialogOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const auth = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const hasPageLevelTopbar = ["/service-gap-ranking", "/impor-data"].includes(location.pathname)
    || location.pathname.startsWith("/destinasi/");
  const isRegionOverview = location.pathname === "/ringkasan-wilayah";
  const showGlobalSearch = !isRegionOverview;

  useEffect(() => {
    if (!logoutDialogOpen) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !isLoggingOut) setLogoutDialogOpen(false);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [logoutDialogOpen, isLoggingOut]);

  async function confirmLogout() {
    setIsLoggingOut(true);
    try {
      await auth.logout();
    } catch {
      // AuthContext still clears the local session when the API cannot respond.
    } finally {
      setIsLoggingOut(false);
      setLogoutDialogOpen(false);
      setOpen(false);
      navigate("/ringkasan-wilayah");
    }
  }

  return (
    <div className="min-h-screen bg-white lg:grid lg:grid-cols-[208px_minmax(0,1fr)] xl:grid-cols-[232px_minmax(0,1fr)] 2xl:grid-cols-[256px_minmax(0,1fr)]">
      <aside className="sticky top-0 z-40 hidden h-screen min-h-0 flex-col border-r border-[#E4E7EC] bg-white lg:flex">
        <SidebarContent onRequestLogout={() => setLogoutDialogOpen(true)} />
      </aside>

      <div className="min-w-0 bg-white">
        <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-[#E4E7EC] bg-white px-4 lg:hidden">
          <div className="min-w-0 pr-2">
            <p className={`${isRegionOverview ? "text-sm" : "text-xl"} truncate font-bold leading-none text-[#071A33]`}>{isRegionOverview ? "Ikhtisar Destinasi" : "PodaNauli"}</p>
            <p className="mt-1 line-clamp-2 text-[9px] leading-3 text-[#667085]">{isRegionOverview ? "Gambaran umum kinerja layanan wisata di kawasan Danau Toba." : "Akselerator Pariwisata Toba"}</p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" aria-label={auth.isAdmin ? "Logout" : "Login"} title={auth.isAdmin ? "Logout" : "Login"} onClick={() => auth.isAdmin ? setLogoutDialogOpen(true) : navigate("/login")} className="inline-flex h-10 cursor-pointer items-center gap-1.5 rounded-lg border border-[#D0D5DD] px-2.5 text-xs font-semibold text-[#475467]">
              {auth.isAdmin ? <LogOut size={17} /> : <LogIn size={17} />}
              {auth.isAdmin ? "Logout" : "Login"}
            </button>
            <button
              type="button"
              aria-label="Buka navigasi"
              title="Buka navigasi"
              onClick={() => setOpen(true)}
              className="grid size-10 place-items-center rounded-lg border border-[#D0D5DD] text-[#475467]"
            >
              <Menu size={21} />
            </button>
          </div>
        </header>

        {!hasPageLevelTopbar && <Topbar showSearch={showGlobalSearch} onRequestLogout={() => setLogoutDialogOpen(true)} />}

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
              <SidebarContent onNavigate={() => setOpen(false)} onRequestLogout={() => setLogoutDialogOpen(true)} />
            </aside>
          </div>
        )}

        <main className="min-w-0 px-4 pb-8 pt-6 sm:px-5 lg:px-4 lg:pt-5 xl:px-5 xl:pt-6 2xl:px-8 2xl:pt-7">
          <Outlet />
        </main>
      </div>

      {logoutDialogOpen && (
        <div className="fixed inset-0 z-[1000] flex items-center justify-center p-4">
          <button
            type="button"
            aria-label="Tutup konfirmasi logout"
            disabled={isLoggingOut}
            onClick={() => setLogoutDialogOpen(false)}
            className="absolute inset-0 cursor-default bg-[#071A33]/35 backdrop-blur-[3px]"
          />
          <section role="dialog" aria-modal="true" aria-labelledby="logout-dialog-title" className="relative z-10 w-full max-w-sm rounded-xl border border-[#E4E7EC] bg-white p-6 shadow-[0_20px_48px_rgba(7,26,51,0.22)]">
            <span className="grid size-11 place-items-center rounded-lg bg-red-50 text-[#D92D20]">
              <LogOut aria-hidden="true" size={20} strokeWidth={1.8} />
            </span>
            <h2 id="logout-dialog-title" className="mt-5 text-lg font-semibold text-[#101828]">Konfirmasi logout</h2>
            <p className="mt-2 text-sm leading-6 text-[#667085]">Sesi stakeholder akan diakhiri. Anda perlu login kembali untuk menambah atau mengelola ulasan.</p>
            <div className="mt-6 grid grid-cols-2 gap-3">
              <button type="button" autoFocus disabled={isLoggingOut} onClick={() => setLogoutDialogOpen(false)} className="inline-flex h-10 cursor-pointer items-center justify-center rounded-lg border border-[#D0D5DD] px-4 text-xs font-semibold text-[#344054] hover:bg-[#F8FAFC] disabled:cursor-not-allowed disabled:opacity-60">Batal</button>
              <button type="button" disabled={isLoggingOut} onClick={confirmLogout} className="inline-flex h-10 cursor-pointer items-center justify-center gap-2 rounded-lg bg-[#D92D20] px-4 text-xs font-semibold text-white hover:bg-[#B42318] disabled:cursor-not-allowed disabled:opacity-60"><LogOut size={15} /> {isLoggingOut ? "Memproses" : "Logout"}</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
