import { ArrowLeft, ArrowRight, Eye, EyeOff, LockKeyhole } from "lucide-react";
import { type FormEvent, useState } from "react";
import { Link, Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

export function LoginPage() {
  const auth = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [username, setUsername] = useState("stakeholder");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const requestedNext = searchParams.get("next") ?? "/impor-data";
  const next = requestedNext.startsWith("/") && !requestedNext.startsWith("//")
    ? requestedNext
    : "/impor-data";

  if (!auth.isLoading && auth.isAdmin) return <Navigate to={next} replace />;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await auth.login(username.trim(), password);
      navigate(next, { replace: true });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Login tidak berhasil.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#F8FAFC] px-5 py-6 sm:px-8 lg:px-12 lg:py-8">
      <div className="mx-auto flex min-h-[calc(100vh-48px)] max-w-6xl flex-col lg:min-h-[calc(100vh-64px)]">
        <header className="flex items-center justify-between border-b border-[#E4E7EC] pb-5">
          <Link to="/ringkasan-wilayah" className="text-xl font-bold text-[#071A33] sm:text-2xl">PodaNauli</Link>
          <Link to="/ringkasan-wilayah" className="inline-flex items-center gap-2 text-xs font-semibold text-[#475467] hover:text-[#071A33] sm:text-sm">
            <ArrowLeft size={16} /> Kembali ke Ikhtisar
          </Link>
        </header>

        <div className="grid flex-1 items-center gap-10 py-10 lg:grid-cols-[minmax(0,1fr)_420px] lg:gap-20">
          <section className="max-w-xl">
            <p className="text-xs font-semibold uppercase text-[#1666D8]">Akses stakeholder</p>
            <h1 className="mt-4 text-[32px] font-semibold leading-tight text-[#071A33] sm:text-4xl lg:text-[42px]">
              Kelola masukan dengan akses yang bertanggung jawab.
            </h1>
            <p className="mt-5 max-w-lg text-sm leading-7 text-[#667085] sm:text-base">
              Ikhtisar dan prioritas tetap dapat dilihat publik. Login hanya diperlukan untuk menambah ulasan serta mengelola data yang masuk ke analisis.
            </p>
            <div className="mt-8 hidden border-l-2 border-[#1666D8] pl-4 text-sm leading-6 text-[#475467] lg:block">
              Setiap perubahan dicatat agar proses pembaruan data dapat ditelusuri kembali.
            </div>
          </section>

          <section className="rounded-xl border border-[#E4E7EC] bg-white p-6 shadow-[0_8px_28px_rgba(16,24,40,0.06)] sm:p-8">
            <span className="grid size-11 place-items-center rounded-lg bg-[#EAF2FC] text-[#1666D8]"><LockKeyhole size={20} /></span>
            <h2 className="mt-5 text-xl font-semibold text-[#101828]">Login stakeholder</h2>
            <p className="mt-2 text-xs leading-5 text-[#667085]">Gunakan akun admin yang telah dikonfigurasi pada server.</p>

            <form onSubmit={submit} className="mt-7 space-y-5">
              <label className="block">
                <span className="mb-2 block text-xs font-semibold text-[#344054]">Nama pengguna</span>
                <input
                  autoComplete="username"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  required
                  className="h-11 w-full rounded-lg border border-[#D0D5DD] px-3.5 text-sm text-[#101828] outline-none transition focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC]"
                />
              </label>
              <label className="block">
                <span className="mb-2 block text-xs font-semibold text-[#344054]">Password</span>
                <span className="relative block">
                  <input
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                    minLength={8}
                    className="h-11 w-full rounded-lg border border-[#D0D5DD] px-3.5 pr-11 text-sm text-[#101828] outline-none transition focus:border-[#1666D8] focus:ring-2 focus:ring-[#EAF2FC]"
                  />
                  <button
                    type="button"
                    aria-label={showPassword ? "Sembunyikan password" : "Tampilkan password"}
                    title={showPassword ? "Sembunyikan password" : "Tampilkan password"}
                    onClick={() => setShowPassword((value) => !value)}
                    className="absolute right-1.5 top-1/2 grid size-8 -translate-y-1/2 place-items-center rounded-md text-[#667085] hover:bg-[#F8FAFC] hover:text-[#101828]"
                  >
                    {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                  </button>
                </span>
              </label>

              {error && <p role="alert" className="rounded-lg bg-red-50 px-3 py-2.5 text-xs leading-5 text-[#B42318]">{error}</p>}

              <button
                type="submit"
                disabled={submitting || !username.trim() || !password}
                className="inline-flex h-11 w-full cursor-pointer items-center justify-center gap-2 rounded-lg bg-[#1666D8] px-4 text-sm font-semibold text-white transition hover:bg-[#0B54B8] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submitting ? "Memeriksa akses..." : "Login"}
                {!submitting && <ArrowRight size={17} />}
              </button>
            </form>

            <p className="mt-5 border-t border-[#E4E7EC] pt-4 text-[11px] leading-5 text-[#667085]">
              Sesi akan berakhir otomatis. Jangan membagikan akun kepada pihak yang tidak berkepentingan.
            </p>
          </section>
        </div>
      </div>
    </main>
  );
}
