import { Navigate, useLocation } from "react-router-dom";
import { LoadingState } from "../components/UI";
import { useAuth } from "./AuthContext";

export function RequireAdmin({ children }: { children: React.ReactNode }) {
  const auth = useAuth();
  const location = useLocation();

  if (auth.isLoading) return <LoadingState label="Memeriksa akses stakeholder" />;
  if (!auth.isAdmin) {
    const next = `${location.pathname}${location.search}`;
    return <Navigate to={`/login?next=${encodeURIComponent(next)}`} replace />;
  }
  return children;
}
