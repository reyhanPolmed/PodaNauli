import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, type ReactNode, useContext, useEffect } from "react";
import { api } from "../lib/api";
import type { AuthStatus, AuthUser } from "../types/api";

interface AuthContextValue {
  user: AuthUser | null;
  isAdmin: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const signedOut: AuthStatus = { authenticated: false, user: null };
const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const auth = useQuery({
    queryKey: ["auth", "session"],
    queryFn: api.authMe,
    retry: false,
    staleTime: 5 * 60_000,
  });
  const user = auth.data?.authenticated ? auth.data.user : null;

  useEffect(() => {
    const signedOutByApi = () => queryClient.setQueryData(["auth", "session"], signedOut);
    window.addEventListener("podanauli:unauthorized", signedOutByApi);
    return () => window.removeEventListener("podanauli:unauthorized", signedOutByApi);
  }, [queryClient]);

  async function login(username: string, password: string) {
    const status = await api.login(username, password);
    queryClient.setQueryData(["auth", "session"], status);
  }

  async function logout() {
    try {
      await api.logout();
    } finally {
      queryClient.setQueryData(["auth", "session"], signedOut);
      queryClient.removeQueries({ queryKey: ["data-imports"] });
      queryClient.removeQueries({ queryKey: ["data-import"] });
    }
  }

  return (
    <AuthContext.Provider value={{
      user,
      isAdmin: user?.role === "admin",
      isLoading: auth.isLoading,
      login,
      logout,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth harus digunakan di dalam AuthProvider.");
  return context;
}
