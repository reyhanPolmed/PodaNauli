import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { LoadingState } from "./components/UI";

const RegionOverviewPage = lazy(() => import("./pages/RegionOverviewPage").then((module) => ({ default: module.RegionOverviewPage })));
const ServiceGapRankingPage = lazy(() => import("./pages/ServiceGapRankingPage").then((module) => ({ default: module.ServiceGapRankingPage })));
const PlacePage = lazy(() => import("./pages/PlacePage").then((module) => ({ default: module.PlacePage })));

export default function App() {
  return (
    <Suspense fallback={<LoadingState label="Memuat halaman" />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/ringkasan-wilayah" replace />} />
          <Route path="ringkasan-wilayah" element={<RegionOverviewPage />} />
          <Route path="service-gap-ranking" element={<ServiceGapRankingPage />} />
          <Route path="destinasi/:placeId" element={<PlacePage />} />
          <Route path="*" element={<Navigate to="/ringkasan-wilayah" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
