const CLUSTER_AREA_NAMES: Record<number, string> = {
  0: "Pangururan - Sianjur Mula-mula",
  1: "Bakti Raja - Tipang",
  2: "Silalahi - Paropo",
  3: "Balige dan Sekitarnya",
  4: "Tuktuk - Tomok",
  5: "Parapat - Ajibata",
  6: "Porsea - Sigumpar - Silaen",
  7: "Tigaras - Simarjarunjung",
  8: "Tarutung dan Sekitarnya",
};

export function clusterAreaName(clusterId: number | null | undefined): string {
  if (clusterId === null || clusterId === undefined) return "Koordinat belum tersedia";
  return CLUSTER_AREA_NAMES[clusterId] ?? `Klaster lokasi ${clusterId + 1}`;
}
