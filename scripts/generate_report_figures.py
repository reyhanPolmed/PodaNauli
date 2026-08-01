"""Generate report-only figures from verified TobaPulse artifacts.

This script does not train models or mutate source datasets. It reads the
locked evaluation, ranking, validation, and geospatial outputs produced by the
pipeline and writes presentation-ready PNG files for the LaTeX report.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "scratch" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures" / "report"
MAP_DIR = ROOT / "outputs" / "maps"
PREDICTION_DIR = ROOT / "outputs" / "predictions"
PROCESSED_DIR = ROOT / "data" / "processed"

NAVY = "#17324D"
BLUE = "#2D6A9F"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
ORANGE = "#F4A261"
RED = "#C94C4C"
LIGHT = "#E9EEF3"
GRAY = "#667788"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def decimal_comma(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(axis=grid_axis, color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8B4BF")
    ax.spines["bottom"].set_color("#A8B4BF")
    ax.tick_params(colors=NAVY, labelsize=9)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / name,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "TobaPulse report figure generator"},
    )
    plt.close(fig)


def model_metrics_summary() -> None:
    sentiment = load_json(REPORT_DIR / "sentiment_metrics.json")
    complaint = load_json(REPORT_DIR / "complaint_metrics.json")
    aspect = load_json(REPORT_DIR / "aspect_metrics.json")

    rows = [
        ("Sentiment", "Macro F1", sentiment["champion"]["metrics"]["macro_f1"]),
        ("Sentiment", "Recall negatif", sentiment["champion"]["metrics"]["negative_recall"]),
        ("Complaint", "Macro F1", complaint["test_metrics"]["macro_f1"]),
        ("Complaint", "Recall negatif", complaint["test_metrics"]["negative_recall"]),
        ("Aspect", "Micro F1", aspect["test_metrics"]["micro_f1"]),
        ("Aspect", "Macro F1", aspect["test_metrics"]["macro_f1"]),
    ]
    frame = pd.DataFrame(rows, columns=["model", "metric", "value"])

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(3)
    width = 0.32
    first = frame.groupby("model", sort=False).nth(0)["value"].to_numpy()
    second = frame.groupby("model", sort=False).nth(1)["value"].to_numpy()
    bars_a = ax.bar(x - width / 2, first, width, color=BLUE, label="Metrik utama 1")
    bars_b = ax.bar(x + width / 2, second, width, color=TEAL, label="Metrik utama 2")

    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.014,
                decimal_comma(bar.get_height(), 4),
                ha="center",
                va="bottom",
                fontsize=9,
                color=NAVY,
                fontweight="bold",
            )

    ax.set_xticks(x, ["Sentiment", "Complaint", "Aspect"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Skor pada locked test", color=NAVY)
    ax.set_title("Ringkasan Evaluasi Model pada Human Gold", color=NAVY, fontsize=15, fontweight="bold")
    ax.legend(
        [bars_a, bars_b],
        ["Macro F1 / Micro F1", "Recall negatif / Macro F1"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
    )
    style_axis(ax)
    save_figure(fig, "model_metrics_summary.png")


def eda_data_funnel() -> None:
    cleaning = load_json(REPORT_DIR / "cleaning_summary.json")
    topic = load_json(REPORT_DIR / "topic_modeling_summary.json")
    sentiment_gold = load_json(REPORT_DIR / "gold_dataset_manifest.json")
    geospatial = load_json(REPORT_DIR / "geospatial_clustering_summary.json")

    review_labels = [
        "Baris ulasan\nterintegrasi",
        "Ulasan\nberteks",
        "Korpus NLP\nnonduplikat",
        "Gold\nsentiment",
    ]
    review_values = [
        int(cleaning["review_rows"]),
        int(cleaning["review_rows_with_text"]),
        int(topic["corpus_rows"]),
        int(sentiment_gold["validation"]["rows"]),
    ]
    place_labels = ["Baris sumber\ntempat", "Tempat\nkanonis", "Koordinat\nvalid"]
    place_values = [
        int(cleaning["source_place_rows"]),
        int(cleaning["canonical_place_count"]),
        int(geospatial["valid_coordinate_places"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), gridspec_kw={"width_ratios": [1.35, 1]})
    review_bars = axes[0].bar(review_labels, review_values, color=[NAVY, BLUE, TEAL, GOLD], width=0.68)
    place_bars = axes[1].bar(place_labels, place_values, color=[NAVY, BLUE, TEAL], width=0.62)

    for ax, bars in ((axes[0], review_bars), (axes[1], place_bars)):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{int(bar.get_height()):,}".replace(",", "."),
                ha="center",
                va="bottom",
                fontsize=9,
                color=NAVY,
                fontweight="bold",
            )
        style_axis(ax)

    axes[0].set_ylim(0, max(review_values) * 1.15)
    axes[0].set_ylabel("Jumlah baris/ulasan", color=NAVY)
    axes[0].set_title("Funnel Ulasan", color=NAVY, fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, max(place_values) * 1.18)
    axes[1].set_ylabel("Jumlah entitas", color=NAVY)
    axes[1].set_title("Integrasi Tempat", color=NAVY, fontsize=13, fontweight="bold")
    fig.suptitle("Transformasi Data dari Sumber ke Korpus Analitik", color=NAVY, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "eda_data_funnel.png")


def eda_gold_distributions() -> None:
    sentiment_gold = load_json(REPORT_DIR / "gold_dataset_manifest.json")
    aspect_gold = load_json(REPORT_DIR / "aspect_gold_manifest.json")

    sentiment_counts = sentiment_gold["validation"]["label_counts"]
    sentiment_order = ["negative", "neutral", "positive"]
    sentiment_labels = ["Negatif", "Netral", "Positif"]
    sentiment_values = [int(sentiment_counts[label]) for label in sentiment_order]

    aspect_counts = aspect_gold["validation"]["label_counts"]
    aspect_frame = pd.DataFrame(
        [
            {"aspect": key.replace("_", " ").title(), "count": int(value)}
            for key, value in aspect_counts.items()
        ]
    ).sort_values("count", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8.2), gridspec_kw={"width_ratios": [0.8, 1.6]})
    sentiment_bars = axes[0].bar(sentiment_labels, sentiment_values, color=[RED, GOLD, TEAL], width=0.62)
    for bar in sentiment_bars:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(int(bar.get_height())),
            ha="center",
            color=NAVY,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(sentiment_values) * 1.18)
    axes[0].set_ylabel("Jumlah ulasan", color=NAVY)
    axes[0].set_title("Gold Sentiment (n=900)", color=NAVY, fontsize=13, fontweight="bold")
    style_axis(axes[0])

    aspect_colors = [GOLD if label == "None" else BLUE for label in aspect_frame["aspect"]]
    aspect_bars = axes[1].barh(aspect_frame["aspect"], aspect_frame["count"], color=aspect_colors)
    for bar in aspect_bars:
        axes[1].text(
            bar.get_width() + 4,
            bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())),
            va="center",
            fontsize=8,
            color=NAVY,
        )
    axes[1].set_xlim(0, max(aspect_frame["count"]) * 1.15)
    axes[1].set_xlabel("Jumlah label (multi-label)", color=NAVY)
    axes[1].set_title("Gold Aspect per Label (1.200 klausa)", color=NAVY, fontsize=13, fontweight="bold")
    style_axis(axes[1], grid_axis="x")

    fig.suptitle("Distribusi Human Gold untuk Evaluasi Model", color=NAVY, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "eda_gold_distributions.png")


def eda_coordinate_coverage() -> None:
    places = pd.read_parquet(PROCESSED_DIR / "places_master.parquet")
    places["valid_coordinate"] = (
        places["coordinate_parsing_success"].fillna(False)
        & places["latitude"].notna()
        & places["longitude"].notna()
    )
    frame = (
        places.groupby("place_category", dropna=False)
        .agg(total=("canonical_place_id", "size"), valid=("valid_coordinate", "sum"))
        .reset_index()
    )
    frame["missing"] = frame["total"] - frame["valid"]
    frame["coverage"] = frame["valid"] / frame["total"]
    frame["label"] = frame["place_category"].astype(str).str.replace("_", "-", regex=False).str.title()

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    x = np.arange(len(frame))
    valid_bars = ax.bar(x, frame["valid"], color=TEAL, width=0.62, label="Koordinat valid")
    missing_bars = ax.bar(
        x,
        frame["missing"],
        bottom=frame["valid"],
        color=RED,
        width=0.62,
        label="Kosong/tidak valid",
    )
    for index, (valid_bar, missing_bar) in enumerate(zip(valid_bars, missing_bars)):
        total = int(frame.iloc[index]["total"])
        coverage = float(frame.iloc[index]["coverage"])
        ax.text(
            valid_bar.get_x() + valid_bar.get_width() / 2,
            total + 4,
            f"{decimal_comma(coverage * 100, 1)}%\n({int(valid_bar.get_height())}/{total})",
            ha="center",
            va="bottom",
            color=NAVY,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x, frame["label"])
    ax.set_ylim(0, max(frame["total"]) * 1.22)
    ax.set_ylabel("Jumlah tempat", color=NAVY)
    ax.set_title("Kelengkapan Metadata Koordinat per Kategori", color=NAVY, fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax)
    save_figure(fig, "eda_coordinate_coverage.png")


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    scale = 2**zoom
    lat_rad = math.radians(max(min(lat, 85.05112878), -85.05112878))
    x = (lon + 180.0) / 360.0 * scale
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return x, y


def tile_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = 2**zoom
    lon = x / scale * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / scale))))
    return lon, lat


def osm_basemap(bounds: tuple[float, float, float, float], zoom: int = 9) -> tuple[Image.Image, list[float]]:
    west, south, east, north = bounds
    x_left, y_bottom = lonlat_to_tile(west, south, zoom)
    x_right, y_top = lonlat_to_tile(east, north, zoom)
    x_min, x_max = math.floor(x_left), math.floor(x_right)
    y_min, y_max = math.floor(y_top), math.floor(y_bottom)
    cache = ROOT / "scratch" / "osm_tiles"
    cache.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new("RGB", ((x_max - x_min + 1) * 256, (y_max - y_min + 1) * 256), "white")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "TobaPulse-report-map/1.0")]
    for tile_x in range(x_min, x_max + 1):
        for tile_y in range(y_min, y_max + 1):
            path = cache / f"{zoom}_{tile_x}_{tile_y}.png"
            if not path.exists():
                url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
                with opener.open(url, timeout=30) as response:
                    path.write_bytes(response.read())
            tile = Image.open(path).convert("RGB")
            mosaic.paste(tile, ((tile_x - x_min) * 256, (tile_y - y_min) * 256))
    left, top = tile_to_lonlat(x_min, y_min, zoom)
    right, bottom = tile_to_lonlat(x_max + 1, y_max + 1, zoom)
    return mosaic, [left, right, bottom, top]


def eda_coordinate_map() -> None:
    places = pd.read_parquet(PROCESSED_DIR / "places_master.parquet")
    valid = places[
        places["coordinate_parsing_success"].fillna(False)
        & places["latitude"].notna()
        & places["longitude"].notna()
    ].copy()
    bounds = (
        float(valid["longitude"].min()) - 0.08,
        float(valid["latitude"].min()) - 0.08,
        float(valid["longitude"].max()) + 0.08,
        float(valid["latitude"].max()) + 0.08,
    )
    basemap, extent = osm_basemap(bounds)
    colors = {"wisata": TEAL, "restoran": ORANGE, "hotel": BLUE, "hotel_resto": RED}
    labels = {"wisata": "Wisata", "restoran": "Restoran", "hotel": "Hotel", "hotel_resto": "Hotel-resto"}

    fig, ax = plt.subplots(figsize=(9.4, 8.2))
    ax.imshow(basemap, extent=extent, origin="upper", alpha=0.88, aspect="auto")
    for category, group in valid.groupby("place_category"):
        ax.scatter(
            group["longitude"],
            group["latitude"],
            s=28,
            color=colors.get(str(category), GRAY),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            label=f"{labels.get(str(category), str(category))} (n={len(group)})",
        )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xlabel("Bujur", color=NAVY)
    ax.set_ylabel("Lintang", color=NAVY)
    ax.set_title("Peta Sebaran 315 Tempat dengan Koordinat Valid", color=NAVY, fontsize=15, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower left")
    ax.text(
        0.995,
        0.008,
        "Basemap: (C) OpenStreetMap contributors",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=NAVY,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
    )
    style_axis(ax, grid_axis="both")
    save_figure(fig, "eda_coordinate_map.png")


def aspect_f1_by_class() -> None:
    aspect = load_json(REPORT_DIR / "aspect_metrics.json")
    per_aspect = aspect["test_metrics"]["per_aspect"]
    frame = pd.DataFrame(
        [
            {"aspect": key.replace("_", " ").title(), "f1": value["f1"], "support": value["support"]}
            for key, value in per_aspect.items()
        ]
    ).sort_values(["f1", "support"], ascending=True)

    colors = [RED if value < 0.55 else ORANGE if value < 0.70 else TEAL for value in frame["f1"]]
    fig, ax = plt.subplots(figsize=(10.2, 8.2))
    bars = ax.barh(frame["aspect"], frame["f1"], color=colors)
    for bar, support in zip(bars, frame["support"]):
        ax.text(
            min(bar.get_width() + 0.015, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{decimal_comma(bar.get_width(), 3)}  (n={int(support)})",
            va="center",
            fontsize=8.5,
            color=NAVY,
        )
    ax.axvline(0.50, color=RED, linewidth=1, linestyle="--", label="Batas F1 aspek kunci")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 pada locked test", color=NAVY)
    ax.set_title("F1 per Aspek dan Dukungan Kelas", color=NAVY, fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    style_axis(ax, grid_axis="x")
    save_figure(fig, "aspect_f1_by_class.png")


def service_gap_top10() -> None:
    frame = pd.read_csv(PREDICTION_DIR / "service_gap_rankings.csv").head(10).copy()
    frame["label"] = (
        frame["place_name"].str.slice(0, 30)
        + " | "
        + frame["aspect"].str.replace("_", " ", regex=False)
    )
    frame = frame.sort_values("service_gap_score", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(frame["label"], frame["service_gap_score"], color=BLUE)
    for bar in bars:
        ax.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            decimal_comma(bar.get_width(), 2),
            va="center",
            fontsize=8.5,
            color=NAVY,
            fontweight="bold",
        )
    ax.set_xlim(0, max(frame["service_gap_score"]) + 8)
    ax.set_xlabel("Service Gap Score (0-100)", color=NAVY)
    ax.set_title("Sepuluh Prioritas Service Gap Tertinggi", color=NAVY, fontsize=15, fontweight="bold")
    style_axis(ax, grid_axis="x")
    save_figure(fig, "service_gap_top10.png")


def service_gap_validation() -> None:
    pending = REPORT_DIR / "service_gap_top20_validation.pending.csv"
    primary = REPORT_DIR / "service_gap_top20_validation.csv"
    path = pending if pending.exists() else primary
    frame = pd.read_csv(path)
    reviewed = len(frame)
    evidence = frame["manual_evidence_valid"].astype(str).str.lower().eq("yes").sum()
    priority = frame["manual_priority_valid"].astype(str).str.lower().eq("yes").sum()
    both = (
        frame["manual_evidence_valid"].astype(str).str.lower().eq("yes")
        & frame["manual_priority_valid"].astype(str).str.lower().eq("yes")
    ).sum()
    values = np.array([evidence, priority, both], dtype=float) / max(reviewed, 1)
    labels = ["Validitas bukti", "Validitas prioritas", "Validitas keseluruhan"]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(labels, values, color=[BLUE, TEAL, GOLD], width=0.58)
    ax.axhline(0.80, color=RED, linestyle="--", linewidth=1.5, label="Acceptance gate 0,80")
    for bar, count in zip(bars, [evidence, priority, both]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{decimal_comma(bar.get_height(), 2)} ({count}/{reviewed})",
            ha="center",
            va="bottom",
            color=NAVY,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Proporsi valid", color=NAVY)
    ax.set_title("Validasi Manusia atas 20 Ranking Teratas", color=NAVY, fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    style_axis(ax)
    save_figure(fig, "service_gap_validation.png")


def geospatial_clusters() -> None:
    geojson = load_json(MAP_DIR / "place_clusters.geojson")
    points = []
    for feature in geojson["features"]:
        if feature.get("geometry", {}).get("type") != "Point":
            continue
        lon, lat = feature["geometry"]["coordinates"]
        props = feature.get("properties", {})
        points.append(
            {
                "longitude": lon,
                "latitude": lat,
                "cluster": int(props.get("geo_cluster_id", -1)),
                "noise": bool(props.get("is_noise", False)),
            }
        )
    frame = pd.DataFrame(points)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    palette = [BLUE, TEAL, GOLD, ORANGE, "#7A5195", "#4D908E", "#577590", "#F94144", "#90BE6D"]
    for cluster_id, group in frame.groupby("cluster"):
        if cluster_id == -1:
            ax.scatter(
                group["longitude"],
                group["latitude"],
                s=14,
                color="#A8B4BF",
                alpha=0.7,
                label="Noise",
            )
        else:
            ax.scatter(
                group["longitude"],
                group["latitude"],
                s=22,
                color=palette[cluster_id % len(palette)],
                alpha=0.82,
                label=f"Klaster {cluster_id}",
            )
    ax.set_xlabel("Bujur", color=NAVY)
    ax.set_ylabel("Lintang", color=NAVY)
    ax.set_title("Klaster Spasial Lokasi dengan Koordinat Valid", color=NAVY, fontsize=15, fontweight="bold")
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="best")
    style_axis(ax, grid_axis="both")
    save_figure(fig, "geospatial_clusters.png")


def pipeline_lineage() -> None:
    stages = [
        ("Excel mentah", "14 sheet", NAVY),
        ("Profiling & cleaning", "22.407 ulasan", BLUE),
        ("Human gold", "900 ulasan\n1.200 klausa", TEAL),
        ("Model NLP", "sentiment\ncomplaint\naspect", ORANGE),
        ("Analitik konteks", "aspect-sentiment\nDBSCAN", "#7A5195"),
        ("Prioritas", "Service Gap\nTop-20 tervalidasi", RED),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, len(stages) * 2.0)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for index, (title, subtitle, color) in enumerate(stages):
        x = 0.25 + index * 2.0
        box = FancyBboxPatch(
            (x, 0.85),
            1.55,
            1.35,
            boxstyle="round,pad=0.035,rounding_size=0.06",
            linewidth=1.3,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + 0.775, 1.72, title, ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)
        ax.text(x + 0.775, 1.20, subtitle, ha="center", va="center", color=NAVY, fontsize=8.5)
        if index < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + 1.58, 1.52),
                (x + 1.97, 1.52),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color=GRAY,
            )
            ax.add_patch(arrow)
    ax.text(
        0.25,
        2.72,
        "Lineage Transformasi Dataset menjadi Prioritas Layanan",
        color=NAVY,
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, "pipeline_lineage.png")


def main() -> None:
    eda_data_funnel()
    eda_gold_distributions()
    eda_coordinate_coverage()
    eda_coordinate_map()
    model_metrics_summary()
    aspect_f1_by_class()
    service_gap_top10()
    service_gap_validation()
    geospatial_clusters()
    pipeline_lineage()
    print(f"Generated 10 report figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
