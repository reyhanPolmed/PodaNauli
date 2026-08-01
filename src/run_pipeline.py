"""Tahap 10 CLI: run TobaPulse pipeline stages end-to-end."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.data_loader import run_profile
from src.aspect_sentiment import infer_aspect_sentiment
from src.aspect_annotation import (
    prepare_aspect_annotation_queue,
    suggest_aspect_annotation_queue,
)
from src.aspect_gold_dataset import prepare_aspect_gold_dataset
from src.gap_scoring import run_gap_scoring
from src.geospatial import run_geospatial_clustering
from src.gold_comparison import compare_gold_and_weak_models
from src.gold_dataset import prepare_gold_dataset
from src.model_evaluation import run_model_evaluation
from src.run_cleaning import run_cleaning
from src.sentiment_annotation import prepare_annotation_queue, suggest_annotation_queue
from src.sentiment_labeling import run_sentiment_labeling
from src.topic_modeling import run_topic_modeling
from src.train_aspect import run_aspect_training
from src.train_complaint import train_complaint_detector
from src.train_sentiment import train_and_evaluate


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"
REPORT_DIR = ROOT / "outputs" / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

STAGE_ORDER = [
    "profile",
    "clean",
    "label-sentiment",
    "prepare-annotations",
    "prepare-gold",
    "train-sentiment",
    "train-complaint",
    "compare-gold",
    "suggest-annotations",
    "topic-modeling",
    "prepare-aspect-annotations",
    "prepare-aspect-gold",
    "train-aspect",
    "suggest-aspect-annotations",
    "aspect-sentiment",
    "geospatial",
    "gap-scoring",
    "evaluation",
    "export",
]

ALIASES = {
    "sentiment-labeling": "label-sentiment",
    "annotation": "prepare-annotations",
    "gold": "prepare-gold",
    "complaint": "train-complaint",
    "ai-annotation": "suggest-annotations",
    "aspect-annotation": "prepare-aspect-annotations",
    "aspect-gold": "prepare-aspect-gold",
    "aspect-ai-annotation": "suggest-aspect-annotations",
    "evaluate": "evaluation",
}

REQUIRED_ARTIFACTS = {
    "profile": [
        REPORT_DIR / "data_profile.json",
        REPORT_DIR / "data_quality_summary.csv",
    ],
    "clean": [
        PROCESSED_DIR / "reviews_clean.parquet",
        PROCESSED_DIR / "places_master.parquet",
        PROCESSED_DIR / "entity_mapping.parquet",
        REPORT_DIR / "cleaning_summary.json",
    ],
    "label-sentiment": [
        REPORT_DIR / "sentiment_annotation_sample.csv",
        REPORT_DIR / "sentiment_labeling_summary.json",
    ],
    "prepare-annotations": [
        ROOT / "data" / "annotations" / "sentiment_annotation_queue.csv",
        ROOT / "data" / "annotations" / "sentiment_gold.csv",
        REPORT_DIR / "sentiment_annotation_workflow.json",
    ],
    "prepare-gold": [
        REPORT_DIR / "gold_dataset_manifest.json",
    ],
    "train-sentiment": [
        MODEL_DIR / "sentiment_champion.joblib",
        MODEL_DIR / "sentiment_label_encoder.joblib",
        REPORT_DIR / "sentiment_metrics.json",
        REPORT_DIR / "sentiment_model_comparison.csv",
    ],
    "train-complaint": [
        MODEL_DIR / "complaint_detector.joblib",
        PROCESSED_DIR / "review_complaint_predictions.parquet",
        REPORT_DIR / "complaint_metrics.json",
        REPORT_DIR / "complaint_model_comparison.csv",
    ],
    "compare-gold": [
        REPORT_DIR / "gold_vs_weak_model_comparison.json",
        REPORT_DIR / "gold_vs_weak_model_comparison.csv",
    ],
    "suggest-annotations": [
        ROOT / "data" / "annotations" / "sentiment_annotation_queue.csv",
        REPORT_DIR / "ai_annotation_summary.json",
    ],
    "topic-modeling": [
        REPORT_DIR / "topic_summary.csv",
        REPORT_DIR / "topic_keywords.csv",
        PROCESSED_DIR / "review_topics.parquet",
    ],
    "prepare-aspect-annotations": [
        ROOT / "data" / "annotations" / "aspect_clause_annotation_queue.csv",
        ROOT / "data" / "annotations" / "aspect_gold.csv",
        REPORT_DIR / "aspect_annotation_workflow.json",
    ],
    "prepare-aspect-gold": [
        REPORT_DIR / "aspect_gold_manifest.json",
    ],
    "train-aspect": [
        MODEL_DIR / "aspect_champion.joblib",
        MODEL_DIR / "aspect_multilabel_binarizer.joblib",
        MODEL_DIR / "aspect_metadata.json",
        PROCESSED_DIR / "review_aspect_predictions.parquet",
        REPORT_DIR / "aspect_metrics.json",
    ],
    "suggest-aspect-annotations": [
        ROOT / "data" / "annotations" / "aspect_clause_annotation_queue.csv",
        REPORT_DIR / "aspect_ai_annotation_summary.json",
    ],
    "aspect-sentiment": [
        PROCESSED_DIR / "review_aspect_sentiment.parquet",
        REPORT_DIR / "aspect_sentiment_summary.json",
    ],
    "geospatial": [
        PROCESSED_DIR / "place_clusters.parquet",
        REPORT_DIR / "geospatial_clustering_summary.json",
        ROOT / "outputs" / "maps" / "place_clusters.geojson",
    ],
    "gap-scoring": [
        PROCESSED_DIR / "service_gap_scores.parquet",
        ROOT / "outputs" / "predictions" / "service_gap_rankings.csv",
        REPORT_DIR / "service_gap_methodology.json",
    ],
    "evaluation": [
        REPORT_DIR / "model_evaluation_summary.json",
        REPORT_DIR / "sentiment_error_analysis.csv",
        REPORT_DIR / "project_readiness.json",
        MODEL_DIR / "model_registry.json",
    ],
    "export": [
        REPORT_DIR / "pipeline_export_summary.json",
    ],
}


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        ],
    )


def check_dependencies() -> dict[str, bool]:
    modules = ["pandas", "numpy", "openpyxl", "pyarrow", "sklearn", "rapidfuzz", "matplotlib", "yaml", "joblib"]
    status = {}
    for module in modules:
        try:
            __import__(module)
            status[module] = True
        except ImportError:
            status[module] = False
    missing = [module for module, ok in status.items() if not ok]
    if missing:
        raise RuntimeError(f"Missing required dependencies: {missing}")
    return status


def validate_artifacts(stage: str) -> None:
    missing = [path for path in REQUIRED_ARTIFACTS.get(stage, []) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage {stage} did not produce required artifacts: {[str(path) for path in missing]}")


def run_stage(stage: str) -> dict:
    stage = ALIASES.get(stage, stage)
    logging.info("Starting stage: %s", stage)
    start = time.perf_counter()
    runners: dict[str, Callable[[], dict]] = {
        "profile": run_profile,
        "clean": run_cleaning,
        "label-sentiment": run_sentiment_labeling,
        "prepare-annotations": prepare_annotation_queue,
        "prepare-gold": prepare_gold_dataset,
        "train-sentiment": train_and_evaluate,
        "train-complaint": train_complaint_detector,
        "compare-gold": compare_gold_and_weak_models,
        "suggest-annotations": suggest_annotation_queue,
        "topic-modeling": run_topic_modeling,
        "prepare-aspect-annotations": prepare_aspect_annotation_queue,
        "prepare-aspect-gold": prepare_aspect_gold_dataset,
        "train-aspect": run_aspect_training,
        "suggest-aspect-annotations": suggest_aspect_annotation_queue,
        "aspect-sentiment": infer_aspect_sentiment,
        "geospatial": run_geospatial_clustering,
        "gap-scoring": run_gap_scoring,
        "evaluation": run_model_evaluation,
        "export": run_export_summary,
    }
    if stage not in runners:
        raise ValueError(f"Unknown stage: {stage}")
    result = runners[stage]()
    validate_artifacts(stage)
    elapsed = time.perf_counter() - start
    logging.info("Finished stage: %s in %.2fs", stage, elapsed)
    return {"stage": stage, "elapsed_seconds": elapsed, "result": result}


def run_export_summary() -> dict:
    """Collect an artifact index for API/dashboard stages to consume later."""
    artifacts = {}
    for stage, paths in REQUIRED_ARTIFACTS.items():
        artifacts[stage] = [
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for path in paths
        ]
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "pipeline_stage_count": len(STAGE_ORDER),
        "artifacts": artifacts,
        "dashboard_data_ready": all(item["exists"] for item in artifacts["gap-scoring"]),
        "api_data_ready": all(item["exists"] for item in artifacts["evaluation"]),
        "model_and_ranking_pipeline_ready": (
            json.loads((REPORT_DIR / "project_readiness.json").read_text(encoding="utf-8"))
            .get("model_and_ranking_pipeline_ready", False)
            if (REPORT_DIR / "project_readiness.json").exists()
            else False
        ),
        "production_application_ready": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "pipeline_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_tests() -> dict:
    start = time.perf_counter()
    completed = subprocess.run(["pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    return {
        "command": "pytest -q",
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - start,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def resolve_stages(stage: str) -> list[str]:
    stage = ALIASES.get(stage, stage)
    if stage == "all":
        return STAGE_ORDER.copy()
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {stage}. Valid stages: {STAGE_ORDER + ['all']}")
    return [stage]


def run_pipeline(stage: str, run_tests_after: bool = False) -> dict:
    check_dependencies()
    stages = resolve_stages(stage)
    runs = [run_stage(item) for item in stages]
    test_result = run_tests() if run_tests_after else None
    if test_result and test_result["returncode"] != 0:
        raise RuntimeError(f"Tests failed after pipeline run:\n{test_result['stdout']}\n{test_result['stderr']}")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "requested_stage": stage,
        "stages_run": [run["stage"] for run in runs],
        "runs": runs,
        "tests": test_result,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "pipeline_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TobaPulse pipeline stages.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGE_ORDER + ["all", *ALIASES.keys()],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--run-tests", action="store_true", help="Run pytest after requested stage(s).")
    args = parser.parse_args()
    setup_logging()
    summary = run_pipeline(args.stage, run_tests_after=args.run_tests)
    print(
        json.dumps(
            {
                "requested_stage": summary["requested_stage"],
                "stages_run": summary["stages_run"],
                "summary_path": str(REPORT_DIR / "pipeline_run_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
