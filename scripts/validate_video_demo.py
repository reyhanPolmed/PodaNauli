"""Execute and validate the PodaNauli video demo package."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "scratch" / "matplotlib_demo"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / "scratch" / "jupyter_runtime"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / "scratch" / "jupyter_config"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / "scratch" / "ipython_demo"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import joblib
import pandas as pd
from PIL import Image

from demo.demo_runtime import load_demo_bundle, predict_reviews


DEMO_DIR = ROOT / "demo"
OUTPUT_DIR = DEMO_DIR / "outputs"
REPORT_PATH = OUTPUT_DIR / "demo_validation_report.json"
NOTEBOOK_PATH = DEMO_DIR / "01_podanauli_video_demo.ipynb"
EXECUTED_NOTEBOOK_PATH = OUTPUT_DIR / "01_podanauli_video_demo_executed.ipynb"

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?62|0)[\s.-]?(?:\d[\s.-]?){8,13}(?!\d)")
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"(?:^|[\\/])Users[\\/]", re.IGNORECASE),
    re.compile(r"(?:^|/)home/", re.IGNORECASE),
]
FORBIDDEN_FIELDS = ["reviewer_id", "reviewer_name"]
INTERNET_TOKENS = ["requests.", "urllib", "http://", "https://", "!pip", "%pip", "pip install", "wget ", "curl "]


REQUIRED_FILES = [
    "demo/01_podanauli_video_demo.ipynb",
    "demo/demo_config.yaml",
    "demo/demo_reviews.csv",
    "demo/demo_metrics.json",
    "demo/demo_service_gap.csv",
    "demo/demo_data_quality.json",
    "demo/demo_place_detail.json",
    "demo/demo_runtime.py",
    "demo/requirements-demo.txt",
    "demo/figures/pipeline_overview.png",
    "demo/figures/data_transformation.png",
    "demo/figures/model_metrics.png",
    "demo/figures/sentiment_confusion_matrix.png",
    "demo/figures/complaint_confusion_matrix.png",
    "demo/figures/aspect_metrics.png",
    "demo/figures/service_gap_validation.png",
    "demo/figures/service_gap_top10.png",
    "scripts/prepare_video_demo.py",
    "scripts/validate_video_demo.py",
    "scripts/run_video_demo.ps1",
]


class Validation:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str, critical: bool = True) -> None:
        self.checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "critical": bool(critical),
                "detail": sanitize(detail),
            }
        )

    @property
    def critical_failures(self) -> list[str]:
        return [item["name"] for item in self.checks if item["critical"] and not item["passed"]]

    @property
    def warnings(self) -> list[str]:
        return [item["name"] for item in self.checks if not item["critical"] and not item["passed"]]


def sanitize(value: Any) -> str:
    text = str(value).replace(str(ROOT), "<PROJECT_ROOT>")
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        text = text.replace(user_profile, "<USER_PROFILE>")
    text = re.sub(r"[A-Za-z]:[\\/][^\r\n\"']+", "<LOCAL_PATH>", text)
    return text[:1000]


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_enough(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def check_required_files(validation: Validation) -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    validation.add(
        "required_demo_files",
        not missing,
        "Semua file wajib tersedia." if not missing else "File hilang: " + ", ".join(missing),
    )


def check_models(validation: Validation) -> dict[str, Any] | None:
    try:
        bundle = load_demo_bundle(print_status=False)
        texts = [
            "Pemandangannya sangat indah, tetapi toilet kurang bersih dan area parkir sempit.",
            "Pelayanannya ramah, tempatnya nyaman, dan makanan disajikan dengan cepat.",
            "Akses jalan rusak, petunjuk arah kurang jelas, dan kendaraan sulit mencapai lokasi.",
        ]
        output = predict_reviews(texts, bundle)
        valid = len(output) == 3 and set(output.columns) == {"Review", "Sentimen", "Complaint", "Aspek", "Catatan"}
        validation.add("champion_models_load_and_predict", valid, "Tiga model champion memproses tiga input manual.")
        return bundle
    except Exception as exc:
        validation.add("champion_models_load_and_predict", False, f"Model gagal dimuat atau diprediksi: {exc}")
        return None


def check_metrics(validation: Validation) -> None:
    try:
        demo = load_json("demo/demo_metrics.json")
        sentiment = load_json("outputs/reports/sentiment_metrics.json")
        complaint = load_json("outputs/reports/complaint_metrics.json")
        aspect = load_json("outputs/reports/aspect_metrics.json")
        readiness = load_json("outputs/reports/project_readiness.json")
        expected = {
            "sentiment.macro_f1": sentiment["champion"]["metrics"]["macro_f1"],
            "sentiment.negative_recall": sentiment["champion"]["metrics"]["negative_recall"],
            "complaint.macro_f1": complaint["test_metrics"]["macro_f1"],
            "complaint.negative_recall": complaint["test_metrics"]["negative_recall"],
            "aspect.micro_f1": aspect["test_metrics"]["micro_f1"],
            "aspect.macro_f1": aspect["test_metrics"]["macro_f1"],
            "aspect.hamming_loss": aspect["test_metrics"]["hamming_loss"],
            "aspect.subset_accuracy": aspect["test_metrics"]["subset_accuracy"],
            "service_gap_validation.evidence_validity": readiness["service_gap_validation"]["evidence_validity_rate"],
            "service_gap_validation.priority_validity": readiness["service_gap_validation"]["priority_validity_rate"],
            "service_gap_validation.overall_validity": readiness["service_gap_validation"]["validity_rate"],
        }
        actual = {
            "sentiment.macro_f1": demo["sentiment"]["macro_f1"],
            "sentiment.negative_recall": demo["sentiment"]["negative_recall"],
            "complaint.macro_f1": demo["complaint"]["macro_f1"],
            "complaint.negative_recall": demo["complaint"]["negative_recall"],
            "aspect.micro_f1": demo["aspect"]["micro_f1"],
            "aspect.macro_f1": demo["aspect"]["macro_f1"],
            "aspect.hamming_loss": demo["aspect"]["hamming_loss"],
            "aspect.subset_accuracy": demo["aspect"]["subset_accuracy"],
            "service_gap_validation.evidence_validity": demo["service_gap_validation"]["evidence_validity"],
            "service_gap_validation.priority_validity": demo["service_gap_validation"]["priority_validity"],
            "service_gap_validation.overall_validity": demo["service_gap_validation"]["overall_validity"],
        }
        mismatches = [key for key in expected if not close_enough(expected[key], actual[key])]
        validation.add(
            "metrics_match_source_artifacts",
            not mismatches,
            "Sebelas metrik demo cocok dengan artifact sumber." if not mismatches else "Metrik berbeda: " + ", ".join(mismatches),
        )
        split_ok = (
            demo["sentiment"]["group_overlap_count"] == 0
            and demo["aspect"]["group_overlap_count"] == 0
            and demo["sentiment"]["split_rows"] == sentiment["split"]["rows"]
            and demo["aspect"]["split_rows"] == aspect["split"]["rows"]
        )
        validation.add("locked_split_matches_artifacts", split_ok, "Split place-disjoint dan jumlah baris cocok dengan sumber.")
    except Exception as exc:
        validation.add("metrics_match_source_artifacts", False, f"Validasi metrik gagal: {exc}")


def check_demo_reviews(validation: Validation) -> None:
    try:
        frame = pd.read_csv(DEMO_DIR / "demo_reviews.csv")
        required = {
            "demo_id", "place_name", "review_text", "expected_sentiment", "predicted_sentiment",
            "predicted_complaint", "predicted_aspects", "is_error_example", "safe_for_video",
        }
        labels_ok = set(frame["predicted_sentiment"]).issubset({"Negatif", "Netral", "Positif"})
        complaint_ok = set(frame["predicted_complaint"]).issubset({"Terdeteksi", "Tidak terdeteksi", "Perlu tinjau"})
        safe_flags = frame["safe_for_video"].astype(str).str.lower().isin({"true", "1"}).all()
        valid = (
            required.issubset(frame.columns)
            and 5 <= len(frame) <= 10
            and frame["is_error_example"].astype(str).str.lower().isin({"true", "1"}).any()
            and safe_flags
            and labels_ok
            and complaint_ok
        )
        validation.add("safe_actual_review_examples", valid, f"{len(frame)} contoh locked test tersedia, termasuk contoh error.")
    except Exception as exc:
        validation.add("safe_actual_review_examples", False, f"Pemeriksaan demo_reviews gagal: {exc}")


def normalize_aspect(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ")


def check_ranking(validation: Validation) -> None:
    try:
        demo = pd.read_csv(DEMO_DIR / "demo_service_gap.csv").sort_values("rank").reset_index(drop=True)
        source = pd.read_csv(ROOT / "outputs" / "predictions" / "service_gap_rankings.csv").sort_values("rank").head(10).reset_index(drop=True)
        rows_match = len(demo) == 10
        if rows_match:
            for index in range(10):
                rows_match = rows_match and (
                    int(demo.loc[index, "rank"]) == int(source.loc[index, "rank"])
                    and str(demo.loc[index, "place_name"]) == str(source.loc[index, "place_name"])
                    and normalize_aspect(demo.loc[index, "aspect"]) == normalize_aspect(source.loc[index, "aspect"])
                    and close_enough(demo.loc[index, "service_gap_score"], round(float(source.loc[index, "service_gap_score"]), 2), 1e-8)
                )
        validation.add("top10_ranking_matches_source", rows_match, "Top-10, tempat, aspek, dan skor cocok dengan ranking sumber.")

        detail = load_json("demo/demo_place_detail.json")
        checked = pd.read_csv(ROOT / "outputs" / "reports" / "service_gap_top20_validation.pending.csv")
        first_check = checked[checked["rank"].astype(int).eq(1)].iloc[0]
        detail_ok = (
            int(detail["rank"]) == 1
            and str(detail["place_name"]) == str(source.loc[0, "place_name"])
            and str(first_check["manual_evidence_valid"]).strip().lower() == "yes"
            and str(first_check["manual_priority_valid"]).strip().lower() == "yes"
            and len(detail["evidence_snippets"]) >= 1
        )
        validation.add("validated_place_detail", detail_ok, "Detail ranking pertama cocok dan memiliki validasi manusia serta evidence.")
    except Exception as exc:
        validation.add("top10_ranking_matches_source", False, f"Pemeriksaan ranking gagal: {exc}")


def check_figures(validation: Validation) -> None:
    names = [
        "pipeline_overview.png", "data_transformation.png", "model_metrics.png",
        "sentiment_confusion_matrix.png", "complaint_confusion_matrix.png",
        "aspect_metrics.png", "service_gap_validation.png", "service_gap_top10.png",
    ]
    failures: list[str] = []
    dimensions: dict[str, list[int]] = {}
    for name in names:
        path = DEMO_DIR / "figures" / name
        if not path.exists():
            failures.append(name + " hilang")
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
            dimensions[name] = [width, height]
            if width < 1400 or height < 750:
                failures.append(f"{name} terlalu kecil ({width}x{height})")
        except Exception as exc:
            failures.append(f"{name} tidak dapat dibaca: {exc}")
    validation.add("video_figures_readable", not failures, "Delapan figure lolos dimensi minimum." if not failures else "; ".join(failures))


def notebook_text(notebook: dict[str, Any], include_outputs: bool) -> str:
    pieces: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        pieces.append("".join(source) if isinstance(source, list) else str(source))
        if not include_outputs:
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                value = output.get("text", "")
                pieces.append("".join(value) if isinstance(value, list) else str(value))
            elif output.get("output_type") in {"display_data", "execute_result"}:
                data = output.get("data", {})
                for mime in ["text/plain", "text/html", "text/markdown"]:
                    value = data.get(mime, "")
                    pieces.append("".join(value) if isinstance(value, list) else str(value))
            elif output.get("output_type") == "error":
                pieces.extend(output.get("traceback", []))
    return "\n".join(pieces)


def execute_notebook(validation: Validation) -> dict[str, Any]:
    result = {"runtime_seconds": None, "cell_count": 0, "code_cell_count": 0, "failed_cells": 0, "stderr_streams": 0}
    try:
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        result["cell_count"] = len(notebook.cells)
        result["code_cell_count"] = sum(cell.cell_type == "code" for cell in notebook.cells)
        started = time.perf_counter()
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
            extra_arguments=["--log-level=ERROR"],
        )
        executed = client.execute()
        result["runtime_seconds"] = round(time.perf_counter() - started, 3)
        for cell in executed.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    result["failed_cells"] += 1
                if output.get("output_type") == "stream" and output.get("name") == "stderr":
                    result["stderr_streams"] += 1
        nbformat.write(executed, EXECUTED_NOTEBOOK_PATH)
        passed = result["failed_cells"] == 0 and result["stderr_streams"] == 0
        validation.add(
            "notebook_executes_without_error",
            passed,
            f"Notebook selesai dalam {result['runtime_seconds']:.3f} detik; {result['code_cell_count']} sel kode; {result['failed_cells']} gagal; {result['stderr_streams']} stderr.",
        )
    except Exception as exc:
        validation.add("notebook_executes_without_error", False, f"Eksekusi notebook gagal: {exc}")
    return result


def scan_text_for_privacy(relative_path: str, text: str) -> list[str]:
    findings: list[str] = []
    if EMAIL_RE.search(text):
        findings.append(relative_path + ": email")
    if PHONE_RE.search(text):
        findings.append(relative_path + ": nomor telepon")
    for pattern in ABSOLUTE_PATH_PATTERNS:
        if pattern.search(text):
            findings.append(relative_path + ": path absolut")
            break
    lowered = text.lower()
    username = os.environ.get("USERNAME", "").strip().lower()
    if username and len(username) >= 3 and re.search(rf"(?<![a-z0-9_]){re.escape(username)}(?![a-z0-9_])", lowered):
        findings.append(relative_path + ": nama akun komputer")
    for field in FORBIDDEN_FIELDS:
        if field in lowered:
            findings.append(relative_path + ": field identitas reviewer")
    return findings


def check_privacy_and_offline(validation: Validation) -> None:
    findings: list[str] = []
    internet_findings: list[str] = []
    text_suffixes = {".md", ".json", ".csv", ".yaml", ".yml", ".py", ".txt"}
    for path in DEMO_DIR.rglob("*"):
        if not path.is_file() or path == REPORT_PATH or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        relative = path.relative_to(ROOT).as_posix()
        findings.extend(scan_text_for_privacy(relative, text))

    notebook_paths = [NOTEBOOK_PATH]
    if EXECUTED_NOTEBOOK_PATH.exists():
        notebook_paths.append(EXECUTED_NOTEBOOK_PATH)
    for path in notebook_paths:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            text = notebook_text(notebook, include_outputs=path == EXECUTED_NOTEBOOK_PATH)
            relative = path.relative_to(ROOT).as_posix()
            findings.extend(scan_text_for_privacy(relative, text))
            code_text = "\n".join(
                "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            ).lower()
            for token in INTERNET_TOKENS:
                if token in code_text:
                    internet_findings.append(relative + ": " + token.strip())
        except Exception as exc:
            findings.append(path.relative_to(ROOT).as_posix() + f": gagal dipindai ({exc})")
    validation.add("privacy_scan", not findings, "Tidak ada identitas, email, nomor telepon, institusi, atau path absolut." if not findings else "; ".join(sorted(set(findings))))
    validation.add("offline_notebook", not internet_findings, "Notebook tidak memuat dependency internet." if not internet_findings else "; ".join(sorted(set(internet_findings))))


def check_notebook_structure(validation: Validation) -> None:
    try:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = notebook_text(notebook, include_outputs=False)
        section_count = len(re.findall(r"^## \d+\.", sources, flags=re.MULTILINE))
        code_cells = sum(cell.get("cell_type") == "code" for cell in notebook["cells"])
        source_has_absolute = any(pattern.search(sources) for pattern in ABSOLUTE_PATH_PATTERNS)
        valid = notebook.get("nbformat") == 4 and section_count >= 17 and code_cells >= 10 and not source_has_absolute
        validation.add("notebook_structure", valid, f"Notebook memiliki {section_count} bagian dan {code_cells} sel kode dengan path relatif.")
    except Exception as exc:
        validation.add("notebook_structure", False, f"Struktur notebook gagal dibaca: {exc}")


def write_report(validation: Validation, notebook_result: dict[str, Any]) -> str:
    status = "READY" if not validation.critical_failures else "NOT_READY"
    artifact_hashes = {}
    for relative in [
        "demo/01_podanauli_video_demo.ipynb",
        "demo/demo_metrics.json",
        "demo/demo_reviews.csv",
        "demo/demo_service_gap.csv",
    ]:
        path = ROOT / relative
        if path.exists():
            artifact_hashes[relative] = sha256(path)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "summary": {
            "checks_total": len(validation.checks),
            "checks_passed": sum(item["passed"] for item in validation.checks),
            "critical_failures": validation.critical_failures,
            "warnings": validation.warnings,
        },
        "notebook_execution": notebook_result,
        "loaded_artifacts": [
            "models/sentiment_champion.joblib",
            "models/complaint_detector.joblib",
            "models/aspect_champion.joblib",
            "models/aspect_multilabel_binarizer.joblib",
            "outputs/predictions/service_gap_rankings.csv",
        ],
        "checks": validation.checks,
        "artifact_sha256": artifact_hashes,
        "privacy": {
            "reviewer_identity_in_demo": False if status == "READY" else None,
            "absolute_path_in_notebook_or_output": False if status == "READY" else None,
            "internet_required_by_notebook": False if status == "READY" else None,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_path = OUTPUT_DIR / "demo_execution_log.txt"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"notebook_runtime_seconds={notebook_result.get('runtime_seconds')}\n")
        handle.write(f"validation_status={status}\n")
    return status


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for directory in [
        ROOT / "scratch" / "jupyter_runtime",
        ROOT / "scratch" / "jupyter_config",
        ROOT / "scratch" / "ipython_demo",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    validation = Validation()
    check_required_files(validation)
    check_models(validation)
    check_metrics(validation)
    check_demo_reviews(validation)
    check_ranking(validation)
    check_figures(validation)
    check_notebook_structure(validation)
    notebook_result = execute_notebook(validation)
    check_privacy_and_offline(validation)
    status = write_report(validation, notebook_result)
    print(f"Status demo: {status}")
    print(f"Pemeriksaan: {sum(item['passed'] for item in validation.checks)}/{len(validation.checks)} lulus")
    if validation.critical_failures:
        print("Masalah kritis: " + ", ".join(validation.critical_failures))
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
