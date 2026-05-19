#!/usr/bin/env python3
"""Professional multi-tab frontend for THRP and the four classical baselines."""
from __future__ import annotations

from pathlib import Path
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
UPLOAD_ROOT = OUTPUTS / "frontend_uploads"
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


METHODS = [
    {
        "key": "thrp",
        "title": "THRP",
        "subtitle": "Two-stage aircraft recognition using the project pipeline",
        "accent": "#7c3aed",
        "command": ["scripts/predict.py", "--visualize"],
    },
    {
        "key": "decision_tree",
        "title": "Decision Tree",
        "subtitle": "A simple baseline that splits on handcrafted features",
        "accent": "#2563eb",
        "command": ["scripts/predict_auto_decision_tree.py"],
    },
    {
        "key": "knn",
        "title": "K-Nearest Neighbors",
        "subtitle": "Predicts from nearby examples in feature space",
        "accent": "#0f766e",
        "command": ["scripts/predict_auto_knn.py"],
    },
    {
        "key": "bayes",
        "title": "Naive Bayes",
        "subtitle": "A fast probabilistic baseline for comparison",
        "accent": "#ea580c",
        "command": ["scripts/predict_auto_bayes.py"],
    },
    {
        "key": "svm",
        "title": "Support Vector Machine",
        "subtitle": "Strong classical baseline on the report data",
        "accent": "#7c2d12",
        "command": ["scripts/predict_auto_svm.py"],
    },
]


st.set_page_config(page_title="THRP Aircraft Studio", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #081120 0%, #0f172a 40%, #f8fafc 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1360px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.88));
            color: white;
            padding: 1.75rem 1.8rem;
            border-radius: 26px;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 24px 70px rgba(2,6,23,0.35);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.2rem;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .hero p {
            margin: 0.55rem 0 0 0;
            color: rgba(226,232,240,0.92);
            font-size: 1.01rem;
            line-height: 1.6;
        }
        .pill-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .pill {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.14);
            color: #e2e8f0;
            font-size: 0.83rem;
        }
        .metric-card {
            background: white;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 12px 30px rgba(15,23,42,0.08);
            min-height: 86px;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.18rem;
            font-weight: 700;
            color: #0f172a;
        }
        .section-card {
            background: rgba(255,255,255,0.96);
            border-radius: 22px;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 14px 35px rgba(15,23,42,0.08);
        }
        .method-title {
            font-size: 1.4rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            color: #0f172a;
        }
        .method-subtitle {
            color: #475569;
            margin-bottom: 1rem;
        }
        .command-box {
            background: #0f172a;
            color: #e2e8f0;
            padding: 0.9rem 1rem;
            border-radius: 14px;
            font-size: 0.88rem;
            overflow-x: auto;
            border: 1px solid rgba(148,163,184,0.22);
        }
        .result-box {
            background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(241,245,249,0.96));
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(148,163,184,0.20);
        }
        .prediction-lines {
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 14px;
            padding: 1rem 1.05rem;
            font-size: 1rem;
            line-height: 1.8;
            white-space: pre-wrap;
            margin: 0.6rem 0 1rem 0;
        }
        .small-note {
            color: #64748b;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>THRP Aircraft Studio</h1>
            <p>
                Upload an aircraft image and run it through the THRP pipeline or one of the classical baselines.
                Each tab runs the project code directly and returns the prediction, confidence, and a compact
                result summary.
            </p>
            <div class="pill-row">
                <span class="pill">5 tabs</span>
                <span class="pill">Upload image and run</span>
                <span class="pill">THRP + Decision Tree + KNN + Naive Bayes + SVM</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics() -> None:
    cols = st.columns(4)
    values = ["THRP pipeline", "Decision Tree / KNN / Bayes / SVM", "Local execution", "Prediction preview"]
    labels = ["Demo", "Methods", "Mode", "Output"]
    for col, label, value in zip(cols, labels, values):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def save_upload(uploaded_file, method_key: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    target_dir = UPLOAD_ROOT / method_key
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_specialist_preview(image_path: Path, bbox: List[Any], method_key: str) -> Optional[Path]:
    try:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            x1, y1, x2, y2 = [int(float(piece)) for piece in bbox]
            x1 = max(0, min(x1, image.width))
            y1 = max(0, min(y1, image.height))
            x2 = max(x1 + 1, min(x2, image.width))
            y2 = max(y1 + 1, min(y2, image.height))
            roi = image.crop((x1, y1, x2, y2))

            side = max(roi.width, roi.height)
            canvas = Image.new("RGB", (side, side), (10, 15, 30))
            offset_x = (side - roi.width) // 2
            offset_y = (side - roi.height) // 2
            canvas.paste(roi, (offset_x, offset_y))

            preview_dir = OUTPUTS / "frontend_previews" / method_key
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"specialist_{image_path.stem}.png"
            canvas.save(preview_path)
            return preview_path
    except Exception:
        return None


def run_command(command: List[str], cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def parse_baseline_output(stdout: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return ast.literal_eval(line)
        except Exception:
            continue
    raise ValueError("Could not parse baseline prediction output")


def parse_thrp_output(stdout: str) -> Dict[str, Any]:
    detections: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"SuperClass: (?P<superclass>.+?) \((?P<super_conf>[\d.]+)%\)\s+"
        r"Aircraft: (?P<aircraft>.+?) \((?P<aircraft_conf>[\d.]+)%\)\s+"
        r"Combined: (?P<combined>[\d.]+)%\s+"
        r"BBox: (?P<bbox>.+?)(?:\n|$)",
        re.S,
    )

    def parse_bbox(text: str) -> List[int]:
        explicit = re.findall(r"np\.int64\(([-\d]+)\)", text)
        if explicit:
            return [int(piece) for piece in explicit]
        fallback = re.findall(r"[-]?\d+", text)
        return [int(piece) for piece in fallback]

    for match in pattern.finditer(stdout):
        bbox_text = match.group("bbox").strip()
        bbox_vals = parse_bbox(bbox_text)
        detections.append(
            {
                "superclass": match.group("superclass").strip(),
                "superclass_confidence": float(match.group("super_conf")) / 100.0,
                "aircraft_model": match.group("aircraft").strip(),
                "aircraft_confidence": float(match.group("aircraft_conf")) / 100.0,
                "combined_confidence": float(match.group("combined")) / 100.0,
                "bbox": bbox_vals,
            }
        )

    if detections:
        detections.sort(key=lambda item: item["combined_confidence"], reverse=True)
        best = detections[0]
        return {
            "success": True,
            "results": detections,
            "top_result": best,
        }
    return {"success": False, "results": []}


def render_prediction_summary(method_key: str, result: Dict[str, Any], preview_path: Optional[Path] = None) -> None:
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    if method_key == "thrp":
        top = result.get("top_result") or (result.get("results") or [None])[0]
        if top:
            prediction_text = (
                f"SuperClass: {top['superclass']} ({top['superclass_confidence']:.2%})\n"
                    f"Aircraft: {top['aircraft_model']} ({top['aircraft_confidence']:.2%})"
            )
            st.markdown(f"<div class='prediction-lines'>{prediction_text}</div>", unsafe_allow_html=True)
            st.caption("THRP result from the two-stage pipeline")
        else:
            st.warning("No confident THRP detection was returned for this image.")
    else:
        if result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Superclass", result.get("superclass", "N/A"))
            c2.metric("Prediction", result.get("predicted_name", "N/A"))
            confidence = result.get("confidence")
            c3.metric("Confidence", f"{confidence:.1%}" if confidence is not None else "N/A")
            st.caption("Classical baseline result")
            st.write(
                f"The selected method predicts **{result.get('predicted_name', 'N/A')}** for the **{result.get('superclass', 'N/A')}** branch."
            )
            st.json(to_json_safe(result))
        else:
            st.warning("No result returned.")

    if preview_path and preview_path.exists():
        st.image(str(preview_path), caption="THRP visualization", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def run_method(method: Dict[str, str], uploaded_file) -> tuple[Optional[Dict[str, Any]], str, Optional[Path], str]:
    input_path = save_upload(uploaded_file, method["key"])
    if method["key"] == "thrp":
        preview_dir = OUTPUTS / "frontend_previews" / method["key"]
        preview_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/predict.py",
            "--image",
            str(input_path),
            "--generalist",
            "models/generalist_model.pt",
            "--specialists",
            "models",
            "--output",
            str(preview_dir),
            "--visualize",
        ]
        proc = run_command(command, ROOT)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "THRP command failed").strip())

        result = parse_thrp_output(proc.stdout)
        generated_preview = preview_dir / f"pred_{input_path.name}"
        if not generated_preview.exists():
            generated_preview = None

        command_text = " ".join([str(piece) for piece in command])
        raw_output = proc.stdout.strip()
        return result, command_text, generated_preview, raw_output

    # classical baselines
    command = [
        sys.executable,
        "scripts/" + {"decision_tree": "predict_auto_decision_tree.py", "knn": "predict_auto_knn.py", "bayes": "predict_auto_bayes.py", "svm": "predict_auto_svm.py"}[method["key"]],
        "--image",
        str(input_path),
    ]
    proc = run_command(command, ROOT)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"{method['title']} command failed").strip())
    result = parse_baseline_output(proc.stdout)
    command_text = " ".join([str(piece) for piece in command])
    return result, command_text, None, proc.stdout.strip()


def render_method_tab(method: Dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="method-title">{method['title']}</div>
            <div class="method-subtitle">{method['subtitle']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1.3], gap="large")
    with left:
        uploaded = st.file_uploader(
            f"Upload an image for {method['title']}",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"uploader_{method['key']}",
            label_visibility="visible",
        )
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image preview", use_container_width=True)
            run_clicked = st.button(f"Run {method['title']} prediction", key=f"run_{method['key']}", use_container_width=True)
        else:
            st.info("Upload an image to enable prediction.")
            run_clicked = False

    with right:
        if uploaded is not None and run_clicked:
            with st.spinner(f"Running {method['title']}..."):
                try:
                    result, command_text, preview_path, raw_output = run_method(method, uploaded)
                    st.success("Prediction completed")
                    render_prediction_summary(method["key"], result, preview_path)
                    with st.expander("Command used"):
                        st.code(command_text, language="bash")
                    with st.expander("Raw output"):
                        st.code(raw_output or "", language="text")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.markdown(
                """
                <div class="result-box">
                    <h4 style="margin-top:0;">Prediction output</h4>
                    <p class="small-note">
                        Once you upload an image and click run, the command output, predicted label,
                        confidence, and a preview will appear here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_styles()
    render_header()
    render_metrics()
    st.write("")
    tabs = st.tabs([item["title"] for item in METHODS])

    for tab, method in zip(tabs, METHODS):
        with tab:
            render_method_tab(method)


if __name__ == "__main__":
    main()
