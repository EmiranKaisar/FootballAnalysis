from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import streamlit as st

from football_analysis.analyzer import AnalysisConfig, analyze_video
from football_analysis.detection import UltralyticsTracker
from football_analysis.domain import AnalysisResult, Team
from football_analysis.exports import events_csv, result_json
from football_analysis.preflight import inspect_video
from football_analysis.teams import suggest_team_colors

PROJECT_ROOT = Path(__file__).resolve().parent
PRIVATE_INPUT = PROJECT_ROOT / "data" / "private" / "input"
FOOTBALL_MODEL = PROJECT_ROOT / "models" / "yolo-football-player-detection.pt"
BALL_MODEL = PROJECT_ROOT / "yolo11n.pt"
PRIVATE_INPUT.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    job_id: str
    video_path: Path
    config: AnalysisConfig
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting"
    result: AnalysisResult | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


@st.cache_resource
def job_registry() -> dict[str, Job]:
    return {}


@st.cache_resource
def calibration_tracker(model_path: str) -> UltralyticsTracker:
    return UltralyticsTracker(model_path)


def run_job(job: Job) -> None:
    def update(stage: str, value: float, message: str) -> None:
        with job.lock:
            job.stage = stage
            job.progress = value
            job.message = message

    try:
        result = analyze_video(job.video_path, job.config, progress=update, cancel=job.cancel)
        with job.lock:
            job.result = result
    except InterruptedError:
        with job.lock:
            job.stage = "cancelled"
            job.message = "Analysis cancelled"
    except Exception as exc:  # surfaced in the local UI with no remote logging
        with job.lock:
            job.stage = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "Analysis failed"


def start_job(video_path: Path, config: AnalysisConfig) -> str:
    registry = job_registry()
    job_id = uuid.uuid4().hex
    job = Job(job_id=job_id, video_path=video_path, config=config)
    registry[job_id] = job
    threading.Thread(target=run_job, args=(job,), daemon=True, name=f"analysis-{job_id[:8]}").start()
    return job_id


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    target = PRIVATE_INPUT / f"session-{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as destination:
        shutil.copyfileobj(uploaded_file, destination)
    return target


def first_frame(path: Path):
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def propose_colors(path: Path, model_path: str) -> tuple[str, str] | None:
    report = inspect_video(path, sample_count=6)
    if not report.suitable:
        return None
    tracker = calibration_tracker(model_path)
    tracker.reset()
    samples = [(frame, tracker.track(frame)) for frame in report.sampled_frames]
    tracker.reset()
    return suggest_team_colors(samples)


def render_result(result: AnalysisResult, video_path: Path) -> None:
    st.success("Estimated analysis complete")
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    st.caption(
        f"Measurable possession coverage: {result.measurable_coverage_percent:.1f}% · "
        f"Analyzed frames: {result.analyzed_frames:,}"
    )
    columns = st.columns(2)
    for column, team in zip(columns, (Team.A, Team.B)):
        summary = result.team_summary(team)
        with column:
            st.subheader(summary["team"])
            first, second, third = st.columns(3)
            first.metric("Shots", summary["shots"])
            second.metric("On target", summary["shots_on_target"])
            third.metric("Possession", f'{summary["possession_percent"]:.1f}%')
            first, second = st.columns(2)
            first.metric("Passes", summary["passes"])
            second.metric("Pass success", f'{summary["pass_success_percent"]:.1f}%')

    rows = [event.as_dict(result.team_names) for event in result.events]
    st.subheader("Estimated event timeline")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        labels = [
            f'{event.timestamp_seconds:06.2f}s · {result.team_names[event.team]} · '
            f'{event.event_type.value} · {event.outcome.value}'
            for event in result.events
        ]
        selected = st.selectbox("Review an event", range(len(labels)), format_func=lambda index: labels[index])
        st.video(str(video_path), start_time=max(0, int(result.events[selected].timestamp_seconds) - 2))
    else:
        st.info("No events met the conservative detection rules in this clip.")

    left, right = st.columns(2)
    left.download_button(
        "Download JSON",
        result_json(result),
        file_name=f"{video_path.stem}-analysis.json",
        mime="application/json",
        use_container_width=True,
    )
    right.download_button(
        "Download events CSV",
        events_csv(result),
        file_name=f"{video_path.stem}-events.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.set_page_config(page_title="Local Football Analysis", page_icon="⚽", layout="wide")
st.title("Local Football Analysis")
st.caption("Private on-device proof of concept · results are estimates, not official match statistics")

local_files = sorted(PRIVATE_INPUT.glob("*.mp4"))
upload = st.file_uploader("Upload a private MP4", type=["mp4"], help="The file stays on this machine.")
if upload is not None and st.session_state.get("uploaded_name") != upload.name:
    st.session_state.video_path = str(save_upload(upload))
    st.session_state.uploaded_name = upload.name

if local_files:
    options = {path.name: path for path in local_files}
    default_name = Path(st.session_state.get("video_path", local_files[0])).name
    default_index = list(options).index(default_name) if default_name in options else 0
    selected_name = st.selectbox("Or choose a private local clip", list(options), index=default_index)
    if upload is None:
        st.session_state.video_path = str(options[selected_name])

video_path_value = st.session_state.get("video_path")
if not video_path_value:
    st.info(f"Place an MP4 in {PRIVATE_INPUT} or upload it above.")
    st.stop()

video_path = Path(video_path_value)
report = inspect_video(video_path)
if report.metadata:
    metadata = report.metadata
    st.write(
        f"**{video_path.name}** — {metadata.width}×{metadata.height}, "
        f"{metadata.fps:.2f} FPS, {metadata.duration_seconds:.1f}s"
    )
for error in report.errors:
    st.error(error)
for warning in report.warnings:
    st.warning(warning)
if not report.suitable:
    st.stop()

with st.expander("Preview", expanded=False):
    st.video(str(video_path))

st.header("Calibration")
model_path = st.text_input(
    "Local model path",
    value=str(FOOTBALL_MODEL if FOOTBALL_MODEL.exists() else "yolo11n.pt"),
    help="Run scripts/download_models.py first. The generic fallback is useful only for checking the UI.",
)
ball_model_path = st.text_input(
    "Local ball model path",
    value=str(BALL_MODEL) if BALL_MODEL.exists() else "",
    help="The COCO model detects this clip's visible ball better than the current specialist checkpoint.",
)
if "team_a_color" not in st.session_state:
    st.session_state.team_a_color = "#d7263d"
    st.session_state.team_b_color = "#1b6ca8"

color_source = f"{video_path.resolve()}::{model_path}"
if st.session_state.get("color_source") != color_source and Path(model_path).exists():
    with st.spinner("Sampling the clip for initial jersey-color suggestions…"):
        try:
            initial_suggestion = propose_colors(video_path, model_path)
        except Exception:
            initial_suggestion = None
    if initial_suggestion:
        st.session_state.team_a_color, st.session_state.team_b_color = initial_suggestion
    st.session_state.color_source = color_source

if st.button("Suggest jersey colors from the clip"):
    with st.spinner("Loading the local model and sampling player crops…"):
        suggestion = propose_colors(video_path, model_path)
    if suggestion:
        st.session_state.team_a_color, st.session_state.team_b_color = suggestion
        st.success("Suggested colors are ready; confirm or correct them below.")
    else:
        st.warning("Not enough player crops were found. Choose the colors manually.")

frame = first_frame(video_path)
if frame is not None:
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Opening frame for calibration", width="stretch")

left, right = st.columns(2)
with left:
    team_a_name = st.text_input("Team A name", "Team A")
    team_a_color = st.color_picker("Team A jersey color", st.session_state.team_a_color)
    team_a_attack = st.selectbox("Team A attacks", ["right", "left"])
with right:
    team_b_name = st.text_input("Team B name", "Team B")
    team_b_color = st.color_picker("Team B jersey color", st.session_state.team_b_color)
    team_b_attack = st.selectbox("Team B attacks", ["left", "right"])

if team_a_color.lower() == team_b_color.lower():
    st.error("The two confirmed jersey colors must differ.")
    st.stop()
if team_a_attack == team_b_attack:
    st.error("The teams must attack opposite goals.")
    st.stop()

active_job_id = st.session_state.get("active_job_id")
active_job = job_registry().get(active_job_id) if active_job_id else None
if active_job is None or active_job.stage in {"complete", "failed", "cancelled"}:
    if st.button("Analyze clip", type="primary", use_container_width=True):
        config = AnalysisConfig(
            team_names={Team.A: team_a_name, Team.B: team_b_name},
            team_colors={Team.A: team_a_color, Team.B: team_b_color},
            attacks={Team.A: team_a_attack, Team.B: team_b_attack},
            model_path=model_path,
            ball_model_path=ball_model_path or None,
        )
        st.session_state.active_job_id = start_job(video_path, config)
        st.rerun()


@st.fragment(run_every=1.0)
def job_panel() -> None:
    job_id = st.session_state.get("active_job_id")
    job = job_registry().get(job_id) if job_id else None
    if job is None:
        return
    with job.lock:
        stage, value, message = job.stage, job.progress, job.message
        result, error = job.result, job.error
    if result is not None:
        render_result(result, job.video_path)
    elif stage == "failed":
        st.error(error or "Analysis failed")
    elif stage == "cancelled":
        st.warning("Analysis was cancelled.")
    else:
        st.progress(value, text=f"{stage.title()}: {message}")
        if st.button("Cancel analysis"):
            job.cancel.set()
            st.warning("Cancellation requested; the current model frame will finish first.")


job_panel()
