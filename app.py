from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import streamlit as st
from streamlit.components.v2 import component

from football_analysis.analyzer import AnalysisConfig, analyze_video
from football_analysis.detection import UltralyticsTracker
from football_analysis.domain import AnalysisResult, Team
from football_analysis.preflight import inspect_video
from football_analysis.teams import suggest_team_colors

PROJECT_ROOT = Path(__file__).resolve().parent
PRIVATE_INPUT = PROJECT_ROOT / "data" / "private" / "input"
FOOTBALL_MODEL = PROJECT_ROOT / "models" / "yolo-football-player-detection.pt"
BALL_MODEL = PROJECT_ROOT / "yolo11n.pt"
PRIVATE_INPUT.mkdir(parents=True, exist_ok=True)

ANALYSIS_RANGE_COMPONENT = component(
    "analysis_range_selector",
    html="""
        <div class="range-component">
            <div class="range-labels" aria-hidden="true">
                <output id="start-label"></output>
                <output id="end-label"></output>
            </div>
            <div class="range-track">
                <input id="start" type="range" aria-label="Analysis range start">
                <input id="end" type="range" aria-label="Analysis range end">
            </div>
            <div class="range-summary" aria-live="polite"></div>
        </div>
    """,
    css="""
        .range-component {
            box-sizing: border-box;
            padding: 1.75rem 0.65rem 0;
            width: 100%;
            color: var(--st-text-color);
            font-family: var(--st-font);
        }
        .range-labels, .range-track {
            position: relative;
            width: 100%;
        }
        .range-labels output {
            position: absolute;
            bottom: 0.25rem;
            transform: translateX(-50%);
            padding: 0.15rem 0.4rem;
            border: 1px solid color-mix(in srgb, var(--st-primary-color) 45%, transparent);
            border-radius: 0.35rem;
            background: var(--st-secondary-background-color);
            font-size: 0.78rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .range-labels output.active {
            border-color: var(--st-primary-color);
            color: var(--st-primary-color);
            font-weight: 700;
        }
        .range-track {
            height: 2.25rem;
            border-radius: 999px;
            background: var(--range-gradient);
        }
        .range-track input[type="range"] {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 2.25rem;
            margin: 0;
            appearance: none;
            -webkit-appearance: none;
            background: transparent;
            pointer-events: none;
        }
        .range-track input[type="range"]::-webkit-slider-runnable-track {
            height: 0.55rem;
            background: transparent;
        }
        .range-track input[type="range"]::-moz-range-track {
            height: 0.55rem;
            background: transparent;
        }
        .range-track input[type="range"]::-webkit-slider-thumb {
            width: 1.25rem;
            height: 1.25rem;
            margin-top: -0.35rem;
            appearance: none;
            -webkit-appearance: none;
            border: 0.18rem solid white;
            border-radius: 50%;
            background: var(--st-primary-color);
            box-shadow: 0 0 0 1px var(--st-primary-color), 0 2px 5px rgba(0, 0, 0, 0.25);
            cursor: grab;
            pointer-events: auto;
        }
        .range-track input[type="range"]::-moz-range-thumb {
            width: 1.25rem;
            height: 1.25rem;
            border: 0.18rem solid white;
            border-radius: 50%;
            background: var(--st-primary-color);
            box-shadow: 0 0 0 1px var(--st-primary-color), 0 2px 5px rgba(0, 0, 0, 0.25);
            cursor: grab;
            pointer-events: auto;
        }
        .range-track input[type="range"]:focus-visible::-webkit-slider-thumb {
            outline: 3px solid color-mix(in srgb, var(--st-primary-color) 35%, transparent);
            outline-offset: 3px;
        }
        .range-track input[type="range"]:focus-visible::-moz-range-thumb {
            outline: 3px solid color-mix(in srgb, var(--st-primary-color) 35%, transparent);
            outline-offset: 3px;
        }
        #start { z-index: 3; }
        #end { z-index: 2; }
        .range-summary {
            margin-top: 0.15rem;
            text-align: center;
            color: var(--st-text-color);
            font-size: 0.78rem;
            font-variant-numeric: tabular-nums;
        }
    """,
    js="""
        export default function(component) {
            const { data, parentElement, setStateValue } = component;
            const options = data.options;
            const minimumDuration = data.minimumDuration;
            const startInput = parentElement.querySelector('#start');
            const endInput = parentElement.querySelector('#end');
            const startLabel = parentElement.querySelector('#start-label');
            const endLabel = parentElement.querySelector('#end-label');
            const summary = parentElement.querySelector('.range-summary');
            const track = parentElement.querySelector('.range-track');

            const closestIndex = (seconds) => options.reduce(
                (best, value, index) => Math.abs(value - seconds) < Math.abs(options[best] - seconds) ? index : best,
                0,
            );
            const formatTime = (seconds) => {
                const minutes = Math.floor(Math.max(0, seconds) / 60);
                const remainder = Math.max(0, seconds) - minutes * 60;
                return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(1).padStart(4, '0')}`;
            };
            const maxIndex = options.length - 1;
            startInput.min = endInput.min = 0;
            startInput.max = endInput.max = maxIndex;
            startInput.step = endInput.step = 1;
            startInput.value = closestIndex(data.start);
            endInput.value = closestIndex(data.end);

            const constrain = (active) => {
                let startIndex = Number(startInput.value);
                let endIndex = Number(endInput.value);
                if (options[endIndex] - options[startIndex] >= minimumDuration) return;
                if (active === startInput) {
                    while (startIndex > 0 && options[endIndex] - options[startIndex] < minimumDuration) startIndex -= 1;
                    startInput.value = startIndex;
                } else {
                    while (endIndex < maxIndex && options[endIndex] - options[startIndex] < minimumDuration) endIndex += 1;
                    endInput.value = endIndex;
                }
            };
            const render = (active = null) => {
                const startIndex = Number(startInput.value);
                const endIndex = Number(endInput.value);
                const startPercent = 100 * startIndex / maxIndex;
                const endPercent = 100 * endIndex / maxIndex;
                startLabel.textContent = formatTime(options[startIndex]);
                endLabel.textContent = formatTime(options[endIndex]);
                startLabel.style.left = `${startPercent}%`;
                endLabel.style.left = `${endPercent}%`;
                startLabel.classList.toggle('active', active === startInput);
                endLabel.classList.toggle('active', active === endInput);
                track.style.setProperty(
                    '--range-gradient',
                    `linear-gradient(to right, var(--st-secondary-background-color) 0 ${startPercent}%, ` +
                    `var(--st-primary-color) ${startPercent}% ${endPercent}%, ` +
                    `var(--st-secondary-background-color) ${endPercent}% 100%)`,
                );
                const duration = options[endIndex] - options[startIndex];
                summary.textContent = `Selected duration · ${duration.toFixed(1)}s`;
                startInput.setAttribute(
                    'aria-valuetext',
                    `Start ${formatTime(options[startIndex])}; minimum range 30 seconds`,
                );
                endInput.setAttribute(
                    'aria-valuetext',
                    `End ${formatTime(options[endIndex])}; minimum range 30 seconds`,
                );
            };
            const commit = () => setStateValue('range', [
                options[Number(startInput.value)],
                options[Number(endInput.value)],
            ]);
            const moveOneSecond = (event, input) => {
                if (!event.shiftKey || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
                event.preventDefault();
                const direction = event.key === 'ArrowRight' ? 1 : -1;
                const currentIndex = Number(input.value);
                const target = options[currentIndex] + direction;
                let nextIndex = closestIndex(target);
                if (direction > 0) nextIndex = Math.max(currentIndex + 1, nextIndex);
                else nextIndex = Math.min(currentIndex - 1, nextIndex);
                input.value = Math.max(0, Math.min(maxIndex, nextIndex));
                constrain(input);
                render(input);
                commit();
            };

            for (const input of [startInput, endInput]) {
                input.oninput = () => { constrain(input); render(input); };
                input.onchange = () => { render(); commit(); };
                input.onfocus = () => render(input);
                input.onblur = () => render();
                input.onkeydown = (event) => moveOneSecond(event, input);
            }
            render();
        }
    """,
)


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


def format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{remainder:04.1f}"


def analysis_range_options(result: AnalysisResult) -> tuple[float, ...]:
    timestamps = {0.0, result.duration_seconds}
    timestamps.update(round(value, 6) for value in result.analyzed_timestamps)
    return tuple(sorted(value for value in timestamps if 0.0 <= value <= result.duration_seconds))


def initialize_analysis_range(result: AnalysisResult, job_id: str) -> tuple[str, tuple[float, ...]]:
    widget_key = f"analysis_range_{job_id}"
    options = analysis_range_options(result)
    default = (options[0], options[-1])
    if widget_key not in st.session_state:
        st.session_state[widget_key] = default
    return widget_key, options


def reset_analysis_range(widget_key: str, options: tuple[float, ...]) -> None:
    full_clip = (options[0], options[-1])
    st.session_state[widget_key] = full_clip
    generation_key = f"{widget_key}_generation"
    st.session_state[generation_key] = st.session_state.get(generation_key, 0) + 1


def render_result(result: AnalysisResult, video_path: Path) -> None:
    st.success("Estimated analysis complete")
    st.header(
        "Analysis report · "
        f"{format_timestamp(result.range_start_seconds)}–{format_timestamp(result.selected_end_seconds)} "
        f"({result.selected_duration_seconds:.1f}s)"
    )
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
            possession = summary["possession_percent"]
            third.metric("Possession", "N/A" if possession is None else f"{possession:.1f}%")
            first, second = st.columns(2)
            first.metric("Passes", summary["passes"])
            second.metric("Pass success", f'{summary["pass_success_percent"]:.1f}%')

    rows = [event.as_dict(result.team_names) for event in result.events]
    st.subheader("Estimated event timeline")
    if rows:
        st.table(rows)
        labels = [
            f'{event.timestamp_seconds:06.2f}s · {result.team_names[event.team]} · '
            f'{event.event_type.value} · {event.outcome.value}'
            for event in result.events
        ]
        selected = st.selectbox("Review an event", range(len(labels)), format_func=lambda index: labels[index])
        st.video(str(video_path), start_time=max(0, int(result.events[selected].timestamp_seconds) - 2))
    else:
        st.info("No events met the conservative detection rules in this clip.")

    if result.measurable_seconds == 0:
        st.info("This interval lacks sufficient visible evidence to measure possession.")


def render_analysis_range_selector(
    widget_key: str,
    options: tuple[float, ...],
) -> None:
    st.markdown(
        """
        <style>
        .st-key-analysis_range_selector {
            position: sticky;
            bottom: 0.75rem;
            z-index: 50;
            padding: 0.75rem 1rem 0.25rem;
            border: 1px solid color-mix(in srgb, var(--primary-color) 35%, transparent);
            border-radius: 0.75rem;
            background: color-mix(in srgb, var(--background-color) 94%, transparent);
            backdrop-filter: blur(10px);
            box-shadow: 0 0.35rem 1.25rem rgba(0, 0, 0, 0.14);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="analysis_range_selector"):
        st.subheader("Analysis range")
        st.caption(
            "Drag either boundary to filter the cached full-clip report. "
            "The video is not reanalyzed or seeked. Minimum range: 30 seconds."
        )
        start_seconds, end_seconds = st.session_state[widget_key]
        generation = st.session_state.get(f"{widget_key}_generation", 0)
        component_result = ANALYSIS_RANGE_COMPONENT(
            key=f"{widget_key}_{generation}",
            data={
                "options": options,
                "start": start_seconds,
                "end": end_seconds,
                "minimumDuration": 30.0,
            },
            default={"range": [start_seconds, end_seconds]},
            on_range_change=lambda: None,
        )
        selected_range = tuple(component_result.range or (start_seconds, end_seconds))
        if selected_range != (start_seconds, end_seconds):
            selected_start, selected_end = selected_range
            if selected_end - selected_start >= 30.0:
                st.session_state[widget_key] = (selected_start, selected_end)
                st.rerun()
        left, middle, right = st.columns([1, 2, 1])
        left.caption(f"Start · {format_timestamp(start_seconds)}")
        middle.caption(f"Selected duration · {end_seconds - start_seconds:.1f}s")
        right.button(
            "Reset to full clip",
            key=f"reset_{widget_key}",
            on_click=reset_analysis_range,
            args=(widget_key, options),
            width="stretch",
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
    if st.button("Analyze clip", type="primary", width="stretch"):
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
        widget_key, options = initialize_analysis_range(result, job.job_id)
        start_seconds, end_seconds = st.session_state[widget_key]
        filtered_result = result.filtered(start_seconds, end_seconds)
        render_result(filtered_result, job.video_path)
        render_analysis_range_selector(widget_key, options)
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
