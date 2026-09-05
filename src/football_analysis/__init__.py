"""Local football analysis proof of concept."""

from .analyzer import AnalysisConfig, analyze_video
from .domain import AnalysisResult, Event, EventType, Outcome, Team
from .preflight import PreflightReport, inspect_video

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "Event",
    "EventType",
    "Outcome",
    "PreflightReport",
    "Team",
    "analyze_video",
    "inspect_video",
]

