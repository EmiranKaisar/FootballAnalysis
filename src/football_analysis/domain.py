from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Team(StrEnum):
    A = "team_a"
    B = "team_b"


class EventType(StrEnum):
    PASS = "pass"
    SHOT = "shot"


class Outcome(StrEnum):
    SUCCESSFUL = "successful"
    UNSUCCESSFUL = "unsuccessful"
    ON_TARGET = "on_target"
    OFF_TARGET = "off_target"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Event:
    timestamp_seconds: float
    event_type: EventType
    team: Team
    outcome: Outcome
    evidence_score: float
    evidence: tuple[str, ...] = ()

    def as_dict(self, team_names: dict[Team, str] | None = None) -> dict[str, Any]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        value["team"] = team_names.get(self.team, self.team.value) if team_names else self.team.value
        value["outcome"] = self.outcome.value
        value["evidence"] = list(self.evidence)
        return value


@dataclass
class AnalysisResult:
    source_name: str
    duration_seconds: float
    team_names: dict[Team, str]
    events: list[Event] = field(default_factory=list)
    controlled_seconds: dict[Team, float] = field(
        default_factory=lambda: {Team.A: 0.0, Team.B: 0.0}
    )
    eligible_seconds: float = 0.0
    analyzed_frames: int = 0
    source_frames: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def measurable_seconds(self) -> float:
        return sum(self.controlled_seconds.values())

    @property
    def measurable_coverage_percent(self) -> float:
        if self.eligible_seconds <= 0:
            return 0.0
        return min(100.0, 100.0 * self.measurable_seconds / self.eligible_seconds)

    def team_summary(self, team: Team) -> dict[str, Any]:
        events = [event for event in self.events if event.team == team]
        passes = [event for event in events if event.event_type == EventType.PASS]
        resolved_passes = [
            event
            for event in passes
            if event.outcome in (Outcome.SUCCESSFUL, Outcome.UNSUCCESSFUL)
        ]
        successful_passes = sum(event.outcome == Outcome.SUCCESSFUL for event in passes)
        shots = [event for event in events if event.event_type == EventType.SHOT]
        on_target = sum(event.outcome == Outcome.ON_TARGET for event in shots)
        possession = (
            100.0 * self.controlled_seconds[team] / self.measurable_seconds
            if self.measurable_seconds
            else 0.0
        )
        return {
            "team": self.team_names[team],
            "shots": len(shots),
            "shots_on_target": on_target,
            "passes": len(passes),
            "successful_passes": successful_passes,
            "unknown_passes": sum(event.outcome == Outcome.UNKNOWN for event in passes),
            "pass_success_percent": (
                100.0 * successful_passes / len(resolved_passes) if resolved_passes else 0.0
            ),
            "possession_percent": possession,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "duration_seconds": round(self.duration_seconds, 3),
            "estimated": True,
            "teams": [self.team_summary(Team.A), self.team_summary(Team.B)],
            "quality": {
                "measurable_coverage_percent": round(self.measurable_coverage_percent, 2),
                "analyzed_frames": self.analyzed_frames,
                "source_frames": self.source_frames,
                "warnings": self.warnings,
            },
            "events": [event.as_dict(self.team_names) for event in self.events],
        }

