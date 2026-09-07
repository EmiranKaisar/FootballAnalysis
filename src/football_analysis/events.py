from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot

from .detection import TrackedObject
from .domain import AnalysisInterval, Event, EventType, Outcome, Team


@dataclass(frozen=True)
class PlayerState:
    track: TrackedObject
    team: Team
    team_confidence: float

    @property
    def is_goalkeeper(self) -> bool:
        return self.track.label == "goalkeeper"


@dataclass
class PendingRelease:
    player_id: int
    team: Team
    timestamp: float
    is_shot: bool
    evidence_score: float


class EventEngine:
    """Interpretable, deliberately conservative event and possession state machine."""

    def __init__(self, width: int, height: int, fps: float, attacks: dict[Team, str]) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.attacks = attacks
        self.events: list[Event] = []
        self.ball_history: deque[tuple[float, float, float]] = deque(maxlen=max(4, int(fps)))
        self.current_controller: int | None = None
        self.current_team: Team | None = None
        self.control_candidate: int | None = None
        self.control_candidate_frames = 0
        self.missing_control_frames = 0
        self.pending: PendingRelease | None = None
        self.possession_seconds = {Team.A: 0.0, Team.B: 0.0}
        self.eligible_seconds = 0.0
        self.intervals: list[AnalysisInterval] = []
        self.last_timestamp: float | None = None
        self.shot_cooldown_until = 0.0

    def _controller(self, players: list[PlayerState], ball: TrackedObject | None) -> PlayerState | None:
        if ball is None:
            return None
        bx, by = ball.center
        candidates: list[tuple[float, PlayerState]] = []
        for player in players:
            px, py = player.track.foot
            distance = hypot(bx - px, by - py)
            threshold = max(24.0, player.track.height * 0.85)
            if distance <= threshold:
                candidates.append((distance / threshold, player))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _ball_motion(self) -> tuple[float, float, float]:
        if len(self.ball_history) < 2:
            return 0.0, 0.0, 0.0
        start = self.ball_history[0]
        end = self.ball_history[-1]
        elapsed = max(1e-6, end[0] - start[0])
        vx = (end[1] - start[1]) / elapsed
        vy = (end[2] - start[2]) / elapsed
        return vx, vy, hypot(vx, vy)

    def _looks_like_shot(self, team: Team, ball: TrackedObject | None, timestamp: float) -> tuple[bool, float]:
        if ball is None or timestamp < self.shot_cooldown_until:
            return False, 0.0
        x, _ = ball.center
        vx, _, speed = self._ball_motion()
        direction = self.attacks[team]
        in_attacking_area = x >= self.width * 0.55 if direction == "right" else x <= self.width * 0.45
        toward_goal = vx > self.width * 0.07 if direction == "right" else vx < -self.width * 0.07
        fast = speed > self.width * 0.10
        score = 0.45 + 0.2 * in_attacking_area + 0.2 * toward_goal + 0.15 * fast
        return in_attacking_area and toward_goal and fast, min(0.92, score)

    def _shot_outcome(self, ball: TrackedObject | None, new_controller: PlayerState | None) -> Outcome | None:
        if self.pending is None or not self.pending.is_shot:
            return None
        if new_controller is not None and new_controller.team != self.pending.team:
            return Outcome.ON_TARGET if new_controller.is_goalkeeper else Outcome.BLOCKED
        if ball is None:
            return None
        x, y = ball.center
        direction = self.attacks[self.pending.team]
        reached_end = x >= self.width * 0.94 if direction == "right" else x <= self.width * 0.06
        if not reached_end:
            return None
        # A ball leaving the visible goalward edge does not prove a goal or save.
        # Only clearly missing the broad goal band is enough to call it off target.
        in_goal_band = self.height * 0.20 <= y <= self.height * 0.72
        return None if in_goal_band else Outcome.OFF_TARGET

    def _finish_pending(self, timestamp: float, controller: PlayerState | None, ball: TrackedObject | None) -> None:
        if self.pending is None:
            return
        age = timestamp - self.pending.timestamp
        if self.pending.is_shot:
            outcome = self._shot_outcome(ball, controller)
            if outcome is not None or age >= 2.5:
                self.events.append(
                    Event(
                        self.pending.timestamp,
                        EventType.SHOT,
                        self.pending.team,
                        outcome or Outcome.UNKNOWN,
                        self.pending.evidence_score,
                        (
                            "release",
                            "goalward_trajectory",
                            "goalkeeper_control" if outcome == Outcome.ON_TARGET and controller and controller.is_goalkeeper else (
                                "terminal_observation" if outcome else "outcome_not_visible"
                            ),
                        ),
                    )
                )
                self.pending = None
            return
        if controller is not None:
            if controller.track.track_id == self.pending.player_id:
                self.pending = None
                return
            outcome = Outcome.SUCCESSFUL if controller.team == self.pending.team else Outcome.UNSUCCESSFUL
            self.events.append(
                Event(
                    self.pending.timestamp,
                    EventType.PASS,
                    self.pending.team,
                    outcome,
                    min(self.pending.evidence_score, 0.55 + 0.35 * controller.team_confidence),
                    ("controlled_release", "different_next_controller", f"receiver_{outcome.value}"),
                )
            )
            self.pending = None
        elif age >= 3.0:
            self.events.append(
                Event(
                    self.pending.timestamp,
                    EventType.PASS,
                    self.pending.team,
                    Outcome.UNKNOWN,
                    self.pending.evidence_score * 0.75,
                    ("controlled_release", "next_controller_not_visible"),
                )
            )
            self.pending = None
            self.current_team = None

    def update(
        self,
        timestamp: float,
        players: list[PlayerState],
        ball: TrackedObject | None,
    ) -> None:
        dt = 0.0 if self.last_timestamp is None else max(0.0, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp
        self.eligible_seconds += dt
        if ball is not None:
            bx, by = ball.center
            self.ball_history.append((timestamp, bx, by))

        raw_controller = self._controller(players, ball)
        raw_id = raw_controller.track.track_id if raw_controller else None
        if raw_id is not None and raw_id == self.control_candidate:
            self.control_candidate_frames += 1
        else:
            self.control_candidate = raw_id
            self.control_candidate_frames = 1 if raw_id is not None else 0

        stable_controller = raw_controller if raw_controller and self.control_candidate_frames >= 2 else None
        self._finish_pending(timestamp, stable_controller, ball)

        if stable_controller is not None:
            self.missing_control_frames = 0
            self.current_controller = stable_controller.track.track_id
            self.current_team = stable_controller.team
        elif self.current_controller is not None:
            self.missing_control_frames += 1
            if self.missing_control_frames >= 2 and self.pending is None:
                is_shot, score = self._looks_like_shot(self.current_team, ball, timestamp)
                self.pending = PendingRelease(
                    self.current_controller,
                    self.current_team,
                    timestamp,
                    is_shot,
                    score if is_shot else 0.55,
                )
                if is_shot:
                    self.shot_cooldown_until = timestamp + 2.0
                self.current_controller = None
                if is_shot:
                    self.current_team = None

        possession_team = self.current_team
        if possession_team is None and self.pending is not None and not self.pending.is_shot:
            if timestamp - self.pending.timestamp <= 3.0:
                possession_team = self.pending.team
        if possession_team is not None:
            self.possession_seconds[possession_team] += dt
        if dt > 0:
            self.intervals.append(
                AnalysisInterval(timestamp - dt, timestamp, possession_team, eligible=True)
            )

    def finish(self, timestamp: float) -> None:
        self._finish_pending(timestamp + 3.1, None, None)
