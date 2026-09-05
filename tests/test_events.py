from football_analysis.detection import TrackedObject
from football_analysis.domain import EventType, Outcome, Team
from football_analysis.events import EventEngine, PlayerState


def player(track_id: int, team: Team, foot_x: float, foot_y: float = 300) -> PlayerState:
    track = TrackedObject(track_id, "player", 0.9, (foot_x - 20, foot_y - 100, foot_x + 20, foot_y))
    return PlayerState(track, team, 0.9)


def goalkeeper(track_id: int, team: Team, foot_x: float, foot_y: float = 300) -> PlayerState:
    track = TrackedObject(track_id, "goalkeeper", 0.9, (foot_x - 20, foot_y - 100, foot_x + 20, foot_y))
    return PlayerState(track, team, 0.9)


def ball(track_id: int, x: float, y: float = 295) -> TrackedObject:
    return TrackedObject(track_id, "ball", 0.8, (x - 5, y - 5, x + 5, y + 5))


def test_same_team_control_transition_is_successful_pass() -> None:
    engine = EventEngine(1000, 600, 10, {Team.A: "right", Team.B: "left"})
    engine.update(0.0, [player(1, Team.A, 300)], ball(90, 300))
    engine.update(0.1, [player(1, Team.A, 305)], ball(90, 305))
    engine.update(0.2, [], ball(90, 360))
    engine.update(0.3, [], ball(90, 420))
    engine.update(0.4, [player(2, Team.A, 480)], ball(90, 480))
    engine.update(0.5, [player(2, Team.A, 485)], ball(90, 485))

    assert len(engine.events) == 1
    assert engine.events[0].event_type == EventType.PASS
    assert engine.events[0].outcome == Outcome.SUCCESSFUL


def test_fast_goalward_release_saved_by_goalkeeper_is_on_target() -> None:
    engine = EventEngine(1000, 600, 10, {Team.A: "right", Team.B: "left"})
    engine.update(0.0, [player(1, Team.A, 560, 300)], ball(90, 560, 295))
    engine.update(0.1, [player(1, Team.A, 570, 300)], ball(90, 570, 295))
    engine.update(0.2, [], ball(90, 680, 300))
    engine.update(0.3, [], ball(90, 800, 310))
    engine.update(0.4, [goalkeeper(2, Team.B, 950, 320)], ball(90, 950, 315))
    engine.update(0.5, [goalkeeper(2, Team.B, 950, 320)], ball(90, 950, 315))

    assert len(engine.events) == 1
    assert engine.events[0].event_type == EventType.SHOT
    assert engine.events[0].outcome == Outcome.ON_TARGET
