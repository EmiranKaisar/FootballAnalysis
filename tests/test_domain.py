import pytest

from football_analysis.domain import AnalysisInterval, AnalysisResult, Event, EventType, Outcome, Team


def test_pass_success_excludes_unknown_outcomes() -> None:
    result = AnalysisResult(
        source_name="private.mp4",
        duration_seconds=30,
        team_names={Team.A: "Red", Team.B: "Blue"},
        events=[
            Event(1, EventType.PASS, Team.A, Outcome.SUCCESSFUL, 0.8),
            Event(2, EventType.PASS, Team.A, Outcome.UNSUCCESSFUL, 0.8),
            Event(3, EventType.PASS, Team.A, Outcome.UNKNOWN, 0.4),
        ],
        controlled_seconds={Team.A: 12, Team.B: 8},
        eligible_seconds=25,
    )

    summary = result.team_summary(Team.A)

    assert summary["passes"] == 3
    assert summary["unknown_passes"] == 1
    assert summary["pass_success_percent"] == 50
    assert summary["possession_percent"] == 60
    assert result.measurable_coverage_percent == 80


def test_result_is_explicitly_estimated() -> None:
    result = AnalysisResult(
        source_name="private.mp4",
        duration_seconds=5,
        team_names={Team.A: "A", Team.B: "B"},
    )

    assert result.as_dict()["estimated"] is True


def ranged_result() -> AnalysisResult:
    return AnalysisResult(
        source_name="private.mp4",
        duration_seconds=60,
        team_names={Team.A: "Red", Team.B: "Blue"},
        events=[
            Event(10, EventType.PASS, Team.A, Outcome.SUCCESSFUL, 0.8),
            Event(30, EventType.SHOT, Team.B, Outcome.ON_TARGET, 0.9),
            Event(40, EventType.PASS, Team.A, Outcome.UNSUCCESSFUL, 0.7),
            Event(60, EventType.SHOT, Team.A, Outcome.UNKNOWN, 0.5),
        ],
        controlled_seconds={Team.A: 20, Team.B: 20},
        eligible_seconds=60,
        analyzed_frames=7,
        intervals=[
            AnalysisInterval(0, 20, Team.A),
            AnalysisInterval(20, 40, Team.B),
            AnalysisInterval(40, 60, None),
        ],
        analyzed_timestamps=[0, 10, 20, 30, 40, 50, 60],
    )


def test_filter_uses_half_open_event_boundaries() -> None:
    filtered = ranged_result().filtered(20, 40)

    assert [event.timestamp_seconds for event in filtered.events] == [30]
    assert filtered.analyzed_timestamps == [20, 30]
    assert filtered.analyzed_frames == 2


def test_filter_includes_physical_clip_endpoint() -> None:
    filtered = ranged_result().filtered(30, 60)

    assert [event.timestamp_seconds for event in filtered.events] == [30, 40, 60]
    assert filtered.analyzed_timestamps == [30, 40, 50, 60]


def test_filter_clips_duration_intervals_and_recalculates_metrics() -> None:
    filtered = ranged_result().filtered(15, 45)

    assert filtered.eligible_seconds == 30
    assert filtered.controlled_seconds == {Team.A: 5, Team.B: 20}
    assert filtered.measurable_coverage_percent == pytest.approx(83.3333)
    assert filtered.team_summary(Team.A)["possession_percent"] == 20
    assert filtered.team_summary(Team.B)["possession_percent"] == 80


def test_possession_is_unavailable_without_measurable_evidence() -> None:
    filtered = ranged_result().filtered(40, 60)

    assert filtered.measurable_coverage_percent == 0
    assert filtered.team_summary(Team.A)["possession_percent"] is None
    assert filtered.team_summary(Team.B)["possession_percent"] is None


def test_filter_rejects_ranges_outside_the_clip() -> None:
    with pytest.raises(ValueError):
        ranged_result().filtered(-1, 40)
