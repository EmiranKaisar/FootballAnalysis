from football_analysis.domain import AnalysisResult, Event, EventType, Outcome, Team


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

