from football_analysis.jobs import SingleJobGate


def test_single_job_gate_allows_only_one_owner() -> None:
    gate = SingleJobGate()

    assert gate.try_acquire("first") is True
    assert gate.current_owner() == "first"
    assert gate.try_acquire("second") is False

    gate.release("second")
    assert gate.current_owner() == "first"

    gate.release("first")
    assert gate.current_owner() is None
    assert gate.try_acquire("second") is True
