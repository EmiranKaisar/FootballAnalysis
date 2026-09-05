from __future__ import annotations

import csv
import io
import json

from .domain import AnalysisResult


def result_json(result: AnalysisResult) -> str:
    return json.dumps(result.as_dict(), indent=2)


def events_csv(result: AnalysisResult) -> str:
    output = io.StringIO()
    fields = ["timestamp_seconds", "event_type", "team", "outcome", "evidence_score", "evidence"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for event in result.events:
        row = event.as_dict(result.team_names)
        row["evidence"] = " | ".join(row["evidence"])
        writer.writerow(row)
    return output.getvalue()

