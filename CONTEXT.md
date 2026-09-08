# Football Match Analysis

This context defines the football concepts used by the local video-analysis demo so that its reported statistics have consistent meanings.

## Language

**Analysis Clip**:
A continuous live-play segment of no more than two minutes from a fixed or smoothly panning/zooming wide-angle tactical camera, in which most active players, the ball, and enough pitch markings for calibration are normally visible. The demo does not accept cuts, abrupt viewpoint changes, broadcast replays, or close-up edits as analysis clips.
_Avoid_: Broadcast clip, arbitrary match video

**Clip Preview**:
A single small, static image of the analysis clip's opening frame, shown after preflight so the user can confirm the selected clip without playing it.
_Avoid_: Video preview, preview player, event-review player

**Team-Level Analysis**:
Statistics and timestamped events attributed to one of the two teams without asserting a real-world player identity. Temporary tracks may support classification but are not part of the promised result.
_Avoid_: Player analysis, player identification

**Player-Level Analysis**:
Future analysis that attributes events and statistics to a persistent, identifiable player across the clip.
_Avoid_: Temporary track statistics

**Shot**:
An intentional attempt directed toward the opposing goal. A shot may have an on-target, off-target, blocked, or unknown target outcome.
_Avoid_: Any fast ball movement toward the goal

**Shot On Target**:
A shot visibly resulting in a goal or visibly stopped by the goalkeeper when it would otherwise enter the goal. Blocked shots do not qualify.
_Avoid_: Shot toward goal, probable shot on target

**Unknown Target Outcome**:
A shot whose target outcome cannot be established from the visible evidence in the analysis clip. It is reported separately rather than treated as off target.
_Avoid_: Assumed off-target shot

**Pass**:
An intentional ball release from one temporary player track after which a different player becomes the next controller. Dribbles, accidental deflections, and touches without an established next controller are not passes.
_Avoid_: Any change in nearest player

**Successful Pass**:
A pass whose next controlling player belongs to the passing team.
_Avoid_: Unintercepted ball movement

**Unsuccessful Pass**:
A pass after which the opposing team gains control or the ball goes out of play before a teammate controls it.
_Avoid_: Unknown pass outcome

**Unknown Pass Outcome**:
A pass whose receiver or terminal outcome cannot be established from the visible evidence. It is excluded from pass-success-rate calculation.
_Avoid_: Failed pass

**Pass Success Rate**:
Successful passes divided by passes with either a successful or unsuccessful outcome. Passes with an unknown outcome are reported separately and excluded.
_Avoid_: Successful passes divided by all detected pass attempts

**Controlled Possession**:
A live-play interval during which one team clearly controls the ball. It continues while that team's successful pass is in flight and changes only when the opposing team establishes control.
_Avoid_: Nearest-player possession

**Unknown Possession**:
A live-play interval in which control is contested, the ball is not observable, or the evidence is otherwise insufficient. It is excluded from possession-rate calculation.
_Avoid_: Neutral-team possession

**Possession Rate**:
A team's controlled-possession duration divided by the combined controlled-possession duration of both teams. Dead-ball and unknown-possession intervals are excluded.
_Avoid_: Percentage of all video frames

**Measurable Coverage**:
The percentage of eligible live-play time for which controlled possession can be assigned confidently to either team.
_Avoid_: Model confidence

**Estimated Event**:
A timestamped football event inferred from the visible evidence in an analysis clip and accompanied by a confidence level. It is inspectable evidence from a proof of concept, not an official match record.
_Avoid_: Official event, verified event

**Evidence Score**:
A zero-to-one relative indication of how strongly the prototype's visible rule evidence supports an estimated event. It is useful for review and ranking, but it is not a calibrated probability or an accuracy guarantee.
_Avoid_: Probability, model accuracy

**Estimated Statistic**:
A team-level aggregate calculated from estimated events and measurable portions of an analysis clip. It must not be presented as an official match statistic.
_Avoid_: Official statistic, exact statistic

**Clip Team**:
One of the two teams participating in an analysis clip, identified by a user-confirmed name and jersey appearance.
_Avoid_: Automatically inferred team identity

**Attacking Direction**:
The goal toward which a clip team is attacking at the beginning of an analysis clip, confirmed by the user before analysis.
_Avoid_: Camera direction, team side

**Analysis Result**:
The team summary, timestamped estimated-event timeline, measurable-coverage information, and structured exports produced for an analysis clip. It does not include a separately rendered annotated video in the first version.
_Avoid_: Annotated output video, official match report

**Analysis Session**:
The temporary association between one uploaded analysis clip, its calibration choices, and its analysis result while the local demo is running. It is not a persistent match record.
_Avoid_: Saved match, analysis history

**Analysis Range**:
A user-selected, clip-relative interval of at least thirty seconds within a completed full-clip analysis. Its start is included and its end is excluded unless the end coincides with the physical end of the clip. It derives a report from already analyzed evidence and never causes the clip to be analyzed again.
_Avoid_: Playback range, trimmed clip, reanalysis window

**Range-Filtered Analysis Report**:
An Analysis Result view whose events and duration-based estimates are limited to an Analysis Range while retaining timestamps relative to the original clip.
_Avoid_: New analysis, excerpt timeline

**Clip Suitability**:
The pre-analysis assessment of whether a video satisfies the analysis-clip boundary and contains enough visible evidence for meaningful estimates. An unsuitable clip is rejected with reasons; a marginal clip may proceed with explicit warnings.
_Avoid_: Analysis accuracy, model confidence
