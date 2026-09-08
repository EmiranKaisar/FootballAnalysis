# Local Football Analysis — Product Design

## Product intent

Local Football Analysis is a private, on-device proof of concept for reviewing team-level football activity in a short tactical-camera clip. It produces inspectable estimates rather than official match statistics. The first version favors conservative, explainable results over broad footage support or player identification.

## Goals

- Estimate team-level shots, shots on target, passes, pass success, possession, and measurable coverage.
- Keep private footage and derived analysis on the user's computer.
- Let the user inspect estimated events at their original clip-relative timestamps.
- Let the user inspect any valid 30-second-or-longer portion of a completed analysis without rerunning model inference.
- Make uncertainty and unsuitable evidence visible rather than silently guessing.

## Non-goals

- Identifying real-world players or producing player-level statistics.
- Producing an official match record or calibrated probability of correctness.
- Supporting broadcast edits, replays, close-ups, abrupt camera cuts, or arbitrary match videos.
- Rendering an annotated output video.
- Persisting an analysis history.
- Exporting JSON or CSV in the current interface.

## Supported input

The product accepts one MP4 **Analysis Clip** at a time. A valid clip:

- lasts from 30 through 120 seconds;
- has a resolution of at least 640×360;
- uses a fixed or smoothly moving wide-angle tactical view;
- is continuous live play without cuts, replays, or close-up edits; and
- normally shows most active players, the ball, and enough pitch markings for useful inference.

Preflight rejects unreadable files, unsupported formats, invalid metadata, clips outside the duration boundary, and undersized video. It warns about unusual frame rates and probable viewpoint cuts. Rejected clips are not analyzed.

## Primary workflow

1. The user uploads a private MP4 or chooses one from the local private-input folder.
2. The product inspects the clip and reports errors or warnings.
3. The user confirms the selected clip from a compact static opening-frame preview and calibrates the two teams:
   - confirms a name and jersey color for each team;
   - confirms opposite attacking directions; and
   - optionally accepts locally generated jersey-color suggestions.
4. The user starts analysis. The application processes the entire clip once, showing stage progress and allowing cancellation.
5. The completed Analysis Report presents team metrics, quality information, and a timestamped estimated-event timeline.
6. An Analysis Range Selector below the report lets the user derive a filtered report from cached, time-indexed full-clip evidence. Moving the selector never reruns inference and never seeks the video.

## Analysis behavior

The application samples the source at a target analysis rate and uses replaceable local object trackers for players, goalkeepers, and the ball. User-confirmed jersey appearance classifies temporary player tracks by team. An explicit temporal state machine derives possession intervals and estimated passes and shots.

The completed full-clip result retains:

- clip-relative event timestamps;
- time-indexed eligibility and possession intervals;
- analyzed-frame timestamps;
- team calibration and source metadata; and
- warnings about evidence quality.

This time-indexed evidence is the only source for range-filtered reports. Range selection does not invoke detection, tracking, or event inference again.

## Analysis Report

The report heading identifies its clip-relative start, end, and selected duration. The report contains:

- measurable possession coverage and analyzed-frame count;
- per-team shots and shots on target;
- per-team passes and pass-success rate;
- per-team possession rate; and
- a timestamped estimated-event timeline.

The timeline is the complete event-review interface in this version. It does not include an event selector, event-specific image, or video player.

For a filtered report, event counts include only events within the Analysis Range. Duration-based metrics clip eligibility and possession intervals at the range boundaries and recompute their values from the overlapping durations.

When no event meets the conservative evidence rules, the timeline explains that no qualifying event was found. When no measurable possession exists in the selected range, each team's possession is shown as **N/A**, coverage is `0.0%`, event counts remain valid, and the report explains that the interval lacks sufficient visible evidence.

## Clip Preview

The interface contains exactly one Clip Preview. It is a left-aligned static image of the Analysis Clip's opening frame, positioned after preflight metadata, errors, and warnings and before Calibration. It is not a video player and has a maximum displayed width of approximately 360 pixels so it does not compete visually with the Analysis Report. It remains visible throughout the temporary Analysis Session so the clip associated with the report stays identifiable. Its caption follows `<filename> · opening frame`. The user may expand the still image to fullscreen for inspection, but the preview provides no playback behavior or playback controls.

If the opening frame cannot be decoded despite otherwise successful preflight, the preview area is omitted and a concise **Opening-frame preview unavailable** warning is shown. This preview failure does not block Calibration or analysis.

## Analysis Range Selector

### Placement and presentation

The selector appears only after full-clip analysis completes. Its document position is beneath the Analysis Report, and it remains sticky at the bottom of the viewport while the report is being viewed.

It resembles a video progress bar but is not a playback control. Two draggable handles define the selected interval. The segment between them uses the product accent color; excluded segments are visually muted. A clip-relative `MM:SS.s` timestamp remains visible above each handle, with the active handle emphasized during dragging.

The report heading uses the same timestamp format and also shows selected duration.

### Initial and reset state

- The initial range spans the full clip.
- **Reset to full clip** restores both handles to the clip boundaries.
- Ordinary UI rerenders preserve the current range within the Analysis Session.
- Selecting another clip or starting a new full analysis resets the range to the full clip.

### Interaction rules

- Handles snap to analyzed-frame timestamps; the UI does not imply precision beyond the cached evidence.
- Handles cannot cross.
- The selected duration must be at least 30 seconds.
- Handle timestamps update continuously while dragging.
- The report updates immediately when the handle is released, avoiding a model rerun and avoiding expensive report reconstruction during pointer movement.
- Dragging a handle does not seek or otherwise change video playback.
- Each handle is keyboard-focusable. Left and Right move one analyzed-frame step; Shift+Left and Shift+Right move one second while respecting the clip boundary and minimum duration.
- Accessible handle names announce whether the boundary is the start or end, its current timestamp, and the 30-second constraint.

### Boundary semantics

The Analysis Range is start-inclusive and end-exclusive: `[start, end)`. When the end handle is at the physical end of the clip, evidence timestamped exactly at that endpoint is included. All displayed event timestamps remain relative to the original clip and are never rebased to the selected start.

## Report filtering

On handle release, every visible report value is derived for the selected range:

- shots, shots on target, passes, pass outcomes, and the event timeline are filtered by event timestamp;
- possession and measurable coverage are recomputed from overlapping cached intervals;
- analyzed-frame count includes cached analyzed-frame timestamps within the boundary rule; and
- warnings remain visible because they qualify the underlying full-clip evidence.

The original full-clip result remains available in memory so resetting the selector is immediate.

## Progress, cancellation, and failures

Analysis runs as a local background job. The interface reports its current stage and progress. Cancellation finishes the current model frame, then stops the job and reports cancellation. Failures are shown locally without remote error logging. The range selector is unavailable for queued, running, failed, or cancelled jobs.

## Privacy and storage

After initial model setup, inference and video processing occur locally. Uploaded clips are copied to the ignored private-input directory. Match footage and derived artifacts must not be committed to the repository or uploaded by the application. An Analysis Session is temporary and is not a saved match or analysis-history record.

## Accessibility and interaction quality

- Pointer use is not required for range adjustment.
- Focus state and selected/excluded track regions have distinguishable visual treatment.
- Timestamp labels do not rely on color alone.
- Status, validation, empty-evidence, cancellation, and failure messages use text.
- The 30-second constraint is enforced during pointer, touch, and keyboard interaction rather than reported only after submission.

## Acceptance criteria for range filtering

- Completing analysis of a valid clip shows the full-clip report and a full-span selector beneath it.
- Selecting a valid range updates every report output on handle release without calling the analyzer again.
- An event on the start boundary is included; an event on an internal end boundary is excluded.
- Possession intervals crossing either boundary contribute only their overlapping duration.
- The handles cannot produce a range shorter than 30 seconds or cross each other.
- Resetting, selecting a different clip, and starting a new analysis follow the state rules above.
- A range with no measurable possession shows **N/A**, not `0.0%`, for possession.
- No JSON or CSV download control appears.

## Acceptance criteria for clip presentation

- Exactly one Clip Preview appears for a successfully decoded Analysis Clip.
- The preview is a left-aligned opening-frame still no wider than approximately 360 pixels and is captioned with the filename.
- No video player or event-review selector appears before or after analysis.
- Estimated events remain inspectable as clip-relative timestamps in the timeline.
- Failure to decode the opening frame shows a concise warning and does not block an otherwise suitable clip.

## Related domain and architecture documents

Canonical football and product terms are defined in [`CONTEXT.md`](./CONTEXT.md). Architectural trade-offs are recorded in [`docs/adr/`](./docs/adr/).
