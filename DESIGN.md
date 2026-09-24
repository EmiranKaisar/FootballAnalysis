# Local Football Analysis — Product Design

## Product intent

Local Football Analysis is a private, on-device proof of concept for reviewing team-level football activity in a continuous tactical-camera clip of up to 20 minutes. It produces inspectable estimates rather than official match statistics. The first version favors conservative, explainable results over broad footage support or player identification.

## Goals

- Estimate team-level shots, shots on target, passes, pass success, possession, and measurable coverage.
- Keep private footage and derived analysis on the user's computer.
- Let the user inspect estimated events at their original clip-relative timestamps.
- Let the user inspect any valid 30-second-or-longer portion of a completed analysis without rerunning model inference.
- Process a valid 20-minute clip as one continuous analysis without requiring the user to split it.
- Make uncertainty and unsuitable evidence visible rather than silently guessing.

## Non-goals

- Identifying real-world players or producing player-level statistics.
- Producing an official match record or calibrated probability of correctness.
- Supporting broadcast edits, replays, close-ups, abrupt camera cuts, or arbitrary match videos.
- Rendering an annotated output video.
- Persisting an analysis history.
- Resuming an interrupted analysis after a server or machine restart in the first 20-minute release.
- Analyzing across halftime or another change in attacking direction.
- Exporting JSON or CSV in the current interface.

## Supported input

The product accepts one MP4 **Analysis Clip** at a time. A valid clip:

- lasts from 30 through 1,200 seconds;
- falls within one match half and does not contain a change in attacking direction;
- is no larger than 4 GB;
- has a resolution of at least 640×360;
- uses a fixed or smoothly moving wide-angle tactical view;
- is continuous live play without cuts, replays, or close-up edits; and
- normally shows most active players, the ball, and enough pitch markings for useful inference.

Preflight rejects unreadable files, unsupported formats, invalid metadata, clips outside the duration or file-size boundary, and undersized video. It warns about unusual frame rates, probable viewpoint cuts, and evidence suggesting an attacking-direction change. Sampling adapts to clip duration and retains only a bounded set of full-resolution frames; cut inspection retains compact derived evidence rather than every sampled image. Rejected clips are not analyzed.

## Primary workflow

1. The user uploads a private MP4 or chooses one from the local private-input folder.
2. The product inspects the clip and reports errors or warnings.
3. The user confirms the selected clip from a compact static opening-frame preview and calibrates the two teams:
   - confirms a name and jersey color for each team;
   - confirms opposite attacking directions; and
   - optionally accepts locally generated jersey-color suggestions.
4. The user starts analysis. The application processes the entire clip once, showing elapsed clip time, logical work-unit progress, measured throughput, device, estimated time remaining, and a cancellation control.
5. The completed Analysis Report presents team metrics, quality information, and a timestamped estimated-event timeline.
6. An Analysis Range Selector below the report lets the user derive a filtered report from cached, time-indexed full-clip evidence. Moving the selector never reruns inference and never seeks the video.

## Analysis behavior

The application samples the source at a target analysis rate and uses replaceable local object trackers for players, goalkeepers, and the ball. User-confirmed jersey appearance classifies temporary player tracks by team. An explicit temporal state machine derives possession intervals and estimated passes and shots.

The initial 20-minute implementation retains an analysis target of approximately eight frames per second. A maximum-duration clip therefore produces roughly 9,600 analyzed frames and, with both reference detectors enabled, approximately 19,200 inference calls. This target may change only after representative CPU, Metal, and CUDA benchmarks assess both processing time and event-quality impact.

### Long-clip processing

The product never requires the user to split an Analysis Clip. It opens one source through a continuous decoder and presents individual sampled frames to the trackers. For progress and operational control, the 20-minute job is divided into ten logical two-minute work units; these are not separate media files or independent analyses.

One tracker, classifier, and event engine remain alive for the complete clip. Logical boundaries must not reset or discard:

- ByteTrack tracking state;
- the current controller, team, or control candidate;
- pending pass or shot evidence;
- ball history, the last detected ball, or the short missing-ball bridge;
- possession accumulation and interval continuity; or
- shot cooldown and original clip-relative timestamps.

This continuity allows an event beginning before a logical boundary and resolving after it to be reported exactly once. Logical work units provide progress reporting, cancellation points, bounded diagnostic flushing, and performance measurement, but do not reduce inference work.

Only one inference job may be active per local server. A new request is rejected or queued while another job owns the worker, and repeated submissions from the same session cannot create duplicate work. Completed, failed, cancelled, and abandoned jobs are removed according to a bounded cleanup policy.

The first 20-minute release uses one continuous in-memory run and does not promise restart recovery. Later resume support may restart before the latest completed boundary, warm tracking state over an overlap, and deduplicate boundary evidence; that behavior requires a separate design decision and acceptance criteria.

### Reference models and replacement boundary

The reference configuration uses three upstream artifacts:

- [`martinjolif/yolo-football-player-detection`](https://huggingface.co/martinjolif/yolo-football-player-detection), pinned to revision [`5e83fafa8d564243001ce8e063612a618a138fbe`](https://huggingface.co/martinjolif/yolo-football-player-detection/tree/5e83fafa8d564243001ce8e063612a618a138fbe), for football-specific player, goalkeeper, referee, and fallback ball observations;
- the official [`yolo11n.pt` model from the Ultralytics assets v8.3.0 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) for supplementary generic person and sports-ball coverage; and
- the [`ultralytics/ultralytics`](https://github.com/ultralytics/ultralytics) runtime for local inference and ByteTrack-based tracking.

These models are replaceable reference defaults. The interface accepts an alternative local Ultralytics-compatible primary model and an optional secondary ball/coverage model. A one-model configuration must cover every analysis object it intends to provide. In the two-model configuration, the primary model provides football-specific roles while the secondary model provides generic person and ball coverage.

The canonical tracker output consists of a stable track ID, canonical object label, confidence score, and pixel-space `(x1, y1, x2, y2)` bounding box. Accepted label aliases normalize to **player**, **goalkeeper**, **ball**, or **official**. Unrecognized classes are ignored, and team assignment remains a separate jersey-color classification step.

An inference runtime other than Ultralytics is replaceable behind the `ObjectTracker` boundary but is not selectable from the current interface. Such a provider must adapt its detections and tracking state to the canonical tracker output. The analyzer accepts an injected tracker so alternative providers can be tested independently before interface-level provider selection is added.

The default downloader owns only the two pinned reference filenames and restores them when their checksums differ. Custom weights use distinct filenames and remain the responsibility of the user. Every replacement model and runtime retains its own license; the project's MIT license does not relicense third-party artifacts.

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

Analysis runs as a local background job. During inference, the interface reports elapsed source time, total source duration, current two-minute logical work unit, measured analyzed frames per second, selected device, and an estimated completion time derived from the current run. Cancellation finishes the current model frame, releases decoder and model-job resources, then reports cancellation. Failures are shown locally without remote error logging. Partial diagnostic state may be retained for troubleshooting, but it is not presented as a completed Analysis Result. The range selector is unavailable for queued, running, failed, or cancelled jobs.

## Privacy and storage

After initial model setup, inference and video processing occur locally. Uploaded clips are copied to the ignored private-input directory only after file-size and available-disk checks succeed. Multi-gigabyte clips placed directly in the private-input directory avoid browser-upload duplication and are the preferred local-server workflow. Match footage and derived artifacts must not be committed to the repository or uploaded by the application. Temporary, abandoned, and superseded uploads are removed according to a documented age and ownership policy. An Analysis Session is temporary and is not a saved match or analysis-history record.

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

## Acceptance criteria for model replacement

- A user can select an alternative compatible Ultralytics model by local path without modifying application code.
- A one-model configuration works when the selected model supplies the required canonical objects.
- A two-model configuration distinguishes the football-role model from the supplementary generic player/ball model.
- Unsupported class names are ignored rather than silently mapped to an unrelated analysis object.
- A non-Ultralytics provider can be tested by injecting an `ObjectTracker` implementation that returns the canonical tracker output.
- Documentation identifies the exact reference repositories, pinned model revision or release, replacement contract, and third-party licensing boundary.

## Acceptance criteria for 20-minute analysis

- A valid clip lasting exactly 20 minutes is accepted; a clip beyond the defined metadata tolerance is rejected.
- The complete source is decoded continuously and is not physically divided into temporary video files.
- The interface reports ten logical two-minute work units while tracker and event-engine state reset only once, before the full analysis.
- A pass, shot, ball-history window, or possession interval crossing a logical boundary remains continuous and is counted exactly once.
- Approximately eight source frames per second are analyzed until benchmark and accuracy evidence authorizes another rate.
- Only one inference job can own the local worker, and repeated submissions do not create duplicate jobs.
- Progress includes elapsed source time, logical work-unit count, measured throughput, selected device, and estimated time remaining.
- Cancellation releases the video decoder and job resources after the current model frame.
- Preflight sampling remains memory-bounded as duration increases and covers the full 20-minute source.
- A completed 20-minute result supports immediate Analysis Range filtering without invoking either model again.
- Browser interaction with approximately 9,600 analyzed timestamps remains responsive.
- Clean-machine testing covers representative CPU, Metal, and CUDA systems, multi-gigabyte input, insufficient disk space, cancellation, and server interruption.
- Restart resume is not advertised in the first 20-minute release.

## Related domain and architecture documents

Canonical football and product terms are defined in [`CONTEXT.md`](./CONTEXT.md). Architectural trade-offs are recorded in [`docs/adr/`](./docs/adr/).
