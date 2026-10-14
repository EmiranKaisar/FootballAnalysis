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
3. After a successful preflight, the Analysis Workspace opens with the selected clip in the central video player and a right-side panel. The user opens the collapsible Clip Setup section below the player to calibrate the two teams:
   - confirms a name and jersey color for each team;
   - confirms opposite attacking directions; and
   - optionally accepts locally generated jersey-color suggestions.
4. The user starts analysis from Clip Setup. The application processes the entire clip once and shows elapsed clip time, logical work-unit progress, measured throughput, device, estimated time remaining, and a cancellation control in the Analysis tab.
5. On completion, the Analysis tab presents the two teams' metrics and quality information; the Events tab presents the timestamped estimated events. Selecting an event seeks the player to its reference frame, pauses playback, and marks the event selected.
6. The Analysis Range control appears directly below the video controls. It filters the completed analysis and Events list from cached, time-indexed full-clip evidence. Changing the range never reruns inference or seeks the player.

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

## Analysis Workspace

After preflight succeeds, the workspace presents the Analysis Video Player in the main area and a right-side panel. On wide screens, the player is centered in the main column beside the panel. On narrow screens, the player appears above the panel; the panel retains the same tabs and content.

The right-side panel has two tabs:

- **Analysis** is the initial tab. While analysis is running, it shows job progress and cancellation. After completion, it shows the Analysis Report: measurable possession coverage, analyzed-frame count, quality warnings, and metric cards for both teams (shots, shots on target, passes, pass-success rate, and possession rate).
- **Events** shows estimated events in chronological order. Each selectable event identifies its clip-relative timestamp, event type, team, and outcome. Before results are available, the tab shows an empty or not-yet-analyzed state.

Selecting an event seeks to its source-video reference frame, pauses playback, and marks that event as selected. If its timestamp does not match an analyzed-frame timestamp, the nearest analyzed frame is used. Selecting an event does not change the Analysis Range. Playback remains available over the full Analysis Clip even when the Analysis Range is shorter.

The completed Analysis Report heading identifies the Analysis Range start, end, and selected duration. If no event meets the conservative evidence rules, the Events tab explains that no qualifying event was found. If no measurable possession exists in the selected range, each team's possession is shown as **N/A**, coverage is `0.0%`, event counts remain valid, and the Analysis tab explains that the interval lacks sufficient visible evidence.

## Analysis Video Player

The Analysis Video Player displays the selected local Analysis Clip in the main workspace after preflight succeeds and remains associated with the Analysis Session. It supports ordinary playback and seeking across the complete clip. Seeking the video does not change the Analysis Range or the displayed metrics. The player uses the local clip and does not upload video to a remote service.

Clip selection and preflight remain before the workspace. Team names, jersey colors, and attacking directions are in a compact, collapsible **Clip Setup** section below the player. The user can confirm color suggestions and start analysis there. After analysis completes, the Analysis Range control takes the position immediately below the playback controls; Clip Setup remains available below it in its collapsed state.

If the browser cannot play the selected clip despite successful preflight, the player shows a concise **Video playback unavailable** message. This does not block local analysis or access to the Analysis and Events tabs.

## Analysis Range Selector

### Placement and presentation

The selector appears only after full-clip analysis completes. It is directly below the Analysis Video Player's playback controls and visually distinct from the player's seek bar. It is not sticky to the viewport. The collapsed Clip Setup section follows it.

It resembles a video progress bar but is not a playback control. Two draggable handles define the selected interval. The segment between them uses the product accent color; excluded segments are visually muted. A clip-relative `MM:SS.s` timestamp remains visible above each handle, with the active handle emphasized during dragging.

The Analysis Report heading uses the same timestamp format and also shows selected duration.

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
- The Analysis tab and Events list update when the handle is released, without a model rerun or expensive report reconstruction during pointer movement.
- Dragging a handle does not seek or otherwise change video playback.
- Each handle is keyboard-focusable. Left and Right move one analyzed-frame step; Shift+Left and Shift+Right move one second while respecting the clip boundary and minimum duration.
- Accessible handle names announce whether the boundary is the start or end, its current timestamp, and the 30-second constraint.

### Boundary semantics

The Analysis Range is start-inclusive and end-exclusive: `[start, end)`. When the end handle is at the physical end of the clip, evidence timestamped exactly at that endpoint is included. All displayed event timestamps remain relative to the original clip and are never rebased to the selected start.

## Report filtering

On handle release, the Analysis tab and Events list are derived for the selected range:

- shots, shots on target, passes, pass outcomes, and visible Events list entries are filtered by event timestamp;
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
- The Analysis and Events tabs, event list, and event selection are keyboard-operable; selected-event state is conveyed without relying on color alone.
- Focus state and selected/excluded track regions have distinguishable visual treatment.
- Timestamp labels do not rely on color alone.
- Status, validation, empty-evidence, cancellation, and failure messages use text.
- The 30-second constraint is enforced during pointer, touch, and keyboard interaction rather than reported only after submission.

## Acceptance criteria for range filtering

- Completing analysis of a valid clip shows full-clip metrics in the Analysis tab and a full-span selector directly below the player controls.
- Selecting a valid range updates the Analysis tab and Events list on handle release without calling the analyzer again or changing player position.
- An event on the start boundary is included; an event on an internal end boundary is excluded.
- Possession intervals crossing either boundary contribute only their overlapping duration.
- The handles cannot produce a range shorter than 30 seconds or cross each other.
- Resetting, selecting a different clip, and starting a new analysis follow the state rules above.
- A range with no measurable possession shows **N/A**, not `0.0%`, for possession.
- No JSON or CSV download control appears.

## Acceptance criteria for clip presentation

- After successful preflight, the workspace shows the Analysis Video Player beside the Analysis/Events panel on a wide window and stacks the player above the panel on a narrow window.
- The Analysis and Events tabs remain available before results; after completion, Analysis shows both teams' metrics and Events lists estimated events chronologically.
- Each event identifies its clip-relative timestamp, type, team, and outcome; selecting it seeks to the reference frame, pauses playback, and visibly marks the selected event.
- The player remains seekable across the full clip regardless of the Analysis Range; selecting an event does not alter that range.
- Clip Setup is available in a compact, collapsible section below the player. After analysis completes, the Analysis Range sits between the player controls and collapsed Clip Setup.
- If browser playback is unavailable, a concise warning appears while local analysis and both tabs remain usable.

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
