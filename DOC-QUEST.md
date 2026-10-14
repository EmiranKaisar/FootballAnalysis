# Doc Quest: Analysis Workspace Redesign

**Status:** Complete

## Questions

No questions remain.

## Established Understanding

- Upload and preflight happen before entering the analysis workspace. Once the selected clip is valid, its video appears in the main, centered player area. Team names, jersey colors, and attacking directions live in a compact, collapsible setup section below the player.
- On wide screens, the player and a right-side panel form a two-column workspace. On narrow screens, the player stacks above the panel. The panel retains its two tabs: **Analysis** and **Events**. Analysis shows both teams' statistics; Events lists detected events.
- The Analysis Range control sits directly below the player and is visually distinct from the playback seek bar. It filters the completed report using cached analysis data and does not rerun inference.
- Playback remains available over the full clip regardless of the selected Analysis Range. Selecting an event seeks to its clip-relative timestamp, pauses playback, and marks the event selected without changing the range.
- The term **Clip Preview** is replaced by **Analysis Video Player**. **Selected Event** is added to the glossary. The existing local-processing, team-level, event-definition, and range-calculation decisions remain in force.
- “Like GameNote” is interpreted as the visual hierarchy of a central primary player beside a right rail; no pixel-level reference was supplied.

The user confirmed this synthesis in Q6.

## Answered Questions

### Q6 — Confirm the complete workspace design

**Decision:** _Confirmed._

**Rationale / constraints:** The synthesis accurately captures the five prior decisions.

### Q1 — Setup and calibration in the two-pane workspace

**Question:** Once a clip passes preflight, where should clip details and team calibration appear?

**Why it matters:** The current design puts upload, preview, and calibration in a vertical setup flow, while the proposed center player and right sidebar are an analysis workspace. The placement determines whether setup competes with the video or the team metrics.

**Recommendation:** Keep upload and preflight before the workspace; show the selected clip in the center player as soon as it is valid; put team names, jersey colors, and attack directions in a compact, collapsible setup section below the player. The right sidebar can then remain dedicated to analysis and events.

**Decision:** _Keep upload and preflight before the workspace; show the selected clip in the center player as soon as it is valid; put team names, jersey colors, and attack directions in a compact, collapsible setup section below the player. The right sidebar can then remain dedicated to analysis and events._

**Rationale / constraints:** None provided.

### Q2 — Analysis Range placement

**Question:** Where should the existing Analysis Range selector appear in the new layout?

**Why it matters:** A video player's seek bar changes playback position, while the Analysis Range changes which cached statistics and events are included. Placing them too close or giving them similar visual treatment could make users confuse the two.

**Recommendation:** Put a clearly labeled Analysis Range control directly below the video, visually separate from the player's playback controls. Keep its current behavior: it filters the completed report and does not rerun inference.

**Decision:** _Put a clearly labeled Analysis Range control directly below the video, visually separate from the player's playback controls. Keep its current behavior: it filters the completed report and does not rerun inference._

**Rationale / constraints:** None provided.

### Q3 — Playback outside the selected Analysis Range

**Question:** When the user selects a shorter Analysis Range, should the video player still allow playback and seeking across the entire clip?

**Why it matters:** The Analysis Range currently filters report data only. Restricting playback would make it harder to review the surrounding play, while full-clip playback could show moments that are not represented in the filtered report.

**Recommendation:** Keep playback available for the entire clip. Treat the Analysis Range as a report filter only, and make the selected range visible in its own control and the report heading.

**Decision:** _Keep playback available for the entire clip. Treat the Analysis Range as a report filter only, and make the selected range visible in its own control and the report heading._

**Rationale / constraints:** None provided.

### Q4 — Selecting an event

**Question:** When a user clicks an event in the Events tab, what should happen to playback?

**Why it matters:** Seeking to the event is specified, but whether playback starts, pauses, or keeps running affects review accuracy and the meaning of the selected event.

**Recommendation:** Seek to the event's clip-relative timestamp, pause there, and visually mark that event as selected. Do not change the Analysis Range. Use the nearest available analyzed frame if the event timestamp does not exactly match a decoded frame.

**Decision:** _Seek to the event's clip-relative timestamp, pause there, and visually mark that event as selected. Do not change the Analysis Range. Use the nearest available analyzed frame if the event timestamp does not exactly match a decoded frame._

**Rationale / constraints:** None provided.

### Q5 — Narrow-window layout

**Question:** What should happen when the browser is too narrow to show a useful video and right sidebar side by side?

**Why it matters:** Preserving a literal right rail at every width can shrink the player and event details until neither is usable.

**Recommendation:** Keep the two-column layout on wide screens; below a practical breakpoint, stack the player above the sidebar while keeping the Analysis and Events tabs. This preserves the same information and avoids hiding controls.

**Decision:** _Keep the two-column layout on wide screens; below a practical breakpoint, stack the player above the sidebar while keeping the Analysis and Events tabs. This preserves the same information and avoids hiding controls._

**Rationale / constraints:** None provided.
