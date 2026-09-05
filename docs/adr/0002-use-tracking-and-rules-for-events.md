# Use tracking and explicit rules for first-version event detection

The first demo will infer passes, pass outcomes, possession, shots, and shot outcomes from detected player and ball tracks, user-confirmed teams and attacking directions, pitch geometry, and an explicit temporal state machine. We accept lower generality than a trained temporal action-spotting model because this approach is inspectable, tunable on a small number of clips, and practical on the local Apple M1; SoccerNet-style learned action spotting remains a future replacement behind the analysis boundary.
