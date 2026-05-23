Read the following documents to establish full context on the project's current architecture and model design, then present the information clearly.

Documents to read (in order):
1. `docs/Research_Ideas_Writeup.md` — sections 3.1 (what the model addresses), 3.2 (architecture diagram), 3.3 (training overview), 3.4 (innovation claims)
2. `docs/design.tex` — Architecture Design section and Representation Learning bullets
3. `CLAUDE.md` — Architecture section: multi-branch representation table, aggregator modes, adding new features guide

After reading, present:
- The current named branch set, their source modules, and their output dimensions
- The two aggregator modes (concat vs gated) and when each is appropriate
- The three downstream tasks (price prediction, volatility prediction, trend classification) and their metrics
- Any components or methods explicitly marked TBD or open for future expansion

If the user asks a specific design question, answer it using information drawn from these documents rather than assumptions.
