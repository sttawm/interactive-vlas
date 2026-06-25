# molmo_shared

Code shared across the Molmo interactive instances
([`molmobot_molmospaces/`](../molmobot_molmospaces/),
[`molmoact2_maniskill/`](../molmoact2_maniskill/),
[`molmoact2_molmospaces/`](../molmoact2_molmospaces/)).

The interactive pattern is the same for every instance (it mirrors the working
[`pi05_libero/`](../pi05_libero/) app): a background worker owns the simulator and
steps a VLA against a **live, user-typed instruction**, while a small Flask app
streams the rendered camera view to the browser and exposes the prompt as a text box.

What is intended to live here (factored out as the instances are built):

- `webviewer.py` — the Flask app + MJPEG/`/frame.jpg` browser viewer + instruction box
  (the `build_app(...)` + `INDEX_HTML` half of the pi05 app, made policy-agnostic).
- `runlog.py` — per-rollout `runs/` logging: `instructions.txt`, `rollout.mp4`,
  `actions.npy`, `metadata.json`.
- `worker.py` — the threaded rollout-worker scaffolding (lock-guarded shared state,
  reset/pause/instruction plumbing) with the sim-specific step left abstract.

Each instance keeps only its **sim + policy glue** (how to build the env, what the
observation dict looks like, how to render a frame) and reuses the above.

> Built incrementally: the first instance (`molmobot_molmospaces/`) may inline some of
> this; pieces move here once a second instance needs them, to avoid premature abstraction.
