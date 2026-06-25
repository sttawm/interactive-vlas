# shared

Code shared across the interactive VLA instances. The headline piece is the **one web
page** every instance serves.

`webui.py` is policy/sim-agnostic: a live MJPEG camera stream, an instruction text box,
**config-driven cascading selectors**, play/pause, and a captioned **Save video** export.
Each instance implements a small **worker contract** and gets the identical page.

## Worker contract

```python
config() -> dict
    {
      "title": str,
      "selectors": [                                  # ordered, cascading dropdowns
        {"name": "suite", "label": "Suite", "options": ["libero_10", ...]},
        {"name": "task",  "label": "Task",  "depends_on": "suite",
         "options_by": {"libero_10": [{"value": "0", "label": "Task 0 — ...",
                                       "default_prompt": "..."}], ...}},
      ],
      "instruction_label": str,        # optional
      "instruction_placeholder": str,  # optional
    }
snapshot_status() -> dict        # free-form; rendered verbatim as "key: value" rows.
                                 #   keys "paused"/"limit_reached" (bool) are consumed by
                                 #   the Play button + step-limit handling, not displayed.
latest_jpeg() -> bytes
set_instruction(text) -> None
request_reset(selection, instruction="") -> None   # selection = {selector_name: value}
set_paused(paused: bool) -> None
save_video(name, speed) -> path | None             # optional; enables the Save-video button
```

- **Selectors** are generic and cascade: a selector with `depends_on` reads its options
  from `options_by[<parent value>]`, so the whole catalog ships in one `/config` call.
  pi05 supplies `suite → task`; MolmoBot supplies `vla → env → scene`. An option's
  `default_prompt` prefills the blank instruction.
- **`request_reset(selection, instruction="")`** lands paused & ready; resolve the default
  prompt synchronously from `selection` so a `Send` during env load isn't clobbered.
- **`save_video`** can call the reusable `webui.compose_video(frames, prompts, name, speed,
  runs_dir)` helper (captions the active prompt under each frame), or just return an
  already-written per-run `rollout.mp4`.

`webui.serve()` isn't used — each instance starts its own worker thread and calls
`app = webui.build_app(worker); app.run(...)` (see `pi05_libero/app/interactive_libero.py`).

## Status

- [`pi05_libero/`](../pi05_libero/) — ✅ on this contract (reference implementation).
- [`molmobot_molmospaces/`](../molmobot_molmospaces/) — needs migration from the old
  `{vlas, envs, scenes}` config to the `selectors` config + `request_reset(selection,
  instruction)` + optional `save_video`. The frontend, status rendering, play/pause, and
  MJPEG are unchanged.
- `molmoact2_*` — will use it when built.
