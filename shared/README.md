# shared

Code shared across the interactive VLA instances. The headline piece is the **one web
page** every instance serves.

- `webui.py` — the policy/sim-agnostic Flask app + browser frontend: a live MJPEG camera
  stream, an instruction text box, and **VLA · Env · Scene selectors** at the top. It talks
  to a small backend contract (`config()`, `snapshot_status()`, `latest_jpeg()`,
  `set_instruction()`, `request_reset(selection)`, `set_paused()`), so each instance only
  implements that contract for its own VLA + simulator and gets the identical page.

The selectors only advertise what the *running* server can do (a MolmoBot pod won't list
LIBERO), via the `config()` the instance returns. Switching VLA/scene posts the selection to
`/reset`, which the backend acts on.

Used by:
- [`molmobot_molmospaces/`](../molmobot_molmospaces/) — ✅ on the shared UI.
- [`pi05_libero/`](../pi05_libero/) — has its own copy of this page; migration onto
  `shared/webui.py` is pending (needs a pod to retest the working instance safely).
- `molmoact2_*` — will use it when built.
