# molmoact2_maniskill — 🗓️ planned

Interactive playground for **MolmoAct2** in **ManiSkill** (its own native sim eval).

This is the most *turnkey* MolmoAct2 path: the repo already ships a closed-loop
`sim_eval` that talks to a FastAPI `/act` server, and the language instruction is
already a CLI flag. The catch is that **ManiSkill/SAPIEN requires an NVIDIA GPU + Vulkan**,
so the sim *and* the model server both run on the GPU pod and you watch via synced
video (no Mac-local window).

Reference repo (gitignored clone): `../molmo/molmoact2/sim_eval/`.

Planned build (mirrors [`pi05_libero/`](../pi05_libero/)):
- `setup.sh` — on the GPU pod: `uv sync` in `molmoact2`, download robot assets
  (`sim_eval/scripts/download_assets.py`) + the `MolmoAct2-DROID` checkpoint (~22 GB),
  start `examples/droid/host_server_droid.py`.
- `run.sh` / `app/` — wrap `sim_eval/run_eval.py` so the instruction is live
  (the hook is `run_eval.py:194`, where `instruction` is resolved) and stream the
  rollout frames to the browser instead of only writing `outputs/.../*.mp4`.

Not started yet — `molmobot_molmospaces/` first.
