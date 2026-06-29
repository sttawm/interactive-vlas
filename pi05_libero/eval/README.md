# pi0.5 · LIBERO evaluation

Two questions, one harness (`libero_eval.py`), run against a live policy server
(`scripts/serve_policy.py --env LIBERO` on :8000). **Pause/stop the web UI first** so
it doesn't contend with the eval for the single-threaded server.

`libero_eval.py` mirrors OpenPI's `examples/libero/main.py` exactly — object-settle,
image rotate+resize_with_pad(224), `[eef_pos, axisangle(eef_quat), gripper_qpos]` state,
replan every 5 — so the canonical numbers are comparable to the published benchmark.

## 1. Reproduce in-distribution (canonical tasks)

```bash
cd /workspace/interactive-vlas/pi05_libero/eval
LV=/workspace/openpi/examples/libero/.venv
PYTHONPATH=/workspace/openpi/third_party/libero MUJOCO_GL=egl \
  $LV/bin/python libero_eval.py --suite libero_goal --trials 5 \
    --prompt-mode canonical --out results_goal_canonical.json
```
This runs the **same tasks the model was trained on**, from the tasks' init states —
i.e. in-distribution. Expect it to approach the published numbers (LIBERO avg ~96.8%).
`--trials` is rollouts/task (benchmark uses 50; 5–10 gives a quick signal). `--max-tasks N`
limits to the first N tasks for speed.

## 2. Language generalization (paraphrases)

Same scenes and init states, but the instruction is reworded. Run **paired** so wording
is the only variable:

```bash
PYTHONPATH=/workspace/openpi/third_party/libero MUJOCO_GL=egl \
  $LV/bin/python libero_eval.py --suite libero_goal --trials 5 \
    --prompt-mode both --paraphrases paraphrases_libero_goal.json \
    --out results_goal_paired.json
```
`--prompt-mode both` runs canonical *and* paraphrase from each init state, so the output
reports both success rates side by side. The gap = the language-generalization cost.

`paraphrases_<suite>.json` maps `{canonical_instruction: paraphrase}` (keyed by the exact
`task.language` string, so it's robust to task ordering). Generate it from the actual
task strings; missing entries fall back to canonical (with a warning).
