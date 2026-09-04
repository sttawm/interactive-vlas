#!/usr/bin/env python3
"""Start frame, success frame and demo clip for an arbitrary LIBERO task list.

Generalises capture_frames.py to any suite: renders the t=0 agentview frame,
replays a human teleop demo's final sim state for a sharp "done" frame, and
resamples 32 demo states into a 4s clip. Demo hdf5s are pulled per task from the
HF mirror (the stock LIBERO downloader's hosting is dead).

  MUJOCO_GL=egl python3 make_media.py --tasks tasks.json --out /workspace/media
  tasks.json: [{"suite": "...", "task_id": N}, ...]
"""
import argparse, json, pathlib, subprocess
import numpy as np
from PIL import Image
import h5py
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RES, HF = 256, "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True); p.add_argument("--out", required=True)
    p.add_argument("--settle", type=int, default=10); p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dsroot = pathlib.Path("/workspace/libero_datasets"); dsroot.mkdir(exist_ok=True)
    suites = {}
    for t in json.load(open(a.tasks)):
        suite, tid = t["suite"], int(t["task_id"])
        tag = f"{suite}__{tid:02d}"
        if (out / f"{tag}.mp4").exists():
            print("skip", tag, flush=True); continue
        if suite not in suites: suites[suite] = benchmark.get_benchmark_dict()[suite]()
        task = suites[suite].get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        env.seed(a.seed); env.reset()
        inits = suites[suite].get_task_init_states(tid)
        obs = env.set_init_state(inits[0])
        for _ in range(a.settle): obs, _, _, _ = env.step([0.0]*6 + [-1.0])
        Image.fromarray(np.asarray(obs["agentview_image"][::-1], dtype=np.uint8)).save(out / f"{tag}_start.png")
        # demo -> done frame + clip
        demo = task.bddl_file.replace(".bddl", "_demo.hdf5")
        (dsroot / suite).mkdir(exist_ok=True)
        dst = dsroot / suite / demo
        if not dst.exists() or dst.stat().st_size == 0:
            for _ in range(3):
                if subprocess.run(["curl","-fL","--retry","3","-o",str(dst),
                                   f"{HF}/{suite}/{demo}"]).returncode == 0: break
        with h5py.File(dst, "r") as f:
            states = np.array(f["data"][sorted(f["data"].keys())[0]]["states"])
        fdir = out / f"frames_{tag}"; fdir.mkdir(exist_ok=True)
        for k, si in enumerate(np.linspace(0, len(states)-1, 32).astype(int)):
            o = env.set_init_state(states[si])
            Image.fromarray(np.asarray(o["agentview_image"][::-1], dtype=np.uint8)).save(fdir / f"{k:03d}.png")
        Image.fromarray(np.asarray(env.set_init_state(states[-1])["agentview_image"][::-1],
                                   dtype=np.uint8)).save(out / f"{tag}_done.png")
        subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","8","-i",str(fdir/"%03d.png"),
                        "-vf","scale=240:240","-c:v","libx264","-pix_fmt","yuv420p","-crf","30",
                        str(out / f"{tag}.mp4")], check=True)
        env.close(); print("made", tag, flush=True)
    print("MEDIA-DONE", flush=True)

if __name__ == "__main__":
    main()
