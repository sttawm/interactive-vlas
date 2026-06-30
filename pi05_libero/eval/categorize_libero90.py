#!/usr/bin/env python3
"""Categorize each LIBERO-90 task by how novel it is vs the 40 finetuning tasks
(libero_spatial/object/goal/10), then (if results exist) report success per bucket.

Novelty is read OBJECTIVELY from each task's BDDL goal predicates:
  - actions  = the set of goal predicate types  (On, In, Open, Close, Turnon, Stack, Turnoff, ...)
  - objects  = the categories of every argument of those predicates (moved objects + receptacles),
               with trailing instance ids stripped (akita_black_bowl_1 -> akita_black_bowl)

A LIBERO-90 task is then bucketed (priority order):
  new action + new object  | new action | new object | new scene / composition only
where "new" = not present anywhere in the 40 training tasks.

Usage:  python categorize_libero90.py [results_libero90_finetuned.json]
"""
import json, pathlib, re, sys, collections
from libero.libero import benchmark, get_libero_path

TRAIN_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
BDDL = pathlib.Path(get_libero_path("bddl_files"))
bd = benchmark.get_benchmark_dict()

PRED_RE = re.compile(r"\(([A-Za-z]+)((?:\s+[A-Za-z0-9_]+)*)\)")


def parse_goal(task):
    """Return (set_of_predicates, set_of_object_categories) from a task's BDDL goal."""
    txt = (BDDL / task.problem_folder / task.bddl_file).read_text()
    i = txt.find("(:goal")
    g = txt[i:txt.find("(:", i + 6)] if i >= 0 else ""
    preds, objs = set(), set()
    for name, args in PRED_RE.findall(g):
        if name in ("goal", "And", "and"):
            continue
        preds.add(name)
        for a in args.split():
            objs.add(re.sub(r"_\d+$", "", a))            # strip instance id
            objs.add(re.sub(r"(_\d+)?(_region|_init_region)$", "", a))  # strip region suffix too
    return preds, objs


def scene_of(task):
    m = re.match(r"^([A-Z_]+SCENE\d+)", task.bddl_file)
    return m.group(1) if m else None


# --- build training inventory ---
TRAIN_PREDS, TRAIN_OBJS, TRAIN_SCENES = set(), set(), set()
for s in TRAIN_SUITES:
    suite = bd[s]()
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        p, o = parse_goal(t)
        TRAIN_PREDS |= p
        TRAIN_OBJS |= o
        sc = scene_of(t)
        if sc:
            TRAIN_SCENES.add(sc)

# --- categorize libero_90 ---
suite = bd["libero_90"]()
rows = []
for tid in range(suite.n_tasks):
    t = suite.get_task(tid)
    preds, objs = parse_goal(t)
    new_acts = sorted(preds - TRAIN_PREDS)
    new_objs = sorted(objs - TRAIN_OBJS - {""})
    sc = scene_of(t)
    new_scene = sc not in TRAIN_SCENES
    if new_acts and new_objs:
        cat = "new action + new object"
    elif new_acts:
        cat = "new action"
    elif new_objs:
        cat = "new object"
    else:
        cat = "new scene / composition only"
    rows.append({"task_id": tid, "lang": str(t.language), "scene": sc, "category": cat,
                 "new_actions": new_acts, "new_objects": new_objs, "new_scene": new_scene})

# --- optional: join with results ---
succ = {}
if len(sys.argv) > 1 and pathlib.Path(sys.argv[1]).exists():
    res = json.load(open(sys.argv[1]))
    for r in res.get("per_task", []):
        c = r["success"].get("canonical", "0/0")
        k, n = (int(x) for x in c.split("/"))
        succ[r["task_id"]] = (k, n)

print("TRAIN inventory: %d scenes, %d preds %s, %d obj-cats"
      % (len(TRAIN_SCENES), len(TRAIN_PREDS), sorted(TRAIN_PREDS), len(TRAIN_OBJS)))
print("=" * 92)
agg = collections.OrderedDict()
for cat in ["new scene / composition only", "new object", "new action", "new action + new object"]:
    agg[cat] = [0, 0, 0]   # tasks, successes, trials
for r in rows:
    mark = ""
    if r["new_actions"]:
        mark += " A:" + ",".join(r["new_actions"])
    if r["new_objects"]:
        mark += " O:" + ",".join(r["new_objects"])
    s = ""
    if r["task_id"] in succ:
        k, n = succ[r["task_id"]]
        agg[r["category"]][0] += 1
        agg[r["category"]][1] += k
        agg[r["category"]][2] += n
        s = "%d/%d" % (k, n)
    print("%2d  %-5s %-28s %4s  %-46s%s"
          % (r["task_id"], r["scene"] or "-", r["category"], s, r["lang"][:46], mark))

print("=" * 92)
print("%-30s %6s %8s %s" % ("CATEGORY", "#tasks", "success", "(succ/trials)"))
for cat, (nt, k, n) in agg.items():
    if n:
        print("%-30s %6d %7.1f%%  (%d/%d)" % (cat, nt, 100.0 * k / n, k, n))
    else:
        ntotal = sum(1 for r in rows if r["category"] == cat)
        print("%-30s %6d   (no results yet; %d tasks total)" % (cat, 0, ntotal))
if succ:
    tot_k = sum(a[1] for a in agg.values()); tot_n = sum(a[2] for a in agg.values())
    print("%-30s %6d %7.1f%%  (%d/%d)" % ("ALL evaluated", sum(a[0] for a in agg.values()),
                                          100.0 * tot_k / tot_n if tot_n else 0, tot_k, tot_n))
json.dump(rows, open("libero90_categories.json", "w"), indent=2)
