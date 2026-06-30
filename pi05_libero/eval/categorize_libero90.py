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

PRED_RE = re.compile(r"\(([A-Za-z]+)((?:\s+[A-Za-z0-9_]+)+)\)")


def _section(txt, header):
    i = txt.find(header)
    if i < 0:
        return ""
    # return up to the next top-level "(:" header
    j = txt.find("(:", i + len(header))
    return txt[i:j if j > 0 else len(txt)]


def parse_task(task):
    """Return (preds, moved_cats, target_cats) read from the BDDL goal, with tokens
    resolved to real object/fixture categories (instance ids and regions normalized)."""
    txt = (BDDL / task.problem_folder / task.bddl_file).read_text()

    # instance/fixture -> category, e.g. "akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl"
    inst2cat = {}
    for sec in ("(:objects", "(:fixtures"):
        for line in _section(txt, sec).splitlines():
            if " - " in line:
                names, cat = line.rsplit(" - ", 1)
                cat = cat.strip()
                for nm in names.split():
                    inst2cat[nm.strip()] = cat

    # region -> its (:target fixture)
    region2tgt = {}
    for rname, rbody in re.findall(r"\(([a-z0-9_]+)\s*\(:target\s+([a-z0-9_]+)\)", _section(txt, "(:regions")):
        region2tgt[rname] = rbody

    def resolve(tok):
        seen = set()
        while tok and tok not in seen:
            seen.add(tok)
            if tok in inst2cat:
                return inst2cat[tok]
            if tok in region2tgt:
                tok = region2tgt[tok]
                continue
            t2 = re.sub(r"_\d+$", "", tok)        # strip a trailing instance id
            if t2 != tok:
                tok = t2
                continue
            break
        return tok  # cleaned fallback (e.g. a region whose target we couldn't follow)

    preds, moved, targets = set(), set(), set()
    for name, args in PRED_RE.findall(_section(txt, "(:goal")):
        if name in ("goal", "And", "and"):
            continue
        preds.add(name)
        a = args.split()
        if a:
            moved.add(resolve(a[0]))
        if len(a) > 1:
            targets.add(resolve(a[1]))
    return preds, moved, targets


def scene_of(task):
    m = re.match(r"^([A-Z_]+SCENE\d+)", task.bddl_file)
    return m.group(1) if m else None


# --- build training inventory ---
TRAIN_PREDS, TRAIN_MOVED, TRAIN_TARGETS, TRAIN_SCENES = set(), set(), set(), set()
for s in TRAIN_SUITES:
    suite = bd[s]()
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        p, mv, tg = parse_task(t)
        TRAIN_PREDS |= p
        TRAIN_MOVED |= mv
        TRAIN_TARGETS |= tg
        sc = scene_of(t)
        if sc:
            TRAIN_SCENES.add(sc)
TRAIN_OBJS = TRAIN_MOVED | TRAIN_TARGETS

# --- categorize libero_90 ---
CATS = ["new scene / arrangement", "new object", "new receptacle",
        "new action", "new action + new object"]
suite = bd["libero_90"]()
rows = []
for tid in range(suite.n_tasks):
    t = suite.get_task(tid)
    preds, moved, targets = parse_task(t)
    new_acts = sorted(preds - TRAIN_PREDS)
    new_moved = sorted(moved - TRAIN_OBJS - {""})        # a manipulated object never trained
    new_tgt = sorted(targets - TRAIN_OBJS - {""})        # a receptacle never trained
    sc = scene_of(t)
    if new_acts and (new_moved or new_tgt):
        cat = "new action + new object"
    elif new_acts:
        cat = "new action"
    elif new_moved:
        cat = "new object"
    elif new_tgt:
        cat = "new receptacle"
    else:
        cat = "new scene / arrangement"
    rows.append({"task_id": tid, "lang": str(t.language), "scene": sc, "category": cat,
                 "new_actions": new_acts, "new_objects": new_moved, "new_receptacles": new_tgt,
                 "new_scene": sc not in TRAIN_SCENES})

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
for cat in CATS:
    agg[cat] = [0, 0, 0]   # tasks, successes, trials
for r in rows:
    mark = ""
    if r["new_actions"]:
        mark += " A:" + ",".join(r["new_actions"])
    if r["new_objects"]:
        mark += " O:" + ",".join(r["new_objects"])
    if r["new_receptacles"]:
        mark += " R:" + ",".join(r["new_receptacles"])
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
