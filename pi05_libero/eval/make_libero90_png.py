import matplotlib.pyplot as plt

HEADER = "#2f3b52"; ALT = "#f3f5f9"
GREEN = "#cdebcd"; YELLOW = "#fdf3d6"; RED = "#fbd5d5"; REF = "#e7ecf5"

# (category, #tasks, pct_str, frac_str, pct_value_for_color)
ROWS = [
    ("New receptacle  —  known object → new target (tray, cabinet shelf)", "13", "41.5%", "27/65", 41.5),
    ("New scene / arrangement  —  same object + action, new scene/layout", "59", "34.9%", "103/295", 34.9),
    ("New action  —  “turn off the stove” (never trained; only turn-on was)", "1", "0%", "0/5", 0.0),
    ("New object  —  frying pan, white bowl, red mug, yellow book, salad dressing", "17", "0%", "0/85", 0.0),
]
TOTAL = ("Held-out total  (all of LIBERO-90)", "90", "28.9%", "130/450", None)
REFROW = ("In-distribution reference  (the 40 finetuning tasks)", "40", "~96%", "—", None)
COLS = ["Novelty vs the 40 finetuning tasks", "#tasks", "success", "succ/trials"]
WID = [8.2, 1.1, 1.2, 1.4]


def cell_color(v):
    if v is None:
        return None
    if v == 0:
        return RED
    if v >= 40:
        return GREEN
    return YELLOW


rows = [list(r[:4]) for r in ROWS] + [list(TOTAL[:4]), list(REFROW[:4])]
pcts = [r[4] for r in ROWS] + [None, None]
nlines = [1] * len(rows)
total_lines = sum(nlines) + 1.0
row_in, title_in = 0.34, 1.15
fig_w = sum(WID) * 1.45
fig_h = title_in + row_in * total_lines + 0.15
fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
fig.text(0.012, 1 - 0.30 / fig_h,
         "pi0.5 · LIBERO-90 — held-out generalization (finetuned on 40 tasks, tested on 90 unseen)",
         fontsize=13.5, fontweight="bold", va="top")
fig.text(0.012, 1 - 0.66 / fig_h,
         "Each held-out task bucketed by what's novel vs the finetune set (objects/actions read from its BDDL goal). "
         "90 tasks × 5 trials = 450 rollouts.\nOverall 28.9% vs ~96% in-distribution — and the collapse is almost "
         "entirely NEW OBJECTS (0/85). Familiar object in a new container or scene fares far better.",
         fontsize=9, color="#555", va="top")
ax = fig.add_axes([0.0, 0.0, 1.0, 1 - title_in / fig_h]); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=COLS, colWidths=[w / sum(WID) for w in WID],
               cellLoc="left", loc="upper center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
unit = 1.0 / total_lines
for (rr, cc), cell in tbl._cells.items():
    cell.set_edgecolor("#d8dde6"); cell.set_text_props(va="center")
    if cc >= 1:
        cell.set_text_props(ha="center", va="center")
    if rr == 0:
        cell.set_height(unit); cell.set_facecolor(HEADER)
        cell.set_text_props(color="white", fontweight="bold", ha="center" if cc >= 1 else "left")
    else:
        dr = rr - 1; cell.set_height(unit * nlines[dr])
        is_total = dr == len(ROWS)
        is_ref = dr == len(ROWS) + 1
        if is_total:
            cell.set_facecolor(REF); cell.get_text().set_fontweight("bold")
        elif is_ref:
            cell.set_facecolor("#eef0f3"); cell.get_text().set_color("#555")
            cell.get_text().set_fontstyle("italic")
        elif cc == 2 and cell_color(pcts[dr]):
            cell.set_facecolor(cell_color(pcts[dr]))
        elif dr % 2 == 1:
            cell.set_facecolor(ALT)
out = "/Users/sttawm/dev/interactive-pi/pi05_libero/eval/libero90_generalization.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
