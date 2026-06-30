import textwrap
import matplotlib.pyplot as plt

HEADER = "#2f3b52"; ALT = "#f3f5f9"; GREEN = "#dff3df"; RED = "#fde2e2"

# (canonical, simplified, canon score, simplified score)
ROWS = [
    ("pick up the alphabet soup and put it in the basket", "put the alphabet soup in the basket", "5/5", "5/5"),
    ("pick up the butter and put it in the basket", "put the butter in the basket", "5/5", "5/5"),
    ("pick up the milk and put it in the basket", "put the milk in the basket", "0/5", "0/5"),
    ("pick up the orange juice and put it in the basket", "put the orange juice in the basket", "0/5", "0/5"),
    ("pick up the tomato sauce and put it in the basket", "put the tomato sauce in the basket", "5/5", "5/5"),
]
TOTAL = ("", "TOTAL", "60%  (15/25)", "60%  (15/25)")
COLS = ["Canonical instruction", "Simplified  (\"put A in the basket\")", "Canon.", "Simpl."]
WID = [5.6, 4.2, 1.1, 1.1]
WRAP = {0: 40, 1: 30}


def wrap(s, w):
    return "\n".join(textwrap.wrap(s, w)) if s else s


def score_color(s):
    try:
        a, b = map(int, s.split("/"))
        return RED if a == 0 else (GREEN if a == b else None)
    except Exception:
        return None


rows = [[wrap(c, WRAP[i]) if i in WRAP else c for i, c in enumerate(r)] for r in ROWS]
rows.append([wrap(c, WRAP[i]) if i in WRAP else c for i, c in enumerate(TOTAL)])
nlines = [max(1, max(str(c).count("\n") + 1 for c in r)) for r in rows]
total_lines = sum(nlines) + 1.0
row_in, title_in = 0.265, 0.95
fig_w = sum(WID) * 1.5
fig_h = title_in + row_in * total_lines + 0.15
fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
fig.text(0.012, 1 - 0.30 / fig_h, "pi0.5 · LIBERO-90 LIVING_ROOM_SCENE2 — simplified phrasing",
         fontsize=14, fontweight="bold", va="top")
fig.text(0.012, 1 - 0.62 / fig_h,
         "\"pick up the A and put it in the basket\"  ->  \"put the A in the basket\". "
         "Same scene & init states, 5 tasks x 5 trials, paired.   Canonical 60%  =  Simplified 60%  (identical per task). "
         "milk & orange juice fail at 0/5 for BOTH wordings -> in-distribution task failures, not language. "
         "On the 3 solvable tasks: 15/15 both ways.",
         fontsize=9, color="#555", va="top")
ax = fig.add_axes([0.0, 0.0, 1.0, 1 - title_in / fig_h]); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=COLS, colWidths=[w / sum(WID) for w in WID],
               cellLoc="left", loc="upper center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
unit = 1.0 / total_lines
for (rr, cc), cell in tbl._cells.items():
    cell.set_edgecolor("#d8dde6"); cell.set_text_props(va="center")
    if cc >= 2:
        cell.set_text_props(ha="center", va="center")
    if rr == 0:
        cell.set_height(unit); cell.set_facecolor(HEADER)
        cell.set_text_props(color="white", fontweight="bold", ha="center" if cc >= 2 else "left")
    else:
        dr = rr - 1; cell.set_height(unit * nlines[dr])
        is_total = dr == len(rows) - 1
        if is_total:
            cell.set_facecolor("#e7ecf5"); cell.get_text().set_fontweight("bold")
        elif cc == 3 and score_color(ROWS[dr][3]):
            cell.set_facecolor(score_color(ROWS[dr][3]))
        elif cc == 2 and score_color(ROWS[dr][2]):
            cell.set_facecolor(score_color(ROWS[dr][2]))
        elif dr % 2 == 1:
            cell.set_facecolor(ALT)
out = "/Users/sttawm/dev/interactive-pi/pi05_libero/eval/livingroom2_simplify.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
