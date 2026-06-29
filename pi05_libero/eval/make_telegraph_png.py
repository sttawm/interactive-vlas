import textwrap
import matplotlib.pyplot as plt

HEADER = "#2f3b52"; ALT = "#f3f5f9"
GREEN = "#dff3df"; YELLOW = "#fdf3d6"; RED = "#fde2e2"

# (canonical, telegraphic, canon score, telegraphic score)
ROWS = [
    ("open the middle drawer of the cabinet", "middle drawer", "5/5", "5/5"),
    ("put the bowl on the stove", "bowl stove", "5/5", "4/5"),
    ("put the wine bottle on top of the cabinet", "wine bottle cabinet", "5/5", "4/5"),
    ("open the top drawer and put the bowl inside", "top drawer bowl", "4/5", "3/5"),
    ("put the bowl on top of the cabinet", "bowl cabinet", "5/5", "4/5"),
    ("push the plate to the front of the stove", "plate stove", "5/5", "0/5"),
    ("put the cream cheese in the bowl", "cream cheese bowl", "5/5", "3/5"),
    ("turn on the stove", "stove", "5/5", "4/5"),
    ("put the bowl on the plate", "bowl plate", "4/5", "4/5"),
    ("put the wine bottle on the rack", "wine bottle rack", "5/5", "4/5"),
]
TOTAL = ("", "TOTAL", "96%  (48/50)", "70%  (35/50)")
COLS = ["Canonical instruction", "Telegraphic (nouns only)", "Canon.", "Telegr."]
WID = [5.6, 4.0, 1.1, 1.1]
WRAP = {0: 38, 1: 26}


def wrap(s, w):
    return "\n".join(textwrap.wrap(s, w)) if s else s


def score_color(s):
    try:
        a, b = map(int, s.split("/"))
        return RED if a == 0 else (GREEN if a == b else YELLOW)
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
fig.text(0.012, 1 - 0.30 / fig_h, "pi0.5 · LIBERO-Goal — how much language is needed?",
         fontsize=14, fontweight="bold", va="top")
fig.text(0.012, 1 - 0.62 / fig_h,
         "Each instruction stripped to bare nouns (\"put the bowl on the stove\" -> \"bowl stove\"). "
         "Same scenes & init states, 10 tasks x 5 trials, paired.   "
         "Canonical 96%  ->  Telegraphic 70%.   Nouns carry most of the signal, but task 5 collapses: "
         "\"push ... to the front of\" -> \"plate stove\" is read as a placement.",
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
out = "/Users/sttawm/dev/interactive-pi/pi05_libero/eval/telegraph_goal.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
