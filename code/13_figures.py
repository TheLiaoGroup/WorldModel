"""Figures 2-5 at print size (Arial 9/8 pt), from the tables in out/.

Figure 2: size and source of the error. Figure 3: why standard checks miss it. Figure 4: two abilities, predicting a
reagent's effect and picking a workable condition. Figure 5: the same-budget remedy and its validation.
Colours: dark blue = literature-style (success-only) records; grey = random records or measured background; teal = the
random well; orange = overestimate. Arrays are listed in the same order in every figure: by the literature-style excess
error, largest first.
Output: figures/Figure2-5 .pdf and .png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import figure_style as S

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "out", ROOT / "figures"
BLUE, GREY, LIGHT, TEAL, ORANGE = "#1f3b73", "#8c8c8c", "#c8c8c8", "#1a9e8f", "#e07b24"
PALEBLUE = "#9fb3d6"
NAMES = {"Acylation": "Amide coupling", "Suzuki": "Suzuki–Miyaura", "Buchwald-Hartwig": "Buchwald–Hartwig",
         "Allylic-substitution": "Allylic substitution", "Buchwald-Hartwig CMC": "Buchwald–Hartwig II",
         "C-O competitive reactions": "C–O coupling (Pd/Cu)", "Chan-Lam": "Chan–Lam", "Esterification": "Esterification",
         "Grignard": "Grignard addition", "Heck": "Heck", "Hiyama": "Hiyama", "Kumada": "Kumada",
         "Pd-carbonylation": "Pd carbonylation", "Reductive-amination": "Reductive amination", "SNAr": r"S$_{\mathrm{N}}$Ar",
         "Sonogashira": "Sonogashira", "Ullmann": "Ullmann"}
# generic equation of each array, read from its substrates (out/wells.csv.gz)
EQUATIONS = {"Acylation": r"RCO$_2$H + amine", "Suzuki": r"ArBr + ArB(OH)$_2$", "Buchwald-Hartwig": r"ArCl + HNR$_2$",
             "Allylic-substitution": r"allyl–Br + ArNHR", "Buchwald-Hartwig CMC": r"ArBr + ArNH$_2$",
             "C-O competitive reactions": r"ArBr + ROH", "Chan-Lam": r"ArB(OH)$_2$ + ArNH$_2$",
             "Esterification": r"RCO$_2$H + R′OH", "Grignard": r"ArMgBr + C=O", "Heck": r"ArBr + CH$_2$=CHR",
             "Hiyama": r"ArX + RSi(OEt)$_3$", "Kumada": r"ArX + RMgBr", "Pd-carbonylation": r"ArX + HCO$_2$R",
             "Reductive-amination": r"RCHO + R′NH$_2$", "SNAr": r"ArF + HNR$_2$", "Sonogashira": r"ArBr + HC≡CR",
             "Ullmann": r"ArX + ArOH", "Ahneman-BH": r"ArX + ArNH$_2$", "Perera-Suzuki": r"ArX + ArB(OR)$_2$"}
ROLE = {"Acylation": "coupling reagent", "Suzuki": "Pd source", "Buchwald-Hartwig": "Pd source"}
PILOT = ["Acylation", "Suzuki", "Buchwald-Hartwig"]
FLOOR = 0.25


def load():
    v = pd.concat([pd.read_csv(OUT / "pilot_verdict.csv"), pd.read_csv(OUT / "pilot_verdict_extension.csv")])
    s = pd.concat([pd.read_csv(OUT / "pilot_summary.csv"), pd.read_csv(OUT / "pilot_summary_extension.csv")])
    lit = v[v.policy == "literature"].set_index("reaction")
    assert set(lit.index) == set(NAMES) and set(lit.primary_model) == {"rf"}
    order = lit.delta.sort_values(ascending=False).index.tolist()
    floor = pd.read_csv(OUT / "blocks_summary.csv").set_index("screen").pairs_best_ge20 < FLOOR
    return v, s[s.model == "rf"], lit, order, floor


def save(fig, name):
    S.align_panels(fig)
    for ax in fig.axes:
        if ax.get_title(loc="left") == "" and ax.get_label() == "panel":
            raise SystemExit(f"{name}: a panel lost its title")
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def inches(fig):
    w, h = fig.get_size_inches()
    return lambda x, y, dx, dy: [x / w, y / h, dx / w, dy / h]


def rows_axis(ax, order, floor):
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES[r] + ("*" if floor[r] else "") for r in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)


# ------------------------------------------------------------------ Figure 2: size and source
def figure2(lit, order, floor):
    w = pd.read_csv(OUT / "fig2a_wells.csv.gz")
    d = pd.read_csv(OUT / "literature_decomposition.csv")
    d = d[d.model == "rf"].pivot(index="reaction", columns="train", values="mae") * 100
    fig = plt.figure(figsize=(7.0, 5.2))
    box = inches(fig)
    # a: predicted vs measured, amide coupling; both models as binned medians with interquartile bands
    ax = fig.add_axes(box(0.55, 2.9, 1.8, 1.8), label="panel")
    bins = np.linspace(0, 100, 11)
    mid = (bins[1:] + bins[:-1]) / 2
    cut = pd.cut(100 * w.measured, bins, include_lowest=True)
    for col, colour in (("pred_literature", BLUE), ("pred_random_8", GREY)):
        q = (100 * w[col]).groupby(cut, observed=False).quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(mid, q[0.25], q[0.75], color=colour, alpha=0.22, lw=0)
        ax.plot(mid, q[0.5], color=colour, lw=1.4)
    ax.plot([0, 100], [0, 100], color="black", lw=0.5, ls="--")
    over_l, over_r = 100 * (w.pred_literature - w.measured).mean(), 100 * (w.pred_random_8 - w.measured).mean()
    ax.text(97, 13, f"literature-style: {over_l:+.1f}", color=BLUE, fontsize=S.SMALL, ha="right", va="bottom")
    ax.text(97, 3, f"8 random wells: {over_r:+.1f}", color=GREY, fontsize=S.SMALL, ha="right", va="bottom")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Measured yield (%)", ylabel="Predicted yield (%)")
    ax.set_aspect("equal")
    S.panel_title(fig, ax, "a", "Unrun yields are\nsystematically overpredicted", 0, 0)
    # b: every array, with its generic reaction
    ax = fig.add_axes(box(4.3, 1.15, 1.3, 3.55), label="panel")
    for k, r in enumerate(order):
        row = lit.loc[r]
        ax.plot([100 * row.lo, 100 * row.hi], [k, k], color=BLUE, lw=0.8)
        ax.plot(100 * row.delta, k, "o", ms=3.2, color=BLUE)
    ext = pd.read_csv(OUT / "external_verdict.csv")
    ext_names = {"Ahneman-BH": "Buchwald–Hartwig†", "Perera-Suzuki": "Suzuki–Miyaura†"}
    ypos, keys = list(range(len(order))), list(order)
    for k, r in enumerate(ext.itertuples()):
        y = len(order) + 0.6 + k
        ax.plot([100 * r.lo, 100 * r.hi], [y, y], color=BLUE, lw=0.8)
        ax.plot(100 * r.delta, y, "s", ms=3.0, mfc="white", mec=BLUE, mew=0.8)
        ypos.append(y)
        keys.append(r.dataset)
    for y, key in zip(ypos, keys):
        ax.text(67, y, EQUATIONS[key], fontsize=S.SMALL, color=GREY, va="center", clip_on=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels([NAMES[r] + ("*" if floor[r] else "") for r in order] + [ext_names[d] for d in ext.dataset])
    ax.set_ylim(ypos[-1] + 0.6, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="Excess error (points)", xlim=(0, 65))
    S.panel_title(fig, ax, "b", "Higher in every array tested", 0, 1)
    # c: which well, not how many; the three steps annotated with their ranges across arrays
    ax = fig.add_axes(box(0.55, 0.55, 2.35, 1.5), label="panel")
    cols = ["random_8", "random_1", "best_1", "literature"]
    for r in order:
        ax.plot(range(4), d.loc[r, cols].to_numpy(), color=LIGHT, lw=0.6, zorder=1)
    ax.plot(range(4), d[cols].median().to_numpy(), color="black", lw=1.8, marker="o", ms=3, zorder=3)
    steps = [(0.5, d.random_1 - d.random_8, GREY, "fewer"), (1.5, d.best_1 - d.random_1, "#9a6a12", "best of 8"),
             (2.5, d.literature - d.best_1, ORANGE, "≥20% only")]
    for xm, g, colour, what in steps:
        ax.text(xm, 72, f"+{g.min():.1f}–{g.max():.1f}", fontsize=S.SMALL, ha="center", color=colour)
        ax.text(xm, 65, what, fontsize=S.SMALL, ha="center", color=colour)
    sp = pd.read_csv(OUT / "silent_pairs.csv")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["8 random", "1 random", "1 best", f"1 best ≥20%\n{100 * sp.share_silent.min():.0f}–"
                        f"{100 * sp.share_silent.max():.0f}% of pairs\nleave no record"])
    ax.get_xticklabels()[3].set_color(ORANGE)
    ax.set(ylabel="Error, unrun (points)", xlim=(-0.25, 3.25), ylim=(0, 78))
    S.panel_title(fig, ax, "c", "Which well, not how many", 1, 0)
    save(fig, "Figure2")


# ------------------------------------------------------------------ Figure 3: why standard checks miss it
def figure3():
    bk = pd.read_csv(OUT / "book_score.csv")
    bk = bk[bk.model == "rf"]
    lit = bk[(bk.policy == "literature") & (bk.arm == "policy") & (bk.book_wells >= 100)].set_index("reaction")
    rnd = bk[(bk.policy == "habit") & (bk.arm == "random")].set_index("reaction").loc[lit.index]
    ratios = lit.cf_mae / lit.book_mae
    ratio = ratios.min()
    fig = plt.figure(figsize=(7.0, 3.5))
    box = inches(fig)
    side, xmax = 2.3, 40  # box sized to the equal aspect (40 x 65 points) so it is never re-centred
    ax = fig.add_axes(box(0.55, 0.55, side * xmax / 65, side), label="panel")
    ax.plot([0, xmax - 1], [0, xmax - 1], color="black", lw=0.5, ls="--")
    ax.plot([0, 52 / ratio], [0, 52], color=GREY, lw=0.5, ls=":")
    ax.text(14 / ratio - 1.5, 14, f"{ratio:.1f}×", fontsize=S.SMALL, color=GREY, ha="right", va="center")
    ax.plot(100 * rnd.book_mae, 100 * rnd.cf_mae, "o", ms=3.4, color=GREY)
    ax.plot(100 * lit.book_mae, 100 * lit.cf_mae, "o", ms=3.4, color=BLUE)
    real = pd.read_csv(OUT / "literature_model_verdict.csv").set_index("reaction")
    ax.plot(100 * real.book_mae, 100 * real.mae_literature, "*", ms=7.5, color=BLUE, mec="white", mew=0.5, zorder=5)
    ax.text(22, 62, f"{len(lit)} arrays,\nliterature-style", fontsize=S.SMALL, color=BLUE, va="top")
    ax.plot(27.3, 45.6, "*", ms=7.5, color=BLUE, mec="white", mew=0.5, clip_on=False)
    ax.text(29.2, 48, "Reaxys-trained\n(substrate-only)", fontsize=S.SMALL, color=BLUE, va="top", clip_on=False)
    ax.text(17, 6, "8 random wells", color=GREY, fontsize=S.SMALL)
    ax.set(xlim=(0, xmax), ylim=(0, 65), xlabel="Benchmark error (points)", ylabel="Error on unrun conditions (points)")
    ax.set_aspect("equal")
    S.panel_title(fig, ax, "a", f"Benchmarks miss a\n{ratio:.1f}–{ratios.max():.1f}-fold error", 0, 0)
    training_curve(fig, box)
    literature_yields(fig.add_axes(box(5.3, 0.95, 0.3, side - 0.4), label="panel"))
    save(fig, "Figure3")


def training_curve(fig, box):
    """Figure 3b: one small panel per reaction class; literature models on 100-3,000 records, five draws each."""
    c = pd.read_csv(OUT / "literature_curve.csv")
    sizes = sorted(c["size"].unique())
    order = ["Acylation", "Suzuki", "Buchwald-Hartwig"]
    assert set(c.reaction) == set(order) and sizes[0] == 100
    axes = []
    for k, rx in enumerate(order):
        ax = fig.add_axes(box(2.81, 0.55 + (2 - k) * 0.8, 1.38, 0.66), label="panel" if k == 0 else "sub")
        g = c[c.reaction == rx].groupby("size")
        for col, filled in (("mae_literature", True), ("book_mae", False)):
            m, lo, hi = (100 * g[col].mean(), 100 * g[col].min(), 100 * g[col].max())
            ax.vlines(sizes, lo, hi, color=BLUE, lw=0.6, alpha=0.6)
            ax.plot(sizes, m, "-" if filled else "--", color=BLUE, lw=0.9, marker="*", ms=5.5,
                    mfc=BLUE if filled else "white", mec=BLUE, mew=0.6)
        ax.text(sizes[-1] * 1.3, 100 * g["mae_literature"].mean().iloc[-1], "unrun", fontsize=S.SMALL, color=BLUE, va="center")
        ax.text(sizes[-1] * 1.3, 100 * g["book_mae"].mean().iloc[-1], "benchmark", fontsize=S.SMALL, color=BLUE, va="center")
        ax.text(sizes[0] * 0.85, 75, NAMES[rx], fontsize=S.SMALL, va="top")
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes] if k == 2 else [])
        ax.minorticks_off()
        ax.set(xlim=(75, 4000), ylim=(0, 76), yticks=[0, 30, 60])
        axes.append(ax)
    axes[1].set_ylabel("Error (points)")
    axes[2].set_xlabel("Substrate-only literature records per class")
    S.panel_title(fig, axes[0], "b", "Thirtyfold more records, same error", 0, 1)


def literature_yields(ax):
    """Figure 3c: where literature yields fall within the measured conditions of the same pairs (recipes not matched)."""
    lp = pd.read_csv(OUT / "literature_check_pairs.csv")
    lv = pd.read_csv(OUT / "literature_check_verdict.csv").iloc[0]
    n = len(lp)
    above = int((lp.lit_value > lp.hte_best).sum())
    within = int(((lp.lit_value <= lp.hte_best) & (lp.lit_value > lp.hte_mean)).sum())
    below = n - above - within
    bottom = 0
    for count, colour, label in ((below, LIGHT, "below\nmeasured mean"), (within, PALEBLUE, "between measured\nmean and best"),
                                 (above, BLUE, "above\nmeasured best")):
        ax.bar(0, 100 * count / n, bottom=bottom, width=0.8, color=colour)
        yt = 2 if colour == LIGHT else max(bottom + 50 * count / n, 18)
        ax.text(0.6, yt, f"{count} {label.replace(chr(10), ' ') if colour == LIGHT else label}", fontsize=S.SMALL,
                va="center", clip_on=False, color=BLUE if colour == BLUE else "black", linespacing=0.95)
        bottom += 100 * count / n
    ax.text(-0.5, -6, f"{n} pairs, recipes not matched\nmedians: literature reports {100 * lv.median_lit_value:.0f}%,"
                      f"\nmeasured array means {100 * lv.median_hte_mean:.0f}%",
            fontsize=S.SMALL, color=GREY, va="top", clip_on=False, linespacing=1.0)
    ax.set(xlim=(-0.5, 0.5), ylim=(0, 100), ylabel="Substrate pairs (%)")
    ax.set_xticks([])
    S.panel_title(ax.figure, ax, "c", f"{100 * lv.share_lit_above_hte_best:.0f}% of pairs beat\nthe measured best", 0, 2)


# ------------------------------------------------------------------ Figure 4: two abilities
def figure4(order, floor):
    gp = pd.read_csv(OUT / "chemistry_gap.csv").set_index(["reaction", "series"])
    d = pd.read_csv(OUT / "literature_decomposition.csv")
    top = d[d.model == "rf"].pivot(index="reaction", columns="train", values="top_hit") * 100
    fig = plt.figure(figsize=(7.0, 3.3))
    box = inches(fig)
    ax = fig.add_axes(box(1.55, 0.55, 1.75, 2.3), label="panel")
    series = (("measured", "black", "measured", -0.22), ("pred_literature", BLUE, "literature-style", 0.0),
              ("pred_random_8", GREY, "8 random wells", 0.22))
    for k, r in enumerate(PILOT):
        for col, colour, _, dy in series:
            row = gp.loc[(r, col)]
            ax.plot([row.lo, row.hi], [k + dy, k + dy], color=colour, lw=0.9)
            ax.plot(row.gap, k + dy, "o", ms=3.6, color=colour)
    ax.axvline(0, color=LIGHT, lw=0.6, zorder=0)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{NAMES[r]}\n({ROLE[r]})" for r in PILOT])
    ax.set_ylim(2.6, -0.6)
    ax.tick_params(axis="y", length=0)
    for col, colour, lab, dy in series:
        ax.text(gp.loc[(PILOT[0], col)].hi + 0.4, dy, lab, fontsize=S.SMALL, color=colour, va="center")
    ax.set(xlabel="Recorded reagent minus other (points)", xlim=(-4, 12))
    S.panel_title(fig, ax, "a", "Predictions miss reagent effects", 0, 0)
    ax = fig.add_axes(box(5.0, 0.55, 1.95, 2.3), label="panel")
    for k, r in enumerate(order):
        ax.plot([top.loc[r, "random_1"], top.loc[r, "literature"]], [k, k], color=LIGHT, lw=0.8)
        ax.plot(top.loc[r, "random_8"], k, "o", ms=3.2, color=GREY)
        ax.plot(top.loc[r, "random_1"], k, "o", ms=3.2, mfc="white", mec=GREY, mew=0.8)
        ax.plot(top.loc[r, "literature"], k, "o", ms=3.2, color=BLUE)
    rows_axis(ax, order, floor)
    gap = (top.literature - top.random_1).median()
    ax.set(xlabel="Top unrun pick reaching 20% (%)", xlim=(0, 75))
    handles = [Line2D([], [], ls="", marker="o", ms=3.2, color=BLUE, label="literature-style"),
               Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec=GREY, label="1 random record"),
               Line2D([], [], ls="", marker="o", ms=3.2, color=GREY, label="8 random wells")]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 1 - 12.5 / len(order)), frameon=False, fontsize=S.SMALL,
              handletextpad=0.2, borderaxespad=0.1)  # rows 11-13 are empty to the right
    S.panel_title(fig, ax, "b", f"Workable picks survive: gap {gap:.1f} points", 0, 1)
    save(fig, "Figure4")


# ------------------------------------------------------------------ Figure 5: the remedy and its validation
def validation_counted():
    vg = pd.read_csv(OUT / "validation_groups.csv")
    vk = pd.read_csv(OUT / "validation_skipped.csv").assign(array=lambda x: x.group.str.split("#").str[0])
    folds = vg.groupby("array").group.count() * 20
    skips = vk[(vk.policy == "literature") & (vk.f == 0)].groupby("array").size().reindex(folds.index, fill_value=0)
    return set(folds.index[skips <= folds / 2])


def figure5(s, order):
    fig = plt.figure(figsize=(7.0, 2.9))
    box = inches(fig)
    # a: the rule on one plate; rings = experiments run, filled = records kept
    ax = fig.add_axes(box(0.1, 0.45, 1.75, 1.75), label="panel")
    picks, best, rand = [(1, 3), (2, 7), (3, 1), (4, 9), (5, 5), (6, 2), (7, 10)], (4, 9), (2, 4)
    for row in range(8):
        for col in range(12):
            ax.add_patch(plt.Circle((col, -row), 0.36, color=LIGHT, lw=0))
    for (row, col) in picks:
        ax.add_patch(plt.Circle((col, -row), 0.42, fill=False, ec=BLUE, lw=1.0))
    ax.add_patch(plt.Circle((best[1], -best[0]), 0.36, color=BLUE, lw=0))
    ax.add_patch(plt.Circle((rand[1], -rand[0]), 0.36, color=TEAL, lw=0))
    ax.add_patch(plt.Circle((rand[1], -rand[0]), 0.62, fill=False, ec=TEAL, lw=1.0))
    ax.text(5.5, -8.4, "ring = run, coloured = recorded", fontsize=S.SMALL, ha="center", color=GREY, va="top")
    ax.text(5.5, -9.6, "7 by history: best kept if ≥20%", fontsize=S.SMALL, ha="center", color=BLUE, va="top")
    ax.text(5.5, -10.8, "1 at random: always kept", fontsize=S.SMALL, ha="center", color=TEAL, va="top")
    ax.set(xlim=(-0.8, 11.8), ylim=(-11.8, 0.8))
    ax.set_aspect("equal")
    ax.axis("off")
    S.panel_title(fig, ax, "a", "Same budget: 7 + 1", 0, 0)
    # b: excess error against the number of random wells, on a true count axis
    ax = fig.add_axes(box(2.45, 0.55, 1.95, 1.75), label="panel")
    lv = s[s.policy == "literature"].pivot(index="reaction", columns="f", values="delta") * 100
    fs, counts = [0.0, 0.10, 0.25, 0.50, 1.0], [0, 1, 2, 4, 8]
    ax.axvspan(0.7, 1.3, color=TEAL, alpha=0.08, lw=0)
    for r in order:
        ax.plot(counts, lv.loc[r, fs], color=LIGHT, lw=0.6)
    ax.plot(counts, lv[fs].median(), color=TEAL, lw=2.0, marker="o", ms=3)
    cut = 1 - lv[0.10] / lv[0.0]
    ax.annotate(f"median {lv[0.0].median():.1f} → {lv[0.10].median():.1f}\npoints of excess error", xy=(1, lv[0.10].max()),
                xytext=(2.4, 34), fontsize=S.SMALL, color=TEAL, arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.6))
    ax.set_xticks(counts)
    ax.set(xlabel="Random wells among the eight", ylabel="Excess error (points)", ylim=(-2, 60), xlim=(-0.3, 8.3))
    S.panel_title(fig, ax, "b", f"One random well removes {100 * cut.min():.0f}–{100 * cut.max():.0f}%", 0, 1)
    # c: unanalysed pairs before and after the remedy
    vsum = pd.read_csv(OUT / "validation_summary.csv")
    vsum = vsum[(vsum.model == "rf") & (vsum.policy == "literature")]
    vs = vsum.pivot(index="reaction", columns="f", values="delta") * 100
    pairs = vsum[vsum.reaction == "ALL"].set_index("f").pairs
    vv = pd.read_csv(OUT / "validation_verdict.csv").iloc[0]
    counted = validation_counted()
    rows = [r for r in order if r in vs.index and not np.isnan(vs.loc[r, 0.0])]
    ax = fig.add_axes(box(5.6, 0.55, 1.35, 1.75), label="panel")
    for k, r in enumerate(rows):
        filled = r in counted
        ax.plot([vs.loc[r, 0.10], vs.loc[r, 0.0]], [k, k], color=LIGHT, lw=0.8)
        ax.plot(vs.loc[r, 0.0], k, "o", ms=3.4, mec=BLUE, mew=0.8, mfc=BLUE if filled else "white")
        ax.plot(vs.loc[r, 0.10], k, "o", ms=3.4, mec=TEAL, mew=0.8, mfc=TEAL if filled else "white")
    k = len(rows) + 0.6
    ax.plot([100 * vv.pooled_delta * (1 - vv.pooled_cut), 100 * vv.pooled_delta], [k, k], color=LIGHT, lw=1.0)
    ax.plot(100 * vv.pooled_delta, k, "o", ms=4.2, color=BLUE)
    ax.plot(100 * vv.pooled_delta * (1 - vv.pooled_cut), k, "o", ms=4.2, color=TEAL)
    ax.set_yticks(list(range(len(rows))) + [k])
    ax.set_yticklabels([NAMES[r] for r in rows] + [f"pooled\n{int(pairs[0.0])} → {int(pairs[0.1])} pairs"])
    ax.set_ylim(k + 0.7, -1.7)
    ax.text(1, -1.1, "7 + 1 rule", fontsize=S.SMALL, color=TEAL, va="center")
    ax.text(64, -1.1, "literature-style", fontsize=S.SMALL, color=BLUE, va="center", ha="right")
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="Excess error (points)", xlim=(-2, 65))
    S.panel_title(fig, ax, "c", f"Unanalysed pairs: {100 * vv.pooled_cut:.0f}% removed", 0, 2)
    save(fig, "Figure5")


def main():
    S.configure_fonts()
    FIG.mkdir(exist_ok=True)
    v, s, lit, order, floor = load()
    figure2(lit, order, floor)
    figure4(order, floor)
    figure5(s, order)
    if (OUT / "literature_check_pairs.csv").exists():
        figure3()
        print("wrote Figures 2-5")
    else:  # per-pair literature yields are licensed and not distributed; see README
        print("wrote Figures 2, 4 and 5; Figure 3 needs out/literature_check_pairs.csv from 17_literature_check.py")


if __name__ == "__main__":
    main()
