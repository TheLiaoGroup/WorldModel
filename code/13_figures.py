"""Figures 2-5 at print size (Arial 9/8 pt), from the tables in out/.

Figure 2: breadth and decomposition. Figure 3: disappearance of whole substrate pairs.
Figure 4: evaluation blind spots. Figure 5: same-budget remedy and validation.
Extended Data 2 preserves literature-yield and reagent-effect comparisons.
Colours: dark blue = literature-style (success-only) records; grey = random records or measured background; teal = the
random well; orange = overestimate. Arrays are listed in the same order in every figure: by the literature-style excess
error, largest first.
Output: figures/Figure2-5 .pdf and .png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import figure_style as S

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "out", ROOT / "figures"
BLUE, GREY, LIGHT, TEAL, ORANGE = "#1f3b73", "#8c8c8c", "#c8c8c8", "#1a9e8f", "#e07b24"
PALEBLUE = "#9fb3d6"
LIGHTORANGE, NEUTRAL, NEARWHITE = "#e9a367", "#f1f2f3", "#f9f9f9"
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
    fig = plt.figure(figsize=(7.0, 4.9))
    box = inches(fig)
    # a: predicted vs measured, amide coupling; both models as binned medians with interquartile bands
    ax = fig.add_axes(box(0.55, 2.55, 1.8, 1.8), label="panel")
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
    S.panel_title(fig, ax, "a", "Untried yields are\nsystematically overpredicted", 0, 0)
    # b: every array, with its generic reaction
    ax = fig.add_axes(box(4.25, 0.65, 1.35, 3.7), label="panel")
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
    ax.text(100 * ext.lo.min(), len(order) - 0.05, "External arrays", fontsize=S.SMALL, color=GREY, va="center")
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="Excess error (points)", xlim=(0, 65))
    S.panel_title(fig, ax, "b", "Excess error is 22–58 points\nin every analysis array", 0, 1)
    # c: which well, not how many; the three steps annotated with their ranges across arrays
    ax = fig.add_axes(box(0.55, 0.65, 2.35, 1.25), label="panel")
    cols = ["random_8", "random_1", "best_1", "literature"]
    for r in order:
        ax.plot(range(4), d.loc[r, cols].to_numpy(), color=LIGHT, lw=0.6, zorder=1)
    ax.plot(range(4), d[cols].median().to_numpy(), color="black", lw=1.8, marker="o", ms=3, zorder=3)
    steps = [(0.5, d.random_1 - d.random_8, GREY, "fewer"), (1.5, d.best_1 - d.random_1, LIGHTORANGE, "best of 8"),
             (2.5, d.literature - d.best_1, ORANGE, "≥20% only")]
    for xm, g, colour, what in steps:
        ax.text(xm, 72, f"+{g.min():.1f}–{g.max():.1f}", fontsize=S.SMALL, ha="center", color=colour)
        ax.text(xm, 65, what, fontsize=S.SMALL, ha="center", color=colour)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["8 random", "1 random", "1 best", "1 best ≥20%"])
    ax.set(ylabel="Error, untried (points)", xlim=(-0.25, 3.25), ylim=(0, 78))
    S.panel_title(fig, ax, "c", "Selection costs more than record count", 1, 0)
    save(fig, "Figure2")


# ------------------------------------------------------------------ Figure 3: why standard checks miss it
def benchmark_panel(fig, rect):
    bk = pd.read_csv(OUT / "book_score.csv")
    bk = bk[bk.model == "rf"]
    lit = bk[(bk.policy == "literature") & (bk.arm == "policy") & (bk.book_wells >= 100)].set_index("reaction")
    rnd = bk[(bk.policy == "habit") & (bk.arm == "random")].set_index("reaction").loc[lit.index]
    ratios = lit.cf_mae / lit.book_mae
    ratio = ratios.min()
    xmax, ymax = 40, 65
    ax = fig.add_axes(rect, label="panel")
    ax.plot([0, xmax - 1], [0, xmax - 1], color="black", lw=0.5, ls="--")
    ax.plot([0, 52 / ratio], [0, 52], color=GREY, lw=0.5, ls=":")
    ax.text(52 / ratio + 1.0, 55, f"{ratio:.1f}×", fontsize=S.SMALL, color=GREY, va="center")
    ax.plot(100 * rnd.book_mae, 100 * rnd.cf_mae, "o", ms=3.4, color=GREY)
    ax.plot(100 * lit.book_mae, 100 * lit.cf_mae, "o", ms=3.4, color=BLUE)
    real = pd.read_csv(OUT / "literature_model_verdict.csv").set_index("reaction")
    ax.plot(100 * real.book_mae, 100 * real.mae_literature, "*", ms=7.5, color=BLUE, mec="white", mew=0.5, zorder=5)
    circle_label = lit.loc[lit.cf_mae.idxmax()]
    ax.annotate(f"{len(lit)} arrays\nliterature-style", xy=(100 * circle_label.book_mae, 100 * circle_label.cf_mae),
                xytext=(6, 0), textcoords="offset points", fontsize=S.SMALL, color=BLUE, va="center")
    star_label = real.loc[real.book_mae.idxmin()]
    ax.annotate("Reaxys-trained\nsubstrate-only", xy=(100 * star_label.book_mae, 100 * star_label.mae_literature),
                xytext=(-6, 0), textcoords="offset points", fontsize=S.SMALL, color=BLUE, ha="right", va="center",
                clip_on=False)
    ax.text(20, 6, "8 random wells", color=GREY, fontsize=S.SMALL)
    ax.set(xlim=(0, xmax), ylim=(0, ymax), xlabel="Benchmark error (points)", ylabel="Error on untried conditions (points)")
    ax.set_aspect("equal")
    S.panel_title(fig, ax, "a", f"Benchmarks miss a\n{ratio:.1f}–{ratios.max():.1f}-fold error", 0, 0)

def training_curve(fig, box):
    """Figure 3b: one small panel per reaction class; literature models on 100-3,000 records, five draws each."""
    c = pd.read_csv(OUT / "literature_curve.csv")
    sizes = sorted(c["size"].unique())
    order = ["Acylation", "Suzuki", "Buchwald-Hartwig"]
    assert set(c.reaction) == set(order) and sizes[0] == 100
    axes = []
    for k, rx in enumerate(order):
        ax = fig.add_axes(box(4.4, 4.12 + (2 - k) * 0.55, 1.65, 0.48), label="panel" if k == 0 else "sub")
        g = c[c.reaction == rx].groupby("size")
        for col, filled in (("mae_literature", True), ("book_mae", False)):
            m, lo, hi = (100 * g[col].mean(), 100 * g[col].min(), 100 * g[col].max())
            ax.vlines(sizes, lo, hi, color=BLUE, lw=0.6, alpha=0.6)
            ax.plot(sizes, m, "-" if filled else "--", color=BLUE, lw=0.9, marker="*", ms=5.5,
                    mfc=BLUE if filled else "white", mec=BLUE, mew=0.6)
        ax.text(sizes[-1] * 1.3, 100 * g["mae_literature"].mean().iloc[-1], "untried", fontsize=S.SMALL, color=BLUE, va="center")
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
    S.panel_title(fig, axes[0], "b", "Thirtyfold more records,\nsame error", 0, 1)


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
    ax.text(-0.5, -6, f"{n} pairs; recipes not matched\nMedians: literature reports {100 * lv.median_lit_value:.0f}%"
                      f"\nmeasured pair means {100 * lv.median_hte_mean:.0f}%",
            fontsize=S.SMALL, color="#595959", va="top", clip_on=False, linespacing=1.05)
    ax.set(xlim=(-0.5, 0.5), ylim=(0, 100), ylabel="Substrate pairs (%)")
    ax.set_xticks([])
    S.panel_title(ax.figure, ax, "a", f"{100 * lv.share_lit_above_hte_best:.0f}% of pairs exceed\nthe measured best", 0, 0)


def silent_pairs(fig, rect):
    """Figure 2d: one amide-coupling split; each row is a logged pair, its eight attempted yields (best first) and its best
    yield anywhere in the library. Rows without an attempted yield of 20% leave no literature-style record."""
    ex = pd.read_csv(OUT / "silent_pairs_example.csv")
    tried = [f"tried_{k}" for k in range(1, 9)]
    ex = ex.sort_values(["recorded", "tried_1"], ascending=[False, False]).reset_index(drop=True)
    n_rec = int(ex.recorded.sum())
    silent = ex[~ex.recorded]
    grid = np.column_stack([ex[tried].to_numpy(), np.full(len(ex), np.nan), ex.best_library.to_numpy()]) * 100
    cmap = mcolors.ListedColormap(np.vstack([plt.cm.Greys(np.linspace(0.0, 0.08, 20)), plt.cm.Blues(np.linspace(0.35, 1.0, 80))]))
    cmap.set_bad("white")
    ax = fig.add_axes(rect, label="panel")
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
    for left, width in ((-0.5, 8), (8.5, 1)):
        ax.add_patch(Rectangle((left, -0.5), width, len(ex), fill=False, edgecolor=LIGHT, lw=0.5, clip_on=False))
    ax.axhline(n_rec - 0.5, color=ORANGE, lw=1.0)
    ax.set_xticks([3.5, 9])
    n_cond = int((pd.read_csv(OUT / "conditions.csv").reaction == "Acylation").sum())
    ax.set_xticklabels(["8 attempted,\nbest first", f"best measured\n({n_cond}-condition library)"])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel(f"{len(ex)} amide-coupling pairs", fontsize=S.SMALL)
    x = 10.0
    ax.text(x, n_rec / 2, f"{n_rec} pairs recorded\nattempted best ≥20%", fontsize=S.SMALL, va="center", color=BLUE,
            clip_on=False, linespacing=1.05)
    fail = (silent.best_library < 0.20).mean()
    ax.text(x, n_rec + len(silent) / 2, f"{len(silent)} pairs ({100 * len(silent) / len(ex):.0f}%) absent;\n"
            f"{100 * fail:.0f}% stay below 20%\nover measured conditions", fontsize=S.SMALL,
            va="center", color=ORANGE, clip_on=False, linespacing=1.05)
    ax.legend(handles=[Patch(facecolor=NEARWHITE, edgecolor=LIGHT, lw=0.5, label="<20%"),
                       Patch(facecolor=BLUE, edgecolor=BLUE, lw=0.5, label="≥20%")],
              loc="lower left", bbox_to_anchor=(1.04, -0.045), ncol=2, frameon=False,
              fontsize=S.SMALL, handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    S.panel_title(fig, ax, "a", "The 20% threshold silences whole pairs", 0, 0)


# ------------------------------------------------------------------ Figure 3: the mechanism
def figure3(order):
    fig = plt.figure(figsize=(7.0, 6.5))
    box = inches(fig)
    silent_pairs(fig, box(0.45, 3.85, 1.45, 2.3))
    d = pd.read_csv(OUT / "silent_pairs.csv").set_index("reaction").loc[order]
    ax = fig.add_axes(box(5.0, 3.85, 1.8, 2.3), label="panel")
    for k, r in enumerate(order):
        ax.plot([100*d.loc[r,"share_silent"],100*d.loc[r,"error_share_silent"]],[k,k],color=LIGHT,lw=0.65)
        ax.plot(100*d.loc[r,"share_silent"],k,"o",color=ORANGE,ms=3.7)
        ax.plot(100*d.loc[r,"error_share_silent"],k,"o",color="black",ms=3.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES[r] for r in order])
    ax.set(ylim=(len(order)-0.4,-1.2),xlim=(0,100),xticks=[0,50,100],xlabel="Share (%)")
    ax.tick_params(axis="y",length=0)
    first = d.loc[order[0]]
    ax.annotate("Pairs absent", xy=(100 * first.share_silent, 0), xytext=(-5, 0),
                textcoords="offset points", fontsize=S.SMALL, color=ORANGE, ha="right", va="center")
    ax.annotate("Share of total error", xy=(100 * first.error_share_silent, 0), xytext=(0, 7),
                textcoords="offset points", fontsize=S.SMALL, color="black", ha="right", va="bottom")
    S.panel_title(fig,ax,"b","Absent pairs carry 28–91% of the error",0,1)
    # The new-pair decision is scored only on pairs absent from model training.
    # Reuse the order above; the neural learner remains in the SI table.
    new = pd.read_csv(OUT / "dead_pairs_summary.csv")
    arms = [("literature", BLUE, "literature-style"), ("random", GREY, "8 random wells"),
            ("seven_plus_one", TEAL, "7 + 1")]
    assert len(new) == 102 and not new.duplicated(["reaction", "learner", "arm"]).any()
    assert set(new.reaction) == set(order) and set(new.learner) == {"rf", "mlp"}
    assert set(new.arm) == {arm for arm, _, _ in arms}
    assert new[["auc_max", "auc_mean", "false_go"]].notna().all().all()
    assert new[["auc_max", "auc_mean", "false_go"]].ge(0).all().all()
    assert new[["auc_max", "auc_mean", "false_go"]].le(1).all().all()
    rf = new[new.learner == "rf"].set_index(["reaction", "arm"])
    ax = fig.add_axes(box(2.10, 0.80, 4.70, 2.35), label="panel")
    ax.vlines(0.5, -0.05, len(order) - 0.4, color=LIGHT, lw=0.7, ls="--", zorder=0)
    for arm, colour, label in arms:
        g = rf.xs(arm, level="arm").loc[order]
        dy = {"literature": -0.17, "random": 0, "seven_plus_one": 0.17}[arm]
        ax.plot(g.auc_max, np.arange(len(order)) + dy, "o", ms=3.0, color=colour,
                label=f"{label} (median {g.auc_max.median():.2f})")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES[r] for r in order])
    ax.tick_params(axis="y", length=0)
    ax.set(ylim=(len(order) - 0.4, -2.5), xlim=(0.40, 1.0), xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
           xlabel="AUROC for new substrate pairs (higher is better)")
    ax.text(0.5, -0.05, "chance", color=GREY, fontsize=S.SMALL, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
    ax.legend(loc="upper left", ncol=3, frameon=False, borderaxespad=0.15,
              handletextpad=0.25, handlelength=0.7, columnspacing=1.0, fontsize=S.SMALL)
    S.panel_title(fig, ax, "c", "New pairs without a workable condition look workable", 1, 0)
    false_go = [100 * rf.xs(arm, level="arm").false_go.median() for arm, _, _ in arms]
    fig.text(0.30 / 7, 0.22 / 6.5,
             "Median false-go rate: " + " / ".join(f"{value:.0f}%" for value in false_go)
             + " (literature-style / random / 7 + 1)", fontsize=S.SMALL, color=GREY)
    fig.text(0.30 / 7, 0.065 / 6.5, "Unworkable: no measured condition reaches 20% yield.",
             fontsize=S.SMALL, color=GREY)
    save(fig,"Figure3")


# ------------------------------------------------------------------ Figure 4: evaluation blind spots
def top_pick_panel(fig,rect,order,floor):
    d=pd.read_csv(OUT/"literature_decomposition.csv")
    top=d[d.model=="rf"].pivot(index="reaction",columns="train",values="top_hit")*100
    ax=fig.add_axes(rect,label="panel")
    vs1,vs8=top.literature-top.random_1,top.literature-top.random_8
    gap=vs1.median()
    ax.vlines(0,-0.5,len(order)-0.5,color=LIGHT,lw=0.6,zorder=0)
    ax.vlines(gap,-0.5,len(order)-0.5,color=BLUE,lw=0.6,ls="--",zorder=0)
    for k,r in enumerate(order):
        ax.plot(vs8[r],k,"o",ms=3.4,mfc="white",mec=GREY,mew=0.8)
        ax.plot(vs1[r],k,"o",ms=3.4,color=BLUE)
    rows_axis(ax,order,floor)
    lo=5*np.floor(min(vs1.min(),vs8.min())/5)-1
    ax.set(xlabel="Top-pick difference (points)",xlim=(lo,10),xticks=[-20,-10,0,10])
    ax.set_ylim(len(order) - 0.5, -2.5)
    ax.text(9.5, -1.8, "vs 1 random record", fontsize=S.SMALL, color=BLUE, ha="right", va="center")
    ax.text(9.5, -0.85, "vs 8 random wells", fontsize=S.SMALL, color=GREY, ha="right", va="center")
    S.panel_title(fig,ax,"c",f"Top-pick success:\nmedian gap {gap:.1f} points",1,1)


def figure4(order,floor):
    fig=plt.figure(figsize=(7.0,6.15))
    box=inches(fig)
    benchmark_panel(fig,box(0.6,0.8,4.9 * 40 / 65,4.9))
    training_curve(fig,box)
    top_pick_panel(fig,box(5.45,0.65,1.4,2.6),order,floor)
    save(fig,"Figure4")


def extended_data2():
    fig=plt.figure(figsize=(7.0,3.0))
    box=inches(fig)
    literature_yields(fig.add_axes(box(0.55,0.85,0.24,1.6),label="panel"))
    gp=pd.read_csv(OUT/"chemistry_gap.csv").set_index(["reaction","series"])
    ax=fig.add_axes(box(4.25,0.85,2.55,1.6),label="panel")
    series=(("measured","black","measured",-0.22),("pred_literature",BLUE,"literature-style",0),
            ("pred_random_8",GREY,"8 random wells",0.22))
    for k,r in enumerate(PILOT):
        for col,colour,_,dy in series:
            row=gp.loc[(r,col)]
            ax.plot([row.lo,row.hi],[k+dy,k+dy],color=colour,lw=0.9)
            ax.plot(row.gap,k+dy,"o",ms=3.6,color=colour)
    ax.axvline(0,color=LIGHT,lw=0.6,zorder=0)
    ax.set_yticks(range(3))
    ax.set_yticklabels([NAMES[r] for r in PILOT])
    ax.set(ylim=(2.6,-0.6),xlabel="Shared minus different\nfirst-reagent (points)",xlim=(-4,8),xticks=[-4,0,4,8])
    ax.tick_params(axis="y",length=0)
    for col,colour,lab,dy in series:
        if col == "measured":
            ax.text(gp.loc[(PILOT[0],col)].gap,-0.53,lab,fontsize=S.SMALL,color=colour,ha="center",va="bottom")
        else:
            ax.text(gp.loc[(PILOT[0],col)].hi+0.35,dy,lab,fontsize=S.SMALL,color=colour,va="center")
    S.panel_title(fig,ax,"b","Predictions miss reagent effects",0,1)
    save(fig,"ExtendedData2")


# ------------------------------------------------------------------ Figure 5: the remedy and its validation
def validation_counted():
    vg = pd.read_csv(OUT / "validation_groups.csv")
    vk = pd.read_csv(OUT / "validation_skipped.csv").assign(array=lambda x: x.group.str.split("#").str[0])
    folds = vg.groupby("array").group.count() * 20
    skips = vk[(vk.policy == "literature") & (vk.f == 0)].groupby("array").size().reindex(folds.index, fill_value=0)
    return set(folds.index[skips <= folds / 2])


def figure5(s, order):
    fig = plt.figure(figsize=(7.0, 2.6))
    box = inches(fig)
    # a: schematic condition yields; every marker is an attempt, filled markers are recorded outcomes.
    ax = fig.add_axes(box(0.1, 0.55, 1.75, 1.75), label="panel")
    x = np.arange(8)
    schematic_yields = np.array([0.86, 0.71, 0.54, 0.36, 0.22, 0.15, 0.08, 0.04])
    for baseline in (0.66, 0.17):
        edges = [BLUE] * 8 if baseline == 0.66 else [BLUE] * 7 + [TEAL]
        ax.bar(x, 0.25 * schematic_yields, bottom=baseline, width=0.64, color=NEUTRAL, edgecolor=edges, lw=0.5)
        ax.hlines(baseline, -0.5, 7.5, color=BLUE, lw=0.5)
    for k, y in enumerate(schematic_yields):
        ax.plot(k, 0.66 + 0.25 * y + 0.022, "o", ms=3.6, mec=BLUE, mew=0.8,
                mfc=BLUE if k == 0 else "white")
        colour = TEAL if k == 7 else BLUE
        ax.plot(k, 0.17 + 0.25 * y + 0.022, "o", ms=3.6, mec=colour, mew=0.8,
                mfc=colour if k in (0, 7) else "white")
    ax.text(-0.5, 0.99, "8 by history", fontsize=S.SMALL, color=BLUE, va="top")
    ax.text(-0.5, 0.62, "Best of 8, if ≥20%", fontsize=S.SMALL, color=BLUE, va="top")
    ax.text(-0.5, 0.50, "7 by history", fontsize=S.SMALL, color=BLUE, va="top")
    ax.text(3.55, 0.50, "+ 1 random", fontsize=S.SMALL, color=TEAL, va="top")
    ax.text(-0.5, 0.13, "Best of 7, if ≥20%", fontsize=S.SMALL, color=BLUE, va="top")
    ax.text(-0.5, 0.045, "+ 1 random, always kept", fontsize=S.SMALL, color=TEAL, va="top")
    handles = [Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec="black", mew=0.8, label="attempts"),
               Line2D([], [], ls="", marker="o", ms=3.6, color="black", label="recorded")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.03, -0.055), ncol=2, frameon=False,
              handletextpad=0.2, columnspacing=0.7, borderaxespad=0, fontsize=S.SMALL)
    ax.text(-0.5, -0.18, "Schematic condition yields", fontsize=S.SMALL, color=GREY, va="top")
    ax.set(xlim=(-0.7, 7.7), ylim=(0, 1.02))
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
    figure3(order)
    figure4(order, floor)
    figure5(s, order)
    if (OUT / "literature_check_pairs.csv").exists():
        extended_data2()
        print("wrote Figures 2–5 and Extended Data 2")
    else:  # per-pair literature yields are licensed and not distributed; see README
        print("wrote Figures 2–5; Extended Data 2 requires licensed per-pair literature yields")


if __name__ == "__main__":
    main()
