"""Supplementary table data, one CSV per table, from the results in out/.

Only aggregated results are written; no literature record or literature yield is written at record or pair level.
Output: out/si/Table_S{number}_{stable_suffix}.csv, numbered by first manuscript citation.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SI = OUT / "si"
NAMES = {"Acylation": "Amide coupling", "Suzuki": "Suzuki–Miyaura coupling", "Buchwald-Hartwig": "Buchwald–Hartwig amination",
         "Allylic-substitution": "Allylic substitution", "Buchwald-Hartwig CMC": "Buchwald–Hartwig II",
         "C-O competitive reactions": "C–O coupling (Pd/Cu)", "Chan-Lam": "Chan–Lam coupling", "Esterification": "Esterification",
         "Grignard": "Grignard addition", "Heck": "Heck reaction", "Hiyama": "Hiyama coupling", "Kumada": "Kumada coupling",
         "Pd-carbonylation": "Pd-catalysed carbonylation", "Reductive-amination": "Reductive amination",
         "SNAr": "Nucleophilic aromatic substitution", "Sonogashira": "Sonogashira coupling", "Ullmann": "Ullmann coupling",
         "Suzuki Bpin": "Suzuki–Miyaura coupling with pinacol boronates"}
EXTERNAL_NAMES = {"Ahneman-BH": "Ahneman C–N coupling", "Perera-Suzuki": "Perera Suzuki–Miyaura"}
DISPLAY_ALIASES = {"Buchwald–Hartwig amination (CMC array)": NAMES["Buchwald-Hartwig CMC"],
                   "C–N/C–O competition": NAMES["C-O competitive reactions"],
                   "Suzuki Bpin": NAMES["Suzuki Bpin"]}
DISPLAY_COLUMNS = {
    "blocks": {
        "Completeness (%)": "Complete (%)",
        "Wells >=20% (%)": "Cells yielding ≥20% (%)",
        "Pairs whose best condition >=20% (%)": "Pairs with best yield ≥20% (%)",
    },
    "policies": {
        "Random-arm MAE (points)": "Random MAE (points)",
        "95% interval low": "95% interval lower",
        "95% interval high": "95% interval upper",
    },
    "random_replacement": {
        f"Excess error with {n} random of 8 (points)": f"{n} random of 8 (points)"
        for n in (0, 1, 2, 4, 8)
    },
    "decomposition": {
        "8 random (MAE)": "Eight random",
        "1 random (MAE)": "One random",
        "best of 8 (MAE)": "Best of eight",
        "best of 8 if >=20% (MAE)": "Best of eight if ≥20%",
    },
    "benchmark": {
        "Benchmark MAE (points)": "Literature-style benchmark MAE (points)",
        "Unrun-condition MAE (points)": "Literature-style unrun MAE (points)",
        "Random-arm benchmark MAE (points)": "Random benchmark MAE (points)",
        "Random-arm unrun-condition MAE (points)": "Random unrun MAE (points)",
        "Recorded benchmark observations": "Scored benchmark observations",
        "Eligible (>=100)": "Eligible (≥100)",
    },
    "validation": {
        "Excess error with 1 random of 8 (points)": "One random of eight (points)",
    },
    "literature_check": {
        "Median HTE mean (%)": "Median array mean (%)",
        "Median HTE best (%)": "Median array best (%)",
        "Pairs with literature above HTE best": "Pairs above array best",
    },
    "literature_models": {
        "Literature-model MAE (points)": "Literature model MAE",
        "Random-8 MAE (points)": "Random-8 MAE",
        "Excess error (points)": "Excess error",
        "Mean signed error (points)": "Signed error",
        "Literature benchmark MAE (points)": "Literature benchmark MAE",
        "True/benchmark": "True / benchmark",
        "Reduction with 1 random record per training pair (%)": "Reduction after random addition (%)",
        "Simulated-literature excess error (points)": "Simulated literature excess error",
    },
    "external": {
        "Random-arm MAE (points)": "Random MAE (points)",
        "95% interval low": "95% interval lower",
        "95% interval high": "95% interval upper",
        "Reduction with 1 random of 8 (%)": "Reduction with one random (%)",
        "Random-replacement criterion met": "Random replacement criterion met",
    },
    "gradient_boosting": {
        "95% interval low": "95% interval lower",
        "95% interval high": "95% interval upper",
        "Reduction with 1 random of 8 (%)": "Reduction with one random (%)",
        "Unrun-condition MAE (points)": "Unrun MAE (points)",
    },
    "literature_curve": {
        "Mean overprediction (points)": "Mean signed error (points)",
        "Minimum overprediction (points)": "Minimum signed error (points)",
        "Maximum overprediction (points)": "Maximum signed error (points)",
        "Mean true-to-benchmark ratio": "Mean true / benchmark",
        "Overprediction criterion (draws/5)": "Excess + signed criterion (draws/5)",
    },
}


def pct(x):
    return (100 * x).round(1)


def dead_pairs_table():
    """Format the stored new-pair summaries without fitting or rescoring models."""
    rows = pd.read_csv(OUT / "dead_pairs_summary.csv")
    return pd.DataFrame({
        "Learner": rows.learner.map({"rf": "Random forest", "mlp": "Neural network (MLP)"}),
        "Array": rows.reaction.map(NAMES),
        "Records": rows.arm.map({"literature": "Literature-style", "random": "Random", "seven_plus_one": "7 + 1"}),
        "AUROC (maximum)": rows.auc_max.round(3),
        "AUROC (mean)": rows.auc_mean.round(3),
        "False-go (%)": pct(rows.false_go),
    })


def numbered_filenames():
    """Table file names by stable suffix: numbered from the manuscript when it is present, otherwise as shipped."""
    if (Path(__file__).with_name("make_submission_nature.py")).exists() and (ROOT / "manuscript" / "manuscript.md").is_file():
        from make_submission_nature import supplementary_layout
        return {Path(filename).stem.split("_", 2)[2]: filename for _, filename, *_ in supplementary_layout()}
    shipped = {path.stem.split("_", 2)[2]: path.name for path in SI.glob("Table_S*.csv")}
    if not shipped:
        raise ValueError("Supplementary table numbering requires manuscript/manuscript.md with its generator, "
                         f"or shipped Table_S*.csv files in {SI}")
    return shipped


def main():
    SI.mkdir(exist_ok=True)
    table_filenames = numbered_filenames()

    def write_table(frame, suffix):
        if suffix not in table_filenames:
            raise ValueError(f"No manuscript or shipped supplementary table filename for {suffix!r}")
        if "Array" in frame:
            frame = frame.copy()
            frame["Array"] = frame.Array.replace(DISPLAY_ALIASES)
            unknown = set(frame.Array) - set(array_rank)
            if unknown:
                raise ValueError(f"Unknown array labels in {suffix}: {sorted(unknown)}")
            if "Learner" in frame:
                learner_rank = {name: i for i, name in enumerate(("Random forest", "Gradient-boosted trees", "Neural network (MLP)"))}
                if not set(frame.Learner) <= set(learner_rank):
                    raise ValueError(f"Unknown learner labels in {suffix}")
                frame = (frame.assign(_learner_rank=frame.Learner.map(learner_rank), _array_rank=frame.Array.map(array_rank))
                         .sort_values(["_learner_rank", "_array_rank"], kind="stable").drop(columns=["_learner_rank", "_array_rank"]))
            else:
                frame = frame.sort_values("Array", key=lambda values: values.map(array_rank), kind="stable")
        frame = frame.rename(columns=DISPLAY_COLUMNS.get(suffix, {}))
        frame.to_csv(SI / table_filenames[suffix], index=False, na_rep="NA")

    b = pd.read_csv(OUT / "blocks_summary.csv")
    b = b[b.screen.isin(NAMES) & (b.screen != "Suzuki Bpin")].copy()
    array_rank = {"Pooled": -1, **{label: i for i, label in enumerate(b.screen.map(NAMES))},
                  NAMES["Suzuki Bpin"]: len(b)}
    t1 = pd.DataFrame({"Array": b.screen.map(NAMES), "Substrate pairs": b.pairs, "Conditions": b.conditions,
                       "Measured cells": b.wells, "Completeness (%)": pct(b.fill),
                       "Substrate 1 molecules": b.substrate_1, "Substrate 2 molecules": b.substrate_2,
                       "Mean yield (%)": pct(b.mean_yield), "Wells >=20% (%)": pct(b.wells_ge20),
                       "Pairs whose best condition >=20% (%)": pct(b.pairs_best_ge20)})
    write_table(t1, "blocks")

    v = pd.concat([pd.read_csv(OUT / "pilot_verdict.csv"), pd.read_csv(OUT / "pilot_verdict_extension.csv")])
    t2 = pd.DataFrame({"Array": v.reaction.map(NAMES), "Policy": v.policy.map({"habit": "historical preference",
                       "first_hit": "first hit", "literature": "literature-style"}),
                       "Random-arm MAE (points)": pct(v.mae_random), "Excess error (points)": pct(v.delta),
                       "95% interval low": pct(v.lo), "95% interval high": pct(v.hi), "Direction A": v.meets_A,
                       "Direction B": v.meets_B})
    write_table(t2, "policies")

    s = pd.concat([pd.read_csv(OUT / "pilot_summary.csv"), pd.read_csv(OUT / "pilot_summary_extension.csv")])
    s = s[(s.model == "rf") & (s.policy == "literature")]
    t3 = s.pivot_table(index="reaction", columns="f", values="delta").mul(100).round(1)
    t3.columns = [f"Excess error with {int(round(8 * f))} random of 8 (points)" for f in t3.columns]
    t3.insert(0, "Array", t3.index.map(NAMES))
    write_table(t3, "random_replacement")

    d = pd.read_csv(OUT / "literature_decomposition.csv")
    d = d[d.model == "rf"].pivot(index="reaction", columns="train", values="mae").mul(100).round(1)
    d = d[["random_8", "random_1", "best_1", "literature"]]
    d.columns = ["8 random (MAE)", "1 random (MAE)", "best of 8 (MAE)", "best of 8 if >=20% (MAE)"]
    d.insert(0, "Array", d.index.map(NAMES))
    write_table(d, "decomposition")

    silent = pd.read_csv(OUT / "silent_pairs.csv")
    ts = pd.DataFrame({"Array": silent.reaction.map(NAMES), "Pair-by-seed rows": silent.pair_rows,
                       "Unrecorded rows (%)": pct(silent.share_silent),
                       "Unrecorded MAE (points)": pct(silent.mae_silent),
                       "Recorded MAE (points)": pct(silent.mae_recorded),
                       "Error from unrecorded rows (%)": pct(silent.error_share_silent),
                       "Unrecorded rows with no ≥20% condition (%)": pct(1 - silent.silent_with_workable)})
    write_table(ts, "silent_pairs")
    write_table(dead_pairs_table(), "dead_pairs")

    book_scores = pd.read_csv(OUT / "book_score.csv")
    bk = book_scores[(book_scores.model == "rf") & (book_scores.policy == "literature") &
                     (book_scores.arm == "policy")].set_index("reaction")
    random_bk = book_scores[(book_scores.model == "rf") & (book_scores.policy == "habit") &
                            (book_scores.arm == "random")].set_index("reaction")
    bk = bk.join(random_bk[["book_mae", "cf_mae"]].add_prefix("random_"), validate="one_to_one")
    t5 = pd.DataFrame({"Array": bk.index.map(NAMES), "Benchmark MAE (points)": pct(bk.book_mae),
                       "Unrun-condition MAE (points)": pct(bk.cf_mae),
                       "Random-arm benchmark MAE (points)": pct(bk.random_book_mae),
                       "Random-arm unrun-condition MAE (points)": pct(bk.random_cf_mae),
                       "Recorded benchmark observations": bk.book_wells,
                       "Eligible (>=100)": bk.book_wells >= 100})
    write_table(t5, "benchmark")

    vs = pd.read_csv(OUT / "validation_summary.csv")
    vs = vs[(vs.model == "rf") & (vs.policy == "literature")]
    t6 = vs.pivot_table(index="reaction", columns="f", values="delta").mul(100).round(1)[[0.0, 0.1]]
    t6.columns = ["Excess error (points)", "Excess error with 1 random of 8 (points)"]
    t6.insert(0, "Array", [NAMES.get(r, "Pooled") for r in t6.index])
    vg = pd.read_csv(OUT / "validation_groups.csv")
    vk = pd.read_csv(OUT / "validation_skipped.csv").assign(array=lambda x: x.group.str.split("#").str[0])
    folds = vg.groupby("array").group.count() * 20
    skips = vk[(vk.policy == "literature") & (vk.f == 0)].groupby("array").size().reindex(folds.index, fill_value=0)
    t6["Counted in array-level criteria"] = [("" if r == "ALL" else bool(skips[r] <= folds[r] / 2)) for r in t6.index]
    write_table(t6, "validation")

    if (OUT / "literature_check_pairs.csv").exists():
        lp = pd.read_csv(OUT / "literature_check_pairs.csv")
        g = lp.groupby("reaction")
        t7 = pd.DataFrame({"Matched pairs": g.size(), "Median literature value (%)": pct(g.lit_value.median()),
                           "Median HTE mean (%)": pct(g.hte_mean.median()), "Median HTE best (%)": pct(g.hte_best.median()),
                           "Pairs with literature above HTE best": g.apply(lambda x: int((x.lit_value > x.hte_best).sum()), include_groups=False)})
        t7.insert(0, "Array", [NAMES.get(r, r) for r in t7.index])
        write_table(t7, "literature_check")
    else:  # per-pair literature yields are licensed and not distributed; the shipped table is kept
        target = SI / table_filenames["literature_check"]
        if not target.exists():
            shipped = list(SI.glob("Table_S*_literature_check.csv"))
            if len(shipped) != 1:
                raise ValueError("Exactly one shipped literature-check table is required when per-pair data are absent")
            target = shipped[0]
        write_table(pd.read_csv(target), "literature_check")
        print("literature-check table kept as shipped; it needs out/literature_check_pairs.csv from 17_literature_check.py")

    mv = pd.read_csv(OUT / "literature_model_verdict.csv")
    t8 = pd.DataFrame({"Array": mv.reaction.map(NAMES), "Literature training records": mv.literature_records,
                       "Literature-model MAE (points)": pct(mv.mae_literature), "Random-8 MAE (points)": pct(mv.mae_random_8),
                       "Excess error (points)": pct(mv.delta), "Mean signed error (points)": pct(mv.signed_literature),
                       "Literature benchmark MAE (points)": pct(mv.book_mae), "True/benchmark": mv.true_over_book.round(1),
                       "Reduction with 1 random record per training pair (%)": pct(mv.reduction_1_random),
                       "Simulated-literature excess error (points)": pct(mv.delta_simulated_literature)})
    write_table(t8, "literature_models")

    lc = pd.read_csv(OUT / "literature_curve.csv")
    curve = lc.groupby(["reaction", "size"]).agg(
        signed_mean=("signed", "mean"), signed_min=("signed", "min"), signed_max=("signed", "max"),
        delta_mean=("delta", "mean"), book_mean=("book_mae", "mean"), ratio_mean=("true_over_book", "mean"),
        overprediction_draws=("B1", "sum"), benchmark_draws=("B2", "sum"), draws=("draw", "size"))
    curve_order = pd.MultiIndex.from_product(
        [["Acylation", "Suzuki", "Buchwald-Hartwig"], [100, 300, 1000, 3000]], names=["reaction", "size"])
    if (set(curve.index) != set(curve_order) or lc.duplicated(["reaction", "size", "draw"]).any()
            or not curve.draws.eq(5).all()):
        raise ValueError("Literature curve requires three arrays, four training sizes and five distinct draws per group")
    curve = curve.reindex(curve_order)
    t_curve = pd.DataFrame({
        "Array": curve.index.get_level_values("reaction").map(NAMES),
        "Training records": curve.index.get_level_values("size"),
        "Mean overprediction (points)": pct(curve.signed_mean),
        "Minimum overprediction (points)": pct(curve.signed_min),
        "Maximum overprediction (points)": pct(curve.signed_max),
        "Mean excess error (points)": pct(curve.delta_mean),
        "Mean benchmark MAE (points)": pct(curve.book_mean),
        "Mean true-to-benchmark ratio": curve.ratio_mean.round(1),
        "Overprediction criterion (draws/5)": curve.overprediction_draws.astype(str) + "/5",
        "Benchmark criterion (draws/5)": curve.benchmark_draws.astype(str) + "/5"})
    write_table(t_curve, "literature_curve")

    xv = pd.read_csv(OUT / "external_verdict.csv")
    gv = pd.read_csv(OUT / "gbm_summary.csv")
    t9a = pd.DataFrame({"Dataset": xv.dataset.map(EXTERNAL_NAMES), "Random-arm MAE (points)": pct(xv.mae_random), "Excess error (points)": pct(xv.delta),
                        "95% interval low": pct(xv.lo), "95% interval high": pct(xv.hi), "Reduction with 1 random of 8 (%)": pct(xv.cut_1_random),
                        "Excess-error criterion met": xv.E1, "Random-replacement criterion met": xv.E2})
    write_table(t9a, "external")
    gb = pd.read_csv(OUT / "gbm_book.csv").set_index("reaction")
    neural = pd.read_csv(OUT / "mlp_summary.csv")
    neural_book = pd.read_csv(OUT / "mlp_book.csv").set_index("reaction")
    learner_frames = []
    for learner, summary, benchmarks in (("Random forest", s, bk), ("Gradient-boosted trees", gv, gb),
                                         ("Neural network (MLP)", neural, neural_book)):
        g0 = summary[summary.f == 0].set_index("reaction")
        g1 = summary[summary.f == 0.1].set_index("reaction")
        if (len(g0) != 17 or not g0.index.is_unique or not g1.index.is_unique or set(g0.index) != set(g1.index)
                or set(g0.index) != set(benchmarks.index)):
            raise ValueError(f"The {learner} comparison must contain exactly the seventeen analysis arrays")
        g0 = g0.join(benchmarks[["book_mae", "cf_mae"]], validate="one_to_one")
        learner_frames.append(pd.DataFrame({
            "Learner": learner, "Array": [NAMES[r] for r in g0.index], "Excess error (points)": pct(g0.delta),
            "95% interval low": pct(g0.lo), "95% interval high": pct(g0.hi),
            "Reduction with 1 random of 8 (%)": pct(1 - g1.delta / g0.delta),
            "Benchmark MAE (points)": pct(g0.book_mae), "Unrun-condition MAE (points)": pct(g0.cf_mae)}))
    t9b = pd.concat(learner_frames, ignore_index=True)
    write_table(t9b, "gradient_boosting")
    for path in SI.glob("Table_S*.csv"):
        parts = path.stem.split("_", 2)
        if len(parts) == 3 and parts[2] in table_filenames and path.name != table_filenames[parts[2]]:
            path.unlink()
    for filename in table_filenames.values():
        f = SI / filename
        print(f.name, len(pd.read_csv(f)))


if __name__ == "__main__":
    main()
