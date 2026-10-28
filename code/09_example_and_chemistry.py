"""Example substrate pair and chemistry readout, by rules fixed before the outcomes were read.

Example pair: acylation, seed 0, random forest, f = 0. Per pair, the literature model's MAE on its unrun conditions in the
fold where the pair is logged; median over all pairs; among pairs where the literature policy recorded a well, the pair
closest to the median (ties: first pair string). Its full condition row is written with the measured yields and the
predictions of three models (literature, literature with one random well, eight random wells).

Chemistry readout: three pilot arrays, seeds 0-9, f = 0, random forest. For each pair with a recorded literature well, its
unrun conditions are split by whether their first reagent equals that of the recorded well (acylation: coupling reagent;
Suzuki, Buchwald-Hartwig: metal source or precatalyst). Per group: mean measured yield and the mean predictions of the
literature model and the eight-random-well model.

Outputs: out/example_pair.csv, out/chemistry_readout.csv (with 95% intervals), out/chemistry_rows.csv.gz.
"""

import numpy as np
import pandas as pd

import pilot_lib as L

PILOT = ["Acylation", "Suzuki", "Buchwald-Hartwig"]


def fold_setup(block, seed, fold):
    P = len(block.pairs)
    perm = L.rng_for(block.name, "split", seed).permutation(P)
    halves = (perm[: P // 2], perm[P // 2:])
    hist, logged = halves[fold], halves[1 - fold]
    hist_mean = np.nan_to_num(np.nanmean(block.Y[hist], 0), nan=0.0)
    return logged, hist_mean


def arms(block, logged, hist_mean, seed, fold):
    Y = block.Y[logged]
    run_p, rep_lit = L.run_policy("literature", Y, hist_mean, 0.0, L.rng_for(block.name, "literature", 0.0, seed, fold))
    counts = run_p.sum(1)
    run_r = L.matched_random(Y, counts, L.rng_for(block.name, "matched", tuple(counts), seed, fold))
    evalm = ~np.isnan(Y) & ~run_p & ~run_r
    return Y, run_p, rep_lit, run_r, evalm


def example():
    e = pd.read_csv(L.OUT / "pilot_errors.csv.gz")
    e = e[(e.reaction == "Acylation") & (e.policy == "literature") & (e.f == 0) & (e.model == "rf")
          & (e.arm == "policy") & (e.seed == 0)].copy()
    if e.empty or e.pair.duplicated().any():
        raise SystemExit("expected one row per acylation pair for seed 0")
    e["mae"] = e.abs_err_sum / e.n_eval
    med = e.mae.median()
    cand = e[e.n_train_wells > 0].assign(gap=lambda d: (d.mae - med).abs()).sort_values(["gap", "pair"])
    pick = cand.iloc[0]
    print(f"example pair: {pick.pair} (fold {pick.fold}); MAE {pick.mae:.4f}, median {med:.4f}, {len(e)} pairs")

    block = L.load_block("Acylation")
    seed, fold = 0, int(pick.fold)
    logged, hist_mean = fold_setup(block, seed, fold)
    Y, run_p, rep_lit, run_r, evalm = arms(block, logged, hist_mean, seed, fold)
    i = [block.pairs[j] for j in logged].index(pick.pair)
    run_fix, rep_fix, rnd_fix = L.run_policy("literature", Y, hist_mean, 0.10,
                                             L.rng_for(block.name, "literature", 0.10, seed, fold), return_random=True)
    pred = {k: L.fit_predict("rf", block, logged, rep, seed)[i]
            for k, rep in (("literature", rep_lit), ("literature_plus_1_random", rep_fix), ("random_8", run_r))}
    check = np.abs(pred["literature"] - Y[i])[evalm[i]].mean()
    if not np.isclose(check, pick.mae, atol=1e-9):
        raise SystemExit(f"refit does not reproduce the pilot error: {check} vs {pick.mae}")
    cond = pd.read_csv(L.OUT / "conditions.csv")
    cond = cond[cond.reaction == "Acylation"].set_index("condition_id").loc[block.conds]
    out = pd.DataFrame({
        "condition_id": block.conds, "coupling_reagent": cond.reagent1_name.values,
        "additive": cond.reagent2_name.values, "base": cond.reagent3_name.values,
        "measured": Y[i], "recorded_literature": rep_lit[i], "run_by_literature": run_p[i],
        "run_by_literature_plus_1_random": run_fix[i], "recorded_with_1_random": rep_fix[i], "random_well": rnd_fix[i],
        "run_by_random_8": run_r[i],
        # one scoring set for all three predictions: conditions run by none of the three arms
        "unrun_scored": evalm[i] & ~run_fix[i],
        **{f"pred_{k}": v for k, v in pred.items()}})
    out.insert(0, "pair", pick.pair)
    out.to_csv(L.OUT / "example_pair.csv", index=False)
    rec = out[out.recorded_literature].iloc[0]
    u = out[out.unrun_scored]
    print(f"recorded: {rec.condition_id} {rec.coupling_reagent}/{rec.additive}/{rec.base} measured {rec.measured:.3f}")
    print(f"unrun scored {len(u)}: measured mean {u.measured.mean():.3f}; literature pred {u.pred_literature.mean():.3f}; "
          f"+1 random {u.pred_literature_plus_1_random.mean():.3f}; random 8 {u.pred_random_8.mean():.3f}")


def chemistry():
    rows = []
    cond = pd.read_csv(L.OUT / "conditions.csv")
    for rx in PILOT:
        block = L.load_block(rx)
        first = cond[cond.reaction == rx].set_index("condition_id").loc[block.conds].reagent1_id.to_numpy()
        for seed in L.SEEDS:
            for fold in (0, 1):
                logged, hist_mean = fold_setup(block, seed, fold)
                Y, run_p, rep_lit, run_r, evalm = arms(block, logged, hist_mean, seed, fold)
                p_lit = L.fit_predict("rf", block, logged, rep_lit, seed)
                p_rnd = L.fit_predict("rf", block, logged, run_r, seed)
                for i in np.flatnonzero(rep_lit.any(1)):
                    c_star = np.flatnonzero(rep_lit[i])[0]
                    same = first == first[c_star]
                    for group, mask in (("same_first_reagent", same), ("different_first_reagent", ~same)):
                        m = evalm[i] & mask
                        if m.any():
                            rows.append(dict(reaction=rx, seed=seed, fold=fold, group=group, pair=block.pairs[logged[i]], n=int(m.sum()),
                                             measured=Y[i, m].sum(), pred_literature=p_lit[i, m].sum(),
                                             pred_random_8=p_rnd[i, m].sum()))
        print(rx, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(L.OUT / "chemistry_rows.csv.gz", index=False)
    g = d.groupby(["reaction", "group"])
    s = pd.DataFrame({"wells": g.n.sum(), "pair_folds": g.n.size(),
                      **{c: g[c].sum() / g.n.sum() for c in ("measured", "pred_literature", "pred_random_8")}}).reset_index()
    # 95% intervals: resample seeds, then whole substrate pairs within them (same scheme as the main scores)
    for k, row in s.iterrows():
        sub = d[(d.reaction == row.reaction) & (d.group == row.group)]
        cube = sub.groupby(["pair", "seed"])[["n", "measured", "pred_literature", "pred_random_8"]].sum()
        grids = {c: cube[c].unstack("seed").fillna(0).to_numpy() for c in cube.columns}
        rng = L.rng_for("chemistry_boot", row.reaction, row.group)
        boot = {c: [] for c in ("measured", "pred_literature", "pred_random_8")}
        P, S = grids["n"].shape
        for _ in range(1000):
            ix = np.ix_(rng.integers(0, P, P), rng.integers(0, S, S))
            n = grids["n"][ix].sum()
            for c in boot:
                boot[c].append(grids[c][ix].sum() / n)
        for c, v in boot.items():
            s.loc[k, c + "_lo"], s.loc[k, c + "_hi"] = np.quantile(v, [0.025, 0.975])
    s.to_csv(L.OUT / "chemistry_readout.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    import sys
    example()
    if "--example-only" not in sys.argv:
        chemistry()
