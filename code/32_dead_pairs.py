"""Dead substrate pairs (rules frozen on 2026-10-28 before results were computed): can models trained
on success-only records rank new substrate pairs by whether any condition will reach 20%?

Within each pilot split the logged half supplies training records (literature-style, matched random, 7 + 1) and the
history half supplies test pairs whose yields never enter training targets, but guide historical condition preferences
for attempts on the logged half. Score: maximum (and mean) predicted yield over the library. Dead means that none of
the pair's measured library conditions reaches 20%, without a claim about other conditions.
Outputs: out/dead_pairs_splits.csv.gz, out/dead_pairs_summary.csv, out/dead_pairs_verdict.csv.
"""

import importlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

import pilot_lib as L

RUN = importlib.import_module("04_run_pilot")
REACTIONS = RUN.PILOT + RUN.EXTENSION
LEARNERS = ("rf", "mlp")
MIN_CLASS, MIN_SPLITS, NEED = 3, 10, 2 / 3


def predict_new(learner, block, logged, report, new, seed):
    """Fit on the recorded wells of the logged pairs; predict every condition for the new pairs."""
    pi, ci = np.nonzero(report)
    y = block.Y[logged][pi, ci]
    X = np.hstack([block.pair_fp[logged][pi], block.cond_x[ci]])
    C = block.Y.shape[1]
    Xa = np.hstack([np.repeat(block.pair_fp[new], C, 0), np.tile(block.cond_x, (len(new), 1))])
    if learner == "rf":
        rf = RandomForestRegressor(n_estimators=200, max_features=0.33, n_jobs=-1, random_state=seed).fit(X, y)
        return rf.predict(Xa).reshape(len(new), C)
    return L.mlp(X, y, Xa, seed).reshape(len(new), C)


def main():
    rows = []
    for rx in REACTIONS:
        block = L.load_block(rx)
        P = len(block.pairs)
        best = np.nanmax(np.where(np.isnan(block.Y), -1.0, block.Y), 1)
        for seed in L.SEEDS:
            perm = L.rng_for(block.name, "split", seed).permutation(P)
            halves = (perm[: P // 2], perm[P // 2:])
            for fold in (0, 1):
                hist, logged = halves[fold], halves[1 - fold]
                Y = block.Y[logged]
                hist_mean = np.nan_to_num(np.nanmean(block.Y[hist], 0), nan=0.0)
                run_p, rep_p = L.run_policy("literature", Y, hist_mean, 0.0, L.rng_for(block.name, "literature", 0.0, seed, fold))
                if not rep_p.any():
                    continue
                counts = run_p.sum(1)
                run_r = L.matched_random(Y, counts, L.rng_for(block.name, "matched", tuple(counts), seed, fold))
                _, rep_f = L.run_policy("literature", Y, hist_mean, 0.10, L.rng_for(block.name, "literature", 0.10, seed, fold))
                new = hist[best[hist] >= 0]                  # pairs with at least one measured outcome
                dead = best[new] < L.HIT
                for arm, rep in (("literature", rep_p), ("random", run_r), ("seven_plus_one", rep_f)):
                    for learner in LEARNERS:
                        pred = predict_new(learner, block, logged, rep, new, seed)
                        mx, mean = pred.max(1), pred.mean(1)
                        ok = dead.sum() >= MIN_CLASS and (~dead).sum() >= MIN_CLASS
                        rows.append(dict(reaction=rx, seed=seed, fold=fold, arm=arm, learner=learner,
                                         n_dead=int(dead.sum()), n_live=int((~dead).sum()), eligible=ok,
                                         auc_max=roc_auc_score(~dead, mx) if ok else np.nan,
                                         auc_mean=roc_auc_score(~dead, mean) if ok else np.nan,
                                         false_go=float((mx[dead] >= L.HIT).mean()) if dead.any() else np.nan))
        print(rx, flush=True)
    sp = pd.DataFrame(rows)
    sp.to_csv(L.OUT / "dead_pairs_splits.csv.gz", index=False)
    e = sp[sp.eligible]
    n_ok = e[(e.arm == "literature") & (e.learner == "rf")].groupby("reaction").size()
    keep = n_ok[n_ok >= MIN_SPLITS].index
    s = (e[e.reaction.isin(keep)].groupby(["reaction", "learner", "arm"])[["auc_max", "auc_mean", "false_go"]].mean()
         .reset_index())
    s.to_csv(L.OUT / "dead_pairs_summary.csv", index=False)
    rf = s[s.learner == "rf"].pivot(index="reaction", columns="arm")
    a_lit, a_rnd, a_7 = rf[("auc_max", "literature")], rf[("auc_max", "random")], rf[("auc_max", "seven_plus_one")]
    fg_lit, fg_rnd = rf[("false_go", "literature")], rf[("false_go", "random")]
    n = len(rf)
    gap = a_rnd - a_lit
    pos = gap > 0
    rec = ((a_7 - a_lit) >= 0.5 * gap)[pos]
    v = dict(eligible_arrays=n, D1_arrays=int((a_lit < a_rnd).sum()), D1=bool((a_lit < a_rnd).sum() >= np.ceil(NEED * n)),
             D2_arrays=int((fg_lit - fg_rnd >= 0.20).sum()), D2=bool((fg_lit - fg_rnd >= 0.20).sum() >= np.ceil(NEED * n)),
             D3_arrays=int(rec.sum()), D3_of=int(pos.sum()), D3=bool(rec.sum() >= np.ceil(NEED * pos.sum())),
             auc_lit_median=a_lit.median(), auc_random_median=a_rnd.median(), auc_7p1_median=a_7.median(),
             false_go_lit_median=fg_lit.median(), false_go_random_median=fg_rnd.median())
    pd.DataFrame([v]).to_csv(L.OUT / "dead_pairs_verdict.csv", index=False)
    print(pd.Series(v).to_string())
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
