"""Book score: what a benchmark built from the recorded data would report, next to the counterfactual score.

For each policy at f = 0 (and its matched random arm), the recorded wells of the logged half are split five ways at
random, as a benchmark split of a reaction dataset would be; the model is trained on four fifths and scored on the
fifth (out-of-fold MAE over all recorded wells). The counterfactual score is the pilot's main score: the model trained
on all recorded wells, scored on the wells neither the policy nor its matched random arm ran.

Usage: python 07_book_score.py [reaction ...]   (default: the three pilot blocks)
Output: out/book_score.csv (one row per reaction, policy, arm and model; existing rows of other reactions are kept).
"""

import sys

import numpy as np
import pandas as pd

import pilot_lib as L

MODELS = ["additive", "rf"]
POLICIES = ["habit", "first_hit", "literature"]


def book(block, logged, rep, model, seed):
    pi, ci = np.nonzero(rep)
    if len(pi) < 10:
        return np.nan, 0
    fold = L.rng_for("book", seed, len(pi)).permutation(len(pi)) % 5
    Y = block.Y[logged]
    err = np.empty(len(pi))
    for k in range(5):
        train = np.zeros_like(rep)
        train[pi[fold != k], ci[fold != k]] = True
        pred = L.fit_predict(model, block, logged, train, seed)
        te = fold == k
        err[te] = np.abs(pred[pi[te], ci[te]] - Y[pi[te], ci[te]])
    return err.sum(), len(err)


def main(reactions):
    rows = []
    for rx in reactions:
        block = L.load_block(rx)
        P = len(block.pairs)
        for seed in L.SEEDS:
            perm = L.rng_for(block.name, "split", seed).permutation(P)
            halves = (perm[: P // 2], perm[P // 2:])
            for fold in (0, 1):
                hist, logged = halves[fold], halves[1 - fold]
                Y = block.Y[logged]
                hist_mean = np.nan_to_num(np.nanmean(block.Y[hist], 0), nan=0.0)
                for pol in POLICIES:
                    run_p, rep_p = L.run_policy(pol, Y, hist_mean, 0.0, L.rng_for(block.name, pol, 0.0, seed, fold))
                    if not rep_p.any():
                        continue
                    counts = run_p.sum(1)
                    run_r = L.matched_random(Y, counts, L.rng_for(block.name, "matched", tuple(counts), seed, fold))
                    evalm = ~np.isnan(Y) & ~run_p & ~run_r
                    for model in MODELS:
                        for arm, rep in (("policy", rep_p), ("random", run_r)):
                            b_sum, b_n = book(block, logged, rep, model, seed)
                            pred = L.fit_predict(model, block, logged, rep, seed)
                            rows.append(dict(reaction=rx, seed=seed, fold=fold, policy=pol, arm=arm, model=model,
                                             book_sum=b_sum, book_n=b_n,
                                             cf_sum=np.abs(pred - np.nan_to_num(Y))[evalm].sum(), cf_n=int(evalm.sum())))
        print(rx, flush=True)
    d = pd.DataFrame(rows)
    g = d.groupby(["reaction", "policy", "arm", "model"])
    s = pd.DataFrame({"book_mae": g.book_sum.sum() / g.book_n.sum(), "cf_mae": g.cf_sum.sum() / g.cf_n.sum(),
                      "book_wells": g.book_n.sum()}).reset_index()
    s["inflation"] = s.cf_mae - s.book_mae
    path = L.OUT / "book_score.csv"
    if path.exists():
        old = pd.read_csv(path)
        s = pd.concat([old[~old.reaction.isin(reactions)], s])
    s.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(s[s.reaction.isin(reactions)].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["Acylation", "Suzuki", "Buchwald-Hartwig"])
