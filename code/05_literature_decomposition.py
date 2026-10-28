"""Split the literature-style loss into fewer recorded wells and success-only recording.

Same pairs, seeds, literature runs and evaluation wells as the pilot (f = 0). Four training sets per logged half:
  random_8    eight random wells, all recorded (the pilot's comparator)
  random_1    one of those random wells per pair
  best_1      the best of the eight literature runs, recorded whatever its yield
  literature  the best of the eight runs, recorded only if it reaches 20%
Folds where the literature set records no well are left out for all four sets (same rule as the pilot).
Output: out/literature_decomposition.csv (MAE, mean signed error and top-pick hit rate on identical wells), all 17 arrays.
"""

import numpy as np
import pandas as pd

import pilot_lib as L

import importlib

_run = importlib.import_module("04_run_pilot")
REACTIONS = _run.PILOT + _run.EXTENSION
MODELS = ["additive", "rf"]


def main():
    rows = []
    for rx in REACTIONS:
        block = L.load_block(rx)
        P = len(block.pairs)
        for seed in L.SEEDS:
            perm = L.rng_for(block.name, "split", seed).permutation(P)
            halves = (perm[: P // 2], perm[P // 2:])
            for fold in (0, 1):
                hist, logged = halves[fold], halves[1 - fold]
                Y = block.Y[logged]
                hist_mean = np.nan_to_num(np.nanmean(block.Y[hist], 0), nan=0.0)
                run_p, rep_lit = L.run_policy("literature", Y, hist_mean, 0.0, L.rng_for(block.name, "literature", 0.0, seed, fold))
                if not rep_lit.any():
                    continue
                counts = run_p.sum(1)
                run_r = L.matched_random(Y, counts, L.rng_for(block.name, "matched", tuple(counts), seed, fold))
                evalm = ~np.isnan(Y) & ~run_p & ~run_r
                r1 = np.zeros_like(run_r)
                rng = L.rng_for(block.name, "random_1", seed, fold)
                for i in range(len(Y)):
                    r1[i, rng.choice(np.flatnonzero(run_r[i]))] = True
                b1 = np.zeros_like(run_p)
                for i in range(len(Y)):
                    cols = np.flatnonzero(run_p[i])
                    b1[i, cols[np.argmax(Y[i, cols])]] = True
                sets = {"random_8": run_r, "random_1": r1, "best_1": b1, "literature": rep_lit}
                truth = np.nan_to_num(Y)
                for model in MODELS:
                    for name, rep in sets.items():
                        pred = L.fit_predict(model, block, logged, rep, seed)
                        top = np.argmax(np.where(evalm, pred, -np.inf), 1)
                        has = evalm.any(1)
                        rows.append(dict(reaction=rx, seed=seed, fold=fold, model=model, train=name,
                                         abs_err=np.abs(pred - truth)[evalm].sum(), signed_err=(pred - truth)[evalm].sum(),
                                         n=int(evalm.sum()), top_hit=(Y[np.arange(len(Y)), top] >= L.HIT)[has].mean(),
                                         wells=int(rep.sum())))
        print(rx, flush=True)
    d = pd.DataFrame(rows)
    g = d.groupby(["reaction", "model", "train"])
    s = pd.DataFrame({"mae": g.abs_err.sum() / g.n.sum(), "mean_signed_error": g.signed_err.sum() / g.n.sum(),
                      "top_hit": g.top_hit.mean(), "wells_per_half": g.wells.mean()}).reset_index()
    s.to_csv(L.OUT / "literature_decomposition.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
