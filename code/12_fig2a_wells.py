"""Per-well predictions behind Figure 2a: acylation, seeds 0-9, f = 0, random forest.

Same splits, literature runs, matched random arms and scored wells as the pilot, so the pooled mean signed
errors equal those in out/literature_decomposition.csv. Output: out/fig2a_wells.csv.gz.
"""

import numpy as np
import pandas as pd

import pilot_lib as L


def main():
    block = L.load_block("Acylation")
    P = len(block.pairs)
    parts = []
    for seed in L.SEEDS:
        perm = L.rng_for(block.name, "split", seed).permutation(P)
        halves = (perm[: P // 2], perm[P // 2:])
        for fold in (0, 1):
            hist, logged = halves[fold], halves[1 - fold]
            Y = block.Y[logged]
            hist_mean = np.nan_to_num(np.nanmean(block.Y[hist], 0), nan=0.0)
            run_p, rep_lit = L.run_policy("literature", Y, hist_mean, 0.0, L.rng_for(block.name, "literature", 0.0, seed, fold))
            counts = run_p.sum(1)
            run_r = L.matched_random(Y, counts, L.rng_for(block.name, "matched", tuple(counts), seed, fold))
            evalm = ~np.isnan(Y) & ~run_p & ~run_r
            p_lit = L.fit_predict("rf", block, logged, rep_lit, seed)
            p_rnd = L.fit_predict("rf", block, logged, run_r, seed)
            i, c = np.nonzero(evalm)
            parts.append(pd.DataFrame({"seed": seed, "fold": fold, "measured": Y[i, c],
                                       "pred_literature": p_lit[i, c], "pred_random_8": p_rnd[i, c]}))
    d = pd.concat(parts)
    d.to_csv(L.OUT / "fig2a_wells.csv.gz", index=False)
    ref = pd.read_csv(L.OUT / "literature_decomposition.csv").set_index(["reaction", "model", "train"])
    for col, key in (("pred_literature", "literature"), ("pred_random_8", "random_8")):
        mse = (d[col] - d.measured).mean()
        assert np.isclose(mse, ref.loc[("Acylation", "rf", key), "mean_signed_error"], atol=1e-9), (col, mse)
    print(f"{len(d)} wells; mean signed error literature {(d.pred_literature - d.measured).mean():.4f}, "
          f"random 8 {(d.pred_random_8 - d.measured).mean():.4f}")


if __name__ == "__main__":
    main()
