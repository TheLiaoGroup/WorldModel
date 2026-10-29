"""Neural-model check (rules fixed before outcomes were read): a multilayer perceptron replaces the random forest
on the 17 analysis arrays; literature-style policy with and without one random well, and its book score.
Outputs: out/mlp_errors.csv.gz, out/mlp_summary.csv, out/mlp_book.csv, out/mlp_verdict.csv.
"""

import importlib

import numpy as np
import pandas as pd

import pilot_lib as L

RUN = importlib.import_module("04_run_pilot")
BOOK = importlib.import_module("07_book_score")
REACTIONS = RUN.PILOT + RUN.EXTENSION
REL, RATIO, CUT, MIN_BOOK = 0.10, 1.5, 0.50, 100


def main():
    frames, book_rows = [], []
    for rx in REACTIONS:
        block = L.load_block(rx)
        frames.append(L.experiment(block, ["mlp"], policies=("literature",), fractions=(0.0, 0.10)))
        P = len(block.pairs)
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
                evalm = ~np.isnan(Y) & ~run_p & ~run_r
                b_sum, b_n = BOOK.book(block, logged, rep_p, "mlp", seed)
                pred = L.fit_predict("mlp", block, logged, rep_p, seed)
                book_rows.append(dict(reaction=rx, seed=seed, fold=fold, book_sum=b_sum, book_n=b_n,
                                      cf_sum=np.abs(pred - np.nan_to_num(Y))[evalm].sum(), cf_n=int(evalm.sum())))
        print(rx, flush=True)
    df = pd.concat(frames)
    df.to_csv(L.OUT / "mlp_errors.csv.gz", index=False)
    s = L.summarise(df)
    s.to_csv(L.OUT / "mlp_summary.csv", index=False)
    bk = pd.DataFrame(book_rows)
    g = bk.groupby("reaction")
    book = pd.DataFrame({"book_mae": g.book_sum.sum() / g.book_n.sum(), "cf_mae": g.cf_sum.sum() / g.cf_n.sum(),
                         "book_wells": g.book_n.sum()}).reset_index()
    book.to_csv(L.OUT / "mlp_book.csv", index=False)
    r0 = s[s.f == 0].set_index("reaction")
    r1 = s[s.f == 0.10].set_index("reaction")
    a = (r0.lo > 0) & (r0.delta >= REL * r0.mae_random)
    cut = 1 - r1.delta / r0.delta
    ok_book = book[book.book_wells >= MIN_BOOK].set_index("reaction")
    ratio = ok_book.cf_mae / ok_book.book_mae
    need = lambda n: int(np.ceil(2 / 3 * n))
    v = dict(arrays=len(r0), meets_A=int(a.sum()), N1=int(a.sum()) >= need(len(r0)),
             book_arrays=len(ok_book), book_ratio_ge_1_5=int((ratio >= RATIO).sum()),
             N2=int((ratio >= RATIO).sum()) >= need(len(ok_book)), book_ratio_min=ratio.min(), book_ratio_max=ratio.max(),
             cut_over_half=int((cut > CUT).sum()), N3=int((cut > CUT).sum()) >= need(len(r0)),
             cut_min=cut.min(), cut_max=cut.max(), delta_min=r0.delta.min(), delta_max=r0.delta.max())
    pd.DataFrame([v]).to_csv(L.OUT / "mlp_verdict.csv", index=False)
    print(pd.Series(v).to_string())
    print(pd.DataFrame({"delta": r0.delta, "lo": r0.lo, "A": a, "cut": cut, "book_ratio": ratio}).round(3).to_string())


if __name__ == "__main__":
    main()
