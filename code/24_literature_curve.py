"""Training-size curve of the literature-trained models (rules fixed before running; see Methods).

Each size is a subset of the training set already drawn for 19_literature_model.py, so no further literature record
is read: per array and draw d, one fixed permutation of that set, and each size takes its first n records. Models,
features, book score, HTE comparator and intervals are those of 19_literature_model.py; the random-eight errors are read
from out/literature_model_errors.csv.gz so every size is compared with the same comparator on the same wells.
Outputs: out/literature_curve.csv, out/literature_curve_summary.csv.
"""

import importlib

import numpy as np
import pandas as pd

import pilot_lib as L

LM = importlib.import_module("19_literature_model")
OUT = L.OUT
SIZES, DRAWS = (100, 300, 1000, 3000), 5


def training_sets():
    """Rebuild exactly the training sets of 19_literature_model.py (same filters, same capped sample)."""
    pool = pd.read_csv(OUT / "literature_pool.csv.gz", dtype=str)
    blocks = {}
    for rx in LM.ARRAYS:
        b = L.load_block(rx)
        w = pd.read_csv(OUT / "wells.csv.gz", usecols=["reaction", "pair", "sub_1_smiles", "sub_2_smiles"])
        w = w[w.reaction == rx].drop_duplicates("pair").set_index("pair").loc[b.pairs]
        blocks[rx] = dict(block=b, s1=w.sub_1_smiles.tolist(), s2=w.sub_2_smiles.tolist())
    sets, _ = LM.literature_sets(pool, blocks)
    out = {}
    for rx, s in sets.items():
        train, _ = LM.capped_training(rx, s)
        X, keep = LM.features(train, s["c1"], s["c2"])
        out[rx] = (X, train.y.to_numpy()[keep], blocks[rx]["block"])
    return out


def score(rx, p_lit, b, r8, book_mae, n, d):
    """Excess error, signed error and benchmark ratio of one fitted literature model, against the stored random-eight errors."""
    Y = b.Y
    rows = []
    for seed in L.SEEDS:
        perm = L.rng_for(b.name, "split", seed).permutation(len(b.pairs))
        for i in perm[len(b.pairs) // 2:]:
            ok = ~np.isnan(Y[i])
            rows.append((b.pairs[i], seed, int(ok.sum()), np.abs(p_lit[i] - Y[i, ok]).sum(), (p_lit[i] - Y[i, ok]).sum()))
    lit = pd.DataFrame(rows, columns=["pair", "seed", "n", "abs_err", "signed_err"]).set_index(["pair", "seed"])
    ref = r8.set_index(["pair", "seed"])
    if not lit.index.sort_values().equals(ref.index.sort_values()) or not (lit.n == ref.n.reindex(lit.index)).all():
        raise SystemExit(f"{rx}: test wells differ from the stored comparator")
    pairs = lit.index.get_level_values(0).unique()
    seeds = sorted(lit.index.get_level_values(1).unique())
    grid = lambda frame, v: frame[v].unstack("seed").reindex(index=pairs, columns=seeds).fillna(0).to_numpy()
    N, E_lit, S_lit, E_r8 = grid(lit, "n"), grid(lit, "abs_err"), grid(lit, "signed_err"), grid(ref.reindex(lit.index), "abs_err")

    def stats(ix):
        m = N[ix].sum()
        return E_lit[ix].sum() / m - E_r8[ix].sum() / m, S_lit[ix].sum() / m, E_lit[ix].sum() / m, E_r8[ix].sum() / m

    delta, signed, mae_lit, mae_r8 = stats(np.ix_(range(len(pairs)), range(len(seeds))))
    rng = L.rng_for("litcurve_boot", rx, n, d)
    boots = np.array([stats(np.ix_(rng.integers(0, len(pairs), len(pairs)), rng.integers(0, len(seeds), len(seeds))))
                      for _ in range(LM.N_BOOT)])
    dlo, slo = np.quantile(boots[:, 0], 0.025), np.quantile(boots[:, 1], 0.025)
    return dict(reaction=rx, size=n, draw=d, delta=delta, delta_lo=dlo, signed=signed, signed_lo=slo, mae_literature=mae_lit,
                mae_random_8=mae_r8, book_mae=book_mae, true_over_book=mae_lit / book_mae,
                B1=bool(dlo > 0 and delta >= LM.REL * mae_r8 and slo > 0), B2=bool(mae_lit >= LM.RATIO * book_mae))


def main():
    sets = training_sets()
    err = pd.read_csv(OUT / "literature_model_errors.csv.gz")
    for rx, (X, y, _) in sets.items():
        print(f"{rx}: full training set {len(y)} records; sizes {[n for n in SIZES if n <= len(y)]}, {DRAWS} draws each")
    rows = []
    for rx, (X, y, b) in sets.items():
        r8 = err[(err.reaction == rx) & (err.model == "random_8")][["pair", "seed", "n", "abs_err"]]
        for d in range(DRAWS):
            order = L.rng_for("litcurve", rx, d).permutation(len(y))
            for n in SIZES:
                idx = order[:n]
                folds = L.rng_for("litcurve_book", rx, n, d).permutation(n) % 5
                book = np.empty(n)
                for k in range(5):
                    m = LM.rf(k).fit(X[idx][folds != k], y[idx][folds != k])
                    book[folds == k] = np.abs(m.predict(X[idx][folds == k]) - y[idx][folds == k])
                p = LM.rf(d).fit(X[idx], y[idx]).predict(b.pair_fp)
                rows.append(score(rx, p, b, r8, book.mean(), n, d))
            print(f"{rx}: draw {d} done", flush=True)
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "literature_curve.csv", index=False)

    full = pd.read_csv(OUT / "literature_model_verdict.csv").set_index("reaction")
    passing = {n: all(int(g.B1.mul(g.B2).sum()) >= 2 for _, g in c[(c["size"] == n)].groupby("draw")) for n in SIZES}
    passing_all3 = {n: all(bool(g.B1.mul(g.B2).all()) for _, g in c[(c["size"] == n)].groupby("draw")) for n in SIZES}
    smallest = lambda ok: next((n for i, n in enumerate(SIZES) if all(ok[m] for m in SIZES[i:])), None)
    summary = c.groupby(["reaction", "size"]).agg(signed_mean=("signed", "mean"), signed_min=("signed", "min"),
                                                  signed_max=("signed", "max"), delta_mean=("delta", "mean"),
                                                  ratio_mean=("true_over_book", "mean"), ratio_min=("true_over_book", "min"),
                                                  B1_draws=("B1", "sum"), B2_draws=("B2", "sum")).reset_index()
    summary["signed_full"] = summary.reaction.map(full.signed_literature)
    summary["ratio_full"] = summary.reaction.map(full.true_over_book)
    summary["n_star"] = smallest(passing)
    summary["n_star_all_three_arrays"] = smallest(passing_all3)
    summary.to_csv(OUT / "literature_curve_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print("n* =", smallest(passing), "; all three arrays:", smallest(passing_all3))


if __name__ == "__main__":
    main()
