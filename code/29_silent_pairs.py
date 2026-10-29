"""Substrate pairs that leave no record under literature-style reporting (no attempted condition reaches 20%).

For each array, over the pilot's pair-by-seed rows (random forest, no replacement): the share of pairs with no recorded
outcome, the mean absolute error on unrun conditions for those pairs and for recorded pairs, the share of the total
error they carry, and the share of them with any condition at or above 20% anywhere in the array.

Input: out/pilot_errors*.csv.gz (04_run_pilot.py), out/wells.csv.gz (01_build_blocks.py).
Example for Figure 2d: amide coupling, seed 0, fold 0 (the fold of the Figure 1 pair is not needed; the rows are the
logged pairs of that split). For each logged pair, its eight attempted yields (sorted), whether a record is kept and its
best yield over the whole condition library.

Output: out/silent_pairs.csv (shares as fractions, errors as yield fractions); out/silent_pairs_example.csv.
"""

import importlib

import numpy as np
import pandas as pd

import pilot_lib as L


def main():
    e = pd.concat([pd.read_csv(L.OUT / "pilot_errors.csv.gz"), pd.read_csv(L.OUT / "pilot_errors_extension.csv.gz")])
    e = e[(e.policy == "literature") & (e.f == 0) & (e.model == "rf") & (e.arm == "policy")].copy()
    assert e.n_train_wells.isin([0, 1]).all(), "a literature-style record holds at most one outcome per pair"
    best = pd.read_csv(L.OUT / "wells.csv.gz", usecols=["reaction", "pair", "yield_fraction"]) \
        .groupby(["reaction", "pair"]).yield_fraction.max().rename("best")
    e = e.join(best, on=["reaction", "pair"])
    assert e.best.notna().all()
    e["silent"] = e.n_train_wells == 0
    rows = []
    for rx, g in e.groupby("reaction"):
        s, k = g[g.silent], g[~g.silent]
        rows.append(dict(reaction=rx, pair_rows=len(g), share_silent=g.silent.mean(),
                         mae_silent=(s.abs_err_sum / s.n_eval).mean(), mae_recorded=(k.abs_err_sum / k.n_eval).mean(),
                         error_share_silent=s.abs_err_sum.sum() / g.abs_err_sum.sum(),
                         silent_with_workable=(s.best >= L.HIT).mean()))
    out = pd.DataFrame(rows)
    assert len(out) == 17 and np.isfinite(out.drop(columns="reaction").to_numpy()).all()
    out.to_csv(L.OUT / "silent_pairs.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"pooled share of silent pair rows with a workable condition anywhere: {(e[e.silent].best >= L.HIT).mean():.3f}")
    example()


def example(name="Acylation", seed=0, fold=0):
    ex = importlib.import_module("09_example_and_chemistry")
    block = L.load_block(name)
    logged, hist_mean = ex.fold_setup(block, seed, fold)
    Y = block.Y[logged]
    run, report = L.run_policy("literature", Y, hist_mean, 0.0, L.rng_for(block.name, "literature", 0.0, seed, fold))
    rows = []
    for i, j in enumerate(logged):
        tried = np.sort(Y[i][run[i]])[::-1]
        assert len(tried) == 8 and report[i].sum() == (tried[0] >= L.HIT)
        rows.append(dict(pair=block.pairs[j], recorded=bool(report[i].any()), best_library=np.nanmax(block.Y[j]),
                         **{f"tried_{k + 1}": v for k, v in enumerate(tried)}))
    out = pd.DataFrame(rows)
    out.to_csv(L.OUT / "silent_pairs_example.csv", index=False)
    silent = out[~out.recorded]
    print(f"example {name}: {len(out)} pairs, {len(silent)} without a record; "
          f"{(silent.best_library < L.HIT).mean():.3f} of them below 20% across the library")


if __name__ == "__main__":
    main()
