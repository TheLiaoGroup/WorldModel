"""Reagent effect read by the models: mean yield of unrun conditions sharing the first reagent with the recorded success,
minus that of conditions with a different first reagent, for the measured yields and for each model's predictions.

Input: out/chemistry_rows.csv.gz (09_example_and_chemistry.py). Intervals resample whole substrate pairs and seeds
together (1,000 draws), as for the group means. Output: out/chemistry_gap.csv (points as percentage points).
"""

import numpy as np
import pandas as pd

import pilot_lib as L

COLS = ["measured", "pred_literature", "pred_random_8"]
N_BOOT = 1000


def main():
    r = pd.read_csv(L.OUT / "chemistry_rows.csv.gz")
    rows = []
    for rx, g in r.groupby("reaction"):
        cube = g.pivot_table(index=["pair", "seed"], columns="group", values=COLS + ["n"], aggfunc="sum").fillna(0)
        pairs = cube.index.get_level_values(0).unique()
        seeds = sorted(cube.index.get_level_values(1).unique())
        grid = {(c, k): cube[(c, k)].unstack("seed").reindex(index=pairs, columns=seeds).fillna(0).to_numpy()
                for c in COLS + ["n"] for k in ("same_first_reagent", "different_first_reagent")}

        def gaps(ix):
            ns, nd = grid[("n", "same_first_reagent")][ix].sum(), grid[("n", "different_first_reagent")][ix].sum()
            return [100 * (grid[(c, "same_first_reagent")][ix].sum() / ns - grid[(c, "different_first_reagent")][ix].sum() / nd)
                    for c in COLS]

        point = gaps(np.ix_(range(len(pairs)), range(len(seeds))))
        rng = L.rng_for("chemistry_gap", rx)
        boot = np.array([gaps(np.ix_(rng.integers(0, len(pairs), len(pairs)), rng.integers(0, len(seeds), len(seeds))))
                         for _ in range(N_BOOT)])
        for j, c in enumerate(COLS):
            rows.append(dict(reaction=rx, series=c, gap=point[j], lo=np.quantile(boot[:, j], 0.025),
                             hi=np.quantile(boot[:, j], 0.975)))
    out = pd.DataFrame(rows)
    out.to_csv(L.OUT / "chemistry_gap.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
