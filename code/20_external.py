"""External complete arrays (rules fixed on 2026-10-25 before outcomes were read; see Methods): Ahneman/Doyle Buchwald-Hartwig and
Perera Suzuki, run with the pilot protocol unchanged; substrates and condition components encoded by identity.
Outputs: out/external_errors.csv.gz, out/external_summary.csv, out/external_verdict.csv.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as L

ROOT = Path(__file__).resolve().parents[1]
EXT = Path(os.environ.get("WM_EXTERNAL_DIR", ROOT / "data" / "external"))
REL, CUT = 0.10, 0.50


def onehot(values):
    cats = sorted(set(values))
    return np.array([[float(v == c) for c in cats] for v in values], dtype=np.float32)


def make_block(name, d, pair_cols, cond_cols, ycol):
    d = d.copy()
    d["pair"] = d[pair_cols].astype(str).agg("|".join, axis=1)
    d["cond"] = d[cond_cols].astype(str).agg("|".join, axis=1)
    if d.duplicated(["pair", "cond"]).any():
        raise SystemExit(f"{name}: repeated pair-condition cells")
    d["y"] = pd.to_numeric(d[ycol], errors="coerce") / 100.0
    if d.y.max() > 1.05 or d.y.min() < -0.05:
        raise SystemExit(f"{name}: yields outside 0-100%")
    m = d.pivot(index="pair", columns="cond", values="y")
    first = d.drop_duplicates("pair").set_index("pair").loc[m.index]
    pair_fp = np.hstack([onehot(first[c].astype(str).tolist()) for c in pair_cols])
    conds = d.drop_duplicates("cond").set_index("cond").loc[m.columns]
    cond_x = np.hstack([onehot(conds[c].astype(str).tolist()) for c in cond_cols])
    print(f"{name}: {m.shape[0]} pairs x {m.shape[1]} conditions, {int(m.notna().values.sum())} measured cells")
    return L.Block(name, m.to_numpy(float), list(m.index), list(m.columns), pair_fp, cond_x)


def main():
    ahn = pd.read_excel(EXT / "Dreher_and_Doyle_input_data.xlsx", sheet_name="FullCV_01")
    per = pd.read_excel(EXT / "aap9112_Data_File_S1.xlsx").fillna({"Ligand_Short_Hand": "none",
                                                                   "Reagent_1_Short_Hand": "none",
                                                                   "Solvent_1_Short_Hand": "none"})
    blocks = [make_block("Ahneman-BH", ahn, ["Aryl halide"], ["Ligand", "Base", "Additive"], "Output"),
              make_block("Perera-Suzuki", per, ["Reactant_1_Short_Hand", "Reactant_2_Name"],
                         ["Ligand_Short_Hand", "Reagent_1_Short_Hand", "Solvent_1_Short_Hand"], "Product_Yield_PCT_Area_UV")]
    frames = [L.experiment(b, ["additive", "rf"]) for b in blocks]
    df = pd.concat(frames)
    df.to_csv(L.OUT / "external_errors.csv.gz", index=False)
    s = L.summarise(df)
    s.to_csv(L.OUT / "external_summary.csv", index=False)
    rows = []
    rf = s[s.model == "rf"]
    for b in blocks:
        r = rf[rf.reaction == b.name].set_index(["policy", "f"])
        lit0, lit1, null = r.loc[("literature", 0.0)], r.loc[("literature", 0.10)], r.loc[("random_control", 0.0)]
        rows.append(dict(dataset=b.name, mae_random=lit0.mae_random, delta=lit0.delta, lo=lit0.lo, hi=lit0.hi,
                         E1=bool(lit0.lo > 0 and lit0.delta >= REL * lit0.mae_random),
                         cut_1_random=1 - lit1.delta / lit0.delta, E2=bool(1 - lit1.delta / lit0.delta > CUT),
                         habit_delta=r.loc[("habit", 0.0)].delta, first_hit_delta=r.loc[("first_hit", 0.0)].delta,
                         null_lo=null.lo, null_hi=null.hi, null_ok=bool(null.lo <= 0 <= null.hi),
                         skipped_literature_folds=sum(1 for k in L.SKIPPED if k[0] == b.name and k[1] == "literature" and k[2] == 0.0)))
    v = pd.DataFrame(rows)
    v.to_csv(L.OUT / "external_verdict.csv", index=False)
    pd.set_option("display.width", 220)
    print(v.round(4).T.to_string())


if __name__ == "__main__":
    main()
