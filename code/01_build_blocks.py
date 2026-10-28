"""Build the three complete substrate-pair x condition blocks of the counterfactual pilot from the raw tables.

Measurement convention (every record):
  numerical yield                      -> that yield (fraction, 0-1)
  blank yield, yield_clf = 0           -> 0   (screen found no product)
  blank yield, yield_clf = 1           -> missing (product detected, not quantified)
  blank yield, no yield_clf            -> missing
Missing records are never imputed. Wells on the acylation exclusion list (version conflicts) are removed first.

Block rules. Membership never depends on a yield value:
  Acylation, Buchwald-Hartwig: the substrate pairs whose attempted condition set equals the array's most common one.
  Suzuki: conditions attempted for >= 80% of pairs, then pairs with an analysed yield for >= 95% of those conditions.
  The other 16 arrays (added after the pilot): whichever of those two rules gives more analysed wells.
  Pairs measured more than once under the same condition (on separate plates) leave the block; their wells go to
  replicates.csv and serve only as the repeatability reference.

Outputs (out/):
  wells.csv.gz       one row per analysed record in the three blocks
  replicates.csv     analysed records of the pairs measured twice under the same condition
  conditions.csv     one row per block condition: each reagent, solvent, temperature and time
  blocks_summary.csv the per-block description reported in Supplementary Table S1
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("WM_RAW_DIR", ROOT / "data" / "raw"))
EXCLUDE = Path(os.environ.get("WM_EXCLUDE", ROOT / "data" / "EXCLUDE_wells.csv"))
OUT = ROOT / "out"

DENSE = (0.80, 0.95)
ARRAYS = {  # block name: (raw table, reaction_type in the condition table, block rule)
    "Acylation": ("Acylation_260805.xlsx", "acylation", "library"),
    "Suzuki": ("Suzuki_260806.xlsx", "suzuki", DENSE),
    "Buchwald-Hartwig": ("Buchwald-Hartwig_260805.xlsx", "buchwald", "library"),
    "Allylic-substitution": ("Allylic-substitution_260731.csv", "allylic-substitution", "auto"),
    "Buchwald-Hartwig CMC": ("Buchwald-Hartwig(CMC)_260805.csv", "B-H (CMC)", "auto"),
    "C-O competitive reactions": ("C-O competitive reactions_230730.csv", "C-N or C-O competitive reactions", "auto"),
    "Chan-Lam": ("Chan-Lam_260731.csv", "chan-lam", "auto"),
    "Esterification": ("Esterification_260731.csv", "esterification", "auto"),
    "Grignard": ("Grignard_260731.csv", "grignard", "auto"),
    "Heck": ("Heck_260810.csv", "heck", "auto"),
    "Hiyama": ("Hiyama_260729.csv", "hiyama", "auto"),
    "Kumada": ("Kumada_.xlsx", "kumada", "auto"),
    "Minisci": ("Minisci_260731.csv", "minisci", "auto"),
    "Pd-carbonylation": ("Pd-carbonylation_260731.csv", "pd-carbonylation", "auto"),
    "Reductive-amination": ("Reductive-amination_260729.csv", "reductive-amination", "auto"),
    "SNAr": ("SNAr_260729.csv", "snar", "auto"),
    "Sonogashira": ("Sonogashira_260729.csv", "sonogashira", "auto"),
    "Suzuki Bpin": ("Suzuki Bpin_260806.csv", "suzuki(bpin)", "auto"),
    "Ullmann": ("Ullmann.xlsx", "ullmann", "auto"),
}
REAGENTS = [f"reagent{i}_id" for i in range(1, 7)]
COLS = ["reaction", "pair", "sub_1_smiles", "sub_2_smiles", "product_smiles", "condition_id",
        "yield_fraction", "status", "name0", "plate", "archive"]


def measurement(frame):
    value = pd.to_numeric(frame["yield"], errors="coerce")
    clf = pd.to_numeric(frame["yield_clf"], errors="coerce")
    q = value.notna()
    status = np.select([q, ~q & clf.eq(0), ~q & clf.eq(1)],
                       ["quantified", "zero_by_rule", "detected_not_quantified"], default="no_screen_record")
    out = frame.assign(status=status, yield_fraction=np.where(q, value, np.where(clf.eq(0) & ~q, 0.0, np.nan)))
    if out.loc[q, "yield_fraction"].max() > 1.05:
        raise SystemExit("yield column is not a 0-1 fraction")
    return out


def load(name, table):
    raw = pd.read_csv(RAW / table, encoding_errors="replace") if table.endswith(".csv") else pd.read_excel(RAW / table)
    raw["plate"] = raw["name0"].str.split("-LC").str[0]
    if name == "Acylation":
        ex = pd.read_csv(EXCLUDE)
        ex = set(ex.loc[ex.rxn == "Acylation", "name0"])
        hit = raw["name0"].isin(ex)
        if hit.sum() != len(ex):
            raise SystemExit(f"exclusion list: {len(ex)} wells listed, {hit.sum()} found")
        raw = raw[~hit]
    raw["pair"] = raw["sub_1_smiles"] + ">>" + raw["sub_2_smiles"]
    return measurement(raw)


def select(records, rule):
    if rule == "auto":
        a, b = select(records, "library"), select(records, DENSE)
        return a if a.yield_fraction.notna().sum() >= b.yield_fraction.notna().sum() else b
    if rule == "library":
        lib_of_pair = records.groupby("pair").condition_id.apply(frozenset)
        lib = lib_of_pair.value_counts().index[0]
        keep = records.pair.isin(lib_of_pair.index[lib_of_pair == lib]) & records.condition_id.isin(lib)
        return records[keep]
    c_th, p_th = rule
    attempted = records.groupby(["pair", "condition_id"]).size().unstack().notna()
    cols = attempted.columns[attempted.mean(0) >= c_th]
    sub = records[records.condition_id.isin(cols)]
    analysed = sub[sub.yield_fraction.notna()].groupby("pair").condition_id.nunique() / len(cols)
    return sub[sub.pair.isin(analysed.index[analysed >= p_th])]


def conditions(block, name, rtype, ctab):
    rec = block[["condition_id"] + REAGENTS + ["solvent_id", "temp_id", "time_id"]].drop_duplicates()
    if rec.condition_id.duplicated().any():
        raise SystemExit(f"{name}: a condition id carries more than one recipe")
    ref = ctab[ctab.reaction_type == rtype].set_index("condition_id")
    names = {}
    for i in range(1, 7):
        names.update(dict(zip(ref[f"Reagent{i}_id"], ref[f"Reagent{i}_Standardized Name"])))
    names.update(dict(zip(ref["solvent_id"], ref["Standardized Name"])))
    names = {k: v for k, v in names.items() if isinstance(k, str)}
    rec = rec.assign(reaction=name).set_index("condition_id")
    for col in REAGENTS + ["solvent_id"]:
        rec[col.replace("_id", "_name")] = rec[col].map(names)
        unnamed = rec[col].notna() & rec[col.replace("_id", "_name")].isna()
        if unnamed.any():
            raise SystemExit(f"{name}: {col} ids without a name: {sorted(rec.loc[unnamed, col].unique())}")
    return rec.reset_index()


def describe(name, wells):
    m = wells.groupby(["pair", "condition_id"]).yield_fraction.mean().unstack()
    v = m.values
    ok = np.isfinite(v)
    pm, cm, gm = np.nanmean(v, 1, keepdims=True), np.nanmean(v, 0, keepdims=True), np.nanmean(v)
    best = np.nanargmax(np.where(ok, v, -1), 1)
    return dict(screen=name, pairs=m.shape[0], conditions=m.shape[1], wells=int(ok.sum()), fill=ok.mean(),
                substrate_1=wells.sub_1_smiles.nunique(), substrate_2=wells.sub_2_smiles.nunique(),
                mean_yield=np.nanmean(v), wells_ge20=(v[ok] >= 0.2).mean(),
                pairs_best_ge20=(np.nanmax(v, 1) >= 0.2).mean(), distinct_best_conditions=len(set(best)),
                interaction_share=np.nanvar(v - pm - cm + gm) / np.nanvar(v))


def main():
    OUT.mkdir(exist_ok=True)
    ctab = pd.read_excel(RAW / "0_total condition_260810.xlsx")
    wells, conds, summary, replicates = [], [], [], []
    for name, (table, rtype, rule) in ARRAYS.items():
        block = select(load(name, table), rule)
        conds.append(conditions(block, name, rtype, ctab))
        w = block[block.yield_fraction.notna()].assign(reaction=name)
        repeated = w.groupby(["pair", "condition_id"]).name0.transform("size") > 1
        rep_pairs = set(w.loc[repeated, "pair"])
        replicates.append(w[w.pair.isin(rep_pairs)][COLS])
        w = w[~w.pair.isin(rep_pairs)]
        if w.empty:
            raise SystemExit(f"{name}: empty block")
        wells.append(w[COLS])
        summary.append(describe(name, w))
        print(f"{name}: {len(block)} records in block, {len(w)} analysed, {len(rep_pairs)} repeated pairs set aside")
    pd.concat(wells).to_csv(OUT / "wells.csv.gz", index=False)
    pd.concat(replicates).to_csv(OUT / "replicates.csv", index=False)
    pd.concat(conds).to_csv(OUT / "conditions.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "blocks_summary.csv", index=False)
    print(s.round(3).T.to_string())


if __name__ == "__main__":
    main()
