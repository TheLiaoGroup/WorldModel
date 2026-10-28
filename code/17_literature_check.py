"""Real-literature check of the simulated reporting rule (rules fixed on 2026-10-25 before outcomes were read; see Methods).

Literature side: single-step, deduplicated records of the HTE pairs found by 16_literature_overlap.py, with yield_reg
in [0, 1]; the literature value of a pair is the median of its records. HTE side: every analysed condition of the same
pair (raw tables, analysis conventions). Criteria L1-L3 as fixed with the rules.
Outputs: out/literature_check_pairs.csv, out/literature_check_verdict.csv.
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as L

B = importlib.import_module("01_build_blocks")
OV = importlib.import_module("16_literature_overlap")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
EXPECTED_RECORDS = (147, 349)  # matched pairs, records, fixed with the rules
KEY = ["reaction", "pair", "source_raw", "reagent_txt_solvent_exclusive", "solvent_txt", "temperature_celsius_raw", "time_hrs_raw"]
N_BOOT = 1000


def literature_records():
    r = pd.read_csv(OUT / "literature_overlap_records.csv", dtype=str)
    r = r[r.number_of_steps == "1.0"].drop_duplicates(KEY)
    want = EXPECTED_RECORDS
    got = (r.groupby(["reaction", "pair"]).ngroups, len(r))
    if got != want:
        raise SystemExit(f"records differ from the fixed counts: expected {want}, now {got}")
    print(f"records match the fixed counts: {got[0]} pairs, {got[1]} records")
    # yields are read only now, after the rules were fixed
    cache = {}
    vals = []
    for f, rows in r.groupby("source_file"):
        part = pd.read_csv(OV.LIT / f, usecols=["yield_reg"], dtype=str)
        cache[f] = part
        for idx in rows.index:
            vals.append((idx, part.yield_reg.iloc[int(rows.loc[idx, "source_row"]) - 1]))
    y = pd.Series(dict(vals))
    r["yield_lit"] = pd.to_numeric(y.reindex(r.index), errors="coerce")
    return r


def hte_yields(reactions, pairs):
    out = {}
    for name in reactions:
        table = B.ARRAYS[name][0]
        rec = B.load(name, table)
        a = rec[rec.yield_fraction.notna()]
        cond_mean = a.groupby("condition_id").yield_fraction.agg(["sum", "count"])
        for p in pairs[pairs.reaction == name].pair:
            g = a[a.pair == p].groupby("condition_id").yield_fraction.mean()
            others = cond_mean.loc[g.index]
            own = a[a.pair == p].groupby("condition_id").yield_fraction.agg(["sum", "count"])
            hist = ((others["sum"] - own["sum"]) / (others["count"] - own["count"]).replace(0, np.nan)).fillna(0.0)
            out[(name, p)] = (g.to_numpy(), hist.to_numpy())
    return out


def simulated_record(y, hist, key):
    """Pilot literature rule on one pair: 8 habit-weighted conditions, best kept if >= 20%; mean over 10 seeds."""
    rec = []
    w = hist + L.HABIT_FLOOR
    for seed in L.SEEDS:
        rng = L.rng_for("litcheck", key, seed)
        pick = rng.choice(len(y), size=min(L.K, len(y)), replace=False, p=w / w.sum())
        best = y[pick].max()
        if best >= L.HIT:
            rec.append(best)
    return np.mean(rec) if rec else np.nan


def main():
    r = literature_records()
    valid = r[r.yield_lit.between(0, 1)]
    dropped = len(r) - len(valid)
    pairs = valid.groupby(["reaction", "pair"]).yield_lit.agg(["median", "size"]).reset_index()
    pairs.columns = ["reaction", "pair", "lit_value", "lit_records"]
    hte = hte_yields(pairs.reaction.unique(), pairs)
    rows = []
    for t in pairs.itertuples():
        y, hist = hte[(t.reaction, t.pair)]
        rows.append(dict(reaction=t.reaction, pair=t.pair, lit_value=t.lit_value, lit_records=t.lit_records,
                         hte_conditions=len(y), hte_mean=y.mean(), hte_median=np.median(y), hte_best=y.max(),
                         percentile=(y < t.lit_value).mean(), real_gap=t.lit_value - y.mean(),
                         sim_gap=simulated_record(y, hist, (t.reaction, t.pair)) - y.mean()))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "literature_check_pairs.csv", index=False)

    l1_share = (valid.yield_lit >= L.HIT).mean()
    rng = L.rng_for("litcheck_boot")
    n = len(d)
    both = d.dropna(subset=["sim_gap"])
    boot = {"pct": [], "real": [], "sim": [], "ratio": []}
    for _ in range(N_BOOT):
        s = d.iloc[rng.integers(0, n, n)]
        sb = s.dropna(subset=["sim_gap"])
        boot["pct"].append(s.percentile.median())
        boot["real"].append(s.real_gap.median())
        boot["sim"].append(sb.sim_gap.median())
        boot["ratio"].append(s.real_gap.median() / sb.sim_gap.median())
    q = {k: np.quantile(v, [0.025, 0.975]) for k, v in boot.items()}
    pct_med, real_med, sim_med = d.percentile.median(), d.real_gap.median(), both.sim_gap.median()
    ratio = real_med / sim_med
    v = dict(records_valid=len(valid), records_dropped=dropped, pairs=n,
             l1_share_ge20=l1_share, L1=l1_share >= 0.90,
             median_percentile=pct_med, pct_lo=q["pct"][0], pct_hi=q["pct"][1],
             L2=pct_med >= 0.75 and q["pct"][0] > 0.50,
             median_real_gap=real_med, real_lo=q["real"][0], real_hi=q["real"][1],
             median_sim_gap=sim_med, sim_lo=q["sim"][0], sim_hi=q["sim"][1], pairs_with_sim=len(both),
             ratio=ratio, ratio_lo=q["ratio"][0], ratio_hi=q["ratio"][1],
             L3=q["real"][0] > 0 and q["sim"][0] > 0 and 0.5 <= ratio <= 2,
             share_lit_above_hte_best=(d.lit_value > d.hte_best).mean(),
             median_hte_mean=d.hte_mean.median(), median_lit_value=d.lit_value.median(), median_hte_best=d.hte_best.median())
    v["supported"] = bool(v["L1"] and v["L2"])
    pd.DataFrame([v]).to_csv(OUT / "literature_check_verdict.csv", index=False)
    print(pd.Series(v).to_string())
    print(d.groupby("reaction").agg(pairs=("pair", "size"), pct=("percentile", "median"), real_gap=("real_gap", "median"),
                                    sim_gap=("sim_gap", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
