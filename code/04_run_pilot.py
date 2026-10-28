"""Run the counterfactual pilot on the three real blocks and apply the criteria (fixed on 2026-10-25 before outcomes were read; see Methods).

Outputs (out/): pilot_errors.csv.gz (per pair, seed, fold, policy, fraction, model, arm), pilot_summary.csv,
pilot_verdict.csv.
"""

import sys

import numpy as np
import pandas as pd

import pilot_lib as L

PILOT = ["Acylation", "Suzuki", "Buchwald-Hartwig"]
EXTENSION = ["Allylic-substitution", "Buchwald-Hartwig CMC", "C-O competitive reactions", "Chan-Lam", "Esterification",
             "Grignard", "Heck", "Hiyama", "Kumada", "Pd-carbonylation", "Reductive-amination", "SNAr", "Sonogashira",
             "Ullmann"]
MODELS = ["additive", "rf", "lowrank"]
REAL = ["habit", "first_hit", "literature"]
REL = 0.10


def main():
    ext = "--extension" in sys.argv
    REACTIONS = EXTENSION if ext else PILOT
    tag = "_extension" if ext else ""
    frames = []
    for rx in REACTIONS:
        block = L.load_block(rx)
        df = L.experiment(block, MODELS)
        frames.append(df)
        print(f"{rx}: {len(df)} rows", flush=True)
    df = pd.concat(frames)
    skipped = pd.DataFrame(L.SKIPPED, columns=["reaction", "policy", "f", "seed", "fold"])
    skipped.to_csv(L.OUT / f"pilot_skipped{tag}.csv", index=False)
    print(f"{len(skipped)} (policy, fraction, seed, fold) runs recorded no well and were left out", flush=True)
    df.to_csv(L.OUT / f"pilot_errors{tag}.csv.gz", index=False)
    s = L.summarise(df)
    s.to_csv(L.OUT / f"pilot_summary{tag}.csv", index=False)

    verdict = []
    for rx in REACTIONS:
        r = s[s.reaction == rx]
        base = r[(r.policy == "habit") & (r.f == 0) & r.model.isin(["rf", "lowrank"])].set_index("model").mae_random
        primary = base.idxmin()
        p = r[(r.model == primary)]
        null = p[(p.policy == "random_control") & (p.f == 0)].iloc[0]
        for pol in REAL:
            row = p[(p.policy == pol) & (p.f == 0)].iloc[0]
            a = row.lo > 0 and row.delta >= REL * row.mae_random
            b = row.hi < REL * row.mae_random
            fix = p[(p.policy == pol) & (p.hi < REL * p.mae_random)].f
            verdict.append(dict(reaction=rx, primary_model=primary, policy=pol, mae_random=row.mae_random,
                                delta=row.delta, lo=row.lo, hi=row.hi, rel=row.rel, meets_A=a, meets_B=b,
                                smallest_fix_fraction=fix.min() if len(fix) else np.nan,
                                null_lo=null.lo, null_hi=null.hi, null_ok=null.lo <= 0 <= null.hi))
    v = pd.DataFrame(verdict)
    v.to_csv(L.OUT / f"pilot_verdict{tag}.csv", index=False)
    per_rx = v.groupby("reaction").agg(any_A=("meets_A", "any"), all_B=("meets_B", "all"))
    outcome = "A" if per_rx.any_A.sum() >= 2 else "B" if per_rx.all_B.sum() >= 2 else "neither"
    pd.set_option("display.width", 250)
    print(v.round(4).to_string(index=False))
    print(per_rx)
    print(f"\nOutcome: {outcome}; null control holds in {int(v.drop_duplicates("reaction").null_ok.sum())} of {len(REACTIONS)} reactions")
    sys.exit(0)


if __name__ == "__main__":
    main()
