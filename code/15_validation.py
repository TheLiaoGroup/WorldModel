"""Frozen validation on substrate pairs this project never analysed (rules fixed on 2026-10-25 before outcomes were read; see Methods).

Step 1 builds the groups from structure only and checks them against the counts fixed with the rules; step 2 runs the pilot's
protocol unchanged on every group and applies criteria V1-V3. Outputs (out/): validation_groups.csv,
validation_errors.csv.gz, validation_summary.csv, validation_verdict.csv, validation_book.csv,
validation_chemistry.csv, validation_skipped.csv.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as L

B = importlib.import_module("01_build_blocks")
BOOK = importlib.import_module("07_book_score")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
EXPECTED_GROUPS = (9, 25, 483, 37518)  # arrays, groups, substrate pairs, records, fixed with the rules
REAGENTS = [f"reagent{i}_id" for i in range(1, 7)]
RECIPE = REAGENTS + ["solvent_id", "temp_id", "time_id"]
MIN_PAIRS, MIN_CONDS, REL, CUT, BOOK_RATIO, BOOK_WELLS = 10, 24, 0.10, 0.50, 1.5, 100


def build_groups():
    used = pd.read_csv(OUT / "wells.csv.gz", usecols=["reaction", "pair"])
    rep = pd.read_csv(OUT / "replicates.csv", usecols=["reaction", "pair"])
    groups, rows, dropped = [], [], []
    for name, (table, _, _) in B.ARRAYS.items():
        rec = B.load(name, table)
        seen = set(used.loc[used.reaction == name, "pair"]) | set(rep.loc[rep.reaction == name, "pair"])
        rest = rec[~rec.pair.isin(seen)]
        lib = rest.groupby("pair").condition_id.apply(frozenset)
        for k, (conds, n) in enumerate(lib.value_counts().items()):
            if n < MIN_PAIRS or len(conds) < MIN_CONDS:
                continue
            gname = f"{name}#{k + 1}"
            g = rest[rest.pair.isin(lib.index[lib == conds]) & rest.condition_id.isin(conds)].copy()
            recipes = g[["condition_id"] + RECIPE].astype(str).drop_duplicates()
            if recipes.condition_id.duplicated().any():
                dropped.append(gname)
                continue
            if g.groupby(["pair", "condition_id"]).size().gt(1).any():
                raise SystemExit(f"{gname}: repeated pair-condition cells")
            groups.append((name, gname, g))
            rows.append(dict(array=name, group=gname, pairs=n, conditions=len(conds), records=len(g),
                             analysed_share=g.yield_fraction.notna().mean()))
    table = pd.DataFrame(rows)
    return groups, table, dropped


def check_against_plan(table, dropped):
    want = list(EXPECTED_GROUPS)
    got = [table.array.nunique(), len(table), int(table.pairs.sum()), int(table.records.sum())]
    if dropped or got != want:
        raise SystemExit(f"groups differ from the plan: plan {want}, built {got}, dropped {dropped}")
    print(f"groups match the plan: {got}")


def make_block(gname, g):
    m = g.pivot(index="pair", columns="condition_id", values="yield_fraction")
    recipes = g[["condition_id"] + RECIPE].drop_duplicates("condition_id").set_index("condition_id").loc[m.columns]
    pair_fp, cond_x = L.features(recipes, g.drop_duplicates("pair").set_index("pair").loc[m.index])
    return L.Block(gname, m.to_numpy(float), list(m.index), list(m.columns), pair_fp, cond_x), recipes


def cut_interval(df, n_boot=1000):
    """Pooled 1 - Delta(1 random well)/Delta(0) with seeds and pairs resampled together."""
    lit = df[(df.policy == "literature") & (df.model == "rf") & df.f.isin([0.0, 0.10])]
    piv = lit.pivot_table(index=["pair", "seed"], columns=["f", "arm"], values=["abs_err_sum", "n_eval"], aggfunc="sum")
    pairs = piv.index.get_level_values(0).unique()
    seeds = sorted(piv.index.get_level_values(1).unique())
    grid = lambda v, f, arm: piv[(v, f, arm)].unstack("seed").reindex(index=pairs, columns=seeds).fillna(0).to_numpy()
    Ep0, Er0, N0 = grid("abs_err_sum", 0.0, "policy"), grid("abs_err_sum", 0.0, "random"), grid("n_eval", 0.0, "policy")
    Ep1, Er1, N1 = grid("abs_err_sum", 0.1, "policy"), grid("abs_err_sum", 0.1, "random"), grid("n_eval", 0.1, "policy")
    point = 1 - ((Ep1.sum() - Er1.sum()) / N1.sum()) / ((Ep0.sum() - Er0.sum()) / N0.sum())
    rng = L.rng_for("validation_cut")
    boot = np.empty(n_boot)
    for b in range(n_boot):
        ix = np.ix_(rng.integers(0, len(pairs), len(pairs)), rng.integers(0, len(seeds), len(seeds)))
        d0 = (Ep0[ix].sum() - Er0[ix].sum()) / N0[ix].sum()
        d1 = (Ep1[ix].sum() - Er1[ix].sum()) / N1[ix].sum()
        boot[b] = 1 - d1 / d0
    return point, np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def main():
    groups, table, dropped = build_groups()
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    check_against_plan(table, dropped)
    table.to_csv(OUT / "validation_groups.csv", index=False)
    if "--structure-only" in sys.argv:
        return

    frames, book_rows, chem_rows = [], [], []
    for array, gname, g in groups:
        block, recipes = make_block(gname, g)
        df = L.experiment(block, ["additive", "rf"])
        df["group"], df["reaction"] = gname, array
        df["pair"] = array + "|" + df["pair"]
        frames.append(df)
        first = recipes.reagent1_id.astype(str).to_numpy()
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
                preds = {}
                for arm, rep in (("policy", rep_p), ("random", run_r)):
                    b_sum, b_n = BOOK.book(block, logged, rep, "rf", seed)
                    preds[arm] = L.fit_predict("rf", block, logged, rep, seed)
                    book_rows.append(dict(reaction=array, group=gname, seed=seed, fold=fold, arm=arm,
                                          book_sum=b_sum, book_n=b_n,
                                          cf_sum=np.abs(preds[arm] - np.nan_to_num(Y))[evalm].sum(), cf_n=int(evalm.sum())))
                if array in ("Acylation", "Buchwald-Hartwig"):
                    for i in np.flatnonzero(rep_p.any(1)):
                        same = first == first[np.flatnonzero(rep_p[i])[0]]
                        for grp, mask in (("same_first_reagent", same), ("different_first_reagent", ~same)):
                            mm = evalm[i] & mask
                            if mm.any():
                                chem_rows.append(dict(reaction=array, group=grp, n=int(mm.sum()), measured=Y[i, mm].sum(),
                                                      pred_literature=preds["policy"][i, mm].sum(),
                                                      pred_random_8=preds["random"][i, mm].sum()))
        print(gname, flush=True)

    df = pd.concat(frames)
    df.to_csv(OUT / "validation_errors.csv.gz", index=False)
    skipped = pd.DataFrame(L.SKIPPED, columns=["group", "policy", "f", "seed", "fold"])
    skipped.to_csv(OUT / "validation_skipped.csv", index=False)

    per_array = L.summarise(df)
    pooled = L.summarise(df.assign(reaction="ALL"))
    summary = pd.concat([per_array, pooled])
    summary.to_csv(OUT / "validation_summary.csv", index=False)

    # skip rule: literature folds skipped in more than half of an array's folds -> reported, not counted
    folds = table.groupby("array").group.count() * len(L.SEEDS) * 2
    lit_skip = skipped[(skipped.policy == "literature") & (skipped.f == 0)].assign(
        array=lambda d: d.group.str.split("#").str[0]).groupby("array").size().reindex(folds.index, fill_value=0)
    counted = folds.index[lit_skip <= folds / 2].tolist()

    rf = summary[summary.model == "rf"]
    lit0 = rf[(rf.policy == "literature") & (rf.f == 0)].set_index("reaction")
    lit1 = rf[(rf.policy == "literature") & (rf.f == 0.10)].set_index("reaction")
    meets_a = (lit0.lo > 0) & (lit0.delta >= REL * lit0.mae_random)
    cut = 1 - lit1.delta / lit0.delta
    v1 = bool(meets_a["ALL"]) and meets_a.loc[counted].sum() >= np.ceil(2 / 3 * len(counted))
    c_point, c_lo, c_hi = cut_interval(df)
    v2 = c_lo > CUT and (cut.loc[counted] > CUT).sum() >= np.ceil(2 / 3 * len(counted))

    bk = pd.DataFrame(book_rows)
    g = bk.groupby(["reaction", "arm"])
    book = pd.DataFrame({"book_mae": g.book_sum.sum() / g.book_n.sum(), "cf_mae": g.cf_sum.sum() / g.cf_n.sum(),
                         "book_wells": g.book_n.sum()}).reset_index()
    book.to_csv(OUT / "validation_book.csv", index=False)
    lb = book[(book.arm == "policy") & (book.book_wells >= BOOK_WELLS)]
    v3 = bool(len(lb)) and bool((lb.cf_mae >= BOOK_RATIO * lb.book_mae).all())
    rb = book[(book.arm == "random") & book.reaction.isin(lb.reaction)]

    null = rf[(rf.policy == "random_control") & (rf.f == 0) & (rf.reaction == "ALL")].iloc[0]
    null_ok = null.lo <= 0 <= null.hi

    ch = pd.DataFrame(chem_rows)
    if len(ch):
        gc = ch.groupby(["reaction", "group"])
        pd.DataFrame({"wells": gc.n.sum(), **{c: gc[c].sum() / gc.n.sum() for c in ("measured", "pred_literature", "pred_random_8")}}
                     ).reset_index().to_csv(OUT / "validation_chemistry.csv", index=False)

    verdict = pd.DataFrame([dict(
        arrays_counted=len(counted), arrays_meeting_A=int(meets_a.loc[counted].sum()),
        pooled_delta=lit0.loc["ALL", "delta"], pooled_lo=lit0.loc["ALL", "lo"], pooled_hi=lit0.loc["ALL", "hi"],
        pooled_mae_random=lit0.loc["ALL", "mae_random"], V1=v1,
        pooled_cut=c_point, cut_lo=c_lo, cut_hi=c_hi, arrays_cut_over_half=int((cut.loc[counted] > CUT).sum()), V2=v2,
        book_arrays=len(lb), book_ratio_min=(lb.cf_mae / lb.book_mae).min() if len(lb) else np.nan, V3=v3,
        random_book_gap_max=(rb.cf_mae - rb.book_mae).abs().max() if len(rb) else np.nan,
        null_lo=null.lo, null_hi=null.hi, null_ok=null_ok, validation_holds=bool(v1 and v2))])
    verdict.to_csv(OUT / "validation_verdict.csv", index=False)
    print(verdict.T.to_string())
    print(lit0[["delta", "lo", "hi", "mae_random"]].assign(cut=cut).round(4).to_string())


if __name__ == "__main__":
    main()
