"""Models trained on real literature records, scored on measured HTE conditions of pairs they never saw
(rules fixed on 2026-10-25 before outcomes were read; see Methods).

All models see substrates only (ECFP4 of both substrates). Literature model trained once per array on a fixed random
sample of 3,000 records per reaction class (yields are read only for sampled records); HTE comparators and
"literature + one random well" per seed on a random half of the block's pairs; scoring on the other half.
Outputs: out/literature_model_errors.csv.gz, out/literature_model_summary.csv, out/literature_model_verdict.csv.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

import pilot_lib as L

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
# literature records per class, fixed with the rules (Suzuki and C-N arylation: palladium-containing records)
EXPECTED_POOL = {"amide": 112976, "suzuki": 83571, "cn_arylation": 26054}
LIT_DIR = Path(os.environ["WM_LIT_DIR"])
CAP, REL, RATIO, CUT, N_BOOT = 3000, 0.10, 1.5, 0.50, 1000
ARRAYS = {  # block: (literature class, palladium only, literature column for HTE substrate 1, for substrate 2)
    "Acylation": ("amide", False, "substrate_b", "substrate_a"),
    "Suzuki": ("suzuki", True, "substrate_a", "substrate_b"),
    "Buchwald-Hartwig": ("cn_arylation", True, "substrate_b", "substrate_a"),
}
DEDUP = ["substrate_a", "substrate_b", "product", "reagent_txt_solvent_exclusive", "solvent_txt",
         "temperature_celsius_raw", "time_hrs_raw", "source_raw"]
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
_fp_cache = {}


def fp(smiles):
    if smiles not in _fp_cache:
        m = Chem.MolFromSmiles(smiles)
        _fp_cache[smiles] = GEN.GetFingerprintAsNumPy(m).astype(np.float32) if m is not None else None
    return _fp_cache[smiles]


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def rf(seed):
    return RandomForestRegressor(n_estimators=200, max_features=0.33, n_jobs=-1, random_state=seed)


def literature_sets(pool, blocks):
    want = EXPECTED_POOL
    sets, notes = {}, []
    for rx, (cls, pd_only, c1, c2) in ARRAYS.items():
        g = pool[pool.reaction_class == cls]
        if pd_only:
            g = g[g.reagent_txt_solvent_exclusive.fillna("").str.contains("pallad|Pd", case=False)]
        n_cls = len(g) if pd_only else len(pool[pool.reaction_class == cls])
        if n_cls != want[cls]:
            raise SystemExit(f"{rx}: literature pool {n_cls} differs from the fixed count {want[cls]}")
        g = g.drop_duplicates(DEDUP)
        b = blocks[rx]
        hte_pairs = {frozenset((canon(a), canon(c))) for a, c in zip(b["s1"], b["s2"])}
        same = g.apply(lambda r: frozenset((r.substrate_a, r.substrate_b)) in hte_pairs, axis=1)
        sets[rx] = dict(train=g[~same].copy(), same_pair=g[same].copy(), c1=c1, c2=c2)
        notes.append(dict(reaction=rx, pool=n_cls, after_dedup=len(g), same_hte_pair=int(same.sum())))
    print(pd.DataFrame(notes).to_string(index=False))
    return sets, notes


def read_yields(frames):
    need = pd.concat(frames)
    out = {}
    for f, rows in need.groupby("source_file"):
        col = pd.read_csv(LIT_DIR / f, usecols=["yield_reg"], dtype=str).yield_reg
        for i, r in zip(rows.index, rows.source_row.astype(int)):
            out[(f, r)] = col.iloc[r - 1]
    return out


def capped_training(rx, s):
    """A fixed random order of the eligible records; yields are read in that order only until CAP valid ones are held."""
    order = s["train"].sample(frac=1, random_state=int(L.rng_for("litmodel_cap", rx).integers(1e9)))
    kept, dropped, start = [], 0, 0
    while sum(map(len, kept)) < CAP and start < len(order):
        need = CAP - sum(map(len, kept))
        chunk = order.iloc[start:start + need].copy()
        start += need
        y = read_yields([chunk])
        chunk["y"] = pd.to_numeric([y[(f, int(r))] for f, r in zip(chunk.source_file, chunk.source_row)], errors="coerce")
        ok = chunk.y.between(0, 1)
        dropped += int((~ok).sum())
        kept.append(chunk[ok])
    return pd.concat(kept), dropped


def features(df, c1, c2):
    keep, X = [], []
    for i, (a, b) in enumerate(zip(df[c1], df[c2])):
        fa, fb = fp(a), fp(b)
        if fa is not None and fb is not None:
            keep.append(i)
            X.append(np.concatenate([fa, fb]))
    return np.array(X), keep


def main():
    pool = pd.read_csv(OUT / "literature_pool.csv.gz", dtype=str)
    blocks = {}
    for rx in ARRAYS:
        b = L.load_block(rx)
        w = pd.read_csv(OUT / "wells.csv.gz", usecols=["reaction", "pair", "sub_1_smiles", "sub_2_smiles"])
        w = w[w.reaction == rx].drop_duplicates("pair").set_index("pair").loc[b.pairs]
        blocks[rx] = dict(block=b, s1=w.sub_1_smiles.tolist(), s2=w.sub_2_smiles.tolist())
    sets, notes = literature_sets(pool, blocks)

    rows, summary_rows = [], []
    for rx, s in sets.items():
        t0 = time.time()
        b = blocks[rx]["block"]
        # yields are read only now, after the rules were fixed, and only for the sampled records
        train, dropped = capped_training(rx, s)
        X_lit, keep = features(train, s["c1"], s["c2"])
        y_lit = train.y.to_numpy()[keep]
        pair_X = b.pair_fp  # ECFP4 of HTE substrate 1 and 2, same featurisation
        # book score: five random folds of the literature training set
        folds = L.rng_for("litmodel_book", rx).permutation(len(y_lit)) % 5
        book_err = np.empty(len(y_lit))
        for k in range(5):
            m = rf(k).fit(X_lit[folds != k], y_lit[folds != k])
            book_err[folds == k] = np.abs(m.predict(X_lit[folds == k]) - y_lit[folds == k])
        lit_model = rf(0).fit(X_lit, y_lit)
        p_lit = lit_model.predict(pair_X)
        Y = b.Y
        P, C = Y.shape
        for seed in L.SEEDS:
            perm = L.rng_for(b.name, "split", seed).permutation(P)
            tr, te = perm[: P // 2], perm[P // 2:]
            rng = L.rng_for("litmodel_wells", rx, seed)
            r8_rows, r1_rows, sim_rows = [], [], []
            Ytr = Y[tr]
            cond_sum, cond_n = np.nansum(Ytr, 0), np.sum(~np.isnan(Ytr), 0)
            for k, i in enumerate(tr):
                avail = np.flatnonzero(~np.isnan(Y[i]))
                pick = rng.choice(avail, size=min(L.K, len(avail)), replace=False)
                r8_rows += [(i, Y[i, c]) for c in pick]
                r1_rows.append((i, Y[i, pick[0]]))
                own = ~np.isnan(Y[i])
                hist = np.where(cond_n - own > 0, (cond_sum - np.nan_to_num(Y[i])) / np.maximum(cond_n - own, 1), 0.0)[avail]
                w = hist + L.HABIT_FLOOR
                hp = rng.choice(avail, size=min(L.K, len(avail)), replace=False, p=w / w.sum())
                best = np.nanmax(Y[i, hp])
                if best >= L.HIT:
                    sim_rows.append((i, best))
            preds = {"literature": p_lit[te]}
            for name, rws in (("random_8", r8_rows), ("simulated_literature", sim_rows)):
                idx = np.array([r[0] for r in rws])
                preds[name] = rf(seed).fit(pair_X[idx], np.array([r[1] for r in rws])).predict(pair_X[te])
            idx1 = np.array([r[0] for r in r1_rows])
            fixm = rf(seed).fit(np.vstack([X_lit, pair_X[idx1]]), np.concatenate([y_lit, [r[1] for r in r1_rows]]))
            preds["literature_plus_1_random"] = fixm.predict(pair_X[te])
            for name, p in preds.items():
                for j, i in enumerate(te):
                    ok = ~np.isnan(Y[i])
                    rows.append(dict(reaction=rx, seed=seed, model=name, pair=b.pairs[i], n=int(ok.sum()),
                                     abs_err=np.abs(p[j] - Y[i, ok]).sum(), signed_err=(p[j] - Y[i, ok]).sum()))
        summary_rows.append(dict(reaction=rx, literature_records=len(y_lit), out_of_range_dropped=dropped,
                                 book_mae=book_err.mean()))
        print(f"{rx}: done in {time.time() - t0:.0f} s", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "literature_model_errors.csv.gz", index=False)
    summ = pd.DataFrame(summary_rows).set_index("reaction")

    verdict = []
    for rx in ARRAYS:
        g = d[d.reaction == rx]
        cube = g.pivot_table(index=["pair", "seed"], columns="model", values=["abs_err", "signed_err", "n"], aggfunc="sum")
        pairs = cube.index.get_level_values(0).unique()
        seeds = sorted(cube.index.get_level_values(1).unique())
        grid = lambda v, m: cube[(v, m)].unstack("seed").reindex(index=pairs, columns=seeds).fillna(0).to_numpy()
        N = grid("n", "literature")
        E = {m: grid("abs_err", m) for m in ("literature", "random_8", "simulated_literature", "literature_plus_1_random")}
        S_lit = grid("signed_err", "literature")

        def stats(ix):
            n = N[ix].sum()
            mae = {m: E[m][ix].sum() / n for m in E}
            delta = mae["literature"] - mae["random_8"]
            return dict(delta=delta, signed=S_lit[ix].sum() / n, mae_lit=mae["literature"], mae_r8=mae["random_8"],
                        delta_sim=mae["simulated_literature"] - mae["random_8"],
                        reduction=1 - (mae["literature_plus_1_random"] - mae["random_8"]) / delta)

        point = stats(np.ix_(range(len(pairs)), range(len(seeds))))
        rng = L.rng_for("litmodel_boot", rx)
        boots = [stats(np.ix_(rng.integers(0, len(pairs), len(pairs)), rng.integers(0, len(seeds), len(seeds)))) for _ in range(N_BOOT)]
        q = lambda k: np.quantile([x[k] for x in boots], [0.025, 0.975])
        dq, sq, rq = q("delta"), q("signed"), q("reduction")
        b1 = dq[0] > 0 and point["delta"] >= REL * point["mae_r8"] and sq[0] > 0
        b2 = point["mae_lit"] >= RATIO * summ.loc[rx, "book_mae"]
        b3 = point["reduction"] > CUT
        verdict.append(dict(reaction=rx, literature_records=int(summ.loc[rx, "literature_records"]),
                            out_of_range_dropped=int(summ.loc[rx, "out_of_range_dropped"]), test_pairs_all_seeds=len(pairs),
                            mae_literature=point["mae_lit"], mae_random_8=point["mae_r8"], delta=point["delta"],
                            delta_lo=dq[0], delta_hi=dq[1], signed_literature=point["signed"], signed_lo=sq[0], signed_hi=sq[1],
                            book_mae=summ.loc[rx, "book_mae"], true_over_book=point["mae_lit"] / summ.loc[rx, "book_mae"],
                            reduction_1_random=point["reduction"], reduction_lo=rq[0], reduction_hi=rq[1],
                            delta_simulated_literature=point["delta_sim"], B1=b1, B2=b2, B3=b3))
    v = pd.DataFrame(verdict)
    v.to_csv(OUT / "literature_model_verdict.csv", index=False)
    overall = dict(B1=int(v.B1.sum()) >= 2, B2=int(v.B2.sum()) >= 2, B3=int(v.B3.sum()) >= 2)
    pd.DataFrame([overall]).to_csv(OUT / "literature_model_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(v.round(4).T.to_string())
    print(overall)


if __name__ == "__main__":
    main()
