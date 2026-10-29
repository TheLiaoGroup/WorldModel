"""Policies, models and scoring for the counterfactual pilot.

A block is a pair x condition yield matrix Y (NaN = not analysed). Substrate pairs are split in half: the history half
only ranks conditions; the logged half is sampled by a policy, and models trained on what the policy recorded are
scored on the wells that neither the policy nor its matched random comparator ran.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
K = 8              # experiments per substrate pair
HIT = 0.20         # yield that counts as a usable result
FRACTIONS = (0.0, 0.10, 0.25, 0.50, 1.0)
POLICIES = ("habit", "first_hit", "literature", "random_control")
SEEDS = range(10)
SKIPPED = []       # (reaction, policy, f, seed, fold) whose policy recorded no well; both arms are left out


@dataclass
class Block:
    name: str
    Y: np.ndarray            # pairs x conditions
    pairs: list
    conds: list
    pair_fp: np.ndarray      # pairs x bits (ECFP4 of both substrates)
    cond_x: np.ndarray       # conditions x components (reagents and solvent present)


def features(recipes, subs):
    """Pair fingerprints (ECFP4, 1,024 bits per substrate) and condition-component indicators.

    recipes: one row per condition (index = condition id, in matrix column order); subs: one row per pair (matrix row order).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    comp_cols = [f"reagent{i}_id" for i in range(1, 7)] + ["solvent_id"]
    comps = sorted({v for col in comp_cols for v in recipes[col].dropna()})
    cond_x = np.array([[float(x in set(recipes.loc[cid, comp_cols].dropna())) for x in comps] for cid in recipes.index])
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = lambda s: gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32)
    pair_fp = np.array([np.concatenate([fp(a), fp(b)]) for a, b in zip(subs.sub_1_smiles, subs.sub_2_smiles)])
    return pair_fp, cond_x


def load_block(name: str) -> Block:
    w = pd.read_csv(OUT / "wells.csv.gz")
    w = w[w.reaction == name]
    c = pd.read_csv(OUT / "conditions.csv")
    c = c[c.reaction == name].set_index("condition_id")
    m = w.pivot(index="pair", columns="condition_id", values="yield_fraction").reindex(columns=c.index)
    pair_fp, cond_x = features(c, w.drop_duplicates("pair").set_index("pair").loc[m.index])
    return Block(name, m.to_numpy(float), list(m.index), list(c.index), pair_fp, cond_x)


HABIT_FLOOR = 0.01  # added to historical condition means so that never-productive conditions keep a small chance


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


# ---------------------------------------------------------------- policies
def run_policy(policy, Y, hist_mean, f, rng, return_random=False):
    """Return (run, report) boolean masks over the logged pairs' matrix Y (and the random-slice mask if asked)."""
    P, C = Y.shape
    run = np.zeros((P, C), bool)
    report = np.zeros((P, C), bool)
    random_slice = np.zeros((P, C), bool)
    order = np.argsort(-hist_mean, kind="stable")
    weight = hist_mean + HABIT_FLOOR
    n_rand = int(round(f * K))
    for i in range(P):
        avail = np.flatnonzero(~np.isnan(Y[i]))
        rand = rng.choice(avail, size=min(n_rand, len(avail)), replace=False)
        run[i, rand] = report[i, rand] = random_slice[i, rand] = True   # the random slice is always recorded in full
        rest = np.setdiff1d(avail, rand)
        budget = K - len(rand)
        if budget <= 0 or len(rest) == 0:
            continue
        if policy == "random_control":
            pick = rng.choice(rest, size=min(budget, len(rest)), replace=False)
            run[i, pick] = report[i, pick] = True
        elif policy in ("habit", "literature"):
            p = weight[rest] / weight[rest].sum()
            pick = rng.choice(rest, size=min(budget, len(rest)), replace=False, p=p)
            run[i, pick] = True
            if policy == "habit":
                report[i, pick] = True
            else:
                best = pick[np.argmax(Y[i, pick])]
                if Y[i, best] >= HIT:
                    report[i, best] = True
        elif policy == "first_hit":
            for j in [j for j in order if j in set(rest)][:budget]:
                run[i, j] = report[i, j] = True
                if Y[i, j] >= HIT:
                    break
        else:
            raise ValueError(policy)
    return (run, report, random_slice) if return_random else (run, report)


def matched_random(Y, counts, rng):
    run = np.zeros(Y.shape, bool)
    for i, n in enumerate(counts):
        avail = np.flatnonzero(~np.isnan(Y[i]))
        run[i, rng.choice(avail, size=min(n, len(avail)), replace=False)] = True
    return run


# ---------------------------------------------------------------- models
def fit_predict(model, block, pair_idx, report, seed):
    """Fit on the recorded wells of the logged pairs (rows pair_idx) and predict their full matrix."""
    Y = block.Y[pair_idx]
    pi, ci = np.nonzero(report)
    y = Y[pi, ci]
    P, C = Y.shape
    if model == "additive":
        return additive(pi, ci, y, P, C)
    if model == "rf":
        X = np.hstack([block.pair_fp[pair_idx][pi], block.cond_x[ci]])
        rf = RandomForestRegressor(n_estimators=200, max_features=0.33, n_jobs=-1, random_state=seed).fit(X, y)
        Xa = np.hstack([np.repeat(block.pair_fp[pair_idx], C, 0), np.tile(block.cond_x, (P, 1))])
        return rf.predict(Xa).reshape(P, C)
    if model == "lowrank":
        return lowrank(pi, ci, y, P, block.cond_x, seed)
    if model == "gbm":   # robustness check; hyperparameters fixed before reading results
        from lightgbm import LGBMRegressor
        if len(y) < 2:   # LightGBM needs two rows; a one-row training set predicts its value, as a tree ensemble does
            return np.full((P, C), float(y.mean()))
        X = np.hstack([block.pair_fp[pair_idx][pi], block.cond_x[ci]])
        g = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.33, random_state=seed, verbose=-1).fit(X, y)
        Xa = np.hstack([np.repeat(block.pair_fp[pair_idx], C, 0), np.tile(block.cond_x, (P, 1))])
        return g.predict(Xa).reshape(P, C)
    if model == "mlp":   # neural check; architecture and training fixed before results were read
        return mlp(np.hstack([block.pair_fp[pair_idx][pi], block.cond_x[ci]]), y,
                   np.hstack([np.repeat(block.pair_fp[pair_idx], C, 0), np.tile(block.cond_x, (P, 1))]), seed).reshape(P, C)
    raise ValueError(model)


def mlp(X, y, Xa, seed, epochs=300):
    import torch
    torch.manual_seed(seed)
    X, y, Xa = (torch.tensor(a, dtype=torch.float32) for a in (X, y, Xa))
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(64, 1), torch.nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(X).squeeze(1), y).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xa).squeeze(1).numpy().astype(float)


def additive(pi, ci, y, P, C, lam=1.0, iters=20):
    mu, bp, bc = y.mean(), np.zeros(P), np.zeros(C)
    for _ in range(iters):
        r = y - mu - bc[ci]
        bp = np.bincount(pi, r, P) / (np.bincount(pi, minlength=P) + lam)
        r = y - mu - bp[pi]
        bc = np.bincount(ci, r, C) / (np.bincount(ci, minlength=C) + lam)
    return mu + bp[:, None] + bc[None, :]


def lowrank(pi, ci, y, P, cond_x, seed, d=4, lam=1e-4):
    """y = mu + b_pair + w.x_c + u_pair . (V' x_c); a condition is the sum of its components."""
    R = cond_x.shape[1]
    X = cond_x[ci]
    n = len(y)
    rng = np.random.default_rng(seed)
    sizes = [1, P, R, P * d, R * d]
    cuts = np.cumsum(sizes)[:-1]
    theta0 = np.concatenate([[y.mean()], np.zeros(P + R), 0.01 * rng.standard_normal(P * d + R * d)])

    def unpack(t):
        mu, bp, w, U, V = np.split(t, cuts)
        return mu[0], bp, w, U.reshape(P, d), V.reshape(R, d)

    def loss(t):
        mu, bp, w, U, V = unpack(t)
        Z = X @ V
        r = mu + bp[pi] + X @ w + np.sum(U[pi] * Z, 1) - y
        L = np.mean(r ** 2) + lam * (bp @ bp + w @ w + np.sum(U ** 2) + np.sum(V ** 2))
        g_mu = 2 * r.mean()
        g_bp = 2 * np.bincount(pi, r, P) / n + 2 * lam * bp
        g_w = 2 * X.T @ r / n + 2 * lam * w
        g_U = np.zeros((P, d))
        np.add.at(g_U, pi, 2 * r[:, None] * Z / n)
        g_U += 2 * lam * U
        g_V = 2 * X.T @ (r[:, None] * U[pi]) / n + 2 * lam * V
        return L, np.concatenate([[g_mu], g_bp, g_w, g_U.ravel(), g_V.ravel()])

    t = minimize(loss, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    mu, bp, w, U, V = unpack(t)
    return mu + bp[:, None] + (cond_x @ w)[None, :] + U @ (cond_x @ V).T


# ---------------------------------------------------------------- one experiment
def experiment(block, models, policies=POLICIES, fractions=FRACTIONS, seeds=SEEDS):
    """Per-pair error sums for every (seed, fold, policy, fraction, model, arm)."""
    rows = []
    P = len(block.pairs)
    for seed in seeds:
        perm = rng_for(block.name, "split", seed).permutation(P)
        halves = (perm[: P // 2], perm[P // 2:])
        for fold in (0, 1):
            hist, logged = halves[fold], halves[1 - fold]
            Y = block.Y[logged]
            hist_mean = np.nan_to_num(np.nanmean(block.Y[hist], 0), nan=0.0)
            cache = {}

            def predictions(model, report, key):
                k = (model, key)
                if k not in cache:
                    cache[k] = fit_predict(model, block, logged, report, seed)
                return cache[k]

            for policy in policies:
                for f in fractions:
                    run_p, rep_p = run_policy(policy, Y, hist_mean, f, rng_for(block.name, policy, f, seed, fold))
                    if not rep_p.any():   # nothing recorded (no pair reached the hit line): no model to score
                        SKIPPED.append((block.name, policy, f, seed, fold))
                        continue
                    counts = run_p.sum(1)
                    run_r = matched_random(Y, counts, rng_for(block.name, "matched", tuple(counts), seed, fold))
                    evalm = ~np.isnan(Y) & ~run_p & ~run_r
                    for model in models:
                        for arm, rep, key in (("policy", rep_p, rep_p.tobytes()), ("random", run_r, run_r.tobytes())):
                            pred = predictions(model, rep, key)
                            err = np.where(evalm, np.abs(pred - np.nan_to_num(Y)), 0.0)
                            masked = np.where(evalm, pred, -np.inf)
                            top = np.argmax(masked, 1)
                            has = evalm.any(1)
                            for i in np.flatnonzero(has):
                                rows.append((block.name, seed, fold, policy, f, model, arm, block.pairs[logged[i]],
                                             err[i].sum(), int(evalm[i].sum()), int(Y[i, top[i]] >= HIT),
                                             int(np.nanmax(np.where(evalm[i], Y[i], np.nan)) >= HIT),
                                             int(rep[i].sum())))
    return pd.DataFrame(rows, columns=["reaction", "seed", "fold", "policy", "f", "model", "arm", "pair",
                                       "abs_err_sum", "n_eval", "top_hit", "any_hit", "n_train_wells"])


# ---------------------------------------------------------------- summary
def summarise(df, n_boot=1000):
    """Delta = MAE(policy) - MAE(random) on identical wells.

    Interval: resample seeds (each seed refits every model, so it carries the fit-to-fit variation), then whole
    substrate pairs within them. Resampling pairs alone ignored the fit-to-fit variation and failed the null control.
    """
    out = []
    for key, g in df.groupby(["reaction", "policy", "f", "model"]):
        piv = g.pivot_table(index=["pair", "seed", "fold"], columns="arm", values=["abs_err_sum", "n_eval", "top_hit"])
        cube = piv.groupby(level=["pair", "seed"]).sum()
        grid = lambda col: cube[col].unstack("seed").fillna(0).to_numpy()   # pairs x seeds
        E_p, E_r, N = grid(("abs_err_sum", "policy")), grid(("abs_err_sum", "random")), grid(("n_eval", "policy"))
        e_p, e_r, n = E_p.sum(1), E_r.sum(1), N.sum(1)
        rng = rng_for("boot", key)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            sd = rng.integers(0, N.shape[1], N.shape[1])
            pr = rng.integers(0, N.shape[0], N.shape[0])
            boot[b] = (E_p[np.ix_(pr, sd)].sum() - E_r[np.ix_(pr, sd)].sum()) / N[np.ix_(pr, sd)].sum()
        cnt = piv[("n_eval", "policy")]
        out.append(dict(zip(["reaction", "policy", "f", "model"], key),
                        mae_policy=e_p.sum() / n.sum(), mae_random=e_r.sum() / n.sum(),
                        delta=(e_p.sum() - e_r.sum()) / n.sum(),
                        lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975),
                        top_hit_policy=piv[("top_hit", "policy")].mean(), top_hit_random=piv[("top_hit", "random")].mean(),
                        pairs=len(n), eval_wells=int(cnt.sum())))
    s = pd.DataFrame(out)
    s["rel"] = s.delta / s.mae_random
    return s
