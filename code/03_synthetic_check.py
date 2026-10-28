"""Check the pilot pipeline on planted data before it touches real yields.

1. Policy mechanics: budgets, stopping rule, literature reporting, the random slice.
2. Planted interaction: yields built from a known pair x component interaction on the real acylation layout.
   Random data must let the low-rank model beat the additive one (the interaction is learnable), and a policy that
   concentrates on few conditions must show Delta > 0 for both learning models.
3. Null: random_control against its matched random arm must give intervals that include zero.
Exits non-zero on any failure.
"""

import sys

import numpy as np

import pilot_lib as L

FAIL = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def planted(block, seed=0, d=2):
    rng = np.random.default_rng(seed)
    P, C = block.Y.shape
    R = block.cond_x.shape[1]
    h = rng.standard_normal((P, d))
    V = rng.standard_normal((R, d)) * 0.8
    a = rng.normal(-0.8, 0.8, P)
    b = block.cond_x @ rng.normal(0, 0.7, R)
    logit = a[:, None] + b[None, :] + h @ (block.cond_x @ V).T
    Y = 1 / (1 + np.exp(-logit)) + rng.normal(0, 0.03, (P, C))
    return np.where(np.isnan(block.Y), np.nan, np.clip(Y, 0, 1))


def main():
    real = L.load_block("Acylation")
    block = L.Block("synthetic", planted(real), real.pairs, real.conds, real.pair_fp, real.cond_x)

    # 1. mechanics
    Y = block.Y[:60]
    hist = np.nanmean(block.Y[60:], 0)
    for f in (0.0, 0.25, 1.0):
        n_rand = int(round(f * L.K))
        run, rep = L.run_policy("habit", Y, hist, f, L.rng_for("t", f))
        check((run.sum(1) == L.K).all() and (rep == run).all(), f"habit f={f}: 8 runs, all recorded")
        run, rep = L.run_policy("literature", Y, hist, f, L.rng_for("t", f))
        extra = rep & ~run
        check(not extra.any() and (run.sum(1) == L.K).all(), f"literature f={f}: 8 runs, records only run wells")
        pol_part = rep.sum(1) - n_rand
        check(((pol_part == 0) | (pol_part == 1)).all(), f"literature f={f}: at most one policy well recorded per pair")
        rec_vals = np.where(rep, Y, np.nan)
        if f == 0:
            check(np.nanmin(rec_vals) >= L.HIT, "literature f=0: every recorded well reaches the hit line")
        run, rep = L.run_policy("first_hit", Y, hist, 0.0, L.rng_for("t"))
        order = np.argsort(-hist, kind="stable")
        ok = True
        for i in range(len(Y)):
            seq = [j for j in order if run[i, j]]
            ys = Y[i, seq]
            ok &= len(seq) <= L.K and all(v < L.HIT for v in ys[:-1]) and (ys[-1] >= L.HIT or len(seq) == L.K)
        check(ok, "first_hit: runs in history order, stops at the first hit or at 8")
    run, _ = L.run_policy("habit", Y, hist, 1.0, L.rng_for("t1"))
    counts = run.sum(1)
    mr = L.matched_random(Y, counts, L.rng_for("m"))
    check((mr.sum(1) == counts).all(), "matched random reproduces per-pair run counts")

    # 2 and 3. planted interaction and null
    df = L.experiment(block, ["additive", "rf", "lowrank"], policies=("habit", "first_hit", "random_control"),
                      fractions=(0.0,), seeds=range(6))
    s = L.summarise(df, n_boot=500)
    print(s.round(4).to_string(index=False))
    rnd = s[(s.policy == "habit")].set_index("model").mae_random
    check(rnd["lowrank"] < rnd["additive"], f"planted interaction learnable: low-rank {rnd['lowrank']:.4f} < additive {rnd['additive']:.4f}")
    for model in ("rf", "lowrank"):
        row = s[(s.policy == "first_hit") & (s.model == model)].iloc[0]
        check(row.lo > 0, f"first_hit on planted data: Delta > 0 for {model} (lo {row.lo:.4f})")
        row = s[(s.policy == "random_control") & (s.model == model)].iloc[0]
        check(row.lo <= 0 <= row.hi, f"null: random_control interval includes 0 for {model} ({row.lo:.4f}, {row.hi:.4f})")
    print(f"\n{len(FAIL)} failures")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
