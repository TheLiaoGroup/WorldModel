"""Exact-pair overlap between the HTE arrays and the literature database (97 CSV parts). Yields are not read.

HTE side: every substrate pair of the 19 two-substrate arrays with at least 24 analysed conditions (raw tables, same
measurement convention and exclusions as the analysis). A literature record matches a pair when one of its products
equals the pair's HTE product and both HTE substrates are among its substrates (RDKit canonical SMILES without
stereochemistry; isomeric agreement is recorded separately). Literature location: WM_LIT_DIR (required).
Outputs: out/literature_overlap_records.csv (one row per matched record: identifiers and conditions, no yield),
out/literature_overlap_pairs.csv (one row per matched HTE pair).
"""

import importlib
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
LIT = Path(os.environ["WM_LIT_DIR"])
MIN_CONDS = 24
COLS = ["substrates", "products", "reagent_txt_solvent_exclusive", "solvent_txt", "temperature_celsius_raw",
        "time_hrs_raw", "number_of_steps", "number_of_stages", "record_id", "reaction_id",
        "source_type", "journal", "year", "patent_number", "source_raw"]
# File-part order of the existing overlap output. Preserve it so deduplication
# replays the same records regardless of multiprocessing completion order.
# This metadata contains file numbers only, never reaction records or outcomes.
MATCHED_PART_ORDER = (
    3, 9, 5, 8, 1, 7, 2, 6, 11, 19, 15, 12, 16, 17, 18, 14, 20, 21, 22, 26,
    25, 28, 29, 30, 35, 23, 36, 33, 32, 37, 38, 39, 40, 42, 41, 53, 57, 58,
    56, 55, 59, 61, 60, 62, 63, 65, 66, 67, 68, 69, 71, 70, 72, 73, 75, 74,
    76, 77, 78, 79, 80, 83, 82, 86, 97, 89, 92, 93, 94, 96, 95,
)
PART_ORDER = MATCHED_PART_ORDER + tuple(i for i in range(1, 98) if i not in MATCHED_PART_ORDER)
PART_NAMES = tuple(f"lit_hq_y_sx_pt_part_{i:03d}.csv" for i in PART_ORDER)


def validate_parts(paths):
    paths = list(paths)
    if len(PART_ORDER) != 97 or set(PART_ORDER) != set(range(1, 98)):
        raise ValueError("literature part-order metadata must cover each part from 001 to 097 exactly once")
    if len(paths) != 97 or {p.name for p in paths} != set(PART_NAMES):
        raise ValueError("expected exactly literature input parts 001 to 097; missing or unexpected parts")
    return paths


def order_records(records):
    """Restore the established file/row order without depending on a private old output."""
    rank = {name: i for i, name in enumerate(PART_NAMES)}
    if not set(records.source_file).issubset(rank):
        raise ValueError("records refer to an unexpected literature input part")
    keys = [(rank[name], int(row)) for name, row in zip(records.source_file, records.source_row)]
    return records.iloc[sorted(range(len(records)), key=keys.__getitem__)].reset_index(drop=True)


def canon(smiles, iso=False):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    return Chem.MolToSmiles(m, isomericSmiles=iso) if m is not None else None


def hte_pairs():
    B = importlib.import_module("01_build_blocks")
    rows = []
    for name, (table, _, _) in B.ARRAYS.items():
        rec = B.load(name, table)
        a = rec[rec.yield_fraction.notna()]
        n = a.groupby("pair").condition_id.nunique()
        keep = a[a.pair.isin(n.index[n >= MIN_CONDS])].drop_duplicates("pair")
        for r in keep.itertuples():
            rows.append(dict(reaction=name, pair=r.pair, product=canon(r.product_smiles),
                             s1=canon(r.sub_1_smiles), s2=canon(r.sub_2_smiles),
                             product_iso=canon(r.product_smiles, True), n_conditions=int(n[r.pair])))
    return pd.DataFrame(rows).dropna(subset=["product", "s1", "s2"])


def scan(args):
    path, products = args
    hits = []
    for chunk in pd.read_csv(path, usecols=COLS, dtype=str, chunksize=20000):
        for idx, r in chunk.iterrows():
            prods = str(r.products).replace(";", ".").split(".")
            found = None
            for p in prods:
                c = canon(p)
                if c in products:
                    found = c
                    break
            if found is None:
                continue
            subs = {canon(s) for s in str(r.substrates).replace(";", ".").split(".")}
            prod_iso = {canon(p, True) for p in prods}
            hits.append(dict(source_file=path.name, source_row=int(idx) + 1, product=found,
                             substrates_canon="|".join(sorted(x for x in subs if x)),
                             products_iso="|".join(sorted(x for x in prod_iso if x)),
                             **{c: r[c] for c in COLS if c not in ("substrates", "products")}))
    return hits


def main():
    pairs = hte_pairs()
    products = set(pairs["product"])
    print(f"HTE pairs with >= {MIN_CONDS} analysed conditions: {len(pairs)}; distinct products {len(products)}", flush=True)
    files = validate_parts(sorted(LIT.glob("lit_hq_y_sx_pt_part_*.csv")))
    with Pool(10) as pool:
        hits = [h for part in pool.imap_unordered(scan, [(f, products) for f in files]) for h in part]
    rec = pd.DataFrame(hits)
    print(f"literature records whose product is an HTE product: {len(rec)}", flush=True)
    m = rec.merge(pairs, on="product")
    both = m.apply(lambda r: r.s1 in r.substrates_canon.split("|") and r.s2 in r.substrates_canon.split("|"), axis=1)
    m = order_records(m[both].copy())
    m["isomeric_product_match"] = m.apply(lambda r: r.product_iso in r.products_iso.split("|"), axis=1)
    m.drop(columns=["products_iso"]).to_csv(OUT / "literature_overlap_records.csv", index=False)
    g = m.groupby(["reaction", "pair"])
    summary = pd.DataFrame({"records": g.size(), "sources": g.source_raw.nunique(),
                            "articles": g.apply(lambda d: (d.source_type == "article").sum()),
                            "n_conditions": g.n_conditions.first()}).reset_index()
    summary.to_csv(OUT / "literature_overlap_pairs.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.groupby("reaction").agg(pairs=("pair", "size"), records=("records", "sum"),
                                          sources=("sources", "sum")).to_string())
    print(f"total: {len(summary)} HTE pairs reported in the literature, {int(summary.records.sum())} records")


if __name__ == "__main__":
    main()
