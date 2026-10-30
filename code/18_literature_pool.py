"""Literature training pool for the literature-trained models: single-step amide couplings, Suzuki couplings and C-N arylations in the
literature database, identified by reaction templates whose product must equal a recorded product.

Yield values are not read; only whether a yield is present. Literature location: WM_LIT_DIR (required).
Output: out/literature_pool.csv.gz (one row per classified record: class, canonical substrates, product, reagent and
solvent text, temperature, time, source, identifiers, has_yield).
"""

import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
LIT = Path(os.environ["WM_LIT_DIR"])
COLS = ["substrates", "products", "reagent_txt_solvent_exclusive", "reagents_smiles", "solvent_txt",
        "temperature_celsius_raw", "time_hrs_raw", "number_of_steps", "record_id",
        "source_type", "journal", "year", "patent_number", "source_raw", "yield_reg"]
# File-part order of the existing analysis pool. Fixed-seed sampling depends on
# row order, so retain it independently of worker completion order. These are
# file numbers only; no reaction records, selected identities or yields are stored.
PART_ORDER = (
    3, 4, 5, 1, 2, 6, 9, 8, 7, 10, 11, 12, 14, 15, 16, 17, 19, 18, 20, 13,
    22, 21, 25, 24, 26, 27, 29, 28, 31, 23, 30, 35, 34, 33, 38, 36, 32, 37,
    39, 40, 42, 46, 45, 41, 43, 44, 48, 47, 49, 53, 50, 51, 52, 58, 57, 54,
    56, 59, 55, 61, 62, 60, 63, 65, 67, 64, 68, 66, 70, 71, 69, 72, 73, 76,
    74, 77, 75, 83, 81, 80, 82, 79, 84, 86, 78, 87, 85, 97, 90, 88, 89, 91,
    93, 92, 96, 94, 95,
)
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
TEMPLATES = {
    "amide": ("[C:1](=[O:2])[OX2H1].[N;H2,H1;!$(N-C=O);!$(N-S(=O)=O):3]>>[C:1](=[O:2])[N:3]", "acid", "amine"),
    "suzuki": ("[c:1][Br,I,Cl].[c:2][BX3]([OX2])[OX2]>>[c:1]-[c:2]", "aryl_halide", "boron"),
    "cn_arylation": ("[c:1][Br,I,Cl].[N;H2,H1;!$(N-C=O);!$(N-S(=O)=O):2]>>[c:1]-[N:2]", "aryl_halide", "amine"),
}
RXN = {k: AllChem.ReactionFromSmarts(v[0]) for k, v in TEMPLATES.items()}


def canon(m):
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def classify(subs, prods):
    mols = [Chem.MolFromSmiles(s) for s in subs]
    if any(m is None for m in mols):
        return None
    targets = {canon(Chem.MolFromSmiles(p)) for p in prods} - {None}
    for cls, rxn in RXN.items():
        for i in range(len(mols)):
            for j in range(len(mols)):
                if i == j:
                    continue
                try:
                    outs = rxn.RunReactants((mols[i], mols[j]))
                except Exception:
                    continue
                for o in outs:
                    m = o[0]
                    try:
                        Chem.SanitizeMol(m)
                    except Exception:
                        continue
                    if canon(m) in targets:
                        return cls, canon(mols[i]), canon(mols[j]), canon(m)
    return None


def scan(path):
    rows = []
    for chunk in pd.read_csv(path, usecols=COLS, dtype=str, chunksize=20000):
        chunk = chunk[chunk.number_of_steps == "1.0"]
        for idx, r in chunk.iterrows():
            subs = [s for s in str(r.substrates).replace(";", ".").split(".") if s]
            if not 2 <= len(subs) <= 3:
                continue
            prods = [p for p in str(r.products).replace(";", ".").split(".") if p]
            hit = classify(subs, prods)
            if hit is None:
                continue
            cls, a, b, p = hit
            rows.append(dict(reaction_class=cls, substrate_a=a, substrate_b=b, product=p,
                             source_file=path.name, source_row=int(idx) + 1, has_yield=pd.notna(r.yield_reg),
                             **{c: r[c] for c in COLS if c not in ("substrates", "products", "yield_reg", "number_of_steps")}))
    return rows


def main():
    files = validate_parts(sorted(LIT.glob("lit_hq_y_sx_pt_part_*.csv")))
    with Pool(10) as pool:
        rows = [r for part in pool.imap_unordered(scan, files) for r in part]
    d = order_records(pd.DataFrame(rows))
    d.to_csv(OUT / "literature_pool.csv.gz", index=False)
    print(d.groupby("reaction_class").agg(records=("product", "size"), with_yield=("has_yield", "sum"),
                                          products=("product", "nunique")).to_string())


if __name__ == "__main__":
    main()
