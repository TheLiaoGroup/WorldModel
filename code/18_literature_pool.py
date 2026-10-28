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
        "temperature_celsius_raw", "time_hrs_raw", "number_of_steps", "record_id", "true_reaxys_id",
        "source_type", "journal", "year", "patent_number", "source_raw", "yield_reg"]
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
    files = sorted(LIT.glob("lit_hq_y_sx_pt_part_*.csv"))
    if len(files) != 97:
        raise SystemExit(f"expected 97 literature parts, found {len(files)}")
    with Pool(10) as pool:
        rows = [r for part in pool.imap_unordered(scan, files) for r in part]
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "literature_pool.csv.gz", index=False)
    print(d.groupby("reaction_class").agg(records=("product", "size"), with_yield=("has_yield", "sum"),
                                          products=("product", "nunique")).to_string())


if __name__ == "__main__":
    main()
