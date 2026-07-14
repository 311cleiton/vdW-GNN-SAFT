#!/usr/bin/env python3
"""
verify_dataset.py  --  independent re-derivation of every dataset number in the paper.

This script does not trust the manuscript: it recomputes each quantity from the frozen CSVs
and asserts it. If any assertion fires, the shipped data and the paper disagree and one of
them is wrong. It is the dataset half of the framework's self-check contract, and it runs in
CI on every push.

It needs only pandas + numpy + rdkit -- no torch, no FeOs -- so it is fast and always runnable.

    python scripts/verify_dataset.py

Notes on definitions (these are exactly the traps that make "obvious" numbers disagree):
  * vdW median/mean are PER MOLECULE (n = 1,092), not per row (n = 26,724).
  * The vdW standardization buffers are computed per MODELING UNIT (smiles, vdW) over the
    TRAIN split only (n = 763), with the SAMPLE standard deviation (ddof = 1), because that is
    what torch.Tensor.std() does and what train.compute_vdw_stats() stores in the checkpoint.
  * test2.csv is a small debugging set. It is EXCLUDED from every number reported in the paper.
    The frozen dataset is train_val.csv + test1.csv.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

FAILURES: list[str] = []


def check(label: str, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:52s} {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, expected {want!r}")
    return ok


def main() -> int:
    tv = pd.read_csv(os.path.join(DATA, "train_val.csv"))
    t1 = pd.read_csv(os.path.join(DATA, "test1.csv"))
    t2 = pd.read_csv(os.path.join(DATA, "test2.csv"))

    tr, va = tv[tv.split == "train"], tv[tv.split == "val"]
    frozen = pd.concat([tv, t1], ignore_index=True)          # test2 deliberately excluded
    per_mol = frozen.drop_duplicates("smiles")

    print("=" * 78)
    print("1. SPLIT SIZES  (Table 2 / Section 3.1)")
    print("=" * 78)
    check("train_val rows", len(tv), 21556)
    check("train rows", len(tr), 18259)
    check("val rows", len(va), 3297)
    check("test1 rows", len(t1), 5168)
    check("train_val ionic liquids", tv.smiles.nunique(), 922)
    check("test1 ionic liquids", t1.smiles.nunique(), 170)
    check("frozen rows (train_val + test1)", len(frozen), 26724)
    check("frozen ionic liquids", frozen.smiles.nunique(), 1092)
    check("test2 rows (debug set, excluded)", len(t2), 439)
    check("test2 ionic liquids (excluded)", t2.smiles.nunique(), 30)

    print()
    print("=" * 78)
    print("2. MOLECULE-DISJOINT SPLITS  (no leakage)")
    print("=" * 78)
    s_tr, s_va, s_t1, s_t2 = (set(x.smiles) for x in (tr, va, t1, t2))
    check("train ∩ val", len(s_tr & s_va), 0)
    check("train ∩ test1", len(s_tr & s_t1), 0)
    check("val ∩ test1", len(s_va & s_t1), 0)
    check("test2 ∩ frozen", len(s_t2 & (s_tr | s_va | s_t1)), 0)

    print()
    print("=" * 78)
    print("3. vdW DESCRIPTOR  (per molecule, n = 1,092)")
    print("=" * 78)
    v = per_mol.vdW
    check("vdW min  [A^3]", round(float(v.min()), 2), 69.22, tol=5e-3)
    check("vdW median [A^3]", round(float(v.median()), 2), 209.62, tol=5e-3)
    check("vdW mean [A^3]", round(float(v.mean()), 2), 233.49, tol=5e-3)
    check("vdW max  [A^3]", round(float(v.max()), 2), 1114.61, tol=5e-3)

    print()
    print("=" * 78)
    print("4. STATE RANGES  (applicability domain)")
    print("=" * 78)
    check("T min [K]", round(float(frozen.T_K.min()), 2), 250.0, tol=1e-6)
    check("T max [K]", round(float(frozen.T_K.max()), 2), 443.46, tol=1e-6)
    check("P min [kPa]", round(float(frozen.P_kPa.min()), 1), 95.0, tol=1e-6)
    check("P max [kPa]", round(float(frozen.P_kPa.max()), 1), 10001.0, tol=1e-6)
    check("rows above 106 kPa", int((frozen.P_kPa > 106.0).sum()), 4945)
    print("         -> build_dataset.py --p-max 10001.0 reproduces this exactly (bound inclusive).")

    print()
    print("=" * 78)
    print("5. PACKING SCREEN  phi = 0.602214 * vdW * rho / (1000 * MW)")
    print("=" * 78)
    phi = 0.602214 * frozen.vdW * frozen.density_kg_m3 / (1000.0 * frozen.MW_g_mol)
    check("max retained phi (< 0.6022)", round(float(phi.max()), 4), 0.5983, tol=5e-4)
    check("rows failing the screen", int((phi >= 0.602214).sum()), 0)

    print()
    print("=" * 78)
    print("6. FRAGMENTS  (Section 3.8: the two dications)")
    print("=" * 78)
    vc = per_mol.n_frags.value_counts()
    check("ILs with 2 fragments", int(vc.get(2, 0)), 1090)
    check("ILs with 3 fragments (dications)", int(vc.get(3, 0)), 2)
    check("both dications live in test1",
          int(per_mol[per_mol.n_frags == 3].smiles.isin(s_t1).sum()), 2)

    print()
    print("=" * 78)
    print("7. SMILES CANONICALITY")
    print("=" * 78)
    bad = [s for s in per_mol.smiles if Chem.MolToSmiles(Chem.MolFromSmiles(s)) != s]
    check("non-canonical SMILES", len(bad), 0)

    print()
    print("=" * 78)
    print("8. MODELING UNITS + vdW STANDARDIZATION BUFFERS")
    print("   (per (smiles, vdW) unit; TRAIN split; SAMPLE std, ddof = 1)")
    print("=" * 78)
    u_tr = tr.drop_duplicates(subset=["smiles", "vdW"])
    u_va = va.drop_duplicates(subset=["smiles", "vdW"])
    check("train modeling units", len(u_tr), 763)
    check("val modeling units", len(u_va), 159)
    check("vdw_mean [A^3]", float(u_tr.vdW.mean()), 231.22710353866, tol=1e-9)
    check("vdw_std  [A^3]", float(u_tr.vdW.std(ddof=1)), 91.46474411564, tol=1e-9)

    print()
    print("=" * 78)
    if FAILURES:
        print(f"DATASET VERIFICATION FAILED -- {len(FAILURES)} mismatch(es):")
        for f in FAILURES:
            print("  -", f)
        print("=" * 78)
        return 1
    print("DATASET VERIFICATION PASSED -- all 30 checks reproduce the manuscript.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
