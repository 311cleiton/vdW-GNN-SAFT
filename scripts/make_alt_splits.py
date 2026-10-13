#!/usr/bin/env python3
"""
make_alt_splits.py  --  re-partition the frozen dataset into 65/15/20 and 75/15/10
(train/val/test, by ionic liquid) by moving whole ionic liquids between train and test1.

The frozen 70/15/15 split (data/train_val.csv + data/test1.csv) is the result of the curation of
Section 2.1, and that curation is deliberately split-ASYMMETRIC:

  tier 3  physically corrupt   packing vdW*rho/(1000*MW) >= 1; large residuals from the
                               compound's linear rho(T) fit; a source > 5 % from the
                               consensus                                          -> removed everywhere
  tier 2  outside model scope  metal-anion systems; n_frags >= 3; anomalously dense
                               (rho > 1900 kg/m3, the perrhenate series)          -> removed from train/val only
  tier 1  extreme-but-plausible  vdW >= 600 A^3; low-density fatty-acid anions
                               (train/val floor 830.6 kg/m3)                      -> removed from train/val only

So a compound's rows depend on which split it sits in. Moving a test compound that carries tier-1/2
data into train would delete rows (or the whole compound); moving a train compound that lost
tier-1/2 rows into test would need rows the frozen CSVs no longer hold. Only compounds that sit
WELL WITHIN every rule have identical rows under both regimes, so only those are moved -- every
row is carried over byte-for-byte, with only its `split` field rewritten.

Selection: val is untouched. Test sizes are 20 / 10 % of the 1,092 ionic liquids; train takes
the rest. Moves are drawn at random (fixed seed) from the eligible compounds, stratified by cation
class so that every new test set keeps the class mix of the original test1, and the test sets are
nested: test(75/15/10) < test1 < test(65/15/20).

    python scripts/make_alt_splits.py                      # writes data/alt_splits/
    python scripts/make_alt_splits.py --outdir /tmp/splits --seed 42

Outputs, per scheme: <outdir>/split_<train>_<val>_<test>/{train_val.csv,test1.csv}, plus
<outdir>/moves.csv (every moved compound) and <outdir>/eligibility.csv (every compound, its
diagnostics, and why it may or may not move). The frozen data/*.csv files are never modified.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

SCHEMES = [(65, 15, 20), (75, 15, 10)]

# --- Section 2.1 thresholds, and the margins that define "sits well within every rule" ----------
PHI_CONST = 0.602214          # N_A * 1e-24; the packing screen is phi >= PHI_CONST
PHI_MARGIN = 0.57             # ~5 % below the tier-3 packing threshold
VDW_TIER1 = 600.0             # tier 1: vdW >= 600 A^3
VDW_MARGIN = 550.0            # ~8 % below it
RHO_LOW_MARGIN = 850.0        # tier-1 low-density floor is 830.6 kg/m3 in the frozen train/val
RHO_HIGH_MARGIN = 1850.0      # tier-2 anomalous-density line is 1900 kg/m3
RHOT_DEV_MARGIN = 2.0         # % max point deviation from the per-compound rho(T) line (p99 = 1.9 %)
BRANCH_MARGIN = 2.5           # % branch deviation from the pooled consensus; tier-3 threshold is 5 %
AMBIENT_KPA = 200.0           # rho(T) and reference-branch diagnostics use P <= 200 kPa (Section 3.1)

METALS = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Al", "Ga", "In", "Tl", "Sn",
          "Pb", "Bi", "Sb", "Ge", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr",
          "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "La", "Ce", "Hf", "Ta", "W", "Re", "Os",
          "Ir", "Pt", "Au", "Hg", "U"}

# Passes every Section 2.1 screen, but Section 3.4 names it a data-quality candidate (an aromatic
# iodide at packing fraction 0.38, placed 25-55 % denser by all three model families, in every
# run). It is kept out of training rather than moved into it.
KEEP_IN_TEST = {
    "C[N+]1(C)CCN(c2ccccc2)CC1.[I-]": "Section 3.4 data-quality candidate",
}

CATION_CLASSES = [
    ("imidazolium", "[n+]1ccnc1"), ("pyridinium", "[n+]1ccccc1"), ("pyrrolidinium", "[N+]1CCCC1"),
    ("piperidinium", "[N+]1CCCCC1"), ("morpholinium", "[N+]1CCOCC1"), ("phosphonium", "[P+]"),
    ("sulfonium", "[S+]"), ("cyclopropenium", "[c+]1cc1"), ("guanidinium", "[NX3]C(=[N+])[NX3]"),
    ("ammonium", "[N+;X4;!$([N+][O-]);!$([N+]=*)]"),
]
CATION_CLASSES = [(n, Chem.MolFromSmarts(s)) for n, s in CATION_CLASSES]


# ------------------------------------------------------------------------------------------------
# raw-line I/O: rows are carried byte-for-byte; only the trailing `split` field is ever rewritten
# ------------------------------------------------------------------------------------------------
def read_raw(path: str):
    with open(path, "rb") as fh:
        blob = fh.read()
    ends_nl = blob.endswith(b"\n")
    lines = blob.decode("utf-8").split("\n")
    if ends_nl:
        lines = lines[:-1]
    return lines[0], lines[1:], ends_nl


def row_key(line: str):
    f = next(csv.reader([line]))
    return f[10], f[0], float(f[5]), float(f[6])          # split, smiles, T_K, P_kPa


def with_split(line: str, split: str) -> str:
    return line[: line.rfind(",") + 1] + split


def write_raw(path: str, header: str, lines: list[str], ends_nl: bool):
    text = "\n".join([header] + sorted(lines, key=row_key)) + ("\n" if ends_nl else "")
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))


# ------------------------------------------------------------------------------------------------
# per-compound diagnostics (Section 2.1 / 3.1)
# ------------------------------------------------------------------------------------------------
def cation_class(smiles: str) -> str:
    frags = [Chem.MolFromSmiles(s) for s in smiles.split(".")]
    cations = [m for m in frags if Chem.GetFormalCharge(m) > 0]
    for name, patt in CATION_CLASSES:
        if any(m.HasSubstructMatch(patt) for m in cations):
            return name
    return "other"


def ions(smiles: str):
    frags = smiles.split(".")
    q = [Chem.GetFormalCharge(Chem.MolFromSmiles(s)) for s in frags]
    return ({s for s, c in zip(frags, q) if c > 0}, {s for s, c in zip(frags, q) if c < 0})


def diagnose(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(phi=PHI_CONST * df.vdW * df.density_kg_m3 / (1000.0 * df.MW_g_mol))
    recs = []
    for smi, g in df.groupby("smiles", sort=True):
        mol = Chem.MolFromSmiles(smi)
        metals = sorted({a.GetSymbol() for a in mol.GetAtoms()} & METALS)
        amb = g[g.P_kPa <= AMBIENT_KPA]
        slope = rhot_dev = branch = np.nan
        if amb.T_K.nunique() >= 5:
            b, a = np.polyfit(amb.T_K, amb.density_kg_m3, 1)
            fit = a + b * amb.T_K
            slope = b
            rhot_dev = float((abs(amb.density_kg_m3 - fit) / fit).max() * 100)
            if amb.reference.nunique() >= 2:          # pooled consensus (reproduces the 2.2 % of 3.1)
                branch = max(abs(((m.density_kg_m3 - (a + b * m.T_K)) / (a + b * m.T_K)).mean()) * 100
                             for _, m in amb.groupby("reference"))
        # every isobar (source, pressure) with >= 3 temperatures must fall with T, at any pressure
        isobar_ok = all(np.polyfit(s.T_K, s.density_kg_m3, 1)[0] < 0
                        for _, s in g.groupby([g.reference, g.P_kPa.round(0)]) if s.T_K.nunique() >= 3)
        recs.append(dict(
            smiles=smi, name=g.name.iloc[0], split=g.split.iloc[0], cation=cation_class(smi),
            n_rows=len(g), n_refs=g.reference.nunique(), vdW=g.vdW.iloc[0], MW=g.MW_g_mol.iloc[0],
            n_frags=int(g.n_frags.iloc[0]), metals=",".join(metals), rho_min=g.density_kg_m3.min(),
            rho_max=g.density_kg_m3.max(), phi_max=g.phi.max(), rhoT_slope=slope,
            rhoT_maxdev_pct=rhot_dev, branch_dev_pct=branch, isobars_monotone=isobar_ok,
            high_pressure=bool((g.P_kPa > 106.0).any()),
        ))
    c = pd.DataFrame(recs).set_index("smiles")
    # what the frozen train/val still holds is, by construction, inside the tier-1/2 rules
    tv = df[df.split != "test"]
    tv_floor, tv_ceiling = tv.density_kg_m3.min(), tv.density_kg_m3.max()
    tv_metals = set(",".join(c[c.split != "test"].metals).split(",")) - {""}

    def reasons(r) -> str:
        why = []
        if r.n_frags >= 3:
            why.append("tier 2: n_frags >= 3")
        if r.metals:
            tier = "tier 2" if set(r.metals.split(",")) - tv_metals else "margin"
            why.append(f"{tier}: metal-bearing ({r.metals})")
        if r.vdW >= VDW_TIER1:
            why.append("tier 1: vdW >= 600")
        elif r.vdW >= VDW_MARGIN:
            why.append("margin: vdW 550-600")
        if r.rho_min < tv_floor:
            why.append(f"tier 1: rho_min below the train/val floor ({tv_floor:g})")
        elif r.rho_min < RHO_LOW_MARGIN:
            why.append("margin: rho_min < 850")
        if r.rho_max > tv_ceiling:
            why.append(f"tier 2: rho_max above the train/val ceiling ({tv_ceiling:g})")
        elif r.rho_max > RHO_HIGH_MARGIN:
            why.append("margin: rho_max > 1850")
        if r.phi_max >= PHI_MARGIN:
            why.append("margin: packing phi >= 0.57")
        if np.isnan(r.rhoT_slope):
            why.append("rho(T) not assessable (< 5 T at P <= 200 kPa)")
        elif r.rhoT_slope >= 0 or r.rhoT_maxdev_pct > RHOT_DEV_MARGIN:
            why.append("margin: rho(T) deviation > 2 %")
        if not r.isobars_monotone:
            why.append("tier 3: non-monotone isobar")
        if not np.isnan(r.branch_dev_pct) and r.branch_dev_pct > BRANCH_MARGIN:
            why.append("margin: reference branch > 2.5 %")
        return "; ".join(why)

    c["not_movable_because"] = [reasons(r) for r in c.reset_index().itertuples()]
    for smi, why in KEEP_IN_TEST.items():
        if smi in c.index:
            c.loc[smi, "not_movable_because"] = "; ".join(filter(None, [c.loc[smi, "not_movable_because"], why]))
    c["movable"] = c.not_movable_because == ""
    return c


# ------------------------------------------------------------------------------------------------
# class-stratified, nested selection
# ------------------------------------------------------------------------------------------------
def apportion(weights: dict, n: int) -> dict:
    """Largest-remainder apportionment of n over the classes, proportional to weights."""
    tot = sum(weights.values())
    quota = {k: n * w / tot for k, w in weights.items()}
    out = {k: int(np.floor(q)) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - out[k]), k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def fit_counts(target: dict, lo: dict, hi: dict, n: int) -> dict:
    """Class counts closest to target, within [lo, hi] per class, summing to n."""
    got = {k: min(max(target[k], lo[k]), hi[k]) for k in target}
    while sum(got.values()) > n:
        k = max((k for k in got if got[k] > lo[k]), key=lambda k: (got[k] - target[k], k))
        got[k] -= 1
    while sum(got.values()) < n:
        k = max((k for k in got if got[k] < hi[k]), key=lambda k: (target[k] - got[k], k))
        got[k] += 1
    return got


def resize_test(test: set, n: int, c: pd.DataFrame, train_pool: set, ref_mix: dict, rng) -> set:
    """Grow (from train_pool) or shrink (movable members only) `test` to n compounds, steering its
    cation-class mix toward ref_mix."""
    classes = sorted(set(c.cation))
    cls = lambda s: c.loc[s, "cation"]
    have = {k: sum(cls(s) == k for s in test) for k in classes}
    target = apportion({k: ref_mix.get(k, 0) for k in classes}, n)
    if n >= len(test):
        pool = {k: sorted(s for s in train_pool if cls(s) == k and c.loc[s, "movable"]) for k in classes}
        want = fit_counts(target, have, {k: have[k] + len(pool[k]) for k in classes}, n)
        add = [str(s) for k in classes for s in rng.choice(pool[k], want[k] - have[k], replace=False)]
        return test | set(add)
    stuck = {k: sum(cls(s) == k and not c.loc[s, "movable"] for s in test) for k in classes}
    want = fit_counts(target, stuck, have, n)
    drop = []
    for k in classes:
        pool = sorted(s for s in test if cls(s) == k and c.loc[s, "movable"])
        drop += [str(s) for s in rng.choice(pool, have[k] - want[k], replace=False)]
    return test - set(drop)


# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Build 65/15/20 and 75/15/10 splits under the curation rules.")
    ap.add_argument("--outdir", default=os.path.join(DATA, "alt_splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tv_head, tv_lines, tv_nl = read_raw(os.path.join(DATA, "train_val.csv"))
    t1_head, t1_lines, t1_nl = read_raw(os.path.join(DATA, "test1.csv"))
    assert tv_head == t1_head, "train_val.csv and test1.csv must share one header"
    assert sorted(tv_lines, key=row_key) == tv_lines and sorted(t1_lines, key=row_key) == t1_lines, \
        "frozen files are expected in (split, smiles, T_K, P_kPa) order"

    df = pd.concat([pd.read_csv(os.path.join(DATA, f)) for f in ("train_val.csv", "test1.csv")],
                   ignore_index=True)
    c = diagnose(df)
    split_of = c.split.to_dict()
    train0 = {s for s, v in split_of.items() if v == "train"}
    val0 = {s for s, v in split_of.items() if v == "val"}
    test0 = {s for s, v in split_of.items() if v == "test"}
    n_all = len(c)
    ref_mix = c.loc[sorted(test0)].cation.value_counts().to_dict()

    print("=" * 78)
    print("ELIGIBILITY  (moved only if within every Section 2.1 rule, with margin)")
    print("=" * 78)
    for s, grp in (("train", train0), ("val", val0), ("test", test0)):
        mov = int(c.loc[sorted(grp), "movable"].sum())
        print(f"  {s:5s} {len(grp):4d} ILs   movable {mov:4d}   stay put {len(grp) - mov:3d}")

    # nested: larger test sets grow from test1; smaller ones shrink from test1 (in turn, if there are several)
    tests, base = {}, test0
    for scheme in sorted(SCHEMES, key=lambda x: -x[2]):
        n_test = int(round(scheme[2] / 100 * n_all))
        rng = np.random.default_rng([args.seed, scheme[2]])
        if n_test >= len(test0):
            tests[scheme] = resize_test(test0, n_test, c, train0, ref_mix, rng)
        else:
            tests[scheme] = base = resize_test(base, n_test, c, train0, ref_mix, rng)

    os.makedirs(args.outdir, exist_ok=True)
    moves, summary = [], []
    for scheme in SCHEMES:
        test = tests[scheme]
        train = (train0 | test0) - test
        tag = "split_{}_{}_{}".format(*scheme)
        d = os.path.join(args.outdir, tag)
        os.makedirs(d, exist_ok=True)
        new_split = {**{s: "train" for s in train}, **{s: "val" for s in val0}, **{s: "test" for s in test}}
        out_tv, out_t1 = [], []
        for line in tv_lines + t1_lines:
            old_split, smi = row_key(line)[:2]
            assert line.endswith("," + old_split)
            new = with_split(line, new_split[smi])
            (out_t1 if new_split[smi] == "test" else out_tv).append(new)
        write_raw(os.path.join(d, "train_val.csv"), tv_head, out_tv, tv_nl)
        write_raw(os.path.join(d, "test1.csv"), t1_head, out_t1, t1_nl)
        for s in sorted((test - test0) | (test0 - test)):
            moves.append(dict(scheme=tag, smiles=s, name=c.loc[s, "name"], cation=c.loc[s, "cation"],
                              n_rows=int(c.loc[s, "n_rows"]), moved_from=split_of[s],
                              moved_to=new_split[s]))
        summary.append((scheme, tag, train, test))

    pd.DataFrame(moves).to_csv(os.path.join(args.outdir, "moves.csv"), index=False)
    elig = c.reset_index()[["smiles", "name", "split", "cation", "n_rows", "n_refs", "vdW", "MW", "n_frags",
                            "metals", "rho_min", "rho_max", "phi_max", "rhoT_slope", "rhoT_maxdev_pct",
                            "branch_dev_pct", "isobars_monotone", "movable", "not_movable_because"]]
    elig.rename(columns={"split": "frozen_split"}).to_csv(os.path.join(args.outdir, "eligibility.csv"),
                                                           index=False, float_format="%.6g")
    return verify(args.outdir, summary, c, train0, val0, test0, df)


# ------------------------------------------------------------------------------------------------
def verify(outdir, summary, c, train0, val0, test0, frozen) -> int:
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'OK ' if ok else 'FAIL'}] {label:58s} {detail}")
        if not ok:
            fails.append(label)

    strip = lambda line: line[: line.rfind(",")]
    frozen_rows = sorted(strip(l) for l in read_raw(os.path.join(DATA, "train_val.csv"))[1]
                         + read_raw(os.path.join(DATA, "test1.csv"))[1])
    frozen_val = sorted(l for l in read_raw(os.path.join(DATA, "train_val.csv"))[1] if l.endswith(",val"))
    ftv = frozen[frozen.split != "test"]
    tier12_in_test = {s for s in test0 if c.loc[s, "n_frags"] >= 3 or c.loc[s, "vdW"] >= VDW_TIER1
                      or c.loc[s, "rho_min"] < ftv.density_kg_m3.min()
                      or any(m in c.loc[s, "metals"].split(",") for m in ("Fe", "Sb", "Cr"))}
    n_all = len(c)

    for scheme, tag, train, test in summary:
        d = os.path.join(outdir, tag)
        tv = pd.read_csv(os.path.join(d, "train_val.csv"))
        t1 = pd.read_csv(os.path.join(d, "test1.csv"))
        tr, va = tv[tv.split == "train"], tv[tv.split == "val"]
        raw_tv, raw_t1 = read_raw(os.path.join(d, "train_val.csv"))[1], read_raw(os.path.join(d, "test1.csv"))[1]
        print()
        print("=" * 78)
        print(f"{tag}   (target {scheme[0]}/{scheme[1]}/{scheme[2]} by ionic liquid)")
        print("=" * 78)
        check("rows carried byte-for-byte (only `split` rewritten)",
              sorted(strip(l) for l in raw_tv + raw_t1) == frozen_rows, f"{len(raw_tv) + len(raw_t1)} rows")
        check("validation rows identical to the frozen val", sorted(l for l in raw_tv if l.endswith(",val")) == frozen_val)
        check("test1.csv holds only split=test", set(t1.split) == {"test"})
        check("molecule-disjoint train / val / test",
              not (set(tr.smiles) & set(va.smiles) or set(tr.smiles) & set(t1.smiles) or set(va.smiles) & set(t1.smiles)))
        check("test size", t1.smiles.nunique() == len(test), f"{t1.smiles.nunique()} ILs")
        check("all 1,092 ILs / 26,724 rows present",
              tv.smiles.nunique() + t1.smiles.nunique() == n_all and len(tv) + len(t1) == len(frozen))
        check("no train/val compound breaches a tier-1/2 bound",
              (tv.vdW < VDW_TIER1).all() and (tv.n_frags == 2).all()
              and tv.density_kg_m3.between(ftv.density_kg_m3.min(), ftv.density_kg_m3.max()).all()
              and not any(m in ("Fe", "Sb", "Cr") for s in set(tv.smiles) for m in c.loc[s, "metals"].split(",")))
        check("tier-1/2 test compounds retained in test", tier12_in_test <= set(t1.smiles), f"{len(tier12_in_test)} ILs")
        phi = PHI_CONST * pd.concat([tv, t1]).pipe(lambda x: x.vdW * x.density_kg_m3 / (1000 * x.MW_g_mol))
        check("packing screen: no row with phi >= 0.602214", (phi < PHI_CONST).all(), f"max {phi.max():.4f}")
        moved = (set(t1.smiles) - test0) | (test0 - set(t1.smiles))
        check("every moved compound is movable", all(c.loc[s, "movable"] for s in moved), f"{len(moved)} moved")

    tsets = {tag: set(pd.read_csv(os.path.join(outdir, tag, "test1.csv")).smiles) for _, tag, _, _ in summary}
    print()
    check("nested: test(75/15/10) < test1 < test(65/15/20)",
          tsets["split_75_15_10"] <= test0 <= tsets["split_65_15_20"])

    print()
    print("=" * 78)
    print("SUMMARY  (ILs / rows, % of the 1,092 ILs and 26,724 rows)")
    print("=" * 78)
    print(f"  {'scheme':16s} {'train':>21s} {'val':>21s} {'test':>21s}  moved")
    rows_of = frozen.groupby("smiles").size()
    for scheme, tag, train, test in [((70, 15, 15), "frozen", train0, test0)] + summary:
        cells = []
        for grp in (train, val0, test):
            ni, nr = len(grp), int(rows_of[list(grp)].sum())
            cells.append(f"{ni:4d} / {nr:5d} ({100 * ni / n_all:4.1f}%)")
        mv = len((test - test0) | (test0 - test))
        print(f"  {tag:16s} " + "  ".join(cells) + f"  {mv:4d}")
    print()
    if fails:
        print(f"ALT SPLITS FAILED -- {len(fails)} check(s): {', '.join(fails)}")
        return 1
    print(f"ALT SPLITS WRITTEN AND VERIFIED -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
