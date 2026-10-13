#!/usr/bin/env python3
"""
make_val_test_splits.py  --  splits that move whole ionic liquids between VALIDATION and test, leaving the
training split exactly as it was:

  70/10/20 = the frozen 70/15/15 split (data/)                     with 50 val ILs moved to test
  75/10/15 = data/alt_splits/split_75_15_10 (make_alt_splits.py)   with the same 50 val ILs moved to test
  65/10/25 = data/alt_splits/split_65_15_20 (make_alt_splits.py)   with the same 50 val ILs moved to test
  70/20/10 = the frozen 70/15/15 split                             with 59 test ILs moved to val
  75/20/5  = data/alt_splits/split_75_15_10                        with the same 59 test ILs moved to val
  65/20/15 = data/alt_splits/split_65_15_20                        with the same 59 test ILs moved to val

Section 2.1 curates validation like training (tier-1/2 data removed) and test differently (kept), so
the rule of make_alt_splits.py applies in both directions: only compounds that sit well within every
rule -- whose rows are identical under both regimes -- may move (its `movable` flag, same thresholds
and margins). Validation becomes 10 % or 20 % of the 1,092 ILs (109 or 218) and test takes the rest.
Moves are drawn at random (fixed seed) within each cation class, so validation keeps its original
class mix. Each group of three moves one shared set -- test -> val moves are drawn from test(75/15/10),
which lies inside test1 and test(65/15/20) -- so a group shares one validation set and its test sets
are nested.

    python scripts/make_val_test_splits.py      # writes the six split_* folders into data/alt_splits

Rows are carried byte-for-byte; only the `split` field is rewritten. Nothing else in data/ changes.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_alt_splits as mas   # noqa: E402  shared raw I/O, Section 2.1 diagnostics, apportionment

# new split -> (the split it is built from, validation % of the ILs); None = the frozen data/ files
SCHEMES = {"split_70_10_20": (None, 10), "split_75_10_15": ("split_75_15_10", 10),
           "split_70_20_10": (None, 20), "split_75_20_5": ("split_75_15_10", 20),
           "split_65_10_25": ("split_65_15_20", 10), "split_65_20_15": ("split_65_15_20", 20)}
MOVES_FILE = {10: "moves_val_to_test.csv", 20: "moves_test_to_val.csv"}


def split_sets(tv: pd.DataFrame, t1: pd.DataFrame):
    return (set(tv[tv.split == "train"].smiles), set(tv[tv.split == "val"].smiles), set(t1.smiles))


def parent_dir(outdir: str, parent: str | None) -> str:
    return mas.DATA if parent is None else os.path.join(outdir, parent)


def parent_sets(outdir: str, parent: str | None):
    d = parent_dir(outdir, parent)
    return split_sets(pd.read_csv(os.path.join(d, "train_val.csv")), pd.read_csv(os.path.join(d, "test1.csv")))


def pick_moves(c: pd.DataFrame, val0: list, pool: set, n_val: int, rng):
    """ILs to move so that validation holds n_val, keeping its cation-class mix.

    Shrinking moves movable val ILs out (non-movable ones must stay); growing moves movable ILs in
    from `pool`, the test ILs present in every parent. Returns (moved, class counts before, after)."""
    classes = sorted(set(c.cation))
    have = {k: sum(c.loc[s, "cation"] == k for s in val0) for k in classes}
    target = mas.apportion(have, n_val)
    moved = set()
    if n_val < len(val0):
        stuck = {k: sum(c.loc[s, "cation"] == k and not c.loc[s, "movable"] for s in val0) for k in classes}
        after = mas.fit_counts(target, stuck, have, n_val)
        for k in classes:
            src = sorted(s for s in val0 if c.loc[s, "cation"] == k and c.loc[s, "movable"])
            moved |= {str(s) for s in rng.choice(src, have[k] - after[k], replace=False)}
    else:
        avail = {k: sorted(s for s in pool if c.loc[s, "cation"] == k and c.loc[s, "movable"]) for k in classes}
        after = mas.fit_counts(target, have, {k: have[k] + len(avail[k]) for k in classes}, n_val)
        for k in classes:
            moved |= {str(s) for s in rng.choice(avail[k], after[k] - have[k], replace=False)}
    return moved, have, after


def main() -> int:
    ap = argparse.ArgumentParser(description="Build 70/10/20, 75/10/15, 65/10/25, 70/20/10, 75/20/5 and "
                                             "65/20/15 splits by moving ILs between validation and test.")
    ap.add_argument("--outdir", default=os.path.join(mas.DATA, "alt_splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tv_head, tv_lines, tv_nl = mas.read_raw(os.path.join(mas.DATA, "train_val.csv"))
    t1_head, t1_lines, t1_nl = mas.read_raw(os.path.join(mas.DATA, "test1.csv"))
    df = pd.concat([pd.read_csv(os.path.join(mas.DATA, f)) for f in ("train_val.csv", "test1.csv")],
                   ignore_index=True)
    c = mas.diagnose(df)
    n_all = len(c)
    val0 = sorted(s for s, v in c.split.items() if v == "val")

    moved = {}
    for pct in sorted({pct for _, pct in SCHEMES.values()}):
        n_val = int(round(pct / 100 * n_all))
        pool = set.intersection(*[parent_sets(args.outdir, p)[2] for p, v in SCHEMES.values() if v == pct])
        moved[pct], have, after = pick_moves(c, val0, pool, n_val, np.random.default_rng([args.seed, pct]))
        grow = n_val > len(val0)
        print("=" * 78)
        print(f"{'TEST -> VALIDATION' if grow else 'VALIDATION -> TEST'}  (val {len(val0)} -> {n_val} ILs; "
              f"moving {len(moved[pct])}, all movable, from "
              f"{'the ' + str(len(pool)) + ' test ILs every parent shares' if grow else 'val'})")
        print("=" * 78)
        for k in sorted(have):
            if have[k] or after[k]:
                print(f"  {k:14s} val {have[k]:3d} -> {after[k]:3d}   ({100 * have[k] / len(val0):4.1f}% -> "
                      f"{100 * after[k] / n_val:4.1f}%)   moved {abs(after[k] - have[k]):2d}")

    moves, built = {pct: [] for pct in moved}, []
    for tag, (parent, pct) in SCHEMES.items():
        p_train, p_val, p_test = parent_sets(args.outdir, parent)
        assert p_val == set(val0), f"{parent}: expected the frozen validation set"
        mv = moved[pct]
        grow = pct / 100 * n_all > len(p_val)
        val, test = (p_val | mv, p_test - mv) if grow else (p_val - mv, p_test | mv)
        new_split = {**{s: "train" for s in p_train}, **{s: "val" for s in val}, **{s: "test" for s in test}}
        out_tv, out_t1 = [], []
        for line in tv_lines + t1_lines:
            smi = mas.row_key(line)[1]
            new = mas.with_split(line, new_split[smi])
            (out_t1 if new_split[smi] == "test" else out_tv).append(new)
        d = os.path.join(args.outdir, tag)
        os.makedirs(d, exist_ok=True)
        mas.write_raw(os.path.join(d, "train_val.csv"), tv_head, out_tv, tv_nl)
        mas.write_raw(os.path.join(d, "test1.csv"), t1_head, out_t1, t1_nl)
        for s in sorted(mv):
            moves[pct].append(dict(scheme=tag, parent=parent or "frozen", smiles=s, name=c.loc[s, "name"],
                                   cation=c.loc[s, "cation"], n_rows=int(c.loc[s, "n_rows"]),
                                   moved_from="test" if grow else "val", moved_to="val" if grow else "test"))
        built.append((tag, parent, grow, p_val, p_test, mv))

    for pct, rows in moves.items():
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, MOVES_FILE[pct]), index=False)
    return verify(args.outdir, built, c, df, set(val0))


def verify(outdir, built, c, frozen, val0) -> int:
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'OK ' if ok else 'FAIL'}] {label:58s} {detail}")
        if not ok:
            fails.append(label)

    strip = lambda line: line[: line.rfind(",")]
    frozen_rows = sorted(strip(l) for l in mas.read_raw(os.path.join(mas.DATA, "train_val.csv"))[1]
                         + mas.read_raw(os.path.join(mas.DATA, "test1.csv"))[1])
    ftv = frozen[frozen.split != "test"]
    test0 = {s for s, v in c.split.items() if v == "test"}
    tier12_in_test = {s for s in test0 if c.loc[s, "n_frags"] >= 3 or c.loc[s, "vdW"] >= mas.VDW_TIER1
                      or c.loc[s, "rho_min"] < ftv.density_kg_m3.min()
                      or any(m in c.loc[s, "metals"].split(",") for m in ("Fe", "Sb", "Cr"))}
    unmovable_test = {s for s in test0 if not c.loc[s, "movable"]}
    n_all = len(c)
    vals, tests = {}, {}

    for tag, parent, grow, p_val, p_test, mv in built:
        d, pdir = os.path.join(outdir, tag), parent_dir(outdir, parent)
        tv, t1 = pd.read_csv(os.path.join(d, "train_val.csv")), pd.read_csv(os.path.join(d, "test1.csv"))
        raw_tv, raw_t1 = mas.read_raw(os.path.join(d, "train_val.csv"))[1], mas.read_raw(os.path.join(d, "test1.csv"))[1]
        p_raw_tv, p_raw_t1 = mas.read_raw(os.path.join(pdir, "train_val.csv"))[1], mas.read_raw(os.path.join(pdir, "test1.csv"))[1]
        tr, va = tv[tv.split == "train"], tv[tv.split == "val"]
        vals[tag], tests[tag] = set(va.smiles), set(t1.smiles)
        n_val = int(round((20 if grow else 10) / 100 * n_all))
        print()
        print("=" * 78)
        print(f"{tag}   (from {parent or 'the frozen split'}: {len(mv)} {'test' if grow else 'val'} ILs moved "
              f"to {'val' if grow else 'test'})")
        print("=" * 78)
        check("rows carried byte-for-byte (only `split` rewritten)",
              sorted(strip(l) for l in raw_tv + raw_t1) == frozen_rows, f"{len(raw_tv) + len(raw_t1)} rows")
        check("training rows identical to the parent's, same order",
              [l for l in raw_tv if l.endswith(",train")] == [l for l in p_raw_tv if l.endswith(",train")],
              f"{len(tr)} rows, {tr.smiles.nunique()} ILs")
        if grow:
            check("val = parent val + the moved test ILs, rows unchanged",
                  sorted(l for l in raw_tv if l.endswith(",val"))
                  == sorted([l for l in p_raw_tv if l.endswith(",val")]
                            + [mas.with_split(l, "val") for l in p_raw_t1 if mas.row_key(l)[1] in mv]),
                  f"{len(p_val)} + {len(mv)}")
            check("test = parent test minus the moved ILs", set(t1.smiles) == p_test - mv, f"{len(p_test)} - {len(mv)}")
        else:
            check("remaining val rows identical to the parent's",
                  [l for l in raw_tv if l.endswith(",val")]
                  == [l for l in p_raw_tv if l.endswith(",val") and mas.row_key(l)[1] not in mv])
            check("test = parent test + the moved val ILs", set(t1.smiles) == p_test | mv, f"{len(p_test)} + {len(mv)}")
        check("test1.csv holds only split=test", set(t1.split) == {"test"})
        check("molecule-disjoint train / val / test",
              not (set(tr.smiles) & set(va.smiles) or set(tr.smiles) & set(t1.smiles) or set(va.smiles) & set(t1.smiles)))
        check(f"val size = {'20' if grow else '10'} % of the ILs", va.smiles.nunique() == n_val, f"{va.smiles.nunique()} ILs")
        check("all 1,092 ILs / 26,724 rows present",
              tv.smiles.nunique() + t1.smiles.nunique() == n_all and len(tv) + len(t1) == len(frozen))
        check("no train/val compound breaches a tier-1/2 bound",
              (tv.vdW < mas.VDW_TIER1).all() and (tv.n_frags == 2).all()
              and tv.density_kg_m3.between(ftv.density_kg_m3.min(), ftv.density_kg_m3.max()).all()
              and not any(m in ("Fe", "Sb", "Cr") for s in set(tv.smiles) for m in c.loc[s, "metals"].split(",")))
        check("tier-1/2 test compounds retained in test", tier12_in_test <= set(t1.smiles), f"{len(tier12_in_test)} ILs")
        phi = mas.PHI_CONST * pd.concat([tv, t1]).pipe(lambda x: x.vdW * x.density_kg_m3 / (1000 * x.MW_g_mol))
        check("packing screen: no row with phi >= 0.602214", (phi < mas.PHI_CONST).all(), f"max {phi.max():.4f}")
        check(f"every moved compound came from {'test' if grow else 'val'} and is movable",
              (mv <= p_test if grow else mv <= val0) and all(c.loc[s, "movable"] for s in mv), f"{len(mv)} moved")

    print()
    check("65/10/25, 70/10/20 and 75/10/15 share one validation set",
          vals["split_65_10_25"] == vals["split_70_10_20"] == vals["split_75_10_15"])
    check("nested: test(75/10/15) < test(70/10/20) < test(65/10/25)",
          tests["split_75_10_15"] <= tests["split_70_10_20"] <= tests["split_65_10_25"])
    check("65/20/15, 70/20/10 and 75/20/5 share one validation set",
          vals["split_65_20_15"] == vals["split_70_20_10"] == vals["split_75_20_5"])
    check("nested: test(75/20/5) < test(70/20/10) < test(65/20/15)",
          tests["split_75_20_5"] <= tests["split_70_20_10"] <= tests["split_65_20_15"])

    print()
    print("=" * 78)
    print("SUMMARY  (ILs / rows, % of the 1,092 ILs; last column: test ILs that can never move)")
    print("=" * 78)
    rows_of = frozen.groupby("smiles").size()
    print(f"  {'scheme':16s} {'train':>21s} {'val':>21s} {'test':>21s}  unmovable in test")
    for name in ("split_65_15_20", "split_65_10_25", "split_65_20_15", "frozen 70/15/15", "split_70_10_20",
                 "split_70_20_10", "split_75_15_10", "split_75_10_15", "split_75_20_5"):
        sets = parent_sets(outdir, None if name.startswith("frozen") else name)
        cells = "  ".join(f"{len(g):4d} / {int(rows_of[list(g)].sum()):5d} ({100 * len(g) / n_all:4.1f}%)" for g in sets)
        stuck = len(unmovable_test & sets[2])
        print(f"  {name:16s} {cells}  {stuck:3d} ({100 * stuck / len(sets[2]):2.0f}%)")
    print()
    if fails:
        print(f"VAL<->TEST SPLITS FAILED -- {len(fails)} check(s): {', '.join(fails)}")
        return 1
    print(f"VAL<->TEST SPLITS WRITTEN AND VERIFIED -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
