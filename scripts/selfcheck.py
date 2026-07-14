#!/usr/bin/env python3
"""
selfcheck.py  --  run the framework's self-checks and report honestly which ones exist.

Contribution (vi) of the paper is an "open, self-checking framework". This script is the
entry point for that claim. It runs every self-check that exists and states plainly, for the
modules that have none, why.

    python scripts/selfcheck.py                      # data + physics + model (fast, ~1 min)
    python scripts/selfcheck.py --with-training     # + a headless end-to-end training run
    python scripts/selfcheck.py --with-baselines    # + the RF/MLP fit (slow, several minutes)
    python scripts/selfcheck.py --list              # just show the self-check inventory

Coverage (see docs/known-issues.md for the full picture):

  module                self-check?  how it is reached
  --------------------  -----------  ---------------------------------------------------------
  verify_dataset.py     yes          python scripts/verify_dataset.py     (30 dataset assertions)
  pcsaft.py             yes          python src/pcsaft.py                 (density + grad + FeOs xcheck)
  model.py              yes          python src/model.py                  (24 arch configs + grads)
  baselines.py          yes          python src/baselines.py --self-check (fit + MAPE plausibility)
  train.py              yes          python scripts/smoke_train.py        (headless end-to-end chain)
  evaluate.py           in-run       asserted per test file inside main()
  export_params.py      in-run       asserted per exported file inside main()
  build_dataset.py      in-run       asserted after the pull (no NaNs, no leakage, valid SMILES)
  tune.py               NONE         thin wrapper over train.train_model; nothing of its own
  smiles2vdW_volume.py  NONE         GUI-only script; no headless entry point

"in-run" means the assertions fire during a real run of that module and abort it on failure --
they are not separately invocable, so this script cannot exercise them without a full run of
that module. Two modules have no self-check at all. Both facts are stated rather than papered
over.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "src")
SCRIPTS = os.path.join(HERE, "scripts")

INVENTORY = [
    ("verify_dataset.py", "standalone", "30 dataset assertions vs. the manuscript"),
    ("pcsaft.py", "standalone", "density, gradient flow, plain-FeOs cross-check"),
    ("model.py", "standalone", "24 architecture configs, gradients, bias-at-priors"),
    ("baselines.py", "standalone", "RF + ECFP-MLP fit, finite preds, plausible MAPE"),
    ("train.py", "standalone", "headless end-to-end chain (scripts/smoke_train.py)"),
    ("evaluate.py", "in-run", "per test file, inside main()"),
    ("export_params.py", "in-run", "per exported parameter file, inside main()"),
    ("build_dataset.py", "in-run", "no NaNs, no split leakage, all SMILES valid"),
    ("tune.py", "none", "thin wrapper over train.train_model"),
    ("smiles2vdW_volume.py", "none", "GUI-only; no headless entry point"),
]


def run(label: str, cmd: list[str]) -> bool:
    print()
    print("=" * 78)
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 78)
    r = subprocess.run(cmd, cwd=HERE)
    ok = r.returncode == 0
    print(f"  --> {'PASSED' if ok else 'FAILED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the vdW-GNN-SAFT self-checks.")
    ap.add_argument("--with-baselines", action="store_true",
                    help="Also fit the RF/MLP baselines (slow: several minutes).")
    ap.add_argument("--with-training", action="store_true",
                    help="Also run the headless training smoke test (slow: a minute or two).")
    ap.add_argument("--list", action="store_true", help="Print the self-check inventory and exit.")
    args = ap.parse_args()

    if args.list:
        print(f"{'module':22s} {'kind':11s} what it asserts")
        print("-" * 78)
        for m, kind, what in INVENTORY:
            print(f"{m:22s} {kind:11s} {what}")
        return 0

    py = sys.executable
    results = []
    results.append(("dataset", run("DATA  -- dataset provenance",
                                   [py, os.path.join(SCRIPTS, "verify_dataset.py")])))
    results.append(("pcsaft", run("PHYSICS -- differentiable PC-SAFT bridge",
                                  [py, os.path.join(SRC, "pcsaft.py")])))
    results.append(("model", run("MODEL -- GNN architecture + gradients",
                                 [py, os.path.join(SRC, "model.py")])))
    if args.with_training:
        results.append(("training", run("TRAINING -- headless end-to-end chain",
                                        [py, os.path.join(SCRIPTS, "smoke_train.py")])))
    else:
        print()
        print("  (skipping the training smoke test; add --with-training to include it)")

    if args.with_baselines:
        results.append(("baselines", run("BASELINES -- RF + ECFP-MLP",
                                         [py, os.path.join(SRC, "baselines.py"), "--self-check",
                                          "--train-val-csv", "data/train_val.csv",
                                          "--test-csv", "data/test1.csv"])))
    else:
        print()
        print("  (skipping the baselines fit; add --with-baselines to include it)")

    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for name, ok in results:
        print(f"  {name:12s} {'PASS' if ok else 'FAIL'}")
    bad = [n for n, ok in results if not ok]
    print("=" * 78)
    if bad:
        print(f"  {len(bad)} self-check(s) FAILED: {', '.join(bad)}")
        return 1
    print("  All self-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
