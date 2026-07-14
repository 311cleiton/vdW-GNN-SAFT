#!/usr/bin/env python3
"""
smoke_train.py  --  prove the training path runs end to end, headlessly.

Until train.py gained `--no-gui` this test was impossible: training required a display and a
human click, so no automated check could ever touch it. It can now, and CI runs this on every
push.

It is a SMOKE test, not a reproduction. It carves a tiny subset out of the training data and
runs a handful of epochs on a deliberately small network. The MAPEs it prints are meaningless
-- what is being asserted is that the pipeline holds together:

    CLI config -> graph build -> GNN -> differentiable PC-SAFT -> loss -> backward
              -> checkpoint written -> checkpoint reloaded -> val MAPE reproduced

train.py's own per-seed self-check does the last two, and it aborts the run if the training
loss rises. So if this script exits 0, the whole chain is intact.

    python scripts/smoke_train.py            # ~1-2 min
    python scripts/smoke_train.py --epochs 6 --n-train 32
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "src")
DATA = os.path.join(HERE, "data")


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless end-to-end smoke test of train.py.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=16, help="training ILs to keep")
    ap.add_argument("--n-val", type=int, default=4, help="validation ILs to keep")
    args = ap.parse_args()

    import pandas as pd
    import torch

    tmp = tempfile.mkdtemp(prefix="vdwsaft_smoke_")
    subset = os.path.join(tmp, "smoke.csv")
    base = os.path.join(tmp, "smoke.pt")

    try:
        df = pd.read_csv(os.path.join(DATA, "train_val.csv"))
        tr, va = df[df.split == "train"], df[df.split == "val"]
        keep_tr = tr.smiles.drop_duplicates().head(args.n_train)
        keep_va = va.smiles.drop_duplicates().head(args.n_val)
        sub = pd.concat([tr[tr.smiles.isin(keep_tr)], va[va.smiles.isin(keep_va)]],
                        ignore_index=True)
        sub.to_csv(subset, index=False)
        print(f"== smoke subset: {len(sub)} rows | "
              f"{sub[sub.split == 'train'].smiles.nunique()} train ILs / "
              f"{sub[sub.split == 'val'].smiles.nunique()} val ILs ==")

        # Both vdW arms x 2 seeds -> 4 checkpoints, exactly the shape of the paper's sweep.
        cmd = [
            sys.executable, os.path.join(SRC, "train.py"), "--no-gui",
            "--train-val-csv", subset,
            "--vdw", "both", "--bounds", "on", "--seeds", "0", "1",
            "--epochs", str(args.epochs),
            "--hidden", "32", "--depth", "2", "--batch-size", "8",
            "--out", base,
        ]
        print("$ " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=tmp)
        if r.returncode != 0:
            print("\nFAILED: headless training exited non-zero.")
            return 1

        # train.py has already asserted (per seed) that the loss fell and the checkpoint
        # round-trips. Here we only confirm the sweep produced the files it promised.
        root, ext = os.path.splitext(base)
        expected = [f"{root}{tag}_bounded_s{s}{ext}"
                    for tag in ("", "_vdw") for s in (0, 1)]

        print()
        print("== checkpoints ==")
        for p in expected:
            if not os.path.exists(p):
                print(f"  MISSING  {p}")
                return 1
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            assert cfg["use_bounds"] is True
            assert cfg["predict_association"] is False
            assert "model_state" in ck and "deg" in ck
            print(f"  OK  {os.path.basename(p)}   "
                  f"use_vdw={ck['use_vdw']!s:5s} seed={ck['seed']} "
                  f"best_val_mape={ck['best_val_mape']:.1f}%")

        print()
        print("=" * 70)
        print("SMOKE TEST PASSED -- headless training runs end to end.")
        print("(The MAPEs above are meaningless: 4 epochs on 16 molecules. The point is")
        print(" that the chain holds: config -> graph -> GNN -> PC-SAFT -> loss -> ckpt.)")
        print("=" * 70)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
