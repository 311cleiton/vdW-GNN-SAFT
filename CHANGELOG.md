# Changelog

## [Unreleased]

- train.py: new `--threads` option for PyTorch's CPU thread count, which the run header now
  logs. It defaults to 16, the thread count of the published runs, on every machine. Training
  used to take every logical CPU, and because the thread count changes the order in which sums
  are accumulated, a machine with another CPU count trained a different trajectory from the same
  seed. `--threads 0` restores the old behaviour.
- train.py: a run that collapses while FeOs still converges (the bounded head saturates and the val
  MAPE freezes: the "Esper mode") now stops and fails after 5 frozen epochs. It used to train to
  the end, where the rising-loss self-check failed it -- except when the collapse came in the first
  epoch: the loss then never rises above the collapsed level, and the run passed as OK with a
  best val MAPE of 40 % or more at epoch 0. Stopping saves up to ~115 epochs per collapsed run.
- New scripts/train_multi_val.py: one training run for several splits that share their training
  rows and differ only in validation and test. Validation never shapes training, so the run keeps
  each split's best epoch, read off that split's own validation set: the checkpoint a separate
  train.py run on that split would keep.
- Reproduction uses the paper's seeds 0, 3 and 4. The commands in the README, docs/ and train.py
  said `--seeds 0 1 2 3`: that trained seed 2, whose two runs both collapse, and seed 1, whose
  vdW-on run collapses after the epoch-30 restart, and it left out seed 4. The README results
  table now quotes the three-seed means: point MAPE 5.32 ± 0.25 % (vdW off) and 4.02 ± 0.14 %
  (vdW on), molecule MAPE 6.02 ± 0.27 % and 4.98 ± 0.11 %.
- train.py: its docstring and end-of-run message no longer say that evaluate.py computes a
  molecule-level cluster bootstrap. evaluate.py reports point- and molecule-level MAPE per seed.
- .gitignore: checkpoints in subfolders of checkpoints/, and the logs/ folder, are ignored too.

## [1.0.0] - 2026-07-14

Initial public release.

- Nine-module pipeline: dataset construction from NIST ILThermo, van der Waals volume
  computation, GNN (PNA / GATv2 / TransformerConv) with bounded prior-initialized heads,
  differentiable PC-SAFT bridge (feos-torch), physics-in-the-loop training, BOHB
  hyperparameter search, evaluation against classical baselines, and PC-SAFT parameter
  export.
- Frozen benchmark: 26,724 liquid-density measurements over 1,092 ionic liquids,
  molecule-disjoint train/val/test splits, full provenance (ILThermo entry IDs and
  primary references) carried in the CSVs.
- Headless operation: train.py, evaluate.py and export_params.py accept --no-gui and run
  on CI, over SSH, and on batch nodes; the Tk GUIs remain the default.
- Self-checks: scripts/verify_dataset.py (30 assertions reproducing every dataset number
  in the paper), scripts/selfcheck.py (runner), scripts/smoke_train.py (headless
  end-to-end training test); pcsaft.py and model.py carry their own standalone checks.
- CI verifies the frozen dataset on every push.
- CPU-only throughout; no accelerator required.
