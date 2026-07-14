# Changelog

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
