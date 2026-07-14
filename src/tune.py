#!/usr/bin/env python3
"""
tune.py  --  Group C (2/2) of the GNN / PC-SAFT density factory.

Hyperparameter search over the GNN, reusing train.py's `train_model` so the per-trial
training is identical to a normal run. The tuning metric is validation density MAPE
(not vapor pressure -- P_sat needs iterative solves that don't converge early in training).

Two modes
---------
  --dry-run : compare a few hand-picked configs sequentially with NO Ray dependency.
              Useful as a quick conv-type bake-off and for environments without Ray.
  (default) : full BOHB search (TuneBOHB + HyperBandForBOHB). HyperBand reads the per-epoch
              `density_mape` we report and kills weak trials early.

Environment note
----------------
BOHB requires `hpbandster` + `ConfigSpace`, and these (with TuneBOHB) need Python <= 3.11.
The heavy Ray/BOHB imports live inside run_bohb() so this file imports fine anywhere; the
--dry-run path only needs torch + the Group B/C modules. If your Ray version names the
reporting call differently, adjust _ray_report (this mirrors the Module 5 setup).

Run e.g.:
  python tune.py --train-val-csv data/train_val.csv --dry-run --epochs 30
  python tune.py --train-val-csv data/train_val.csv --trials 32 --epochs 40
  # tune the 5-parameter (core + association) model instead:
  python tune.py --train-val-csv data/train_val.csv --dry-run --epochs 30 --params assoc
  # tune with the vdW input on (independent of --params):
  python tune.py --train-val-csv data/train_val.csv --dry-run --epochs 30 --vdw on
"""

from __future__ import annotations
import argparse

from train import build_datasets, train_model, compute_vdw_stats, DEFAULT_TRAIN_VAL_CSV

# fixed (not searched): towers/heads are 4 and every candidate `hidden` is divisible by 4,
# which keeps PNA towers and attention heads valid for all sampled configs.
FIXED = {"towers": 4, "heads": 4}


def _ray_report(metrics):
    """Report per-epoch metrics to Ray Tune, resilient to API renames across Ray versions."""
    try:
        from ray import train as ray_train
        ray_train.report(metrics)
    except Exception:
        from ray.air import session
        session.report(metrics)


def _trainable(config, train_ds=None, val_ds=None, deg=None, epochs=10,
               predict_association=False, use_vdw=False, vdw_mean=0.0, vdw_std=1.0):
    cfg = {**config, **FIXED, "predict_association": predict_association,
           "use_vdw": use_vdw, "vdw_mean": vdw_mean, "vdw_std": vdw_std}
    train_model(cfg, train_ds, val_ds, deg, epochs=epochs, report_fn=_ray_report, verbose=False)


def dry_run(train_val_csv, epochs, predict_association=False, use_vdw=False):
    """Ray-free: train a few configs and rank them by validation density MAPE."""
    train_ds, val_ds, deg = build_datasets(train_val_csv, use_vdw=use_vdw)
    vdw_mean, vdw_std = compute_vdw_stats(train_ds) if use_vdw else (0.0, 1.0)
    configs = [
        {"conv_type": "PNA", "hidden": 128, "depth": 4, "lr": 1e-3, "batch_size": 16},
        {"conv_type": "GATv2", "hidden": 128, "depth": 4, "lr": 1e-3, "batch_size": 16},
        {"conv_type": "Transformer", "hidden": 128, "depth": 4, "lr": 1e-3, "batch_size": 16},
    ]
    print(f"parameter set: {'5 (core + association)' if predict_association else '3 (core)'}"
          f"   vdW input: {'on' if use_vdw else 'off'}")
    results = []
    for cfg in configs:
        full = {**cfg, **FIXED, "predict_association": predict_association,
                "use_vdw": use_vdw, "vdw_mean": vdw_mean, "vdw_std": vdw_std}
        best, _, _ = train_model(full, train_ds, val_ds, deg, epochs=epochs, verbose=False)
        print(f"  {cfg['conv_type']:11s} hidden={cfg['hidden']} depth={cfg['depth']} "
              f"lr={cfg['lr']}  ->  best val MAPE {best:.2f}%")
        results.append((best, cfg))
    results.sort(key=lambda x: x[0])
    print(f"\nbest (dry-run): {results[0][1]}  ->  {results[0][0]:.2f}% MAPE")
    return results[0]


def run_bohb(train_val_csv, num_trials, epochs, predict_association=False, use_vdw=False):
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    from ray import tune
    from ray.tune.search.bohb import TuneBOHB
    from ray.tune.schedulers import HyperBandForBOHB

    train_ds, val_ds, deg = build_datasets(train_val_csv, use_vdw=use_vdw)
    vdw_mean, vdw_std = compute_vdw_stats(train_ds) if use_vdw else (0.0, 1.0)
    print(f"parameter set: {'5 (core + association)' if predict_association else '3 (core)'}"
          f"   vdW input: {'on' if use_vdw else 'off'}")

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter("conv_type", ["PNA", "GATv2", "Transformer"]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter("hidden", [64, 128, 256]))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("depth", lower=2, upper=6))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter("batch_size", [8, 16, 32]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("lr", lower=1e-4, upper=5e-3, log=True))

    searcher = TuneBOHB(space=cs, metric="density_mape", mode="min")
    scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=epochs, reduction_factor=3)

    tuner = tune.Tuner(
        tune.with_parameters(_trainable, train_ds=train_ds, val_ds=val_ds, deg=deg,
                             epochs=epochs, predict_association=predict_association,
                             use_vdw=use_vdw, vdw_mean=vdw_mean, vdw_std=vdw_std),
        tune_config=tune.TuneConfig(metric="density_mape", mode="min",
                                    search_alg=searcher, scheduler=scheduler, num_samples=num_trials),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="density_mape", mode="min")
    print("best config:", best.config)
    print(f"best density MAPE: {best.metrics['density_mape']:.2f}%")
    return best


def main():
    ap = argparse.ArgumentParser(description="BOHB search for the physics-in-loop GNN.")
    ap.add_argument("--train-val-csv", default=DEFAULT_TRAIN_VAL_CSV,
                    help="CSV with the train+val rows (a `split` column tags each as train/val).")
    ap.add_argument("--dry-run", action="store_true", help="No-Ray sequential bake-off of a few configs.")
    ap.add_argument("--trials", type=int, default=24, help="BOHB num_samples.")
    ap.add_argument("--epochs", type=int, default=40, help="Max epochs per trial (HyperBand max_t).")
    ap.add_argument("--params", choices=["core", "assoc"], default="core",
                    help="PC-SAFT parameters the GNN predicts: 'core' = 3, 'assoc' = 5 "
                         "(+ kappa_ab, epsilon_k_ab). Fixed for the whole search, not a searched "
                         "dimension -- pick the mode, then tune the architecture for it.")
    ap.add_argument("--vdw", choices=["off", "on"], default="off",
                    help="Use the van der Waals volume as an extra GNN input. Fixed for the whole "
                         "search (like --params), not a searched dimension. 'on' needs a `vdW` column.")
    args = ap.parse_args()

    assoc = (args.params == "assoc")
    use_vdw = (args.vdw == "on")
    if args.dry_run:
        dry_run(args.train_val_csv, args.epochs, predict_association=assoc, use_vdw=use_vdw)
    else:
        run_bohb(args.train_val_csv, args.trials, args.epochs,
                 predict_association=assoc, use_vdw=use_vdw)


if __name__ == "__main__":
    main()
