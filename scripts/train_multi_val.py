#!/usr/bin/env python3
"""
train_multi_val.py  --  one training run, the best epoch for every split that shares its training rows.

The validation <-> test splits (scripts/make_val_test_splits.py) keep their parent's training rows
byte for byte; only validation and test differ. Validation never shapes training: it feeds no
gradient and no learning-rate schedule, and train.py draws the same single random number per epoch
for its validation pass whatever the validation set. So the weights after every epoch are the same
for all splits of a family, and one run yields each split's best epoch, read off its own validation
set: exactly the epoch a separate `train.py` run on that split would keep.

How: train.train_model runs the training loop unchanged and evaluates the FIRST split's validation
set itself. Its per-epoch hook evaluates the other splits' validation sets on the same weights, with
the global RNG state saved and restored around them (a DataLoader draws a seed from it whenever it
starts iterating), so the next epoch's batch order is untouched. Every split then gets train.py's own
treatment: best-epoch tracking (strictly lower val MAPE; the first epoch wins a tie), a
FrozenCollapseGuard of its own, and at the end the checkpoint train.py would leave on disk plus the
self-check (the train loss fell over the run; the checkpoint reloads and reproduces its val MAPE).
Training stops early only once every split has stopped.

Checkpoints are written once, when the run ends, with the content train.py leaves behind: the final
save (full history) for a run that completed, or the last best-so-far flush (history up to the best
epoch) for a run the collapse guard failed. train.py's interim flushes exist for crash safety only;
a crashed run here is simply rerun.

    python scripts/train_multi_val.py --splits 70_15_15 70_10_20 70_20_10 --vdw off --seed 1

Split names: 70_15_15 is the frozen split (data/train_val.csv); any other NAME is
data/alt_splits/split_NAME/train_val.csv. Checkpoints: checkpoints/gnn_core_NAME[_vdw]_bounded_sSEED.pt,
the name `train.py --out checkpoints/gnn_core_NAME.pt` gives. A JSON record of the run (outcome, best
epoch and per-epoch val MAPE for every split) goes to logs/multi_val/. Model and optimizer settings
are train.py's defaults (the published configuration: PNA, 3 core parameters, bounded head).
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch                                     # noqa: E402
from torch.utils.data import DataLoader          # noqa: E402
import train as T                                # noqa: E402

FROZEN = "70_15_15"


def split_csv(name: str) -> str:
    if name == FROZEN:
        return os.path.join("data", "train_val.csv")
    return os.path.join("data", "alt_splits", f"split_{name}", "train_val.csv")


def same_units(a: "T.ILDensityDataset", b: "T.ILDensityDataset") -> bool:
    """True if two datasets hold the same modeling units, in the same order, with identical points."""
    if len(a) != len(b):
        return False
    for x, y in zip(a.mols, b.mols):
        if x["smiles"] != y["smiles"]:
            return False
        if not (x["vdw"] == y["vdw"] or (math.isnan(x["vdw"]) and math.isnan(y["vdw"]))):
            return False
        if not all(torch.equal(x[k], y[k]) for k in ("T", "P", "rho", "MW")):
            return False
    return True


class AllStopped(Exception):
    """Every split has stopped (collapse guard): end training."""


class SplitRun:
    """One split's view of the shared run: its val set, best epoch, history, guard and outcome."""

    def __init__(self, name, val_ds, out_path, write_ckpt, batch_size):
        self.name = name
        self.csv = split_csv(name)
        self.val_ds = val_ds
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=T.collate)
        self.out_path = out_path
        self.write_ckpt = write_ckpt
        self.history = []              # train.py's per-epoch history, for this split's val set
        self.best = float("inf")
        self.best_state = None
        self.best_len = 0              # len(history) when the best epoch was recorded
        self.guard = T.FrozenCollapseGuard()
        self.stop_kind = None          # None while running; "collapsed" (guard) or "nan" (val collapse)
        self.stop_msg = ""
        self.ok = None
        self.reason = ""
        self.announced = False

    @property
    def running(self):
        return self.stop_kind is None

    def record(self, epoch, train_loss, mape, conv, model):
        """train_model's per-epoch bookkeeping, in train.py's order, for this split's val set."""
        if (not math.isfinite(mape)) or conv <= 0.0:
            # train.py's own collapse guard: stop, and do NOT record this epoch
            self.stop_kind = "nan"
            self.stop_msg = (f"validation collapsed at epoch {epoch} (val MAPE {mape}, "
                             f"converged {conv * 100:.1f}%); stopped early")
            return
        self.history.append({"epoch": epoch, "train_loss": train_loss, "val_mape": mape, "conv": conv})
        if mape < self.best:
            self.best = mape
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_len = len(self.history)
        try:
            self.guard({"density_mape": mape, "train_loss": train_loss})
        except T.TrainingCollapsed as e:
            self.stop_kind = "collapsed"
            self.stop_msg = str(e)

    def best_epoch(self):
        return min(self.history, key=lambda h: h["val_mape"])["epoch"] if self.history else None


def build_ckpt(model, config, deg, use_vdw, seed, best, history, csv):
    """train.py's checkpoint dict (train_one_seed.build_ckpt), field for field."""
    return {"model_state": model.state_dict(), "config": config, "deg": deg,
            "predict_association": config["predict_association"], "use_vdw": use_vdw,
            "use_bounds": config.get("use_bounds", False),
            "positive": True, "best_val_mape": best, "history": history,
            "seed": seed, "train_val_csv": os.path.abspath(csv)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", nargs="+", required=True,
                    help="splits sharing one training set; the first one's val set is train.py's own")
    ap.add_argument("--vdw", choices=["off", "on"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--threads", type=int, default=16, help="as train.py --threads (16 = published runs)")
    ap.add_argument("--no-ckpt", nargs="*", default=[], metavar="SPLIT",
                    help="splits to track and report without writing a checkpoint (e.g. one already on disk)")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--record-dir", default=os.path.join("logs", "multi_val"))
    args = ap.parse_args()
    if len(set(args.splits)) != len(args.splits):
        ap.error("--splits has duplicates")
    if set(args.no_ckpt) - set(args.splits):
        ap.error(f"--no-ckpt names splits not in --splits: {sorted(set(args.no_ckpt) - set(args.splits))}")
    if args.threads < 0:
        ap.error("--threads must be 0 or positive")

    os.chdir(REPO)
    log_filename = T.setup_logging()
    logging.info(f"Logging initialized. Saving logs to: {log_filename}")
    torch.set_num_threads(args.threads or max(1, os.cpu_count() or 1))
    use_vdw = args.vdw == "on"
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    logging.info("multi-val run: one training run, best epoch picked per split "
                 "(scripts/train_multi_val.py)")
    logging.info(f"splits: {args.splits}  (train.py's own val pass: {args.splits[0]})")
    logging.info("parameter set: 3 (core)")
    logging.info("parameter bounds: on (physical box, sigmoid-squashed head)")
    logging.info(f"CPU threads: {torch.get_num_threads()}  "
                 f"(this machine has {os.cpu_count() or '?'} logical CPUs)")
    logging.info(f"vdW {args.vdw} | seed {args.seed} | {args.epochs} epochs")

    # ---- data: every split must hold the same training units, in the same order ----
    built = {}
    for name in args.splits:
        csv = split_csv(name)
        if not os.path.exists(csv):
            raise SystemExit(f"no such split: {csv}")
        built[name] = T.build_datasets(csv, use_vdw=use_vdw)
    train_ds, _, deg = built[args.splits[0]]
    for name in args.splits[1:]:
        tr, _, dg = built[name]
        if not same_units(train_ds, tr) or not torch.equal(dg, deg):
            raise SystemExit(f"split {name} does not share {args.splits[0]}'s training rows; "
                             f"train it with train.py instead.")
    vdw_mean, vdw_std = T.compute_vdw_stats(train_ds) if use_vdw else (0.0, 1.0)
    # train.py main()'s config, with its CLI defaults
    config = {"conv_type": "PNA", "hidden": 256, "depth": 6, "towers": 4, "heads": 4, "lr": 1e-3,
              "batch_size": 16, "grad_clip": 0.0, "predict_association": False,
              "use_vdw": use_vdw, "vdw_mean": vdw_mean, "vdw_std": vdw_std, "use_bounds": True}
    logging.info(f"modeling units: train {len(train_ds)} | val "
                 + ", ".join(f"{n} {len(built[n][1])}" for n in args.splits)
                 + f"; deg histogram len {len(deg)}")
    logging.info(f"vdW input: {args.vdw}"
                 + (f"  (train mean {vdw_mean:.2f}, std {vdw_std:.2f} A^3)" if use_vdw else ""))
    logging.info(f"config: {config}")

    runs = []
    for name in args.splits:
        out_path = T.checkpoint_path(os.path.join(args.ckpt_dir, f"gnn_core_{name}.pt"),
                                     args.seed, use_vdw, True)
        runs.append(SplitRun(name, built[name][1], out_path, name not in args.no_ckpt,
                             config["batch_size"]))
        logging.info(f"output [{name}] -> "
                     + (os.path.abspath(out_path) if name not in args.no_ckpt else "(no checkpoint written)"))

    # ---- train once; hooks see the model and train.py's own val result ----
    captured = {}
    orig_make_model, orig_evaluate = T.make_model, T.evaluate

    def make_model_hook(cfg, dg):
        model = orig_make_model(cfg, dg)
        captured["model"] = model
        return model

    def evaluate_hook(model, loader):
        out = orig_evaluate(model, loader)
        captured["eval"] = out
        return out

    n_done = [0]
    train_losses = []
    t_epoch = [time.time()]

    def report(metrics):
        epoch = n_done[0]
        n_done[0] += 1
        model = captured["model"]
        loss = metrics["train_loss"]
        train_losses.append(loss)
        results = {runs[0].name: captured["eval"]}
        rng = torch.get_rng_state()
        try:
            for r in runs[1:]:
                if r.running:
                    results[r.name] = orig_evaluate(model, r.val_loader)
        finally:
            torch.set_rng_state(rng)      # the extra val passes must not move the batch order
        for r in runs:
            if r.running:
                r.record(epoch, loss, *results[r.name], model)
        now = time.time()
        logging.info("          by split: "
                     + " | ".join(f"{r.name} {results[r.name][0]:6.2f}%" + ("" if r.running else " stopped")
                                  if r.name in results else f"{r.name} stopped" for r in runs)
                     + f"   ({now - t_epoch[0]:.0f} s)")
        t_epoch[0] = now
        for r in runs:
            if r.stop_kind and not r.announced:
                r.announced = True
                logging.warning(f"[{r.name}] stopped: {r.stop_msg}")
        if not any(r.running for r in runs):
            raise AllStopped()

    T.make_model, T.evaluate = make_model_hook, evaluate_hook
    try:
        T.train_model(config, train_ds, runs[0].val_ds, deg, args.epochs, report_fn=report,
                      verbose=True, seed=args.seed, on_improve=None)
        if n_done[0] < args.epochs:
            # train_model broke out of its loop on its own collapse guard (first split's val set)
            r0 = runs[0]
            if r0.running:
                r0.stop_kind = "nan"
                r0.stop_msg = f"validation collapsed at epoch {n_done[0]}; stopped early"
            for r in runs[1:]:
                if r.running:
                    r.stop_kind = "nan"
                    r.stop_msg = f"run ended at epoch {n_done[0]} when {r0.name}'s validation collapsed"
    except AllStopped:
        pass
    finally:
        T.make_model, T.evaluate = orig_make_model, orig_evaluate

    # ---- per split: the checkpoint train.py leaves on disk, then its self-check ----
    model = captured["model"]
    for r in runs:
        logging.info("")
        logging.info(f"--- [{r.name}] vdW {args.vdw} | seed {args.seed} ---")
        if not r.history or r.best_state is None:
            r.ok, r.reason = False, "training collapsed before reaching a single valid epoch; no checkpoint"
            logging.error(f"[{r.name}] FAILED, {r.reason}")
            continue
        collapsed = r.stop_kind == "collapsed"
        history = r.history[:r.best_len] if collapsed else r.history
        model.load_state_dict(r.best_state)
        if r.write_ckpt:
            os.makedirs(os.path.dirname(os.path.abspath(r.out_path)), exist_ok=True)
            torch.save(build_ckpt(model, config, deg, use_vdw, args.seed, r.best, history, r.csv), r.out_path)
        logging.info(f"[{r.name}] best val MAPE: {r.best:.2f}%   (epoch {r.best_epoch()})")
        if collapsed:
            r.ok, r.reason = False, r.stop_msg
            logging.error(f"[{r.name}] vdW {args.vdw} seed {args.seed} FAILED, {r.stop_msg}")
        else:
            losses = [h["train_loss"] for h in r.history]
            early = len(r.history) < args.epochs
            if not all(math.isfinite(x) for x in losses):
                r.ok, r.reason = False, "non-finite train loss"
            elif not early and not losses[-1] <= losses[0] + 1e-9:
                r.ok, r.reason = False, "training loss increased overall (diverged)"
            else:
                r.ok = True
                if early:
                    r.reason = f"early-stopped after {len(r.history)}/{args.epochs} epochs: {r.stop_msg}"
            logging.info(f"[{r.name}] train loss {losses[0]:.5f} -> {losses[-1]:.5f}"
                         + (f"   [{len(r.history)} epochs, early-stopped]" if early else ""))
            if not r.ok:
                logging.error(f"[{r.name}] vdW {args.vdw} seed {args.seed} FAILED self-check: {r.reason}")
        # reload check (train.py's last self-check), from disk when a checkpoint was written
        reloaded = orig_make_model(config, deg)
        state = torch.load(r.out_path)["model_state"] if r.write_ckpt else r.best_state
        reloaded.load_state_dict(state)
        rm, _ = orig_evaluate(reloaded, r.val_loader)
        if abs(rm - r.best) < 1e-6:
            logging.info(f"[{r.name}] checkpoint reloads and reproduces val MAPE ({rm:.2f}%).")
        else:
            r.ok, r.reason = False, f"reloaded MAPE {rm} != best {r.best}"
            logging.error(f"[{r.name}] FAILED: {r.reason}")
        if r.write_ckpt:
            logging.info(f"[{r.name}] wrote {os.path.abspath(r.out_path)}")

    # ---- summary + JSON record ----
    logging.info("")
    logging.info("================  SUMMARY  ================")
    for r in runs:
        tag = f"val MAPE {r.best:6.2f}% (epoch {r.best_epoch()})" if r.history else "no valid epoch"
        logging.info(f"  {r.name:>9}  vdW {args.vdw}  seed {args.seed}:  {'OK    ' if r.ok else 'FAILED'}  "
                     f"{tag}" + (f"   [{r.reason}]" if r.reason else ""))
    os.makedirs(args.record_dir, exist_ok=True)
    rec_path = os.path.join(args.record_dir,
                            f"{'+'.join(args.splits)}_vdw_{args.vdw}_s{args.seed}.json")
    record = {
        "script": "scripts/train_multi_val.py", "splits": args.splits, "vdw": args.vdw,
        "seed": args.seed, "epochs": args.epochs, "epochs_run": n_done[0],
        "threads": torch.get_num_threads(), "log": log_filename,
        "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": True,
        "train_loss": train_losses,
        "per_split": {r.name: {
            "train_val_csv": r.csv, "val_units": len(r.val_ds), "ok": r.ok, "reason": r.reason,
            "stop": r.stop_kind, "best_val_mape": r.best if r.history else None,
            "best_epoch": r.best_epoch(), "epochs_recorded": len(r.history),
            "checkpoint": r.out_path if r.write_ckpt and r.best_state is not None else None,
            "val_mape": [h["val_mape"] for h in r.history], "conv": [h["conv"] for h in r.history],
        } for r in runs},
    }
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    logging.info(f"record -> {os.path.abspath(rec_path)}")


if __name__ == "__main__":
    main()
