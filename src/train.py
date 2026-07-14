#!/usr/bin/env python3
"""
train.py  --  Group C (1/2) of the GNN / PC-SAFT density factory.

Physics-in-the-loop training: the GNN predicts PC-SAFT pure-component parameters per molecule
-- the 3 core params (m, sigma, epsilon_k) by default, or all 5 (m, sigma, epsilon_k, kappa_ab,
epsilon_k_ab) when association is turned on with `--params assoc` -- FeOs turns them into a
liquid density at each experimental (T, P), and the modified-Huber relative-error loss is taken
on density vs. experiment. Gradients flow density -> parameters -> GNN weights (verified in
Group B). Optionally (`--vdw on`) the molecule's van der Waals volume [cubic angstrom] is fed to
the GNN as an extra input. Both choices are stored in the checkpoint `config`, so evaluate.py and
export_params.py rebuild the matching model automatically.

Data layout
-----------
Training and validation rows live together in ONE file (default: data/train_val.csv); a `split`
column tags every row as "train" or "val". Test molecules are held out in their own files and
are NOT touched here -- evaluate.py (Group D) loads them. The default location is the
DEFAULT_TRAIN_VAL_CSV constant below; override it on the CLI with --train-val-csv. With
`--vdw on` a `vdW` column is also required (column 2 in the provided CSVs).

Per-molecule vs. per-point
--------------------------
The GNN runs once per modeling unit (per graph). The unit is the molecule (grouped by `smiles`),
or the (molecule, vdW) pair when `--vdw on`, since the same SMILES can carry several vdW values
(multi-fragment mixtures such as deep eutectic solvents). Each unit has many (T, P, rho) points,
so a batch carries B graphs plus all their points concatenated, with `mol_index` mapping every
point back to its graph and a per-graph vdW vector. After model(batch, vdw) -> [B, 3] (or [B, 5]
with association), we gather params[mol_index] -> [P, 3]/[P, 5] and evaluate density at the P
points. One GNN forward per unit, one density solve per point.

Sweeps: seeds and vdW
---------------------
Two axes can be swept in a single run, each producing one checkpoint per combination:

  * Seeds -- training is stochastic (weight init + batch shuffling), so the GUI lets you tick
    several seeds (0 .. 4). Each checkpoint's file name gets a `_s{seed}` tag and the seed is
    stored inside it.
  * vdW   -- tick "off" and/or "on" to train the with- and without-vdW models side by side for a
    like-for-like ablation. "on" adds a `_vdw` tag; the vdW setting (and its train-set
    standardization stats) is stored in each checkpoint's `config`, so evaluate.py and
    export_params.py rebuild the matching model automatically.

Ticking both axes trains the full grid (one checkpoint per vdW setting x seed) -- equivalent to
invoking this script once per (`--vdw`, `--seed`) pair. The datasets are rebuilt once per vdW
setting (the modeling unit differs: per-SMILES with vdW off, per-(SMILES, vdW) with vdW on) and
reused across that setting's seeds. The val MAPE mean +/- std across seeds is printed per vdW
setting at the end. (The headline vdW ablation is still computed on the held-out test set in
evaluate.py -- cluster-bootstrap at the molecule level -- not from these val numbers.)

Numerical stability
-------------------
A batch on which FeOs fails to converge everywhere can yield a non-finite loss; stepping on it
poisons the weights (NaN spreads, every later epoch stays NaN). Two guards prevent that: batches
with a non-finite loss are skipped (a no-op for healthy batches), and if validation collapses
(non-finite MAPE / zero FeOs coverage) training stops early and keeps the best pre-collapse
weights. `--grad-clip` (off by default) additionally caps the gradient norm.
"""

from __future__ import annotations
import argparse
import os
import math
import logging
import statistics
from datetime import datetime

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from model import PCSAFTGNN, smiles_to_data, compute_degree_histogram
from pcsaft import (assemble_pcsaft_params, mass_density_kg_per_m3,
                    huber_relative_loss, density_mape)

# ---------------------------------------------------------------------------
# data locations  (edit here, or override on the CLI with --train-val-csv)
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_VAL_CSV = "data/train_val.csv"

# ---------------------------------------------------------------------------
# sweep helpers (seeds x vdW)
# ---------------------------------------------------------------------------
MAX_SEEDS = 5                             # the GUI exposes seeds 0 .. MAX_SEEDS-1
VDW_LABELS = {False: "off", True: "on"}   # use_vdw boolean -> human label


def checkpoint_path(base_path: str, seed: int, use_vdw: bool, use_bounds: bool) -> str:
    """Insert the config tags before the extension of `base_path`, in a fixed order.

    A single base name fans out over the sweep, so every checkpoint gets a distinct,
    self-describing name. Order is `_vdw` (only when vdW is on) -> `_bounded` (only when bounds
    are on) -> `_s{seed}`, matching the names the previous single-run CLI/GUI produced so existing
    checkpoints line up.

    e.g. checkpoints/gnn_core.pt, seed 2, vdw on,  bounds on  -> checkpoints/gnn_core_vdw_bounded_s2.pt
         checkpoints/gnn_core.pt, seed 0, vdw off, bounds off -> checkpoints/gnn_core_s0.pt
         checkpoints/gnn_core,    seed 1, vdw off, bounds on  -> checkpoints/gnn_core_bounded_s1.pt  (.pt default)
    """
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    tag = ""
    if use_vdw:
        tag += "_vdw"
    if use_bounds:
        tag += "_bounded"
    tag += f"_s{seed}"
    return f"{root}{tag}{ext}"


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
def setup_logging():
    """Configures dynamic, timestamped logging to both file and console."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class ILDensityDataset(Dataset):
    """One item per modeling unit, with all its (T, P, rho, MW) points."""

    def __init__(self, df: pd.DataFrame, use_vdw: bool = False):
        self.mols = []
        has_vdw = "vdW" in df.columns
        keys = ["smiles", "vdW"] if use_vdw else ["smiles"]
        for _, g in df.groupby(keys, sort=False):
            smiles = g["smiles"].iloc[0]
            self.mols.append({
                "smiles": smiles,
                "data": smiles_to_data(smiles),
                "vdw": float(g["vdW"].iloc[0]) if has_vdw else float("nan"),
                "T": torch.tensor(g["T_K"].to_numpy(), dtype=torch.float64),
                "P": torch.tensor(g["P_kPa"].to_numpy(), dtype=torch.float64),
                "rho": torch.tensor(g["density_kg_m3"].to_numpy(), dtype=torch.float64),
                "MW": torch.tensor(g["MW_g_mol"].to_numpy(), dtype=torch.float64),
            })

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, i):
        return self.mols[i]


def collate(batch):
    """Batch B graphs and concatenate their points; build point->graph index and per-graph vdW."""
    graphs = [b["data"] for b in batch]
    big = Batch.from_data_list(graphs)
    T = torch.cat([b["T"] for b in batch])
    P = torch.cat([b["P"] for b in batch])
    rho = torch.cat([b["rho"] for b in batch])
    MW = torch.cat([b["MW"] for b in batch])
    vdw = torch.tensor([b["vdw"] for b in batch], dtype=torch.float64)   # [B], one per graph
    counts = torch.tensor([len(b["T"]) for b in batch])
    mol_index = torch.repeat_interleave(torch.arange(len(batch)), counts)
    return big, T, P, rho, MW, vdw, mol_index


def compute_vdw_stats(train_ds: "ILDensityDataset"):
    """Mean/std of vdW over the training items (used to standardize the vdW input)."""
    vals = torch.tensor([float(m["vdw"]) for m in train_ds.mols], dtype=torch.float64)
    if not torch.isfinite(vals).all():
        raise ValueError("vdW stats requested but some training items have no vdW value.")
    mean = float(vals.mean())
    std = float(vals.std())
    return mean, (std if std > 1e-8 else 1.0)


def build_datasets(train_val_csv: str = DEFAULT_TRAIN_VAL_CSV, use_vdw: bool = False):
    """Build the train + val datasets (and the PNA degree histogram) from train_val.csv."""
    df = pd.read_csv(train_val_csv)
    need = {"smiles", "T_K", "P_kPa", "density_kg_m3", "MW_g_mol", "split"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{train_val_csv} missing columns: {missing}")
    if use_vdw and "vdW" not in df.columns:
        raise ValueError(f"{train_val_csv} has no `vdW` column but vdW was requested (--vdw on).")
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"{train_val_csv} must contain both 'train' and 'val' rows in its split column "
            f"(got {len(train_df)} train / {len(val_df)} val).")
    train_ds = ILDensityDataset(train_df, use_vdw=use_vdw)
    val_ds = ILDensityDataset(val_df, use_vdw=use_vdw)

    seen = {}
    for m in train_ds.mols:
        seen.setdefault(m["smiles"], m["data"])
    deg = compute_degree_histogram(list(seen.values()))
    return train_ds, val_ds, deg


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------
def _density_pass(model, big, T, P, rho, MW, vdw, mol_index):
    params = model(big, vdw=vdw)[mol_index]        # [P, 3] or [P, 5] per-point params
    params8 = assemble_pcsaft_params(params)       # [P, 8] (zero-fills any params not predicted)
    rho_pred, valid = mass_density_kg_per_m3(params8, T, P, MW)
    return rho_pred, valid


@torch.no_grad()
def evaluate(model, loader):
    """Return (density MAPE %, fraction of points where FeOs converged)."""
    model.eval()
    preds, tgts, masks = [], [], []
    for big, T, P, rho, MW, vdw, idx in loader:
        rho_pred, valid = _density_pass(model, big, T, P, rho, MW, vdw, idx)
        preds.append(rho_pred); tgts.append(rho); masks.append(valid)
    pred = torch.cat(preds); tgt = torch.cat(tgts); mask = torch.cat(masks)
    return density_mape(pred, tgt, mask), float(mask.double().mean())


def make_model(config, deg):
    return PCSAFTGNN(
        hidden=config["hidden"], depth=config["depth"], conv_type=config["conv_type"],
        predict_association=config.get("predict_association", False),
        use_vdw=config.get("use_vdw", False),
        vdw_mean=config.get("vdw_mean", 0.0), vdw_std=config.get("vdw_std", 1.0),
        use_bounds=config.get("use_bounds", False),
        towers=config.get("towers", 4),
        heads=config.get("heads", 4), deg=deg, positive=True)


def train_model(config, train_ds, val_ds, deg, epochs, report_fn=None, verbose=True,
                seed=0, on_improve=None):
    """Reusable training core. Returns (best_val_mape, model_at_best, history).

    `on_improve(best, model, history)`, if given, is called whenever the val MAPE improves
    (used by the CLI to flush the best checkpoint to disk for crash-safety). Training stops
    early and keeps the best weights if validation collapses (non-finite MAPE / zero FeOs
    coverage); the collapse epoch is not recorded in `history`.
    """
    torch.manual_seed(seed)
    model = make_model(config, deg)
    opt = Adam(model.parameters(), lr=config["lr"])
    sched = CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2, eta_min=1e-6)
    grad_clip = float(config.get("grad_clip", 0.0))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=collate)

    history, best, best_state = [], float("inf"), None
    for epoch in range(epochs):
        model.train()
        run_loss, n_batches, n_skipped = 0.0, 0, 0
        for big, T, P, rho, MW, vdw, idx in train_loader:
            opt.zero_grad()
            rho_pred, valid = _density_pass(model, big, T, P, rho, MW, vdw, idx)
            loss = huber_relative_loss(rho_pred, rho, delta=0.01, mask=valid)
            # A non-finite loss (e.g. FeOs failed to converge on every point in the batch) would
            # write NaNs into the weights on opt.step() and poison the whole run. Skip the step
            # for that batch instead -- a no-op for healthy batches (their loss is always finite).
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run_loss += float(loss.detach()); n_batches += 1
        sched.step()

        train_loss = run_loss / max(n_batches, 1)
        val_mape, conv = evaluate(model, val_loader)

        # Collapse guard: if FeOs no longer converges on any val point (val MAPE non-finite /
        # coverage zero), the run has broken down and will stay broken (NaNs don't heal). Stop
        # now and keep the best pre-collapse weights. This epoch is NOT appended to `history`,
        # so the best state and the loss trace stay clean.
        if (not math.isfinite(val_mape)) or conv <= 0.0:
            if verbose:
                logging.warning(
                    f"epoch {epoch:3d}  COLLAPSE detected (val_MAPE {val_mape}, "
                    f"converged {conv*100:.1f}%"
                    + (f", {n_skipped} non-finite batch(es) skipped this epoch" if n_skipped else "")
                    + "). Stopping early; keeping best val_MAPE "
                    + (f"{best:.2f}%." if math.isfinite(best) else "(none reached)."))
            break

        history.append({"epoch": epoch, "train_loss": train_loss, "val_mape": val_mape, "conv": conv})
        if val_mape < best:
            best = val_mape
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if on_improve is not None:
                on_improve(best, model, history)   # crash-safe flush of the new best
        if verbose:
            msg = (f"epoch {epoch:3d}  train_loss {train_loss:.5f}  "
                   f"val_MAPE {val_mape:6.2f}%  converged {conv*100:5.1f}%")
            if n_skipped:
                msg += f"  (skipped {n_skipped} non-finite batch(es))"
            logging.info(msg)
        if report_fn is not None:
            report_fn({"density_mape": val_mape, "train_loss": train_loss})

    if best_state is not None:
        model.load_state_dict(best_state)
    return best, model, history


# ---------------------------------------------------------------------------
# per-seed pipeline (train -> save -> self-check)
# ---------------------------------------------------------------------------
def train_one_seed(seed, config, train_ds, val_ds, deg, epochs, use_vdw,
                   train_val_csv, out_path, val_loader):
    """Train a single seed, write its checkpoint, and run the end-to-end self-check.

    Returns the best val MAPE. Raises on failure (the caller decides whether to continue)."""

    def build_ckpt(best, model, history):
        return {"model_state": model.state_dict(), "config": config, "deg": deg,
                "predict_association": config["predict_association"], "use_vdw": use_vdw,
                "use_bounds": config.get("use_bounds", False),
                "positive": True, "best_val_mape": best, "history": history,
                "seed": seed, "train_val_csv": os.path.abspath(train_val_csv)}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def on_improve(best, model, history):
        # Flush the current best to disk the moment it improves, so an interrupted or killed
        # run still leaves the best model on disk. A write hiccup here must not abort training.
        try:
            torch.save(build_ckpt(best, model, history), out_path)
        except Exception:
            logging.warning(f"could not flush interim checkpoint to {out_path}", exc_info=True)

    best, model, history = train_model(config, train_ds, val_ds, deg, epochs,
                                       verbose=True, seed=seed, on_improve=on_improve)

    if not history or not math.isfinite(best):
        raise RuntimeError("training collapsed before reaching a single valid epoch; "
                           "no usable checkpoint was written for this seed.")

    # Final authoritative save (best weights already reloaded into `model` by train_model).
    torch.save(build_ckpt(best, model, history), out_path)

    # ---- self-check (per seed) ----
    logging.info(f"--- SELF-CHECK (seed {seed}) ---")
    early_stopped = len(history) < epochs
    losses = [h["train_loss"] for h in history]
    assert all(map(lambda x: x == x and abs(x) != float("inf"), losses)), "non-finite train loss"
    if not early_stopped:
        assert losses[-1] <= losses[0] + 1e-9, "training loss increased overall (diverged)"
    else:
        logging.warning(f"early-stopped after {len(history)}/{epochs} epochs (collapse guard); "
                        f"skipping the monotonic-loss check for this seed.")
    logging.info(f"best val MAPE: {best:.2f}%   (epoch {min(history, key=lambda h: h['val_mape'])['epoch']})")
    logging.info(f"train loss {losses[0]:.5f} -> {losses[-1]:.5f}"
                 + (f"   [{len(history)} epochs, early-stopped]" if early_stopped else ""))

    # checkpoint round-trips and reproduces the same val MAPE
    reloaded = make_model(config, deg)
    reloaded.load_state_dict(torch.load(out_path)["model_state"])
    reloaded_mape, _ = evaluate(reloaded, val_loader)
    assert abs(reloaded_mape - best) < 1e-6, f"reloaded MAPE {reloaded_mape} != best {best}"
    logging.info(f"checkpoint reloads and reproduces val MAPE ({reloaded_mape:.2f}%).")
    logging.info(f"wrote {os.path.abspath(out_path)}")
    return best


# ---------------------------------------------------------------------------
# CLI + GUI self-check
# ---------------------------------------------------------------------------
# vdW settings to sweep, per --vdw. "off" is listed first, matching the GUI's checkbox order.
VDW_SWEEPS = {"off": [False], "on": [True], "both": [False, True]}


def configure_headless(args) -> None:
    """Configure the run from the CLI: the --no-gui path.

    Sets exactly the same five attributes that the GUI's Run button sets -- params, vdws,
    bounds, out_base, seeds -- so everything downstream is identical. This exists because the
    GUI needs a display, which CI runners, plain SSH sessions, HPC batch nodes and containers
    do not have. It changes nothing about training: it only lets the CLI values through in
    place of the widgets.

        python train.py --no-gui --vdw both --bounds on --seeds 0 1 2 3 --epochs 120
    """
    args.params = args.cli_params
    args.vdws = VDW_SWEEPS[args.vdw]
    args.seeds = list(args.cli_seeds) if args.cli_seeds else [args.seed]
    # `--bounds` is already the "off"/"on" string the GUI radio would have written.
    # Default output base mirrors the GUI's update_filename(): checkpoints/gnn_{params}.pt
    args.out_base = (args.cli_out or f"checkpoints/gnn_{args.params}.pt").strip()

    # the same guards the GUI's on_run() enforces, as hard errors instead of message boxes
    if not args.vdws:
        raise SystemExit("no vdW setting selected: --vdw must be off, on, or both.")
    if not args.seeds:
        raise SystemExit("no seeds selected: pass --seeds 0 1 2 3 (or --seed N).")
    if any(s < 0 for s in args.seeds):
        raise SystemExit(f"seeds must be non-negative, got {args.seeds}.")
    if not args.out_base:
        raise SystemExit("--out is empty: pass a checkpoint base name.")


def run_gui(args) -> bool:
    """Configure the run from the Tk widgets: the interactive path, and the default.

    Populates exactly five attributes on `args` -- params, vdws, bounds, out_base, seeds --
    from the widgets, which are authoritative here (the CLI flags only pre-tick them).
    Returns False if the window was closed without clicking Run.

    Needs a display. Use --no-gui (see configure_headless) on CI, over SSH without X
    forwarding, on a batch node, or in a container.
    """
    # --- GUI START ---
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    # root.tk.call('tk', 'scaling', 2.0)  # doubles fonts/widget sizing
    root.title("GNN Training Configuration")
    root.geometry("600x600")

    # GUI Variables
    params_var = tk.StringVar(value="core")
    bounds_var = tk.StringVar(value=args.bounds)             # "off"/"on"; --bounds sets the default
    out_var = tk.StringVar(value="checkpoints/gnn_core.pt")   # BASE name; the sweep appends tags
    user_modified_filename = tk.BooleanVar(value=False)
    preview_var = tk.StringVar()   # live list of files that will be written

    # vdW is sweepable like seeds: one checkbox per setting (off / on). Pre-tick from --vdw.
    vdw_off_var = tk.BooleanVar(value=(args.vdw in ("off", "both")))
    vdw_on_var = tk.BooleanVar(value=(args.vdw in ("on", "both")))
    vdw_vars = [(False, vdw_off_var), (True, vdw_on_var)]     # (use_vdw, checkbox); off listed first

    def selected_vdws():
        """vdW settings to sweep, as use_vdw booleans (False = off, True = on); off first."""
        return [use for use, v in vdw_vars if v.get()]

    # One checkbox per seed (0 .. MAX_SEEDS-1). Pre-tick the CLI --seed if in range, else seed 0.
    default_seed = args.seed if 0 <= args.seed < MAX_SEEDS else 0
    seed_vars = {s: tk.BooleanVar(value=(s == default_seed)) for s in range(MAX_SEEDS)}

    def selected_seeds():
        return [s for s in range(MAX_SEEDS) if seed_vars[s].get()]

    # Dynamically update the BASE output name ONLY if the user hasn't typed a custom one. The base
    # carries just the parameter set; vdW, bounds, and seed tags are appended per checkpoint by
    # checkpoint_path(), so one base fans out over the whole sweep.
    def update_filename(*_):
        if not user_modified_filename.get():
            out_var.set(f"checkpoints/gnn_{params_var.get()}.pt")

    # Live preview of the exact files that will be written (one per vdW setting x seed).
    def refresh_preview(*_):
        vdws = selected_vdws()
        seeds = selected_seeds()
        base = out_var.get().strip()
        use_bounds = (bounds_var.get() == "on")
        if not vdws:
            preview_var.set("(no vdW setting selected)")
        elif not seeds:
            preview_var.set("(no seeds selected)")
        elif not base:
            preview_var.set("(enter an output base name)")
        else:
            preview_var.set("\n".join(
                checkpoint_path(base, s, use_vdw, use_bounds)
                for use_vdw in vdws for s in seeds))

    params_var.trace_add("write", update_filename)
    bounds_var.trace_add("write", refresh_preview)   # bounds changes the _bounded tag in the names
    out_var.trace_add("write", refresh_preview)
    for _, v in vdw_vars:
        v.trace_add("write", refresh_preview)
    for v in seed_vars.values():
        v.trace_add("write", refresh_preview)

    # Styling and Layout
    tk.Label(root, text="Select PC-SAFT Parameters:", font=("Arial", 10, "bold")).pack(pady=(10, 0))
    tk.Radiobutton(root, text="3 Core Parameters (m, sigma, epsilon_k)", variable=params_var, value="core").pack()
    tk.Radiobutton(root, text="5 Parameters (+ kappa_ab, epsilon_k_ab)", variable=params_var, value="assoc").pack()

    tk.Label(root, text="Select van der Waals (vdW) volume (one checkpoint each):",
             font=("Arial", 10, "bold")).pack(pady=(10, 0))
    vdw_frame = tk.Frame(root)
    vdw_frame.pack()
    tk.Checkbutton(vdw_frame, text="vdW off", variable=vdw_off_var).pack(side="left", padx=4)
    tk.Checkbutton(vdw_frame, text="vdW on", variable=vdw_on_var).pack(side="left", padx=4)

    tk.Label(root, text="Bound PC-SAFT parameters to a physical box:", font=("Arial", 10, "bold")).pack(pady=(10, 0))
    tk.Radiobutton(root, text="Unbounded (off) - softplus head", variable=bounds_var, value="off").pack()
    tk.Radiobutton(root, text="Bounded (on) - keeps FeOs converging", variable=bounds_var, value="on").pack()

    tk.Label(root, text="Select seeds to train (one checkpoint each):", font=("Arial", 10, "bold")).pack(pady=(10, 0))
    seed_frame = tk.Frame(root)
    seed_frame.pack()
    for s in range(MAX_SEEDS):
        tk.Checkbutton(seed_frame, text=f"seed {s}", variable=seed_vars[s]).pack(side="left", padx=4)

    tk.Label(root, text="Output base name (sweep appends _vdw/_bounded/_s{n}):",
             font=("Arial", 10, "bold")).pack(pady=(15, 5))

    # Frame for Entry and Browse Button
    file_frame = tk.Frame(root)
    file_frame.pack(fill="x", padx=30)

    out_entry = tk.Entry(file_frame, textvariable=out_var, font=("Consolas", 10))
    out_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

    # Flag to stop auto-updating if the user types in the box manually
    def on_entry_key(event):
        user_modified_filename.set(True)
    out_entry.bind("<Key>", on_entry_key)

    def browse_file():
        filename = filedialog.asksaveasfilename(
            initialdir="checkpoints",
            title="Save Checkpoint Base Name As",
            defaultextension=".pt",
            filetypes=(("PyTorch Models", "*.pt"), ("All Files", "*.*"))
        )
        if filename:
            out_var.set(filename)
            user_modified_filename.set(True)  # Lock the choice so radio buttons don't override it

    browse_btn = tk.Button(file_frame, text="Browse...", command=browse_file, font=("Arial", 9))
    browse_btn.pack(side="right")

    # Preview of files that will be written
    tk.Label(root, text="Files that will be written:", font=("Arial", 9, "italic")).pack(pady=(12, 2))
    tk.Label(root, textvariable=preview_var, font=("Consolas", 9), fg="#333333",
             justify="left", anchor="w").pack(fill="x", padx=30)

    def on_run():
        vdws = selected_vdws()
        seeds = selected_seeds()
        if not vdws:
            messagebox.showwarning("No vdW setting selected",
                                   "Tick at least one vdW setting (off and/or on) to train.")
            return
        if not seeds:
            messagebox.showwarning("No seeds selected", "Tick at least one seed to train.")
            return
        if not out_var.get().strip():
            messagebox.showwarning("No output name", "Enter an output base file name.")
            return
        args.params = params_var.get()
        args.vdws = vdws            # list of use_vdw booleans to sweep (e.g. [False, True])
        args.bounds = bounds_var.get()
        args.out_base = out_var.get().strip()
        args.seeds = seeds
        root.destroy()

    tk.Button(root, text="Run Training", command=on_run, bg="#4CAF50", fg="white",
              font=("Arial", 10, "bold"), width=15).pack(pady=20)

    refresh_preview()   # populate the preview before the first interaction

    # Start the GUI Loop
    root.mainloop()

    # If the user closed the window without clicking 'Run Training', fall back safely.
    if not hasattr(args, "seeds"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Train the physics-in-loop GNN (core PC-SAFT params).")
    ap.add_argument("--train-val-csv", default=DEFAULT_TRAIN_VAL_CSV)
    ap.add_argument("--conv", default="PNA", choices=["PNA", "GATv2", "Transformer"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--towers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0,
                    help="Pre-ticks this seed in the GUI (0..4); the GUI's checkboxes are authoritative there. With --no-gui, trains this seed unless --seeds is given.")
    ap.add_argument("--vdw", choices=["off", "on", "both"], default="off",
                    help="Pre-ticks the vdW checkbox(es) in the GUI (the GUI is authoritative there; this "
                         "flag is authoritative with --no-gui). vdW is sweepable like seeds: 'both' "
                         "trains one checkpoint per setting (off and on) for a like-for-like ablation.")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="Max gradient norm (0 = off). Turn on (e.g. 1.0) if training blows up to NaN. "
                         "Off by default so it doesn't change ongoing experiments' dynamics.")
    ap.add_argument("--bounds", choices=["off", "on"], default="on",
                    help="Pre-selects the bounds radio in the GUI (the GUI is authoritative). A bounded "
                         "head keeps predicted PC-SAFT params inside a physical box so FeOs stays in its "
                         "convergence region (prevents the NaN-density collapse). Authoritative "
                         "with --no-gui.")

    # ---- headless mode -------------------------------------------------------------------
    # The GUI needs a display. These flags let the run be configured entirely from the command
    # line, so training works on CI, over SSH without X forwarding, on a batch node, or in a
    # container. They are read ONLY with --no-gui; otherwise the widgets win, as before.
    ap.add_argument("--no-gui", action="store_true",
                    help="Skip the GUI and configure the run from --params/--seeds/--out/--vdw/"
                         "--bounds. Required to train without a display (CI, SSH, HPC, Docker).")
    ap.add_argument("--params", choices=["core", "assoc"], default="core", dest="cli_params",
                    help="[--no-gui] PC-SAFT parameters the GNN predicts: 'core' = 3 "
                         "(m, sigma, epsilon_k), 'assoc' = 5 (+ kappa_ab, epsilon_k_ab).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, dest="cli_seeds",
                    help="[--no-gui] Seeds to train, one checkpoint each, e.g. --seeds 0 1 2 3 "
                         "(the paper's four). Defaults to the single --seed.")
    ap.add_argument("--out", default=None, dest="cli_out",
                    help="[--no-gui] Checkpoint BASE name; the sweep appends _vdw/_bounded/_s{n}. "
                         "Default: checkpoints/gnn_{params}.pt, matching the GUI.")

    # We parse the core arguments first, then run the GUI to populate params, vdw, seeds, and out.
    args = ap.parse_args()

    # Configure the run: headlessly from the CLI (--no-gui), or interactively from the GUI.
    if args.no_gui:
        configure_headless(args)
    else:
        if not run_gui(args):
            print("Training cancelled by user.")   # standard print: logging isn't set up yet
            return

    # Initialize logging now that we know the user intends to proceed
    log_filename = setup_logging()
    logging.info(f"Logging initialized. Saving logs to: {log_filename}")

    torch.set_num_threads(max(1, os.cpu_count() or 1))   # feos-torch + GNN run on CPU

    n_runs = len(args.vdws) * len(args.seeds)
    logging.info(f"parameter set: {'5 (core + association)' if args.params == 'assoc' else '3 (core)'}")
    logging.info("parameter bounds: "
                 + ("on (physical box, sigmoid-squashed head)" if args.bounds == "on"
                    else "off (softplus head)"))
    logging.info(f"gradient clipping: {'off' if args.grad_clip <= 0 else args.grad_clip}")
    logging.info(f"vdW settings to sweep: {[VDW_LABELS[u] for u in args.vdws]}")
    logging.info(f"seeds to train: {args.seeds}")
    logging.info(f"total checkpoints: {n_runs}  "
                 f"({len(args.vdws)} vdW setting(s) x {len(args.seeds)} seed(s), "
                 f"{args.epochs} epochs each)")

    results = []   # list of (use_vdw, seed, best_val_mape, out_path, ok)
    run_i = 0
    for use_vdw in args.vdws:
        # Datasets, the PNA degree histogram, the vdW standardization stats, the config, and the
        # val loader ALL depend on the vdW setting -- the modeling unit is per-SMILES with vdW off
        # but per-(SMILES, vdW) with vdW on -- so build them once per setting and reuse across that
        # setting's seeds. (Within a setting, only weight-init and batch shuffling use the RNG, both
        # seeded inside train_model, so the seeds legitimately share the same datasets.) This is
        # exactly what a separate CLI run per (vdW, seed) would rebuild.
        logging.info("")
        logging.info(f"############  vdW {VDW_LABELS[use_vdw].upper()}  ############")
        try:
            train_ds, val_ds, deg = build_datasets(args.train_val_csv, use_vdw=use_vdw)
            logging.info(f"modeling units: train {len(train_ds)} | val {len(val_ds)}; "
                         f"deg histogram len {len(deg)}")
            vdw_mean, vdw_std = compute_vdw_stats(train_ds) if use_vdw else (0.0, 1.0)
            config = {"conv_type": args.conv, "hidden": args.hidden, "depth": args.depth,
                      "towers": args.towers, "heads": args.heads, "lr": args.lr,
                      "batch_size": args.batch_size, "grad_clip": args.grad_clip,
                      "predict_association": (args.params == "assoc"),
                      "use_vdw": use_vdw, "vdw_mean": vdw_mean, "vdw_std": vdw_std,
                      "use_bounds": (args.bounds == "on")}
            logging.info(f"vdW input: {'on' if use_vdw else 'off'}"
                         + (f"  (train mean {vdw_mean:.2f}, std {vdw_std:.2f} A^3)" if use_vdw else ""))
            logging.info(f"config: {config}")
            # Val loader is seed-independent within a setting; build it once for the self-checks.
            val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=collate)
        except Exception:
            # e.g. --vdw on but the CSV has no `vdW` column. Don't kill the other setting's runs;
            # mark this setting's seeds as failed and move on so the summary still prints.
            logging.exception(f"vdW {VDW_LABELS[use_vdw]}: setup failed; "
                              f"skipping all {len(args.seeds)} seed(s) for this setting")
            for seed in args.seeds:
                run_i += 1
                out_path = checkpoint_path(args.out_base, seed, use_vdw, args.bounds == "on")
                results.append((use_vdw, seed, float("nan"), out_path, False))
            continue

        for seed in args.seeds:
            run_i += 1
            out_path = checkpoint_path(args.out_base, seed, use_vdw, config["use_bounds"])
            logging.info("")
            logging.info(f"========  vdW {VDW_LABELS[use_vdw]} | SEED {seed}  "
                         f"({run_i}/{n_runs})  ========")
            logging.info(f"output -> {os.path.abspath(out_path)}")
            try:
                best = train_one_seed(seed, config, train_ds, val_ds, deg, args.epochs,
                                      use_vdw, args.train_val_csv, out_path, val_loader)
                results.append((use_vdw, seed, best, out_path, True))
            except Exception:
                # A failed run shouldn't waste the ones that already trained; log loudly, continue.
                logging.exception(f"vdW {VDW_LABELS[use_vdw]} seed {seed} FAILED "
                                  f"(continuing with remaining runs)")
                results.append((use_vdw, seed, float("nan"), out_path, False))

    # ---- summary (grouped by vdW setting; mean +/- std over that setting's seeds) ----
    logging.info("")
    logging.info("================  SUMMARY  ================")
    for use_vdw in args.vdws:
        rows = [(seed, best, out_path, ok)
                for (u, seed, best, out_path, ok) in results if u == use_vdw]
        logging.info("")
        logging.info(f"vdW {VDW_LABELS[use_vdw]}:")
        for seed, best, out_path, ok in rows:
            if ok:
                logging.info(f"  seed {seed}:  val MAPE {best:6.2f}%   {out_path}")
            else:
                logging.info(f"  seed {seed}:  FAILED               {out_path}")
        good = [best for _, best, _, ok in rows if ok]
        if good:
            mean = statistics.mean(good)
            std = statistics.stdev(good) if len(good) > 1 else 0.0
            logging.info(f"  -> val MAPE across {len(good)} seed(s): {mean:.2f}% +/- {std:.2f}% "
                         f"(mean +/- sample std)")
        else:
            logging.info("  -> no seeds completed for this setting.")

    if any(ok for *_, ok in results):
        logging.info("")
        logging.info("The headline vdW ablation belongs on the held-out test set via evaluate.py "
                     "(cluster-bootstrap at the molecule level), not these per-setting val numbers.")
    else:
        logging.error("No runs completed successfully.")


if __name__ == "__main__":
    main()