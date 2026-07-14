#!/usr/bin/env python3
"""
evaluate.py  --  Group D (2/2) of the GNN / PC-SAFT density factory.

Loads the trained GNN checkpoint, predicts on the held-out TEST molecules (never seen in
training), and compares it against the Habicht-style baselines on liquid density:

  GNN  : graph [+ vdW] -> PC-SAFT params -> FeOs -> density   (physics-in-the-loop)
         core (m, sigma, epsilon_k) or core + association (+ kappa_ab, epsilon_k_ab), and with or
         without the vdW input -- whichever the checkpoint was trained with; both are read
         automatically from its `config`. The baselines below are fit with the SAME vdW setting,
         so the comparison is on equal inputs.
  RF   : RandomForest on fingerprint + descriptors + (T, P)[ + vdW] -> density
  MLP  : ECFP-MLP, same features -> density

Data layout
-----------
Test data now lives in one or more dedicated files (default: data/test1.csv and data/test2.csv).
Each test file is evaluated on its OWN, and then all of them together as a "combined" set, so
you get three metric tables: test1, test2, combined. The baselines are fit on the rows tagged
"train" inside the train+val file (default: data/train_val.csv) -- the same molecules the GNN
trained on. Locations are the DEFAULT_* constants below; override on the CLI with
--train-val-csv / --test-csv.

Outputs (in --outdir, one set of artifacts per label in {test1, test2, combined}):
  - comparison_<label>.csv                : point MAPE, molecule-level MAPE (the paper's two-level
                                            averaging), AAD, bias, RMSE, and FeOs coverage
  - parity_<label>.png                    : predicted vs experimental density for all three methods
  - density_vs_T_<label>.png              : experimental points vs the GNN curve for a few example ILs
  - test_predictions_<label>.csv          : per-row predictions for all three methods
  - comparison_all.csv                    : every table stacked, indexed by (test_set, method)
  - terminal_evaluate_<checkpoint_name>.out: Log of the terminal output generated during the run

Run:
  python evaluate.py --train-val-csv data/train_val.csv \
      --test-csv data/test1.csv data/test2.csv \
      --checkpoint checkpoints/gnn_core.pt
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import smiles_to_data
from train import make_model
from pcsaft import assemble_pcsaft_params, mass_density_kg_per_m3
from baselines import fit_baselines, predict_baselines, load_train_split


# --- Custom Logger to redirect print statements to both terminal and a file ---
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensures output is written to the file in real-time

    def flush(self):
        self.terminal.flush()
        self.log.flush()
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# data locations  (edit here, or override on the CLI with --train-val-csv / --test-csv)
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_VAL_CSV = "data/train_val.csv"
DEFAULT_TEST_CSVS = ["data/test1.csv", "data/test2.csv"]


def load_gnn(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(ckpt["config"], ckpt["deg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def gnn_predict(model, df):
    """GNN density per test row (one forward per modeling unit, expanded to its points).

    The unit is the molecule, or the (molecule, vdW) pair when the model uses vdW -- so DES-type
    mixtures that share a SMILES but differ in vdW get distinct predictions.
    """
    use_vdw = getattr(model, "use_vdw", False)
    if use_vdw and "vdW" not in df.columns:
        raise ValueError("the checkpoint uses vdW but this test CSV has no `vdW` column.")
    pred = np.full(len(df), np.nan)
    keys = ["smiles", "vdW"] if use_vdw else ["smiles"]
    for _, sub in df.groupby(keys, sort=False):
        smiles = sub["smiles"].iloc[0]
        big = Batch.from_data_list([smiles_to_data(smiles)])
        vdw = torch.tensor([float(sub["vdW"].iloc[0])], dtype=torch.float64) if use_vdw else None
        params = model(big, vdw=vdw)                            # [1, 3] or [1, 5]
        params8 = assemble_pcsaft_params(params.repeat(len(sub), 1))
        T = torch.tensor(sub.T_K.to_numpy(), dtype=torch.float64)
        P = torch.tensor(sub.P_kPa.to_numpy(), dtype=torch.float64)
        MW = torch.tensor(sub.MW_g_mol.to_numpy(), dtype=torch.float64)
        rho, valid = mass_density_kg_per_m3(params8, T, P, MW)
        out = rho.detach().numpy()
        out[~valid.numpy()] = np.nan
        pred[[df.index.get_loc(i) for i in sub.index]] = out
    return pred


def predict_all(test_df, model, base_models):
    """Return a copy of test_df with gnn / rf / mlp density columns added."""
    out = test_df.copy()
    out["gnn"] = gnn_predict(model, out)
    preds = predict_baselines(base_models, out)
    out["rf"] = preds["RandomForest"]
    out["mlp"] = preds["ECFP-MLP"]
    return out


def metrics(exp, pred, groups):
    """Point + molecule-level MAPE, AAD, bias, RMSE, coverage. NaN predictions are dropped."""
    exp = np.asarray(exp, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(pred)
    e, p, g = exp[mask], pred[mask], groups[mask]
    ape = np.abs(p - e) / e * 100.0
    mol_mape = pd.Series(ape).groupby(pd.Series(g)).mean().mean()
    return {
        "point_MAPE_%": ape.mean(),
        "mol_MAPE_%": mol_mape,
        "AAD_kg/m3": np.abs(p - e).mean(),
        "bias_kg/m3": (p - e).mean(),
        "RMSE_kg/m3": np.sqrt(((p - e) ** 2).mean()),
        "coverage_%": 100.0 * mask.mean(),
    }


def parity_plot(test_df, methods, path):
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.6), squeeze=False)
    lo = test_df["density_kg_m3"].min() * 0.97
    hi = test_df["density_kg_m3"].max() * 1.03
    for ax, (name, col) in zip(axes[0], methods.items()):
        sub = test_df[np.isfinite(test_df[col])]
        ax.scatter(sub["density_kg_m3"], sub[col], s=9, alpha=0.5, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
        mape = np.mean(np.abs(sub[col] - sub["density_kg_m3"]) / sub["density_kg_m3"]) * 100
        ax.set_title(f"{name}  (MAPE {mape:.2f}%)")
        ax.set_xlabel("experimental density [kg/m$^3$]")
        ax.set_ylabel("predicted density [kg/m$^3$]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def tcurve_plot(test_df, gnn_col, path, n=3):
    counts = test_df.groupby("smiles").size().sort_values(ascending=False)
    chosen = counts.head(n).index.tolist()
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.6 * len(chosen), 4.2), squeeze=False)
    for ax, smiles in zip(axes[0], chosen):
        sub = test_df[test_df.smiles == smiles].sort_values("T_K")
        ax.scatter(sub["T_K"], sub["density_kg_m3"], s=18, label="experimental", color="k")
        ax.plot(sub["T_K"], sub[gnn_col], "-", label="GNN", color="tab:blue")
        name = sub["name"].iloc[0] if "name" in sub else smiles
        ax.set_title((str(name)[:34]) if isinstance(name, str) else smiles, fontsize=9)
        ax.set_xlabel("T [K]"); ax.set_ylabel("density [kg/m$^3$]"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report_set(label, aug_df, method_cols, outdir):
    """Compute the metrics table for one already-predicted set, write its artifacts, return it."""
    table = pd.DataFrame({name: metrics(aug_df["density_kg_m3"], aug_df[col], aug_df["smiles"])
                          for name, col in method_cols.items()}).T
    table = table[["point_MAPE_%", "mol_MAPE_%", "AAD_kg/m3", "bias_kg/m3", "RMSE_kg/m3", "coverage_%"]]

    print(f"\n=== TEST-SET DENSITY COMPARISON [{label}] "
          f"({aug_df['smiles'].nunique()} molecules / {len(aug_df)} points) ===")
    print(table.to_string())

    table.to_csv(os.path.join(outdir, f"comparison_{label}.csv"))
    parity_plot(aug_df, method_cols, os.path.join(outdir, f"parity_{label}.png"))
    tcurve_plot(aug_df, "gnn", os.path.join(outdir, f"density_vs_T_{label}.png"))
    aug_df.to_csv(os.path.join(outdir, f"test_predictions_{label}.csv"), index=False)
    return table


def run_gui(args) -> bool:
    """Configure the run from the Tk widgets: the interactive path, and the default.

    The widgets only ever overwrite four attributes that argparse already defines --
    train_val_csv, test_csv, checkpoint and outdir -- so --no-gui simply skips this and uses
    the command line as given. Returns False if the window was closed without clicking Run.

    Needs a display. Use --no-gui on CI, over SSH without X forwarding, on a batch node, or in
    a container.
    """
    # --- GUI START ---
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("GNN Evaluation Configuration")
    root.geometry("600x300")
    
    # GUI Variables
    train_var = tk.StringVar(value=args.train_val_csv)
    test_var = tk.StringVar(value=" ".join(args.test_csv))
    ckpt_var = tk.StringVar(value=args.checkpoint)
    outdir_var = tk.StringVar(value=args.outdir)

    def browse_file(var, title, filetypes):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(path)

    def browse_files(var, title, filetypes):
        paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
        if paths:
            var.set(" ".join(paths))

    def browse_checkpoint(var, title, filetypes):
        """Browse for a checkpoint and automatically update the output directory."""
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(path)
            # Update output directory based on the selected checkpoint's filename
            new_ckpt_name = os.path.splitext(os.path.basename(path))[0]
            outdir_var.set(os.path.join("results", new_ckpt_name))

    def browse_dir(var, title):
        path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)
            
    # Layout Frame
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)

    # 1. Train/Val CSV
    tk.Label(frame, text="Train/Val CSV:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="e", pady=10, padx=5)
    tk.Entry(frame, textvariable=train_var, width=50).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_file(train_var, "Select Train/Val CSV", [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=0, column=2)

    # 2. Test CSV(s)
    tk.Label(frame, text="Test CSV(s):", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="e", pady=10, padx=5)
    tk.Entry(frame, textvariable=test_var, width=50).grid(row=1, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_files(test_var, "Select Test CSV(s)", [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=1, column=2)

    # 3. Checkpoint (Uses customized browse_checkpoint to auto-update OutDir)
    tk.Label(frame, text="Checkpoint (.pt):", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="e", pady=10, padx=5)
    tk.Entry(frame, textvariable=ckpt_var, width=50).grid(row=2, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_checkpoint(ckpt_var, "Select Checkpoint", [("PyTorch Models", "*.pt"), ("All Files", "*.*")])).grid(row=2, column=2)

    # 4. Output Directory
    tk.Label(frame, text="Output Directory:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="e", pady=10, padx=5)
    tk.Entry(frame, textvariable=outdir_var, width=50).grid(row=3, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_dir(outdir_var, "Select Output Directory")).grid(row=3, column=2)

    # State flag to ensure the user clicked run
    run_flag = [False]

    def on_run():
        args.train_val_csv = train_var.get()
        args.test_csv = test_var.get().split()  # Convert back to list of strings
        args.checkpoint = ckpt_var.get()
        args.outdir = outdir_var.get()
        run_flag[0] = True
        root.destroy()
        
    tk.Button(root, text="Run Evaluation", command=on_run, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=20).pack(pady=10)
    
    # Start the GUI Loop
    root.mainloop()
    
    # Check if the user closed the window without running
    if not run_flag[0]:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Evaluate the GNN vs baselines on held-out test ILs.")
    ap.add_argument("--train-val-csv", default=DEFAULT_TRAIN_VAL_CSV,
                    help="CSV with train+val rows; baselines fit on its split=='train' rows.")
    ap.add_argument("--test-csv", nargs="+", default=list(DEFAULT_TEST_CSVS),
                    help="One or more test CSVs; each is scored separately, plus all combined.")
    ap.add_argument("--checkpoint", default="checkpoints/gnn_core.pt")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--ecfp-bits", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gui", action="store_true",
                    help="Skip the GUI and use the flags above. Required without a display "
                         "(CI, SSH, HPC, Docker).")
    args = ap.parse_args()

    # Dynamically set the default output directory based on the checkpoint name
    if args.outdir == "results":
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.outdir = os.path.join("results", ckpt_name)

    # Configure the run: headlessly from the CLI (--no-gui), or interactively from the GUI.
    if not args.no_gui:
        if not run_gui(args):
            print("Evaluation cancelled by user.")
            return

    os.makedirs(args.outdir, exist_ok=True)
    
    # --- Intercept terminal output and log it to the output directory ---
    # Extract filename without extension from the finalized args.checkpoint path
    ckpt_file_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    log_path = os.path.join(args.outdir, f"terminal_evaluate_{ckpt_file_name}.out")
    sys.stdout = Logger(log_path)
    print(f"Terminal output is also being saved to: {log_path}\n")
    # --------------------------------------------------------------------

    pd.set_option("display.width", 160, "display.float_format", lambda v: f"{v:8.3f}")

    # GNN
    model, ckpt = load_gnn(args.checkpoint)
    n_params = 5 if ckpt["config"].get("predict_association", False) else 3
    use_vdw = ckpt["config"].get("use_vdw", False)
    # Epoch at which the best val MAPE was reached. It isn't stored as its own checkpoint key, but the
    # per-epoch `history` the checkpoint carries lets us recover it -- the first epoch that hit the
    # minimum val MAPE, exactly as train.py selects `best` (strict `<`) and reports in its self-check.
    # 0-indexed, matching train.py's logs. Missing/empty history (very old checkpoints) -> unknown.
    history = ckpt.get("history", [])
    best_epoch = min(history, key=lambda h: h["val_mape"])["epoch"] if history else None
    best_epoch_str = f", epoch {best_epoch}" if best_epoch is not None else ""
    print(f"loaded checkpoint: {ckpt['config']}  "
          f"(best val MAPE {ckpt.get('best_val_mape', float('nan')):.2f}%{best_epoch_str})")
    print(f"GNN parameter set: {n_params} ({'core + association' if n_params == 5 else 'core'})")
    print(f"GNN vdW input: {'on' if use_vdw else 'off'}")

    # baselines fit on the GNN's training molecules (split=='train' in the train+val file).
    # They mirror the checkpoint's vdW setting so the GNN and the baselines see the same inputs.
    train_df = load_train_split(args.train_val_csv)
    models = fit_baselines(train_df, ecfp_bits=args.ecfp_bits, seed=args.seed, use_vdw=use_vdw)
    print(f"baselines fit on {len(train_df)} train rows / {train_df.smiles.nunique()} molecules "
          f"(vdW feature {'on' if use_vdw else 'off'}, matching the GNN)")

    method_cols = {"GNN (PC-SAFT)": "gnn", "RandomForest": "rf", "ECFP-MLP": "mlp"}

    # predict on each test file once (the GNN forward is the costly part), then concat for combined
    augmented = {}
    for path in args.test_csv:
        label = os.path.splitext(os.path.basename(path))[0]
        tdf = pd.read_csv(path).reset_index(drop=True)
        n_units = tdf.groupby(["smiles", "vdW"]).ngroups if use_vdw and "vdW" in tdf.columns \
            else tdf["smiles"].nunique()
        unit = "(smiles,vdW) units" if use_vdw else "molecules"
        print(f"test set [{label}]: {n_units} {unit} / {len(tdf)} points")
        augmented[label] = predict_all(tdf, model, models)
    combined = pd.concat(augmented.values(), ignore_index=True)

    # per-file tables + the combined table
    tables = {label: report_set(label, adf, method_cols, args.outdir)
              for label, adf in augmented.items()}
    tables["combined"] = report_set("combined", combined, method_cols, args.outdir)

    # one stacked summary across all sets, indexed by (test_set, method)
    stacked = []
    for label, tbl in tables.items():
        t = tbl.copy()
        t.insert(0, "test_set", label)
        t.index.name = "method"
        stacked.append(t.reset_index())
    summary = pd.concat(stacked, ignore_index=True).set_index(["test_set", "method"])
    summary_path = os.path.join(args.outdir, "comparison_all.csv")
    summary.to_csv(summary_path)

    # ---- self-check ----
    print("\n=== SELF-CHECK ===")
    expected = list(augmented) + ["combined"]
    assert set(tables) == set(expected), "missing a test set in the tables"
    for label, tbl in tables.items():
        assert set(tbl.index) == set(method_cols), f"[{label}] missing a method"
        assert np.isfinite(tbl[["point_MAPE_%", "mol_MAPE_%", "AAD_kg/m3", "RMSE_kg/m3"]].to_numpy()).all(), \
            f"[{label}] non-finite metric"
        for fn in (f"comparison_{label}.csv", f"parity_{label}.png",
                   f"density_vs_T_{label}.png", f"test_predictions_{label}.csv"):
            assert os.path.exists(os.path.join(args.outdir, fn)), f"missing output {fn}"
    assert os.path.exists(summary_path), "missing comparison_all.csv"
    print(f"all outputs written to {os.path.abspath(args.outdir)}")
    for label, tbl in tables.items():
        best = tbl["point_MAPE_%"].idxmin()
        print(f"  [{label:8s}] lowest point MAPE: {best} ({tbl.loc[best, 'point_MAPE_%']:.2f}%)")
    print("evaluate.py self-check passed.")


if __name__ == "__main__":
    main()