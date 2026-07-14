#!/usr/bin/env python3
"""
export_params.py  --  materialize the GNN's PC-SAFT parameters as a per-molecule table.

The trained GNN predicts the PC-SAFT pure-component parameters from molecular structure alone
-- the 3 core params (m, sigma, epsilon_k), or all 5 (+ kappa_ab, epsilon_k_ab) if the
checkpoint was trained with `--params assoc`. They do not depend on temperature or pressure, and
training does not store them as a table. They are simply the model's output: this script runs
one forward pass per molecule and writes the parameter table that the literature usually
reports. The core/association mode is read from the checkpoint `config`, so the same command
covers both.

van der Waals volume (vdW)
--------------------------
If the checkpoint was trained with `--vdw on` (config `use_vdw=True`), the molecule's van der
Waals volume [cubic angstrom] is a MODEL INPUT and the predicted parameters depend on it. That
volume is NOT stored per molecule in the checkpoint -- it lives in the input CSV's `vdW` column
(the same source training read it from); the checkpoint only carries the train-set
standardization mean/std and the `use_vdw` flag, both applied inside the model. So for a
vdW-trained checkpoint this script reads `vdW` from the CSV, feeds it through the forward pass
exactly as in training, and -- because the same SMILES can carry several vdW values (deep-eutectic
mixtures), each giving different parameters -- the modeling unit becomes (smiles, vdW): one row
per distinct pair. The written table gains a `vdW` column right after `smiles`. Core (non-vdW)
checkpoints are unchanged (one row per molecule); a `vdW` column present in the CSV is still
carried through as metadata but does not affect the parameters.

Output CSVs (one row per molecule, or per (smiles, vdW) unit for a vdW-trained checkpoint):
  - results/<checkpoint_name>/pcsaft_params_train_val.csv
  - results/<checkpoint_name>/pcsaft_params_test1.csv
  - results/<checkpoint_name>/pcsaft_params_test2.csv
"""
# Run: python export_params.py --train-val-csv data/train_val.csv --test-csv data/test1.csv data/test2.csv --checkpoint checkpoints/gnn_core.pt --outdir results

from __future__ import annotations
import argparse
import atexit
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from model import smiles_to_data
from train import make_model
from pcsaft import assemble_pcsaft_params, PARAM_NAMES

# ---------------------------------------------------------------------------
# data locations  (edit here, or override on the CLI with --train-val-csv / --test-csv)
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_VAL_CSV = "data/train_val.csv"
DEFAULT_TEST_CSVS = ["data/test1.csv", "data/test2.csv"]


# ---------------------------------------------------------------------------
# terminal logging: mirror everything printed here to a file too (like `tee`)
# ---------------------------------------------------------------------------
class _Tee:
    """Write to the real stream and a log file at once, so console output is also
    captured verbatim on disk."""
    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()


def _tee_terminal_output(log_path):
    """Duplicate stdout+stderr into `log_path` for the rest of the process.

    Stream restoration and file close are registered with atexit, so a normal return
    or an exception both leave a complete, flushed log behind."""
    logfile = open(log_path, "w", encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, logfile)
    sys.stderr = _Tee(orig_err, logfile)

    def _restore():
        sys.stdout, sys.stderr = orig_out, orig_err
        logfile.flush()
        logfile.close()

    atexit.register(_restore)
    print(f"Terminal output is also being saved to: {log_path}")


@torch.no_grad()
def predict_params(model, smiles_list, vdw_list=None):
    """Return an [N, 8] array of PC-SAFT parameters, one row per (molecule[, vdW]).

    When `vdw_list` is given (checkpoint trained with vdW), each molecule's van der Waals volume
    [cubic angstrom] is fed to the model exactly as in training -- as a raw value the model
    standardizes with the train-set mean/std baked into the checkpoint -- so the predicted
    parameters match what the trained model actually produces. The same SMILES with different vdW
    values then yields different parameters (e.g. deep-eutectic mixtures). With no `vdw_list` the
    call is vdW-free (core / non-vdW checkpoints), identical to before.
    """
    rows = []
    for i, smiles in enumerate(smiles_list):
        big = Batch.from_data_list([smiles_to_data(smiles)])
        if vdw_list is None:
            core = model(big)                      # [1, 3] core, or [1, 5] core+association
        else:
            vdw = torch.tensor([float(vdw_list[i])], dtype=torch.float64)   # [1], one per graph
            core = model(big, vdw=vdw)             # feos-torch needs float64; model standardizes it
        params8 = assemble_pcsaft_params(core)[0]  # [8] full feos-torch vector
        rows.append(params8.numpy())
    return np.asarray(rows)


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
    root.title("GNN Parameter Export Configuration")
    root.geometry("650x300")
    
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

    # 3. Checkpoint (Uses customized browse_checkpoint to auto-update target folder)
    tk.Label(frame, text="Checkpoint (.pt):", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="e", pady=10, padx=5)
    tk.Entry(frame, textvariable=ckpt_var, width=50).grid(row=2, column=1, padx=5)
    tk.Button(frame, text="Browse", command=lambda: browse_checkpoint(ckpt_var, "Select Checkpoint", [("PyTorch Models", "*.pt"), ("All Files", "*.*")])).grid(row=2, column=2)

    # 4. Output Directory (Explicitly renamed to convey it handles all 3 output files)
    tk.Label(frame, text="Output Directory (All CSVs):", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="e", pady=10, padx=5)
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
        
    tk.Button(root, text="Run Parameter Export", command=on_run, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=20).pack(pady=10)
    
    # Start the GUI Loop
    root.mainloop()
    
    # Check if the user closed the window without running
    if not run_flag[0]:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Export GNN-predicted PC-SAFT parameters per molecule split by input files.")
    ap.add_argument("--train-val-csv", default=DEFAULT_TRAIN_VAL_CSV,
                    help="CSV with the train+val molecules (a `split` column tags train/val).")
    ap.add_argument("--test-csv", nargs="+", default=list(DEFAULT_TEST_CSVS),
                    help="Test CSVs; their molecules are included in the table with split tags.")
    ap.add_argument("--checkpoint", default="checkpoints/gnn_core.pt")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-gui", action="store_true",
                    help="Skip the GUI and use the flags above. Required without a display "
                         "(CI, SSH, HPC, Docker).")
    args = ap.parse_args()

    # Dynamically set the default output directory based on the default checkpoint name
    if args.outdir == "results":
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.outdir = os.path.join("results", ckpt_name)

    # Configure the run: headlessly from the CLI (--no-gui), or interactively from the GUI.
    if not args.no_gui:
        if not run_gui(args):
            print("Export cancelled by user.")
            return

    os.makedirs(args.outdir, exist_ok=True)

    # Extract the base name of the checkpoint file (without extension)
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # Mirror everything printed below into results/<checkpoint_name>/terminal_params_<checkpoint_name>.out
    _tee_terminal_output(os.path.join(args.outdir, f"terminal_params_{checkpoint_name}.out"))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(ckpt["config"], ckpt["deg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Map each target file to its respective updated output filename logic
    export_jobs = [
        (args.train_val_csv, os.path.join(args.outdir, "pcsaft_params_train_val.csv"))
    ]
    for test_path in args.test_csv:
        filename = os.path.basename(test_path)
        export_jobs.append((test_path, os.path.join(args.outdir, f"pcsaft_params_{filename}")))

    assoc = ckpt["config"].get("predict_association", False)
    # Whether the checkpoint consumes vdW as an input. If so, vdW is read per molecule from the CSV
    # and fed through the model (the checkpoint only holds the standardization stats, not the values).
    use_vdw = ckpt["config"].get("use_vdw", False)

    # Epoch at which the best val MAPE was reached. It isn't stored as its own checkpoint key, but the
    # per-epoch `history` the checkpoint carries lets us recover it -- the first epoch that hit the
    # minimum val MAPE, exactly as train.py selects `best` (strict `<`) and reports in its self-check.
    # 0-indexed, matching train.py's logs. Missing/empty history (very old checkpoints) -> unknown.
    history = ckpt.get("history", [])
    best_epoch = min(history, key=lambda h: h["val_mape"])["epoch"] if history else None

    # Process and evaluate each file individually
    for src_csv, dest_csv in export_jobs:
        df = pd.read_csv(src_csv)
        has_vdw = "vdW" in df.columns

        # The per-molecule vdW volume is a MODEL INPUT read from the CSV (column 2 in the provided
        # files) -- it is NOT stored per molecule in the checkpoint, which carries only the train-set
        # standardization mean/std and the use_vdw flag (both applied inside the model). A vdW-trained
        # checkpoint therefore cannot reproduce its parameters without this column.
        if use_vdw and not has_vdw:
            raise ValueError(
                f"{src_csv} has no `vdW` column, but the checkpoint was trained with vdW "
                f"(config['use_vdw']=True). The van der Waals volume is a per-molecule model input "
                f"that lives in the CSV, not the checkpoint, so the parameters can't be reproduced "
                f"without it. Re-export with the vdW-bearing CSVs, or use a core (non-vdW) checkpoint.")

        # With vdW on, the modeling unit is (smiles, vdW): the same SMILES can carry several vdW
        # values (deep-eutectic mixtures), each giving different parameters. Dedup on the same key
        # the GNN trained with so every distinct unit gets its own row; core is one row per SMILES.
        dedup_keys = ["smiles", "vdW"] if use_vdw else ["smiles"]
        mol = df.drop_duplicates(dedup_keys).copy()

        # Metadata columns, with `vdW` placed right after `smiles` whenever the CSV provides it.
        meta_cols = ["smiles"] + (["vdW"] if has_vdw else []) + ["name", "split", "MW_g_mol"]
        for col in meta_cols:
            if col not in mol.columns:
                mol[col] = np.nan
        mol = mol[meta_cols].reset_index(drop=True)

        # Feed vdW to the model exactly as in training when the checkpoint uses it; otherwise the
        # forward pass is vdW-free and any passthrough vdW column is metadata the model ignores.
        vdw_arg = mol["vdW"].tolist() if use_vdw else None
        params8 = predict_params(model, mol["smiles"].tolist(), vdw_arg)
        for j, pname in enumerate(PARAM_NAMES):
            mol[pname] = params8[:, j]

        mol.to_csv(dest_csv, index=False)

        # ---- self-check per file ----
        print(f"\n== export_params.py [{os.path.basename(dest_csv)}] ==")
        print(f"parameter set: {'5 (core + association)' if assoc else '3 (core)'}")
        vdw_mode = ("on -- fed to the model; unit = (smiles, vdW)" if use_vdw
                    else "carried through as metadata (core checkpoint ignores it)" if has_vdw
                    else "off")
        print(f"vdW: {vdw_mode}")
        n_label = "(smiles, vdW) rows" if use_vdw else "molecules"
        best_epoch_str = f", epoch {best_epoch}" if best_epoch is not None else ""
        print(f"{n_label}: {len(mol)}   (checkpoint best val MAPE "
              f"{ckpt.get('best_val_mape', float('nan')):.2f}%{best_epoch_str})")
        pd.set_option("display.width", 160, "display.max_colwidth", 36)
        cols = (["name"] + (["vdW"] if has_vdw else []) + ["m", "sigma", "epsilon_k"]
                + (["kappa_ab", "epsilon_k_ab"] if assoc else []))
        print(mol[cols].head(8).to_string(index=False))

        if use_vdw:
            assert mol["vdW"].notna().all(), f"NaN vdW input in {dest_csv}"
            assert (mol["vdW"] > 0).all(), f"non-positive vdW input in {dest_csv}"

        core = mol[["m", "sigma", "epsilon_k"]]
        assert core.notna().all().all(), f"NaN parameter encountered in {dest_csv}"
        assert (core > 0).all().all(), f"non-positive parameter encountered in {dest_csv}"
        assert mol["m"].between(0, 100).all(), f"m outside sane (0,100] in {dest_csv}"
        assert mol["sigma"].between(1.0, 10.0).all(), f"sigma outside sane [1,10] A in {dest_csv}"
        assert mol["epsilon_k"].between(10.0, 1000.0).all(), f"epsilon_k outside sane [10,1000] K in {dest_csv}"
        print(f"ranges: m {mol['m'].min():.2f}-{mol['m'].max():.2f}   "
              f"sigma {mol['sigma'].min():.2f}-{mol['sigma'].max():.2f} A   "
              f"epsilon_k {mol['epsilon_k'].min():.1f}-{mol['epsilon_k'].max():.1f} K")

        assoc_cols = mol[["kappa_ab", "epsilon_k_ab", "na", "nb"]]
        if assoc:
            ab = mol[["kappa_ab", "epsilon_k_ab"]]
            assert ab.notna().all().all(), f"NaN association parameter in {dest_csv}"
            assert (ab > 0).all().all(), f"non-positive association parameter in {dest_csv}"
            assert mol["kappa_ab"].between(0.0, 1.0).all(), f"kappa_ab outside sane (0,1] in {dest_csv}"
            assert mol["epsilon_k_ab"].between(100.0, 6000.0).all(), f"epsilon_k_ab outside sane [100,6000] K in {dest_csv}"
            assert (mol["na"] == 1).all() and (mol["nb"] == 1).all(), f"expected 2B scheme (na=nb=1) in {dest_csv}"
            print(f"         kappa_ab {mol['kappa_ab'].min():.4f}-{mol['kappa_ab'].max():.4f}   "
                  f"epsilon_k_ab {mol['epsilon_k_ab'].min():.0f}-{mol['epsilon_k_ab'].max():.0f} K")
        else:
            assert (assoc_cols == 0).all().all(), f"core-only checkpoint must have zeroed association columns in {dest_csv}"
        print(f"wrote {os.path.abspath(dest_csv)}")


if __name__ == "__main__":
    main()