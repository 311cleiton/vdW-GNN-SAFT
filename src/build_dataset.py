#!/usr/bin/env python3
"""
build_dataset.py  --  Group A of the GNN / PC-SAFT density factory.

Builds a *faithful*, reproducible ionic-liquid liquid-density dataset by pulling
experimental data from the NIST ILThermo 2.0 database via ILThermoPy, validating
chemistry with RDKit, and writing a frozen CSV with molecule-level train/val/test
splits. Everything downstream (graphs, training, evaluation) reads this CSV.

Data provenance & units (confirmed against ILThermoPy 1.1.2 / ILThermo 2.0):
  - Source            : pure-compound (n_compounds=1) datasets with property "Density".
  - Per-point columns : read *dynamically* from each entry's header. We extract
                        Temperature [K], Pressure [kPa], Specific (mass) density [kg/m^3].
  - SMILES            : provided by ILThermoPy as a dot-separated ion pair, e.g.
                        [BMIM][BF4] -> "CCCC[n+]1ccn(C)c1.F[B-](F)(F)F".

Output CSV schema (one row per experimental (molecule, T, P) point):
  smiles, name, ilt_id ((ILThermo entry ID)), reference, T_K, P_kPa, density_kg_m3, MW_g_mol, n_frags, split
"""

from __future__ import annotations
import argparse
import logging
import os
import pickle
import random
import re
import sys
import time
import threading
from dataclasses import dataclass

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")  # we handle parse failures ourselves

import ilthermopy as ilt

# GUI Imports
import tkinter as tk
from tkinter import ttk, messagebox

log = logging.getLogger("build_dataset")

# Pressure unit -> kPa
PRESSURE_TO_KPA = {"kpa": 1.0, "mpa": 1000.0, "pa": 0.001, "bar": 100.0, "hpa": 0.1, "atm": 101.325}


def parse_label(label: str):
    """Split an ILThermo header label 'Quantity, Unit => Phase' into (quantity_lower, unit_clean)."""
    main = label.split("=>")[0]
    parts = [p.strip() for p in main.split(",")]
    quantity = parts[0].lower() if parts else ""
    unit = parts[1] if len(parts) > 1 else ""
    unit = re.sub(r"<[^>]+>", "", unit).strip()  # strip HTML like <SUP>3</SUP>
    return quantity, unit


def identify_columns(header: dict):
    """Map an entry's variable columns to (tcol, pcol, dcol, p_unit)."""
    tcol = pcol = dcol = None
    p_unit = None
    for col, label in header.items():
        quantity, unit = parse_label(label)
        if "error" in quantity or "uncertainty" in quantity:
            continue
        u = unit.lower()
        if tcol is None and "temperature" in quantity and u == "k":
            tcol = col
        elif pcol is None and "pressure" in quantity and u in PRESSURE_TO_KPA:
            pcol = col
            p_unit = u
        elif dcol is None and "density" in quantity and u.startswith("kg/m"):
            dcol = col
    return tcol, pcol, dcol, p_unit


@dataclass
class RawEntry:
    id: str
    smiles: str
    name: str
    reference: str
    data: pd.DataFrame
    header: dict


def fetch_entry(code, smiles, name, reference, cache_dir, refresh, sleep):
    """Load an entry from cache or NIST. Returns RawEntry or None on failure."""
    path = os.path.join(cache_dir, f"{code}.pkl")
    if (not refresh) and os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                return RawEntry(**pickle.load(fh))
        except Exception:
            pass  # corrupt cache -> refetch
    try:
        e = ilt.GetEntry(code)
        time.sleep(sleep)
    except Exception as ex:
        log.warning("  GetEntry(%s) failed: %s", code, ex)
        return None
    raw = RawEntry(id=code, smiles=smiles, name=name, reference=reference,
                   data=e.data.copy(), header=dict(e.header))
    try:
        with open(path, "wb") as fh:
            pickle.dump(raw.__dict__, fh)
    except Exception:
        pass
    return raw


def entry_to_rows(raw: RawEntry, default_pressure: float):
    """Turn a RawEntry into a list of {T_K, P_kPa, density_kg_m3} rows."""
    tcol, pcol, dcol, p_unit = identify_columns(raw.header)
    if tcol is None or dcol is None:
        return []  # not a usable temperature/density table
    pfac = PRESSURE_TO_KPA.get(p_unit, 1.0) if pcol else None
    rows = []
    for _, r in raw.data.iterrows():
        try:
            T = float(r[tcol]); D = float(r[dcol])
        except (TypeError, ValueError):
            continue
        if pcol is not None:
            try:
                P = float(r[pcol]) * pfac
            except (TypeError, ValueError):
                P = default_pressure
        else:
            P = default_pressure
        rows.append({"T_K": T, "P_kPa": P, "density_kg_m3": D})
    return rows


def validate_smiles(smiles, require_ions):
    """Return (canonical_smiles, MW, n_frags) or None if invalid / filtered."""
    if not smiles or not isinstance(smiles, str):
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    n_frags = len(Chem.GetMolFrags(m))
    if require_ions and (n_frags < 2 or Chem.GetFormalCharge(m) != 0):
        return None  # keep only net-neutral multi-ion salts (genuine ILs)
    return Chem.MolToSmiles(m), Descriptors.MolWt(m), n_frags


def molecule_level_split(smiles_list, splits, seed):
    """Return a function smiles -> {'train','val','test'} with no molecule shared."""
    tr, va, _ = splits
    mols = sorted(set(smiles_list))
    random.Random(seed).shuffle(mols)
    n = len(mols)
    n_tr, n_va = int(round(tr * n)), int(round(va * n))
    train, val, test = set(mols[:n_tr]), set(mols[n_tr:n_tr + n_va]), set(mols[n_tr + n_va:])
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    return lambda s: "train" if s in train else ("val" if s in val else "test")


def main():
    ap = argparse.ArgumentParser(description="Build a faithful IL liquid-density dataset from NIST ILThermo.")
    ap.add_argument("--out", default="data/il_density.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--max-entries", type=int, default=200, help="Cap ILThermo entries pulled (0 = all).")
    ap.add_argument("--min-points", type=int, default=5, help="Skip an entry if fewer points survive filtering.")
    ap.add_argument("--t-min", type=float, default=250.0)
    ap.add_argument("--t-max", type=float, default=450.0)
    ap.add_argument("--p-min", type=float, default=95.0, help="kPa")
    ap.add_argument("--p-max", type=float, default=106.0, help="kPa")
    ap.add_argument("--default-pressure", type=float, default=101.325, help="kPa, when entry has no pressure column.")
    ap.add_argument("--rho-min", type=float, default=500.0, help="kg/m^3 sanity lower bound.")
    ap.add_argument("--rho-max", type=float, default=2500.0, help="kg/m^3 sanity upper bound.")
    ap.add_argument("--keep-nonionic", action="store_true", help="Keep non-salt pure compounds too (default: ILs only).")
    ap.add_argument("--splits", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds between NIST calls (be polite).")
    ap.add_argument("--refresh", action="store_true", help="Ignore cache and re-pull.")
    args = ap.parse_args()

    # Clear existing handlers to prevent duplicate GUI logs if run multiple times
    if log.hasHandlers():
        log.handlers.clear()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    splits = tuple(float(x) for x in args.splits.split(","))
    assert len(splits) == 3 and abs(sum(splits) - 1.0) < 1e-6, "--splits must be three fractions summing to 1"
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1) search ----------------------------------------------------------------
    log.info("Searching ILThermo for pure-compound density datasets ...")
    try:
        search = ilt.Search(n_compounds=1, prop="Density")
    except Exception as ex:
        log.error("ILThermo search failed (%s). ILThermo has no official API; this is usually a "
                  "transient/network issue -- check your network settings and retry.", ex)
        sys.exit(1)
    log.info("  %d candidate density datasets found.", len(search))
    if args.max_entries and args.max_entries > 0:
        search = search.head(args.max_entries)
        log.info("  capped to %d entries (--max-entries).", len(search))

    # 2) fetch + parse ---------------------------------------------------------
    records = []
    n_fail = n_nosmiles = n_badsmiles = n_notable = n_fewpts = 0
    n = len(search)
    for i, row in enumerate(search.itertuples(index=False), 1):
        code = getattr(row, "id")
        smiles = getattr(row, "cmp1_smiles", None)
        name = getattr(row, "cmp1", None)
        reference = getattr(row, "reference", None)
        if i == 1 or i % 25 == 0:
            log.info("  [%d/%d] fetching %s ...", i, n, code)
        if not smiles:
            n_nosmiles += 1
            continue
        v = validate_smiles(smiles, require_ions=not args.keep_nonionic)
        if v is None:
            n_badsmiles += 1
            continue
        can, mw, n_frags = v
        raw = fetch_entry(code, smiles, name, reference, args.cache_dir, args.refresh, args.sleep)
        if raw is None:
            n_fail += 1
            continue
        kept_rows = []
        for r in entry_to_rows(raw, args.default_pressure):
            if not (args.t_min <= r["T_K"] <= args.t_max):
                continue
            if not (args.p_min <= r["P_kPa"] <= args.p_max):
                continue
            if not (args.rho_min <= r["density_kg_m3"] <= args.rho_max):
                continue
            kept_rows.append({"smiles": can, "name": name, "ilt_id": code, "reference": reference,
                              "T_K": r["T_K"], "P_kPa": r["P_kPa"], "density_kg_m3": r["density_kg_m3"],
                              "MW_g_mol": round(mw, 4), "n_frags": n_frags})
        if not kept_rows:
            n_notable += 1
            continue
        if len(kept_rows) < args.min_points:
            n_fewpts += 1
            continue
        records.extend(kept_rows)

    if not records:
        log.error("No usable data points after filtering. Widen --t-* / --p-* / --rho-* "
                  "or raise --max-entries.")
        sys.exit(1)

    df = pd.DataFrame.from_records(records)

    # 3) dedupe identical (smiles, T, P): average density ----------------------
    before = len(df)
    df = (df.groupby(["smiles", "T_K", "P_kPa"], as_index=False)
            .agg({"density_kg_m3": "mean", "name": "first", "ilt_id": "first",
                  "reference": "first", "MW_g_mol": "first", "n_frags": "first"}))
    log.info("Deduplicated %d -> %d rows.", before, len(df))

    # 4) molecule-level split --------------------------------------------------
    which = molecule_level_split(df["smiles"].tolist(), splits, args.seed)
    df["split"] = df["smiles"].map(which)

    df = df[["smiles", "name", "ilt_id", "reference", "T_K", "P_kPa",
             "density_kg_m3", "MW_g_mol", "n_frags", "split"]]
    df = df.sort_values(["split", "smiles", "T_K", "P_kPa"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    # 5) self-check ------------------------------------------------------------
    log.info("\n=== SELF-CHECK ===")
    log.info("skipped: %d fetch-fail, %d no-SMILES, %d non-IL/bad-SMILES, %d no-T/rho-table, %d too-few-points",
             n_fail, n_nosmiles, n_badsmiles, n_notable, n_fewpts)
    log.info("rows: %d   molecules: %d", len(df), df["smiles"].nunique())
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        log.info("  %-5s : %5d rows / %4d molecules", sp, len(sub), sub["smiles"].nunique())
    log.info("ranges: T %.1f-%.1f K   P %.2f-%.2f kPa   rho %.1f-%.1f kg/m^3",
             df["T_K"].min(), df["T_K"].max(), df["P_kPa"].min(), df["P_kPa"].max(),
             df["density_kg_m3"].min(), df["density_kg_m3"].max())

    assert df[["T_K", "P_kPa", "density_kg_m3", "MW_g_mol"]].notna().all().all(), "NaNs in numeric columns"
    assert (df.groupby("smiles")["split"].nunique() == 1).all(), "a molecule leaked across splits"
    assert df["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None).all(), "invalid SMILES slipped through"
    log.info("assertions passed: no NaNs, no split leakage, all SMILES valid.")
    log.info("wrote %s", os.path.abspath(args.out))


def launch_gui():
    """Launches a Tkinter GUI to configure and run the dataset builder."""
    root = tk.Tk()
    root.title("Dataset Builder Configuration")
    root.geometry("550x300")
    root.resizable(False, False)

    # Header / Output info
    info_frame = tk.Frame(root, padx=20, pady=15)
    info_frame.pack(fill="x")
    
    tk.Label(info_frame, text="Dataset Generation Tool", font=("Helvetica", 14, "bold")).pack(anchor="w")
    tk.Label(info_frame, text="Output file will be saved to: data/il_density.csv", fg="blue").pack(anchor="w", pady=(5, 0))
    tk.Label(info_frame, text="Cache files will be saved to: data/cache/").pack(anchor="w")

    # Options
    opt_frame = tk.LabelFrame(root, text="Select Configuration", padx=10, pady=10)
    opt_frame.pack(fill="x", padx=20, pady=5)

    mode_var = tk.IntVar(value=1)

    rb1 = tk.Radiobutton(
        opt_frame, 
        text="Default Range (--max-entries 200, --p-min 95.0, --p-max 106.0, --sleep 0.2)", 
        variable=mode_var, 
        value=1
    )
    rb1.pack(anchor="w", pady=2)

    rb2 = tk.Radiobutton(
        opt_frame, 
        text="Up to 100 bar (--max-entries 0, default --p-min to --p-max 10001.0, --sleep 2.0)", 
        variable=mode_var, 
        value=2
    )
    rb2.pack(anchor="w", pady=2)

    # Status & Run
    status_var = tk.StringVar(value="Ready to build.")
    
    def on_run_clicked():
        btn_run.config(state="disabled")
        rb1.config(state="disabled")
        rb2.config(state="disabled")
        status_var.set("Running... Please check the terminal/console for live progress logs.")

        # Modify sys.argv based on user choice so main() parses it correctly
        if mode_var.get() == 1:
            sys.argv = ["build_dataset.py"]
        elif mode_var.get() == 2:
            sys.argv = [
                "build_dataset.py", 
                "--max-entries", "0", 
                "--p-max", "10001.0", 
                "--sleep", "2.0"
            ]

        def thread_target():
            try:
                main()
                status_var.set("Successfully built dataset!")
                messagebox.showinfo("Success", "Dataset has been built successfully.\nCheck data/il_density.csv")
            except SystemExit as e:
                # sys.exit() called inside main()
                if e.code == 0:
                    status_var.set("Finished.")
                else:
                    status_var.set("Failed. Check terminal logs.")
                    messagebox.showerror("Error", "Process exited with an error. Check terminal output.")
            except Exception as e:
                status_var.set("An error occurred.")
                messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
            finally:
                btn_run.config(state="normal")
                rb1.config(state="normal")
                rb2.config(state="normal")

        # Run in a separate thread so the GUI doesn't freeze
        threading.Thread(target=thread_target, daemon=True).start()

    bottom_frame = tk.Frame(root, padx=20, pady=15)
    bottom_frame.pack(fill="x")

    btn_run = tk.Button(bottom_frame, text="Build Dataset", font=("Helvetica", 10, "bold"), command=on_run_clicked, width=15)
    btn_run.pack(side="left")

    lbl_status = tk.Label(bottom_frame, textvariable=status_var, fg="gray")
    lbl_status.pack(side="left", padx=15)

    root.mainloop()


if __name__ == "__main__":
    # If arguments are passed via CLI, run normally without GUI
    if len(sys.argv) > 1:
        main()
    else:
        # Launch GUI by default
        launch_gui()