#!/usr/bin/env python3
"""
baselines.py  --  Group D (1/2) of the GNN / PC-SAFT density factory.

Habicht-style classical baselines for IL liquid density. Because the physics-in-loop GNN was
trained without reference PC-SAFT parameters, the fair comparison is direct density prediction:
each baseline maps (molecular fingerprint + simple descriptors + state) -> density [kg/m^3].

Data layout
-----------
The baselines are fit on the SAME training molecules the GNN saw -- the rows tagged "train" in
the train+val file (default: data/train_val.csv) -- and evaluated on the held-out test files
(default: data/test1.csv and data/test2.csv), each on its own and combined. Locations are the
DEFAULT_* constants below; override them on the CLI with --train-val-csv / --test-csv.

Features per data point:
  - ECFP (Morgan radius 2, `ecfp_bits` bits; default here 2048 for speed)
  - atom counts (C, N, O, F, Si, P, S, Cl, Br, I), ring count, rotatable bonds, molar weight
  - the state variables T [K] and P [kPa]
  - the van der Waals volume vdW [cubic angstrom], appended only when --vdw on

Models:
  - RandomForestRegressor (raw features)
  - ECFP-MLP: sklearn MLPRegressor on standardized features and standardized target

Note on the two toggles
-----------------------
- PC-SAFT parameter count (`--params core|assoc`): IRRELEVANT here. These baselines never go
  through PC-SAFT -- they map fingerprint + descriptors + (T, P) straight to density -- so the
  SAME baseline is the reference for both GNN parameter modes.
- vdW (`--vdw on|off`): DOES apply. vdW is a generic molecular descriptor, so it can be appended
  to the baseline feature vector exactly like the other descriptors. It is appended per row, so
  multi-fragment mixtures (e.g. DES) that share a SMILES but differ in vdW are distinguished here
  too. For a fair head-to-head, evaluate.py fits these baselines with the SAME vdW setting as the
  GNN checkpoint it is scoring (no normalization is needed: RandomForest is scale-invariant and
  the MLP already standardizes its inputs).

Run `python baselines.py` to fit on the train split and report test MAPE (per test file and
combined) for both baselines.
"""

from __future__ import annotations
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

RDLogger.DisableLog("rdApp.*")
ATOM_SYMBOLS = ["C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]

# ---------------------------------------------------------------------------
# data locations  (edit here, or override on the CLI with --train-val-csv / --test-csv)
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_VAL_CSV = "data/train_val.csv"
DEFAULT_TEST_CSVS = ["data/test1.csv", "data/test2.csv"]


def _mol_features(smiles, gen):
    """Per-molecule feature vector: ECFP bits ++ atom counts ++ rings/rotbonds/MW."""
    m = Chem.MolFromSmiles(smiles)
    fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
    counts = np.zeros(len(ATOM_SYMBOLS), dtype=np.float32)
    for atom in m.GetAtoms():
        s = atom.GetSymbol()
        if s in ATOM_SYMBOLS:
            counts[ATOM_SYMBOLS.index(s)] += 1.0
    extra = np.array([rdMolDescriptors.CalcNumRings(m),
                      rdMolDescriptors.CalcNumRotatableBonds(m),
                      Descriptors.MolWt(m)], dtype=np.float32)
    return np.concatenate([fp, counts, extra])


def featurize(df: pd.DataFrame, gen, use_vdw: bool = False):
    """Rows aligned to df: [molecule features ++ (T_K, P_kPa)[ ++ vdW]]; returns (X, y)."""
    if use_vdw and "vdW" not in df.columns:
        raise ValueError("use_vdw=True but the dataframe has no `vdW` column.")
    cache = {}
    feats, y = [], []
    for row in df.itertuples(index=False):
        if row.smiles not in cache:
            cache[row.smiles] = _mol_features(row.smiles, gen)
        state = [row.T_K, row.P_kPa]
        if use_vdw:
            state.append(row.vdW)          # van der Waals volume [A^3], per row (per formulation)
        feats.append(np.concatenate([cache[row.smiles], np.array(state, dtype=np.float32)]))
        y.append(row.density_kg_m3)
    return np.asarray(feats, dtype=np.float32), np.asarray(y, dtype=np.float32)


def fit_baselines(train_df: pd.DataFrame, ecfp_bits: int = 2048, seed: int = 0, use_vdw: bool = False):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ecfp_bits)
    X, y = featurize(train_df, gen, use_vdw=use_vdw)

    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    rf.fit(X, y)

    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=800, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(x_scaler.transform(X), y_scaler.transform(y.reshape(-1, 1)).ravel())

    return {"gen": gen, "rf": rf, "mlp": mlp, "x_scaler": x_scaler,
            "y_scaler": y_scaler, "ecfp_bits": ecfp_bits, "use_vdw": use_vdw}


def predict_baselines(models, df: pd.DataFrame):
    """Return {name: density predictions} aligned to df rows."""
    X, _ = featurize(df, models["gen"], use_vdw=models.get("use_vdw", False))
    rf = models["rf"].predict(X)
    mlp_std = models["mlp"].predict(models["x_scaler"].transform(X)).reshape(-1, 1)
    mlp = models["y_scaler"].inverse_transform(mlp_std).ravel()
    return {"RandomForest": rf, "ECFP-MLP": mlp}


def _mape(exp, pred):
    return float(np.mean(np.abs(pred - exp) / exp) * 100.0)


def load_train_split(train_val_csv):
    """The rows the GNN trained on: split == 'train' inside the train+val file."""
    df = pd.read_csv(train_val_csv)
    if "split" not in df.columns:
        raise ValueError(f"{train_val_csv} has no `split` column")
    return df[df.split == "train"]


def load_test_frames(test_csvs):
    """{label: dataframe} for each test file (label = filename stem) plus a 'combined' concat."""
    frames = {os.path.splitext(os.path.basename(p))[0]: pd.read_csv(p) for p in test_csvs}
    frames["combined"] = pd.concat(frames.values(), ignore_index=True)
    return frames


def _self_check(use_vdw: bool = False):
    print("== baselines.py self-check ==")
    tv = os.environ.get("IL_TRAIN_VAL_CSV", DEFAULT_TRAIN_VAL_CSV)
    test_csvs = os.environ.get("IL_TEST_CSVS")
    test_csvs = test_csvs.split(",") if test_csvs else DEFAULT_TEST_CSVS
    models = fit_baselines(load_train_split(tv), ecfp_bits=2048, seed=0, use_vdw=use_vdw)
    for label, te in load_test_frames(test_csvs).items():
        preds = predict_baselines(models, te)
        for name, yhat in preds.items():
            assert np.isfinite(yhat).all(), f"{label}/{name}: non-finite predictions"
            mape = _mape(te.density_kg_m3.to_numpy(), yhat)
            assert 0 < mape < 100, f"{label}/{name}: implausible MAPE {mape}"
            print(f"  [{label:8s}] {name:13s} test MAPE {mape:5.2f}%")
    print("baselines.py self-check passed.")


def main():
    ap = argparse.ArgumentParser(description="Fit RF + ECFP-MLP density baselines and report test MAPE.")
    ap.add_argument("--train-val-csv", default=DEFAULT_TRAIN_VAL_CSV,
                    help="CSV with train+val rows; baselines fit on its split=='train' rows.")
    ap.add_argument("--test-csv", nargs="+", default=list(DEFAULT_TEST_CSVS),
                    help="One or more test CSVs; each is reported separately, plus combined.")
    ap.add_argument("--ecfp-bits", type=int, default=2048,
                    help="Morgan/ECFP fingerprint length. The paper uses this 2048-bit default.")
    ap.add_argument("--vdw", choices=["off", "on"], default="off",
                    help="Append the van der Waals volume [cubic angstrom] to the features. "
                         "'on' requires a `vdW` column. Use the same setting as the GNN you compare to.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true",
                    help="Run the module self-check (fit, assert finite + plausible MAPE) and exit.")
    args = ap.parse_args()

    if args.self_check:
        _self_check(use_vdw=(args.vdw == "on"))
        return

    use_vdw = (args.vdw == "on")
    train_df = load_train_split(args.train_val_csv)
    models = fit_baselines(train_df, ecfp_bits=args.ecfp_bits, seed=args.seed, use_vdw=use_vdw)

    print(f"ECFP bits: {args.ecfp_bits}   vdW feature: {'on' if use_vdw else 'off'}   "
          f"(fit on {len(train_df)} train rows / {train_df.smiles.nunique()} molecules)")
    for label, te in load_test_frames(args.test_csv).items():
        preds = predict_baselines(models, te)
        exp = te.density_kg_m3.to_numpy()
        for name, yhat in preds.items():
            print(f"  [{label:8s}] {name:13s} test density MAPE {_mape(exp, yhat):.2f}%")


if __name__ == "__main__":
    main()
