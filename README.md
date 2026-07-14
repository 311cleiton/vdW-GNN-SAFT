# vdW-GNN-SAFT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](pyproject.toml)
[![CI](https://github.com/311cleiton/vdW-GNN-SAFT/actions/workflows/ci.yml/badge.svg)](https://github.com/311cleiton/vdW-GNN-SAFT/actions/workflows/ci.yml)

**Train equation-of-state parameters with the equation of state inside the loop.**

A graph neural network reads a molecule and predicts PC-SAFT pure-component parameters — *m*, *σ*,
*ε/k*. A **differentiable** PC-SAFT engine ([FeOs](https://github.com/feos-org/feos), via
`feos-torch`) turns those parameters into a density at each experimental (*T*, *P*), and the loss is
taken on **the measured property, not on reference parameters**. Gradients flow density →
parameters → network weights.

That inversion is the point. A conventional parameter-prediction model needs a table of fitted
parameters to learn from — which for ionic liquids barely exists. Their vapour pressure is
negligible, so the usual regression target is unavailable, and published parameter sets are sparse
and mutually inconsistent. Here the supervision is the density measurement itself, which is
abundant. **The model never sees a reference parameter set.**

This repository ships the full pipeline, a curated ionic-liquid density benchmark, and per-module
self-checks. It is meant to be **used and extended**, not merely re-run:

- **Point it at your own molecules** — any RDKit-parseable SMILES with (T, P, ρ) measurements.
  → [Using your own data](#using-your-own-data)
- **Supervise on a different property** — density is one target; the loss doesn't care.
- **Swap the equation of state** — the bridge is ~200 lines with two entry points.
- **Turn on the association head** — implemented and bounded, deliberately inactive here.

  → [Extending it](#extending-it)

The ionic-liquid benchmark is a worked example, not the boundary.

---

## Install

Python 3.10–3.12. Use a dedicated virtual environment (see the note under
[Repository layout](#repository-layout)).

```bash
git clone https://github.com/311cleiton/vdW-GNN-SAFT.git
cd vdW-GNN-SAFT

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

python scripts/selfcheck.py --with-training            # confirm it actually works
```

<details>
<summary><b>Two install gotchas</b></summary>

- **Tkinter.** `train.py`, `build_dataset.py` and `smiles2vdW_volume.py` offer a GUI and need Tk.
  It ships with python.org and conda builds; on Debian/Ubuntu install `python3-tk`. Training also
  runs fully headless with `--no-gui`, which needs none of this.
- **BOHB / Ray.** `tune.py`'s full BOHB search needs `hpbandster` + `ConfigSpace`, which require
  **Python ≤ 3.11**. Install with `pip install -e ".[tune]"`. The `--dry-run` mode has no Ray
  dependency and works on any supported Python.

</details>

`requirements-lock.txt` pins the exact environment used for the published benchmark, if you need
bit-level agreement.

---

## Quickstart

```bash
# Data integrity: 30 assertions, no torch needed (~20 s)
python scripts/verify_dataset.py

# The physics bridge: density, gradients, cross-check against plain FeOs
python src/pcsaft.py

# The GNN: 24 architecture configurations, gradient flow, bias-at-priors
python src/model.py

# Train, headless. Two vdW arms x four seeds = 8 checkpoints.
python src/train.py --no-gui --train-val-csv data/train_val.csv \
    --vdw both --bounds on --seeds 0 1 2 3 --epochs 120
# (drop --no-gui to configure the same run through a Tk GUI instead)

# Score a checkpoint against the classical baselines on held-out molecules
python src/evaluate.py --no-gui --train-val-csv data/train_val.csv --test-csv data/test1.csv \
    --checkpoint checkpoints/gnn_core_vdw_bounded_s0.pt --outdir results

# Materialize the predicted PC-SAFT parameters as a per-molecule table
python src/export_params.py --no-gui --train-val-csv data/train_val.csv --test-csv data/test1.csv \
    --checkpoint checkpoints/gnn_core_vdw_bounded_s0.pt --outdir results
```

---

## Using your own data

Nothing in the pipeline is specific to the bundled dataset. Every script reads a CSV, and the
schema is the whole contract:

| Column | Units | Required | Notes |
|---|---|---|---|
| `smiles` | — | yes | RDKit-parseable. Multi-fragment (`cation.anion`) is fine — it is the normal case here. |
| `T_K` | K | yes | |
| `P_kPa` | kPa | yes | |
| `density_kg_m3` | kg/m³ | yes | the supervision target |
| `MW_g_mol` | g/mol | yes | converts molar → mass density inside the bridge |
| `split` | — | yes | `train` / `val` in the train file |
| `vdW` | Å³ | only with `--vdw on` | from `smiles2vdW_volume.py` |
| `name`, `ilt_id`, `reference`, `n_frags` | — | no | provenance; carried through, never seen by the model |

From a list of SMILES to a trained model:

```bash
# 1. Compute the vdW volume for each unique SMILES (opens a small file picker)
python src/smiles2vdW_volume.py

# 2. Train on your CSV
python src/train.py --no-gui --train-val-csv mydata/train_val.csv \
    --vdw on --bounds on --seeds 0 1 2 --epochs 120 --out checkpoints/mine.pt

# 3. Score it
python src/evaluate.py --train-val-csv mydata/train_val.csv --test-csv mydata/test.csv \
    --checkpoint checkpoints/mine_vdw_bounded_s0.pt --outdir results
```

Three things that will bite you, in order of how often they do:

1. **Split by molecule, never by row.** A random row split leaks the same molecule into train and
   test at a different temperature, and the reported error stops meaning anything.
   `build_dataset.py::molecule_level_split` does it correctly — copy it.
2. **Check the bounds box.** `BOUNDS` in `model.py` is *m* ∈ [1,15], *σ* ∈ [2.5,4.5] Å,
   *ε/k* ∈ [150,500] K, chosen so FeOs always finds a liquid root **for ionic liquids**. Move to a
   different chemical class and that box may be wrong. If training collapses, suspect this first.
   [`docs/reproducing-the-paper.md`](docs/reproducing-the-paper.md) shows two published boxes that
   fail under this scheme, and why.
3. **Watch coverage — but don't trust it.** Coverage (the fraction of points where FeOs converged)
   catches one collapse mode and is completely blind to another: a model can report 100 % coverage
   while being 78 % wrong. What catches both is the **training loss rising**, which `train_one_seed`
   asserts on at the end of every run and aborts.

`build_dataset.py` will also pull ionic-liquid density straight from NIST ILThermo if you'd rather
extend the bundled set than bring your own.

---

## Extending it

The pieces are deliberately separable. Each of these is a contained change:

| Want to… | Touch | Notes |
|---|---|---|
| **Predict a different property** | `pcsaft.py` | `mass_density_kg_per_m3` is the only place density is special. FeOs also exposes enthalpy, heat capacity, speed of sound, vapour pressure… Swap the forward call and the target column; the gradient path is unchanged. |
| **Use a different equation of state** | `pcsaft.py` | The bridge is ~200 lines and exposes two things: `assemble_pcsaft_params` and a property call. Any differentiable EoS of the same shape drops in. |
| **Train on several properties at once** | `train.py` | The loss is one `huber_relative_loss` on one tensor. Sum a few. |
| **Activate association** (κ_AB, ε_AB) | `--params assoc` | Implemented, bounded, and self-checked. Off in the benchmark because density alone does not *identify* association parameters — see [Known limitations](#known-limitations). Supervise on something else and it becomes meaningful. |
| **Change the architecture** | `--conv {PNA,GATv2,Transformer}` | All three wired and self-checked across 24 configurations. `model.py::_make_conv` is where a fourth goes. |
| **Change the molecular descriptor** | `model.py` | `vdW` is one scalar standardized and concatenated to the pooled embedding. Anything per-molecule fits the same slot — COSMO volume, dipole moment, a learned fingerprint. |
| **Re-tune hyperparameters** | `tune.py` | BOHB over conv type, width, depth, batch size, learning rate. `--dry-run` needs no Ray. |
| **Add a baseline** | `baselines.py` | `fit_baselines` / `predict_baselines` is the interface `evaluate.py` expects. |

The self-checks are the safety net for all of it:

```bash
python scripts/selfcheck.py --with-training --with-baselines
```

That exercises the data, the physics bridge, the GNN across 24 configurations, a full headless
training run, and the baselines. If a change breaks something structural, you find out in minutes
instead of after a seven-hour training run.

---

## How it performs

On the bundled benchmark — 170 **unseen** ionic liquids, 5,168 points, four seeds, mean ± sample SD:

| Model | Point MAPE (%) | Molecule MAPE (%) | Non-monotone isobars (of 311) |
|---|---|---|---|
| GNN, vdW **off** | 5.36 ± 0.23 | 6.04 ± 0.22 | **0** |
| GNN, vdW **on** | **4.01 ± 0.12** | **4.94 ± 0.13** | **0** |
| Random forest (vdW on) | 3.65 | 3.67 | 61–63 |
| ECFP-MLP (vdW on) | **2.82** | 4.02 | 0 *(no guarantee)* |

Two things are true at once, and both get said:

1. **The volume-aware descriptor works.** It cuts point MAPE by 1.35 ± 0.19 points and molecule
   MAPE by 1.10 ± 0.17. The seed bands are fully disjoint — *every* vdW-on run beats *every* vdW-off
   run, on both metrics.

2. **The baselines beat the GNN on raw interpolation MAPE.** They do. An ECFP-MLP fitting density
   directly is more accurate point-for-point than routing through an equation of state. That is
   stated first rather than buried.

The case for physics-in-the-loop is not raw MAPE. It is:

- **Thermodynamic consistency by construction.** Density comes from a real equation of state, so it
  is monotone in temperature along an isobar because PC-SAFT is. The GNN violates this on **0 of
  311** held-out isobars in **all eight** runs. The random forest steps on **61–63** of them (jumps
  up to +16.4 kg/m³). The MLP happens to pass — but nothing in it *guarantees* that, and no such
  claim is made.
- **The output is a parameter set, not a lookup table.** Valid across the whole (*T*, *P*) surface
  and usable in any PC-SAFT code — including for properties the model was never trained on.
- **The descriptor pays off *through* the EoS.** Appending vdW to the baselines' feature vector buys
  them ~0.1 points. Through the equation of state it buys 1.35 / 1.10. The volume information is
  doing physical work, not just adding a feature column.

The full recipe — including two published bounds boxes that **collapse** under this training scheme,
and why coverage alone fails to notice one of them — is in
[`docs/reproducing-the-paper.md`](docs/reproducing-the-paper.md).

---

## The data

`data/` holds the frozen, provenance-tracked benchmark.

| File | Rows | ILs | Role |
|---|---|---|---|
| `train_val.csv` | 21,556 | 922 | train (18,259 rows / 763 ILs) + val (3,297 / 159) |
| `test1.csv` | 5,168 | 170 | the held-out test set — every reported number |
| `test2.csv` | 439 | 30 | debugging set, **excluded from all reported results** |

26,724 rows / 1,092 ILs (`train_val` + `test1`), from NIST ILThermo 2.0 via ILThermoPy. Splits are
**molecule-disjoint**. Ranges: *T* 250.0–443.46 K, *P* 95.0–10,001.0 kPa, *ρ* 815–1,899 kg/m³.

**Read [`data/README.md`](data/README.md) before using the `vdW` column.** It is not the additive
hard-core volume of the ion pair, and the reason is worth knowing.

---

## Repository layout

```
├── src/                     the nine pipeline modules
│   ├── build_dataset.py       pull + filter NIST ILThermo -> frozen CSV
│   ├── smiles2vdW_volume.py   RDKit vdW volume for each SMILES
│   ├── model.py               the GNN (PNA / GATv2 / TransformerConv) + bounded heads
│   ├── pcsaft.py              the differentiable PC-SAFT bridge (feos-torch)
│   ├── train.py               physics-in-the-loop training, seed x vdW sweep
│   ├── tune.py                BOHB hyperparameter search
│   ├── evaluate.py            GNN vs. RF vs. ECFP-MLP on held-out molecules
│   ├── baselines.py           the classical baselines
│   └── export_params.py       predicted PC-SAFT parameters as a table
├── data/                    the frozen benchmark (+ its own README — read it)
├── scripts/
│   ├── verify_dataset.py      30 assertions on the shipped data
│   ├── smoke_train.py         headless end-to-end training run
│   ├── selfcheck.py           runs every self-check; names the modules that have none
├── docs/
│   ├── reproducing-the-paper.md
│   └── known-issues.md        the honest list
└── .github/workflows/ci.yml   verifies the frozen dataset on every push
```

The modules import each other flat (`from model import ...`), which is how they were written and
run. `pip install -e .` therefore exposes them as **top-level** modules (`model`, `train`,
`pcsaft`, …). Those are generic names — install into a dedicated virtual environment. Running by
path (`python src/train.py`) works without installing at all.

### The benchmark configuration

PNA, hidden 256, depth 6, towers 4, heads 4, lr 1e-3, batch 16, 120 epochs, gradient clipping off,
association head off, **bounded heads on**, with the parameter box

> *m* ∈ [1, 15]  ·  *σ* ∈ [2.5, 4.5] Å  ·  *ε/k* ∈ [150, 500] K

The box is not cosmetic. Bounds keep the predicted parameters inside the region where FeOs finds a
liquid root; outside it, training collapses to NaN. But the bound alone is not what does the work —
a *bounded* literature box also collapses. The box has to sit where a liquid root exists.

**CPU only.** No CUDA call anywhere, and FeOs is a compiled CPU library, so the EoS solve runs on
CPU regardless. One checkpoint ≈ 53 min (median 26 s/epoch) on an Intel Core i7-10700; all eight
≈ 7 h. No accelerator required.

---

## Known limitations

Stated up front, because they affect what you can do with this. Full list in
[`docs/known-issues.md`](docs/known-issues.md).

- **Curation is a documented manual act, not a script.** `build_dataset.py` performs the pull and
  the range/sanity filters. The physically-motivated curation applied afterwards was done by hand,
  guided by a written specification. The frozen CSVs are the artifact — every criterion is
  recomputable against them (`scripts/verify_dataset.py` does exactly that) and the counts reproduce
  exactly — but no button regenerates them. This is the one place in the pipeline that is not
  push-button, and it is better said than implied.
- **The association head is inactive.** Exported as zeros throughout. Density alone does not
  *identify* association parameters; activating it against density would widen an already degenerate
  family. It is there for a different supervision signal, not for these results.
- **The predicted parameters are one member of a density-equivalent family.** Across seeds,
  per-molecule *m* varies by ~30 % while *m·σ³* varies by ~3 % and the resulting density by ~0.7 %.
  Density supervision pins the *combination*, not the individual parameters. Do not read a single
  predicted *m* as a physical measurement.
- **The `vdW` column is not an additive ion-pair volume.** RDKit embeds dot-separated ions
  concentrically. The descriptor is deterministic, strictly size-ordered, and consumed identically
  by every model in the comparison — so no result is biased — but it is not a hard-core volume.
  [`data/README.md`](data/README.md) has the full treatment.

---

## Citation

`CITATION.cff` is machine-readable; GitHub renders a "Cite this repository" button from it.

```bibtex
@software{beraldo_vdw_gnn_saft,
  author  = {Beraldo, Cleiton S.},
  title   = {vdW-GNN-SAFT: Physics-in-the-Loop Graph Neural Networks for Ionic Liquids},
  year    = {2026},
  url     = {https://github.com/311cleiton/vdW-GNN-SAFT}
}
```

A Zenodo DOI is minted from the first tagged release and will be added here and to
`CITATION.cff` once it exists (see `docs/release-checklist.md`).

A manuscript describing the method is under review. Once it appears, please cite that as well — it
is listed under `preferred-citation` in `CITATION.cff`.

**Author** — Cleiton S. Beraldo
([ORCID 0009-0008-8798-2882](https://orcid.org/0009-0008-8798-2882) ·
[Google Scholar](https://scholar.google.com/citations?user=GXXG9RYAAAAJ&hl=en) ·
[cleitonberaldo@alumni.usp.br](mailto:cleitonberaldo@alumni.usp.br)),
Department of Chemical Engineering (PQI), Polytechnic School, University of São Paulo.
*This v1.0.0 release credits the lead author only; co-authors will be added in a subsequent
version.*

## License

[MIT](LICENSE) — use it, fork it, build on it.

The dataset is derived from **NIST ILThermo 2.0**; please also cite the ILThermo database and the
primary sources listed in the `reference` column of each CSV.
