# Known issues and limitations

The paper claims an *open, self-checking, end-to-end reproducible framework*. This file is what
keeps that claim honest. Everything here is a real constraint on what you can do with this
repository, stated plainly rather than discovered by a frustrated user.

---

## 1. ~~`train.py` requires a graphical display~~ — **FIXED in v1.0.0**

Historical note, kept because it explains why `train.py` looks the way it does.

`train.py` used to parse its command line and then **unconditionally** open a Tkinter window. The
CLI flags only *pre-ticked* the widgets — the help text said so outright — and the seeds, vdW arms,
parameter mode and output path were all read back from the widgets after the user clicked **Run**.
There was no code path that skipped the GUI, so training could not run on CI, over plain SSH, on an
HPC batch node, or in a container. (`xvfb-run` did not help: a virtual display renders the window,
but nothing clicks the button.)

**It was broader than training.** `evaluate.py` and `export_params.py` opened unconditional Tk
windows too, so *scoring a checkpoint* and *exporting the parameters* were equally impossible
headless. Four of the nine modules were display-locked.

`--no-gui` now configures each of them entirely from the command line:

```bash
python src/train.py --no-gui --train-val-csv data/train_val.csv \
    --vdw both --bounds on --seeds 0 1 2 3 --epochs 120

python src/evaluate.py --no-gui --train-val-csv data/train_val.csv --test-csv data/test1.csv \
    --checkpoint checkpoints/gnn_core_vdw_bounded_s0.pt --outdir results

python src/export_params.py --no-gui --train-val-csv data/train_val.csv --test-csv data/test1.csv \
    --checkpoint checkpoints/gnn_core_vdw_bounded_s0.pt --outdir results
```

The GUIs are unchanged and remain the default. In `train.py`, `configure_headless()` sets exactly
the five attributes the Run button set — `params`, `vdws`, `bounds`, `out_base`, `seeds`. In
`evaluate.py` and `export_params.py` the widgets only ever overwrote four attributes that argparse
already defines, so `--no-gui` is a pure skip. Nothing in the training or scoring mathematics moved.

`build_dataset.py` and `smiles2vdW_volume.py` still need a display. `build_dataset.py` accepts CLI
arguments; `smiles2vdW_volume.py` is GUI-only. Neither is on the critical path for training or
scoring, so they are left as they are — but say so rather than let you discover it.

`scripts/smoke_train.py` now exercises the whole chain headlessly, and CI runs it on every push —
a test that was **impossible** while the GUI was mandatory.

## 2. Curation is a manual, documented act — there is no curation script

`build_dataset.py` performs the ILThermo pull, the range and sanity filters, the deduplication and
the molecule-level split. The physically-motivated curation applied afterwards — the packing
screen, the two-source disagreement handling, the tiered removals — was carried out **by hand**,
guided by a written specification.

The frozen CSVs in `data/` are the artifact. Every criterion is recomputable and checkable against
them (`scripts/verify_dataset.py` does exactly that, and the counts reproduce exactly), but no
button regenerates them from the raw pull.

This is the one place in the pipeline that is not push-button, and it is better said than implied.

## 3. There is no one-command paper regeneration

No Makefile or script rebuilds the figures and tables end-to-end from the raw data. Earlier drafts
of the supporting information promised one; it does not exist and the claim was removed.

## 4. Self-check coverage is uneven

`python scripts/selfcheck.py --list` prints the inventory. Summarised:

| Module | Self-check | Reached by |
|---|---|---|
| `scripts/verify_dataset.py` | ✅ standalone | `python scripts/verify_dataset.py` |
| `pcsaft.py` | ✅ standalone | `python src/pcsaft.py` |
| `model.py` | ✅ standalone | `python src/model.py` |
| `baselines.py` | ✅ standalone | `python src/baselines.py --self-check` |
| `train.py` | ✅ standalone | `python scripts/smoke_train.py` (headless end-to-end) |
| `evaluate.py` | ⚠️ in-run | asserts per test file, inside `main()` |
| `export_params.py` | ⚠️ in-run | asserts per exported file, inside `main()` |
| `build_dataset.py` | ⚠️ in-run | asserts after the pull, inside `main()` |
| `tune.py` | ❌ none | thin wrapper over `train.train_model` |
| `smiles2vdW_volume.py` | ❌ none | GUI-only; no headless entry point |

"In-run" assertions are real and they abort the run on failure — the training loop's rising-loss
assertion is what catches the equation-of-state collapse that coverage alone misses — but they
cannot be exercised without a full run of that module. Two modules have no self-check at all.
(`train.py`'s in-run assertions still exist and still abort a bad run; `smoke_train.py` is what
now drives them automatically.)

## 5. The `vdW` column is not an additive ion-pair volume

RDKit's ETKDG embeds dot-separated ion pairs concentrically, so the stored volume is the union of
two interpenetrating envelopes (V_stored / V_additive = 0.731 ± 0.070). It is deterministic,
strictly size-ordered (Spearman 0.967), standardized before use, and consumed identically by every
model in the comparison — so no result is biased — but it must not be read as a hard-core volume.
Full treatment in [`../data/README.md`](../data/README.md).

## 6. The predicted parameters are one member of a density-equivalent family

Across seeds, per-molecule *m* varies by ~30 % while *m·σ³* varies by ~3 % and the resulting density
by ~0.7 %. Density supervision pins the combination, not the individual parameters. Do not read a
single predicted *m* or *σ* as a physical measurement.

## 7. The association head is inactive

Implemented, wired, and exported as zeros in every run. Density alone does not identify association
parameters. Activating it is future work, not a result.

## 8. `tune.py` imports from `train.py`, not the reverse

`tune.py` is a module of the framework in its own right, built on `train.py`'s training routine.
Nothing in the repository imports `tune.py`.

