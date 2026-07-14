# Reproducing the benchmark

This reproduces the numbers in the accompanying manuscript. It is **one** use of the framework, not
the point of it — for pointing it at your own molecules, see the README's *Using your own data*; for
changing the physics, the descriptor or the target, see *Extending it*.

Read this anyway if you are extending: the **bounds-box** section at the end is the single most
transferable thing here. Two published boxes collapse under this training scheme, and one of them
collapses while reporting perfect coverage.

Eight checkpoints: 2 vdW arms (off / on) x 4 seeds (0, 1, 2, 3). CPU only.
~53 min per checkpoint (median 26 s/epoch), ~7 h total on an Intel Core i7-10700 @ 2.90 GHz / 16 GB.

## 0. Check the data before you spend seven hours

```bash
python scripts/verify_dataset.py
```

Thirty assertions. If any fires, stop: the shipped data and the paper disagree.

## 1. Train the eight checkpoints

```bash
python src/train.py --no-gui \
    --train-val-csv data/train_val.csv \
    --conv PNA --hidden 256 --depth 6 --towers 4 --heads 4 \
    --lr 1e-3 --batch-size 16 --epochs 120 \
    --bounds on --vdw both --seeds 0 1 2 3
```

That is the whole sweep: 2 vdW arms × 4 seeds = 8 checkpoints, no display required.

> **Prefer the GUI?** Drop `--no-gui` and a Tk window opens. There the flags only *pre-tick* the
> widgets and the **checkboxes are authoritative**: tick seeds 0, 1, 2 and 3 (not 4), tick **both**
> vdW arms, leave bounds **on**, press Run. Same eight checkpoints, same names.

Checkpoints land in `checkpoints/` with self-describing names:

```
gnn_core_bounded_s0.pt  ...  gnn_core_bounded_s3.pt        (vdW off)
gnn_core_vdw_bounded_s0.pt  ...  gnn_core_vdw_bounded_s3.pt (vdW on)
```

Each seed runs its own self-check at the end: the training loss must fall, and FeOs coverage is
asserted. A run that collapses **aborts** rather than reporting a plausible-looking number.

## 2. Evaluate against the baselines

Once per checkpoint:

```bash
python src/evaluate.py \
    --train-val-csv data/train_val.csv \
    --test-csv data/test1.csv \
    --checkpoint checkpoints/gnn_core_vdw_bounded_s0.pt \
    --outdir results
```

Do **not** pass `--ecfp-bits`. The paper used the 2048-bit default, and passing anything else
changes the baselines.

`test2.csv` is a debugging set and is excluded from every reported number. If you pass it, you
will get a `test2` table and a `combined` table that correspond to nothing in the paper.

## 3. Export the PC-SAFT parameters

```bash
python src/export_params.py \
    --train-val-csv data/train_val.csv \
    --test-csv data/test1.csv \
    --checkpoint checkpoints/gnn_core_vdw_bounded_s0.pt \
    --outdir results
```

The association columns (kappa_ab, epsilon_k_ab, mu, na, nb) will be **all zero**. That is
correct: the association head is implemented but deliberately inactive.

## 4. The baselines on their own

```bash
python src/baselines.py --train-val-csv data/train_val.csv --test-csv data/test1.csv --vdw on
python src/baselines.py --train-val-csv data/train_val.csv --test-csv data/test1.csv --vdw off
```

One fit per arm, fixed seed, fit on the 18,259-row train split — byte-identical across GNN seeds,
which is why the baseline rows carry no error bars.

## 5. Hyperparameter search (optional; not needed to reproduce the results)

```bash
pip install -e ".[tune]"                      # needs Python <= 3.11 for BOHB
python src/tune.py --train-val-csv data/train_val.csv --trials 32 --epochs 40 --vdw on
python src/tune.py --train-val-csv data/train_val.csv --dry-run --epochs 30 --vdw on   # no Ray
```

---

## The bounds box, and why it is not cosmetic

The adopted box is

| | m | sigma (A) | eps/k (K) |
|---|---|---|---|
| **This work** | 1 – 15 | 2.5 – 4.5 | 150 – 500 |

Two literature boxes were run under otherwise identical conditions (seed 0, vdW on, 120 epochs).
**Both collapse.**

| Box | m | sigma (A) | eps/k (K) | Outcome |
|---|---|---|---|---|
| **This work** | 1–15 | 2.5–4.5 | 150–500 | Completes. Coverage 100.0 % at every epoch. Best val **3.08 %** @ epoch 97. |
| Felton et al. | 1–10 | 2.5–5.0 | 100–1000 | **Collapses at epoch 65.** Coverage 100.0 % -> **0.4 %**; val frozen at 61.98 % for 55 epochs. |
| Esper et al. | 1–23.32 | 1.9–4.5 | 50–550 | **Collapses at epoch 34.** Coverage stays **100.0 %**; val frozen at 78.47 % for 86 epochs. |

The methodological point, and the reason the framework's self-checks are load-bearing:
**coverage alone is blind to the Esper mode.** It catches Felton — the liquid root is lost — and
reports a perfect 100 % while the Esper model is 78 % wrong. What catches **both** is the training
loss rising, which is precisely the assertion `train_one_seed` makes at the end of every run, and
which aborts both.

Honest detail: the Felton box reaches **2.79 % at epoch 53 — better than the adopted box ever
attains** — before it collapses. The box is chosen for **survival, not peak accuracy**.

The bound alone is not what does the work. Felton's box was bounded and still collapsed. The box
must lie inside the region where a liquid root exists.
