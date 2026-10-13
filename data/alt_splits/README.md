# Alternative splits: 65/15/20, 75/15/10, 65/10/25, 70/10/20, 75/10/15, 65/20/15, 70/20/10, 75/20/5

Two re-partitions of the frozen dataset, made by moving **whole ionic liquids** between the train
split and `test1` under the curation rules of the manuscript (Section 2.1). Validation is untouched.
Six more leave training untouched and instead move ionic liquids between validation and test:
65/10/25, 70/10/20 and 75/10/15 (validation → test), 65/20/15, 70/20/10 and 75/20/5 (test →
validation); see *Validation ↔ test* below. Every row is carried over byte-for-byte from
`data/train_val.csv` / `data/test1.csv`; only its `split` field changes. The frozen files themselves
are not modified.

```bash
python scripts/make_alt_splits.py        # the first two, verified (seed 42)
python scripts/make_val_test_splits.py   # the six validation <-> test splits, verified (run after the line above)
```

| Scheme | Train ILs / rows | Val ILs / rows | Test ILs / rows | Moved |
|---|---|---|---|---|
| frozen 70/15/15 | 763 / 18,259 (69.9 %) | 159 / 3,297 (14.6 %) | 170 / 5,168 (15.6 %) | — |
| `split_65_15_20` | 715 / 16,580 (65.5 %) | 159 / 3,297 (14.6 %) | 218 / 6,847 (20.0 %) | 48 train → test |
| `split_75_15_10` | 824 / 20,483 (75.5 %) | 159 / 3,297 (14.6 %) | 109 / 2,944 (10.0 %) | 61 test → train |

Percentages are of the 1,092 ionic liquids. The test size is exact, and train absorbs the fact that
the frozen val is 14.6 %, not 15 %. By rows the test shares are 25.6 / 11.0 %.

## Why only some compounds can move

The curation is split-asymmetric by design:

| Tier | Flags | train / val | test1 |
|---|---|---|---|
| 3: physically corrupt | packing vdW·ρ/(1000·MW) ≥ 1; large residuals from the compound's linear ρ(T) fit; a source > 5 % from consensus | removed | removed |
| 2: outside model scope | metal-anion systems; n_frags ≥ 3; anomalously dense (ρ > 1,900 kg/m³) | removed | **kept** |
| 1: extreme but plausible | vdW ≥ 600 Å³; low-density fatty-acid anions (train/val floor 830.6 kg/m³) | removed | **kept** |

So a compound's rows depend on which split it sits in:
- **test → train:** a test compound carrying tier-1/2 data would lose rows, or vanish entirely.
- **train → test:** a train compound that lost tier-1/2 rows would need rows the frozen CSVs no longer hold. The 1-(2-ethoxyethyl)-3-ethylimidazolium perrhenate series at 1,899 kg/m³ is the concrete case.

Only compounds that sit **well within every rule** have identical rows under both regimes, and only
those were moved. "Well within" means clearing each threshold with a margin:

| Criterion | Section 2.1 threshold | Margin required to move |
|---|---|---|
| Packing φ = 0.602214·vdW·ρ/(1000·MW) | φ ≥ 0.6022 → tier 3 | φ_max < 0.57 |
| Size | vdW ≥ 600 Å³ → tier 1 | vdW < 550 Å³ |
| Low density | below the 830.6 kg/m³ train/val floor → tier 1 | ρ_min ≥ 850 kg/m³ |
| High density | ρ > 1,900 kg/m³ → tier 2 | ρ_max ≤ 1,850 kg/m³ |
| Fragments / metals | n_frags ≥ 3, metal anions → tier 2 | n_frags = 2, no metal atom |
| ρ(T), P ≤ 200 kPa, ≥ 5 T | large residuals from the compound's line → tier 3 | slope < 0, every point ≤ 2 % from the compound's line, every isobar falling |
| Reference branch | a source > 5 % from the pooled consensus → tier 3 | ≤ 2.5 % |

These diagnostics reproduce the manuscript on the frozen data:
- 1,084 ILs with ≥ 5 temperatures at P ≤ 200 kPa, all with density falling with temperature;
- the worst point is [BMIM][HSO₄], 6.3 % from its line relative to the measured density, as in the paper
  (6.0 % relative to the fitted value, the measure `eligibility.csv` reports as `rhoT_maxdev_pct`);
- that same compound's branches deviate 2.19 % from the pooled consensus (paper: 2.2 %).

The one exception outside Section 2.1 is 1,1-dimethyl-4-phenylpiperazinium iodide. It passes every screen, but Section 3.4 names it a data-quality candidate, so it is kept in test rather than trained on.

**Compounds that never move** (`eligibility.csv` gives the reasons, compound by compound):
- **24 of the 170 test ILs.**
  - 10 are tier 1/2 by rule: vdW ≥ 600 Å³ (5), the two dications, FeCl₄⁻, SbF₆⁻, and trihexylammonium octanoate at 815 kg/m³.
  - 13 are within the rules, but only just. Most sit close to the packing, size or ρ(T) limits; one is the Al-bearing [C₅MIM][AlCl₄].
  - 1 is the iodide above.
- **60 of the 763 train ILs** sit near a threshold, carry a metal, or have too few ambient temperatures for the ρ(T) test.

## How the moves were chosen

- **Seeded random draws.** Moves are drawn at random (seed 42) from the movable compounds.
- **Stratified by cation class.** Every new test set is steered toward the cation-class mix of the original test1: imidazolium 38.8 → 39.0 / 38.5 %, ammonium 20.0 → 20.2 / 20.2 %, phosphonium 10.0 → 10.1 / 10.1 %.
- **Nested test sets.** test(75/15/10) ⊂ test1 ⊂ test(65/15/20). So a compound in a smaller test set is in every larger one, and results across schemes differ only by what was added or removed.

## What shifts, and why it matters for comparisons

The compounds that can't move stay in test, so the smaller test sets are **harder** than test1:

| | test1 | 65/15/20 | 75/15/10 |
|---|---|---|---|
| Unmovable (tier 1/2 or near a threshold) | 24 (14 %) | 24 (11 %) | 24 (22 %) |
| of which tier 1/2 by rule | 10 (5.9 %) | 10 (4.6 %) | 10 (9.2 %) |
| ILs measured above 106 kPa | 19 | 28 | 9 |
| Test ILs with a cation unseen in train | 36 % | 37 % | 34 % |
| Test ILs with an anion unseen in train | 11 % | 11 % | 15 % |

The train split always keeps the frozen envelope (vdW 69–599 Å³, ρ 831–1,899 kg/m³, n_frags = 2).

What this means for results:
- **Molecule-level metrics on 75/15/10** mix a larger training set with a harder test set.
- **For a like-for-like comparison, use the nesting.** The 109 compounds of test(75/15/10) are unseen by all three models (frozen, 65/15/20, 75/15/10). Scoring each model on those 109 isolates the effect of training-set size. Restricting further to the 85 movable compounds among them also removes the edge-case enrichment.
- **The manuscript** reports seed 3 on all eight alternative splits (Section 3.2, Table 4). Its other numbers, and
  `scripts/verify_dataset.py`, refer to the frozen split.

## Validation ↔ test: 65/10/25, 70/10/20, 75/10/15, 65/20/15, 70/20/10, 75/20/5

These six keep the training split exactly as it was and move whole ionic liquids between validation
and test, so validation becomes 10 % or 20 % of the 1,092 ILs and test takes the rest:

| Scheme | Built from | Moved | Train ILs / rows | Val ILs / rows | Test ILs / rows | Unmovable in test |
|---|---|---|---|---|---|---|
| `split_65_10_25` | `split_65_15_20` | 50 val → test | 715 / 16,580 (65.5 %) | 109 / 1,980 (10.0 %) | 268 / 8,164 (24.5 %) | 24 (9 %) |
| `split_70_10_20` | frozen 70/15/15 | 50 val → test | 763 / 18,259 (69.9 %) | 109 / 1,980 (10.0 %) | 220 / 6,485 (20.1 %) | 24 (11 %) |
| `split_75_10_15` | `split_75_15_10` | 50 val → test | 824 / 20,483 (75.5 %) | 109 / 1,980 (10.0 %) | 159 / 4,261 (14.6 %) | 24 (15 %) |
| `split_65_20_15` | `split_65_15_20` | 59 test → val | 715 / 16,580 (65.5 %) | 218 / 4,266 (20.0 %) | 159 / 5,878 (14.6 %) | 24 (15 %) |
| `split_70_20_10` | frozen 70/15/15 | 59 test → val | 763 / 18,259 (69.9 %) | 218 / 4,266 (20.0 %) | 111 / 4,199 (10.2 %) | 24 (22 %) |
| `split_75_20_5` | `split_75_15_10` | 59 test → val | 824 / 20,483 (75.5 %) | 218 / 4,266 (20.0 %) | 50 / 1,975 (4.6 %) | 24 (48 %) |

- **Three training sets, three splits each.** With their parents, these splits form three families that each share one training set: 65 % (65/15/20, 65/10/25, 65/20/15), 70 % (the frozen 70/15/15, 70/10/20, 70/20/10) and 75 % (75/15/10, 75/10/15, 75/20/5). `scripts/train_multi_val.py` trains a family once and keeps each split's best epoch on its own validation set.
- **Same eligibility rule, both directions.** Section 2.1 curates validation like training (tier-1/2 data removed) and test differently, so only compounds that sit well within every rule move: the same `movable` flag as above. The iodide of Section 3.4 stays in test.
- **One shared move set per direction.** 65/10/25, 70/10/20 and 75/10/15 move the same 50 validation ILs. 65/20/15, 70/20/10 and 75/20/5 move the same 59 test ILs, drawn from test(75/15/10), which lies inside test1 and test(65/15/20). So the three splits of each direction share one validation set, and their test sets are nested: test(75/10/15) ⊂ test(70/10/20) ⊂ test(65/10/25) and test(75/20/5) ⊂ test(70/20/10) ⊂ test(65/20/15).
- **Class mix kept.** Moves are drawn at random (seed 42) within each cation class, so validation keeps its original mix in both directions: imidazolium 45.9 % in all three validation sets, ammonium 15.1 → 14.7 / 15.1 %, phosphonium 10.7 → 11.0 / 10.1 %.
- **Small test sets fill up with edge cases.** The 24 compounds that can never move stay in test, so the 50-IL test set of 75/20/5 is 48 % edge cases.
- **The training trajectory doesn't change.** Validation never feeds a gradient, the learning-rate schedule ignores it, and the vdW statistics and degree histogram come from training only. Two epochs on `split_70_10_20` give bit-identical training losses to the frozen split (seed 0, 16 threads).
- **What validation does decide** is which epoch's weights the checkpoint keeps. So the saved model, and its test scores, can still differ. Val MAPEs on a 109- or 218-IL set aren't comparable with those on the 159-IL set.

## Files

| Path | Content |
|---|---|
| `split_<train>_<val>_<test>/train_val.csv`, `test1.csv` | drop-in replacements, same schema and row order as `data/` |
| `moves.csv` | every train ↔ test move of 65/15/20 and 75/15/10, with its cation class, rows, and direction |
| `moves_val_to_test.csv` | the 50 validation → test moves of 65/10/25, 70/10/20 and 75/10/15 (the same 50 for each) |
| `moves_test_to_val.csv` | the 59 test → validation moves of 65/20/15, 70/20/10 and 75/20/5 (the same 59 for each) |
| `eligibility.csv` | every compound in the frozen data: diagnostics, `movable`, `not_movable_because` |

## Training and scoring on a split

The three splits of a training set share their training rows, so `scripts/train_multi_val.py` trains each
training set once and keeps every split's best epoch on its own validation set. The manuscript's runs use seed 3
and 16 threads; the 75 % training set was trained with the gradient norm clipped at 0.05 and a one-epoch
learning-rate warm-up at the start of every cosine cycle:

```bash
python scripts/train_multi_val.py --splits 65_15_20 65_10_25 65_20_15 --vdw off --seed 3 --threads 16
python scripts/train_multi_val.py --splits 70_15_15 70_10_20 70_20_10 --vdw off --seed 3 --threads 16
python scripts/train_multi_val.py --splits 75_10_15 75_20_5 75_15_10 --vdw off --seed 3 --threads 16 \
    --grad-clip 0.05 --warmup-epochs 1
```

Repeat each with `--vdw on`. `70_15_15` is the frozen split. The checkpoints are
`checkpoints/gnn_core_<split>[_vdw]_bounded_s3.pt`, so the benchmark checkpoints stay intact, and the vdW
standardization buffers come from each training set itself. Score every split on its own test set:

```bash
python src/evaluate.py --no-gui --train-val-csv data/alt_splits/split_65_10_25/train_val.csv \
    --test-csv data/alt_splits/split_65_10_25/test1.csv \
    --checkpoint checkpoints/gnn_core_65_10_25_vdw_bounded_s3.pt --outdir results/gnn_core_65_10_25_vdw_bounded_s3
```

Expected test results (manuscript Table 4; every evaluation covers 100 % of the test points):

| Split | Test ILs / points | Point MAPE, vdW off → on | Molecule MAPE, vdW off → on |
|---|---|---|---|
| 65/15/20 | 218 / 6,847 | 5.44 → 3.50 % | 5.92 → 4.49 % |
| 65/10/25 | 268 / 8,164 | 4.85 → 3.40 % | 5.47 → 4.43 % |
| 65/20/15 | 159 / 5,878 | 4.91 → 3.58 % | 5.52 → 4.69 % |
| 70/15/15 (frozen) | 170 / 5,168 | 5.04 → 3.88 % | 5.77 → 4.86 % |
| 70/10/20 | 220 / 6,485 | 4.99 → 3.79 % | 5.71 → 4.83 % |
| 70/20/10 | 111 / 4,199 | 5.27 → 4.03 % | 6.25 → 5.24 % |
| 75/15/10 | 109 / 2,944 | 6.78 → 5.16 % | 6.96 → 5.56 % |
| 75/10/15 | 159 / 4,261 | 5.81 → 4.48 % | 6.42 → 5.12 % |
| 75/20/5 | 50 / 1,975 | 7.33 → 6.09 % | 8.36 → 7.32 % |
