# The frozen dataset

Three CSVs, one schema, no hidden state. Everything downstream reads these files.

| File | Rows | ILs | Role |
|---|---|---|---|
| `train_val.csv` | 21,556 | 922 | `split` column tags each row `train` (18,259 rows / 763 ILs) or `val` (3,297 / 159) |
| `test1.csv` | 5,168 | 170 | the held-out test set — **every number reported in the paper** |
| `test2.csv` | 439 | 30 | a small debugging set, **excluded from all reported results** |

The frozen dataset is `train_val.csv` + `test1.csv` = **26,724 rows / 1,092 ionic liquids**.
Splits are molecule-disjoint: no ionic liquid appears in more than one split, and `test2` shares
no ionic liquid with any of them.

`python scripts/verify_dataset.py` re-derives all thirty of these numbers from the CSVs and
asserts them. If it passes, the data and the paper agree.

## Schema

| Column | Units | Meaning |
|---|---|---|
| `smiles` | — | RDKit-canonical, dot-separated ion pair, e.g. `CCCC[n+]1ccn(C)c1.F[B-](F)(F)F` |
| `vdW` | Å³ | van der Waals volume — **read the caveat below** |
| `name` | — | compound name as reported by ILThermo |
| `ilt_id` | — | NIST ILThermo 2.0 entry ID (provenance) |
| `reference` | — | primary literature source for the measurement (provenance) |
| `T_K` | K | temperature (250.0 – 443.46) |
| `P_kPa` | kPa | pressure (95.0 – 10,001.0; 4,945 rows above 106 kPa) |
| `density_kg_m3` | kg/m³ | experimental liquid mass density (815 – 1,899) — the supervision target |
| `MW_g_mol` | g/mol | molecular weight of the neutral ion pair |
| `n_frags` | — | disconnected fragments; 2 for a normal IL, 3 for the two dications (both in `test1`) |
| `split` | — | `train` / `val` / `test` |

Source: **NIST ILThermo 2.0**, pulled through [ILThermoPy](https://pypi.org/project/ilthermopy/).
Cite the database and the primary sources in the `reference` column, not just this repository.

---

## ⚠️ What the `vdW` column actually is

**It is not the additive hard-core volume of the ion pair.** It is the volume of the *union of two
interpenetrating envelopes*. If you reuse this column, you need to know why.

`smiles2vdW_volume.py` embeds each SMILES in 3D with RDKit's ETKDG distance geometry and
integrates the resulting envelope with `AllChem.ComputeMolVolume`. But an ionic liquid SMILES is
**dot-separated** — two disconnected fragments — and ETKDG's bounds matrix has no term that keeps
disconnected fragments apart. So it embeds the cation and the anion **concentrically**:

- fragment centroid separation across a random sample: **0.00 – 0.24 Å**
- ethylammonium nitrate: closest cation–anion contact **0.61 Å**, against a van der Waals radius
  sum of **3.30 Å**; **38 of 44** inter-ion atom pairs overlap
- the `useRandomCoords` fallback does not fix it either (~1.4 Å)

Quantified against the true additive volume (each ion embedded separately and summed, n = 150):

| Quantity | Value |
|---|---|
| V_stored / V_additive | **0.731 ± 0.070** (median 0.715, range 0.632 – 0.932) |
| Pearson *r* | 0.977 |
| **Spearman rank *r*** | **0.967** |
| OLS | V_additive ≈ 1.13 · V_stored + 51 Å³ |

Ethylammonium nitrate, worked: RDKit returns 69.22 Å³; cation 59.91 + anion 44.06 = 103.97 Å³.
That is **33.5 % overlap**.

### Why the results still stand

The descriptor is **deterministic**, **strictly size-ordered** (Spearman 0.967 against the true
additive volume), standardized to zero mean / unit variance before it reaches any weight, and —
this is the load-bearing part — **both GNN arms and both baselines consume the identical column**.
No comparison in the paper is biased by it. What the network receives is a monotone,
structure-derived measure of molecular *size*, and that is exactly how the paper describes it.

The column was therefore **documented, not recomputed**. Recomputing it would be a different
feature, requiring every model to be retrained.

### If you are building on this

Do not read `vdW` as a hard-core volume in a packing or free-volume argument without correcting
it. The paper's packing screen is written to be safe under this: because the stored envelope
*understates* the true occupied volume, the screen is a **lower bound** on occupancy and therefore
**errs toward retention** — it never discards a record on the basis of an inflated volume.

---

## Reproducing the CSVs

`build_dataset.py` performs the ILThermo pull, the range filters and the molecule-level split:

```bash
python src/build_dataset.py --max-entries 0 --p-min 95.0 --p-max 10001.0 --sleep 2.0
```

(`--p-max 10001.0` reproduces the paper's set exactly: the bound is inclusive and the highest
pressure in the data is 10,001.0 kPa.)

**But that is not the whole story, and we will not pretend otherwise.** The physically-motivated
curation applied afterwards — the packing screen, the two-source disagreement handling, the
tiered removals — was performed **by hand**, guided by a written specification, not by a script.
The frozen CSVs are the artifact. Every criterion is recomputable and checkable against them, and
the counts reproduce exactly, but the cut is a documented editorial act rather than an executable
one. This is the one place in the pipeline that is not push-button.
