# Contributing

Thanks for looking. This framework is meant to be **used and extended** — pointed at other
molecules, other properties, other equations of state. Forks and pull requests are welcome.

One thing is unusual, and it is worth stating plainly: `src/` is also the code that produced a
published benchmark. Changes are welcome, but a change that **moves a reported number** must say so
out loud. See below.

## Before you open a PR

Run the self-checks. They are the contract:

```bash
python scripts/selfcheck.py --with-baselines
```

All of them must pass. `scripts/verify_dataset.py` in particular asserts thirty numbers against the
manuscript — if it goes red, either the data changed or the paper is wrong, and both are serious.

## What is welcome

- **Extensions.** A different equation of state, a different supervision target, a different
  descriptor, a new conv type, a new baseline. The `Extending it` table in the README maps each of
  these to the one file it touches.
- **New chemistry.** The bounds box in `model.py` is tuned for ionic liquids. If you make this work
  on another class of molecules, that is a genuinely useful contribution — open an issue and say
  what box you needed.
- Bug fixes, especially anything in `docs/known-issues.md`.
- Portability fixes, packaging fixes, documentation.
- **New self-checks.** Two modules still have none (`tune.py`, `smiles2vdW_volume.py`).

## What needs discussion first

Anything that changes a **published number**. That is not a ban — it is a request for honesty. Open
an issue first, and in the PR show the before/after explicitly. A PR that silently shifts a reported
metric will be rejected on principle, even if the new number is better.

Adding capability alongside the existing path (a new flag, a new module, a new EoS) needs no such
ceremony. Prefer that shape where you can.

## Data

Do not modify `data/*.csv`. They are the frozen artifact. If you believe a record is wrong, open an
issue with the ILThermo entry ID (`ilt_id`) and the primary reference.

## Style

Match the file you are editing. The existing modules use a consistent voice: a substantial module
docstring that explains *why*, assertions instead of trust, and honest comments about what does not
work.
