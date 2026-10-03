# Contributing to Q-OMICS

Thanks for looking. This project is a research codebase that takes itself
seriously about not overstating what it measures, and contributions are held
to the same standard.

## Before you start

Open an issue first for anything non-trivial. A short conversation about the
approach is cheaper than a pull request that needs rewriting.

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/Q-OMICS.git
cd Q-OMICS
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"

pytest -q
```

## The standard this project holds itself to

**A result that cannot fail is not a result.** If you add a model, it gets
compared against the existing baselines on identical folds, with a paired test
and a multiplicity correction. If you add a quantum component, it gets a
geometric-difference measurement against a classical reference. If the new
component does not beat what was already there, say so in the pull request —
that is a publishable finding, not a failure.

**Leakage control is not negotiable.** Anything fitted from data — scaler,
variance filter, PCA, quantum embedding, a stacking model — is fitted on
training rows only. When you add a transformation, add a test that would fail
if it were fitted on the full cohort.

**Verify the physics.** If you add a feature map and it factorises, add a
closed-form check of the Gram matrix and assert the simulator agrees. The
existing `sep_rx` control exists precisely so the backend can be checked.

**State limitations.** Every module that could mislead a reader has a
limitations note. If you find a case not covered, add it.

## Code conventions

- Type hints on public functions. Line length 100.
- Docstrings explain *why*, not *what*. The code already says what.
- `ruff check qomics tests` must pass. `ruff format` for formatting.
- Comment the non-obvious invariant, not the obvious mechanic. A comment that
  restates the next line is noise; a comment explaining why a test-time kernel
  row is divided by the training norms is documentation.

## Tests

Every bug fix gets a regression test. If you fixed something, the test must
fail on the old code — otherwise it is not testing the bug.

```bash
pytest -q                          # everything
pytest tests/test_quantum.py -v    # one module
pytest -q -k "cross_block"         # one behaviour
```

The suite runs without network access and without a trained model. If your
change makes a test require either, that is a signal the test belongs in
`scripts/` as a diagnostic rather than in `tests/`.

## Commit messages

```
fix(kernel): normalise cross blocks by training-set state norms

A test-time kernel row divided by its own diagonal norm is scaled
differently from every training row, so the SVM computes its margin in
the wrong geometry. Pass the training norms through instead.

Fixes #42
```

Short imperative subject, blank line, body explaining the reasoning, issue
reference.

## Pull requests

- One logical change per PR.
- Update `docs/METHODOLOGY.md` if the method changes.
- Update `docs/API_CONTRACT.md` **and** `qomics/serving/app.py` together if the
  wire format changes — they are two views of one contract, and the frontend
  is written against the document.
- Include the command you ran and the output, so a reviewer can reproduce it
  without a GPU.

## Adding a cohort or a modality

See `docs/ARCHITECTURE.md#extending`. The short version: a new cohort is a
`CohortConfig` change, a new modality is a block in `Cohort.blocks`, and both
propagate through the preprocessor, the token lift and the biomarker
annotation without further edits.

## Reporting a security or privacy issue

Do not open a public issue. Email the maintainers directly. This project handles
no patient data, but a bug that caused it to persist identifiable data from a
public archive would still be a serious problem.

## Licence

Contributions are accepted under the MIT licence in `LICENSE`. By contributing
you confirm you have the right to license the code under those terms.
