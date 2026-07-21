## Running the Tests

Install the package with test dependencies:

```bash
pip install -e ".[test]"
```

Run the full test suite from the repository root:

```bash
pytest tests
```

To see a coverage report:

```bash
pytest tests --cov --cov-report=term-missing
```

Coverage settings live in `pyproject.toml` under `[tool.coverage.run]` and
`[tool.coverage.report]`. Current coverage of the in-scope modules (see
below) is around 94%, above the `fail_under = 85` threshold enforced there
and in CI.

Tests are organized to mirror the package layout:

```
tests/loss_functions/
tests/signal_models/
tests/networks/
tests/utils/
tests/test_model_maker.py
tests/test_net_maker.py
tests/test_train.py
tests/integration/
```

`tests/integration/test_fit_pipeline.py` is an end-to-end test: it simulates
a known ground truth with `make_test_image`, fits it with `run_fit` (the
same function `microtorch.main` calls), and checks the fitted parameters
recover the simulated ground truth. It trains a small network for real, so
it's slower than the unit tests, but needs no external data.

Installing `pip install -e ".[dev]"` gets you pytest, pytest-cov, and ruff
together - this is what CI runs.

## Adding Tests

All new compartments must include appropriate unit tests.

Tests should be added to:

    microtorch/tests/signal_models/

Please follow the structure and conventions of existing tests. Tests
should verify:

-   Correct parameter handling
-   Numerical stability
-   Expected output shape
-   Basic sanity checks of signal behaviour

## What Isn't Covered, and Why

Coverage targets the scientific and training logic: signal models,
`ModelMaker`, the network/training pipeline (`net_maker.py`, `train.py`,
`networks/`, `network_constraints.py`), preprocessing/acquisition utilities,
and the end-to-end fitting pipeline (`run_fit.py`).

A few modules are excluded from the coverage target (`[tool.coverage.run]
omit` in `pyproject.toml`) because they're thin orchestration/IO layers
around logic that's already tested elsewhere, rather than algorithmic code
in their own right:

-   `main.py` - the Hydra CLI entry point. Its only logic is argument
    validation plus calls to `run_fit` and `plot_param_maps`, both tested
    directly; testing it further would mostly test Hydra's own
    decorator/argument-parsing machinery.
-   `utils/plot_results.py` - matplotlib plotting functions that produce
    figures for visual inspection (see the example notebooks). Correctness
    here is about the figure being informative, which visual review covers
    better than an automated test would.
-   `utils/optuna_search.py` - the Optuna hyperparameter search loop. The
    hyperparameter-resolution logic it's called through
    (`get_model_hyperparams`, `tune="default"`) is exercised by the
    integration test; the search loop itself (`tune="optuna_tuner"`) mostly
    orchestrates Optuna's own optimization machinery, and running real
    trials in CI would be slow without adding much confidence in
    microtorch-specific logic.
-   `utils/create_all_test_images.py` - the batch-generation script behind
    the `microtorch-create-test-images` command. It loops `make_test_image`
    (which does have direct tests) over every model/config combination; it's
    I/O and looping around an already-tested function.

If you add non-trivial logic to any of these files, please add tests for it
and remove it from the `omit` list.
