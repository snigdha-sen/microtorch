# Contributing

We welcome contributions from the diffusion MRI community.

To propose a new feature or improvement:

1.  Fork the repository.
2.  Create a new branch named after your feature
    (e.g. `new-compartment`).
3.  Implement your changes.
4.  Open a Pull Request (PR) to the `main` branch.
5.  A maintainer will review your contribution.

Please ensure your code is well documented and tested before submitting
a PR. We encourage contributors to open an issue first if they would like to
discuss substantial changes before implementation.

### Where to Start

-   Adding a new microstructure compartment? See
    [adding_compartments.md](docs/developer/adding_compartments.md).
-   Adding a new model (a combination of compartments)? See
    [adding_models.md](docs/developer/adding_models.md).
-   Writing or running tests? See [testing.md](docs/developer/testing.md).

### Before Opening a PR

-   Run `pytest tests -v --cov --cov-report=term-missing` and make sure
    it passes.
-   Run `ruff check .` and `ruff format --check .` - these are the same
    checks CI runs on every PR.
-   Add tests for any new compartments, models, or other non-trivial logic.