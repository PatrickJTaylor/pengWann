# Contributing

Contributions both big and small to the `pengwann` codebase and/or its associated documentation are very welcome!

## Guidelines

If you are going to submit a pull request to `pengwann`, please first check that it meets these general guidelines if possible:

1. New/modified functionality should be accompanied by new/appropriately modified **tests**.

2. New/modified functionality should be accompanied by new/appropriately modified **documentation**.
    - In keeping with the current codebase, please use [numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings to document all Python source code.

3. Where possible, write new code in a somewhat [functional](https://en.wikipedia.org/wiki/Functional_programming) style[^1].

4. Write new code for Python >= 3.10 only i.e., avoid deprecated code and coding patterns designed for Python <= 3.9.

[^1]: More specifically, write **pure** functions (i.e., no side effects) and use **immutable** data structures. Python is not a functional language so there are naturally limits, but the main codebase generally adheres to these requirements, so contributions should ideally reflect this.

## Recommended developer workflow

Contributions to `pengwann` should be made via the [fork and pull](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) framework:

1. Fork the `pengwann` repository on GitHub.

2. Clone a local version of your new fork:

```shell
git clone git@github.com:your_username_here/pengWann.git
```

3. Set up a development environment with [uv](https://docs.astral.sh/uv/):

```shell
cd pengWann
uv sync
```

Note that, as mentioned in the [installation guide](./installation), building `pengwann` from source (which will happen automatically when `uv` syncs the environment) requires the Rust compiler (see [rustup](https://rustup.rs/) for installation).

4. Create and switch to a new branch:

```shell
git switch -c name_of_new_feature
```

5. Implement the changes that you would like to contribute.

6. If you have modified the Python source, run `ruff`, `basedpyright` and `pytest` to lint, format, type check and run the test suite:

```shell
uv run ruff check
uv run ruff format
uv run basedpyright
uv run pytest
```

If you have modified the Rust source, run `clippy` and `rustfmt` to lint and format:

```shell
cargo clippy
cargo fmt
```

Regardless of whether or not you validate your changes in the manner shown above, they will be checked in CI when you open a pull request.

7. Commit your changes and push to GitHub:

```shell
git add src/pengwann/changed_file.py
git commit -m 'Short description of changes made.'
git push origin name_of_new_feature
```

8. Submit a [pull request](https://github.com/PatrickJTaylor/pengWann/pulls).
