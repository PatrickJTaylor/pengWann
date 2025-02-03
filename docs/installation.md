# Installation

To install the latest tagged release of `pengwann`, use `pip`:

```
pip install pengwann
```

## Latest

Alternatively, if you would like to install the latest version of the code (i.e. the current Git commit):

```
pip install git+https://github.com/PatrickJTaylor/pengwann.git
```

## Extras

Various optional dependencies are accessible for development and documentation purposes. If you would like to build the documentation locally, install `pengwann` with:

```
git clone https://github.com/PatrickJTaylor/pengWann.git
cd pengWann
pip install -e '.[docs]'
```

Once `pip` has finished, navigate to the docs and run `make` to generate whatever format you would like:

```
cd docs
make html
```

Note that you will also need to install [pandoc](https://pandoc.org/installing.html) for the above to run without error (unfortunately we cannot package this as a dependency directly as it is written in Haskell, not Python).

If you are interested in contributing to `pengwann`, then you may want to install the developer extras via:

```
git clone https://github.com/PatrickJTaylor/pengWann.git
cd pengWann
pip install -e '.[dev]'
```

This will install a number of optional dependencies for static type checking, unit testing, code styling etc.
