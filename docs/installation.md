# Installation

To install the most recent tagged release of `pengwann`, use `pip`:

```
pip install pengwann
```

Alternatively, if you would like to install the bleeding-edge latest version of the code (i.e. the current Git commit):

```
pip install git+https://github.com/PatrickJTaylor/pengwann.git
```

## Building the documentation

If you would like to build a local version of the `pengwann` documentation, you should install the `docs` extras:

```
pip install pengwann[docs]
```

Once `pip` has finished, navigate to the docs and run `make` to build the documentation:

```
cd docs
make html
```

## Setting up a development environment

If you are interested in contributing to `pengwann`, a suitable development environment can be easily set up using [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/PatrickJTaylor/pengWann.git
cd pengWann
uv sync
```
