# Installation

To install the most recent tagged release of `pengwann`, use `pip`:

```shell
pip install pengwann
```

Alternatively, if you would like to install the bleeding-edge latest version of the code (i.e. the current Git commit), you can build from source with:

```shell
pip install git+https://github.com/PatrickJTaylor/pengwann.git
```

:::{important}
Note that building `pengwann` from source entails compiling a small Rust extension, meaning that a suitable version of the Rust compiler must be available on the host machine.
This requirement does not usually hold when installing from PyPI, as pre-built wheels are provided for most commonly used platforms and architectures.

If you do need to build from source and your host machine is lacking the Rust compiler, the [rustup](https://rustup.rs/) utility can easily be used to install it.
:::

## Platform support

`pengwann` has **Tier 1** support for:

- 🐧 Linux (x86_64)
- 🍎 MacOS (x86_64)
- 🍎 MacOS (aarch64)
- 🪟 Windows (x86_64)

Providing **Tier 1** support means that `pengwann` is continuously built and tested against all the above platforms.

`pengwann` has **Tier 2** support for:

- 🐧 Linux (x86)
- 🐧 Linux (aarch64)
- 🐧 Linux (armv7)
- 🐧 Linux (ppc64le)
- 🐧 Linux (s390x)
- 🪟 Windows (x86)

Providing **Tier 2** support means that `pengwann` provides pre-built wheels for the above platforms, but they are not extensively tested in a continuous manner.
Practically speaking this means that the stability of `pengwann` on **Tier 2** systems is likely to vary.

## Building the documentation

If you would like to build a local version of the `pengwann` documentation, you should install the `docs` extras:

```shell
git clone https://github.com/PatrickJTaylor/pengWann.git
cd pengWann
pip install -e '.[docs]'
```

Once `pip` has finished, navigate to the docs and run `make` to build the documentation:

```shell
cd docs
make html
```

## Setting up a development environment

If you are interested in contributing to `pengwann`, a suitable development environment can be easily set up using [uv](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/PatrickJTaylor/pengWann.git
cd pengWann
uv sync
```

For more details, see the [contributing guide](./CONTRIBUTING).
