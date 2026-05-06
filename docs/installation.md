# Installation

## Stable release

To install PEAgent, run this command in your terminal:

```sh
uv add peagent-tool
```

Or if you prefer to use `pip`:

```sh
pip install peagent-tool
```

Install model prediction support with TensorFlow:

```sh
pip install "peagent-tool[prediction]"
```

Install ISM attribution and plotting support:

```sh
pip install "peagent-tool[ism]"
```

## From source

The source files for PEAgent Tool can be downloaded from the [GitHub repo](https://github.com/YAOJ-bioin/peagent_tool).

You can either clone the public repository:

```sh
git clone git@github.com:YAOJ-bioin/peagent_tool.git
```

Or download the [tarball](https://github.com/YAOJ-bioin/peagent_tool/tarball/main):

```sh
curl -OJL https://github.com/YAOJ-bioin/peagent_tool/tarball/main
```

Once you have a copy of the source, you can install it with:

```sh
cd peagent_tool
uv pip install .
```
