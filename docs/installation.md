# Installation

## Stable release

Use pip to install from PyPI:

Install from pip:

```
pip install pysatproc
```

## From source

The source for satproc can be installed from the GitHub repo.

```
python -m pip install git+git://github.com/dymaxionlabs/satproc.git#egg=pysatproc
```

To install for local development:

```
git clone git@github.com:dymaxionlabs/satproc.git
cd satproc
python -m pip install -e .[dev]
```

Now, whenever you want to bring the latest changes, just run `git pull` from the
cloned repository.