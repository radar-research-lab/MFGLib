## Getting started

`MFGLib` uses `poetry` to manage its dependencies. You should follow their installation
instructions [here](https://python-poetry.org/docs/#installation). 

Once you have `poetry` installed on your machine, run `$ poetry install` from within the project
root to install `MFGLib` and all its dependencies. 

`$ poetry install` will install several tools to lint and test the codebase. These include

- `black`: run via `$ black .`
- `isort`: run via `$ isort .`
- `mypy`: run via `$ mypy`
- `ruff`: run via `$ ruff mfglib tests` 
- `pytest`: run via `$ pytest`

All of the above checks are run in the CI.

## Documentation

By default, `$ poetry install` will not install the necessary documentation dependenices. If you wish to build and 
serve the documentation locally, first run `$ poetry install --with docs` to and then

```shell
$ sphinx-autobuild docs/source docs/build
```

To ensure that the documentation build successfully (with no errors), run

```shell
sphinx-build -EW docs/source docs/build
```
