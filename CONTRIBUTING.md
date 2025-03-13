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

## CHANGELOG

Any pull request that makes a significant change to `MFGLib` should include a new entry in `MFGLib/changelog.d/`.
To generate a new entry, run the command

```shell
$ towncrier create -c "{description of the change}" {issue}.{type}.md
```

If your change corresponds with a GitHub issue, replace `{issue}` with the issue number. If there
is no corresponding GitHub issue, replace `{issue}` with a unique identifier starting with `+`. The 
`{type}` placeholder should be replaced by one of

- `security`
- `removed`
- `deprecated`
- `added`
- `changed`
- `fixed`
