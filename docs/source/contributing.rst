Contributing
============

Environment
-----------

MFGLib uses ``poetry`` to manage its dependencies. You can learn more about ``poetry``, including
installation instructions `here <https://python-poetry.org/>`_.

Once you have ``poetry`` installed on your machine, run

.. code-block:: bash

    $ poetry install

from within MFGLib's project root to install the library and its dependencies.

Documentation
-------------

To keep your environment lightweight, the command ``$ poetry install`` does not install
any dependencies related to documentation. If you wish to build and serve the documentation
locally, run

.. code-block:: bash

    $ poetry install --with doc

When that is complete, you can run

.. code-block:: bash

    $ poetry run sphinx-autobuild docs/source docs/build

from within the project root to serve the site on your local machine. ``sphinx-autobuild`` will
hot-reload your local version of the documentation as you make changes.

Changelog
---------

Any pull request that makes a noteworthy change to MFGLib should include a new entry in
``MFGLib/changelog.d/``. This directory stores "newsfragments" which get compiled and
prepended to ``CHANGELOG.md`` when a new version of MFGLib is released.

To generate a newsfragment, run the command

.. code-block::

    $ poetry run towncrier create -c "{description of the change}" {issue}.{type}.md

If your change corresponds with a GitHub issue, replace ``{issue}`` with the issue number.
If there is no corresponding GitHub issue, replace ``{issue}`` with a unique identifier starting
with ``+``. The ``{type}`` placeholder should be replaced by one of

* ``security``
* ``removed``
* ``deprecated``
* ``added``
* ``changed``
* ``fixed``


Lints and Checks
----------------

Pull requests are subjected to several code quality checks. To ensure your
code adheres to our formatting requirements, run

.. code-block:: bash

    $ poetry run black .
    $ poetry run isort .

To lint the code, run

.. code-block:: bash

    $ poetry run mypy
    $ poetry run ruff mfglib tests

Finally, you can run the tests with

.. code-block:: bash

    $ poetry run pytest

As written, all commands are assuming execution from the project root.