Contributing to shoot
=====================

We welcome contributions to shoot! This document provides guidelines for contributing to the project.

Reporting Issues
----------------

If you find a bug or have a feature request, please open an issue on the GitHub repository:

1. Check if the issue already exists in the `issue tracker <https://github.com/yourusername/shoot/issues>`_
2. If not, create a new issue with a clear title and description
3. For bugs, include:

   * A minimal reproducible example
   * Your environment (Python version, OS, relevant package versions)
   * Expected vs. actual behavior
   * Error messages and stack traces

4. For feature requests, explain:

   * The use case and motivation
   * Proposed API or behavior
   * Any alternative solutions you've considered

Getting Started
---------------

Fork and Clone
~~~~~~~~~~~~~~

1. Fork the repository on GitHub by clicking the "Fork" button
2. Clone your fork locally::

    git clone https://github.com/yourusername/shoot.git
    cd shoot

3. Add the upstream repository::

    git remote add upstream https://github.com/originalowner/shoot.git

Create a Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a virtual environment::

    conda create -n shoot-dev python=3.11
    conda activate shoot-dev

2. Install the package in development mode with dependencies::

    pip install -e ".[dev]"

3. Install pre-commit hooks (if available)::

    pre-commit install

Making Changes
--------------

Branching Strategy
~~~~~~~~~~~~~~~~~~

1. Create a new branch for your changes::

    git checkout -b feature/your-feature-name
    # or
    git checkout -b fix/issue-number-description

2. Keep your branch focused on a single feature or fix
3. Use descriptive branch names (e.g., ``feature/add-acoustic-analysis``, ``fix/profile-time-conversion``)

Coding Standards
~~~~~~~~~~~~~~~~

* Follow `PEP 8 <https://pep8.org/>`_ style guidelines
* Write clear, self-documenting code with meaningful variable names
* Add docstrings to all public functions, classes, and methods using NumPy style
* Keep functions focused and modular
* Avoid over-engineering - implement what is needed, not what might be needed

Documentation
~~~~~~~~~~~~~

* Update documentation for any changed functionality
* Add docstrings following the NumPy documentation style::

    def function_name(param1, param2):
        """Short description of function

        Longer description if needed.

        Parameters
        ----------
        param1 : type
            Description of param1.
        param2 : type
            Description of param2.

        Returns
        -------
        type
            Description of return value.
        """

* Document any new features in the appropriate ``.rst`` files in the ``docs/`` directory

Testing
~~~~~~~

* **Important**: Don't generate too many tests (as noted in project guidelines)
* Add tests for new functionality or bug fixes
* Ensure all tests pass before submitting::

    pytest tests/

* Check test coverage if relevant::

    pytest --cov=shoot tests/

Using xoa
~~~~~~~~~

The project uses xoa for metadata handling:

* All metadata/CF-related functionality should use xoa from the ``xoa/`` directory
* Do not add dependencies on cf_xarray or similar packages
* See ``shoot/meta.py`` for examples of wrapping xoa functionality

Committing Changes
------------------

1. Stage your changes::

    git add <files>

2. Commit with a clear, descriptive message::

    git commit -m "Add feature: brief description

    More detailed explanation of what changed and why.
    Fixes #issue-number (if applicable)"

3. Keep commits focused and atomic
4. Write commit messages in the imperative mood ("Add feature" not "Added feature")

Submitting a Pull Request
--------------------------

1. Push your branch to your fork::

    git push origin feature/your-feature-name

2. Go to the GitHub repository and click "New Pull Request"

3. Select your branch and provide a clear description:

   * What changes were made
   * Why the changes were needed
   * Related issue numbers (e.g., "Fixes #123")
   * Any breaking changes or migration notes

4. Ensure all CI checks pass

5. Respond to any code review feedback

6. Once approved, a maintainer will merge your PR

Syncing with Upstream
~~~~~~~~~~~~~~~~~~~~~

Keep your fork up to date with the main repository::

    git fetch upstream
    git checkout main
    git merge upstream/main
    git push origin main

Code Review Process
-------------------

* All contributions require review before merging
* Reviewers may request changes or improvements
* Be patient and responsive to feedback
* Discussion and iteration are part of the process

Questions?
----------

If you have questions about contributing, feel free to:

* Open an issue with the "question" label
* Reach out to the maintainers

Thank you for contributing to shoot!
