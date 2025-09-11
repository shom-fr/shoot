Commandline interface to shoot.

:command:`shoot`
================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :subcommands:eddies


:command:`shoot eddies`
=======================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies
    :subcommands:track detect diags


:command:`shoot eddies detect`
=======================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies detect
    :nosubcommands:

:command:`shoot eddies track`
=======================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies track
    :nosubcommands:

:command:`shoot diags`
=======================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies diags
    :nosubcommands:

