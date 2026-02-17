.. _cli:

Commandline interface
#####################

Available commands:

.. toctree::
    :maxdepth: 1
    
    cli.shoot
    cli.shoot.eddies
    cli.shoot.eddies.detect
    cli.shoot.eddies.track
    cli.shoot.eddies.track-detected
    cli.shoot.eddies.diags



:command:`shoot`
================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot


:command:`shoot eddies`
=======================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies


:command:`shoot eddies detect`
==============================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies detect
    :nosubcommands:

:command:`shoot eddies track`
=============================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies track
    :nosubcommands:

:command:`shoot eddies diags`
=============================

.. argparse::
    :module: shoot.cli
    :func: get_parser
    :prog: shoot
    :path: eddies diags
    :nosubcommands:

