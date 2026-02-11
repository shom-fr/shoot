#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample data management

Functions for downloading and accessing example datasets via pooch.
"""
import os
import pooch

pooch.get_logger().setLevel("DEBUG")

POOCH = pooch.create(
    path=pooch.os_cache("shoot"),
    base_url="https://raw.githubusercontent.com/shom-fr/data-samples/main/OCEANO/",
    # version_dev="main",
    # env="WOOM_SAMPLES",
)

POOCH.load_registry(os.path.join(os.path.dirname(__file__), "samples.txt"))


def get_sample_file(sample_name):
    """Fetch sample data file

    Downloads sample file from repository if needed and returns local path.

    Parameters
    ----------
    sample_name : str
        Sample file name (e.g., "MODELS/CROCO/gigatl1-1000m.nc").

    Returns
    -------
    str
        Absolute path to cached sample file.
    """
    return POOCH.fetch(sample_name)
