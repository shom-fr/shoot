#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Managing data samples
"""
import os
import pooch

pooch.get_logger().setLevel("DEBUG")

POOCH = pooch.create(
    path=pooch.os_cache("woom"),
    base_url="https://raw.githubusercontent.com/shom-fr/data-samples/refs/heads/{version}/",
    version_dev="main",
    env="WOOM_SAMPLES",
)

POOCH.load_registry(os.path.join(os.path.dirname(__file__), "samples.txt"))


def get_sample_file(sample_name):
    """Get the path to the sample file name

    Parameters
    ----------
    sample_name: str
        Base name of the file

    Return
    ------
    str
    """
    return POOCH.fetch(sample_name)
