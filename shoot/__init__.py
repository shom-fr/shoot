"""Shom Ocean Objects Tracker"""
import warnings

from . import eddies


try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"


class ShootError(Exception):
    pass


class ShootWarning(UserWarning):
    pass


def shoot_warn(message, stacklevel=2):
    """Issue a :class:`ShootWarning` warning"""
    warnings.warn(message, ShootWarning, stacklevel=stacklevel)
