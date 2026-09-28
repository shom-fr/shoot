#!/usr/bin/env python3
"""
Parallel processing utilities

Generic helpers to run tasks in a pool of worker processes.
"""

import multiprocessing as mp
import os

import numba
import threadpoolctl


def get_nb_procs(nb_procs=None):
    """Number of processes to use, limited to the available cores

    Parameters
    ----------
    nb_procs : int, optional
        Requested number of processes. Defaults to the available cores.

    Returns
    -------
    int
    """
    ncores = len(os.sched_getaffinity(0))
    return min(nb_procs, ncores) if nb_procs else ncores


def can_auto_paral():
    """Whether parallelism can be automatically switched on

    Only with the "fork" start method: the other methods import the main
    script in the workers, which fails in scripts without an
    ``if __name__ == "__main__":`` guard.

    Returns
    -------
    bool
    """
    return mp.get_start_method() == "fork"


def _as_warmups(warmup):
    """List of warmup functions from None, a function or a sequence of functions"""
    if warmup is None:
        return []
    if callable(warmup):
        return [warmup]
    return list(warmup)


def _run_warmups(warmups):
    for warmup in warmups:
        warmup()


def _init_worker(warmups):
    # One thread per worker: multi-threaded BLAS in each worker
    # oversubscribes the cores and slows down the computations
    threadpoolctl.threadpool_limits(1)
    numba.set_num_threads(1)
    _run_warmups(warmups)


def create_pool(nb_procs=None, warmup=None):
    """Create a pool of single-threaded worker processes

    Each worker limits BLAS, OpenMP and numba to one thread, to avoid
    oversubscribing the cores.

    Parameters
    ----------
    nb_procs : int, optional
        Number of processes, limited to the available cores.
    warmup : callable or list of callable, optional
        Functions without argument that compile the numba kernels used by
        the tasks, like :func:`shoot.core.eddies.warmup`. They are called before
        starting the workers, so that forked workers inherit the compiled
        kernels, and at worker startup for the other start methods.
        They must be picklable, i.e. defined at module level.

    Returns
    -------
    multiprocessing.pool.Pool

    Example
    -------
    >>> from shoot.core.eddies import warmup
    >>> from shoot.paral import create_pool
    >>> with create_pool(4, warmup=warmup) as pool:  # doctest: +SKIP
    ...     eddies = Eddies2D.detect_eddies(u, v, 50, pool=pool)
    """
    warmups = _as_warmups(warmup)
    _run_warmups(warmups)
    return mp.Pool(get_nb_procs(nb_procs), initializer=_init_worker, initargs=(warmups,))
