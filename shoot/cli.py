#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface
"""
import argparse
import logging

import xarray as xr
import cf_xarray  # noqa
import matplotlib.pyplot as plt

from . import log as slog
from . import eddies as seddies
from . import cf as scf
from . import plot as splot

# %% Main


def get_parser():
    parser = argparse.ArgumentParser(
        description="shoot interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    slog.add_logging_parser_arguments(parser)

    subparsers = parser.add_subparsers(help="sub-command help")

    add_parser_eddies(subparsers)

    return parser


def main():
    # Get the parser
    parser = get_parser()

    # Parse args
    args = parser.parse_args()
    slog.main_setup_logging(args)

    # Call subparser function
    if hasattr(args, "func"):
        args.func(parser, args)
    elif hasattr(args, "subcommands"):
        parser.exit(0, "please use one of the subcommands: " f"{args.subcommands}\n")
    else:
        parser.print_usage()


# %% eddies


def add_parser_eddies(subparsers):
    # Setup argument parser
    parser_eddies = subparsers.add_parser(
        "eddies",
        help="eddies detection and tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers_eddies = parser_eddies.add_subparsers(help="sub-command help")
    add_parser_eddies_detect(subparsers_eddies)
    # add_parser_eddies_track(subparsers_eddies)
    # add_parser_eddies_detect_and_track(subparsers_eddies)

    return parser_eddies


# %% eddies detect


def add_parser_eddies_detect(subparsers):
    parser_eddies_detect = subparsers.add_parser(
        "detect",
        help="detect eddies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments_eddies_detect(parser_eddies_detect)
    parser_eddies_detect.set_defaults(func=main_eddies_detect)
    return parser_eddies_detect


def add_arguments_eddies_detect(parser):
    parser.add_argument("nc_data_file", help="input netcdf data file", nargs="+")
    parser.add_argument("--to-netcdf", help="save detections to this netcdf file", default="eddies.detect.nc")
    parser.add_argument("--to-figure", help="save detections to this figure file", default="eddies.detect.png")
    parser.add_argument(
        "--window-center", help="window size in km to find eddy centers", default=50, type=float
    )
    parser.add_argument(
        "--window-fit",
        help="window size in km to fit streamfunction and find contours",
        default=120,
        type=float,
    )
    parser.add_argument("--min-radius", help="minimal eddy radius in km", default=20, type=float)
    parser.add_argument(
        "--max-ellipse-error", help="maximal ellipse relative error (<1)", default=0.05, type=float
    )
    parser.add_argument("--without-ssh", help="do not use dataset ssh to compute contours", action="store_true")
    parser.add_argument("--u-name", help="name of the U variable")
    parser.add_argument("--v-name", help="name of the V variable")
    parser.add_argument("--ssh-name", help="name of the SSH or streamfunction variable")
    parser.add_argument("--parallel", help="use parallel mode", action="store_true")
    parser.add_argument("--nb-procs", help="number of procs to use in parallel mode", type=int)


def main_eddies_detect(parser, args):

    logger = logging.getLogger(__name__)

    # Open files
    if len(args.nc_data_file) == 1:
        ds = xr.open_dataset(args.nc_data_file[0])
    else:
        ds = xr.open_mfdataset(args.nc_data_file)
    time = scf.get_time(ds)
    if time is not None:
        logger.warning("Selecting the first time step")
        ds = ds.isel({time.name: 0})

    # Get variables
    u = ds[args.u_name] if args.u_name else scf.get_u(ds)
    v = ds[args.v_name] if args.v_name else scf.get_v(ds)
    if not args.without_ssh:
        ssh = ds[args.ssh_name] if args.ssh_name else scf.get_ssh(ds, errors="warn")
    else:
        ssh = None

    # Detect
    logger.debug("Starting detections")
    eddies = seddies.Eddies.detect_eddies(u, v, window_center=args.window_center, window_fit=args.window_fit, ssh=ssh, min_radius=args.min_radius, paral=args.parallel, nb_procs=args.nb_procs, ellipse_error=args.max_ellipse_error)
    logger.info("Detections finished")

    # Save
    logger.debug("Saving detections to netcdf")
    eddies.save(args.to_netcdf)
    logger.info(f"Detections saved to: {args.to_netcdf}")

    # Plot
    logger.debug("Plotting detections")
    fig, ax = splot.create_map(ds.cf["longitude"], ds.cf["latitude"], figsize=(8, 5))
    ds.adt.plot(ax=ax, transform=splot.pcarr, add_colorbar=False, cmap="Spectral_r", alpha=0.6)
    plt.quiver(ds.cf["longitude"].values, ds.cf["latitude"].values, u.values, v.values, transform=splot.pcarr)
    for eddy in eddies.eddies:
        eddy.plot(transform=splot.pcarr, lw=1)
    plt.title(f"w_center {args.window_center} km, w_fit {args.window_fit}km, min_rad {args.min_radius}km")
    plt.tight_layout()
    plt.savefig(args.to_figure)
    logger.info(f"Detections plot saved to: {args.to_figure}")

