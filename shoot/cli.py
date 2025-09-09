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
from .eddies import eddies2d as seddies
from .eddies import track as strack
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
        parser.exit(
            0, "please use one of the subcommands: " f"{args.subcommands}\n"
        )
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
    add_parser_eddies_track(subparsers_eddies)
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
    parser.add_argument(
        "nc_data_file", help="input netcdf data file", nargs="+"
    )
    parser.add_argument(
        "--to-netcdf",
        help="save detections to this netcdf file",
        default="eddies.detect.nc",
    )
    parser.add_argument(
        "--to-figure",
        help="save detections to this figure file",
        default="eddies.detect.png",
    )
    parser.add_argument(
        "--window-center",
        help="window size in km to find eddy centers",
        default=50,
        type=float,
    )
    parser.add_argument(
        "--window-fit",
        help="window size in km to fit streamfunction and find contours",
        default=120,
        type=float,
    )
    parser.add_argument(
        "--min-radius",
        help="minimal eddy radius in km",
        default=20,
        type=float,
    )
    parser.add_argument(
        "--max-ellipse-error",
        help="maximal ellipse relative error (<1)",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--without-ssh",
        help="do not use dataset ssh to compute contours",
        action="store_true",
    )
    parser.add_argument("--u-name", help="name of the U variable")
    parser.add_argument("--v-name", help="name of the V variable")
    parser.add_argument(
        "--ssh-name", help="name of the SSH or streamfunction variable"
    )
    parser.add_argument(
        "--parallel", help="use parallel mode", action="store_true"
    )
    parser.add_argument(
        "--nb-procs", help="number of procs to use in parallel mode", type=int
    )
    parser.add_argument(
        "--all", help="Detect on all time steps", action="store_true"
    )
    parser.add_argument(
        "--plot", help="Plot first time detection", action="store_true"
    )


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
        if not args.all:
            ds = ds.isel({time.name: 0})

    # Get variables
    u = ds[args.u_name] if args.u_name else scf.get_u(ds)
    v = ds[args.v_name] if args.v_name else scf.get_v(ds)
    if not args.without_ssh:
        ssh = (
            ds[args.ssh_name]
            if args.ssh_name
            else scf.get_ssh(ds, errors="warn")
        )
    else:
        ssh = None

    # Detect
    logger.debug("Starting detections")
    if args.all:
        eddies = seddies.EvolEddies2D.detect_eddies(
            ds,
            window_center=args.window_center,
            window_fit=args.window_fit,
            u=args.u_name if args.u_name else None,
            v=args.v_name if args.v_name else None,
            ssh=args.ssh_name if args.ssh_name else None,
            min_radius=args.min_radius,
            paral=args.parallel,
            nb_procs=args.nb_procs,
            ellipse_error=args.max_ellipse_error,
        )

    else:
        eddies = seddies.Eddies2D.detect_eddies(
            u,
            v,
            window_center=args.window_center,
            window_fit=args.window_fit,
            ssh=ssh,
            min_radius=args.min_radius,
            paral=args.parallel,
            nb_procs=args.nb_procs,
            ellipse_error=args.max_ellipse_error,
        )

    logger.info("Detections finished")

    # Save
    logger.debug("Saving detections to netcdf")
    eddies.save(args.to_netcdf)
    logger.info(f"Detections saved to: {args.to_netcdf}")

    # Plot
    if args.plot:
        logger.debug("Plotting detections")
        if args.all:
            ds = ds.isel({time.name: 0})
        fig, ax = splot.create_map(
            ds.cf["longitude"], ds.cf["latitude"], figsize=(8, 5)
        )
        ds.adt.plot(
            ax=ax,
            transform=splot.pcarr,
            add_colorbar=False,
            cmap="Spectral_r",
            alpha=0.6,
        )

        if args.all:
            plt.quiver(
                ds.cf["longitude"].values,
                ds.cf["latitude"].values,
                u[0].values,
                v[0].values,
                transform=splot.pcarr,
            )
            for eddy in eddies.eddies[0].eddies:
                eddy.plot(transform=splot.pcarr, lw=1)
        else:
            plt.quiver(
                ds.cf["longitude"].values,
                ds.cf["latitude"].values,
                u.values,
                v.values,
                transform=splot.pcarr,
            )
            for eddy in eddies.eddies:
                eddy.plot(transform=splot.pcarr, lw=1)
        plt.title(
            f"w_center {args.window_center} km, w_fit {args.window_fit}km, min_rad {args.min_radius}km"
        )
        plt.tight_layout()
        plt.savefig(args.to_figure)
        logger.info(f"Detections plot saved to: {args.to_figure}")


# %% eddies track


def add_parser_eddies_track(subparsers):
    parser_eddies_track = subparsers.add_parser(
        "track",
        help="detect and track eddies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments_eddies_track(parser_eddies_track)
    parser_eddies_track.set_defaults(func=main_eddies_track)
    return parser_eddies_track


def add_arguments_eddies_track(parser):
    parser.add_argument(
        "nc_data_file", help="input netcdf data file", nargs="+"
    )
    parser.add_argument(
        "--nbackward",
        help="number of backward step possible or tracking",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--to-netcdf",
        help="save tracking to this netcdf file",
        default="eddies.track.nc",
    )
    parser.add_argument(
        "--to-figure",
        help="save detections to this figure file",
        default="eddies.track.png",
    )
    parser.add_argument(
        "--window-center",
        help="window size in km to find eddy centers",
        default=50,
        type=float,
    )
    parser.add_argument(
        "--window-fit",
        help="window size in km to fit streamfunction and find contours",
        default=120,
        type=float,
    )
    parser.add_argument(
        "--min-radius",
        help="minimal eddy radius in km",
        default=20,
        type=float,
    )
    parser.add_argument(
        "--max-ellipse-error",
        help="maximal ellipse relative error (<1)",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--without-ssh",
        help="do not use dataset ssh to compute contours",
        action="store_true",
    )
    parser.add_argument("--u-name", help="name of the U variable")
    parser.add_argument("--v-name", help="name of the V variable")
    parser.add_argument(
        "--ssh-name", help="name of the SSH or streamfunction variable"
    )
    parser.add_argument(
        "--parallel", help="use parallel mode", action="store_true"
    )
    parser.add_argument(
        "--nb-procs", help="number of procs to use in parallel mode", type=int
    )
    parser.add_argument(
        "--plot", help="Plot first time detection", action="store_true"
    )

    parser.add_argument(
        "--update",
        help="Update tracking based on the already traked file",
        nargs=1,
        type=str,
    )
    parser.add_argument(
        "-b",
        "--begin",
        help="Start date for tracking (yyyy/mm/jj)",
        nargs=1,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--end",
        help="Start date for tracking (yyyy/mm/jj)",
        nargs=1,
        type=str,
    )


def _eddies_track(parser, args, logger, ds):
    # Detect
    logger.debug("Starting detections")
    eddies = seddies.EvolEddies2D.detect_eddies(
        ds,
        window_center=args.window_center,
        window_fit=args.window_fit,
        u=args.u_name if args.u_name else None,
        v=args.v_name if args.v_name else None,
        ssh=args.ssh_name if args.ssh_name else None,
        min_radius=args.min_radius,
        paral=args.parallel,
        nb_procs=args.nb_procs,
        ellipse_error=args.max_ellipse_error,
    )
    logger.info("Detections finished")

    # track
    logger.debug("Starting tracking")
    tracks = strack.track_eddies(eddies, args.nbackward)
    logger.info("tracking finished")

    # Save
    logger.debug("Saving tracking to netcdf")
    tracks.save(args.to_netcdf)
    logger.info(f"Detections saved to: {args.to_netcdf}")
    return eddies, tracks


def _eddies_update(parser, args, logger, ds):

    time = scf.get_time(ds)
    if time is not None:
        logger.warning("Selecting the first time step")
        ds = ds.isel({time.name: -1})

    # Get variables
    u = ds[args.u_name] if args.u_name else scf.get_u(ds)
    v = ds[args.v_name] if args.v_name else scf.get_v(ds)
    if not args.without_ssh:
        ssh = (
            ds[args.ssh_name]
            if args.ssh_name
            else scf.get_ssh(ds, errors="warn")
        )
    else:
        ssh = None

    logger.debug("Starting detection at last day")
    new_eddies = seddies.Eddies2D.detect_eddies(
        u,
        v,
        window_center=args.window_center,
        window_fit=args.window_fit,
        ssh=ssh,
        min_radius=args.min_radius,
        paral=args.parallel,
        nb_procs=args.nb_procs,
        ellipse_error=args.max_ellipse_error,
    )
    logger.info("Detections finished")

    # load past tracks
    logger.debug("Loading already tracked file")
    ds_track = xr.open_dataset(args.update[0])
    # update
    tracks_refresh = strack.update_tracks(ds_track, new_eddies, args.nbackward)
    logger.debug("Update the track finished")

    # Save
    logger.debug("Saving tracking to netcdf")
    tracks_refresh.save(args.to_netcdf)
    logger.info(f"Detections saved to: {args.to_netcdf}")
    return new_eddies, tracks_refresh


def main_eddies_track(parser, args):
    logger = logging.getLogger(__name__)

    # Open files
    if len(args.nc_data_file) == 1:
        ds = xr.open_dataset(args.nc_data_file[0])
    else:
        ds = xr.open_mfdataset(args.nc_data_file)

    # time range
    time = scf.get_time(ds)
    if args.begin or args.end:
        begin = (
            args.begin[0]
            if args.begin
            else str(time[0].dt.strftime("%Y/%m/%d").values)
        )
        end = (
            args.end[0]
            if args.end
            else str(time[-1].dt.strftime("%Y/%m/%d").values)
        )
        print(begin, end)
        ds = ds.sel({time.name: slice(begin, end)})

    if args.update:
        eddies, tracks = _eddies_update(parser, args, logger, ds)
    else:
        eddies, tracks = _eddies_track(parser, args, logger, ds)

    # Plot
    if args.plot:
        logger.debug("Plotting detections")
        ds = ds.isel({time.name: -1})
        u = ds[args.u_name] if args.u_name else scf.get_u(ds)
        v = ds[args.v_name] if args.v_name else scf.get_v(ds)
        fig, ax = splot.create_map(
            ds.cf["longitude"], ds.cf["latitude"], figsize=(8, 5)
        )
        ds.adt.plot(
            ax=ax,
            transform=splot.pcarr,
            add_colorbar=False,
            cmap="Spectral_r",
            alpha=0.6,
        )
        plt.quiver(
            ds.cf["longitude"].values,
            ds.cf["latitude"].values,
            u.values,
            v.values,
            transform=splot.pcarr,
        )

        for eddy in eddies.eddies[-1].eddies:
            eddy.plot(transform=splot.pcarr, lw=1)
            plt.text(
                eddy.glon,
                eddy.glat,
                eddy.track_id,
                c="w",
                transform=splot.pcarr,
            )
            track = tracks.track_eddies[eddy.track_id]
            lon, lat = [], []
            for e in track.eddies:
                lon.append(e.glon)
                lat.append(e.glat)
            plt.plot(lon, lat, transform=splot.pcarr, c="gray", linewidth=2)

        plt.title(
            f"w_center {args.window_center} km, w_fit {args.window_fit}km, min_rad {args.min_radius}km"
        )
        plt.tight_layout()
        plt.savefig(args.to_figure)
        logger.info(f"Detections plot saved to: {args.to_figure}")
