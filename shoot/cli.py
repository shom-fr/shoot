#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface
"""
import argparse
import logging

import xarray as xr
import numpy as np
import cf_xarray  # noqa
import matplotlib.pyplot as plt
import gsw
import cmocean
from matplotlib.colors import ListedColormap, BoundaryNorm

from . import log as slog
from .eddies import eddies2d as seddies
from .eddies import track as strack
from . import hydrology as shydrology
from . import acoustic as sacoustic

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
    add_parser_eddies_track_detected(subparsers_eddies)
    add_parser_eddies_diags(subparsers_eddies)

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
        logger.warning("Selecting the last time step")
        # ds = ds.isel({time.name: -1})
        # print("ds.time", ds.time)

    # Get variables
    u = (
        ds[args.u_name].isel({time.name: -1})
        if args.u_name
        else scf.get_u(ds).isel({time.name: -1})
    )
    v = (
        ds[args.v_name].isel({time.name: -1})
        if args.v_name
        else scf.get_v(ds).isel({time.name: -1})
    )
    if not args.without_ssh:
        ssh = (
            ds[args.ssh_name].isel({time.name: -1})
            if args.ssh_name
            else scf.get_ssh(ds, errors="warn").isel({time.name: -1})
        )
    else:
        ssh = None

    logger.debug("Starting detection at last day")
    print("u", u)
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

        if args.update:
            for eddy in eddies.eddies:
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
                    lon.append(e.lon)
                    lat.append(e.lat)
                plt.plot(
                    lon, lat, transform=splot.pcarr, c="gray", linewidth=2
                )

        else:
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
                    lon.append(e.lon)
                    lat.append(e.lat)
                plt.plot(
                    lon, lat, transform=splot.pcarr, c="gray", linewidth=2
                )

        plt.title(
            f"w_center {args.window_center} km, w_fit {args.window_fit}km, min_rad {args.min_radius}km"
        )
        plt.tight_layout()
        plt.savefig(args.to_figure)
        logger.info(f"Detections plot saved to: {args.to_figure}")



# %% eddies track-detected


def add_parser_eddies_track_detected(subparsers):
    parser_eddies_track_detected = subparsers.add_parser(
        "track-detected",
        help="track already detected eddies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments_eddies_track_detected(parser_eddies_track_detected)
    parser_eddies_track_detected.set_defaults(func=main_eddies_track_detected)
    return parser_eddies_track_detected


def add_arguments_eddies_track_detected(parser):
    parser.add_argument(
        "nc_data_files", help="list of detected netcdf file", nargs="+"
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


def _eddies_track_detected(parser, args, logger, dss):
    # Detect
    logger.debug("Merge detected files")
    eddies = seddies.EvolEddies2D.merge_ds(
        dss
    )
    logger.info("Merge finished")

    # track
    logger.debug("Starting tracking")
    tracks = strack.track_eddies(eddies, args.nbackward)
    logger.info("tracking finished")

    # Save
    logger.debug("Saving tracking to netcdf")
    tracks.save(args.to_netcdf)
    logger.info(f"Detections saved to: {args.to_netcdf}")
    return eddies, tracks

def main_eddies_track_detected(parser, args):
    logger = logging.getLogger(__name__)

    # Open files
    dss=[xr.open_dataset(f) for f in args.nc_data_files]
    eddies, tracks = _eddies_track_detected(parser, args, logger, dss)

 
# %% Eddies diags


def add_parser_eddies_diags(subparsers):
    parser_eddies_diags = subparsers.add_parser(
        "diags",
        help="makes diags on already tracked/detected files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments_eddies_diags(parser_eddies_diags)
    parser_eddies_diags.set_defaults(func=main_eddies_diags)
    return parser_eddies_diags


def add_arguments_eddies_diags(parser):
    parser.add_argument(
        "nc_data_files",
        help="input netcdf 3d data and already tracked eddies files",
        nargs=2,
        type=str,
    )
    parser.add_argument(
        "--rfactor",
        help="distance in radius to search outside profiles (>1)",
        default=1.2,
        type=float,
    )

    parser.add_argument(
        "--to-figure",
        help="save detections to this figure file",
        default="diags.png",
    )

    parser.add_argument(
        "-d",
        "--date",
        help="date (yyyy/mm/jj)",
        nargs=1,
        type=str,
    )

    parser.add_argument(
        "--acoustic", help="Acoustic impact diag", action="store_true"
    )
    parser.add_argument(
        "--density", help="Acoustic impact diag", action="store_true"
    )


def _sel_eddies(eddies, date):
    ind_time = np.where(
        (eddies.time.values >= np.datetime64(date))
        & (eddies.time.values < np.datetime64(date) + np.timedelta64(1, "D"))
    )[0]
    eddies = eddies.isel(obs=ind_time)
    return eddies


def main_eddies_diags(parser, args):
    logger = logging.getLogger(__name__)

    # Open files
    ds_eddies = xr.open_dataset(args.nc_data_files[1])
    ds_3d = xr.open_mfdataset(args.nc_data_files[0])

    # Compute the sound celerity
    if not hasattr(ds_3d, "cs") and args.acoustic:
        ct = gsw.conversions.CT_from_pt(
            scf.get_salt(ds_3d), scf.get_temp(ds_3d)
        )
        pres = gsw.conversions.p_from_z(
            scf.get_depth(ds_3d), scf.get_lat(ds_3d)
        )
        ds_3d["cs"] = gsw.density.sound_speed(ds_3d.salt, ct, pres)

    # time range
    time = ds_3d.time  # scf.get_time(ds_3d)
    try:
        # select in data file
        if args.date:
            date = args.date
            ds_3d = ds_3d.sel({time.name: date})
        else:
            logger.info("select last date")
            ds_3d = ds_3d.isel({time.name: -1})
            # select in track file
        ds_eddies = _sel_eddies(ds_eddies, date)
    except ValueError:
        ...

    # eddies reconstruction
    eddies_r = seddies.Eddies2D.reconstruct(ds_eddies)

    if args.density:
        print("TO BE IMPLEMENTED")

    if args.acoustic:
        # anomaly construction
        shydrology.compute_anomalies(eddies_r, ds_3d.cs, r_factor=args.rfactor)

        # acoustic points
        sacoustic.acoustic_points(eddies_r)

        # plot
        fig, ax = splot.create_map(
            ds_3d.lon_rho, ds_3d.lat_rho, figsize=(8, 5)
        )
        ds_3d.zeta.plot(
            x="lon_rho",
            y="lat_rho",
            cmap="cmo.dense",
            ax=ax,
            add_colorbar=False,
            transform=splot.pcarr,
        )

        colors = ["k", "#009E73", "#E69F00"]
        labels = ["peu impactant", "impactant", "trés impactant"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([0, 1, 2, 3], cmap.N)

        for eddy in eddies_r.eddies:
            # if eddy.acoustic_impact < 1 :
            #     continue

            eddy.plot_previ(transform=splot.pcarr, lw=1)
            cmb = ax.scatter(
                eddy.ellipse.lon,
                eddy.ellipse.lat,
                s=50,
                marker="o",
                c=eddy.acoustic_impact,
                cmap=cmap,
                norm=norm,
                transform=splot.pcarr,
            )

        cbar = plt.colorbar(cmb, ticks=[0.5, 1.5, 2.5])
        cbar.ax.set_yticklabels(labels)
        for label in cbar.ax.get_yticklabels():
            label.set_rotation(-90)
            label.set_va("center")
        plt.title("SSH")
        plt.tight_layout()

        plt.savefig(args.to_figure)
        logger.info(f"Detections plot saved to: {args.to_figure}")
