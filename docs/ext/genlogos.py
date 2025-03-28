#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To generate the shoot logos
"""
import os
import logging

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

shomlightblue = (90, 194, 231)
shomdarkblue = (0, 36, 84)


def genlogo(outfile, dark=False):
    """Generate a shoot logo and save it"""
    font = "Cantarell"
    # font = "Noto sans"
    width, height = (5.2, 2.7)

    if dark:
        fontcolor = "w"
    else:
        fontcolor = tuple(c / 255 for c in shomdarkblue)
    circlecolor = tuple(c / 255 for c in shomlightblue)

    with plt.rc_context({"font.sans-serif": [font]}):
        fig = plt.figure(figsize=(width, height))
        ax = plt.axes([0, 0, 1, 1], aspect=1, facecolor="b")
        kw = dict(
            family="sans-serif",
            size=100,
            # color=fontcolor,
            va="center_baseline",
            weight="extra bold",
            transform=ax.transAxes,
        )
        ax.text(0.05, 0.515, "SH", ha="left", color=fontcolor, **kw)
        # ax.text(0.6, 0.515, "O", ha="center", color=circlecolor, **kw)
        ax.text(0.95, 0.515, "T", ha="right", color=fontcolor, **kw)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        circle = mpatches.Circle(
            (0.59 * width, height / 2),
            radius=height * 0.17,
            facecolor="none",
            linewidth=14,
            ec=circlecolor,
        )
        ax.add_patch(circle)
        ax.plot(
            [0.59 * width] * 2,
            [(0.5 - 0.16) * height, (0.5 + 0.16) * height],
            color=circlecolor,
            lw=5,
        )
        ax.plot(
            [(0.59 - 0.09) * width, (0.59 + 0.09) * width],
            [0.5 * height] * 2,
            color=circlecolor,
            lw=5,
        )

        clip_height = 0.26
        for y0 in (0, 1 - clip_height):
            circle = mpatches.Circle(
                (0.59 * width, height / 2),
                radius=height * 0.5 * 0.81,
                facecolor="none",
                linewidth=14,
                ec=circlecolor,
            )

            ax.add_patch(circle)

            clip = mtransforms.TransformedBbox(
                mtransforms.Bbox([[0, y0], [1, y0 + clip_height]]), ax.transAxes
            )
            circle.set_clip_box(clip)

        ax.axis("off")
        fig.savefig(outfile, transparent=True)
        plt.close(fig)
        del fig


def genlogos(app):
    """Generate light and dark shoot logo during doc compilation"""
    srcdir = app.env.srcdir
    gendir = os.path.join(srcdir, "_static")
    if not os.path.exists(gendir):
        os.mkdir(gendir)

    logging.debug("Generating light shoot logo...")
    genlogo(os.path.join(gendir, "shoot-logo-light.png"))
    logging.info("Generated light shoot logo")

    logging.debug("Generating dark shoot logo...")
    genlogo(os.path.join(gendir, "shoot-logo-dark.png"), dark=True)
    logging.info("Generated dark shoot logo")


def setup(app):
    app.connect("builder-inited", genlogos)
    return {"version": "0.1"}


if __name__ == "__main__":
    genlogo("../_static/shoot-logo-light.png")
    genlogo("../_static/shoot-logo-dark.png", dark=True)
