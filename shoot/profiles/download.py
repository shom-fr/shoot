#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  19 10:21:12 2025

@author: jbroust
"""

import copernicusmarine as cm
import os
import numpy as np
from .. import cf as scf


#### It requires that the user have been logged in the copernicus
#### marine toolbox before


class Download:
    def __init__(
        self,
        years,
        months,
        root_path,
        region="mediterrane",
        data_types=["XB", "PF"],
    ):
        self.path = os.path.join(root_path, region)
        self.region = region
        self.years = years
        self.months = months
        print("years ", years)
        self.data_types = data_types

    def _load(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for dtype in self.data_types:
            for i, year in enumerate(self.years):
                if i == 0:
                    months = np.arange(self.months[0], 13)
                elif i == len(self.years) - 1:
                    months = np.arange(1, self.months[-1] + 1)
                else:
                    months = np.arange(1, 13)
                path_out = os.path.join(self.path, "%s" % year)
                if not os.path.exists(path_out):
                    os.makedirs(path_out)
                print("path :", path_out)
                print(
                    " \n  ##### Downloading : "
                    + dtype
                    + " - year : "
                    + str(year)
                    + "######"
                )
                for m in months:
                    cm.get(
                        dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_cora_irr",
                        filter="*"
                        + self.region
                        + "/"
                        + str(year)
                        + "/CO_DMQCGL01_"
                        + str(year)
                        + f"{m:02d}"
                        + "*_PR_"
                        + dtype
                        + "*",
                        output_directory=path_out,
                        no_directories=True,
                        force_download=True,
                    )

    @staticmethod
    def get_regions():
        return {
            "artic": {"lat": (66, 90), "lon": (-180, 180)},
            "baltic": {"lat": (53, 66), "lon": (9, 30)},
            "blacksea": {"lat": (40, 47), "lon": (27, 42)},
            "mediterrane": {"lat": (30, 46), "lon": (-6, 36)},
            "northwesternshelf": {"lat": (48, 62), "lon": (-25, 10)},
            "southwestshelf": {"lat": (43, 50), "lon": (-17, 5)},
        }


def load(
    years, months, region, root_path="/local/tmp/data", data_types=["XB", "PF"]
):
    Download(
        years, months, root_path, region=region, data_types=data_types
    )._load()
