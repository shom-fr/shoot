#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:25:20 2026

@author: jbroust
"""

import functools
import numpy as np

from .hydrology import Field2D


class DPCEddy:
    """DPC based acoustic analysis for an eddy 

    Computes acoustic parameters both inside and outside an eddy,
    and calculates the acoustic impact (difference between inside/outside).

    Parameters
    ----------
    anomaly : Anomaly
        Eddy anomaly object containing profile data inside and outside the eddy.

    Attributes
    ----------
    ecs_inside : float
        Average Surface duct thickness inside the eddy.
    ecs_outside : float
        Average Surface duct thickness outside the eddy.
    mcp_inside : float
        Average Deep sound speed minimum depth inside the eddy.
    mcp_outside : float
        Average Deep sound speed minimum depth outside the eddy.
    iminc_inside : float
        Average Intermediate minimum depth inside the eddy.
    iminc_outside : float
        Average Intermediate minimum depth outside the eddy.
    acoustic_impact : float
        Combined acoustic impact metric (sum of relative differences).
    """

    def __init__(self, field2d):
        """Initialize acoustic eddy analyzer

        Parameters
        ----------
        field2d : Field2D object
            Dpc output centered around the eddy.
        """
        self.field2d = field2d


    @functools.cached_property
    def ecs_insides(self):
        return self.field2d.ds_inside.z_sld 
    
    @functools.cached_property
    def iminc_insides(self):
        return self.field2d.ds_inside.z_iminc

    @functools.cached_property
    def iminops_insides(self):
        return self.field2d.ds_inside.z_iminops

    @functools.cached_property
    def mcp_insides(self):
        return self.field2d.ds_inside.z_maxdeep

    @functools.cached_property
    def ecs_outsides(self):
        return self.field2d.ds_outside.z_sld
    
    @functools.cached_property
    def iminc_outsides(self):
        return self.field2d.ds_outside.z_iminc

    @functools.cached_property
    def iminops_outsides(self):
        return self.field2d.ds_outside.z_iminops

    @functools.cached_property
    def mcp_outsides(self):
        return self.field2d.ds_outside.z_maxdeep


    @functools.cached_property
    def ecs_inside(self):
        return self.ecs_insides.mean() if np.sum(np.isnan(self.ecs_insides))<0.5*len(self.ecs_insides) else np.nan 
    
    @functools.cached_property
    def iminc_inside(self):
        return self.iminc_insides.mean() if np.sum(np.isnan(self.iminc_insides))<0.5*len(self.iminc_insides) else np.nan 

    @functools.cached_property
    def iminops_inside(self):
        return self.iminops_insides.mean() if np.sum(np.isnan(self.iminops_insides))<0.5*len(self.iminops_insides) else np.nan 

    @functools.cached_property
    def mcp_inside(self):
        return self.mcp_insides.mean() if np.sum(np.isnan(self.mcp_insides))<0.5*len(self.mcp_insides) else np.nan 

    @functools.cached_property
    def ecs_outside(self):
        return self.ecs_outsides.mean() if np.sum(np.isnan(self.ecs_outsides))<0.5*len(self.ecs_outsides) else np.nan 
    
    @functools.cached_property
    def iminc_outside(self):
        return self.iminc_outsides.mean() if np.sum(np.isnan(self.iminc_outsides))<0.5*len(self.iminc_outsides) else np.nan 

    @functools.cached_property
    def iminops_outside(self):
        return self.iminops_outsides.mean() if np.sum(np.isnan(self.iminops_outsides))<0.5*len(self.iminops_outsides) else np.nan 

    @functools.cached_property
    def mcp_outside(self):
        return self.mcp_outsides.mean() if np.sum(np.isnan(self.mcp_outsides))<0.5*len(self.mcp_outsides) else np.nan 

    @staticmethod
    def _distance(e1, e2):
        if np.isnan(e1) and np.isnan(e2):  # point doesn exists at all
            return 0
        elif np.isnan(e1) or np.isnan(e2):  # creation/destruction point
            return 1
        elif e1 == 0 and e2 == 0:
            return 0
        else:
            return np.abs(e1 - e2) / np.abs((0.5 * (e1 + e2)))

    @functools.cached_property
    def acoustic_impact(self):
        ecs_in = self.ecs_inside
        iminops_in = self.iminc_inside
        ecs_out = self.ecs_outside
        iminops_out = self.iminc_outside

        d_iminops = DPCEddy._distance(iminops_in, iminops_out)
        d_ecs = DPCEddy._distance(ecs_in, ecs_out)

        return  d_iminops + d_ecs


def acoustic_points_dpc(eddies, dpc, r_factor = 1.2):
    """Compute acoustic impact for all eddies

    Adds acoustic parameters (ecs, iminc, mcp) inside and outside
    each eddy, plus overall acoustic impact metric.

    Parameters
    ----------
    eddies : Eddies2D
        Collection of eddies with anomaly attributes.

    Notes
    -----
    Modifies eddies in-place by adding acoustic attributes.
    """

    for eddy in eddies.eddies:
        acous = DPCEddy(Field2D(eddy, eddies, dpc, r_factor = r_factor))
        eddy.ecs_insides = acous.ecs_insides
        eddy.ecs_outsides = acous.ecs_outsides
        eddy.iminc_insides = acous.iminc_insides
        eddy.iminc_outsides = acous.iminc_outsides
        eddy.mcp_insides = acous.mcp_insides
        eddy.mcp_outsides = acous.mcp_outsides
        eddy.acoustic_impact = acous.acoustic_impact


