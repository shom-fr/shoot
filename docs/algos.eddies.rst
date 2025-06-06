.. _algos.eddies:

Eddies
======

The eddies detection is performed by :func:`~shoot.eddies.detect_eddies` with the following procedure:

#. Find eddy centers with :func:`~shoot.eddies.find_eddy_centers` in a velocity field and given a window.

   #. Compute the local normalized angular momentum (LNAM) with :func:`~shoot.dyn.get_lnam`.
   #. Compute the Okubo-Weiss parameter (OW)with :func:`~shoot.dyn.get_okuboweiss`.
   #. Mask values of LNAM where the OW is negative.
   #. Find local extrema of LNAM with :func:`~shoot.num.find_signed_peaks_2d` which define eddy centers.
   
#. Loop on eddy centers and initialize an :class:`~shoot.eddies.RawEddy2D` instance from the gridded fields and eddy properties:

   #. Estimate local eddy pseudo-SSH from currents with :func:`~shoot.fit.fit_ssh_from_uv` if not already provided. It is generally useful for subsurface detections. It is possible to directly intergate the stream function based on the velocity field (faster)
   #. Compute SSH closed contours with :func:`~shoot.contours.get_closed_contours` and store them in :attr:`~shoot.eddies.RawEddy2D.contours`. The search is made in a defined window 
   #. Keep the largest contour that defines the eddy bundary and store it in :attr:`~shoot.eddies.RawEddy2D.boundary_contour`.
   #. Interpolate the velocity onto contours with :attr:`~shoot.contours.add_contour_uv`, then find the contour with the maximal average speed and store it in :attr:`~shoot.eddies.RawEddy2D.vmax_contour`.
   #. Check the eddy validity following 4 criteria : existence of at least one closed contour including the centert, minimun radius, avoid costal detection, ellipticity of the velocity contour 
   #. If the velocity contour matches the eddy boundary contour we make the search in a largest window

#. Loop on eddy to avoid eddy inclusion. If two eddies share part of maximum velocity contour only keep the one with largest LNAM. 