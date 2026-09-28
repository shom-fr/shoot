.. _algos.track:

Tracking
========

Eddies are tracked by :func:`~shoot.eddies.track.track_eddies` from detections at
successive time steps (:class:`~shoot.eddies.eddies2d.EvolEddies2D`), following
the approach of Chelton et al. (2011) and Le Vu et al. (2018).

Initialization
--------------

Each eddy of the first time step starts a new track.

Association
-----------

At each following time step, the new eddies are associated with the eddies of the
``nback`` previous time steps that do not already have a successor
(:class:`~shoot.eddies.track.AssociateMulti`).

#. For a new eddy :math:`i` and a previous eddy :math:`j` detected :math:`\Delta t`
   earlier, the squared cost of the association is the sum of:

   - a distance term :math:`(d_{ij} / D_{ij})^2`, where :math:`d_{ij}` is the distance
     between the eddy centers and :math:`D_{ij}` the search distance

     .. math::

        D_{ij} = C \frac{1 + \Delta t}{2} + \overline{R}_j + R_i

     with :math:`C = 6.5` km/day a typical propagation speed, :math:`\overline{R}_j`
     the radius of maximal speed of the track of :math:`j` averaged over its last five
     eddies, and :math:`R_i` the radius of maximal speed of :math:`i`;
   - a dynamical similarity term :math:`\delta R^2 + \delta Ro^2`, with

     .. math::

        \delta R = \frac{R_j - R_i}{\overline{R}_j + R_i}, \quad
        \delta Ro = \frac{Ro_j - Ro_i}{\overline{Ro}_j + Ro_i}

     where :math:`R` is the radius, :math:`Ro` the Rossby number, and the overlines
     denote averages over the last five eddies of the track;
   - a temporal term :math:`(\Delta t / 2 T_c)^2`, with :math:`T_c` = ``nback`` times the
     time step, that favours the most recent eddies.

   The association is impossible when :math:`d_{ij} \geq D_{ij}` or when the eddies
   are not of the same type (cyclone or anticyclone).
   The cost matrix is computed by :func:`shoot.core.track.association_cost`.

#. The new eddies that cannot be associated with any previous eddy are discarded
   from the assignment.
#. The other ones are optimally assigned to previous eddies by minimizing the total
   cost with the Hungarian algorithm (:func:`scipy.optimize.linear_sum_assignment`).
#. An assigned new eddy continues the track of its previous eddy, which cannot be
   assigned again.

Unassigned new eddies start new tracks.

Update
------

Tracks saved to a file can be extended with new detections with
:func:`~shoot.eddies.track.update_tracks`.
