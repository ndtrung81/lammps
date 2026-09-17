.. index:: pair_style gcpm
.. index:: pair_style gcpm/gpu
.. index:: pair_style gcpm/long
.. index:: pair_style gcpm/long/gpu

pair_style gcpm command
=======================

Accelerator Variants: *gcpm/gpu*

pair_style gcpm/long command
============================

Accelerator Variants: *gcpm/long/gpu*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style style args

* style = *gcpm* or *gcpm/long*
* args = list of arguments for a particular style

.. parsed-literal::

     *gcpm* args = enable_polar eps_rf cutoff (cutoff2) (keyword value ...)
       enable_polar = 1 to solve for induced dipoles (polarizable), 0 for charges only
       eps_rf   = dielectric constant of the reaction-field continuum (<= 0 disables the reaction field)
       cutoff   = global cutoff for Buckingham (and Coulombic if only 1 arg) (distance units)
       cutoff2  = global cutoff for Coulombic (optional) (distance units)
       zero or more keyword/value pairs may be appended
       keyword = *cutoff/style* or *polar/tol* or *polar/maxiter*
         *cutoff/style* value = *atom* or *com*
           *atom* = truncate each pair at its atom-atom distance
           *com* = truncate each pair at the center-of-mass distance of the two molecules
         *polar/tol* value = tolerance on the largest induced-dipole change of an iteration (dipole units)
         *polar/maxiter* value = maximum number of induced-dipole iterations per force evaluation
     *gcpm/long* args = enable_polar eps_rf cutoff (cutoff2) (keyword value ...)
       enable_polar = 1 to solve for induced dipoles (polarizable), 0 for charges only
       eps_rf   = must be <= 0; the reaction field is replaced by the k-space solver
       cutoff   = global cutoff for Buckingham (and Coulombic if only 1 arg) (distance units)
       cutoff2  = global cutoff for Coulombic (optional) (distance units)
       keyword = *polar/tol* or *polar/maxiter*, as for *gcpm*

Examples
""""""""

.. code-block:: LAMMPS

   pair_style gcpm   1   78.0  12.0
   pair_coeff 1  1   0.1550  3.1536  12.75  0.0    0.000
   pair_coeff 3  3   0.0     1.0     12.75  1.444  0.610

   pair_style gcpm   1   78.4  11.22  11.22  cutoff/style com  polar/tol 1.8e-5
   pair_style gcpm   0   78.4  12.0

   pair_style gcpm/long   1   0.0   12.0
   kspace_style pppm/dipole 0.0001

Description
"""""""""""

The *gcpm* style implements the Gaussian charge polarizable model (GCPM)
described by :ref:`(Paricaud) <Paricaud>`.  The *gcpm/long* style replaces
the reaction field (RF) approximation with the PPPM for the charge-charge interaction.

The intermolecular (non-bonded
of the GCPM contains three terms:

.. math::

   U_{gcpm} = U_{dispersion}  + U_{Coulomb}  + U_{polar}

The dispersion term are similar to
the :doc:`pair buck6d/coul/gauss/long <pair_buck6d_coul_gauss>`
style in the MOF-FF force field :ref:`(Schmid) <Schmid2>`.

The dispersion term computes a dispersion damped Buckingham potential:

.. math::

   E = A e^{-\kappa r} - \frac{C}{r^6} \cdot \frac{1}{1 + D r^{14}} \qquad r < r_c \\

where A and C are a force constant, :math:`\kappa` is an ionic-pair dependent
reciprocal length parameter, D is a dispersion correction parameter,
and the cutoff :math:`r_c` truncates the interaction distance.
The first term in the potential corresponds to the Buckingham
repulsion term and the second term to the dispersion attraction with
a damping correction analog to the Grimme correction used in DFT.
The latter corrects for artifacts occurring at short distances which
become an issue for soft vdW potentials.

The *gcpm* style uses the reaction field approximation
for the long range contribution as described in :ref:`(Paricaud) <Paricaud>`.
The reaction field is a pairwise term and does not depend on the induced
dipoles, so it may be used either with the polarizable model
(*enable_polar* = 1), where it corrects the charge-charge, charge-dipole and
dipole-dipole interactions alike, or without it (*enable_polar* = 0), where it
is an Onsager reaction field on the Gaussian-smeared Coulomb interaction alone.

.. versionchanged:: TBD

   *eps_rf* > 0 may now be combined with *enable_polar* = 0.  Earlier versions
   of the *gcpm* style required *enable_polar* = 1 whenever a reaction field
   was requested.

The *gcpm/long* style instead uses the real-space term as in
the :doc:`pair buck6d/coul/gauss/long <pair_buck6d_coul_gauss>`
style and requires a :doc:`kspace_style pppm/dipole <kspace_style>`, which
supplies the reciprocal-space charge-charge, charge-dipole and dipole-dipole
contributions.  Because the k-space solver takes over that role, *gcpm/long*
does not accept a reaction field: give it *eps_rf* <= 0.

This pair style include a smoothing function which is invoked
according to the global smoothing parameter within the specified
cutoff.  Hereby a parameter of i.e. 0.9 invokes the smoothing
within 90% of the cutoff.  No smoothing is applied at a value
of 1.0. For the *gcpm* style this smoothing is only applicable
for the dispersion damped Buckingham potential. For the *gcpm/long*
style the smoothing function can also be invoked for the real
space Coulomb interactions, which enforces continuous energies and
forces at the cutoff.

The *gcpm* styles evaluates a Coulomb potential using spherical
Gaussian type charge distributions which effectively dampen
electrostatic interactions for high charges at close distances.
The real-space electrostatic energy is thus evaluated as:

.. math::

   E = \frac{C_{q_i q_j}}{\epsilon r_{ij}}\,\, \textrm{erf}\left(\alpha_{ij} r_{ij}\right)\quad\quad\quad r < r_c

where C is an energy-conversion constant, :math:`q_i` and :math:`q_j`
are the charges on the two atoms, epsilon is the dielectric constant which
can be set by the :doc:`dielectric <dielectric>` command, :math:`\alpha`
is the ion pair dependent damping parameter and erf() is the
error-function.  The cutoff :math:`r_c` truncates the interaction distance.

If one cutoff is specified it is used for both the vdW and Coulomb
terms.  If two cutoffs are specified, the first is used as the cutoff
for the vdW terms, and the second is the cutoff for the Coulombic term.

.. versionadded:: TBD

The *cutoff/style* keyword selects the distance that the cutoffs above are
applied to.  With the default value *atom*, every pair is truncated at the
distance between the two atoms, as in all other LAMMPS pair styles.  With the
value *com*, a pair is instead truncated at the distance between the centers of
mass of the two molecules the atoms belong to, so that the dispersion,
charge-charge, charge-dipole and dipole-dipole interactions between a pair of
molecules are all included or all dropped together.  The interactions
themselves are unchanged: only the cutoff test uses the center-of-mass
distance, while the kernels keep using the atom-atom distance.

The *com* setting reproduces the convention of the original Fortran
implementation of the GCPM model, and is intended for comparing with results
obtained from it.  Note two consequences:

* The centers of mass are computed from the per-atom masses of each molecule
  every time step, and the neighbor list is built with the cutoff enlarged by
  twice the largest atom to center-of-mass distance in the system, so that no
  pair of molecules inside the cutoff is missing from it.  Both add some cost,
  and the enlarged neighbor list needs more memory.
* Because a molecule pair is dropped as a whole, the interaction that is
  discarded at the cutoff is the residual interaction between two neutral
  molecules (predominantly dipole-dipole), which does not vanish there.  The
  energy therefore has a small discontinuity at each cutoff crossing, roughly
  an order of magnitude larger than with the *atom* setting, and the total
  energy of a constant-energy run wanders accordingly.  The trajectory is
  unaffected, but for production runs the *atom* setting conserves energy far
  better.

Using *cutoff/style com* requires molecule IDs, which are defined by the
:doc:`atom_style <atom_style>` used.  It is supported by the *gcpm* and
*gcpm/gpu* styles.  The *gcpm/long* and *gcpm/long/gpu* styles, whose Ewald
sum has to be truncated consistently in real and reciprocal space, accept only
the *atom* setting.

The *polar/tol* and *polar/maxiter* keywords control the iterative solver for
the induced dipoles that runs when *enable_polar* = 1.  The iteration stops
once no induced dipole changes by more than *polar/tol* between two successive
iterations, or after *polar/maxiter* iterations.  A summary of how many
iterations the solver needed is printed at the end of each run.

The polar term is given by

.. math::

   U_{polar} =  \frac{1}{2}\vec{p}_i^{ind} \vec{E}_i^{perm}

where the molecular induced dipoles are iteratively solved for until convergence:

.. math::

   \vec{p}_i^{ind} & = \alpha_i (\vec{E}_i^{ind} +  \vec{E}_i^{perm})
   \vec{E}_i^{ind} = \Sigma_{ij} \vec{T}_{ij} \vec{p}_j^{ind}

The 3-by-3 matrix :math:`\vec{T}` is computed for each pair using
Equations (6) and (7) in :ref:`(Paricaud) <Paricaud>`.

The following coefficients must be defined for each pair of atoms
types via the :doc:`pair_coeff <pair_coeff>` command as in the examples
above, or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands:

* A (energy units)
* :math:`\rho` (distance\^-1 units)
* C (energy-distance\^6 units)
* D (distance\^14 units)
* :math:`\alpha` (distance\^-1 units)
* cutoff (distance units)

The second coefficient, :math:`\rho`, must be greater than zero. The
latter coefficient is optional.  If not specified, the global vdW cutoff
is used.

----------

.. include:: accel_styles.rst

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

These pair styles do not support mixing.  Thus, coefficients for all
I,J pairs must be specified explicitly.

These styles do not support the :doc:`pair_modify <pair_modify>` shift
option for the energy. Instead the smoothing function should be applied
by setting the global smoothing parameter to a value < 1.0.

These styles write their information to :doc:`binary restart files <restart>`, so pair_style and pair_coeff commands do not need
to be specified in an input script that reads a restart file.

Restrictions
""""""""""""

These styles are part of the GCPM package.  They are only
enabled if LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

The *cutoff/style com* setting of the *gcpm* and *gcpm/gpu* styles requires
an :doc:`atom_style <atom_style>` that stores molecule IDs.

The *gcpm/long* style requires :doc:`kspace_style pppm/dipole <kspace_style>`,
and must not be combined with :doc:`neigh_modify exclude <neigh_modify>`:
excluded pairs cannot cancel the k-space contributions.  Intramolecular
Coulomb interactions are already excluded by the pair style using molecule IDs.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

The option defaults are cutoff/style = *atom*, polar/tol = 1.0e-5, and
polar/maxiter = 50.

.. _Paricaud:

**(Paricaud)** P. Paricaud, M. Predota, A. A. Chialvo, P. T. Cummings, J Chem Phys, 122, 244511 (2005).

.. _Schmid2:

**(Schmid)** S. Bureekaew, S. Amirjalayer, M. Tafipolsky, C. Spickermann, T.K. Roy and R. Schmid, Phys. Status Solidi B, 6, 1128 (2013).

