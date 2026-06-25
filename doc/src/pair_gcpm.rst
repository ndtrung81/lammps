.. index:: pair_style gcpm
.. index:: pair_style gcpm/long

pair_style gcpm command
========================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style style args

* style = *gcpm*
* args = list of arguments for a particular style

.. parsed-literal::

     *gcpm* args = enable_polar eps_rf cutoff (cutoff2)
       enable_polar = 1 to solve for induced dipoles (polarizable), 0 for charges only
       eps_rf   = dielectric constant of the reaction-field continuum (<= 0 disables the reaction field)
       cutoff   = global cutoff for Buckingham (and Coulombic if only 1 arg) (distance units)
       cutoff2  = global cutoff for Coulombic (optional) (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style gcpm   1   78.0  12.0
   pair_coeff 1  1   0.1550  3.1536  12.75  0.0    0.000
   pair_coeff 3  3   0.0     1.0     12.75  1.444  0.610

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
style in the MOF-FF force field :ref:`(Schmid) <Schmid>`. 

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
The *gcpm* style uses the real-space term as in
the :doc:`pair buck6d/coul/gauss/long <pair_buck6d_coul_gauss>`
style and requires a kspace style.

This pair style include a smoothing function which is invoked
according to the global smoothing parameter within the specified
cutoff.  Hereby a parameter of i.e. 0.9 invokes the smoothing
within 90% of the cutoff.  No smoothing is applied at a value
of 1.0. For the *gauss/dsf* style this smoothing is only applicable
for the dispersion damped Buckingham potential. For the *gauss/long*
styles the smoothing function can also be invoked for the real
space coulomb interactions which enforce continuous energies and
forces at the cutoff.

The *gcpm/long* style evaluate a Coulomb potential using spherical Gaussian type charge
distributions which effectively dampen electrostatic interactions
for high charges at close distances.  The electrostatic potential
is thus evaluated as:

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

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

none

.. _Paricaud:

.. _Schmid:

**(Paricaud)** P. Paricaud, M. Predota, A. A. Chialvo, P. T. Cummings, J Chem Phys, 122, 244511 (2005).

**(Schmid)** S. Bureekaew, S. Amirjalayer, M. Tafipolsky, C. Spickermann, T.K. Roy and R. Schmid, Phys. Status Solidi B, 6, 1128 (2013).

