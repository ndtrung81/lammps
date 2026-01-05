The GCPM package implements the the Gaussian charge polarizable model (GCPM) described by
Paricaud, Predota, Chiavlo and Cummings, J. Chem. Phys. 122, 244511 (2005):

  * pair lj/cut/coul/gauss/long (in progress)
  * pair buck6/coul/gauss/long (TODO)

The polarizable molecules are modeled as rigid bodies by using, for instance, `fix rigid/small` and a molecular polarability is represented a single site on the molecule, for instance, the M site in the explicit TIP4P water model. The induced dipole at the site is then solved iteratively at every MD timestep.

* Examples

Example input scripts and data files can be found under examples/PACKAGES/gcpm:

mpirun -np 8 lmp_mpi -in in.water_box
