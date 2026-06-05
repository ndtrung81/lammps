The GCPM package implements the Gaussian charge polarizable model (GCPM) 
described in Paricaud, Predota, Chiavlo and Cummings, J. Chem. Phys. 122, 244511 (2005)
in the pair style pair buck6/coul/gauss/long

The polarizable molecules are modeled as rigid bodies by using, `fix rigid` or `fix rigid/small`,
and the molecular polarability is represented a single site on the molecule, for instance,
the M site in the explicit TIP4P water model. The induced dipole at the site is then solved
iteratively at every MD timestep.

* Build

    mkdir build-gcpm && cd build-gcpm
    cmake ../cmake -C ../cmake/presets/basic.cmake -DPKG_GCPM=on -DPKG_DIPOLE=on
    make -j4

* Examples

Example input scripts and data files can be found under examples/PACKAGES/gcpm:

mpirun -np 8 /path/to/lmp_mpi -in examples/PACKAGES/gcpm/in.water_box
