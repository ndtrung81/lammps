The GCPM package implements the Gaussian charge polarizable model (GCPM)
described in Paricaud, Predota, Chiavlo and Cummings, J. Chem. Phys. 122, 244511 (2005)
in the pair style gcpm.

The polarizable molecules are modeled as rigid bodies by using, `fix rigid` or `fix rigid/small`,
and the molecular polarability is represented a single site on the molecule, for instance,
the M site in the explicit TIP4P water model. The induced dipole at the site is then solved
iteratively at every MD timestep.

* Get the source code
```
    git clone https://github.com/ndtrung81/lammps.git lammps-gcpm
    cd lammps-gcpm
    git checkout gcpm
```

* Build

To build the code you need cmake (3.26+), a C/C++ compiler (GNU GCC 12.0 and later),
and a MPI library (OpenMPI or MPICH) installed.

```
    mkdir build && cd build
    cmake ../cmake -C ../cmake/presets/basic.cmake -DPKG_GCPM=on
    make -j4
```

You can also build with GPU suppport for the pair gcpm styles if you have a GPU to run on:

```
    cmake ../cmake -C ../cmake/presets/basic.cmake -DPKG_GCPM=on -DPKG_GPU=on
    make -j4
```

* Test

The example input scripts and data files can be found under examples/PACKAGES/gcpm:

* `in.water_box` and `data.water_box`: input deck for 512 water molecules at density ~1 g/cm^3

```
    mpirun -np 8 lammps-gcpm/build/lmp -in examples/PACKAGES/gcpm/in.water_box -v steps 100000
```

* `in.gcpm` and `data.gcpm`: input deck for 500 water molecules converted from the Fortran code bundle at T = 298 K and density 1 g/cm^3

```
    mpirun -np 8 lammps-gcpm/build/lmp -in examples/PACKAGES/gcpm/in.gcpm  -v steps 1000000
```

To run with GPU acceleration:

```
    mpirun -np 8 lammps-gcpm/build/lmp -in examples/PACKAGES/gcpm/in.gcpm  -v steps 1000000 -sf gpu -pk gpu 1 neigh no
```

Validation 1 (Section IV B, Fig. 5):

The output RDFs of O-O, O-H and H-H from the run with in.gcpm are in `rdf.txt`
is expected to match the reference RDFs of O-O, O-H and H-H from the Fortran run `gofr.dat`.

Plotting column 2 (distance) versus column 3, 5, 7 and 9
shows O-O, O-H, H-O and H-H pairs computed from LAMMPS.

```
   gnuplot> plot 'rdf.txt' u 2:3 w l, 'gofr.dat' u 1:2 w p
   gnuplot> plot 'rdf.txt' u 2:5 w l, 'gofr.dat' u 1:3 w p
   gnuplot> plot 'rdf.txt' u 2:9 w l, 'gofr.dat' u 1:4 w p
```

Validation 2 (Section IV C, Tables V and VI):

Self diffusion coefficents from the LAMMPS code are expected to match the values in these 2 tables.





