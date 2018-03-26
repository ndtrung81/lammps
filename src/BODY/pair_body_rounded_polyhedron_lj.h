/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(body/rounded/polyhedron/lj,PairBodyRoundedPolyhedronLJ)

#else

#ifndef LMP_PAIR_BODY_ROUNDED_POLYHEDRON_LJ_H
#define LMP_PAIR_BODY_ROUNDED_POLYHEDRON_LJ_H

#include "pair_body_rounded_polyhedron.h"

namespace LAMMPS_NS {

class PairBodyRoundedPolyhedronLJ : public PairBodyRoundedPolyhedron {
 public:
  PairBodyRoundedPolyhedronLJ(class LAMMPS *);
  ~PairBodyRoundedPolyhedronLJ();
  void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);

  void kernel_force(double R, int itype, int jtype,
    double cut_inner, double& fpair, double& energy);

 protected:
  double **cut;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair body/rounded/polyhedron requires atom style body rounded/polyhedron

Self-explanatory.

E: Pair body requires body style rounded/polyhedron

This pair style is specific to the rounded/polyhedron body style.

*/
