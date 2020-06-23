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

#ifdef FIX_CLASS

FixStyle(gcmc/electrolyte,FixGCMCElectrolyte)

#else

#ifndef LMP_FIX_GCMC_ELECTROLYTE_H
#define LMP_FIX_GCMC_ELECTROLYTE_H

#include "fix_gcmc.h"

namespace LAMMPS_NS {

class FixGCMCElectrolyte : public FixGCMC {
 public:
  FixGCMCElectrolyte(class LAMMPS *, int, char **);
  virtual ~FixGCMCElectrolyte() {}
  virtual void init();
  virtual void attempt_molecule_deletion_full();
  virtual void attempt_molecule_insertion_full();

 protected:
  static const int nmaxfactorial = 167;
  static const double nfac_table[];
  double factorial(int);
  int num_anions_per_molecule;
  int num_cations_per_molecule;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
