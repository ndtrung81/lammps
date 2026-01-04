/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/coul/gauss/long,PairLJCutCoulGaussLong);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_COUL_GAUSS_LONG_H
#define LMP_PAIR_LJ_CUT_COUL_GAUSS_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutCoulGaussLong : public Pair {

 public:
  PairLJCutCoulGaussLong(class LAMMPS *);
  ~PairLJCutCoulGaussLong() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  int pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/) override;
  void unpack_forward_comm(int n, int first, double *buf) override;
  void *extract(const char *, int &) override;

 protected:
  void dispersion(int, int);
  void charge_charge(int, int);
  void polar(int, int);

  void compute_induced_efield(double**);

  double cut_lj_global;
  double **cut_lj, **cut_ljsq;
  double cut_coul, cut_coulsq;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double **alpha_pol;   // polarizability for each type pair
  double **sigmaM;      // charge spreading width for each type pair
  double *cut_respa;
  double qdist;    // TIP4P distance from O site to negative charge
  double g_ewald;

  double coul_smooth, alpha;
  double c0_c, c1_c, c2_c, c3_c, c4_c, c5_c, rsmooth_sq_c;

  double **efield;      // per-atom electric field due to charges
  double **efield_pol;  // per-atom electric field due to induced dipoles
  double **mu_old;      // per-atom induced dipole from previous iteration
  int nmax;
  int maxiter;          // maximum number of induced dipole iterations
  double tol;           // convergence tolerance for induced dipole iterations
  
  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
