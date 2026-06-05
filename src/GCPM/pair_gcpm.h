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
PairStyle(gcpm,PairGCPM);
// clang-format on
#else

#ifndef LMP_PAIR_GCPM_H
#define LMP_PAIR_GCPM_H

#include "pair.h"

namespace LAMMPS_NS {

class PairGCPM : public Pair {

 public:
  PairGCPM(class LAMMPS *);
  ~PairGCPM() override;
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
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void *extract(const char *, int &) override;

 protected:
  void dispersion(int, int);
  void charge_charge(int, int);
  void polar(int, int);

  void compute_induced_efield();

  double cut_lj_global;
  double **cut_lj, **cut_ljsq;
  double cut_coul, cut_coulsq;

  // Buckingham exp-6 parameters: phi = A*exp(-r/rho) - C6/r^6
  // where A = 6*eps*exp(gamma)/(gamma-6), rho = sigma/gamma, C6 = gamma*eps*sigma^6/(gamma-6)
  double **epsilon, **sigma, **gamma_buck;
  double **buck1, **buck2, **buck3, **offset;

  double **alpha_pol;   // molecular polarizability for each type pair
  double **sigmaM;      // Gaussian charge width of M site (individual, per type pair)
  double **alpha_ij;    // per-pair Ewald Gaussian parameter: 1/sqrt(2*(si^2+sj^2)) [1/A]
  double *cut_respa;
  double g_ewald;
  double qdist;
  double coul_smooth, alpha;
  double c0_c, c1_c, c2_c, c3_c, c4_c, c5_c, rsmooth_sq_c;

  double **efield;      // per-atom electric field due to charges
  double **efield_pol;  // per-atom electric field due to induced dipoles
  double **mu_old;      // per-atom induced dipole from previous iteration
  int nmax;
  int maxiter;
  double tol;
  int enable_polar;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
