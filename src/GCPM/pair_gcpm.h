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
  void polar(int, int, int neigh_half=1);

  void compute_induced_efield(int neigh_half=1);

  // reaction-field correction (Eqs. 11-12 of Paricaud et al.)
  void compute_molecular_dipoles();
  void reaction_field(double **mol_d, double **mol_R);
  void grow_mol_arrays(int n);

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
  double g_ewald;       // g_ewald = 5.6/sigma for Coulomb interactions with Gaussian charge smearing
  double *cut_respa;

  double **efield;      // per-atom electric field due to charges
  double **efield_pol;  // per-atom electric field due to induced dipoles
  double **mu_old;      // per-atom induced dipole from previous iteration
  int nmax;             // maximum number of atoms that can be stored in efield arrays
  int maxiter;          // max iterations for induced dipole convergence
  double tol;           // tolerance for induced dipole convergence
  int enable_polar;     // 1 if polar interactions enabled, 0 if not
  int comm_mode;        // 0 = reverse-comm efield, 1 = reverse-comm efield_pol
  int first_polar;      // 1 on first call to polar(), 0 thereafter (warm-start flag)

  // reaction-field correction (Onsager continuum, Eqs. 11-12 of Paricaud et al.)
  int enable_rf;        // 1 if reaction-field correction enabled, 0 if not
  double eps_rf;        // dielectric constant of the continuum surrounding the cavity
  double c_rf;          // RF prefactor 2*qqrd2e*(eps_rf-1)/((2*eps_rf+1)*rc^3), field units
  int nmol;             // largest molecule id (number of molecules - 1-indexed)
  int nmol_max;         // allocated size of the per-molecule reaction-field tables
  double **mol_mu;      // per-molecule permanent dipole sum_a q_a r_a [e*A]
  double **mol_p;       // per-molecule induced dipole (on the M site) [e*A]
  double **mol_x;       // per-molecule cavity center (the M site, unwrapped) [A]
  double **mol_Rq;      // per-molecule reaction field from permanent dipoles
  double **mol_Rp;      // per-molecule reaction field from induced dipoles

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
