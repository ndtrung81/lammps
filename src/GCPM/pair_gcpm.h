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

// Gaussian charge polarizable model (Paricaud et al., J. Chem. Phys. 122,
// 244511 (2005)), consistent with the original Fortran GCPM code (MD_water/).
//
// This base class is the reaction-field form: the Gaussian-smeared charge-charge
// and charge-dipole interactions are summed in real space to cut_coul, with NO
// Ewald/PPPM, and the long-range tail is supplied by a per-pair Onsager/Tironi
// reaction field (enabled when eps_rf > 0). The exp-6 Buckingham dispersion and
// the self-consistent induced-dipole solver are the shared GCPM machinery.
//
// The long-range (Ewald/PPPM) form is the derived class PairGCPMLong, which
// overrides only the Coulomb method (charge_charge/compute_induced_efield/polar)
// and the compute/init_style flow; everything else is inherited from here.

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
  virtual void charge_charge(int, int);
  virtual void polar(int, int, int neigh_half=1);

  virtual void compute_induced_efield(int neigh_half=1);

  // per-molecule reaction-field machinery (Eqs. 11-12 of Paricaud et al.), used
  // by the long-range (Ewald) derived class PairGCPMLong. The reaction-field
  // base class (this class) instead folds the reaction field into the pairwise
  // Coulomb loops and does not call reaction_field_pre/post().
  void setup_reaction_field();          // size tables, c_rf; call from init_style()
  void reaction_field_pre();            // R_q -> efield; call before polar()
  void reaction_field_post(int eflag);  // U_qq^RF energy + RF site forces; after polar()
  void compute_molecular_dipoles();
  void reaction_field(double **mol_d, double **mol_R);
  void grow_mol_arrays(int n);

  // Tally the virial of a force (fx,fy,fz) on atom i from neighbor j, used
  // instead of virial_fdotr_compute() (which is wrong for the non-central
  // polar force under newton on, and is bypassed by the GPU async path).
  // Half list (neigh_half==1): standard pairwise tally. Full list
  // (neigh_half==0): half the pairwise virial, since each pair is visited twice.
  void vtally_force(int i, int j, int neigh_half,
                    double fx, double fy, double fz,
                    double delx, double dely, double delz);

  double cut_lj_global;
  double **cut_lj, **cut_ljsq;
  double cut_coul, cut_coulsq;

  // Buckingham exp-6 parameters: phi = A*exp(-r/rho) - C6/r^6
  // where A = 6*eps*exp(gamma)/(gamma-6), rho = sigma/gamma, C6 = gamma*eps*sigma^6/(gamma-6)
  double **epsilon, **sigma, **gamma_buck;
  double **buck1, **buck2, **buck3, **offset;

  double **alpha_pol;   // molecular polarizability for each type pair
  double **sigmaM;      // Gaussian charge width of M site (individual, per type pair)
  double **alpha_ij;    // per-pair Gaussian parameter: 1/sqrt(2*(si^2+sj^2)) [1/A]
  double g_ewald;       // Ewald splitting parameter (0 for the reaction-field base)
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
