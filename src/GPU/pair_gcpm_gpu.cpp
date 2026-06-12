// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Trung Nguyen (ndactrung@gmail.com)

   GPU version of pair_gcpm following the pair_amoeba/gpu pattern:
   - dispersion + Gaussian-Coulomb forces computed on GPU (k_gcpm kernel)
   - per-atom efield[] computed on GPU (k_gcpm_efield kernel) and read
     back to CPU before the iterative polar solver
   - polar iterative solver runs on CPU using full neighbor list
   - GPU_FORCE: host builds full neighbor list (REQ_FULL)
   - GPU_NEIGH: GPU builds neighbor list;
                host requests REQ_FULL|REQ_NEWTON_OFF for the polar solver for now
     for the CPU-side polar solver
------------------------------------------------------------------------- */

#include "pair_gcpm_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "gpu_extra.h"
#include "info.h"
#include "kspace.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "suffix.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace EwaldConst;
using namespace MathConst;

// same reverse-comm modes as pair_gcpm.cpp
enum {EFIELD, EFIELD_POL};

// External functions from GPU library (lal_gcpm_ext.cpp)

int gcpm_gpu_init(const int ntypes, double **cutsq,
                  double **host_buck1, double **host_buck2, double **host_buck3,
                  double **host_cut_ljsq, double **offset, double **host_alpha_ij,
                  double *special_lj, const int inum, const int nall,
                  const int max_nbors, const int maxspecial,
                  const double cell_size, int &gpu_mode, FILE *screen,
                  double host_cut_coulsq, double *host_special_coul,
                  const double qqrd2e, const double g_ewald,
                  const double rsmooth_sq,
                  const double c0, const double c1, const double c2,
                  const double c3, const double c4, const double c5);
void gcpm_gpu_clear();
int **gcpm_gpu_compute_n(const int ago, const int inum_full, const int nall,
                         double **host_x, int *host_type, double *sublo,
                         double *subhi, tagint *tag, int **nspecial,
                         tagint **special, const bool eflag, const bool vflag,
                         const bool eatom, const bool vatom, int &host_start,
                         int **ilist, int **jnum, const double cpu_time,
                         bool &success, double *host_q, double *boxlo,
                         double *prd, int *periodicity);
void gcpm_gpu_compute(const int ago, const int inum_full, const int nall,
                      double **host_x, int *host_type, int *ilist, int *numj,
                      int **firstneigh, const bool eflag, const bool vflag,
                      const bool eatom, const bool vatom, int &host_start,
                      const double cpu_time, bool &success, double *host_q,
                      const int nlocal, double *boxlo, double *prd);
void gcpm_gpu_compute_efield(void **efield_ptr);
double gcpm_gpu_bytes();

/* ---------------------------------------------------------------------- */

PairGCPMGPU::PairGCPMGPU(LAMMPS *lmp) : PairGCPM(lmp), gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  reinitflag = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  efield_pinned = nullptr;
  acc_float = false;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ---------------------------------------------------------------------- */

PairGCPMGPU::~PairGCPMGPU()
{
  gcpm_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairGCPMGPU::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Resize per-atom arrays if needed
  if (enable_polar && atom->nmax > nmax) {
    memory->destroy(efield);
    memory->destroy(efield_pol);
    memory->destroy(mu_old);
    nmax = atom->nmax;
    memory->create(efield, nmax, 3, "pair:efield");
    memory->create(efield_pol, nmax, 3, "pair:efield_pol");
    memory->create(mu_old, nmax, 4, "pair:mu_old");
  }

  int nall = atom->nlocal + atom->nghost;
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int host_start;
  bool success = true;

  // GPU: dispersion + Gaussian-Coulomb forces via k_gcpm kernel
  if (gpu_mode == GPU_FORCE) {
    gcpm_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
                     ilist, numneigh, firstneigh,
                     eflag, vflag, eflag_atom, vflag_atom,
                     host_start, cpu_time, success, atom->q,
                     atom->nlocal, domain->boxlo, domain->prd);
  } else {
    // GPU_NEIGH: GPU builds its own neighbor list
    int **tmp = gcpm_gpu_compute_n(neighbor->ago, atom->nlocal, nall,
                                   atom->x, atom->type,
                                   domain->sublo, domain->subhi, atom->tag,
                                   atom->nspecial, atom->special,
                                   eflag, vflag, eflag_atom, vflag_atom,
                                   host_start, &ilist, &numneigh, cpu_time,
                                   success, atom->q, domain->boxlo, domain->prd,
                                   domain->periodicity);
    firstneigh = tmp;
  }
  if (!success) error->one(FLERR, "Insufficient memory on accelerator");

  // GPU: per-atom efield from charge-charge interactions via k_gcpm_efield
  if (enable_polar) {
    gcpm_gpu_compute_efield(&efield_pinned);

    // Read efield back from GPU pinned buffer to CPU efield[] array.
    // efield_pinned layout: [Ex0, Ey0, Ez0, Ex1, Ey1, Ez1, ...]
    int nlocal = atom->nlocal;
    if (acc_float) {
      auto *efld = (float *)efield_pinned;
      for (int i = 0; i < nlocal; i++) {
        efield[i][0] = (double)efld[i*3+0];
        efield[i][1] = (double)efld[i*3+1];
        efield[i][2] = (double)efld[i*3+2];
      }
    } else {
      auto *efld = (double *)efield_pinned;
      for (int i = 0; i < nlocal; i++) {
        efield[i][0] = efld[i*3+0];
        efield[i][1] = efld[i*3+1];
        efield[i][2] = efld[i*3+2];
      }
    }

    // Zero efield_pol for ghost atoms before the iterative solver.
    // (local atoms are zeroed each iteration inside compute_induced_efield_full)
    int ntotal = atom->nlocal + atom->nghost;
    for (int i = atom->nlocal; i < ntotal; i++)
      efield_pol[i][0] = efield_pol[i][1] = efield_pol[i][2] = 0.0;

    // Polar iterative solver: uses full neigh list (REQ_FULL, newton off).
    // Key difference from base polar(): no Newton partner in induced efield,
    // and no reverse_comm (full neigh list gives complete efield for local atoms).
    polar_full(eflag, vflag);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   Iterative dipole solver for full neighbor list (no Newton partner).
   Closely mirrors PairGCPM::polar() but omits reverse_comm and the
   Newton-partner force/torque updates inside the force loop.
------------------------------------------------------------------------- */

void PairGCPMGPU::polar_full(int eflag, int vflag)
{
  int i, ii, j, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, ecoul;
  double rsq, rinv, r2inv, factor_coul;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // On the very first call: estimate mu from efield (E_p = 0)
  if (first_polar) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      if (mu[i][3] != 0.0) {
        mu[i][0] = alpha_pol[itype][itype] * efield[i][0] / qqrd2e;
        mu[i][1] = alpha_pol[itype][itype] * efield[i][1] / qqrd2e;
        mu[i][2] = alpha_pol[itype][itype] * efield[i][2] / qqrd2e;
      }
    }
    first_polar = 0;
  }

  // Seed mu_old from starting guess
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mu[i][3] != 0.0) {
      mu_old[i][0] = mu[i][0];
      mu_old[i][1] = mu[i][1];
      mu_old[i][2] = mu[i][2];
    }
  }

  for (int iter = 0; iter < maxiter; iter++) {

    // Zero efield_pol for local atoms before accumulation
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      efield_pol[i][0] = efield_pol[i][1] = efield_pol[i][2] = 0.0;
    }

    // Compute induced efield from T_ij tensor (full neigh, no Newton partner).
    // T_ij is the dipole-dipole interaction tensor; mu[j] includes ghost atoms
    // which were updated by the previous forward_comm.
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (mu[i][3] == 0.0) continue;

      xtmp = x[i][0]; ytmp = x[i][1]; ztmp = x[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj] & NEIGHMASK;
        if (mu[j][3] == 0.0) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = type[j];

        if (rsq < cutsq[itype][jtype]) {
          r2inv = 1.0/rsq;
          double r = sqrt(rsq);
          double r3inv = 1.0/rsq/r;

          double sij = sigmaM[itype][jtype];
          double sij2 = sij * sij;
          double _erf = erf(r / (2.0 * sij));
          double expmsq = exp(-r*r / (4.0*sij2));
          double rds = r / MY_PIS / sij;
          double f_t = _erf - (rds + rds*rsq/(6.0*sij2)) * expmsq;
          double g_t = _erf - rds * expmsq;

          f_t *= 3.0 * r2inv;
          // T_ij applied to mu[j]: efield_pol[i] += qqrd2e * T_ij . mu_j
          double pidotr_j = delx*mu[j][0] + dely*mu[j][1] + delz*mu[j][2];
          double pre = qqrd2e * r3inv;
          efield_pol[i][0] += pre * (f_t*delx*pidotr_j - g_t*mu[j][0]);
          efield_pol[i][1] += pre * (f_t*dely*pidotr_j - g_t*mu[j][1]);
          efield_pol[i][2] += pre * (f_t*delz*pidotr_j - g_t*mu[j][2]);
          // No Newton partner update: full neigh list handles j's contribution
          // from its own loop iteration.
        }
      }
    }

    // Update mu = alpha * (E_q + E_p)
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      if (mu[i][3] != 0.0) {
        mu[i][0] = alpha_pol[itype][itype] * (efield[i][0] + efield_pol[i][0]) / qqrd2e;
        mu[i][1] = alpha_pol[itype][itype] * (efield[i][1] + efield_pol[i][1]) / qqrd2e;
        mu[i][2] = alpha_pol[itype][itype] * (efield[i][2] + efield_pol[i][2]) / qqrd2e;
      }
    }

    // Propagate updated mu to ghost atoms for next iteration
    comm->forward_comm(this);

    // Convergence check across all ranks
    int converged = 1, all_converged = 0;
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (mu[i][3] != 0.0) {
        if (fabs(mu_old[i][0]-mu[i][0]) > tol ||
            fabs(mu_old[i][1]-mu[i][1]) > tol ||
            fabs(mu_old[i][2]-mu[i][2]) > tol) { converged = 0; break; }
      }
    }
    MPI_Allreduce(&converged, &all_converged, 1, MPI_INT, MPI_MIN, world);
    if (all_converged) break;

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (mu[i][3] != 0.0) {
        mu_old[i][0] = mu[i][0];
        mu_old[i][1] = mu[i][1];
        mu_old[i][2] = mu[i][2];
      }
    }
  }

  // After convergence: polar energy and forces.
  // Energy: U_pol = -1/2 * sum_i p_i . E_q_i  (Eq. 9)

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mu[i][3] == 0.0) continue;
    if (eflag) {
      ecoul = -0.5*(mu[i][0]*efield[i][0] + mu[i][1]*efield[i][1] + mu[i][2]*efield[i][2]);
      if (evflag) ev_tally_full(i, 0.0, ecoul, 0.0, 0.0, 0.0, 0.0);
    }

    xtmp = x[i][0]; ytmp = x[i][1]; ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      if (q[j] == 0.0) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype] && rsq < cut_coulsq) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);
        double r = 1.0/rinv;
        double r3inv = r2inv*rinv;
        double r5inv = r3inv*r2inv;

        double aij = alpha_ij[itype][jtype];
        double grij = g_ewald * r;
        double expm2 = MathSpecial::expmsq(grij);
        double erf_g = 1.0 - MathSpecial::my_erfcx(grij) * expm2;
        double aijr = aij * r;
        double expa = MathSpecial::expmsq(aijr);
        double erfa = 1.0 - MathSpecial::my_erfcx(aijr) * expa;
        double falpha = erfa - EWALD_F*aijr*expa;
        double Phi = falpha - erf_g + EWALD_F*grij*expm2;
        double dPhi_dr = 2.0*EWALD_F*rsq*(aij*aij*aij*expa - g_ewald*g_ewald*g_ewald*expm2);

        double pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
        double fqj = factor_coul * qqrd2e * q[j];
        double pre1 = fqj*r5inv * (3.0*Phi - dPhi_dr) * pidotr;
        double pre2 = fqj*r3inv * Phi;

        double fcx = pre2*mu[i][0] - pre1*delx;
        double fcy = pre2*mu[i][1] - pre1*dely;
        double fcz = pre2*mu[i][2] - pre1*delz;

        f[i][0] += fcx;
        f[i][1] += fcy;
        f[i][2] += fcz;

        // No Newton partner: full neigh list handles j's contribution from j's loop.

        torque[i][0] += pre2 * (mu[i][1]*delz - mu[i][2]*dely);
        torque[i][1] += pre2 * (mu[i][2]*delx - mu[i][0]*delz);
        torque[i][2] += pre2 * (mu[i][0]*dely - mu[i][1]*delx);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairGCPMGPU::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR, "Pair style gcpm/gpu requires atom attribute q");

  if (enable_polar && (!atom->mu_flag || !atom->torque_flag))
    error->all(FLERR,
               "Pair gcpm/gpu requires atom attributes mu and torque for polar");

  // Replicate parameter setup from PairGCPM::init_style() without adding
  // its own neighbor request (we add REQ_FULL below).
  double maxcut = -1.0;
  double cut;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        cut = init_one(i, j);
        cut *= cut;
        if (cut > maxcut) maxcut = cut;
        cutsq[i][j] = cutsq[j][i] = cut;
      } else
        cutsq[i][j] = cutsq[j][i] = 0.0;
    }
  }
  double cell_size = sqrt(maxcut) + neighbor->skin;

  if (force->kspace == nullptr)
    error->all(FLERR, "Pair style gcpm/gpu requires a KSpace style");
  g_ewald = force->kspace->g_ewald;
  cut_coulsq = cut_coul * cut_coul;

  // Smoothing coefficients
  c0_c = c1_c = c2_c = c3_c = c4_c = c5_c = 0.0;
  rsmooth_sq_c = cut_coulsq;
  if (coul_smooth < 1.0) {
    double rsm = coul_smooth * cut_coul;
    double rsm_sq = rsm * rsm;
    double denom = pow((cut_coul - rsm), 5.0);
    c0_c = cut_coul*cut_coulsq*(cut_coulsq - 5.0*cut_coul*rsm + 10.0*rsm_sq) / denom;
    c1_c = -30.0*(cut_coulsq*rsm_sq) / denom;
    c2_c =  30.0*(cut_coulsq*rsm + cut_coul*rsm_sq) / denom;
    c3_c = -10.0*(cut_coulsq + 4.0*cut_coul*rsm + rsm_sq) / denom;
    c4_c =  15.0*(cut_coul + rsm) / denom;
    c5_c =  -6.0 / denom;
    rsmooth_sq_c = rsm_sq;
  }

  int maxspecial = 0;
  if (atom->molecular != Atom::ATOMIC) maxspecial = atom->maxspecial;
  int mnf = 5e-2 * neighbor->oneatom;

  int success = gcpm_gpu_init(
      atom->ntypes + 1, cutsq,
      buck1, buck2, buck3, cut_ljsq, offset, alpha_ij,
      force->special_lj,
      atom->nlocal, atom->nlocal + atom->nghost, mnf, maxspecial,
      cell_size, gpu_mode, screen,
      cut_coulsq, force->special_coul, force->qqrd2e, g_ewald,
      rsmooth_sq_c, c0_c, c1_c, c2_c, c3_c, c4_c, c5_c);
  GPU_EXTRA::check_flag(success, error, world);

  if (gpu_mode == GPU_FORCE) {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_NEWTON_OFF);
  }

  acc_float = Info::has_accelerator_feature("GPU", "precision", "single");
}

/* ---------------------------------------------------------------------- */

double PairGCPMGPU::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes + gcpm_gpu_bytes();
}
