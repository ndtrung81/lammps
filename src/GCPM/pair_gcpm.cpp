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
                        with Claude Code Sonnet 4.6 and Opus 4.8
   Reference: Paricaud et al., J. Chem. Phys. 122, 244511 (2005)

   Gaussian charge polarizable model (GCPM), consistent with the original
   Fortran code (folder MD_water/). This base class is the reaction-field form:
   the Gaussian-smeared charge-charge and charge-dipole interactions are summed
   in real space to cut_coul, with NO Ewald/PPPM long-range summation, and the
   long-range tail is supplied by a per-pair Onsager/Tironi reaction field (the
   Fortran "ferf" terms), enabled when eps_rf > 0.

   Buckingham exp-6 dispersion: Eq. (10) of the reference
     phi = eps/(1-6/gamma) * [6/gamma * exp(gamma*(1-r/sigma)) - (sigma/r)^6]
   stored as the equivalent standard Buckingham A*exp(-r/rho) - C6/r^6 with:
     A   = 6*eps*exp(gamma)/(gamma-6)
     rho = sigma/gamma  (i.e. buck2 = 1/rho = gamma/sigma)
     C6  = gamma*eps*sigma^6/(gamma-6)

   Reaction-field bookkeeping (matching force.f), with
     c_rf = 2*qqrd2e*(eps_rf-1)/((2*eps_rf+1)*rc^3) = qqrd2e * ferf:
     (A) charge-charge   : force -qi*qj*c_rf*r_vec, energy 0.5*qi*qj*c_rf*r^2
     (B) charge->dipole  : efield_i += -c_rf*qj*r_vec  (folded into efield, so
                           the Eq.(9) polarization energy -1/2 p.E_q carries the
                           charge-dipole RF energy)
     (C) dipole->dipole  : efield_pol_i += c_rf*mu_j  (+ self c_rf*mu_i)
     (D) charge-dipole   : a charge in the uniform reaction field c_rf*mu of a
                           dipole feels force q*c_rf*mu
   The reaction field is a continuum-cavity property and is applied to every
   pair within cut_coul (it does NOT use special_coul scaling). For rigid
   molecules the intramolecular RF forces are internal and are projected out by
   the rigid-body integrator; the intramolecular pairs supply the self terms of
   (B)/(D) automatically, so no separate molecular self term is needed (only the
   dipole self term of (C), which has no intramolecular M-M pair, is added
   explicitly).

   The long-range (Ewald/PPPM) form is the derived class PairGCPMLong, which
   overrides only charge_charge()/compute_induced_efield()/polar() and the
   compute()/init_style() flow; the per-molecule reaction-field helpers below
   (setup_reaction_field, reaction_field_pre/post, reaction_field,
   compute_molecular_dipoles, grow_mol_arrays) are used by that derived class.
------------------------------------------------------------------------- */

#include "pair_gcpm.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace EwaldConst;

// reverse-comm selector for comm_mode (EFIELD = 0 -> efield, EFIELD_POL = 1 -> efield_pol)
enum {EFIELD, EFIELD_POL};

/* ---------------------------------------------------------------------- */

PairGCPM::PairGCPM(LAMMPS *lmp) : Pair(lmp)
{
  // reaction-field Coulomb: no long-range k-space (the derived PairGCPMLong sets
  // these to 1). The reaction field replaces Ewald in this base class.
  ewaldflag = pppmflag = 0;
  respa_enable = 0;
  single_enable = 0;
  writedata = 1;

  // The polar force is not a central-pairwise term, and on the GPU path the
  // dispersion/Coulomb forces are applied asynchronously after compute().
  // virial_fdotr_compute() gives the wrong polar virial under newton on (and
  // is bypassed entirely on the GPU). Tally all virials explicitly instead.
  no_virial_fdotr_compute = 1;

  ftable = nullptr;
  cut_respa = nullptr;

  enable_polar = 1;
  efield = nullptr;
  efield_pol = nullptr;
  mu_old = nullptr;
  nmax = 0;
  maxiter = 50;
  tol = 1.0e-5;

  // set comm size needed by this Pair

  comm_forward = 4;
  comm_reverse = 3;
  comm_mode = EFIELD_POL;
  first_polar = 1;

  // induced-dipole solver convergence statistics (reset each run in setup())

  polar_ncalls = 0;
  polar_niter_sum = 0;
  polar_niter_min = 0;
  polar_niter_max = 0;
  polar_nonconv = 0;

  // reaction-field correction (disabled unless eps_rf > 0 is given)

  enable_rf = 0;
  eps_rf = 0.0;
  c_rf = 0.0;
  nmol = 0;
  nmol_max = 0;
  mol_mu = mol_p = mol_x = mol_Rq = mol_Rp = nullptr;
}

/* ---------------------------------------------------------------------- */

PairGCPM::~PairGCPM()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(gamma_buck);
    memory->destroy(buck1);
    memory->destroy(buck2);
    memory->destroy(buck3);
    memory->destroy(offset);
    memory->destroy(alpha_pol);
    memory->destroy(sigmaM);
    memory->destroy(alpha_ij);
  }
  memory->destroy(efield);
  memory->destroy(efield_pol);
  memory->destroy(mu_old);

  memory->destroy(mol_mu);
  memory->destroy(mol_p);
  memory->destroy(mol_x);
  memory->destroy(mol_Rq);
  memory->destroy(mol_Rp);

  if (ftable) free_tables();
}

/* ----------------------------------------------------------------------
   accumulate per-step induced-dipole solver iteration counts (called once per
   polar() invocation). The iteration count is identical on every rank because
   convergence is decided by a global MPI_Allreduce, so no reduction is needed
   here; finish() reports the rank-0 tallies.
------------------------------------------------------------------------- */

void PairGCPM::record_polar_iters(int niter, int converged)
{
  if (polar_ncalls == 0 || niter < polar_niter_min) polar_niter_min = niter;
  if (niter > polar_niter_max) polar_niter_max = niter;
  polar_niter_sum += niter;
  polar_ncalls++;
  if (!converged) polar_nonconv++;
}

/* ----------------------------------------------------------------------
   reset solver statistics at the start of each run
------------------------------------------------------------------------- */

void PairGCPM::setup()
{
  polar_ncalls = 0;
  polar_niter_sum = 0;
  polar_niter_min = 0;
  polar_niter_max = 0;
  polar_nonconv = 0;
}

/* ----------------------------------------------------------------------
   report induced-dipole solver convergence statistics at the end of the run
------------------------------------------------------------------------- */

void PairGCPM::finish()
{
  if (!enable_polar || polar_ncalls == 0) return;
  if (comm->me != 0) return;

  double avg = (double) polar_niter_sum / (double) polar_ncalls;
  utils::logmesg(lmp, "\nGCPM induced-dipole solver stats:\n");
  utils::logmesg(lmp, "  solver calls = {}\n", polar_ncalls);
  utils::logmesg(lmp, "  iterations/call (min/avg/max) = {} {:.2f} {}\n",
                 polar_niter_min, avg, polar_niter_max);
  utils::logmesg(lmp, "  total iterations = {}\n", polar_niter_sum);
  if (polar_nonconv > 0)
    utils::logmesg(lmp, "  WARNING: {} call(s) did not converge to tol {:.3g} "
                   "within maxiter = {}\n", polar_nonconv, tol, maxiter);
}

/* ----------------------------------------------------------------------
   compute(): dispersion + smeared real-space Coulomb + the iterative polar
   solver. The reaction field is handled entirely per pair inside
   charge_charge()/polar() (no per-molecule reaction_field_pre/post passes).
------------------------------------------------------------------------- */

void PairGCPM::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (atom->nmax > nmax) {
    memory->destroy(efield);
    memory->destroy(efield_pol);
    memory->destroy(mu_old);
    nmax = atom->nmax;
    memory->create(efield, nmax, 3, "pair:efield");
    memory->create(efield_pol, nmax, 3, "pair:efield_pol");
    memory->create(mu_old, nmax, 4, "pair:mu_old");
  }

  // zero electric field arrays before accumulation

  int ntotal = atom->nlocal + atom->nghost;
  for (int i = 0; i < ntotal; i++) {
    efield[i][0] = efield[i][1] = efield[i][2] = 0.0;
  }

  dispersion(eflag, vflag);
  charge_charge(eflag, vflag);
  if (enable_polar) {
    if (force->newton_pair) {
      comm_mode = EFIELD;
      comm->reverse_comm(this);
    }
    polar(eflag, vflag, 1);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   Buckingham exp-6 dispersion (Eq. 10 in Paricaud et al.)
   phi = A*exp(-buck2*r) - buck3/r^6
   where buck1=A, buck2=gamma/sigma, buck3=C6 (precomputed in init_one)
------------------------------------------------------------------------- */

void PairGCPM::dispersion(int eflag, int /*vflag*/)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double r,rsq,r2inv,r6inv,rexp,forcebuck,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cut_ljsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        r = sqrt(rsq);
        rexp = exp(-buck2[itype][jtype]*r);

        // forcebuck = -dU/dr * r  (LAMMPS convention: fpair = forcebuck * r2inv)
        forcebuck = buck1[itype][jtype]*buck2[itype][jtype]*r*rexp
                    - 6.0*buck3[itype][jtype]*r6inv;
        fpair = factor_lj*forcebuck*r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = (buck1[itype][jtype]*rexp - buck3[itype][jtype]*r6inv
                   - offset[itype][jtype]) * factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   charge-charge interactions between 2 Gaussian charge distributions
   (Eq. 4 in Paricaud et al.), summed in real space with NO Ewald subtraction.
   Adds the per-pair charge-charge reaction field (term A) and the charge->dipole
   reaction-field contribution to efield (term B).
   special_coul scales the smeared Coulomb (factor_coul = 0 fully excludes
   intramolecular pairs; with no k-space there is nothing to subtract back). The
   reaction-field terms are continuum-cavity contributions and are NOT scaled by
   special_coul -- they apply to every pair within cut_coul.
------------------------------------------------------------------------- */

void PairGCPM::charge_charge(int eflag, int /*vflag*/)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,ecoul,fpair;
  double r,r2inv,forcecoul,factor_coul;
  double arg,expa,erfa,falpha;
  double efield_scalar;
  double rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;

  ecoul = 0.0;
  forcecoul = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      if (qtmp == 0.0 && q[j] == 0.0) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);

        bool has_force = (qtmp != 0.0 && q[j] != 0.0);

        double prefactor = 0.0;
        if (rsq < cut_coulsq) {
          arg = alpha_ij[itype][jtype] * r;
          expa = MathSpecial::expmsq(arg);
          erfa = 1.0 - (MathSpecial::my_erfcx(arg) * expa);

          // smeared Coulomb scalar (no Ewald real-space subtraction)
          falpha = erfa - EWALD_F*arg*expa;

          // charge-independent field scalar (q[j] factored out so the Newton
          // partner can reuse the same value with q[i] in the reverse direction)
          double scale = qqrd2e / r;
          efield_scalar = factor_coul * scale * falpha * r2inv;

          if (has_force) {
            prefactor = qqrd2e*qtmp*q[j]/r;
            forcecoul = factor_coul * prefactor * falpha;
          }

          // (B) charge->dipole reaction field: efield_i += -c_rf*q[j]*r_vec.
          // Folded into efield_scalar (which multiplies r_vec*q[j]); not scaled
          // by factor_coul (the intramolecular pairs supply the self term).
          if (enable_rf) efield_scalar -= c_rf;

        } else {
          forcecoul = 0.0;
          efield_scalar = 0.0;
          erfa = 0.0;
        }

        if (has_force) {
          fpair = forcecoul * r2inv;

          // (A) charge-charge reaction-field force: -qi*qj*c_rf*r_vec
          if (enable_rf && rsq < cut_coulsq) fpair -= qtmp*q[j]*c_rf;

          f[i][0] += delx*fpair;
          f[i][1] += dely*fpair;
          f[i][2] += delz*fpair;

          if (newton_pair || j < nlocal) {
            f[j][0] -= delx*fpair;
            f[j][1] -= dely*fpair;
            f[j][2] -= delz*fpair;
          }

          if (eflag) {
            if (rsq < cut_coulsq) {
              // KNOWN DEFECT (2026-07-16, see the stage 6 section of
              // src/GCPM/gcpm_lammps_vs_fortran.md): this pair energy does
              // not vanish at the cutoff -- there is no shift constant, so
              // E(rc) = qi*qj*qqrd2e*(1 + B0/2)/rc, up to ~66 kcal/mol for
              // an M-M pair. The Fortran GCPM code truncates by molecule
              // COM-COM distance, where these constants sum to exactly zero
              // over each (neutral) molecule pair; the atom-atom truncation
              // used here does not cancel them. Consequence: the reported
              // ecoul/pe (and NVE etotal) jump at every cutoff crossing and
              // random-walk by O(10^4) kcal/mol on bulk decks, while forces
              // (discontinuous only at the (1-B0) ~ 2% level for eps_rf =
              // 78.4), structure, pressure, and induced dipoles remain
              // essentially correct. Planned fix: subtract the per-type-pair
              // constant
              //   qi*qj*(factor_coul*qqrd2e*erfa(alpha_ij*rc)/rc
              //          + 0.5*c_rf*rc^2)
              // so the energy is continuous at rc. The shift is constant
              // inside the cutoff: forces and trajectories are unchanged,
              // only the energy bookkeeping (and the absolute-pe offset
              // against the Fortran single-point records) changes.
              ecoul = factor_coul * prefactor * erfa;
              // (A) charge-charge reaction-field energy: 0.5*qi*qj*c_rf*r^2
              if (enable_rf) ecoul += 0.5*qtmp*q[j]*c_rf*rsq;
            } else ecoul = 0.0;
          }

          if (evflag) ev_tally(i,j,nlocal,newton_pair,
                               0.0,ecoul,fpair,delx,dely,delz);
        }

        // efield at i from q[j]: non-zero only when q[j] != 0
        if (q[j] != 0.0) {
          efield[i][0] += delx * q[j] * efield_scalar;
          efield[i][1] += dely * q[j] * efield_scalar;
          efield[i][2] += delz * q[j] * efield_scalar;
        }

        // efield at j from q[i]: non-zero only when qtmp != 0
        if (newton_pair || j < nlocal) {
          if (qtmp != 0.0) {
            efield[j][0] -= delx * qtmp * efield_scalar;
            efield[j][1] -= dely * qtmp * efield_scalar;
            efield[j][2] -= delz * qtmp * efield_scalar;
          }
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   induced electric field (Eq. 5) from the current dipole estimates, plus the
   per-pair dipole->dipole reaction field (term C): efield_pol_i += c_rf*mu_j
   for every M-M pair within cut_coul, and the self term efield_pol_i += c_rf*mu_i
   (no intramolecular M-M pair exists, so it is added explicitly here).
------------------------------------------------------------------------- */

void PairGCPM::compute_induced_efield(int half)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,r,r2inv,r3inv;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double sigmaM_ij,sigmaM_ij2;
  double _erf,expmsq,rdivsigmaM,f,g;
  double Tij[3][3];

  double **x = atom->x;
  double **mu = atom->mu;
  int *type = atom->type;
  double qqrd2e = force->qqrd2e;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  // zero efield_pol for all atoms (local + ghost) before accumulation

  int ntotal = atom->nlocal + atom->nghost;
  for (i = 0; i < ntotal; i++)
    efield_pol[i][0] = efield_pol[i][1] = efield_pol[i][2] = 0.0;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    if (mu[i][3] == 0.0) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (mu[j][3] == 0.0) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);
        r3inv = 1.0/rsq/r;

        sigmaM_ij = sigmaM[itype][jtype];
        sigmaM_ij2 = sigmaM_ij * sigmaM_ij;

        // Eq. (7): scalars f and g for the T_ij tensor

        _erf = erf(r / (2.0 * sigmaM_ij));
        expmsq = exp(-r * r / 4.0 / sigmaM_ij2);
        rdivsigmaM = r / MY_PIS / sigmaM_ij;
        f = _erf - (rdivsigmaM + rdivsigmaM * rsq / sigmaM_ij2 / 6.0) * expmsq;
        g = _erf - rdivsigmaM * expmsq;

        // Eq. (6): T_ij = 3f*r^-5*r_ij*r_ij - g*r^-3*I (symmetric, T_ij = T_ji)

        f *= 3.0 * r2inv;
        Tij[0][0] = r3inv * (f * delx * delx - g);
        Tij[0][1] = r3inv * f * delx * dely;
        Tij[0][2] = r3inv * f * delx * delz;
        Tij[1][1] = r3inv * (f * dely * dely - g);
        Tij[1][2] = r3inv * f * dely * delz;
        Tij[2][2] = r3inv * (f * delz * delz - g);

        // E_p_i += T_ij . mu_j  (+ the dipole->dipole RF field c_rf*mu_j, term C)

        double rf = (enable_rf && rsq < cut_coulsq) ? c_rf : 0.0;
        efield_pol[i][0] += qqrd2e * (Tij[0][0]*mu[j][0] + Tij[0][1]*mu[j][1] + Tij[0][2]*mu[j][2]) + rf*mu[j][0];
        efield_pol[i][1] += qqrd2e * (Tij[0][1]*mu[j][0] + Tij[1][1]*mu[j][1] + Tij[1][2]*mu[j][2]) + rf*mu[j][1];
        efield_pol[i][2] += qqrd2e * (Tij[0][2]*mu[j][0] + Tij[1][2]*mu[j][1] + Tij[2][2]*mu[j][2]) + rf*mu[j][2];

        // Newton partner: E_p_j += T_ji . mu_i = T_ij . mu_i (T symmetric)

        if ((newton_pair || j < nlocal) && half) {
          efield_pol[j][0] += qqrd2e * (Tij[0][0]*mu[i][0] + Tij[0][1]*mu[i][1] + Tij[0][2]*mu[i][2]) + rf*mu[i][0];
          efield_pol[j][1] += qqrd2e * (Tij[0][1]*mu[i][0] + Tij[1][1]*mu[i][1] + Tij[1][2]*mu[i][2]) + rf*mu[i][1];
          efield_pol[j][2] += qqrd2e * (Tij[0][2]*mu[i][0] + Tij[1][2]*mu[i][1] + Tij[2][2]*mu[i][2]) + rf*mu[i][2];
        }
      }
    }
  }

  // (C) self term: a molecule's own induced dipole reacts on its own M site.
  // There is no intramolecular M-M pair, so add it explicitly (local atoms).

  if (enable_rf) {
    for (i = 0; i < nlocal; i++) {
      if (mu[i][3] == 0.0) continue;
      efield_pol[i][0] += c_rf*mu[i][0];
      efield_pol[i][1] += c_rf*mu[i][1];
      efield_pol[i][2] += c_rf*mu[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   polar interactions: iterative solver for induced dipoles (Eqs. 3 and 5).
   The charge-dipole force kernel uses the pure smeared scalar Phi = falpha (no
   Ewald subtraction); the induced-dipole initial guess is the previous step's
   dipoles (Fortran warm-start, no separate E_p = 0 seeding pass); the per-pair
   charge-dipole reaction-field force (term D) is added in the doA/doB loop. The
   dipole->dipole RF field (term C) is in compute_induced_efield(), and the
   charge->dipole RF field (term B) is folded into efield in charge_charge(), so
   the Eq.(9) energy -1/2 p.E_q already carries all the RF polarization energy.
------------------------------------------------------------------------- */

void PairGCPM::polar(int eflag, int vflag, int neigh_half)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,ecoul;
  double rsq,rinv,r2inv,factor_coul;
  int *ilist,*jlist,*numneigh,**firstneigh;

  ecoul = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Initial guess for the induced dipoles (as in the Fortran GCPM code):
  // start from the dipoles carried over from the previous timestep (warm
  // start). On a cold start mu = 0, so the first iteration below reduces to
  // mu = alpha*E_q automatically. No separate E_p = 0 seeding pass is used.

  for (i = 0; i < nlocal; i++) {
    if (mu[i][3] != 0.0) {
      mu_old[i][0] = mu[i][0];
      mu_old[i][1] = mu[i][1];
      mu_old[i][2] = mu[i][2];
    }
  }

  int iter, converged_run = 0;
  for (iter = 0; iter < maxiter; iter++) {

    // Eq. (5): compute E_p (incl. the dipole->dipole RF field, term C) from the
    // current dipole estimates, then update dipoles from the total field

    compute_induced_efield(neigh_half);

    // communicate and sum per-atom induced efield

    if (newton_pair && neigh_half == 1) {
      comm_mode = EFIELD_POL;
      comm->reverse_comm(this);
    }

    // Eq. (3): p_i = alpha_i * (E_q_i + E_p_i)

    for (i = 0; i < nlocal; i++) {
      itype = type[i];
      if (mu[i][3] != 0.0) {
        mu[i][0] = alpha_pol[itype][itype] * (efield[i][0] + efield_pol[i][0]) / qqrd2e;
        mu[i][1] = alpha_pol[itype][itype] * (efield[i][1] + efield_pol[i][1]) / qqrd2e;
        mu[i][2] = alpha_pol[itype][itype] * (efield[i][2] + efield_pol[i][2]) / qqrd2e;
      }
    }

    // communicate updated dipoles for next iteration of induced field calculation

    comm->forward_comm(this);

    // check for convergence of dipoles: max change in any dipole magnitude < tol
    //  Paricaud et al. before Eq. (8)

    int converged = 1, all_converged = 0;
    for (i = 0; i < nlocal; i++) {
      if (mu[i][3] != 0.0) {
        double diff = (mu_old[i][0]-mu[i][0])*(mu_old[i][0]-mu[i][0]) +
                 (mu_old[i][1]-mu[i][1])*(mu_old[i][1]-mu[i][1]) +
                 (mu_old[i][2]-mu[i][2])*(mu_old[i][2]-mu[i][2]);
        if (diff > tol*tol) {
          converged = 0;
          break;
        }
      }
    }

    MPI_Allreduce(&converged, &all_converged, 1, MPI_INT, MPI_MIN, world);
    if (all_converged) { converged_run = 1; break; }

    for (i = 0; i < nlocal; i++) {
      if (mu[i][3] != 0.0) {
        mu_old[i][0] = mu[i][0];
        mu_old[i][1] = mu[i][1];
        mu_old[i][2] = mu[i][2];
      }
    }
  }

  // record solver convergence statistics for finish(). iterations performed are
  // iter+1 on convergence (loop broke after that pass) or maxiter otherwise.
  record_polar_iters(converged_run ? iter + 1 : maxiter, converged_run);

  // After convergence: forces and energy from charge-induced-dipole interaction.
  // U_pol = -1/2 * sum_i p_i . E_q_i  (Eq. 9 in Paricaud et al.). Because the
  // charge->dipole RF field (term B) is already folded into efield, this energy
  // includes the charge-dipole reaction-field energy.

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    // polarization energy -1/2 p_i.E_q_i (Eq. 9), tallied once per dipole atom.
    if (mu[i][3] != 0.0 && eflag) {
      ecoul = -0.5 * (mu[i][0]*efield[i][0] + mu[i][1]*efield[i][1] + mu[i][2]*efield[i][2]);
      if (eflag_global) eng_coul += ecoul;
      if (eflag_atom) eatom[i] += ecoul;
    }

    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      // a charge-induced-dipole pair contributes a force to BOTH partners.
      //   A: dipole on i with charge on j      B: dipole on j with charge on i

      int doA = (mu[i][3] != 0.0 && q[j] != 0.0);
      int doB = (qtmp != 0.0 && mu[j][3] != 0.0);
      if (!doA && !doB) continue;

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

        // pure smeared charge-dipole kernel (no Ewald subtraction):
        //   Phi      = erf(a r) - (2/sqrt(pi)) a r exp(-(a r)^2)   [= falpha]
        //   dPhi/dr  = 2 (2/sqrt(pi)) a^3 r^2 exp(-(a r)^2)
        double aij = alpha_ij[itype][jtype];
        double aijr = aij * r;
        double expa = MathSpecial::expmsq(aijr);
        double erfa = 1.0 - MathSpecial::my_erfcx(aijr) * expa;
        double Phi = erfa - EWALD_F*aijr*expa;
        double dPhi_dr = 2.0*EWALD_F*rsq*aij*aij*aij*expa;
        double dcoeff = 3.0*Phi - r*dPhi_dr;

        // Interaction A: dipole i with charge j; del = x_i - x_j (charge->dipole)
        if (doA) {
          double pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          double fqj = factor_coul * qqrd2e * q[j];
          double pre1 = fqj*r5inv * dcoeff * pidotr;
          double pre2 = fqj*r3inv * Phi;

          double fcx = pre2*mu[i][0] - pre1*delx;   // force on dipole i
          double fcy = pre2*mu[i][1] - pre1*dely;
          double fcz = pre2*mu[i][2] - pre1*delz;

          // (D) reaction field: charge j sits in the uniform reaction field
          // c_rf*mu_i of dipole i -> force q[j]*c_rf*mu_i on charge j, reaction
          // -q[j]*c_rf*mu_i on dipole i. Not scaled by factor_coul.
          double rfx = 0.0, rfy = 0.0, rfz = 0.0;
          if (enable_rf) {
            double qc = q[j]*c_rf;
            rfx = qc*mu[i][0]; rfy = qc*mu[i][1]; rfz = qc*mu[i][2];
          }

          f[i][0] += fcx - rfx;
          f[i][1] += fcy - rfy;
          f[i][2] += fcz - rfz;

          torque[i][0] += pre2 * (mu[i][1]*delz - mu[i][2]*dely);
          torque[i][1] += pre2 * (mu[i][2]*delx - mu[i][0]*delz);
          torque[i][2] += pre2 * (mu[i][0]*dely - mu[i][1]*delx);

          // torque from the charge->dipole RF field (term B) at dipole i:
          // E_rf = -c_rf*q[j]*del, so torque += mu_i x E_rf. Required so the net
          // torque mu x (E_q + E_p) stays ~0 (the induced dipole has no torque).
          if (enable_rf) {
            double bc = -c_rf*q[j];
            torque[i][0] += bc * (mu[i][1]*delz - mu[i][2]*dely);
            torque[i][1] += bc * (mu[i][2]*delx - mu[i][0]*delz);
            torque[i][2] += bc * (mu[i][0]*dely - mu[i][1]*delx);
          }

          if ((newton_pair || j < nlocal) && neigh_half == 1) {
            f[j][0] -= fcx - rfx;   // reaction on charge j (incl. RF force)
            f[j][1] -= fcy - rfy;
            f[j][2] -= fcz - rfz;
          }

          vtally_force(i, j, neigh_half, fcx, fcy, fcz, delx, dely, delz);
        }

        // Interaction B: dipole j with charge i; the dipole-charge vector is
        // x_j - x_i = -del, so mu[j].(x_j-x_i) = -(mu[j].del)
        if (doB) {
          double pjdotr = -(mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz);
          double fqi = factor_coul * qqrd2e * qtmp;
          double pre1 = fqi*r5inv * dcoeff * pjdotr;
          double pre2 = fqi*r3inv * Phi;

          // force on dipole j = pre2*mu[j] - pre1*(x_j-x_i) = pre2*mu[j] + pre1*del
          double fdx = pre2*mu[j][0] + pre1*delx;
          double fdy = pre2*mu[j][1] + pre1*dely;
          double fdz = pre2*mu[j][2] + pre1*delz;

          // (D) reaction field: charge i sits in the uniform reaction field
          // c_rf*mu_j of dipole j -> force q[i]*c_rf*mu_j on charge i.
          double rfx = 0.0, rfy = 0.0, rfz = 0.0;
          if (enable_rf) {
            double qc = qtmp*c_rf;
            rfx = qc*mu[j][0]; rfy = qc*mu[j][1]; rfz = qc*mu[j][2];
          }

          f[i][0] += -fdx + rfx;   // reaction on charge i (incl. RF force)
          f[i][1] += -fdy + rfy;
          f[i][2] += -fdz + rfz;

          if ((newton_pair || j < nlocal) && neigh_half == 1) {
            f[j][0] += fdx - rfx;   // force on dipole j (incl. RF reaction)
            f[j][1] += fdy - rfy;
            f[j][2] += fdz - rfz;
            // torque on dipole j: tau = mu[j] x E, with E along (x_j-x_i) = -del
            torque[j][0] -= pre2 * (mu[j][1]*delz - mu[j][2]*dely);
            torque[j][1] -= pre2 * (mu[j][2]*delx - mu[j][0]*delz);
            torque[j][2] -= pre2 * (mu[j][0]*dely - mu[j][1]*delx);

            // torque from the charge->dipole RF field (term B) at dipole j:
            // E_rf at j from charge i = -c_rf*q[i]*(x_j-x_i) = +c_rf*qtmp*del
            if (enable_rf) {
              double bc = c_rf*qtmp;
              torque[j][0] += bc * (mu[j][1]*delz - mu[j][2]*dely);
              torque[j][1] += bc * (mu[j][2]*delx - mu[j][0]*delz);
              torque[j][2] += bc * (mu[j][0]*dely - mu[j][1]*delx);
            }
          }

          vtally_force(i, j, neigh_half, -fdx, -fdy, -fdz, delx, dely, delz);
        }
      }
    }
  }

  // dipole-dipole polarization force and torque (gradient at fixed converged
  // dipoles of the Eq. (8) dipole-dipole term). The dipole->dipole reaction
  // field is a uniform field (zero spatial gradient in the cavity bulk), so it
  // contributes no force here; only the smeared T_ij term does.

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mu[i][3] == 0.0) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (mu[j][3] == 0.0) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        double r = sqrt(rsq);
        double r3inv = r2inv/r;
        double r5inv = r3inv*r2inv;

        double s  = sigmaM[itype][jtype];
        double s2 = s*s, s3 = s2*s, s5 = s3*s2;
        double expmsq = exp(-rsq/(4.0*s2));
        double _erf = erf(r/(2.0*s));
        double rds = r/(MY_PIS*s);                       // r/(sqrt(pi)*s)

        // Eq. (7) scalars f, g and their radial derivatives f'(r), g'(r)
        double f_s = _erf - (rds + rds*rsq/(6.0*s2))*expmsq;
        double g_s = _erf - rds*expmsq;
        double df  = rsq*rsq/(12.0*MY_PIS*s5)*expmsq;     // f'(r)
        double dg  = rsq/(2.0*MY_PIS*s3)*expmsq;          // g'(r)

        double pir = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
        double pjr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
        double pij = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];

        // radial coefficients: d(3f/r^5)/dr / r  and  d(g/r^3)/dr / r
        double dA_over_r = 3.0*df*r2inv*r2inv*r2inv - 15.0*f_s*r5inv*r2inv;
        double dB_over_r = dg*r2inv*r2inv - 3.0*g_s*r5inv;
        double rad = qqrd2e*(pir*pjr*dA_over_r - pij*dB_over_r);
        double Aq  = qqrd2e*3.0*f_s*r5inv;

        double fdx = rad*delx + Aq*(pjr*mu[i][0] + pir*mu[j][0]);
        double fdy = rad*dely + Aq*(pjr*mu[i][1] + pir*mu[j][1]);
        double fdz = rad*delz + Aq*(pjr*mu[i][2] + pir*mu[j][2]);

        f[i][0] += fdx;
        f[i][1] += fdy;
        f[i][2] += fdz;

        // torque tau = p x E_p, with E_p = qqrd2e * T_ij . p (Eqs. 5-6, T symmetric)
        double Txx = (3.0*f_s*delx*delx*r2inv - g_s)*r3inv;
        double Tyy = (3.0*f_s*dely*dely*r2inv - g_s)*r3inv;
        double Tzz = (3.0*f_s*delz*delz*r2inv - g_s)*r3inv;
        double Txy = 3.0*f_s*delx*dely*r2inv*r3inv;
        double Txz = 3.0*f_s*delx*delz*r2inv*r3inv;
        double Tyz = 3.0*f_s*dely*delz*r2inv*r3inv;

        double Eix = qqrd2e*(Txx*mu[j][0] + Txy*mu[j][1] + Txz*mu[j][2]);
        double Eiy = qqrd2e*(Txy*mu[j][0] + Tyy*mu[j][1] + Tyz*mu[j][2]);
        double Eiz = qqrd2e*(Txz*mu[j][0] + Tyz*mu[j][1] + Tzz*mu[j][2]);
        // dipole->dipole RF field (term C) at i: E_rf = c_rf*mu_j -> torque
        // mu_i x (c_rf*mu_j). Keeps the net dipole torque mu x (E_q+E_p) ~ 0.
        if (enable_rf && rsq < cut_coulsq) {
          Eix += c_rf*mu[j][0]; Eiy += c_rf*mu[j][1]; Eiz += c_rf*mu[j][2];
        }

        torque[i][0] += mu[i][1]*Eiz - mu[i][2]*Eiy;
        torque[i][1] += mu[i][2]*Eix - mu[i][0]*Eiz;
        torque[i][2] += mu[i][0]*Eiy - mu[i][1]*Eix;

        if ((newton_pair || j < nlocal) && neigh_half == 1) {
          f[j][0] -= fdx;
          f[j][1] -= fdy;
          f[j][2] -= fdz;

          double Ejx = qqrd2e*(Txx*mu[i][0] + Txy*mu[i][1] + Txz*mu[i][2]);
          double Ejy = qqrd2e*(Txy*mu[i][0] + Tyy*mu[i][1] + Tyz*mu[i][2]);
          double Ejz = qqrd2e*(Txz*mu[i][0] + Tyz*mu[i][1] + Tzz*mu[i][2]);
          if (enable_rf && rsq < cut_coulsq) {
            Ejx += c_rf*mu[i][0]; Ejy += c_rf*mu[i][1]; Ejz += c_rf*mu[i][2];
          }
          torque[j][0] += mu[j][1]*Ejz - mu[j][2]*Ejy;
          torque[j][1] += mu[j][2]*Ejx - mu[j][0]*Ejz;
          torque[j][2] += mu[j][0]*Ejy - mu[j][1]*Ejx;
        }

        vtally_force(i, j, neigh_half, fdx, fdy, fdz, delx, dely, delz);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int PairGCPM::pack_forward_comm(int n, int *list, double *buf,
  int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;
  double **mu = atom->mu;
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = mu[j][0];
    buf[m++] = mu[j][1];
    buf[m++] = mu[j][2];
    buf[m++] = mu[j][3];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairGCPM::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  double **mu = atom->mu;
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    mu[i][0] = buf[m++];
    mu[i][1] = buf[m++];
    mu[i][2] = buf[m++];
    mu[i][3] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairGCPM::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  double **arr = (comm_mode == EFIELD) ? efield : efield_pol;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = arr[i][0];
    buf[m++] = arr[i][1];
    buf[m++] = arr[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairGCPM::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  double **arr = (comm_mode == EFIELD) ? efield : efield_pol;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    arr[j][0] += buf[m++];
    arr[j][1] += buf[m++];
    arr[j][2] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   reaction-field setup: validate, compute the constant prefactor c_rf, and
   size the per-molecule tables once (molecule count is fixed for a run).
------------------------------------------------------------------------- */

void PairGCPM::setup_reaction_field()
{
  if (!enable_rf) return;

  if (!enable_polar)
    error->all(FLERR,"Pair gcpm reaction field requires enable_polar = 1");
  if (!atom->molecule_flag)
    error->all(FLERR,"Pair gcpm reaction field requires atom molecule IDs");

  // C_RF = (eps_rf-1)/(2*pi*eps0*(2*eps_rf+1)*rc^3)
  //      = 2*qqrd2e*(eps_rf-1)/((2*eps_rf+1)*rc^3)   since qqrd2e = 1/(4*pi*eps0)
  // qqrd2e is baked in so R shares the field units of efield.

  double qqrd2e = force->qqrd2e;
  double rc3 = cut_coul*cut_coul*cut_coul;
  c_rf = 2.0*qqrd2e*(eps_rf - 1.0) / ((2.0*eps_rf + 1.0)*rc3);

  // size the per-molecule tables once: the molecule count (largest molecule
  // id over all ranks) is fixed for a given run, invariant under migration

  tagint maxmol_local = 0;
  for (int i = 0; i < atom->nlocal; i++)
    if (atom->molecule[i] > maxmol_local) maxmol_local = atom->molecule[i];
  tagint maxmol = 0;
  MPI_Allreduce(&maxmol_local, &maxmol, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  nmol = (int) maxmol;
  grow_mol_arrays(nmol);
}

/* ----------------------------------------------------------------------
   reaction field from the permanent molecular dipoles (Eq. 11, R_i^q),
   folded into efield at the M sites. Used by the per-molecule reaction-field
   path of the derived PairGCPMLong (called before polar()).
------------------------------------------------------------------------- */

void PairGCPM::reaction_field_pre()
{
  if (!enable_rf) return;

  compute_molecular_dipoles();
  reaction_field(mol_mu, mol_Rq);

  double **mu = atom->mu;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (mu[i][3] == 0.0) continue;
    int m = (int) molecule[i];
    efield[i][0] += mol_Rq[m][0];
    efield[i][1] += mol_Rq[m][1];
    efield[i][2] += mol_Rq[m][2];
  }
}

/* ----------------------------------------------------------------------
   permanent-charge reaction-field energy (Eq. 12) and the reaction-field
   site forces F_k = q_k*(R_m^q + 1/2 R_m^p), applied after polar(). Used by
   the per-molecule reaction-field path of the derived PairGCPMLong. mol_Rp
   is built from the converged induced dipoles. The -1/2 p_i.R_i^q part of
   the energy was already tallied inside polar() via the folded efield, so
   here we add only the permanent-permanent term -1/2 sum_i mu_i.R_i^q.
------------------------------------------------------------------------- */

void PairGCPM::reaction_field_post(int eflag)
{
  if (!enable_rf) return;

  double **mu = atom->mu;
  double **f = atom->f;
  double *q = atom->q;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;

  // induced molecular dipoles (on the M sites) -> R_i^p (Eq. 11)
  for (int m = 0; m <= nmol; m++)
    mol_p[m][0] = mol_p[m][1] = mol_p[m][2] = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mu[i][3] == 0.0) continue;
    int m = (int) molecule[i];
    mol_p[m][0] = mu[i][0];
    mol_p[m][1] = mu[i][1];
    mol_p[m][2] = mu[i][2];
  }
  MPI_Allreduce(MPI_IN_PLACE, &mol_p[0][0], 3*(nmol+1), MPI_DOUBLE, MPI_SUM, world);
  reaction_field(mol_p, mol_Rp);

  // U_qq^RF = -1/2 sum_i mu_i.R_i^q, attributed to the M site of molecule i
  if (eflag) {
    double e_rf = 0.0;
    for (int i = 0; i < nlocal; i++) {
      if (mu[i][3] == 0.0) continue;
      int m = (int) molecule[i];
      double e = -0.5 * (mol_mu[m][0]*mol_Rq[m][0] +
                         mol_mu[m][1]*mol_Rq[m][1] +
                         mol_mu[m][2]*mol_Rq[m][2]);
      e_rf += e;
      if (eflag_atom) eatom[i] += e;
    }
    if (eflag_global) eng_coul += e_rf;
  }

  // reaction-field forces on every charged site of each molecule. The RF force
  // is a per-molecule (many-body) field, not a pairwise term; its virial is
  // tallied in the per-atom form x_i (x) f_i (added explicitly since fdotr is
  // disabled). Applied to local atoms only, identically on the CPU and GPU.
  double **x = atom->x;
  for (int i = 0; i < nlocal; i++) {
    if (q[i] == 0.0) continue;
    int m = (int) molecule[i];
    double fx = q[i]*(mol_Rq[m][0] + 0.5*mol_Rp[m][0]);
    double fy = q[i]*(mol_Rq[m][1] + 0.5*mol_Rp[m][1]);
    double fz = q[i]*(mol_Rq[m][2] + 0.5*mol_Rp[m][2]);
    f[i][0] += fx;
    f[i][1] += fy;
    f[i][2] += fz;
    if (vflag_global) {
      virial[0] += x[i][0]*fx; virial[1] += x[i][1]*fy; virial[2] += x[i][2]*fz;
      virial[3] += x[i][0]*fy; virial[4] += x[i][0]*fz; virial[5] += x[i][1]*fz;
    }
    if (vflag_atom) {
      vatom[i][0] += x[i][0]*fx; vatom[i][1] += x[i][1]*fy; vatom[i][2] += x[i][2]*fz;
      vatom[i][3] += x[i][0]*fy; vatom[i][4] += x[i][0]*fz; vatom[i][5] += x[i][1]*fz;
    }
  }
}

/* ----------------------------------------------------------------------
   tally the virial of a force (fx,fy,fz) on atom i interacting with j
------------------------------------------------------------------------- */

void PairGCPM::vtally_force(int i, int j, int neigh_half,
                           double fx, double fy, double fz,
                           double delx, double dely, double delz)
{
  if (!vflag_either) return;

  if (neigh_half) {
    // half list: standard pairwise virial (handles the j partner and newton)
    ev_tally_xyz(i, j, atom->nlocal, force->newton_pair, 0.0, 0.0,
                 fx, fy, fz, delx, dely, delz);
  } else {
    // full list: the pair (i,j) is visited twice (once per center), so add
    // half the pairwise virial del (x) F here; the other half comes from the
    // mirrored visit. This is the correct atomic virial for a non-central
    // force (the per-atom x_i (x) f_i form is wrong for non-central forces).
    double v0 = 0.5*delx*fx, v1 = 0.5*dely*fy, v2 = 0.5*delz*fz;
    double v3 = 0.5*delx*fy, v4 = 0.5*delx*fz, v5 = 0.5*dely*fz;
    if (vflag_global) {
      virial[0] += v0; virial[1] += v1; virial[2] += v2;
      virial[3] += v3; virial[4] += v4; virial[5] += v5;
    }
    if (vflag_atom) {
      vatom[i][0] += v0; vatom[i][1] += v1; vatom[i][2] += v2;
      vatom[i][3] += v3; vatom[i][4] += v4; vatom[i][5] += v5;
    }
  }
}

/* ----------------------------------------------------------------------
   grow the per-molecule reaction-field tables to hold molecule ids 0..n
------------------------------------------------------------------------- */

void PairGCPM::grow_mol_arrays(int n)
{
  if (n <= nmol_max) return;
  nmol_max = n;
  memory->destroy(mol_mu);
  memory->destroy(mol_p);
  memory->destroy(mol_x);
  memory->destroy(mol_Rq);
  memory->destroy(mol_Rp);
  memory->create(mol_mu, nmol_max+1, 3, "pair:mol_mu");
  memory->create(mol_p,  nmol_max+1, 3, "pair:mol_p");
  memory->create(mol_x,  nmol_max+1, 3, "pair:mol_x");
  memory->create(mol_Rq, nmol_max+1, 3, "pair:mol_Rq");
  memory->create(mol_Rp, nmol_max+1, 3, "pair:mol_Rp");
}

/* ----------------------------------------------------------------------
   permanent molecular dipoles mu_i = sum_{a in i} q_a r_a (Eq. 11) and the
   cavity center (the M site, unwrapped) for every molecule. Atoms are
   unwrapped with their image flags so the neutral-molecule dipole is
   periodic-image independent, and the result is reduced across ranks so
   every rank holds the full per-molecule tables.
------------------------------------------------------------------------- */

void PairGCPM::compute_molecular_dipoles()
{
  double **x = atom->x;
  double *q = atom->q;
  double **mu = atom->mu;
  tagint *molecule = atom->molecule;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  // nmol and the per-molecule tables are sized once in init_style() (the
  // molecule count is fixed for a given run); only the per-step accumulation
  // happens here.

  for (int m = 0; m <= nmol; m++) {
    mol_mu[m][0] = mol_mu[m][1] = mol_mu[m][2] = 0.0;
    mol_x[m][0]  = mol_x[m][1]  = mol_x[m][2]  = 0.0;
  }

  double ux[3];
  for (int i = 0; i < nlocal; i++) {
    int m = (int) molecule[i];
    if (m <= 0) continue;
    domain->unmap(x[i], image[i], ux);
    mol_mu[m][0] += q[i]*ux[0];
    mol_mu[m][1] += q[i]*ux[1];
    mol_mu[m][2] += q[i]*ux[2];
    // the M site (only atom carrying an induced dipole) is the cavity center;
    // it is local on exactly one rank, so summing gives the correct value
    if (mu[i][3] != 0.0) {
      mol_x[m][0] = ux[0];
      mol_x[m][1] = ux[1];
      mol_x[m][2] = ux[2];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &mol_mu[0][0], 3*(nmol+1), MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, &mol_x[0][0],  3*(nmol+1), MPI_DOUBLE, MPI_SUM, world);
}

/* ----------------------------------------------------------------------
   reaction field per molecule (Eq. 11): R_i = c_rf * sum_j d_j over molecules
   j (including i itself) whose cavity centers lie within cut_coul of i.
   c_rf already carries the qqrd2e factor so R has the same field units as
   efield. O(nmol^2); each rank computes the full table redundantly.
------------------------------------------------------------------------- */

void PairGCPM::reaction_field(double **mol_d, double **mol_R)
{
  for (int i = 1; i <= nmol; i++) {
    double sx = mol_d[i][0];
    double sy = mol_d[i][1];
    double sz = mol_d[i][2];
    for (int j = 1; j <= nmol; j++) {
      if (j == i) continue;
      double delta[3];
      delta[0] = mol_x[i][0] - mol_x[j][0];
      delta[1] = mol_x[i][1] - mol_x[j][1];
      delta[2] = mol_x[i][2] - mol_x[j][2];
      domain->minimum_image(FLERR, delta);
      double rsq = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
      if (rsq < cut_coulsq) {
        sx += mol_d[j][0];
        sy += mol_d[j][1];
        sz += mol_d[j][2];
      }
    }
    mol_R[i][0] = c_rf * sx;
    mol_R[i][1] = c_rf * sy;
    mol_R[i][2] = c_rf * sz;
  }
}

/* ---------------------------------------------------------------------- */

void PairGCPM::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(gamma_buck,n+1,n+1,"pair:gamma_buck");
  memory->create(buck1,n+1,n+1,"pair:buck1");
  memory->create(buck2,n+1,n+1,"pair:buck2");
  memory->create(buck3,n+1,n+1,"pair:buck3");
  memory->create(offset,n+1,n+1,"pair:offset");
  memory->create(alpha_pol,n+1,n+1,"pair:alpha_pol");
  memory->create(sigmaM,n+1,n+1,"pair:sigmaM");
  memory->create(alpha_ij,n+1,n+1,"pair:alpha_ij");
}

/* ---------------------------------------------------------------------- */

void PairGCPM::settings(int narg, char **arg)
{
  if (narg < 3) error->all(FLERR,"Illegal pair_style command");

  enable_polar = utils::numeric(FLERR,arg[0],false,lmp);
  eps_rf = utils::numeric(FLERR,arg[1],false,lmp);
  cut_lj_global = utils::numeric(FLERR,arg[2],false,lmp);

  // optional 4th positional argument: Coulomb cutoff (defaults to cut_lj);
  // then optional keyword/value pairs controlling the induced-dipole solver:
  //   polar/tol <tol>       convergence tolerance on the max dipole change
  //   polar/maxiter <n>     max SCF iterations per solver call

  int iarg = 3;
  cut_coul = cut_lj_global;
  if ((narg > 3) && (strcmp(arg[3],"polar/tol") != 0) &&
      (strcmp(arg[3],"polar/maxiter") != 0)) {
    cut_coul = utils::numeric(FLERR,arg[3],false,lmp);
    iarg = 4;
  }

  while (iarg < narg) {
    if (strcmp(arg[iarg],"polar/tol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style command: "
                                    "polar/tol requires a value");
      tol = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (tol <= 0.0)
        error->all(FLERR,"Illegal pair_style command: polar/tol must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg],"polar/maxiter") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style command: "
                                    "polar/maxiter requires a value");
      maxiter = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (maxiter < 1)
        error->all(FLERR,"Illegal pair_style command: polar/maxiter must be >= 1");
      iarg += 2;
    } else error->all(FLERR,"Illegal pair_style command: unknown keyword {}",
                      arg[iarg]);
  }

  // reaction-field correction: eps_rf (2nd arg) is the continuum dielectric.
  // eps_rf <= 0 disables the correction.

  enable_rf = 0;
  if (eps_rf > 0.0) enable_rf = 1;

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ---------------------------------------------------------------------- */

void PairGCPM::coeff(int narg, char **arg)
{
  // pair_coeff i j epsilon sigma gamma alpha_pol sigmaM [cut_lj]
  if (narg < 7 || narg > 8)
    error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double epsilon_one   = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one     = utils::numeric(FLERR,arg[3],false,lmp);
  double gamma_one     = utils::numeric(FLERR,arg[4],false,lmp);
  double alpha_pol_one = utils::numeric(FLERR,arg[5],false,lmp);
  double sigmaM_one    = utils::numeric(FLERR,arg[6],false,lmp);

  if (gamma_one <= 6.0)
    error->all(FLERR,"Buckingham exp-6 gamma must be > 6");

  double cut_lj_one = cut_lj_global;
  if (narg == 8) cut_lj_one = utils::numeric(FLERR,arg[7],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j]   = epsilon_one;
      sigma[i][j]     = sigma_one;
      gamma_buck[i][j]= gamma_one;
      alpha_pol[i][j] = alpha_pol_one;
      sigmaM[i][j]    = sigmaM_one;
      cut_lj[i][j]    = cut_lj_one;
      setflag[i][j]   = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ----------------------------------------------------------------------
   init_style for the reaction-field (no k-space) form: the smeared Coulomb is
   summed in real space to cut_coul and the long-range tail comes from the
   reaction-field correction. No KSpace style is required.
------------------------------------------------------------------------- */

void PairGCPM::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair gcpm requires atom attribute q");

  if (enable_polar && (!atom->mu_flag || !atom->torque_flag))
    error->all(FLERR,"Pair gcpm requires atom attributes mu and torque for polarizable simulations");

  neighbor->add_request(this);

  cut_coulsq = cut_coul * cut_coul;

  g_ewald = 0.0;

  setup_reaction_field();
}

/* ---------------------------------------------------------------------- */

double PairGCPM::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j]    = mix_energy(epsilon[i][i],epsilon[j][j],sigma[i][i],sigma[j][j]);
    sigma[i][j]      = mix_distance(sigma[i][i],sigma[j][j]);
    gamma_buck[i][j] = 0.5*(gamma_buck[i][i]+gamma_buck[j][j]);
    alpha_pol[i][j]  = mix_distance(alpha_pol[i][i],alpha_pol[j][j]);
    sigmaM[i][j]     = mix_distance(sigmaM[i][i],sigmaM[j][j]);
    cut_lj[i][j]     = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  double gam = gamma_buck[i][j];
  double sig = sigma[i][j];
  double eps = epsilon[i][j];

  // pre-compute standard Buckingham coefficients
  buck1[i][j] = 6.0*eps*exp(gam) / (gam - 6.0);   // A
  buck2[i][j] = gam / sig;                          // 1/rho
  buck3[i][j] = gam*eps*pow(sig,6.0) / (gam - 6.0); // C6

  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  if (offset_flag && (cut_lj[i][j] > 0.0)) {
    double rexp = exp(-buck2[i][j]*cut_lj[i][j]);
    double r6inv = pow(cut_lj[i][j],-6.0);
    offset[i][j] = buck1[i][j]*rexp - buck3[i][j]*r6inv;
  } else offset[i][j] = 0.0;

  double cut = MAX(cut_lj[i][j], cut_coul);

  // per-pair Gaussian width for charge-charge interactions: 1/sqrt(2*(si^2+sj^2))
  double si = sigmaM[i][i], sj = sigmaM[j][j];
  alpha_ij[i][j] = MY_ISQRT2 / sqrt(si*si + sj*sj);
  alpha_ij[j][i] = alpha_ij[i][j];

  cut_ljsq[j][i]    = cut_ljsq[i][j];
  epsilon[j][i]     = epsilon[i][j];
  sigma[j][i]       = sigma[i][j];
  gamma_buck[j][i]  = gamma_buck[i][j];
  buck1[j][i]       = buck1[i][j];
  buck2[j][i]       = buck2[i][j];
  buck3[j][i]       = buck3[i][j];
  offset[j][i]      = offset[i][j];
  alpha_pol[j][i]   = alpha_pol[i][j];
  sigmaM[j][i]      = sigmaM[i][j];

  return cut;
}

/* ---------------------------------------------------------------------- */

void PairGCPM::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&gamma_buck[i][j],sizeof(double),1,fp);
        fwrite(&alpha_pol[i][j],sizeof(double),1,fp);
        fwrite(&sigmaM[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
      }
    }
}

/* ---------------------------------------------------------------------- */

void PairGCPM::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&epsilon[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&gamma_buck[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&alpha_pol[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigmaM[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut_lj[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma_buck[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&alpha_pol[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigmaM[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ---------------------------------------------------------------------- */

void PairGCPM::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
  fwrite(&enable_polar,sizeof(int),1,fp);
  fwrite(&eps_rf,sizeof(double),1,fp);
  fwrite(&tol,sizeof(double),1,fp);
  fwrite(&maxiter,sizeof(int),1,fp);
}

/* ---------------------------------------------------------------------- */

void PairGCPM::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_lj_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_coul,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tail_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&enable_polar,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&eps_rf,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&tol,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&maxiter,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
  MPI_Bcast(&enable_polar,1,MPI_INT,0,world);
  MPI_Bcast(&eps_rf,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&tol,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&maxiter,1,MPI_INT,0,world);

  // derived flag (settings() computes it from eps_rf)
  enable_rf = (eps_rf > 0.0) ? 1 : 0;
}

/* ---------------------------------------------------------------------- */

void PairGCPM::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,epsilon[i][i],sigma[i][i],gamma_buck[i][i]);
}

/* ---------------------------------------------------------------------- */

void PairGCPM::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g\n",i,j,
              epsilon[i][j],sigma[i][j],gamma_buck[i][j],cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

void *PairGCPM::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  if (strcmp(str,"alpha_pol") == 0) return (void *) alpha_pol;
  return nullptr;
}
