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
   Contributing author: Trung Nguyen (U Chicago) with Claude Code Sonnet 4.6
   Reference: Paricaud et al., J. Chem. Phys. 122, 244511 (2005)
   Buckingham exp-6 dispersion: Eq. (10) of the reference
     phi = eps/(1-6/gamma) * [6/gamma * exp(gamma*(1-r/sigma)) - (sigma/r)^6]
   stored as the equivalent standard Buckingham A*exp(-r/rho) - C6/r^6 with:
     A   = 6*eps*exp(gamma)/(gamma-6)
     rho = sigma/gamma  (i.e. buck2 = 1/rho = gamma/sigma)
     C6  = gamma*eps*sigma^6/(gamma-6)
------------------------------------------------------------------------- */

#include "pair_buck6_coul_gauss_long.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "kspace.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace EwaldConst;

#define EPSILON 1.0e-5
//#define GCPM_DEBUG

/* ---------------------------------------------------------------------- */

PairBuck6CoulGaussLong::PairBuck6CoulGaussLong(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  respa_enable = 0;
  single_enable = 0;
  writedata = 1;
  ftable = nullptr;
  cut_respa = nullptr;
  qdist = 0.0;

  enable_polar = 1;
  efield = nullptr;
  efield_pol = nullptr;
  mu_old = nullptr;
  nmax = 0;
  maxiter = 20;
  tol = EPSILON;

  comm_forward = 4;
}

/* ---------------------------------------------------------------------- */

PairBuck6CoulGaussLong::~PairBuck6CoulGaussLong()
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

  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::compute(int eflag, int vflag)
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
  if (enable_polar) polar(eflag, vflag);

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   Buckingham exp-6 dispersion (Eq. 10 in Paricaud et al.)
   phi = A*exp(-buck2*r) - buck3/r^6
   where buck1=A, buck2=gamma/sigma, buck3=C6 (precomputed in init_one)
------------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::dispersion(int eflag, int /*vflag*/)
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
   charge-charge interactions (identical to lj/cut/coul/gauss/long)
------------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::charge_charge(int eflag, int /*vflag*/)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,ecoul,fpair;
  double r,r2inv,forcecoul,factor_coul;
  double grij,expm2,prefactor;
  double efield_i;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double rcu,rqu,sme,smf;
  double erfa,expa,arg,falpha,ealpha;
  double erf,prefactorE,ealphaE;
  double rsq;

  ecoul = 0.0;

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

    if (qtmp == 0.0) continue;

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

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);

        if (rsq < cut_coulsq) {
          grij = g_ewald * r;
          expm2 = MathSpecial::expmsq(grij);
          erf = 1 - (MathSpecial::my_erfcx(grij) * expm2);

          arg = alpha_ij[itype][jtype] * r;
          expa = MathSpecial::expmsq(arg);
          erfa = 1 - (MathSpecial::my_erfcx(arg) * expa);

          prefactor = qqrd2e*qtmp*q[j]/r;
          falpha = erfa - EWALD_F*arg*expa;
          forcecoul = prefactor * (falpha - erf + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor*falpha;

          prefactorE = qqrd2e*q[j]/r;
          efield_i = prefactorE * (falpha - erf + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) efield_i -= (1.0-factor_coul)*prefactorE*falpha;

          ealpha = prefactor * (erfa-erf);
          ealphaE = prefactorE * (erfa - erf);

          if (rsq > rsmooth_sq_c) {
            rcu = r*rsq;
            rqu = rsq*rsq;
            sme = c5_c*rqu*r + c4_c*rqu + c3_c*rcu + c2_c*rsq + c1_c*r + c0_c;
            smf = 5.0*c5_c*rqu + 4.0*c4_c*rcu + 3.0*c3_c*rsq + 2.0*c2_c*r + c1_c;
            forcecoul = forcecoul*sme - ealpha*smf*r;
            ealpha *= sme;

            efield_i = efield_i*sme - ealphaE*smf*r;
            ealphaE *= sme;
          }

          efield_i = efield_i * r2inv;

        } else {
          forcecoul = 0.0;
          efield_i = 0.0;
        }

        fpair = forcecoul * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;

        efield[i][0] += delx * efield_i;
        efield[i][1] += dely * efield_i;
        efield[i][2] += delz * efield_i;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq) {
            ecoul = ealpha;
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor*erfa;
          } else ecoul = 0.0;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             0.0,ecoul,fpair,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   polar interactions: iterative solver for induced dipoles (Eqs. 3-9)
------------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::polar(int eflag, int vflag)
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

  // initial dipole estimate from charge field only (E_p = 0)

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    if (mu[i][3] != 0.0) {
      mu[i][0] = alpha_pol[itype][itype] * efield[i][0] / qqrd2e;
      mu[i][1] = alpha_pol[itype][itype] * efield[i][1] / qqrd2e;
      mu[i][2] = alpha_pol[itype][itype] * efield[i][2] / qqrd2e;
      mu_old[i][0] = mu[i][0];
      mu_old[i][1] = mu[i][1];
      mu_old[i][2] = mu[i][2];
    }
  }

  for (int iter = 0; iter < maxiter; iter++) {

    compute_induced_efield();

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      if (mu[i][3] != 0.0) {
        mu[i][0] = alpha_pol[itype][itype] * (efield[i][0] + efield_pol[i][0]) / qqrd2e;
        mu[i][1] = alpha_pol[itype][itype] * (efield[i][1] + efield_pol[i][1]) / qqrd2e;
        mu[i][2] = alpha_pol[itype][itype] * (efield[i][2] + efield_pol[i][2]) / qqrd2e;
      }
    }

    comm->forward_comm(this);

    int converged = 1, all_converged = 0;
    double diff = 0.0;
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (mu[i][3] != 0.0) {
        if (fabs(mu_old[i][0] - mu[i][0]) > tol ||
            fabs(mu_old[i][1] - mu[i][1]) > tol ||
            fabs(mu_old[i][2] - mu[i][2]) > tol) {
          converged = 0;
          break;
        }
      }
    }

    MPI_Allreduce(&converged, &all_converged, 1, MPI_INT, MPI_MIN, world);
    if (all_converged) break;

    #ifdef GCPM_DEBUG
    if (comm->me == 0) printf("iter = %d: not converged\n", iter+1);
    #endif

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (mu[i][3] != 0.0) {
        mu_old[i][0] = mu[i][0];
        mu_old[i][1] = mu[i][1];
        mu_old[i][2] = mu[i][2];
        mu_old[i][3] = mu[i][3];
      }
    }
  }

  // After convergence: forces and energy from charge-induced-dipole interaction.
  // U_pol = -1/2 * sum_i p_i . E_q_i  (Eq. 9 in Paricaud et al.)

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    if (mu[i][3] == 0.0) continue;

    if (eflag) {
      ecoul = -0.5 * (mu[i][0]*efield[i][0] + mu[i][1]*efield[i][1] + mu[i][2]*efield[i][2]);
      if (evflag) ev_tally_full(i, 0.0, ecoul, 0.0, 0.0, 0.0, 0.0);
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
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

        if (newton_pair || j < nlocal) {
          f[j][0] -= fcx;
          f[j][1] -= fcy;
          f[j][2] -= fcz;
        }

        torque[i][0] += pre2 * (mu[i][1]*delz - mu[i][2]*dely);
        torque[i][1] += pre2 * (mu[i][2]*delx - mu[i][0]*delz);
        torque[i][2] += pre2 * (mu[i][0]*dely - mu[i][1]*delx);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int PairBuck6CoulGaussLong::pack_forward_comm(int n, int *list, double *buf,
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

void PairBuck6CoulGaussLong::unpack_forward_comm(int n, int first, double *buf)
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

void PairBuck6CoulGaussLong::compute_induced_efield()
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,r,r2inv,r3inv;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double sigmaM_ij,sigmaM_ij2,sigmaM_ij3;
  double _erf,expmsq,rdivsigmaM,f,g;
  double Tij[3][3];

  double **x = atom->x;
  double **mu = atom->mu;
  int *type = atom->type;
  double qqrd2e = force->qqrd2e;

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

    double ex, ey, ez;
    ex = ey = ez = 0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (mu[j][3] == 0.0) continue;

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);
        r3inv = 1.0/rsq/r;

        sigmaM_ij = sigmaM[itype][jtype];
        sigmaM_ij2 = sigmaM_ij * sigmaM_ij;
        sigmaM_ij3 = sigmaM_ij2 * sigmaM_ij;

        _erf = erf(r / (2.0 * sigmaM_ij));          // Eq. (7): erf(r/(2*sigma_M))
        expmsq = exp(-r * r / 4.0 / sigmaM_ij2);    // exp(-r^2/(4*sigma_M^2))
        rdivsigmaM = r / MY_PIS / sigmaM_ij;         // r/(sqrt(pi)*sigma_M)
        f = _erf - (rdivsigmaM + rdivsigmaM * rsq / sigmaM_ij2 / 6.0) * expmsq;
        g = _erf - rdivsigmaM * expmsq;

        f *= 3.0 * r2inv;
        Tij[0][0] = r3inv * (f * delx * delx - g);
        Tij[0][1] = r3inv * f * delx * dely;
        Tij[0][2] = r3inv * f * delx * delz;

        Tij[1][0] = r3inv * f * dely * delx;
        Tij[1][1] = r3inv * (f * dely * dely - g);
        Tij[1][2] = r3inv * f * dely * delz;

        Tij[2][0] = r3inv * f * delz * delx;
        Tij[2][1] = r3inv * f * delz * dely;
        Tij[2][2] = r3inv * (f * delz * delz - g);

        ex += Tij[0][0]*mu[j][0] + Tij[0][1]*mu[j][1] + Tij[0][2]*mu[j][2];
        ey += Tij[1][0]*mu[j][0] + Tij[1][1]*mu[j][1] + Tij[1][2]*mu[j][2];
        ez += Tij[2][0]*mu[j][0] + Tij[2][1]*mu[j][1] + Tij[2][2]*mu[j][2];
      }
    }

    efield_pol[i][0] = ex * qqrd2e;
    efield_pol[i][1] = ey * qqrd2e;
    efield_pol[i][2] = ez * qqrd2e;
  }
}

/* ---------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::allocate()
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

void PairBuck6CoulGaussLong::settings(int narg, char **arg)
{
  if (narg < 4 || narg > 5) error->all(FLERR,"Illegal pair_style command");

  coul_smooth = utils::numeric(FLERR,arg[0],false,lmp);
  alpha = utils::numeric(FLERR,arg[1],false,lmp);
  enable_polar = utils::numeric(FLERR,arg[2],false,lmp);
  cut_lj_global = utils::numeric(FLERR,arg[3],false,lmp);
  if (narg == 4) cut_coul = cut_lj_global;
  else cut_coul = utils::numeric(FLERR,arg[4],false,lmp);

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ---------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::coeff(int narg, char **arg)
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

/* ---------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair buck6/coul/gauss/long requires atom attribute q");

  if (enable_polar && (!atom->mu_flag || !atom->torque_flag))
    error->all(FLERR,"Pair buck6/coul/gauss/long requires atom attributes mu and torque for polarizable simulations");

  neighbor->add_request(this, NeighConst::REQ_FULL);

  cut_coulsq = cut_coul * cut_coul;

  c0_c = c1_c = c2_c = c3_c = c4_c = c5_c = 0.0;
  rsmooth_sq_c = cut_coulsq;
  if (coul_smooth < 1.0) {
    double rsm = coul_smooth * cut_coul;
    double rsm_sq = rsm * rsm;
    double denom = pow((cut_coul-rsm),5.0);
    c0_c = cut_coul*cut_coulsq*(cut_coulsq-
           5.0*cut_coul*rsm+10.0*rsm_sq)/denom;
    c1_c = -30.0*(cut_coulsq*rsm_sq)/denom;
    c2_c = 30.0*(cut_coulsq*rsm + cut_coul*rsm_sq)/denom;
    c3_c = -10.0*(cut_coulsq + 4.0*cut_coul*rsm + rsm_sq)/denom;
    c4_c = 15.0*(cut_coul+rsm)/denom;
    c5_c = -6.0/denom;
    rsmooth_sq_c = rsm_sq;
  }

  if (force->kspace == nullptr)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;
}

/* ---------------------------------------------------------------------- */

double PairBuck6CoulGaussLong::init_one(int i, int j)
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

  double cut = MAX(cut_lj[i][j], cut_coul+2.0*qdist);

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

void PairBuck6CoulGaussLong::write_restart(FILE *fp)
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

void PairBuck6CoulGaussLong::read_restart(FILE *fp)
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

void PairBuck6CoulGaussLong::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&coul_smooth,sizeof(double),1,fp);
  fwrite(&alpha,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ---------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_lj_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_coul,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&coul_smooth,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&alpha,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tail_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&coul_smooth,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&alpha,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,epsilon[i][i],sigma[i][i],gamma_buck[i][i]);
}

/* ---------------------------------------------------------------------- */

void PairBuck6CoulGaussLong::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g\n",i,j,
              epsilon[i][j],sigma[i][j],gamma_buck[i][j],cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

void *PairBuck6CoulGaussLong::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return nullptr;
}
