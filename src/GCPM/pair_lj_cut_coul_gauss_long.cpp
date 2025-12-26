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
   Contributing author: Trung Nguyen (U Chicago)
   Reference: Paricaud et al., J. Chem. Phys. 122, 244511 (2005)
------------------------------------------------------------------------- */

#include "pair_lj_cut_coul_gauss_long.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
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
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace EwaldConst;

#define EPSILON 1.0e-6

/* ---------------------------------------------------------------------- */

PairLJCutCoulGaussLong::PairLJCutCoulGaussLong(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  respa_enable = 1;
  writedata = 1;
  ftable = nullptr;
  qdist = 0.0;
  cut_respa = nullptr;

  efield = nullptr;
  efield_pol = nullptr;
  mu_old = nullptr;
  nmax = 0;

  comm_forward = 4;
}

/* ---------------------------------------------------------------------- */

PairLJCutCoulGaussLong::~PairLJCutCoulGaussLong()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(alpha_pol);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
  memory->destroy(efield);
  memory->destroy(efield_pol);

  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::compute(int eflag, int vflag)
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

  // dispersion interactions

  dispersion(eflag, vflag);

  // charge-charge interactions

  charge_charge(eflag, vflag);

  // polar interactions

  polar(eflag, vflag);

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   dispersion interactions: could be LJ or Buckingham exp-6
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::dispersion(int eflag, int vflag)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double r2inv,r6inv,forcelj,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

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

  // loop over neighbors of my atoms

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

      if (rsq < cutsq[itype][jtype]) {

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        } else forcelj = 0.0;

        fpair = factor_lj*forcelj * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
              offset[itype][jtype];
            evdwl *= factor_lj;
          } else evdwl = 0.0;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   charge-charge interactions
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::charge_charge(int eflag, int vflag)
{
  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  double efield_i;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double rcu,rqu,sme,smf;
  double erfa,expa,arg,falpha,ealpha;
  double erf;
  double rsq;

  ecoul = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **mu = atom->mu;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    efield[i][0] = efield[i][1] = efield[i][2] = 0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);

        if (rsq < cut_coulsq) {
          // long range - real space
          grij = g_ewald * r;
          expm2 = MathSpecial::expmsq(grij);
          erf = 1 - (MathSpecial::my_erfcx(grij) * expm2);

          // gaussian for 1/r alpha_ij contribution
          arg = alpha*r;
          expa = MathSpecial::expmsq(arg);
          erfa = 1 - (MathSpecial::my_erfcx(arg) * expa);

          prefactor = qqrd2e*qtmp*q[j]/r;
          falpha = erfa - EWALD_F*arg*expa;
          forcecoul = prefactor * (falpha - erf + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor*falpha;

          // (q*q/r) * (gauss(alpha_ij) - gauss(alpha_long)
          ealpha = prefactor * (erfa-erf);
          // smoothing term - NOTE: ingnored in special_bonds correction
          // since likely rsmooth_sq_c >> d(special)
          if (rsq > rsmooth_sq_c) {
            rcu = r*rsq;
            rqu = rsq*rsq;
            sme = c5_c*rqu*r + c4_c*rqu + c3_c*rcu + c2_c*rsq + c1_c*r + c0_c;
            smf = 5.0*c5_c*rqu + 4.0*c4_c*rcu + 3.0*c3_c*rsq + 2.0*c2_c*r + c1_c;
            forcecoul = forcecoul*sme - ealpha*smf*r;
            ealpha *= sme;
          }

          
          if (qtmp != 0.0) efield_i = forcecoul / qtmp * r2inv;
          else efield_i = 0.0;

        } else {
          forcecoul = 0.0;
        }

        fpair = forcecoul * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;

        // accumate electric field on atom i efield (Eq. (4) in Paricaud et al.)
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
   polar interactions:
     solve for induced dipoles from electrical fields
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::polar(int eflag, int vflag)
{
  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double rcu,rqu,sme,smf;
  double erfa,expa,arg,falpha,ealpha;
  double erf;

  double rsq;

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **mu = atom->mu;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int maxiter = 10;

  // estimate the initial induced dipoles of the molecules from the electrical fields E = E_q + E_p
  // see Eq. 3 in Paricaud et al. with E_p = 0 for the first iteration
  // note that efield_i computed in charge_charge() is at individual atom level,
  // need to be mapped to the corresponding molecules for E_q

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    // efield here is due to charges only
    // NOTE: using mu[i][3] != 0.0 to indicate a polarizable atom (e.g. the M site of the TIP4P model)

    if (mu[i][3] != 0.0) {
      efield_pol[i][0] = alpha_pol[itype][itype] * efield[i][0];
      efield_pol[i][1] = alpha_pol[itype][itype] * efield[i][1];
      efield_pol[i][2] = alpha_pol[itype][itype] * efield[i][2];
      
      mu[i][0] = efield_pol[i][0];
      mu[i][1] = efield_pol[i][1];
      mu[i][2] = efield_pol[i][2];
      mu_old[i][0] = mu[i][0];
      mu_old[i][1] = mu[i][1];
      mu_old[i][2] = mu[i][2];
    }
    
  }

  for (int iter = 0; iter < maxiter; iter++) {

    // compute the electrical field on each molecule due to the induced dipoles E_p
    // see Eqs. 5-7 in Paricaud et al.

    compute_induced_efield(efield_pol);

    // update the induced dipoles of the molecules from the electrical fields E = E_q + E_p
    // see Eq. 3 in Paricaud et al.
    // note that efield_i computed in charge_charge() is at individual atom level,
    // need to be mapped to the corresponding molecules for E_q
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];

      // efield here is due to charges only
      // NOTE: using mu[i][3] != 0.0 to indicate a polarizable atom (e.g. the M site of the TIP4P model)
      //       so we keep mu[i][3] untouched here

      if (mu[i][3] != 0.0) {
        mu[i][0] = alpha_pol[itype][itype] * (efield[i][0] + efield_pol[i][0]);
        mu[i][1] = alpha_pol[itype][itype] * (efield[i][1] + efield_pol[i][1]);
        mu[i][2] = alpha_pol[itype][itype] * (efield[i][2] + efield_pol[i][2]);
      }
      
    }

    // communicate the updated induced dipoles mu with neighboring processors
    //   the updated mu will be used to compute the induced electrical field E_p in the next iteration

    comm->forward_comm(this);

    // check for convergence and break if not converged

    int converged = 1;
    int all_converged = 0;
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      if (mu[i][3] != 0.0) {
        if (fabs(mu_old[i][0] - mu[i][0]) > EPSILON ||
            fabs(mu_old[i][1] - mu[i][1]) > EPSILON ||
            fabs(mu_old[i][2] - mu[i][2]) > EPSILON ) {
          converged = 0;
          break;
        }
      }
    }

    MPI_Allreduce(&converged, &all_converged, 1, MPI_INT, MPI_MIN, world);
    if (all_converged) break;

    // store the current induced dipoles to mu_old for the next iteration
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      if (mu[i][3] != 0.0) {
        mu_old[i][0] = mu[i][0];
        mu_old[i][1] = mu[i][1];
        mu_old[i][2] = mu[i][2];
        mu_old[i][3] = mu[i][3];
      }
    }
  }

  // compute atom forces and polar energy (Eq. 9 in Paricaud et al.)
  // note: need to project the torques from charge-induced dipole interactions
  // to forces on atoms in each molecule

}

/* ---------------------------------------------------------------------- */

int PairLJCutCoulGaussLong::pack_forward_comm(int n, int *list, double *buf,
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

void PairLJCutCoulGaussLong::unpack_forward_comm(int n, int first, double *buf)
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

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::compute_induced_efield(double **efield_pol)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double r,r2inv,r6inv,efieldx,efieldy,efieldz;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double rsq;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    efield_pol[i][0] = efield_pol[i][1] = efield_pol[i][2] = 0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);

        // compute efield components from induced dipole on atom j

        // accumulate efield on atom i due to dipole on atom j

      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::allocate()
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
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(alpha_pol,n+1,n+1,"pair:alpha_pol");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::settings(int narg, char **arg)
{
 if (narg < 3 || narg > 4) error->all(FLERR,"Illegal pair_style command");

  coul_smooth = utils::numeric(FLERR,arg[0],false,lmp);
  alpha = utils::numeric(FLERR,arg[1],false,lmp);
  cut_lj_global = utils::numeric(FLERR,arg[2],false,lmp);
  if (narg == 3) cut_coul = cut_lj_global;
  else cut_coul = utils::numeric(FLERR,arg[3],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6)
    error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double epsilon_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);
  double alpha_pol_one = utils::numeric(FLERR,arg[4],false,lmp);

  double cut_lj_one = cut_lj_global;
  if (narg == 6) cut_lj_one = utils::numeric(FLERR,arg[5],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      alpha_pol[i][j] = alpha_pol_one;
      cut_lj[i][j] = cut_lj_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::init_style()
{
  if (!atom->q_flag || !atom->mu_flag)
    error->all(FLERR,"Pair dipole/cut requires atom attributes q and mu");

  neighbor->add_request(this, NeighConst::REQ_FULL);

  cut_coulsq = cut_coul * cut_coul;

  // calculation of smoothing coefficients c0_c-c5_c for coulomb smoothing

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

  // ensure use of KSpace long-range solver, set g_ewald

  if (force->kspace == nullptr)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutCoulGaussLong::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    alpha_pol[i][j] = mix_distance(alpha_pol[i][i],alpha_pol[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  // include TIP4P qdist in full cutoff, qdist = 0.0 if not TIP4P

  double cut = MAX(cut_lj[i][j],cut_coul+2.0*qdist);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut_lj[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  alpha_pol[j][i] = alpha_pol[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && MIN(cut_lj[i][j],cut_coul) < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut_lj[i][j]*cut_lj[i][j]*cut_lj[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::read_restart(FILE *fp)
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
          utils::sfread(FLERR,&cut_lj[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&coul_smooth,sizeof(double),1,fp);
  fwrite(&alpha,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
  fwrite(&ncoultablebits,sizeof(int),1,fp);
  fwrite(&tabinner,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_lj_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_coul,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&coul_smooth,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&alpha,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tail_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&ncoultablebits,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tabinner,sizeof(double),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&coul_smooth,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&alpha,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
  MPI_Bcast(&ncoultablebits,1,MPI_INT,0,world);
  MPI_Bcast(&tabinner,1,MPI_DOUBLE,0,world);
}


/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJCutCoulGaussLong::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJCutCoulGaussLong::single(int i, int j, int itype, int jtype,
                                 double rsq,
                                 double factor_coul, double factor_lj,
                                 double &fforce)
{
  double r2inv,r6inv,r,grij,expm2,t,erfc,prefactor;
  double forcecoul,forcelj,phicoul,philj;
  double rcu,rqu,sme,smf;
  double erfa,expa,arg,falpha,ealpha;
  double erf;

  r2inv = 1.0/rsq;
  if (rsq < cut_coulsq) {
    // long range - real space
    grij = g_ewald * r;
    expm2 = MathSpecial::expmsq(grij);
    erf = 1 - (MathSpecial::my_erfcx(grij) * expm2);

    // gaussian for 1/r alpha_ij contribution
    arg = alpha*r;
    expa = MathSpecial::expmsq(arg);
    erfa = 1 - (MathSpecial::my_erfcx(arg) * expa);

    prefactor = force->qqrd2e*atom->q[i]*atom->q[j]/r;
    falpha = erfa - EWALD_F*arg*expa;
    forcecoul = prefactor * (falpha - erf + EWALD_F*grij*expm2);
    if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor*falpha;

    // (q*q/r) * (gauss(alpha_ij) - gauss(alpha_long)
    ealpha = prefactor * (erfa-erf);
    // smoothing term - NOTE: ingnored in special_bonds correction
    // since likely rsmooth_sq_c >> d(special)
    if (rsq > rsmooth_sq_c) {
      rcu = r*rsq;
      rqu = rsq*rsq;
      sme = c5_c*rqu*r + c4_c*rqu + c3_c*rcu + c2_c*rsq + c1_c*r + c0_c;
      smf = 5.0*c5_c*rqu + 4.0*c4_c*rcu + 3.0*c3_c*rsq + 2.0*c2_c*r + c1_c;
      forcecoul = forcecoul*sme - ealpha*smf*r;
      ealpha *= sme;
    }
  } else forcecoul = 0.0;

  if (rsq < cut_ljsq[itype][jtype]) {
    r6inv = r2inv*r2inv*r2inv;
    forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  } else forcelj = 0.0;

  fforce = (forcecoul + factor_lj*forcelj) * r2inv;

  double eng = 0.0;
  if (rsq < cut_coulsq) {
    phicoul = prefactor*erfc;
    if (factor_coul < 1.0) phicoul -= (1.0-factor_coul)*prefactor;
    eng += phicoul;
  }

  if (rsq < cut_ljsq[itype][jtype]) {
    philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
      offset[itype][jtype];
    eng += factor_lj*philj;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairLJCutCoulGaussLong::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return nullptr;
}
