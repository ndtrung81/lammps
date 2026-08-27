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

   Long-range (Ewald/PPPM) variant of pair gcpm. The exp-6 Buckingham dispersion
   and the self-consistent induced-dipole solver are inherited from PairGCPM.
   All three electrostatic channels (charge-charge, charge-dipole,
   dipole-dipole) are Ewald-split: the real-space kernels are the smeared GCPM
   forms minus the point-multipole long-range parts, and kspace_style
   pppm/dipole supplies the reciprocal sums. The induced-dipole SCF loop is
   driven by the total field: real-space screened terms plus the reciprocal
   fields obtained from PPPMDipole::compute_efield_from_charges (once per
   step) and compute_efield_from_dipoles (once per iteration), with the
   dipole Ewald self-field added back so a dipole does not polarize itself.
   Intramolecular electrostatics are excluded by molecule ID inside the pair
   kernels: the excluded pairs stay in the neighbor list and the real-space
   subtraction of the full smeared kernel cancels, pair by pair, the
   contribution the reciprocal sum adds for them.
------------------------------------------------------------------------- */

#include "pair_gcpm_long.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "kspace.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pppm_dipole.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace EwaldConst;
using MathSpecial::expmsq;
using MathSpecial::my_erfcx;

// reverse-comm selector for comm_mode (EFIELD = 0 -> efield, EFIELD_POL = 1 -> efield_pol)
enum {EFIELD, EFIELD_POL};
//#define GCPM_DEBUG

/* ---------------------------------------------------------------------- */

PairGCPMLong::PairGCPMLong(LAMMPS *lmp) : PairGCPM(lmp)
{
  // long-range Coulomb: real-space part is Ewald-screened and kspace_style
  // pppm/dipole supplies the reciprocal sum for all three channels
  // (the base class sets these to 0). dipoleflag makes pair_check() accept
  // the pppm/dipole solver.
  ewaldflag = pppmflag = dipoleflag = 1;
  pppm_dipole = nullptr;
}

/* ----------------------------------------------------------------------
   compute(): dispersion + Ewald-screened smeared Coulomb + the iterative polar
   solver, with the (optional) per-molecule reaction-field passes.
------------------------------------------------------------------------- */

void PairGCPMLong::compute(int eflag, int vflag)
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

    // add the reciprocal-space charge field at local atoms (charges do not
    // move during the SCF loop: once per step). The intramolecular and
    // special-pair contributions included by the k-space sum are cancelled in
    // real space by the exclusion convention in charge_charge().

    pppm_dipole->compute_efield_from_charges(efield);

    polar(eflag, vflag, 1);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   setup(): called from Force::setup() BEFORE the integrator's first pair
   compute. Verlet::setup()/Min::setup() only call kspace->setup() AFTER
   that first pair compute, but the SCF loop already needs the k-space
   coefficient tables (fk vectors, influence functions) during it. Setting
   the solver up here is safe: PPPMDipole::setup() is a pure recomputation
   and is repeated by the integrator right after.
------------------------------------------------------------------------- */

void PairGCPMLong::setup()
{
  PairGCPM::setup();
  if (enable_polar && pppm_dipole) pppm_dipole->setup();
}

/* ----------------------------------------------------------------------
   init_style for the long-range (Ewald/PPPM) form: requires kspace_style
   pppm/dipole (for the reciprocal sums AND the per-iteration SCF fields)
   and takes the splitting parameter g_ewald from it.
------------------------------------------------------------------------- */

void PairGCPMLong::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair gcpm/long requires atom attribute q");

  if (enable_polar && (!atom->mu_flag || !atom->torque_flag))
    error->all(FLERR,"Pair gcpm/long requires atom attributes mu and torque for "
                     "polarizable simulations");

  // intramolecular electrostatics are excluded via molecule IDs (GCPM is
  // intermolecular-only). The excluded pairs must STAY in the neighbor list
  // so the real-space kernels can cancel the reciprocal-space contributions
  // the k-space sum includes for them -- hence no neigh_modify exclude.

  if (!atom->molecule_flag)
    error->all(FLERR,"Pair gcpm/long requires atom attribute molecule");

  if (neighbor->nex_type || neighbor->nex_group || neighbor->nex_mol)
    error->all(FLERR,"Pair gcpm/long is incompatible with neigh_modify exclude: "
                     "removed pairs cannot cancel the k-space contributions; "
                     "intramolecular Coulomb interactions are already excluded "
                     "by the pair style via molecule IDs");

  // the reaction field was the stand-in for the reciprocal-space charge-dipole
  // and dipole-dipole interactions, now fully supplied by pppm/dipole

  if (enable_rf)
    error->all(FLERR,"Pair gcpm/long does not support the reaction field: "
                     "set eps_rf <= 0 (the long-range interactions are computed "
                     "by kspace_style pppm/dipole)");

  neighbor->add_request(this);

  cut_coulsq = cut_coul * cut_coul;

  if (force->kspace == nullptr)
    error->all(FLERR,"Pair style requires a KSpace style");
  pppm_dipole = dynamic_cast<PPPMDipole *>(force->kspace);
  if (enable_polar && (pppm_dipole == nullptr))
    error->all(FLERR,"Pair gcpm/long with polarization requires kspace_style "
                     "pppm/dipole");
  g_ewald = force->kspace->g_ewald;

  // pppm/dipole estimated g_ewald and the grid from the CURRENT dipole
  // moments. Induced-dipole seeds are usually near zero at the start of a
  // fresh run, so the dipole-channel contribution to the error model is
  // invisible and the estimate is effectively charge-only. Warn so users can
  // pin kspace_modify gewald (g_ewald*cut_coul >= ~3.3 recommended) and/or
  // mesh for production-quality accuracy. Runs continued from a restart or
  // data file with converged dipoles are estimated correctly and skip this.

  if (enable_polar) {
    double sums_local[2] = {0.0, 0.0};   // sum mu^2, number of dipole sites
    double **mu = atom->mu;
    for (int i = 0; i < atom->nlocal; i++) {
      if (mu[i][3] == 0.0) continue;
      sums_local[0] += mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2];
      sums_local[1] += 1.0;
    }
    double sums[2];
    MPI_Allreduce(sums_local,sums,2,MPI_DOUBLE,MPI_SUM,world);
    if ((sums[1] > 0.0) && (sums[0]/sums[1] < 0.05*0.05) && (comm->me == 0))
      error->warning(FLERR,"Pair gcpm/long: initial induced dipoles are near "
                     "zero, so the pppm/dipole accuracy estimate is "
                     "effectively charge-only; consider kspace_modify gewald "
                     "and/or mesh for production-quality accuracy");
  }
}

/* ----------------------------------------------------------------------
   charge-charge interactions between 2 Gaussian charge distributions
   (Eq. 4 in Paricaud et al.): the GCPM smeared term minus the Ewald real-space
   screening; the reciprocal part is supplied by the KSpace style.
------------------------------------------------------------------------- */

void PairGCPMLong::charge_charge(int eflag, int /*vflag*/)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,ecoul,fpair;
  double r,r2inv,forcecoul,factor_coul;
  double grij,expm2,prefactor;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double erfa,expa,arg,falpha,ealpha;
  double erf,efield_scalar;
  double rsq;

  ecoul = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  tagint *molecule = atom->molecule;
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

      // GCPM is intermolecular-only: exclude intramolecular Coulomb via the
      // molecule ID. The (1-factor)*full-smeared subtraction below (with
      // factor 0) cancels the contribution the reciprocal sum adds for this
      // pair.
      if ((molecule[i] != 0) && (molecule[i] == molecule[j])) factor_coul = 0.0;

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

        if (rsq < cut_coulsq) {
          grij = g_ewald * r;
          expm2 = expmsq(grij);
          erf = 1 - (my_erfcx(grij) * expm2);

          arg = alpha_ij[itype][jtype] * r;
          expa = expmsq(arg);
          erfa = 1 - (my_erfcx(arg) * expa);

          falpha = erfa - EWALD_F*arg*expa;

          // charge-independent field scalar (q[j] factored out so the Newton
          // partner can reuse the same value with q[i] in the reverse direction)
          double scale = qqrd2e / r;
          efield_scalar = scale * (falpha - erf + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) efield_scalar -= (1.0-factor_coul) * scale * falpha;

          if (has_force) {
            prefactor = qqrd2e*qtmp*q[j]/r;
            forcecoul = prefactor * (falpha - erf + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor*falpha;
            ealpha = prefactor * (erfa-erf);
          }

          efield_scalar *= r2inv;

        } else {
          forcecoul = 0.0;
          efield_scalar = 0.0;
        }

        if (has_force) {
          fpair = forcecoul * r2inv;

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
              ecoul = ealpha;
              if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor*erfa;
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
   compute induced electric field (Eq. 5) using current dipole estimates
     using Eqs. (6) and (7)
   Ewald form: the real-space tensor scalars are the smeared GCPM f and g
   MINUS the point-dipole long-range part (erf(g_ewald*r)-based); the
   reciprocal sum from pppm/dipole restores that part, so real + recip
   equals the full smeared T summed over all periodic images. The same
   splitting as the charge kernel in charge_charge().
   NOTE: no special_coul handling here -- assumes at most one polarizable
   site per molecule (no intramolecular dipole-dipole pairs), as in GCPM.
   The dipole self-image term of the reciprocal sum is corrected by the
   self-field subtraction in the SCF loop.
------------------------------------------------------------------------- */

void PairGCPMLong::compute_induced_efield(int half)
{
  int i,ii,j,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,r,r2inv,r3inv;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double sigmaM_ij,sigmaM_ij2;
  double _erf,expmsq_,rdivsigmaM,f,g;
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

      if (rsq < cut_coulsq) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);
        r3inv = 1.0/rsq/r;

        sigmaM_ij = sigmaM[itype][jtype];
        sigmaM_ij2 = sigmaM_ij * sigmaM_ij;

        // Eq. (7): scalars f and g for the T_ij tensor

        _erf = erf(r / (2.0 * sigmaM_ij));
        expmsq_ = exp(-r * r / 4.0 / sigmaM_ij2);
        rdivsigmaM = r / MathConst::MY_PIS / sigmaM_ij;
        f = _erf - (rdivsigmaM + rdivsigmaM * rsq / sigmaM_ij2 / 6.0) * expmsq_;
        g = _erf - rdivsigmaM * expmsq_;

        // subtract the point-dipole Ewald long-range part:
        //   f -> f - erf(Gr) + EWALD_F*Gr*(1 + 2(Gr)^2/3)*exp(-(Gr)^2)
        //   g -> g - erf(Gr) + EWALD_F*Gr*exp(-(Gr)^2)
        // (equivalent to the b1/b2 real-space kernels of the point-dipole
        // Ewald in pair lj/cut/dipole/long, generalized to smeared dipoles)

        double grij = g_ewald * r;
        double expm2 = expmsq(grij);
        double erf_g = 1.0 - my_erfcx(grij) * expm2;
        double gr2 = grij * grij;
        f += -erf_g + EWALD_F*grij*(1.0 + 2.0*gr2/3.0)*expm2;
        g += -erf_g + EWALD_F*grij*expm2;

        // Eq. (6): T_ij = 3f*r^-5*r_ij*r_ij - g*r^-3*I
        // T_ij is symmetric (T_ij[a][b] = T_ij[b][a]) and T_ij = T_ji

        f *= 3.0 * r2inv;
        Tij[0][0] = r3inv * (f * delx * delx - g);
        Tij[0][1] = r3inv * f * delx * dely;
        Tij[0][2] = r3inv * f * delx * delz;
        Tij[1][1] = r3inv * (f * dely * dely - g);
        Tij[1][2] = r3inv * f * dely * delz;
        Tij[2][2] = r3inv * (f * delz * delz - g);

        // E_p_i += T_ij . mu_j

        efield_pol[i][0] += qqrd2e * (Tij[0][0]*mu[j][0] + Tij[0][1]*mu[j][1] + Tij[0][2]*mu[j][2]);
        efield_pol[i][1] += qqrd2e * (Tij[0][1]*mu[j][0] + Tij[1][1]*mu[j][1] + Tij[1][2]*mu[j][2]);
        efield_pol[i][2] += qqrd2e * (Tij[0][2]*mu[j][0] + Tij[1][2]*mu[j][1] + Tij[2][2]*mu[j][2]);

        // Newton partner: E_p_j += T_ji . mu_i = T_ij . mu_i (T symmetric)

        if ((newton_pair || j < nlocal) && half) {
          efield_pol[j][0] += qqrd2e * (Tij[0][0]*mu[i][0] + Tij[0][1]*mu[i][1] + Tij[0][2]*mu[i][2]);
          efield_pol[j][1] += qqrd2e * (Tij[0][1]*mu[i][0] + Tij[1][1]*mu[i][1] + Tij[1][2]*mu[i][2]);
          efield_pol[j][2] += qqrd2e * (Tij[0][2]*mu[i][0] + Tij[1][2]*mu[i][1] + Tij[2][2]*mu[i][2]);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   polar interactions: iterative solver for induced dipoles (Eqs. 3 and 5)
------------------------------------------------------------------------- */

void PairGCPMLong::polar(int eflag, int vflag, int neigh_half)
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
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  // Ewald self-field: the reciprocal sum includes each dipole's interaction
  // with its own Gaussian screening image, E_self = -eself*mu_i with
  // eself = (4/3)*g_ewald^3/sqrt(pi)*qqrd2e; a dipole must not polarize
  // itself, so eself*mu_i is added back to the SCF field every iteration.

  const double eself =
    4.0*g_ewald*g_ewald*g_ewald/(3.0*MathConst::MY_PIS)*qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // on the very first call: estimate mu from efield only (E_p assumed to be 0)
  // on subsequent time step: start from the previous timestep's converged dipoles

  if (first_polar) {
    for (i = 0; i < nlocal; i++) {
      itype = type[i];
      if (mu[i][3] != 0.0) {
        mu[i][0] = alpha_pol[itype][itype] * efield[i][0] / qqrd2e;
        mu[i][1] = alpha_pol[itype][itype] * efield[i][1] / qqrd2e;
        mu[i][2] = alpha_pol[itype][itype] * efield[i][2] / qqrd2e;
      }
    }
    first_polar = 0;
  }

  // seed mu_old from the starting guess so the convergence check is correct on iter 1
  for (i = 0; i < nlocal; i++) {
    if (mu[i][3] != 0.0) {
      mu_old[i][0] = mu[i][0];
      mu_old[i][1] = mu[i][1];
      mu_old[i][2] = mu[i][2];
    }
  }

  int iter, converged_run = 0;
  for (iter = 0; iter < maxiter; iter++) {

    // Eq. (5): compute E_p from current dipole estimates, then update dipoles from total field

    compute_induced_efield(neigh_half);

    // communicate and sum per-atom induced efield

    if (newton_pair && neigh_half == 1) {
      comm_mode = EFIELD_POL;
      comm->reverse_comm(this);
    }

    // reciprocal-space dipole field from the current dipole estimates
    // (collective call: every rank participates in the FFTs). Recomputed
    // every iteration because it depends on the evolving induced dipoles.
    // Then remove the self-image contribution (see eself above).

    pppm_dipole->compute_efield_from_dipoles(efield_pol);

    for (i = 0; i < nlocal; i++) {
      if (mu[i][3] == 0.0) continue;
      efield_pol[i][0] += eself*mu[i][0];
      efield_pol[i][1] += eself*mu[i][1];
      efield_pol[i][2] += eself*mu[i][2];
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
    if (all_converged) {
      #ifdef GCPM_DEBUG
      if (comm->me == 0) printf("iter = %d: converged\n", iter+1);
      #endif
      converged_run = 1;
      break;
    }

    for (i = 0; i < nlocal; i++) {
      if (mu[i][3] != 0.0) {
        mu_old[i][0] = mu[i][0];
        mu_old[i][1] = mu[i][1];
        mu_old[i][2] = mu[i][2];
      }
    }
  }

  // record solver convergence statistics for finish() (inherited from PairGCPM)
  record_polar_iters(converged_run ? iter + 1 : maxiter, converged_run);

  // After convergence: forces and energy from charge-induced-dipole interaction.
  //
  // Energy bookkeeping (differs from the base RF style): the Eq. (9) shortcut
  // U_pol = -1/2 sum p_i.E_q_i cannot be used here, because E_q now contains
  // the reciprocal-space field while kspace pppm/dipole tallies its own q-mu
  // and mu-mu reciprocal energies (+ dipole self-energy) -- Eq. (9) would
  // double-count them. Instead the pair tallies its real-space pieces
  // explicitly: U_qp_real and U_pp_real per pair (in the loops below, using
  // the same exclusion-corrected kernels as the forces) plus the induction
  // self-energy + sum p_i^2/(2 alpha_i) (the reversible work to create the
  // induced dipoles). At self-consistency
  //   U_qp(real+recip) + U_pp(real+recip+self) + U_ind = -1/2 sum p.E_q,
  // so the grand total (pair + kspace) still equals Eq. (9). Validated by
  // g_ewald-independence of the total energy and by FD F = -dU/dx.

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    // induction self-energy + p_i^2/(2 alpha_i), tallied once per dipole atom
    if (mu[i][3] != 0.0 && eflag) {
      int it = type[i];
      if (alpha_pol[it][it] > 0.0) {
        ecoul = 0.5 * qqrd2e *
          (mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2]) /
          alpha_pol[it][it];
        if (eflag_global) eng_coul += ecoul;
        if (eflag_atom) eatom[i] += ecoul;
      }
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

      // exclude intramolecular charge-dipole interactions by molecule ID
      // (same convention as charge_charge() and the efield accumulation)
      if ((molecule[i] != 0) && (molecule[i] == molecule[j])) factor_coul = 0.0;

      // a charge-induced-dipole pair contributes a force to BOTH partners.
      // Two independent interactions can exist for a pair (each M site carries
      // both a charge and an induced dipole):
      //   A: dipole on i with charge on j      B: dipole on j with charge on i
      // Both partners must receive their force exactly once, otherwise the net
      // force is non-zero (Newton's 3rd law) and the half-list (CPU) and
      // full-list (GPU) paths disagree. With a half list (neigh_half==1) the
      // partner force is applied via f[j]; with a full list (neigh_half==0)
      // each center accumulates only its own force (the partner is handled when
      // it is the center).

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

        double aij = alpha_ij[itype][jtype];
        double grij = g_ewald * r;
        double expm2 = expmsq(grij);
        double erf_g = 1.0 - my_erfcx(grij) * expm2;
        double aijr = aij * r;
        double expa = expmsq(aijr);
        double erfa = 1.0 - my_erfcx(aijr) * expa;
        double falpha = erfa - EWALD_F*aijr*expa;
        double Phi = falpha - erf_g + EWALD_F*grij*expm2;
        double dPhi_dr = 2.0*EWALD_F*rsq*(aij*aij*aij*expa - g_ewald*g_ewald*g_ewald*expm2);

        // excluded (special) pairs: subtract the FULL smeared kernel so that,
        // combined with the point kernel restored by the reciprocal sum, the
        // pair contributes factor_coul * (full smeared interaction) -- the
        // same convention as the q-q loop and the efield accumulation in
        // charge_charge(). Do NOT scale the screened kernel by factor_coul.

        if (factor_coul < 1.0) {
          Phi -= (1.0-factor_coul)*falpha;
          dPhi_dr -= (1.0-factor_coul)*2.0*EWALD_F*aij*aij*aij*rsq*expa;
        }
        double dcoeff = 3.0*Phi - r*dPhi_dr;

        // Interaction A: dipole i with charge j; del = x_i - x_j (charge->dipole)
        if (doA) {
          double pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          double fqj = qqrd2e * q[j];
          double pre1 = fqj*r5inv * dcoeff * pidotr;
          double pre2 = fqj*r3inv * Phi;

          double fcx = pre2*mu[i][0] - pre1*delx;   // force on dipole i
          double fcy = pre2*mu[i][1] - pre1*dely;
          double fcz = pre2*mu[i][2] - pre1*delz;

          f[i][0] += fcx;
          f[i][1] += fcy;
          f[i][2] += fcz;

          torque[i][0] += pre2 * (mu[i][1]*delz - mu[i][2]*dely);
          torque[i][1] += pre2 * (mu[i][2]*delx - mu[i][0]*delz);
          torque[i][2] += pre2 * (mu[i][0]*dely - mu[i][1]*delx);

          if ((newton_pair || j < nlocal) && neigh_half == 1) {
            f[j][0] -= fcx;   // reaction on charge j
            f[j][1] -= fcy;
            f[j][2] -= fcz;
          }

          vtally_force(i, j, neigh_half, fcx, fcy, fcz, delx, dely, delz);

          // real-space charge-dipole energy: U = -p_i . E^real_{q_j -> i}
          // (pre2 already contains the exclusion-corrected kernel)
          if (eflag) {
            ecoul = -pre2 * pidotr;
            if (neigh_half == 0) ecoul *= 0.5;
            ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,0.0,0.0,0.0,0.0);
          }
        }

        // Interaction B: dipole j with charge i; the dipole-charge vector is
        // x_j - x_i = -del, so mu[j].(x_j-x_i) = -(mu[j].del)
        if (doB) {
          double pjdotr = -(mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz);
          double fqi = qqrd2e * qtmp;
          double pre1 = fqi*r5inv * dcoeff * pjdotr;
          double pre2 = fqi*r3inv * Phi;

          // force on dipole j = pre2*mu[j] - pre1*(x_j-x_i) = pre2*mu[j] + pre1*del
          double fdx = pre2*mu[j][0] + pre1*delx;
          double fdy = pre2*mu[j][1] + pre1*dely;
          double fdz = pre2*mu[j][2] + pre1*delz;

          f[i][0] -= fdx;   // reaction on charge i = -(force on dipole j)
          f[i][1] -= fdy;
          f[i][2] -= fdz;

          if ((newton_pair || j < nlocal) && neigh_half == 1) {
            f[j][0] += fdx;   // force on dipole j (partner)
            f[j][1] += fdy;
            f[j][2] += fdz;
            // torque on dipole j: tau = mu[j] x E, with E along (x_j-x_i) = -del
            torque[j][0] -= pre2 * (mu[j][1]*delz - mu[j][2]*dely);
            torque[j][1] -= pre2 * (mu[j][2]*delx - mu[j][0]*delz);
            torque[j][2] -= pre2 * (mu[j][0]*dely - mu[j][1]*delx);
          }

          vtally_force(i, j, neigh_half, -fdx, -fdy, -fdz, delx, dely, delz);

          // real-space charge-dipole energy: U = -p_j . E^real_{q_i -> j};
          // with pjdotr = mu_j.(x_j - x_i) this is -pre2*pjdotr
          if (eflag) {
            ecoul = -pre2 * pjdotr;
            if (neigh_half == 0) ecoul *= 0.5;
            ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,0.0,0.0,0.0,0.0);
          }
        }
      }
    }
  }

  // dipole-dipole polarization force and torque: gradient (at fixed converged
  // dipoles) of the dipole-dipole term of the full polarization energy, Eq. (8):
  //   U_dd = -1/2 sum_i p_i . E_p_i = -sum_{i<j} p_i . T_ij . p_j
  // The real-space part of U_dd is tallied here per pair (see the energy
  // bookkeeping note above); the reciprocal part, force and torque come from
  // pppm/dipole. Uses the same Ewald-screened scalars and the same cutoff
  // cut_coulsq as compute_induced_efield() so that energy, force and induced
  // field stay consistent.

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

      if (rsq < cut_coulsq) {
        r2inv = 1.0/rsq;
        double r = sqrt(rsq);
        double r3inv = r2inv/r;
        double r5inv = r3inv*r2inv;

        double s  = sigmaM[itype][jtype];
        double s2 = s*s, s3 = s2*s, s5 = s3*s2;
        double expmsq_ = exp(-rsq/(4.0*s2));
        double _erf = erf(r/(2.0*s));
        double rds = r/(MathConst::MY_PIS*s);            // r/(sqrt(pi)*s)

        // Eq. (7) scalars f, g and their radial derivatives f'(r), g'(r)
        double f_s = _erf - (rds + rds*rsq/(6.0*s2))*expmsq_;
        double g_s = _erf - rds*expmsq_;
        double df  = rsq*rsq/(12.0*MathConst::MY_PIS*s5)*expmsq_;     // f'(r)
        double dg  = rsq/(2.0*MathConst::MY_PIS*s3)*expmsq_;          // g'(r)

        // subtract the point-dipole Ewald long-range part (identical splitting
        // to compute_induced_efield()) and its radial derivatives

        double grij = g_ewald * r;
        double expm2 = expmsq(grij);
        double erf_g = 1.0 - my_erfcx(grij) * expm2;
        double gr2 = grij * grij;
        double gcube = g_ewald*g_ewald*g_ewald;
        f_s += -erf_g + EWALD_F*grij*(1.0 + 2.0*gr2/3.0)*expm2;
        g_s += -erf_g + EWALD_F*grij*expm2;
        df  -= (4.0/3.0)*EWALD_F*gcube*gr2*rsq*expm2;
        dg  -= 2.0*EWALD_F*gcube*rsq*expm2;

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
          torque[j][0] += mu[j][1]*Ejz - mu[j][2]*Ejy;
          torque[j][1] += mu[j][2]*Ejx - mu[j][0]*Ejz;
          torque[j][2] += mu[j][0]*Ejy - mu[j][1]*Ejx;
        }

        vtally_force(i, j, neigh_half, fdx, fdy, fdz, delx, dely, delz);

        // real-space dipole-dipole energy: U = -p_i . T^real_ij . p_j
        if (eflag) {
          ecoul = -(Aq*pir*pjr - qqrd2e*g_s*r3inv*pij);
          if (neigh_half == 0) ecoul *= 0.5;
          ev_tally(i,j,nlocal,newton_pair,0.0,ecoul,0.0,0.0,0.0,0.0);
        }
      }
    }
  }
}
