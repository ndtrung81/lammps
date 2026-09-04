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
   Gaussian isokinetic (Evans-Hoover) thermostat for rigid bodies.

   The combined translational and rotational kinetic energy of all bodies is
   held exactly constant by a single deterministic friction coefficient

     alpha = ( sum_i F_i.v_i + sum_i tau_i.omega_i )
             / ( sum_i m_i v_i^2 + sum_i omega_i.I_i.omega_i )

   acting on both the center-of-mass momenta and the angular momenta,

     dp_i/dt = F_i - alpha p_i,   dL_i/dt = tau_i - alpha L_i .

   One friction for all degrees of freedom (rather than separate
   translational and rotational baths) is what makes this a single Gaussian
   constraint on the total kinetic energy.  The gyroscopic terms of the Euler
   equations drop out of the numerator identically, so they need not appear in
   alpha.  The half-step momentum update uses the exact propagator of
   Minary et al., as in fix nvk.

   references: Evans and Morriss, Comput. Phys. Rep. 1, 297 (1984)
               Minary et al., J. Chem. Phys. 118, 2510 (2003)
               Kamberaj et al., J. Chem. Phys. 122, 224114 (2005)
------------------------------------------------------------------------- */

#include "fix_rigid_nvk_small.h"

#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "rigid_const.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace RigidConst;

static constexpr double INERTIA_EPSILON = 1.0e-7;

/* ----------------------------------------------------------------------
   the temp keyword of this style sets the kinetic energy the constraint
   holds fixed.  The rigid-body base classes parse the keywords of the
   command and know nothing about it, so it is removed from the argument
   list they are handed and parsed here instead.

   keywords start after the body style and its arguments: fix rigid/small
   body styles are molecule (no arguments) and custom (one argument)
------------------------------------------------------------------------- */

static int first_keyword(int narg, char **arg)
{
  if ((narg > 3) && (strcmp(arg[3],"custom") == 0)) return 5;
  return 4;
}

/* ----------------------------------------------------------------------
   is arg[iarg] the temp keyword of this style, or the value of one of the
   base class keywords that take an arbitrary name?  a group, file, molecule
   template or fix called "temp" is legal, if unkind
------------------------------------------------------------------------- */

static bool is_temp_keyword(int iarg, char **arg)
{
  if (strcmp(arg[iarg],"temp") != 0) return false;

  const char *prev = arg[iarg-1];
  if ((strcmp(prev,"infile") == 0) || (strcmp(prev,"mol") == 0) ||
      (strcmp(prev,"dilate") == 0) || (strcmp(prev,"gravity") == 0)) return false;

  return true;
}

/* ----------------------------------------------------------------------
   the argument list the base classes parse, with the temp keyword and its
   value taken out.  the vector is a temporary in the mem-initializer of the
   constructor below, so it lives until the base class constructor that is
   handed its data has returned
------------------------------------------------------------------------- */

static std::vector<char *> base_args(int narg, char **arg)
{
  std::vector<char *> args;
  const int first = first_keyword(narg,arg);

  for (int iarg = 0; iarg < narg; iarg++) {
    if ((iarg >= first) && is_temp_keyword(iarg,arg)) {
      iarg++;    // also drop the temperature that follows
      continue;
    }
    args.push_back(arg[iarg]);
  }

  return args;
}

/* ---------------------------------------------------------------------- */

static int base_narg(int narg, char **arg)
{
  return (int) base_args(narg,arg).size();
}

/* ---------------------------------------------------------------------- */

FixRigidNVKSmall::FixRigidNVKSmall(LAMMPS *lmp, int narg, char **arg) :
  FixRigidNHSmall(lmp, base_narg(narg,arg), base_args(narg,arg).data()),
  ktarget_flag(0), ktarget_temp(0.0), k_target(0.0)
{
  // parse the temp keyword the base classes were not shown

  const int first = first_keyword(narg,arg);

  for (int iarg = first; iarg < narg; iarg++) {
    if (!is_temp_keyword(iarg,arg)) continue;

    if (iarg+2 > narg)
      utils::missing_cmd_args(FLERR, fmt::format("fix {} temp", style), error);
    if (ktarget_flag)
      error->all(FLERR, iarg, "Fix {} temp keyword used more than once", style);

    ktarget_flag = 1;
    ktarget_temp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    if (ktarget_temp <= 0.0)
      error->all(FLERR, iarg+1, "Fix {} temp value must be > 0.0", style);
    iarg++;
  }

  if (langflag)
    error->all(FLERR,"Must not combine Langevin with the Gaussian isokinetic "
               "thermostat of fix {}", style);
}

/* ----------------------------------------------------------------------
   set the kinetic energy the constraint holds fixed
   with the temp keyword: from the target temperature and the body degrees of
   freedom, rescaling the current velocities to match it exactly
   without it: whatever the bodies carry when the run starts
------------------------------------------------------------------------- */

void FixRigidNVKSmall::setup(int vflag)
{
  FixRigidNHSmall::setup(vflag);

  // total (translational + rotational) kinetic energy of the bodies
  // compute_dof() was called by FixRigidNHSmall::setup(), so nf_t/nf_r are current

  double ke = 0.0;
  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];
    ke += b->mass*(b->vcm[0]*b->vcm[0] + b->vcm[1]*b->vcm[1] + b->vcm[2]*b->vcm[2]);
    ke += b->angmom[0]*b->omega[0] + b->angmom[1]*b->omega[1] +
      b->angmom[2]*b->omega[2];
  }

  double keall;
  MPI_Allreduce(&ke,&keall,1,MPI_DOUBLE,MPI_SUM,world);
  keall *= 0.5 * mvv2e;

  int nf = nf_t + nf_r;
  if (nf == 0) error->all(FLERR,"Fix {} has no degrees of freedom to constrain", style);

  if (ktarget_flag) {
    k_target = 0.5 * nf * boltz * ktarget_temp;

    if (keall <= 0.0)
      error->all(FLERR,"Fix {} temp requires non-zero initial body velocities", style);

    // rescale to land exactly on the target kinetic energy
    // conjqm is linear in angmom, so the same factor applies to it

    const double scale = sqrt(k_target/keall);
    for (int ibody = 0; ibody < nlocal_body; ibody++) {
      Body *b = &body[ibody];
      for (int k = 0; k < 3; k++) {
        b->vcm[k] *= scale;
        b->angmom[k] *= scale;
        b->omega[k] *= scale;
      }
      for (int k = 0; k < 4; k++) b->conjqm[k] *= scale;
    }

    // push the rescaled body velocities back down to the atoms

    commflag = FINAL;
    comm->forward_comm(this,10);
    set_v();

  } else {
    k_target = keall;
  }

  if (comm->me == 0)
    utils::logmesg(lmp,"Fix {} holding the kinetic energy of {} rigid-body degrees "
                   "of freedom at {:.8g} (T = {:.8g})\n", id, nf, k_target,
                   2.0*k_target/(nf*boltz));
}

/* ----------------------------------------------------------------------
   Gaussian isokinetic half-step propagator, Minary et al. Eqs. 4.12-4.13
   s -> dtq and sdot -> 1 when the friction vanishes, which recovers the
   plain rigid-body velocity Verlet half kick
------------------------------------------------------------------------- */

void FixRigidNVKSmall::compute_scale_factors(double &s, double &sdot)
{
  double tbody[3];
  double sums[2],sumall[2];

  sums[0] = sums[1] = 0.0;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];

    // power delivered to this body by its net force and torque

    sums[0] += b->fcm[0]*b->vcm[0] + b->fcm[1]*b->vcm[1] + b->fcm[2]*b->vcm[2] +
      b->torque[0]*b->omega[0] + b->torque[1]*b->omega[1] + b->torque[2]*b->omega[2];

    // |F|^2/m + sum_k tau_k^2/I_k, with the torque in body coordinates
    // where the inertia tensor is diagonal
    // a vanishing principal moment carries no rotational degree of freedom

    MathExtra::transpose_matvec(b->ex_space,b->ey_space,b->ez_space,b->torque,tbody);

    double bsum = (b->fcm[0]*b->fcm[0] + b->fcm[1]*b->fcm[1] +
                   b->fcm[2]*b->fcm[2]) / b->mass;
    for (int k = 0; k < 3; k++)
      if (fabs(b->inertia[k]) > INERTIA_EPSILON) bsum += tbody[k]*tbody[k] / b->inertia[k];
    sums[1] += bsum;
  }

  MPI_Allreduce(sums,sumall,2,MPI_DOUBLE,MPI_SUM,world);

  // a has units of inverse time, b of inverse time squared
  // k_target already carries the mvv2e conversion, so dividing by it once
  // more turns the force-squared term into an acceleration-squared term

  const double a = sumall[0] / (2.0*k_target);
  const double b = sumall[1] / (2.0*k_target*mvv2e);

  if (b <= 0.0) {
    s = dtq;
    sdot = 1.0;
    return;
  }

  const double sqtb = sqrt(b);
  const double x = dtq * sqtb;
  s = a/b * (cosh(x) - 1.0) + sinh(x)/sqtb;
  sdot = a/sqtb * sinh(x) + cosh(x);
}

/* ----------------------------------------------------------------------
   preforce velocity Verlet integration, step references as in Kamberaj et al.
   with the plain half kicks replaced by the isokinetic propagator
------------------------------------------------------------------------- */

void FixRigidNVKSmall::initial_integrate(int vflag)
{
  double s,sdot,mbody[3],tbody[3],fquat[4];

  compute_scale_factors(s,sdot);
  const double sf = s * force->ftm2v;
  const double inv_sdot = 1.0 / sdot;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];

    // step 1.1 - isokinetic update of vcm by 1/2 step

    const double sfm = sf / b->mass;
    b->vcm[0] = (b->vcm[0] + sfm * b->fcm[0]) * inv_sdot;
    b->vcm[1] = (b->vcm[1] + sfm * b->fcm[1]) * inv_sdot;
    b->vcm[2] = (b->vcm[2] + sfm * b->fcm[2]) * inv_sdot;

    // step 1.2 - update xcm by full step

    b->xcm[0] += dtv * b->vcm[0];
    b->xcm[1] += dtv * b->vcm[1];
    b->xcm[2] += dtv * b->vcm[2];

    // step 1.3 - isokinetic update of the quaternion momentum by 1/2 step

    MathExtra::transpose_matvec(b->ex_space,b->ey_space,b->ez_space,
                                b->torque,tbody);
    MathExtra::quatvec(b->quat,tbody,fquat);

    b->conjqm[0] = (b->conjqm[0] + 2.0*sf * fquat[0]) * inv_sdot;
    b->conjqm[1] = (b->conjqm[1] + 2.0*sf * fquat[1]) * inv_sdot;
    b->conjqm[2] = (b->conjqm[2] + 2.0*sf * fquat[2]) * inv_sdot;
    b->conjqm[3] = (b->conjqm[3] + 2.0*sf * fquat[3]) * inv_sdot;

    // step 1.4 to 1.13 - use no_squish rotate to update p and q

    MathExtra::no_squish_rotate(3,b->conjqm,b->quat,b->inertia,dtq);
    MathExtra::no_squish_rotate(2,b->conjqm,b->quat,b->inertia,dtq);
    MathExtra::no_squish_rotate(1,b->conjqm,b->quat,b->inertia,dtv);
    MathExtra::no_squish_rotate(2,b->conjqm,b->quat,b->inertia,dtq);
    MathExtra::no_squish_rotate(3,b->conjqm,b->quat,b->inertia,dtq);

    // update exyz_space
    // transform p back to angmom
    // update angular velocity

    MathExtra::q_to_exyz(b->quat,b->ex_space,b->ey_space,b->ez_space);
    MathExtra::invquatvec(b->quat,b->conjqm,mbody);
    MathExtra::matvec(b->ex_space,b->ey_space,b->ez_space,mbody,b->angmom);

    b->angmom[0] *= 0.5;
    b->angmom[1] *= 0.5;
    b->angmom[2] *= 0.5;

    MathExtra::angmom_to_omega(b->angmom,b->ex_space,b->ey_space,
                               b->ez_space,b->inertia,b->omega);
  }

  // forward communicate updated info of all bodies

  commflag = INITIAL;
  comm->forward_comm(this,29);

  // virial setup before call to set_xv

  v_init(vflag);

  // set coords/orient and velocity/rotation of atoms in rigid bodies
  // from quaternion and omega

  set_xv();
}

/* ---------------------------------------------------------------------- */

void FixRigidNVKSmall::final_integrate()
{
  double s,sdot,mbody[3],tbody[3],fquat[4];

  // late calculation of forces and torques (if requested)

  if (!earlyflag) compute_forces_and_torques();

  compute_scale_factors(s,sdot);
  const double sf = s * force->ftm2v;
  const double inv_sdot = 1.0 / sdot;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];

    // isokinetic update of vcm by 1/2 step

    const double sfm = sf / b->mass;
    b->vcm[0] = (b->vcm[0] + sfm * b->fcm[0]) * inv_sdot;
    b->vcm[1] = (b->vcm[1] + sfm * b->fcm[1]) * inv_sdot;
    b->vcm[2] = (b->vcm[2] + sfm * b->fcm[2]) * inv_sdot;

    // isokinetic update of conjqm, then transform to angmom
    // virial is already setup from initial_integrate

    MathExtra::transpose_matvec(b->ex_space,b->ey_space,b->ez_space,
                                b->torque,tbody);
    MathExtra::quatvec(b->quat,tbody,fquat);

    b->conjqm[0] = (b->conjqm[0] + 2.0*sf * fquat[0]) * inv_sdot;
    b->conjqm[1] = (b->conjqm[1] + 2.0*sf * fquat[1]) * inv_sdot;
    b->conjqm[2] = (b->conjqm[2] + 2.0*sf * fquat[2]) * inv_sdot;
    b->conjqm[3] = (b->conjqm[3] + 2.0*sf * fquat[3]) * inv_sdot;

    MathExtra::invquatvec(b->quat,b->conjqm,mbody);
    MathExtra::matvec(b->ex_space,b->ey_space,b->ez_space,mbody,b->angmom);

    b->angmom[0] *= 0.5;
    b->angmom[1] *= 0.5;
    b->angmom[2] *= 0.5;

    MathExtra::angmom_to_omega(b->angmom,b->ex_space,b->ey_space,
                               b->ez_space,b->inertia,b->omega);
  }

  // forward communicate updated info of all bodies

  commflag = FINAL;
  comm->forward_comm(this,10);

  // set velocity/rotation of atoms in rigid bodies
  // virial is already setup from initial_integrate

  set_v();
}
