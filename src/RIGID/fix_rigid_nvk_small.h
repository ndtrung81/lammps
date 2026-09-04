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

#ifdef FIX_CLASS
// clang-format off
FixStyle(rigid/nvk/small,FixRigidNVKSmall);
// clang-format on
#else

#ifndef LMP_FIX_RIGID_NVK_SMALL_H
#define LMP_FIX_RIGID_NVK_SMALL_H

#include "fix_rigid_nh_small.h"

namespace LAMMPS_NS {

class FixRigidNVKSmall : public FixRigidNHSmall {
 public:
  FixRigidNVKSmall(class LAMMPS *, int, char **);
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

 protected:
  int ktarget_flag;       // 1 if a target temperature was given via the temp keyword
  double ktarget_temp;    // that temperature
  double k_target;        // total kinetic energy held constant by the constraint

  void compute_scale_factors(double &, double &);
};

}    // namespace LAMMPS_NS

#endif
#endif
