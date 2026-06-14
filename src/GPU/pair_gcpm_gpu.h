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

/* ----------------------------------------------------------------------
   Contributing author: Trung Nguyen (ndactrung@gmail.com)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(gcpm/gpu,PairGCPMGPU);
// clang-format on
#else

#ifndef LMP_PAIR_GCPM_GPU_H
#define LMP_PAIR_GCPM_GPU_H

#include "pair_gcpm.h"

namespace LAMMPS_NS {

class PairGCPMGPU : public PairGCPM {
 public:
  PairGCPMGPU(LAMMPS *lmp);
  ~PairGCPMGPU() override;
  void compute(int, int) override;
  void init_style() override;
  double memory_usage() override;

  enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
  void *efield_pinned;
  bool acc_float;
};

}    // namespace LAMMPS_NS
#endif
#endif
