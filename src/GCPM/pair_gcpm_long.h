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
PairStyle(gcpm/long,PairGCPMLong);
// clang-format on
#else

#ifndef LMP_PAIR_GCPM_LONG_H
#define LMP_PAIR_GCPM_LONG_H

#include "pair_gcpm.h"

namespace LAMMPS_NS {

// Long-range (Ewald/PPPM) variant of pair gcpm. The exp-6 Buckingham dispersion
// and the self-consistent induced-dipole solver are inherited from PairGCPM.
// The Coulomb method differs from the reaction-field base: ALL electrostatic
// channels (charge-charge, charge-dipole, dipole-dipole) are Ewald-split.
// The real-space kernels are the smeared GCPM forms minus the point-multipole
// long-range parts; kspace_style pppm/dipole supplies the reciprocal sums and
// the per-iteration reciprocal fields that drive the induced-dipole SCF loop
// (via its compute_efield_from_charges/compute_efield_from_dipoles API).
// Intramolecular electrostatics are excluded by molecule ID inside the pair
// kernels (GCPM is intermolecular-only); the reaction field is not supported
// here (it was the stand-in for the reciprocal dipole interactions).

class PairGCPMLong : public PairGCPM {

 public:
  PairGCPMLong(class LAMMPS *);
  void compute(int, int) override;
  void init_style() override;
  void setup() override;

 protected:
  // Ewald-screened smeared Coulomb (real-space part; KSpace adds the reciprocal)
  void charge_charge(int, int) override;
  // induced dipole-dipole field (Ewald-screened real-space part)
  void compute_induced_efield(int neigh_half) override;
  // charge-dipole and dipole-dipole forces with the Ewald-screened kernels
  void polar(int, int, int neigh_half) override;

  // the KSpace solver; supplies g_ewald, the reciprocal-space fields for the
  // SCF loop, and (after this pair style) the reciprocal forces/torques
  class PPPMDipole *pppm_dipole;
};

}    // namespace LAMMPS_NS

#endif
#endif
