/***************************************************************************
                                   gcpm.h
                             -------------------
                              Trung Dac Nguyen

  Class for acceleration of the gcpm pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : 6/10/2026
    email                : ndactrung@gmail.com
 ***************************************************************************/

#ifndef LAL_GCPM_H
#define LAL_GCPM_H

#include "lal_base_charge.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class GCPM : public BaseCharge<numtyp, acctyp> {
 public:
  GCPM();
  ~GCPM();

  /// Clear any previous data and set up for a new LAMMPS run
  /** \param max_nbors initial number of rows in the neighbor matrix
    * \param cell_size cutoff + skin
    * \param gpu_split fraction of particles handled by device
    *
    * Returns:
    * -  0 if successful
    * - -1 if fix gpu not found
    * - -3 if there is an out of memory error
    * - -4 if the GPU library was not compiled for GPU
    * - -5 Double precision is not supported on card **/
  int init(const int ntypes, double **host_cutsq,
           double **host_buck1, double **host_buck2, double **host_buck3,
           double **host_cut_ljsq, double **host_offset, double **host_alpha_ij,
           double *host_special_lj, const int nlocal,
           const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           const double gpu_split, FILE *screen,
           const double host_cut_coulsq, double *host_special_coul,
           const double qqrd2e, const double g_ewald,
           const double rsmooth_sq,
           const double c0, const double c1, const double c2,
           const double c3, const double c4, const double c5);

  /// Clear all host and device data
  /** \note This is called at the beginning of the init() routine **/
  void clear();

  /// Returns memory usage on device per atom
  int bytes_per_atom(const int max_nbors) const;

  /// Total host memory used by library for pair style
  double host_memory_usage() const;

  // --------------------------- TYPE DATA --------------------------

  /// coeff1.x = buck1*buck2 (force A coeff), coeff1.y = buck2 (1/rho),
  /// coeff1.z = 6*buck3 (force C coeff), coeff1.w = cut_ljsq
  UCL_D_Vec<numtyp4> coeff1;
  /// coeff2.x = buck1 (energy A), coeff2.y = buck3 (energy C6),
  /// coeff2.z = offset, coeff2.w = alpha_ij (Gaussian Coulomb width)
  UCL_D_Vec<numtyp4> coeff2;
  /// cutsq (max of cut_ljsq and cut_coulsq per pair)
  UCL_D_Vec<numtyp> cutsq;
  /// Special LJ values [0-3] and Special Coul values [4-7]
  UCL_D_Vec<numtyp> sp_lj;

  /// If atom type constants fit in shared memory, use fast kernels
  bool shared_types;

  /// Number of atom types
  int _lj_types;

  numtyp _cut_coulsq, _qqrd2e, _g_ewald, _rsmooth_sq;
  acctyp _c0, _c1, _c2, _c3, _c4, _c5;

  /// Compute per-atom efield from charge-charge interactions.
  /// Returns a pinned host pointer via *efield_ptr.
  void compute_efield(void **efield_ptr);

  // Per-atom efield output buffers
  UCL_D_Vec<acctyp> dev_efield;
  UCL_H_Vec<acctyp> host_efield;
  UCL_Kernel k_efield;

 private:
  bool _allocated;
  int loop(const int eflag, const int vflag);
  void loop_efield();
};

}

#endif
