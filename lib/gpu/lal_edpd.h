/***************************************************************************
                                 edpd.h
                             -------------------
                            Trung Dac Nguyen (U Chicago)

  Class for acceleration of the dpd pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : September 2023
    email                : ndactrung@gmail.com
 ***************************************************************************/

#ifndef LAL_DPD_H
#define LAL_DPD_H

#include "lal_base_dpd.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class EDPD : public BaseDPD<numtyp, acctyp> {
 public:
  EDPD();
  ~EDPD();

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
  int init(const int ntypes, double **host_cutsq, double **host_a0,
           double **host_gamma, double **host_cut, double **host_power,
           double **host_kappa, double **host_powerT, double **host_cutT,
           double ***host_sc, double ***host_kc,
           double *host_special_lj, bool tstat_only,
           const int nlocal, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size, const double gpu_split,
           FILE *screen);

  /// Clear all host and device data
  /** \note This is called at the beginning of the init() routine **/
  void clear();

  /// Returns memory usage on device per atom
  int bytes_per_atom(const int max_nbors) const;

  /// Total host memory used by library for pair style
  double host_memory_usage() const;

  /// Update coeff if needed (tstat only)
  void update_coeff(int ntypes, double **host_a0, double **host_gamma,
                    double **host_sigma, double **host_cut);

  void cast_extra_data(double *host_T, double *host_cv);

  /// Get the Q array on the host
  void* get_Q() { return Q.host.begin(); }

  // --------------------------- TYPE DATA --------------------------

  /// coeff.x = a0, coeff.y = gamma, coeff.z = cut
  UCL_D_Vec<numtyp4> coeff;
  /// coeff2.x = power, coeff2.y = kappa, coeff2.z = powerT, coeff2.w = cutT
  UCL_D_Vec<numtyp4> coeff2;

  UCL_D_Vec<numtyp4> kc, sc;

  UCL_D_Vec<numtyp> cutsq;

  /// Special LJ values
  UCL_D_Vec<numtyp> sp_lj, sp_sqrt;

  /// If atom type constants fit in shared memory, use fast kernels
  bool shared_types;

  /// Number of atom types
  int _lj_types;

  /// Only used for thermostat
  int _tstat_only;

  /// Per-atom arrays
  UCL_Vector<acctyp,acctyp> Q;
  int _nmax, _max_q_size;

 private:
  bool _allocated;
  int loop(const int eflag, const int vflag);
};

}

#endif
