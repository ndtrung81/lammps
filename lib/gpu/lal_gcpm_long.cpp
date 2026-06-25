/***************************************************************************
                                gcpm_long.cpp
                             -------------------
                              Trung Dac Nguyen

  Class for acceleration of the gcpm pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : 6/10/2026
    email                : ndactrung@gmail.com
 ***************************************************************************/

#ifdef USE_OPENCL
#include "gcpm_long_cl.h"
#elif defined(USE_CUDART)
const char *gcpm_long=0;
#else
#include "gcpm_long_cubin.h"
#endif

#include "lal_gcpm_long.h"
#include <cassert>
#include <vector>
namespace LAMMPS_AL {
#define GCPMLongT GCPMLong<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
GCPMLongT::GCPMLong() : BaseCharge<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
GCPMLongT::~GCPMLong() {
  clear();
}

template <class numtyp, class acctyp>
int GCPMLongT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int GCPMLongT::init(const int ntypes, double **host_cutsq,
                double **host_buck1, double **host_buck2, double **host_buck3,
                double **host_cut_ljsq, double **host_offset, double **host_alpha_ij,
                double *host_special_lj, const int nlocal,
                const int nall, const int max_nbors,
                const int maxspecial, const double cell_size,
                const double gpu_split, FILE *_screen,
                const double host_cut_coulsq, double *host_special_coul,
                const double qqrd2e, const double g_ewald) {
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                            _screen,gcpm_long,"k_gcpm_long");
  if (success!=0)
    return success;

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  shared_types=false;
  int max_shared_types=this->device->max_shared_types();
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  _lj_types=lj_types;

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<lj_types*lj_types; i++)
    host_write[i]=0.0;

  // Pre-compute derived arrays for coeff1:
  //   coeff1 = {buck1*buck2, buck2, 6*buck3, cut_ljsq}
  // ntypes is atom->ntypes+1 from the caller; host arrays are valid for [0, ntypes-1].
  int n = ntypes;
  std::vector<double> buf_b1b2(n*n), buf_b2(n*n), buf_6b3(n*n);
  std::vector<double*> arr_b1b2(n), arr_b2(n), arr_6b3(n);
  for (int i=0; i<n; i++) {
    arr_b1b2[i] = &buf_b1b2[i*n];
    arr_b2[i]   = &buf_b2[i*n];
    arr_6b3[i]  = &buf_6b3[i*n];
    for (int j=0; j<n; j++) {
      arr_b1b2[i][j] = host_buck1[i][j] * host_buck2[i][j];
      arr_b2[i][j]   = host_buck2[i][j];
      arr_6b3[i][j]  = 6.0 * host_buck3[i][j];
    }
  }

  // Register efield kernel from the same compiled GPU program
  k_efield.set_function(*this->pair_program, "k_gcpm_efield");

  // Allocate per-atom efield buffers (3 acctyp values per atom). These must be
  // resized when the device atom buffers grow (atoms migrate under MPI), see
  // compute_efield().
  _efield_max = nall;
  dev_efield.alloc(3*_efield_max, *(this->ucl_device), UCL_WRITE_ONLY);
  host_efield.alloc(3*_efield_max, *(this->ucl_device), UCL_READ_WRITE);

  coeff1.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,coeff1,host_write,
                         arr_b1b2.data(), arr_b2.data(), arr_6b3.data(), host_cut_ljsq);

  // coeff2 = {buck1, buck3, offset, alpha_ij}
  coeff2.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,coeff2,host_write,
                         host_buck1, host_buck3, host_offset, host_alpha_ij);

  cutsq.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack1(ntypes,lj_types,cutsq,host_write,host_cutsq);

  // sp_lj[0-3] = special_lj, sp_lj[4-7] = special_coul
  sp_lj.alloc(8,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0; i<4; i++) {
    host_write[i]=host_special_lj[i];
    host_write[i+4]=host_special_coul[i];
  }
  ucl_copy(sp_lj,host_write,8,false);

  _cut_coulsq=host_cut_coulsq;
  _qqrd2e=qqrd2e;
  _g_ewald=g_ewald;

  _allocated=true;
  this->_max_bytes=coeff1.row_bytes()+coeff2.row_bytes()+
                   cutsq.row_bytes()+sp_lj.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void GCPMLongT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  coeff1.clear();
  coeff2.clear();
  cutsq.clear();
  sp_lj.clear();
  dev_efield.clear();
  host_efield.clear();
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double GCPMLongT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(GCPMLong<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int GCPMLongT::loop(const int eflag, const int vflag) {
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (shared_types) {
    this->k_pair_sel->set_size(GX,BX);
    this->k_pair_sel->run(&this->atom->x, &coeff1, &coeff2, &sp_lj,
                          &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &eflag, &vflag,
                          &ainum, &nbor_pitch, &this->atom->q, &cutsq,
                          &_cut_coulsq, &_qqrd2e, &_g_ewald,
                          &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x, &coeff1, &coeff2, &_lj_types, &sp_lj,
                     &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                     &this->ans->force, &this->ans->engv, &eflag, &vflag,
                     &ainum, &nbor_pitch, &this->atom->q, &cutsq,
                     &_cut_coulsq, &_qqrd2e, &_g_ewald,
                     &this->_threads_per_atom);
  }
  this->time_pair.stop();
  return GX;
}

// ---------------------------------------------------------------------------
// Compute per-atom electric field (one thread per atom)
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void GCPMLongT::loop_efield() {
  const int BX = this->block_size();
  int ainum = this->ans->inum();
  int GX = static_cast<int>(ceil(static_cast<double>(ainum) /
                                 (BX / this->_threads_per_atom)));
  int nbor_pitch = this->nbor->nbor_pitch();

  this->time_pair.start();
  k_efield.set_size(GX, BX);
  k_efield.run(&this->atom->x, &coeff2, &_lj_types, &sp_lj,
               &this->nbor->dev_nbor, &this->_nbor_data->begin(),
               &dev_efield,
               &ainum, &nbor_pitch, &this->atom->q,
               &_cut_coulsq, &_qqrd2e, &_g_ewald,
               &this->_threads_per_atom);
  this->time_pair.stop();
}

template <class numtyp, class acctyp>
void GCPMLongT::compute_efield(void **efield_ptr) {
  // The efield buffers are allocated in init() at the initial nall. Atoms
  // migrate between subdomains at reneighboring, so a rank's atom count can
  // grow under MPI. Resize the buffers to match the (already-resized) device
  // atom buffers before the kernel writes into them, otherwise the kernel and
  // the host copy run out of bounds (crash under >1 MPI rank).
  int nmax = this->atom->max_atoms();
  if (nmax > _efield_max) {
    _efield_max = nmax;
    dev_efield.clear();
    host_efield.clear();
    dev_efield.alloc(3*_efield_max, *(this->ucl_device), UCL_WRITE_ONLY);
    host_efield.alloc(3*_efield_max, *(this->ucl_device), UCL_READ_WRITE);
  }

  loop_efield();
  int ainum = this->ans->inum();
  ucl_copy(host_efield, dev_efield, 3*ainum, false);
  *efield_ptr = (void *)host_efield.begin();
}

template class GCPMLong<PRECISION,ACC_PRECISION>;
}
