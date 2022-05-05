/***************************************************************************
                            lj_coul_long_soft.cpp
                             -------------------
                            Trung Nguyen (U Chicago)

  Class for acceleration of the lj/cut/coul/long pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : ndtrung@uchicago.edu
 ***************************************************************************/

#if defined(USE_OPENCL)
#include "lj_coul_long_soft_cl.h"
#elif defined(USE_CUDART)
const char *lj_coul_long_soft=0;
#else
#include "lj_coul_long_soft_cubin.h"
#endif

#include "lal_lj_coul_long_soft.h"
#include <cassert>
namespace LAMMPS_AL {
#define LJCoulLongSoftT LJCoulLongSoft<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
LJCoulLongSoftT::LJCoulLongSoft() : LJCoulLong<numtyp,acctyp>() {
}

template <class numtyp, class acctyp>
LJCoulLongSoftT::~LJCoulLongSoft() {
  this->clear();
}

template <class numtyp, class acctyp>
int LJCoulLongSoftT::init(const int ntypes,
                           double **host_cutsq, double **host_lj1,
                           double **host_lj2, double **host_lj3,
                           double **host_lj4, double **host_offset, double **host_epsilon,
                           double *host_special_lj, const int nlocal,
                           const int nall, const int max_nbors,
                           const int maxspecial, const double cell_size,
                           const double gpu_split, FILE *_screen,
                           double **host_cut_ljsq, const double host_cut_coulsq,
                           double *host_special_coul, const double qqrd2e,
                           const double g_ewald) {
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                            _screen,lj_coul_long_soft,"k_lj_coul_long_soft");
  if (success!=0)
    return success;

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  this->shared_types=false;
  int max_shared_types=this->device->max_shared_types();
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    this->shared_types=true;
  }
  this->_lj_types=lj_types;

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<lj_types*lj_types; i++)
    host_write[i]=0.0;

  this->lj1.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,this->lj1,host_write,host_lj1,host_lj2,
           host_cutsq, host_cut_ljsq);

  this->lj3.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,this->lj3,host_write,host_lj3,host_lj4,
                         host_offset,host_epsilon);

  this->sp_lj.alloc(8,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0; i<4; i++) {
    host_write[i]=host_special_lj[i];
    host_write[i+4]=host_special_coul[i];
  }
  ucl_copy(this->sp_lj,host_write,8,false);

  this->_cut_coulsq=host_cut_coulsq;
  this->_qqrd2e=qqrd2e;
  this->_g_ewald=g_ewald;

  this->_allocated=true;
  this->_max_bytes=this->lj1.row_bytes()+this->lj3.row_bytes()+this->sp_lj.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void LJCoulLongSoftT::reinit(const int ntypes, double **host_cutsq, double **host_lj1,
                         double **host_lj2, double **host_lj3, double **host_lj4,
                         double **host_offset, double **host_epsilon, double **host_cut_ljsq) {
  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(this->_lj_types*this->_lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<this->_lj_types*this->_lj_types; i++)
    host_write[i]=0.0;

  this->atom->type_pack4(ntypes,this->_lj_types,this->lj1,host_write,host_lj1,host_lj2,
                         host_cutsq, host_cut_ljsq);
  this->atom->type_pack4(ntypes,this->_lj_types,this->lj3,host_write,host_lj3,host_lj4,
                         host_offset, host_epsilon);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int LJCoulLongSoftT::loop(const int eflag, const int vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (this->shared_types) {
    this->k_pair_sel->set_size(GX,BX);
    this->k_pair_sel->run(&this->atom->x, &this->lj1, &this->lj3, &this->sp_lj,
                          &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &eflag,
                          &vflag, &ainum, &nbor_pitch, &this->atom->q,
                          &this->_cut_coulsq, &this->_qqrd2e, &this->_g_ewald,
                          &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x, &this->lj1, &this->lj3,
                     &this->_lj_types, &this->sp_lj, &this->nbor->dev_nbor,
                     &this->_nbor_data->begin(), &this->ans->force,
                     &this->ans->engv, &eflag, &vflag, &ainum,
                     &nbor_pitch, &this->atom->q, &this->_cut_coulsq,
                     &this->_qqrd2e, &this->_g_ewald, &this->_threads_per_atom);
  }
  this->time_pair.stop();
  return GX;
}

template class LJCoulLongSoft<PRECISION,ACC_PRECISION>;
}
