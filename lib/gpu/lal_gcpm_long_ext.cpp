/***************************************************************************
                                gcpm_long_ext.cpp
                             -------------------
                              Trung Dac Nguyen

  Functions for LAMMPS access to gcpm/long GPU acceleration routines.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : 6/10/2026
    email                : ndactrung@gmail.com
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_gcpm_long.h"

using namespace std;
using namespace LAMMPS_AL;

static GCPMLong<PRECISION,ACC_PRECISION> GCPMLongLMF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int gcpm_long_gpu_init(const int ntypes, double **cutsq,
                  double **host_buck1, double **host_buck2, double **host_buck3,
                  double **host_cut_ljsq, double **offset, double **host_alpha_ij,
                  double *special_lj, const int inum, const int nall,
                  const int max_nbors, const int maxspecial,
                  const double cell_size, int &gpu_mode, FILE *screen,
                  double host_cut_coulsq, double *host_special_coul,
                  const double qqrd2e, const double g_ewald) {
  GCPMLongLMF.clear();
  gpu_mode=GCPMLongLMF.device->gpu_mode();
  double gpu_split=GCPMLongLMF.device->particle_split();
  int first_gpu=GCPMLongLMF.device->first_device();
  int last_gpu=GCPMLongLMF.device->last_device();
  int world_me=GCPMLongLMF.device->world_me();
  int gpu_rank=GCPMLongLMF.device->gpu_rank();
  int procs_per_gpu=GCPMLongLMF.device->procs_per_gpu();

  GCPMLongLMF.device->init_message(screen,"gcpm/long",first_gpu,last_gpu);
  bool message=false;
  if (GCPMLongLMF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=GCPMLongLMF.init(ntypes, cutsq, host_buck1, host_buck2, host_buck3,
                       host_cut_ljsq, offset, host_alpha_ij,
                       special_lj, inum, nall, max_nbors, maxspecial,
                       cell_size, gpu_split, screen,
                       host_cut_coulsq, host_special_coul, qqrd2e, g_ewald);

  GCPMLongLMF.device->world_barrier();
  if (message)
    fprintf(screen,"Done.\n");

  for (int i=0; i<procs_per_gpu; i++) {
    if (message) {
      if (last_gpu-first_gpu==0)
        fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
      else
        fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                last_gpu,i);
      fflush(screen);
    }
    if (gpu_rank==i && world_me!=0)
      init_ok=GCPMLongLMF.init(ntypes, cutsq, host_buck1, host_buck2, host_buck3,
                         host_cut_ljsq, offset, host_alpha_ij,
                         special_lj, inum, nall, max_nbors, maxspecial,
                         cell_size, gpu_split, screen,
                         host_cut_coulsq, host_special_coul, qqrd2e, g_ewald);

    GCPMLongLMF.device->serialize_init();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    GCPMLongLMF.estimate_gpu_overhead();
  return init_ok;
}

void gcpm_long_gpu_clear() {
  GCPMLongLMF.clear();
}

int** gcpm_long_gpu_compute_n(const int ago, const int inum_full,
                         const int nall, double **host_x, int *host_type,
                         double *sublo, double *subhi, tagint *tag, int **nspecial,
                         tagint **special, const bool eflag, const bool vflag,
                         const bool eatom, const bool vatom, int &host_start,
                         int **ilist, int **jnum, const double cpu_time,
                         bool &success, double *host_q, double *boxlo,
                         double *prd, int *periodicity) {
  return GCPMLongLMF.compute(ago, inum_full, nall, host_x, host_type, sublo,
                       subhi, tag, nspecial, special, eflag, vflag, eatom,
                       vatom, host_start, ilist, jnum, cpu_time, success,
                       host_q, boxlo, prd, periodicity);
}

void gcpm_long_gpu_compute(const int ago, const int inum_full, const int nall,
                      double **host_x, int *host_type, int *ilist, int *numj,
                      int **firstneigh, const bool eflag, const bool vflag,
                      const bool eatom, const bool vatom, int &host_start,
                      const double cpu_time, bool &success, double *host_q,
                      const int nlocal, double *boxlo, double *prd) {
  GCPMLongLMF.compute(ago, inum_full, nall, host_x, host_type, ilist, numj,
                firstneigh, eflag, vflag, eatom, vatom, host_start,
                cpu_time, success, host_q, nlocal, boxlo, prd);
}

void gcpm_long_gpu_compute_efield(void **efield_ptr) {
  GCPMLongLMF.compute_efield(efield_ptr);
}

double gcpm_long_gpu_bytes() {
  return GCPMLongLMF.host_memory_usage();
}
