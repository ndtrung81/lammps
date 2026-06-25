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

   GPU version of pair_gcpm (reaction field) following the pair_amoeba/gpu pattern:
   - dispersion + smeared Coulomb forces with the per-pair charge-charge
     reaction field (term A) computed on GPU (k_gcpm kernel)
   - per-atom efield[] (smeared field + charge->dipole RF, term B) computed on
     GPU (k_gcpm_efield kernel) and read back to CPU before the iterative solver
   - polar iterative solver runs on CPU using full neighbor list (terms C, D and
     the dipole-dipole interactions)
   - GPU_FORCE: host builds full neighbor list (REQ_FULL)
   - GPU_NEIGH: GPU builds neighbor list (REQ_FULL|REQ_NEWTON_OFF)
   No KSpace style and no per-molecule reaction-field passes: the reaction field
   is handled entirely per pair (matching PairGCPM::compute()).
------------------------------------------------------------------------- */

#include "pair_gcpm_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "suffix.h"

#include <cmath>

using namespace LAMMPS_NS;

// same reverse-comm modes as pair_gcpm.cpp
enum {EFIELD, EFIELD_POL};

// External functions from GPU library (lal_gcpm_ext.cpp)

int gcpm_gpu_init(const int ntypes, double **cutsq,
                  double **host_buck1, double **host_buck2, double **host_buck3,
                  double **host_cut_ljsq, double **offset, double **host_alpha_ij,
                  double *special_lj, const int inum, const int nall,
                  const int max_nbors, const int maxspecial,
                  const double cell_size, int &gpu_mode, FILE *screen,
                  double host_cut_coulsq, double *host_special_coul,
                  const double qqrd2e, const double c_rf, const int enable_rf);
void gcpm_gpu_clear();
int **gcpm_gpu_compute_n(const int ago, const int inum_full, const int nall,
                         double **host_x, int *host_type, double *sublo,
                         double *subhi, tagint *tag, int **nspecial,
                         tagint **special, const bool eflag, const bool vflag,
                         const bool eatom, const bool vatom, int &host_start,
                         int **ilist, int **jnum, const double cpu_time,
                         bool &success, double *host_q, double *boxlo,
                         double *prd, int *periodicity);
void gcpm_gpu_compute(const int ago, const int inum_full, const int nall,
                      double **host_x, int *host_type, int *ilist, int *numj,
                      int **firstneigh, const bool eflag, const bool vflag,
                      const bool eatom, const bool vatom, int &host_start,
                      const double cpu_time, bool &success, double *host_q,
                      const int nlocal, double *boxlo, double *prd);
void gcpm_gpu_compute_efield(void **efield_ptr);
double gcpm_gpu_bytes();

/* ---------------------------------------------------------------------- */

PairGCPMGPU::PairGCPMGPU(LAMMPS *lmp) : PairGCPM(lmp), gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  reinitflag = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  efield_pinned = nullptr;
  acc_float = false;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ---------------------------------------------------------------------- */

PairGCPMGPU::~PairGCPMGPU()
{
  gcpm_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairGCPMGPU::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Resize per-atom arrays if needed
  if (enable_polar && atom->nmax > nmax) {
    memory->destroy(efield);
    memory->destroy(efield_pol);
    memory->destroy(mu_old);
    nmax = atom->nmax;
    memory->create(efield, nmax, 3, "pair:efield");
    memory->create(efield_pol, nmax, 3, "pair:efield_pol");
    memory->create(mu_old, nmax, 4, "pair:mu_old");
  }

  int nall = atom->nlocal + atom->nghost;
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int host_start;
  bool success = true;

  // GPU: dispersion + smeared Coulomb forces (incl. term A) via k_gcpm kernel
  if (gpu_mode == GPU_FORCE) {
    gcpm_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
                     ilist, numneigh, firstneigh,
                     eflag, vflag, eflag_atom, vflag_atom,
                     host_start, cpu_time, success, atom->q,
                     atom->nlocal, domain->boxlo, domain->prd);
  } else {
    // GPU_NEIGH: GPU builds its own neighbor list
    int **tmp = gcpm_gpu_compute_n(neighbor->ago, atom->nlocal, nall,
                                   atom->x, atom->type,
                                   domain->sublo, domain->subhi, atom->tag,
                                   atom->nspecial, atom->special,
                                   eflag, vflag, eflag_atom, vflag_atom,
                                   host_start, &ilist, &numneigh, cpu_time,
                                   success, atom->q, domain->boxlo, domain->prd,
                                   domain->periodicity);
    firstneigh = tmp;
  }
  if (!success) error->one(FLERR, "Insufficient memory on accelerator");

  // GPU: per-atom efield (smeared field + charge->dipole RF, term B) via
  // k_gcpm_efield
  if (enable_polar) {
    gcpm_gpu_compute_efield(&efield_pinned);

    // Read efield back from GPU pinned buffer to CPU efield[] array.
    // efield_pinned layout: [Ex0, Ey0, Ez0, Ex1, Ey1, Ez1, ...]
    int nlocal = atom->nlocal;
    if (acc_float) {
      auto *efld = (float *)efield_pinned;
      for (int i = 0; i < nlocal; i++) {
        efield[i][0] = (double)efld[i*3+0];
        efield[i][1] = (double)efld[i*3+1];
        efield[i][2] = (double)efld[i*3+2];
      }
    } else {
      auto *efld = (double *)efield_pinned;
      for (int i = 0; i < nlocal; i++) {
        efield[i][0] = efld[i*3+0];
        efield[i][1] = efld[i*3+1];
        efield[i][2] = efld[i*3+2];
      }
    }

    // Zero efield_pol for ghost atoms before the iterative solver.
    // (local atoms are zeroed each iteration inside compute_induced_efield)
    int ntotal = atom->nlocal + atom->nghost;
    for (int i = atom->nlocal; i < ntotal; i++)
      efield_pol[i][0] = efield_pol[i][1] = efield_pol[i][2] = 0.0;

    // Polar iterative solver on the CPU: dipole->dipole RF (term C) is folded
    // into efield_pol inside compute_induced_efield(), the charge-dipole RF
    // force (term D) is applied in the doA/doB loop. Uses the full neighbor
    // list (REQ_FULL), so neigh_half = 0. No per-molecule reaction-field passes.
    polar(eflag, vflag, 0);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairGCPMGPU::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR, "Pair style gcpm/gpu requires atom attribute q");

  if (enable_polar && (!atom->mu_flag || !atom->torque_flag))
    error->all(FLERR,
               "Pair gcpm/gpu requires atom attributes mu and torque for polar");

  // Replicate parameter setup from PairGCPM::init_style() without adding
  // its own neighbor request (we add REQ_FULL below).
  double maxcut = -1.0;
  double cut;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        cut = init_one(i, j);
        cut *= cut;
        if (cut > maxcut) maxcut = cut;
        cutsq[i][j] = cutsq[j][i] = cut;
      } else
        cutsq[i][j] = cutsq[j][i] = 0.0;
    }
  }
  double cell_size = sqrt(maxcut) + neighbor->skin;

  // reaction-field Coulomb: no KSpace style is required (g_ewald unused on GPU)
  g_ewald = 0.0;
  cut_coulsq = cut_coul * cut_coul;

  setup_reaction_field();

  int maxspecial = 0;
  if (atom->molecular != Atom::ATOMIC) maxspecial = atom->maxspecial;
  int mnf = 5e-2 * neighbor->oneatom;

  int success = gcpm_gpu_init(
      atom->ntypes + 1, cutsq,
      buck1, buck2, buck3, cut_ljsq, offset, alpha_ij,
      force->special_lj,
      atom->nlocal, atom->nlocal + atom->nghost, mnf, maxspecial,
      cell_size, gpu_mode, screen,
      cut_coulsq, force->special_coul, force->qqrd2e, c_rf, enable_rf);
  GPU_EXTRA::check_flag(success, error, world);

  if (gpu_mode == GPU_FORCE) {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_NEWTON_OFF);
  }

  acc_float = Info::has_accelerator_feature("GPU", "precision", "single");
}

/* ---------------------------------------------------------------------- */

double PairGCPMGPU::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes + gcpm_gpu_bytes();
}
