/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Trung Nguyen (Northwestern)
------------------------------------------------------------------------- */

#include "fix_gcmc_electrolyte.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "compute.h"
#include "group.h"
#include "domain.h"
#include "region.h"
#include "random_park.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "neighbor.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

// large energy value used to signal overlap

#define MAXENERGYSIGNAL 1.0e100

// this must be lower than MAXENERGYSIGNAL
// by a large amount, so that it is still
// less than total energy when negative
// energy contributions are added to MAXENERGYSIGNAL

#define MAXENERGYTEST 1.0e50

/* ---------------------------------------------------------------------- */

FixGCMCElectrolyte::FixGCMCElectrolyte(LAMMPS *lmp, int narg, char **arg) :
  FixGCMC(lmp, narg, arg)
{
  full_flag = true;
}

/* ---------------------------------------------------------------------- */

void FixGCMCElectrolyte::init()
{
  FixGCMC::init();
  
  num_anions_per_molecule = 0;
  num_cations_per_molecule = 0;
  for (int i = 0; i < natoms_per_molecule; i++) {
    if (onemols[imol]->q[i] < 0) num_anions_per_molecule++;
    if (onemols[imol]->q[i] > 0) num_cations_per_molecule++;
  }
  printf("natoms per molecule = %d; %d %d\n", natoms_per_molecule, num_anions_per_molecule, num_cations_per_molecule);
}

/* ----------------------------------------------------------------------
  attempt to delete a group of ions with zero net charge
------------------------------------------------------------------------- */

void FixGCMCElectrolyte::attempt_molecule_deletion_full()
{
  ndeletion_attempts += 1.0;

  if (ngas == 0 || ngas <= min_ngas) return;

  // work-around to avoid n=0 problem with fix rigid/nvt/small

  if (ngas == natoms_per_molecule) return;

  // select molecules to delete

  tagint deletion_molecule = pick_random_gas_molecule();
  if (deletion_molecule == -1) return;

  double energy_before = energy_stored;

  // check nmolq, grow arrays if necessary

  int nmolq = 0;
  for (int i = 0; i < atom->nlocal; i++)
    if (atom->molecule[i] == deletion_molecule)
      if (atom->q_flag) nmolq++;

  if (nmolq > nmaxmolatoms)
    grow_molecule_arrays(nmolq);

  int m = 0;
  int *tmpmask = new int[atom->nlocal];
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->molecule[i] == deletion_molecule) {
      tmpmask[i] = atom->mask[i];
      atom->mask[i] = exclusion_group_bit;
      toggle_intramolecular(i);
      if (atom->q_flag) {
        molq[m] = atom->q[i];
        m++;
        atom->q[i] = 0.0;
      }
    }
  }
  if (force->kspace) force->kspace->qsum_qsq();
  if (force->pair->tail_flag) force->pair->reinit();
  double energy_after = energy_full();

  // energy_before corrected by energy_intra

  double deltaphi = ngas*exp(beta*((energy_before - energy_intra) - energy_after))/(zz*pow(volume,natoms_per_molecule)*natoms_per_molecule);

  double logdeltaphi = log(ngas/(zz*natoms_per_molecule)) - natoms_per_molecule*log(volume) +
    beta*((energy_before - energy_intra) - energy_after);
  double p = logdeltaphi;
  if (logdeltaphi > 0.0) p = 1.0;
  else p = exp(logdeltaphi);

  if (random_equal->uniform() < p) { // deltaphi

    // accept the trial deletion move

    int i = 0;
    while (i < atom->nlocal) {
      if (atom->molecule[i] == deletion_molecule) {
        atom->avec->copy(atom->nlocal-1,i,1);
        atom->nlocal--;
      } else i++;
    }

    atom->natoms -= natoms_per_molecule;
    if (atom->map_style) atom->map_init();
    ndeletion_successes += 1.0;
    energy_stored = energy_after;

  } else {

    energy_stored = energy_before;
    int m = 0;
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->molecule[i] == deletion_molecule) {
        atom->mask[i] = tmpmask[i];
        toggle_intramolecular(i);
        if (atom->q_flag) {
          atom->q[i] = molq[m];
          m++;
        }
      }
    }
    if (force->kspace) force->kspace->qsum_qsq();
    if (force->pair->tail_flag) force->pair->reinit();
  }
  update_gas_atoms_list();
  delete [] tmpmask;
}

/* ----------------------------------------------------------------------
  attempt to insert a group of ions with zero net charge
------------------------------------------------------------------------- */

void FixGCMCElectrolyte::attempt_molecule_insertion_full()
{
  double lamda[3];
  ninsertion_attempts += 1.0;

  if (ngas >= max_ngas) return;

  double energy_before = energy_stored;

  tagint maxmol = 0;
  for (int i = 0; i < atom->nlocal; i++) maxmol = MAX(maxmol,atom->molecule[i]);
  tagint maxmol_all;
  MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
  maxmol_all++;
  if (maxmol_all >= MAXTAGINT)
    error->all(FLERR,"Fix gcmc ran out of available molecule IDs");
  int insertion_molecule = maxmol_all;

  tagint maxtag = 0;
  for (int i = 0; i < atom->nlocal; i++) maxtag = MAX(maxtag,atom->tag[i]);
  tagint maxtag_all;
  MPI_Allreduce(&maxtag,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

  int nlocalprev = atom->nlocal;

  
  double vnew[3];
  vnew[0] = random_equal->gaussian()*sigma;
  vnew[1] = random_equal->gaussian()*sigma;
  vnew[2] = random_equal->gaussian()*sigma;

  for (int i = 0; i < natoms_per_molecule; i++) {
    double xtmp[3];
    if (triclinic == 0) {
      xtmp[0] = xlo + random_equal->uniform() * (xhi-xlo);
      xtmp[1] = ylo + random_equal->uniform() * (yhi-ylo);
      xtmp[2] = zlo + random_equal->uniform() * (zhi-zlo);
    } else {
      lamda[0] = random_equal->uniform();
      lamda[1] = random_equal->uniform();
      lamda[2] = random_equal->uniform();

      // wasteful, but necessary

      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;

      domain->lamda2x(lamda,xtmp);
    }

    // need to adjust image flags in remap()

    imageint imagetmp = imagezero;
    domain->remap(xtmp,imagetmp);
    if (!domain->inside(xtmp))
      error->one(FLERR,"Fix gcmc put atom outside box");

    int proc_flag = 0;
    if (triclinic == 0) {
      if (xtmp[0] >= sublo[0] && xtmp[0] < subhi[0] &&
          xtmp[1] >= sublo[1] && xtmp[1] < subhi[1] &&
          xtmp[2] >= sublo[2] && xtmp[2] < subhi[2]) proc_flag = 1;
    } else {
      domain->x2lamda(xtmp,lamda);
      if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
          lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
          lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
    }

    if (proc_flag) {
      atom->avec->create_atom(onemols[imol]->type[i],xtmp);
      int m = atom->nlocal - 1;

      // add to groups
      // optionally add to type-based groups

      atom->mask[m] = groupbitall;
      for (int igroup = 0; igroup < ngrouptypes; igroup++) {
        if (ngcmc_type == grouptypes[igroup])
          atom->mask[m] |= grouptypebits[igroup];
      }

      atom->image[m] = imagetmp;
      atom->molecule[m] = insertion_molecule;
      if (maxtag_all+i+1 >= MAXTAGINT)
        error->all(FLERR,"Fix gcmc ran out of available atom IDs");
      atom->tag[m] = maxtag_all + i + 1;
      atom->v[m][0] = vnew[0];
      atom->v[m][1] = vnew[1];
      atom->v[m][2] = vnew[2];

      atom->add_molecule_atom(onemols[imol],i,m,maxtag_all);
      modify->create_attribute(m);
    }
  }

  atom->natoms += natoms_per_molecule;
  if (atom->natoms < 0)
    error->all(FLERR,"Too many total atoms");
  atom->nbonds += onemols[imol]->nbonds;
  atom->nangles += onemols[imol]->nangles;
  atom->ndihedrals += onemols[imol]->ndihedrals;
  atom->nimpropers += onemols[imol]->nimpropers;
  if (atom->map_style) atom->map_init();
  atom->nghost = 0;
  if (triclinic) domain->x2lamda(atom->nlocal);
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (force->kspace) force->kspace->qsum_qsq();
  if (force->pair->tail_flag) force->pair->reinit();
  double energy_after = energy_full();

  // energy_after corrected by energy_intra

  double deltaphi = zz*pow(volume,natoms_per_molecule)*natoms_per_molecule*
    exp(beta*(energy_before - (energy_after - energy_intra)))/(ngas + natoms_per_molecule);

  double logdeltaphi = log(zz*natoms_per_molecule/(ngas + natoms_per_molecule)) + natoms_per_molecule*log(volume) + 
    beta*(energy_before - (energy_after - energy_intra));
  double p = logdeltaphi;
  if (logdeltaphi > 0.0) p = 1.0;
  else p = exp(logdeltaphi);

  if (energy_after < MAXENERGYTEST &&
      random_equal->uniform() < p) { // deltaphi

    ninsertion_successes += 1.0;
    energy_stored = energy_after;

  } else {

    atom->nbonds -= onemols[imol]->nbonds;
    atom->nangles -= onemols[imol]->nangles;
    atom->ndihedrals -= onemols[imol]->ndihedrals;
    atom->nimpropers -= onemols[imol]->nimpropers;
    atom->natoms -= natoms_per_molecule;

    energy_stored = energy_before;
    int i = 0;
    while (i < atom->nlocal) {
      if (atom->molecule[i] == insertion_molecule) {
        atom->avec->copy(atom->nlocal-1,i,1);
        atom->nlocal--;
      } else i++;
    }
    if (force->kspace) force->kspace->qsum_qsq();
    if (force->pair->tail_flag) force->pair->reinit();
  }
  update_gas_atoms_list();
}

