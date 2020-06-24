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
  if (comm->me == 0) 
    printf("Electrolyte with %d cations : %d anions\n", num_cations_per_molecule,
      num_anions_per_molecule);
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
    if (regionflag) {
      int region_attempt = 0;
      xtmp[0] = region_xlo + random_equal->uniform() *
        (region_xhi-region_xlo);
      xtmp[1] = region_ylo + random_equal->uniform() *
        (region_yhi-region_ylo);
      xtmp[2] = region_zlo + random_equal->uniform() *
        (region_zhi-region_zlo);
      while (domain->regions[iregion]->match(xtmp[0],xtmp[1],
                                            xtmp[2]) == 0) {
        xtmp[0] = region_xlo + random_equal->uniform() *
          (region_xhi-region_xlo);
        xtmp[1] = region_ylo + random_equal->uniform() *
          (region_yhi-region_ylo);
        xtmp[2] = region_zlo + random_equal->uniform() *
          (region_zhi-region_zlo);
        region_attempt++;
        if (region_attempt >= max_region_attempts) return;
      }
      if (triclinic) domain->x2lamda(xtmp,lamda);
    } else {
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
      atom->tag[m] = 0; //maxtag_all + i + 1;
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
  atom->tag_extend();
  if (atom->map_style) atom->map_init();
  atom->nghost = 0;
  if (triclinic) domain->x2lamda(atom->nlocal);
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (force->kspace) force->kspace->qsum_qsq();
  if (force->pair->tail_flag) force->pair->reinit();
  double energy_after = energy_full();

  // energy_after corrected by energy_intra

  double permutations = factorial(num_anions_per_molecule)*factorial(num_cations_per_molecule);
  // double deltaphi = zz*pow(volume,natoms_per_molecule)*permutations*natoms_per_molecule *
  //  exp(beta*(energy_before - (energy_after - energy_intra)))/(ngas + natoms_per_molecule);

  double logdeltaphi = log(zz*natoms_per_molecule*permutations/(ngas + natoms_per_molecule)) +
    natoms_per_molecule*log(volume) + beta*(energy_before - (energy_after - energy_intra));
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

  double permutations = factorial(num_anions_per_molecule)*factorial(num_cations_per_molecule);
  //double deltaphi = ngas*exp(beta*((energy_before - energy_intra) - energy_after)) /
  //  (zz*pow(volume,natoms_per_molecule)*permutations*natoms_per_molecule);

  double logdeltaphi = log(ngas/(zz*natoms_per_molecule*permutations)) -
    natoms_per_molecule*log(volume) + beta*((energy_before - energy_intra) - energy_after);
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
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

double FixGCMCElectrolyte::factorial(int n)
{
  if (n < 0 || n > nmaxfactorial) {
    char str[128];
    sprintf(str, "Invalid argument to factorial %d", n);
    error->all(FLERR, str);
  }

  return nfac_table[n];
}

/* ----------------------------------------------------------------------
   factorial n table, size nmaxfactorial+1
------------------------------------------------------------------------- */

const double FixGCMCElectrolyte::nfac_table[] = {
  1,
  1,
  2,
  6,
  24,
  120,
  720,
  5040,
  40320,
  362880,
  3628800,
  39916800,
  479001600,
  6227020800,
  87178291200,
  1307674368000,
  20922789888000,
  355687428096000,
  6.402373705728e+15,
  1.21645100408832e+17,
  2.43290200817664e+18,
  5.10909421717094e+19,
  1.12400072777761e+21,
  2.5852016738885e+22,
  6.20448401733239e+23,
  1.5511210043331e+25,
  4.03291461126606e+26,
  1.08888694504184e+28,
  3.04888344611714e+29,
  8.8417619937397e+30,
  2.65252859812191e+32,
  8.22283865417792e+33,
  2.63130836933694e+35,
  8.68331761881189e+36,
  2.95232799039604e+38,
  1.03331479663861e+40,
  3.71993326789901e+41,
  1.37637530912263e+43,
  5.23022617466601e+44,
  2.03978820811974e+46,
  8.15915283247898e+47,
  3.34525266131638e+49,
  1.40500611775288e+51,
  6.04152630633738e+52,
  2.65827157478845e+54,
  1.1962222086548e+56,
  5.50262215981209e+57,
  2.58623241511168e+59,
  1.24139155925361e+61,
  6.08281864034268e+62,
  3.04140932017134e+64,
  1.55111875328738e+66,
  8.06581751709439e+67,
  4.27488328406003e+69,
  2.30843697339241e+71,
  1.26964033536583e+73,
  7.10998587804863e+74,
  4.05269195048772e+76,
  2.35056133128288e+78,
  1.3868311854569e+80,
  8.32098711274139e+81,
  5.07580213877225e+83,
  3.14699732603879e+85,
  1.98260831540444e+87,
  1.26886932185884e+89,
  8.24765059208247e+90,
  5.44344939077443e+92,
  3.64711109181887e+94,
  2.48003554243683e+96,
  1.71122452428141e+98,
  1.19785716699699e+100,
  8.50478588567862e+101,
  6.12344583768861e+103,
  4.47011546151268e+105,
  3.30788544151939e+107,
  2.48091408113954e+109,
  1.88549470166605e+111,
  1.45183092028286e+113,
  1.13242811782063e+115,
  8.94618213078297e+116,
  7.15694570462638e+118,
  5.79712602074737e+120,
  4.75364333701284e+122,
  3.94552396972066e+124,
  3.31424013456535e+126,
  2.81710411438055e+128,
  2.42270953836727e+130,
  2.10775729837953e+132,
  1.85482642257398e+134,
  1.65079551609085e+136,
  1.48571596448176e+138,
  1.3520015276784e+140,
  1.24384140546413e+142,
  1.15677250708164e+144,
  1.08736615665674e+146,
  1.03299784882391e+148,
  9.91677934870949e+149,
  9.61927596824821e+151,
  9.42689044888324e+153,
  9.33262154439441e+155,
  9.33262154439441e+157,
  9.42594775983835e+159,
  9.61446671503512e+161,
  9.90290071648618e+163,
  1.02990167451456e+166,
  1.08139675824029e+168,
  1.14628056373471e+170,
  1.22652020319614e+172,
  1.32464181945183e+174,
  1.44385958320249e+176,
  1.58824554152274e+178,
  1.76295255109024e+180,
  1.97450685722107e+182,
  2.23119274865981e+184,
  2.54355973347219e+186,
  2.92509369349301e+188,
  3.3931086844519e+190,
  3.96993716080872e+192,
  4.68452584975429e+194,
  5.5745857612076e+196,
  6.68950291344912e+198,
  8.09429852527344e+200,
  9.8750442008336e+202,
  1.21463043670253e+205,
  1.50614174151114e+207,
  1.88267717688893e+209,
  2.37217324288005e+211,
  3.01266001845766e+213,
  3.8562048236258e+215,
  4.97450422247729e+217,
  6.46685548922047e+219,
  8.47158069087882e+221,
  1.118248651196e+224,
  1.48727070609069e+226,
  1.99294274616152e+228,
  2.69047270731805e+230,
  3.65904288195255e+232,
  5.01288874827499e+234,
  6.91778647261949e+236,
  9.61572319694109e+238,
  1.34620124757175e+241,
  1.89814375907617e+243,
  2.69536413788816e+245,
  3.85437071718007e+247,
  5.5502938327393e+249,
  8.04792605747199e+251,
  1.17499720439091e+254,
  1.72724589045464e+256,
  2.55632391787286e+258,
  3.80892263763057e+260,
  5.71338395644585e+262,
  8.62720977423323e+264,
  1.31133588568345e+267,
  2.00634390509568e+269,
  3.08976961384735e+271,
  4.78914290146339e+273,
  7.47106292628289e+275,
  1.17295687942641e+278,
  1.85327186949373e+280,
  2.94670227249504e+282,
  4.71472363599206e+284,
  7.59070505394721e+286,
  1.22969421873945e+289,
  2.0044015765453e+291,
  3.28721858553429e+293,
  5.42391066613159e+295,
  9.00369170577843e+297,
  1.503616514865e+300, // nmaxfactorial = 167
};

