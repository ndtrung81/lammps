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
   Contributing author: Trung Dac Nguyen (ndactrung@gmail.com)
   Ref: Wang, Yu, Langston, Fraige, Particle shape effects in discrete
   element modelling of cohesive angular particles, Granular Matter 2011,
   13:1-12.
   Note: The current implementation has not taken into account
         the contact history for friction forces.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_body_rounded_polyhedron_lj.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_body.h"
#include "body_rounded_polyhedron.h"
#include "comm.h"
#include "force.h"
#include "fix.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "math_extra.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathExtra;
using namespace MathConst;

#define MAX_CONTACTS 32  // for 3D models

//#define _POLYHEDRON_DEBUG

/* ---------------------------------------------------------------------- */

PairBodyRoundedPolyhedronLJ::PairBodyRoundedPolyhedronLJ(LAMMPS *lmp) :
  PairBodyRoundedPolyhedron(lmp)
{
  cut = NULL;
  epsilon = NULL;
  sigma = NULL;
  lj1 = NULL;
  lj2 = NULL;
  lj3 = NULL;
  lj4 = NULL;
  offset = NULL;
}

/* ---------------------------------------------------------------------- */

PairBodyRoundedPolyhedronLJ::~PairBodyRoundedPolyhedronLJ()
{
  if (allocated) {
    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairBodyRoundedPolyhedronLJ::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  int ni,nj,npi,npj,ifirst,jfirst,nei,nej,iefirst,jefirst;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,facc[3];
  double rsq,eradi,eradj,k_nij,k_naij;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **torque = atom->torque;
  double **angmom = atom->angmom;
  int *body = atom->body;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // grow the per-atom lists if necessary and initialize

  if (atom->nmax > nmax) {
    memory->destroy(dnum);
    memory->destroy(dfirst);
    memory->destroy(ednum);
    memory->destroy(edfirst);
    memory->destroy(facnum);
    memory->destroy(facfirst);
    memory->destroy(enclosing_radius);
    memory->destroy(rounded_radius);
    nmax = atom->nmax;
    memory->create(dnum,nmax,"pair:dnum");
    memory->create(dfirst,nmax,"pair:dfirst");
    memory->create(ednum,nmax,"pair:ednum");
    memory->create(edfirst,nmax,"pair:edfirst");
    memory->create(facnum,nmax,"pair:facnum");
    memory->create(facfirst,nmax,"pair:facfirst");
    memory->create(enclosing_radius,nmax,"pair:enclosing_radius");
    memory->create(rounded_radius,nmax,"pair:rounded_radius");
  }

  ndiscrete = nedge = nface = 0;
  for (i = 0; i < nall; i++)
    dnum[i] = ednum[i] = facnum[i] = 0;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    if (body[i] >= 0) {
      if (dnum[i] == 0) body2space(i);
      npi = dnum[i];
      ifirst = dfirst[i];
      nei = ednum[i];
      iefirst = edfirst[i];
      eradi = enclosing_radius[i];
    }

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      // body/body interactions

      evdwl = 0.0;
      facc[0] = facc[1] = facc[2] = 0;

      if (body[i] < 0 || body[j] < 0) continue;

      if (dnum[j] == 0) body2space(j);
      npj = dnum[j];
      jfirst = dfirst[j];
      nej = ednum[j];
      jefirst = edfirst[j];
      eradj = enclosing_radius[j];
      
      // no interaction

      double r = sqrt(rsq);
      if (r > eradi + eradj + cut_inner) continue;

      // sphere-sphere interaction

      if (npi == 1 && npj == 1) {
        sphere_against_sphere(i, j, delx, dely, delz, rsq,
                              lj1[itype][jtype], lj2[itype][jtype], v, f, evflag);
        continue;
      }

      // reset vertex and edge forces

      for (ni = 0; ni < npi; ni++) {
        discrete[ifirst+ni][3] = 0;
        discrete[ifirst+ni][4] = 0;
        discrete[ifirst+ni][5] = 0;
        discrete[ifirst+ni][6] = 0;
      }

      for (nj = 0; nj < npj; nj++) {
        discrete[jfirst+nj][3] = 0;
        discrete[jfirst+nj][4] = 0;
        discrete[jfirst+nj][5] = 0;
        discrete[jfirst+nj][6] = 0;
      }

      for (ni = 0; ni < nei; ni++) {
        edge[iefirst+ni][3] = 0;
        edge[iefirst+ni][4] = 0;
        edge[iefirst+ni][5] = 0;
      }

      for (nj = 0; nj < nej; nj++) {
        edge[jefirst+nj][3] = 0;
        edge[jefirst+nj][4] = 0;
        edge[jefirst+nj][5] = 0;
      }

      // one of the two bodies is a sphere

      if (npj == 1) {
        sphere_against_face(i, j, jtype, x, v, f, torque,
                            angmom, evflag);
        sphere_against_edge(i, j, jtype, x, v, f, torque,
                            angmom, evflag);
        continue;
      } else if (npi == 1) {
        sphere_against_face(j, i, itype, x, v, f, torque,
                            angmom, evflag);
        sphere_against_edge(j, i, itype, x, v, f, torque,
                            angmom, evflag);
        continue;
      }

      int interact, num_contacts;
      Contact contact_list[MAX_CONTACTS];

      num_contacts = 0;

      // check interaction between i's edges and j' faces
      #ifdef _POLYHEDRON_DEBUG
      printf("INTERACTION between edges of %d vs. faces of %d:\n", i, j);
      #endif 
      interact = edge_against_face(i, j, x, contact_list,
                                   num_contacts, evdwl, facc);

      // check interaction between j's edges and i' faces
      #ifdef _POLYHEDRON_DEBUG
      printf("\nINTERACTION between edges of %d vs. faces of %d:\n", j, i);
      #endif
      interact = edge_against_face(j, i, x, contact_list,
                                   num_contacts, evdwl, facc);

      // check interaction between i's edges and j' edges
      #ifdef _POLYHEDRON_DEBUG
      printf("INTERACTION between edges of %d vs. edges of %d:\n", i, j);
      #endif 
      interact = edge_against_edge(i, j, x, contact_list,
                                   num_contacts, evdwl, facc);

      // estimate the contact area
      // also consider point contacts and line contacts

      if (num_contacts > 0) {
        rescale_cohesive_forces(x, f, torque, contact_list, num_contacts, facc);
      }

      if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,evdwl,0.0,
                               facc[0],facc[1],facc[2],delx,dely,delz);

    } // end for jj
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairBodyRoundedPolyhedronLJ::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");

  memory->create(maxerad,n+1,"pair:maxerad");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairBodyRoundedPolyhedronLJ::settings(int narg, char **arg)
{
  if (narg < 5) error->all(FLERR,"Illegal pair_style command");

  c_n = force->numeric(FLERR,arg[0]);
  c_t = force->numeric(FLERR,arg[1]);
  mu = force->numeric(FLERR,arg[2]);
  A_ua = force->numeric(FLERR,arg[3]);
  cut_inner = force->numeric(FLERR,arg[4]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_inner;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairBodyRoundedPolyhedronLJ::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_inner;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairBodyRoundedPolyhedronLJ::init_style()
{
  PairBodyRoundedPolyhedron::init_style();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairBodyRoundedPolyhedronLJ::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  sigma[j][i] = sigma[i][j];

  return (maxerad[i]+maxerad[j]);
}


/* ----------------------------------------------------------------------
   Force kernel
---------------------------------------------------------------------- */

void PairBodyRoundedPolyhedronLJ::kernel_force(double R, int itype, int jtype,
  double cut_inner, double& fpair, double& energy)
{
  if (R > cut_inner) {
    fpair = 0;
    energy = 0;
    return;
  }
  double r = R + sigma[itype][jtype];
  double rsq = r * r;
  double r2inv = 1.0/rsq;
  double r6inv = r2inv*r2inv*r2inv;
  double forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  fpair = forcelj / r;
  energy = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
}


