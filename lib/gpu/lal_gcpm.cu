// **************************************************************************
//                                   gcpm.cu
//                             -------------------
//                               Trung Dac Nguyen
//
//  Device code for acceleration of the gcpm pair style.
//  Computes Buckingham exp-6 dispersion + Gaussian-smeared Coulomb forces.
//  The polarizable (iterative dipole) part runs on the CPU.
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                : 06/10/2026
//    email                : ndactrung@gmail.com
// ***************************************************************************

#if defined(NV_KERNEL) || defined(USE_HIP)

#include "lal_aux_fun1.h"
#ifndef _DOUBLE_DOUBLE
_texture( pos_tex,float4);
_texture( q_tex,float);
#else
_texture_2d( pos_tex,int4);
_texture( q_tex,int2);
#endif

#else
#define pos_tex x_
#define q_tex q_
#endif

/* Reduce per-atom electric field (efx,efy,efz) across t_per_atom threads,
   then write one result per atom.  Mirrors store_answers_eam in lal_eam.cu. */
#if (SHUFFLE_AVAIL == 0)
#define local_allocate_store_efield()                                        \
  __local acctyp red_acc[3][BLOCK_PAIR];

#define store_efield(efx,efy,efz,ii,inum,tid,t_per_atom,offset,efield_out)   \
  if (t_per_atom>1) {                                                        \
    simd_reduce_add3(t_per_atom, red_acc, offset, tid, efx, efy, efz);       \
  }                                                                          \
  if (offset==0 && ii<inum) {                                                \
    efield_out[ii*3  ]=efx;                                                  \
    efield_out[ii*3+1]=efy;                                                  \
    efield_out[ii*3+2]=efz;                                                  \
  }
#else
#define local_allocate_store_efield()
#define store_efield(efx,efy,efz,ii,inum,tid,t_per_atom,offset,efield_out)  \
  if (t_per_atom>1) {                                                        \
    for (unsigned int s=t_per_atom/2; s>0; s>>=1) {                         \
      efx += shfl_down(efx, s, t_per_atom);                                  \
      efy += shfl_down(efy, s, t_per_atom);                                  \
      efz += shfl_down(efz, s, t_per_atom);                                  \
    }                                                                        \
  }                                                                          \
  if (offset==0 && ii<inum) {                                                \
    efield_out[ii*3  ]=efx;                                                  \
    efield_out[ii*3+1]=efy;                                                  \
    efield_out[ii*3+2]=efz;                                                  \
  }
#endif

// k_gcpm: non-fast kernel (global memory for per-type arrays)
// Computes: Buckingham exp-6 dispersion + Gaussian-smeared Coulomb forces
// No efield accumulation (handled on CPU for polar interactions).

__kernel void k_gcpm(const __global numtyp4 *restrict x_,
                     const __global numtyp4 *restrict coeff1,
                     const __global numtyp4 *restrict coeff2,
                     const int lj_types,
                     const __global numtyp *restrict sp_lj_in,
                     const __global int *dev_nbor,
                     const __global int *dev_packed,
                     __global acctyp3 *restrict ans,
                     __global acctyp *restrict engv,
                     const int eflag, const int vflag, const int inum,
                     const int nbor_pitch,
                     const __global numtyp *restrict q_,
                     const __global numtyp *restrict cutsq,
                     const numtyp cut_coulsq, const numtyp qqrd2e,
                     const numtyp g_ewald, const numtyp rsmooth_sq,
                     const acctyp c0, const acctyp c1, const acctyp c2,
                     const acctyp c3, const acctyp c4, const acctyp c5,
                     const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp sp_lj[8];
  int n_stride;
  local_allocate_store_charge();

  sp_lj[0]=sp_lj_in[0];
  sp_lj[1]=sp_lj_in[1];
  sp_lj[2]=sp_lj_in[2];
  sp_lj[3]=sp_lj_in[3];
  sp_lj[4]=sp_lj_in[4];
  sp_lj[5]=sp_lj_in[5];
  sp_lj[6]=sp_lj_in[6];
  sp_lj[7]=sp_lj_in[7];

  acctyp3 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, e_coul, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    e_coul=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }

  if (ii<inum) {
    int nbor, nbor_end;
    int i, numj;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex);
    numtyp qtmp; fetch(qtmp,i,q_tex);
    int itype=ix.w;

    for ( ; nbor<nbor_end; nbor+=n_stride) {
      ucl_prefetch(dev_packed+nbor+n_stride);
      int j=dev_packed[nbor];

      numtyp factor_lj, factor_coul;
      factor_lj = sp_lj[sbmask(j)];
      factor_coul = (numtyp)1.0-sp_lj[sbmask(j)+4];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex);
      int jtype=jx.w;

      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;

      int mtype=itype*lj_types+jtype;
      if (rsq<cutsq[mtype]) {
        numtyp r2inv=ucl_recip(rsq);
        numtyp forcecoul=(numtyp)0.0, force_lj=(numtyp)0.0;
        numtyp r6inv=(numtyp)0.0, rexp=(numtyp)0.0;
        numtyp r=(numtyp)0.0;

        // Buckingham dispersion
        if (rsq < coeff1[mtype].w) {
          r = ucl_sqrt(rsq);
          rexp = ucl_exp(-coeff1[mtype].y*r);
          r6inv = r2inv*r2inv*r2inv;
          // force_lj = (buck1*buck2*r*rexp - 6*buck3*r6inv) * factor_lj
          force_lj = (coeff1[mtype].x*r*rexp - coeff1[mtype].z*r6inv)*factor_lj;
        }

        // Gaussian-smeared Coulomb (Eqs. 4 and 5 from Paricaud et al. 2005)
        // Guard: skip if either charge is zero (matches CPU charge_charge behavior).
        // alpha_ij can be +INF for zero-sigma atoms; INF*0 = NaN in falpha if unchecked.
        numtyp ealpha=(numtyp)0.0;
        numtyp prefactor_erfa=(numtyp)0.0;
        numtyp qj; fetch(qj,j,q_tex);
        if (rsq < cut_coulsq && qtmp != (numtyp)0.0 && qj != (numtyp)0.0) {
          r = ucl_sqrt(rsq);
          numtyp grij = g_ewald * r;
          numtyp expm2 = ucl_exp(-grij*grij);
          // erf(grij) = 1 - erfc(grij)
          numtyp erf_g = (numtyp)1.0 - ucl_erfc(grij);

          numtyp aij = coeff2[mtype].w;
          numtyp aijr = aij * r;
          numtyp expa = ucl_exp(-aijr*aijr);
          // erf(aijr) = 1 - erfc(aijr)
          numtyp erfa = (numtyp)1.0 - ucl_erfc(aijr);

          numtyp falpha = erfa - EWALD_F*aijr*expa;
          numtyp prefactor = qqrd2e * qtmp * qj / r;
          ealpha = prefactor * (erfa - erf_g);
          prefactor_erfa = prefactor * erfa;

          // factor_coul = 1 - special_coul; inline avoids inverted-conditional bug
          forcecoul = prefactor * (falpha - erf_g + EWALD_F*grij*expm2 - factor_coul*falpha);

          // Smoothing: polynomial taper in [rsmooth, cut_coul].
          // Computed in acctyp (double) to avoid catastrophic cancellation:
          // coefficients c4~4512, c5~-77 lead to terms ~1e8 that cancel to [0,1].
          if (rsq > rsmooth_sq) {
            acctyp r_a = (acctyp)r, rsq_a = (acctyp)rsq;
            acctyp rcu_a = r_a * rsq_a;
            acctyp rqu_a = rsq_a * rsq_a;
            acctyp sme = c5*rqu_a*r_a + c4*rqu_a + c3*rcu_a + c2*rsq_a + c1*r_a + c0;
            acctyp smf = (acctyp)5.0*c5*rqu_a + (acctyp)4.0*c4*rcu_a
                       + (acctyp)3.0*c3*rsq_a + (acctyp)2.0*c2*r_a + c1;
            forcecoul = (numtyp)((acctyp)forcecoul*sme - (acctyp)ealpha*smf*r_a);
            ealpha = (numtyp)((acctyp)ealpha * sme);
          }
        }

        numtyp force = (force_lj + forcecoul) * r2inv;

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (EVFLAG && eflag) {
          if (rsq < coeff1[mtype].w) {
            // energy = factor_lj*(buck1*rexp - buck3*r6inv - offset)
            numtyp e = coeff2[mtype].x*rexp - coeff2[mtype].y*r6inv - coeff2[mtype].z;
            energy += factor_lj * e;
          }
          if (rsq < cut_coulsq) {
            e_coul += ealpha;
            if (factor_coul > (numtyp)0.0)
              e_coul -= factor_coul*prefactor_erfa;
          }
        }
        if (EVFLAG && vflag) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
  } // if ii
  store_answers_q(f,energy,e_coul,virial,ii,inum,tid,t_per_atom,offset,eflag,
                  vflag,ans,engv);
}

// k_gcpm_fast: fast kernel (shared memory for per-type arrays when ntypes < MAX_SHARED_TYPES)

__kernel void k_gcpm_fast(const __global numtyp4 *restrict x_,
                          const __global numtyp4 *restrict coeff1_in,
                          const __global numtyp4 *restrict coeff2_in,
                          const __global numtyp *restrict sp_lj_in,
                          const __global int *dev_nbor,
                          const __global int *dev_packed,
                          __global acctyp3 *restrict ans,
                          __global acctyp *restrict engv,
                          const int eflag, const int vflag, const int inum,
                          const int nbor_pitch,
                          const __global numtyp *restrict q_,
                          const __global numtyp *restrict cutsq,
                          const numtyp cut_coulsq, const numtyp qqrd2e,
                          const numtyp g_ewald, const numtyp rsmooth_sq,
                          const acctyp c0, const acctyp c1, const acctyp c2,
                          const acctyp c3, const acctyp c4, const acctyp c5,
                          const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp4 coeff1[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 coeff2[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp sp_lj[8];
  int n_stride;
  local_allocate_store_charge();

  if (tid<8)
    sp_lj[tid]=sp_lj_in[tid];
  if (tid<MAX_SHARED_TYPES*MAX_SHARED_TYPES) {
    coeff1[tid]=coeff1_in[tid];
    coeff2[tid]=coeff2_in[tid];
  }

  acctyp3 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, e_coul, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    e_coul=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }

  __syncthreads();

  if (ii<inum) {
    int nbor, nbor_end;
    int i, numj;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex);
    numtyp qtmp; fetch(qtmp,i,q_tex);
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);

    for ( ; nbor<nbor_end; nbor+=n_stride) {
      ucl_prefetch(dev_packed+nbor+n_stride);
      int j=dev_packed[nbor];

      numtyp factor_lj, factor_coul;
      factor_lj = sp_lj[sbmask(j)];
      factor_coul = (numtyp)1.0-sp_lj[sbmask(j)+4];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex);
      int mtype=itype+jx.w;

      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;

      if (rsq<cutsq[mtype]) {
        numtyp r2inv=ucl_recip(rsq);
        numtyp forcecoul=(numtyp)0.0, force_lj=(numtyp)0.0;
        numtyp r6inv=(numtyp)0.0, rexp=(numtyp)0.0;
        numtyp r=(numtyp)0.0;

        // Buckingham dispersion
        if (rsq < coeff1[mtype].w) {
          r = ucl_sqrt(rsq);
          rexp = ucl_exp(-coeff1[mtype].y*r);
          r6inv = r2inv*r2inv*r2inv;
          force_lj = (coeff1[mtype].x*r*rexp - coeff1[mtype].z*r6inv)*factor_lj;
        }
        // Gaussian-smeared Coulomb (Eqs. 4 and 5 from Paricaud et al. 2005)
        // Guard: skip if either charge is zero (matches CPU charge_charge behavior).
        // alpha_ij can be +INF for zero-sigma atoms; INF*0 = NaN in falpha if unchecked.
        numtyp ealpha=(numtyp)0.0;
        numtyp prefactor_erfa=(numtyp)0.0;
        numtyp qj; fetch(qj,j,q_tex);
        if (rsq < cut_coulsq && qtmp != (numtyp)0.0 && qj != (numtyp)0.0) {
          r = ucl_sqrt(rsq);
          numtyp grij = g_ewald * r;
          numtyp expm2 = ucl_exp(-grij*grij);
          numtyp erf_g = (numtyp)1.0 - ucl_erfc(grij);

          numtyp aij = coeff2[mtype].w;
          numtyp aijr = aij * r;
          numtyp expa = ucl_exp(-aijr*aijr);
          numtyp erfa = (numtyp)1.0 - ucl_erfc(aijr);

          numtyp falpha = erfa - EWALD_F*aijr*expa;
          numtyp prefactor = qqrd2e * qtmp * qj / r;
          ealpha = prefactor * (erfa - erf_g);
          prefactor_erfa = prefactor * erfa;

          // factor_coul = 1 - special_coul; inline avoids inverted-conditional bug
          forcecoul = prefactor * (falpha - erf_g + EWALD_F*grij*expm2 - factor_coul*falpha);

          if (rsq > rsmooth_sq) {
            acctyp r_a = (acctyp)r, rsq_a = (acctyp)rsq;
            acctyp rcu_a = r_a * rsq_a;
            acctyp rqu_a = rsq_a * rsq_a;
            acctyp sme = c5*rqu_a*r_a + c4*rqu_a + c3*rcu_a + c2*rsq_a + c1*r_a + c0;
            acctyp smf = (acctyp)5.0*c5*rqu_a + (acctyp)4.0*c4*rcu_a
                       + (acctyp)3.0*c3*rsq_a + (acctyp)2.0*c2*r_a + c1;
            forcecoul = (numtyp)((acctyp)forcecoul*sme - (acctyp)ealpha*smf*r_a);
            ealpha = (numtyp)((acctyp)ealpha * sme);
          }
        }

        numtyp force = (force_lj + forcecoul) * r2inv;

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (EVFLAG && eflag) {
          if (rsq < coeff1[mtype].w) {
            numtyp e = coeff2[mtype].x*rexp - coeff2[mtype].y*r6inv - coeff2[mtype].z;
            energy += factor_lj * e;
          }
          if (rsq < cut_coulsq) {
            e_coul += ealpha;
            if (factor_coul > (numtyp)0.0)
              e_coul -= factor_coul*prefactor_erfa;
          }
        }
        if (EVFLAG && vflag) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
  } // if ii
  store_answers_q(f,energy,e_coul,virial,ii,inum,tid,t_per_atom,offset,eflag,
                  vflag,ans,engv);
}

// k_gcpm_efield: per-atom electric field from Gaussian-Coulomb interactions.
// Uses t_per_atom threads per atom (neighbor loop is strided); contributions
// from each thread are reduced with store_efield before writing the result.
// Full neighbor list assumed (no Newton partner).

__kernel void k_gcpm_efield(
                const __global numtyp4 *restrict x_,
                const __global numtyp4 *restrict coeff2,
                const int lj_types,
                const __global numtyp *restrict sp_lj_in,
                const __global int *dev_nbor,
                const __global int *dev_packed,
                __global acctyp *restrict dev_efield,
                const int inum, const int nbor_pitch,
                const __global numtyp *restrict q_,
                const numtyp cut_coulsq, const numtyp qqrd2e,
                const numtyp g_ewald, const numtyp rsmooth_sq,
                const acctyp c0, const acctyp c1, const acctyp c2,
                const acctyp c3, const acctyp c4, const acctyp c5,
                const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom, ii, tid, offset);

  __local numtyp sp_lj[8];
  int n_stride;
  local_allocate_store_efield();

  sp_lj[0]=sp_lj_in[0]; sp_lj[1]=sp_lj_in[1];
  sp_lj[2]=sp_lj_in[2]; sp_lj[3]=sp_lj_in[3];
  sp_lj[4]=sp_lj_in[4]; sp_lj[5]=sp_lj_in[5];
  sp_lj[6]=sp_lj_in[6]; sp_lj[7]=sp_lj_in[7];

  acctyp efx=(acctyp)0, efy=(acctyp)0, efz=(acctyp)0;

  if (ii < inum) {
    int nbor, nbor_end, i, numj;
    nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, offset, i,
              numj, n_stride, nbor_end, nbor);

    numtyp4 ix; fetch4(ix, i, pos_tex);
    numtyp qtmp; fetch(qtmp, i, q_tex);
    int itype = (int)ix.w * lj_types;

    if (qtmp != (numtyp)0.0) {
      for (; nbor < nbor_end; nbor += n_stride) {
        ucl_prefetch(dev_packed + nbor + n_stride);
        int j = dev_packed[nbor];
        numtyp factor_coul = (numtyp)1.0 - sp_lj[sbmask(j) + 4];
        j &= NEIGHMASK;

        numtyp qj; fetch(qj, j, q_tex);
        if (qj == (numtyp)0.0) continue;

        numtyp4 jx; fetch4(jx, j, pos_tex);
        int mtype = itype + (int)jx.w;

        numtyp delx = ix.x - jx.x;
        numtyp dely = ix.y - jx.y;
        numtyp delz = ix.z - jx.z;
        numtyp rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < cut_coulsq) {
          numtyp r = ucl_sqrt(rsq);
          numtyp r2inv = ucl_recip(rsq);

          numtyp grij = g_ewald * r;
          numtyp expm2_g = ucl_exp(-grij*grij);
          numtyp erf_g = (numtyp)1.0 - ucl_erfc(grij);

          numtyp aij = coeff2[mtype].w;
          numtyp aijr = aij * r;
          numtyp expa = ucl_exp(-aijr*aijr);
          numtyp erfa = (numtyp)1.0 - ucl_erfc(aijr);
          numtyp falpha = erfa - EWALD_F*aijr*expa;

          numtyp scale = qqrd2e / r;
          // factor_coul = 1 - special_coul; inline avoids inverted-conditional bug
          numtyp es = scale * (falpha - erf_g + EWALD_F*grij*expm2_g - factor_coul*falpha);
          numtyp eas = scale * (erfa - erf_g);

          if (rsq > rsmooth_sq) {
            acctyp r_a = (acctyp)r, rsq_a = (acctyp)rsq;
            acctyp rcu_a = r_a * rsq_a;
            acctyp rqu_a = rsq_a * rsq_a;
            acctyp sme = c5*rqu_a*r_a + c4*rqu_a + c3*rcu_a + c2*rsq_a + c1*r_a + c0;
            acctyp smf = (acctyp)5.0*c5*rqu_a + (acctyp)4.0*c4*rcu_a
                       + (acctyp)3.0*c3*rsq_a + (acctyp)2.0*c2*r_a + c1;
            es = (numtyp)((acctyp)es * sme - (acctyp)eas * smf * r_a);
          }
          es *= r2inv;

          efx += delx * qj * es;
          efy += dely * qj * es;
          efz += delz * qj * es;
        }
      } // for nbor
    } // if qtmp != 0
  } // if ii
  store_efield(efx, efy, efz, ii, inum, tid, t_per_atom, offset, dev_efield);
}
