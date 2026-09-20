#!/usr/bin/env python3
"""Net charge inside the interaction range, atom-atom vs COM-COM truncation.

Brute-force census on a GCPM 5-site data file (default data.gcpm5), used to
support the "Why `com` keeps every pair of an interacting molecule pair" section
of src/GCPM/gcpm_lammps_vs_fortran.md.

For every site i it sums the charges of the sites that the pair style would let
i interact with (intramolecular pairs excluded, as in the example decks):

  atom-atom : sites j with |x_j - x_i|   < rc   -> orientation-dependent fragments
  COM-COM   : sites j with |R_J  - R_I|  < rc   -> whole molecules, hence neutral

Usage: python3 net_charge_census.py [data file] [rc]
"""

import sys
import numpy as np

fname = sys.argv[1] if len(sys.argv) > 1 else "data.gcpm5"
rc = float(sys.argv[2]) if len(sys.argv) > 2 else 11.220684

typ, pos, mol, q, mass = [], [], [], [], []
box = None
started = False
for line in open(fname):
    if "xlo xhi" in line:
        s = line.split()
        box = float(s[1]) - float(s[0])
    if line.startswith("Atoms"):
        started = True
        continue
    s = line.split()
    if not started or len(s) != 12:
        continue
    typ.append(int(s[1]))
    pos.append([float(v) for v in s[2:5]])
    mol.append(int(s[5]))
    q.append(float(s[6]))
    mass.append(float(s[11]))

typ = np.array(typ); pos = np.array(pos); mol = np.array(mol)
q = np.array(q); mass = np.array(mass)
n = len(q); nmol = mol.max()
print(f"{fname}: {n} sites, {nmol} molecules, L = {box:.6f} Ang, "
      f"rc = {rc} (L/2 = {box/2:.3f})")

# per-molecule center of mass and net charge; then neutralize each molecule
# exactly, so the residual reported below is the truncation, not the data
# file's 8-decimal rounding of the site charges.
molq = np.zeros(nmol+1)
com = np.zeros((nmol+1, 3))
tot = np.zeros(nmol+1)
for i in range(n):
    molq[mol[i]] += q[i]
    com[mol[i]] += mass[i]*pos[i]
    tot[mol[i]] += mass[i]
com[1:] /= tot[1:, None]
print(f"molecule net charge from the file: max |sum q| = {np.abs(molq[1:]).max():.1e} e")
qx = q.copy()
for mm in range(1, nmol+1):
    idx = np.where(mol == mm)[0]
    qx[idx[typ[idx] == 3][0]] -= molq[mm]      # put the residual on the M charge site

mic = lambda d: d - box*np.round(d/box)
comi = com[mol]
Qa = np.zeros(n)
Qc = np.zeros(n)
for i in range(n):
    d = mic(pos - pos[i])
    r = np.sqrt((d*d).sum(1))
    dc = mic(comi - comi[i])
    R = np.sqrt((dc*dc).sum(1))
    inter = mol != mol[i]                      # neigh_modify exclude molecule/intra all
    Qa[i] = qx[inter & (r < rc)].sum()
    Qc[i] = qx[inter & (R < rc)].sum()

print("\nnet charge Q_i inside the interaction range of a site:")
for name, Q in (("atom-atom", Qa), ("COM-COM", Qc)):
    print(f"  {name:9s}  mean {Q.mean():+.4e} e   RMS {np.sqrt((Q**2).mean()):.4e} e"
          f"   max |Q| {np.abs(Q).max():.4e} e")

# charge-weighted bias: this is what shows up as an energy, not as noise
bias = 0.5*332.06*(qx*Qa).sum()/rc
print(f"\ncharge-weighted <q_i Q_i> (atom-atom) = {(qx*Qa).mean():+.4f} e^2")
print(f"monopole energy estimate 0.5*332.06*sum_i q_i Q_i / rc = {bias:+.1f} kcal/mol")

# molecule-pair census under atom-atom truncation
sites = [np.where(mol == mm)[0] for mm in range(1, nmol+1)]
cm = com[1:]
D = mic(cm[:, None, :] - cm[None, :, :])
R = np.sqrt((D*D).sum(-1))
nin = nsplit = nout = 0
for a in range(nmol):
    for b in range(a+1, nmol):
        if R[a, b] > rc + 4.0:
            nout += 1
            continue
        dd = mic(pos[sites[b]][None, :, :] - pos[sites[a]][:, None, :])
        c = int((np.sqrt((dd*dd).sum(-1)) < rc).sum())
        if c == 0:
            nout += 1
        elif c == len(sites[a])*len(sites[b]):
            nin += 1
        else:
            nsplit += 1
npair = nmol*(nmol-1)//2
print(f"\nmolecule pairs under atom-atom truncation (total {npair}):")
print(f"  all site pairs inside rc  : {nin:7d} ({100*nin/npair:.1f} %)")
print(f"  split, some in some out   : {nsplit:7d} ({100*nsplit/npair:.1f} %)")
print(f"  none inside rc            : {nout:7d} ({100*nout/npair:.1f} %)")
print(f"  split / fully inside      : {nsplit/nin:.2f}")
