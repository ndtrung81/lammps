# GCPM: `pair_gcpm.cpp` vs. the original Fortran code (`MD_water/`)
This section compares the LAMMPS pair style `src/GCPM/pair_gcpm.cpp` with the
original Fortran reference implementation of the Gaussian Charge Polarizable
Model (GCPM) in `MD_water/MD_water/` (files `force.f`, `init.f`, `main.f`,
`diel_cerf_hbond.f`, `pwat.inc`).

Reference: Paricaud, Predota, Chialvo, Cummings, *J. Chem. Phys.* **122**,
244511 (2005).

PairGCPM (the `gcpm` style) reproduces the Fortran Coulomb method and all the GCPM kernels:
  - smeared real-space Coulomb erf(α_ij·r)/r, no Ewald;
  - the per-pair reaction field reproduces the Fortran ferf physics (the pairwise
    charge-charge term 0.5·qi·qj·ferf·r² is identical; the self terms differ in
    bookkeeping — see difference 5 below);
  - exp-6 Buckingham dispersion;
  - the self-consistent induced-dipole solver with the Fortran warm-start guess.

This is validated by finite-difference F = −dU/dx matching to ~4–5 digits and
clean rigid-NVE conservation — i.e. the forces are self-consistent with the
energy I implemented from `force.f`.

Known differences from the Fortran (confirmed by the single-point comparison in
the last section of this file):
  1. Direct number-to-number comparison: DONE. One pwatin frame was taken through
  both codes (see "Single-point numerical comparison" below). Dispersion matches
  to ~5 digits — validating the geometry, units, charges and exp-6 kernel — and
  the electrostatics differ by ~4–6%, traced to differences 2 and 5 below. (The
  FD F = −dU/dx check above only proves self-consistency of the LAMMPS forces with
  the energy, not equality with the Fortran.)
  2. Cutoff convention (dominant electrostatic difference): the Fortran truncates
  ALL interactions between two molecules by their COM–COM distance
  (if (r2ij(i,j) <= rcut2)); PairGCPM truncates per atom–atom distance. Dispersion
  is immune (O sits ~at the COM); the charge sites (H, M) are ~1 Å off the COM, so
  the two conventions include/exclude different site pairs in the cutoff shell.
  3. Induced-dipole site placement: the Fortran places the induced dipole at the
  molecular COM (x0); PairGCPM places it on the M site (mu[3] != 0), ~0.2 Å away.
  Dipole–dipole distances are COM–COM vs M–M, so the polarization energy and the
  per-atom forces differ by a real (modest) amount.
  4. Reaction-field dielectric is a user input (eps_rf); the Fortran hard-codes
  78.4 (calcul_dielectric → DIELW = 78.4). You match it by passing eps_rf 78.4.
  5. Reaction-field self / intramolecular bookkeeping: the Fortran sums
  intermolecular Coulomb plus EXPLICIT self terms (ferf·d·d, q·mu·ferf); PairGCPM's
  per-pair RF obtains the self terms from intramolecular pairs. So neither
  "include" nor "exclude intramolecular" in LAMMPS reproduces the Fortran exactly.
  (The Fortran is also hard-coded 4-site water summing intermolecular only;
  PairGCPM is generic per-atom and depends on your special_coul setup.)
  6. Dispersion tail: the Fortran sets ercut = 0 and adds analytic exp-6 tail
  corrections (eset/pset); PairGCPM relies on LAMMPS pair_modify shift/tail. The
  short-range force is identical; the long-range dispersion correction differs.
  7. erf evaluation (negligible): the Fortran smeared Coulomb uses the
  Abramowitz–Stegun cerf approximation (~1e-7); PairGCPM uses
  MathSpecial::my_erfcx. A digit-level source of difference only.

To summarize: the physics kernels are consistent and FD-exact, and the direct
single-point comparison confirms the dispersion/geometry/units/charges to ~5
digits. The residual electrostatic difference is NOT a kernel bug — it is the
cutoff convention (molecular vs atomic), the induced-dipole site (COM vs M), and
the RF self / dispersion-tail bookkeeping. An exact electrostatic match would
require aligning the cutoff convention (see the last section).


# GCPMLong: `pair_gcpm_long.cpp` vs. the original Fortran code (`MD_water/`)

This section compares the LAMMPS pair style `src/GCPM/pair_gcpm_long.cpp` with the
original Fortran reference implementation of the Gaussian Charge Polarizable
Model (GCPM) in `MD_water/MD_water/` (files `force.f`, `init.f`, `main.f`,
`diel_cerf_hbond.f`, `pwat.inc`).

Reference: Paricaud, Predota, Chialvo, Cummings, *J. Chem. Phys.* **122**,
244511 (2005).

Both codes implement the **same GCPM physics** — Gaussian-charge smearing,
exp-6 Buckingham dispersion, and a self-consistent (iteratively solved) induced
point dipole per molecule. They differ in **how the long-range electrostatics is
treated** and in the **data model** (LAMMPS per-atom vs. Fortran per-molecule,
hard-coded 4-site water).

---

## 1. Headline difference: long-range Coulomb method

| | Fortran (`force.f`, `main.f`) | C++ (`pair_gcpm_long.cpp`) |
|---|---|---|
| Long-range method | Real-space cutoff **+ Onsager reaction field** (`ferf`). No Ewald. | **Full Ewald / PPPM in ALL three channels** (charge-charge, charge-dipole, dipole-dipole); requires `kspace_style pppm/dipole` when polarization is on (`ewaldflag = pppmflag = dipoleflag = 1`). |
| Smeared Coulomb | `cerf(v)/r` with `v = r/alpha(is,js)`; `cerf` is the Abramowitz–Stegun erf approximation. | `erf(alpha_ij·r)/r` via `MathSpecial::my_erfcx`. |
| Real-space form | full smeared term (no screening subtraction). | GCPM Gaussian **minus** the point-multipole Ewald long-range part in every channel: charges `falpha - erf(g_ewald·r) + EWALD_F·grij·expm2`; dipole tensor scalars `f,g` minus the b1/b2-equivalent erf forms (see status section below). |
| Reaction field | **always on**, dielectric `erf` **hard-coded to 78.4** (`diel_cerf_hbond.f:46`); `ferf = 2(erf-1)/((2·erf+1)·rc³)` (`main.f:324`). | **removed from gcpm/long** (errors if `eps_rf > 0`); the RF was the stand-in for the reciprocal dipole interactions, now supplied exactly by `pppm/dipole`. RF remains available in the base style `gcpm`. |
| Intramolecular exclusion | hard-coded intermolecular-only site loops. | by molecule ID inside the pair kernels (`factor_coul = 0` for same-molecule pairs); `neigh_modify exclude` is rejected because removed pairs could not cancel the k-space contributions. |

**The shared constant:** Fortran `constpi = 1.12837916702184` (commented
"sqrt(2)/pi" but actually `2/sqrt(pi)`) equals LAMMPS `EWALD_F`. In Fortran it
multiplies the physical Gaussian width; in C++ it appears in both the GCPM
Gaussian term and the Ewald-splitting term.

**Consequence:** the absolute Coulomb energy is **not** directly comparable
between the two unless k-space is disabled and the reaction field enabled. They
are alternative long-range treatments; the C++ adds Ewald that the Fortran never
had.

---

## 2. Gaussian-charge width convention (consistent)

The smearing widths are equivalent under a reciprocal convention, because C++
uses `erf(alpha_ij·r)` while Fortran uses `erf(r/alpha)`:

- C++ (`init_one`): `alpha_ij[i][j] = 1/sqrt(2·(si² + sj²))`, `si = sigmaM[i][i]`.
- Fortran (`main.f:114–128`):
  - O–O: `alpha(1,1) = 2·alphao = sqrt(2·(alphao² + alphao²))`
  - O–H: `alpha(1,2) = sqrt(2·(alphao² + alphah²))`
  - H–H: `alpha(2,2) = 2·alphah`

So `alpha_ij (C++) = 1 / alpha(is,js) (Fortran)`, and `sigmaM` ↔ the per-site
Gaussian widths `alphao`/`alphah`. **Consistent.**

---

## 3. Data model / geometry

| | Fortran | C++ |
|---|---|---|
| Water sites | Hard-coded 4 sites: 1 = M (negative charge), 2,3 = H, 4 = O (exp-6 center). | Generic LAMMPS atoms; rigidity via `fix rigid/small`. |
| Induced-dipole site | Molecular center, evaluated with the O/M Gaussian width `alpha(1,·)` (`force.f:206`). | The atom carrying `mu[i][3] != 0` (the M site, width `sigmaM`). |
| Neighbor loop | per-**molecule** list, inner loops over site pairs `is,js`. | per-**atom** half (CPU) / full (GPU) list. |
| Charges | `q(1,1)` on M (negative), `q(1,2)=q(1,3) = -q(1,1)/2` on H (`init.f:79–87`). | per-atom `q` from the data file. |
| Units | reduced (sigma, epsilon); `qqrd2e = 1`. | LAMMPS real units with `force->qqrd2e`. |

---

## 4. Term-by-term correspondence

### Dispersion (exp-6 Buckingham) — equivalent
- Fortran uses the raw Eq. (10) form (`force.f:464–475`):
  `1/(1-6/γ)·[6/γ·exp(γ(1-r)) - (1/r)⁶]`.
- C++ pre-converts to standard Buckingham `A·exp(-r/ρ) - C6/r⁶` in `init_one`
  with `A = 6ε·exp(γ)/(γ-6)`, `1/ρ = γ/σ`, `C6 = γε·σ⁶/(γ-6)`.
- Same potential. Fortran disables the energy shift (`ercut = 0`) and instead
  adds **analytic tail corrections** `eset`/`pset` (`main.f:332–344`); C++ uses
  LAMMPS `offset_flag`/`tail_flag`.

### Iterative induced-dipole solver — same algorithm
- Both compute the charge field once per step (constant) and recompute the
  dipole field each iteration; both converge on the max squared change of the
  dipole vector vs. a tolerance.
- Fortran **caches** the `T` tensor per pair (`txx(i,j)` …) between iterations
  (`force.f:225–230`, reused at `294–299`); C++ recomputes
  `compute_induced_efield()` each pass (with an MPI reverse-comm of the field).
- **Initial guess:** Fortran seeds `mxt` from the previous step's converged
  dipoles (warm start). C++ `first_polar` seeds `mu = α·E_q/qqrd2e` (E_p = 0) on
  the very first call only, then warm-starts thereafter — functionally
  equivalent after step 1.

### Smeared T tensor (Eqs. 6–7) — identical math, Ewald-split in C++
- C++ `compute_induced_efield()` starts from the same smeared scalars
  `f = erf - (rds + rds·r²/(6s²))·exp`, `g = erf - rds·exp`,
  `T = 3f·r⁻⁵·rr - g·r⁻³·I`, then subtracts the point-dipole Ewald
  long-range part (so real + reciprocal = full smeared sum over images):
  `f += -erf(Gr) + EWALD_F*Gr*(1 + 2(Gr)^2/3)*exp(-(Gr)^2)`,
  `g += -erf(Gr) + EWALD_F*Gr*exp(-(Gr)^2)` — the b1/b2 real-space kernels
  of `pair lj/cut/dipole/long`, generalized to smeared dipoles. The
  reciprocal part plus the per-iteration SCF field come from
  `kspace_style pppm/dipole`.
- Fortran `force.f:210–230` uses the same `fv`/`gv` and tensor (no Ewald),
  with `v = r/alpha` and `cerf`.

### Polarization energy — same
`U_pol = -½ Σ p_i·E_q_i` (Eq. 9): Fortran `force.f:371,386`; C++
`pair_gcpm_long.cpp:544`.

### Forces (charge–dipole, dipole–dipole) — same structure
- Same `(3f/r⁵)`, `g/r³` tensor and its radial derivatives.
- C++ additionally computes explicit **torques** and handles half/full-list +
  Newton bookkeeping (`doA`/`doB`). Fortran accumulates per-site forces
  `fxij(i,is)` and per-molecule `fmolx`.
- **Reaction-field self-force:** Fortran adds
  `fxij(i,is) += q(1,is)·mxt(i)·ferf` (`force.f:663–667`, noted as a fix to an
  earlier bug — "affects the torque"). In the C++ base style `gcpm` this is
  subsumed into the reaction-field machinery; `gcpm/long` has no RF (the
  analogous long-range self-consistency is handled exactly by the Ewald
  dipole self-field correction in the SCF loop, see status section below).

---

## 5. Reaction field — base style `gcpm` only

The reaction field now exists ONLY in the base style `gcpm` (`pair_gcpm.cpp`,
folded in per pair, matching the Fortran `ferf` physics — see
`pair_gcpm_summary.md`). `gcpm/long` **rejects** `eps_rf > 0`: the RF was the
stand-in for the reciprocal-space charge-dipole and dipole-dipole
interactions, which `kspace_style pppm/dipole` now supplies exactly (an RF on
top of a complete Ewald sum would double-count the dielectric response).

Historical note: an intermediate version of `gcpm/long` used Ewald for
charge-charge only plus a per-molecule Onsager RF
(`c_rf = 2·qqrd2e·(eps_rf-1)/((2·eps_rf+1)·rc³)` ≡ Fortran `ferf` with
`qqrd2e` baked in) for the dipole channels. That mix-and-match form was
replaced by the full long-range treatment documented in the status section
below.

---

## 6. Charge–charge energy expression

- **Fortran** (`force.f:520`):
  `eqq += 0.5·qi·qj·(erf(v)/r + 0.5·ferf·r²)` — smeared Coulomb **plus** the
  per-pair reaction-field energy.
- **C++** (`charge_charge`): Ewald-split energy `prefactor·(erfa - erf)` for the
  real-space part; the reciprocal part comes from `pppm/dipole`. (Total-energy
  bookkeeping across pair and kspace is stage 4 of the long-range upgrade and
  is NOT finished yet — see the status section below.)

---

## 7. Features present in only one code

**Only in C++:** MPI parallelism (forward/reverse comm of `mu` and the two
fields), GPU path (`gcpm/gpu`), explicit non-central virial tally
(`no_virial_fdotr_compute`, `vtally_force`), per-type mixing, restart I/O, and
the `pair_coeff` interface.

**Only in Fortran:** hard-coded TIP4P-style water geometry, RDF / h-bond /
density-profile / dielectric analysis (`diel_cerf_hbond.f`), analytic exp-6 tail
corrections, and the fixed 78.4 reaction-field dielectric.

---

## 8. Bottom line

The per-pair smeared-Coulomb, exp-6, and self-consistent-dipole **kernels are
equivalent** (modulo the `alpha ↔ 1/alpha_ij` width convention and the reduced
↔ real unit system). The one substantive physics change is that
`pair_gcpm_long.cpp` replaces the Fortran's **cutoff + reaction field**
electrostatics with a **full Ewald / PPPM long-range treatment of all three
channels** (charge-charge, charge-dipole, dipole-dipole) via
`kspace_style pppm/dipole`; the reaction field survives only in the base
style `gcpm`.

For a head-to-head numerical validation against the Fortran reference, use the
reaction-field style **`gcpm`** (`pair_gcpm.cpp`, no k-space) rather than
**`gcpm/long`** (`pair_gcpm_long.cpp`, Ewald/PPPM), because the two long-range
treatments are otherwise not directly comparable.

Note on class layout: `gcpm` (class `PairGCPM`, reaction field) is the base
class holding the shared GCPM machinery; `gcpm/long` (class `PairGCPMLong`,
Ewald/PPPM) derives from it and overrides only the Coulomb method.



# pair gcpm/long full long-range upgrade: staged plan and status

Goal: give `gcpm/long` the complete long-range charge-dipole and dipole-dipole
interactions using the charge+dipole-capable `pppm/dipole` merged from
upstream (PR #5059: three influence functions `greensfn_qq` / `greensfn_qmu` /
`greensfn` for the q-q, q-mu, mu-mu channels, with forces, torques, energy,
virial, per-atom terms, self-energy and slab correction).

7-stage plan agreed 2026-07-16. **Stages 1-5 are COMPLETE; stage 6
(validation) is next -- the user will supply specific test cases for it.**
Nothing is committed to git yet; all changes are in the working tree
(`src/kspace.h`, `src/KSPACE/pppm_dipole.{h,cpp}`,
`src/GCPM/pair_gcpm_long.{h,cpp}`, `examples/PACKAGES/gcpm/in.water_box.long`).

## Stage 1 (DONE) — field-only API on PPPMDipole

`src/KSPACE/pppm_dipole.{h,cpp}` gained two public entry points for SCF
polarization solvers; both ACCUMULATE (`+=`) the reciprocal-space E-field,
scaled by `qqrd2e*scale`, at all LOCAL atom positions (matching the pair's
`efield` convention):

- `compute_efield_from_charges(double **)` — field generated by the point
  charges through the **q-mu cross influence function** (`greensfn_qmu`) —
  the same channel `compute()` uses for the charge contribution to the field
  felt by dipoles, so SCF and final forces/torques are mutually consistent.
  Cost: 1 forward + 3 backward FFTs; charges are static during SCF, so it is
  called once per step.
- `compute_efield_from_dipoles(double **)` — field generated by the current
  `atom->mu` through the dipole-dipole influence function (3 fwd + 3 bwd
  FFTs); called once per SCF iteration. Includes the dipole self-image term
  (see stage 3) and all periodic/excluded-pair contributions.

Supporting pieces: mu-only / q-only density spreading (`make_rho_mu`,
`make_rho_charge`), field-only Poisson solves, a shared interpolator
(`interpolate_efield`), and three comm flags appended to the `kspace.h` enums
(`REVERSE_Q_ONLY`, `REVERSE_MU_ONLY`, `FORWARD_EFIELD`). The
`u{x,y,z}_brick_dipole` arrays are reused as scratch.

Validation hook kept in the code: env var `PPPM_DIPOLE_FIELD_CHECK=1` makes
`compute()` verify split-vs-combined fields (exact by FFT linearity).
Measured 2e-15 on 1/2/4 MPI ranks with `examples/dipole/in.charge_dipole`;
normal-path behavior bit-identical to pristine HEAD.

## Stage 2 (DONE) — Ewald-screened real-space dipole kernels in the pair

`pair_gcpm_long.cpp`: the dipole-field tensor (`compute_induced_efield`) and
the dipole-dipole force/torque loop (`polar`) subtract the point-dipole Ewald
long-range part from the smeared Eq. (7) scalars:

    f += -erf(Gr) + EWALD_F*Gr*(1 + 2(Gr)^2/3)*exp(-(Gr)^2)
    g += -erf(Gr) + EWALD_F*Gr*exp(-(Gr)^2)
    df -= (4/3)*EWALD_F*G^3*(Gr)^2*r^2*exp(-(Gr)^2)     (radial derivative)
    dg -= 2*EWALD_F*G^3*r^2*exp(-(Gr)^2)

(verified numerically to match the b1/b2 kernels of
`pair lj/cut/dipole/long`; derivatives FD-checked to 1e-10; splitting
identity exact). Both dipole loops now gate on `cut_coulsq` (the neighbor
list always covers it: `init_one` returns `MAX(cut_lj, cut_coul)`). The
charge-dipole kernel's exclusion convention was fixed from
`factor_coul * screened` to `screened - (1-factor)*full_smeared` (same
convention as the q-q loop), so excluded pairs cancel the point contribution
the reciprocal sum adds back.

## Stage 3 (DONE) — SCF wired to k-space; RF removed; exclusions by molecule

- `compute()` adds `compute_efield_from_charges(efield)` once per step (after
  the real-space reverse-comm); the SCF loop in `polar()` adds
  `compute_efield_from_dipoles(efield_pol)` each iteration, then the
  **self-field correction**: the reciprocal sum polarizes each dipole with
  its own Gaussian image, `E_self = -(4/3)*g_ewald^3/sqrt(pi)*qqrd2e*mu_i`,
  so `+eself*mu_i` is added back (`eself > 0`). Energy cross-check: this
  corresponds exactly to the `-musqsum*2*g^3/(3*sqrt(pi))` self-energy term
  in `PPPMDipole::compute()`.
- **Reaction field removed from gcpm/long**: `eps_rf > 0` errors out.
- `init_style` requires `kspace_style pppm/dipole` when polarization is on
  (`dipoleflag = 1` set in the constructor so `KSpace::pair_check()` accepts
  it), requires `atom->molecule_flag`, and ERRORS on any
  `neigh_modify exclude` (removed pairs could never cancel the k-space
  contributions).
- **Intramolecular exclusion by molecule ID inside the kernels**: GCPM is
  intermolecular-only, and the decks have no bond topology for
  special_bonds. `charge_charge()` and the q-p loop in `polar()` set
  `factor_coul = 0` when `molecule[i] == molecule[j] != 0`; the
  subtract-full-smeared convention then cancels the reciprocal-space
  intramolecular terms pair by pair. (No mu-mu intramolecular handling:
  GCPM has one polarizable site per molecule — documented assumption.)
- **Critical ordering fix**: `Verlet::setup()` / `Min::setup()` compute pair
  forces BEFORE calling `kspace->setup()`, so the first SCF of every run
  (and every `run 0`) would use uninitialized FFT tables (symptom: spurious
  g^3-scaled induced dipole on an isolated molecule; recip field silently
  ~0). Fixed by `PairGCPMLong::setup()` (invoked from `Force::setup()`,
  which runs first) calling `pppm_dipole->setup()` itself — idempotent, the
  integrator repeats it harmlessly.
- Example deck `examples/PACKAGES/gcpm/in.water_box.long` updated:
  `kspace_style pppm/dipole 0.0001`, `pair_style gcpm/long 1 0.0 12.0`
  (eps_rf 0), `neigh_modify exclude` removed.

### Stage 3 validation results (all on the 500-molecule water box unless noted)

1. **g_ewald independence** (the strong whole-physics test; any missing or
   mis-split term shifts with the splitting parameter): gewald 0.28 vs 0.36
   at accuracy 1e-6 -> forces agree to 2e-5 relative, mu to 2e-5.
   Caveat: need `g_ewald*rc >= ~3.3`; at 0.22*12 = 2.64 the real-space
   truncation (erfc level ~2e-4) dominates and shows up as ~1e-3 apparent
   gewald dependence.
2. **Zero net torque on converged dipoles** (max ~1e-11): at self-consistency
   `mu || E_total`, so total torque per dipole must vanish; this proves the
   pair + kspace torque channels are exactly consistent with the SCF field
   channels. (Useful permanent diagnostic.)
3. **Absolute 2-molecule check** (`data.2water`, 40 A box) vs the
   FD-validated base `gcpm` (cutoff 15, no RF, `neigh_modify exclude` — fine
   there, no kspace): mu matches to 7.6e-5, which is EXACTLY the predicted
   tinfoil-vs-vacuum boundary term `(4*pi/(3V))*M_box*alpha ~ 8.5e-5` — the
   only remaining difference is physical (pppm/dipole = metallic boundary).
4. **Single-molecule exclusion cancellation**: all electrostatics
   intramolecular -> residual mu 4e-5 and gewald-INDEPENDENT (PPPM grid
   error on the short excluded pairs, same class as any coul/long
   exclusion); residual site forces ~0.01 kcal/mol/A canceling within the
   molecule.
5. **Invariance**: net momentum 1.5e-12 (max|F| ~ 96); newton on/off x
   1/2/4 MPI ranks agree to 5e-11 in forces, 2e-13 in mu.
6. 20-step 4-rank dynamics runs cleanly; ~21 SCF iterations/step at the
   default tol 1e-5.

Testing notes: the SCF tolerance is hardcoded (`tol = 1.0e-5`, `maxiter = 50`
in the `PairGCPM` constructor); machine-precision invariance tests require
temporarily setting `tol = 1e-12`, `maxiter = 200` and rebuilding (default
tol limits cross-rank reproducibility to ~1e-3 in forces — SCF noise, not a
bug). Build used: `build-gcpm/` (PKG_GCPM + KSPACE + DIPOLE + RIGID +
MOLECULE).

## Stage 4 (DONE) — energy/virial bookkeeping

The Eq. (9) shortcut `-1/2 sum p.E_q` was dropped from `PairGCPMLong::polar()`
(it would double-count: `E_q` now contains the reciprocal field while
`pppm/dipole` tallies its own q-mu and mu-mu reciprocal energies + dipole
self-energy). The pair now tallies its real-space pieces explicitly:

- U_qq_real: unchanged (`charge_charge()`).
- U_qp_real: in the q-p force loop, `-pre2*pidotr` (interaction A) and
  `-pre2*pjdotr` (interaction B) per pair via `ev_tally()` with zero force
  arguments (energy only; the virial of these forces is already tallied by
  `vtally_force`). `pre2` contains the exclusion-corrected kernel, so the
  intramolecular cancellation against the k-space sum extends to energies.
- U_pp_real: in the dd force loop, `-(Aq*pir*pjr - qqrd2e*g_s*r3inv*pij)`
  per pair.
- Induction self-energy `+ qqrd2e*p_i^2/(2*alpha_i)` once per dipole atom
  (guarded against `alpha_pol == 0`).

At self-consistency U_qp(real+recip) + U_pp(real+recip+self) + U_ind =
-1/2 sum p.E_q, so the grand pair+kspace total still equals Eq. (9).

**Second real bug found (in pppm/dipole, upstream-relevant):**
`PPPMDipole::compute()` refreshed `musum_musq()` only when `atom->natoms`
changed, so the dipole self-energy correction `-musqsum*2g^3/(3 sqrt(pi))`
used the STALE dipoles from init. Harmless for permanent dipoles (rotation
preserves mu^2 — why upstream never saw it), but induced dipoles change
magnitude every step: the total energy was g_ewald-DEPENDENT by ~1%
(residual = exactly `(2/(3 sqrt(pi)))*qqrd2e*g^3*sum mu^2`). Fixed:
`musum_musq(0)` is called every `compute()` (one small allreduce; new
`errorflag` argument suppresses the no-dipoles error for the per-step
refresh since induced dipoles may transiently vanish).

Stage 4 validation (SCF tol temporarily 1e-12):
1. Total energy g_ewald-independence (500-molecule box, gewald 0.28 vs
   0.36, accuracy 1e-6): pe matches to 0.035 kcal/mol out of 9877
   (3.5e-6 relative; was 112.7 kcal/mol before the musqsum fix).
2. Absolute 2-molecule energy vs the FD-validated base `gcpm`:
   pe_long = pe_base - 0.0079 kcal/mol, EXACTLY the predicted
   tinfoil-vs-vacuum surface term `2*pi*M_box^2/(3V)*qqrd2e` (the only
   physical difference); g_ewald-independent to 2e-4 kcal.
3. **FD ground truth F = -dU/dx** (2-molecule, displace single atoms
   +/- 2e-4 A, fixed 48^3 mesh): M-site force -6.9497 vs FD -6.9497
   (1e-5 relative), H site 1.1e-5 relative; small-force probe agrees at
   the same ~6e-5 absolute level (PPPM grid noise). Forces are the exact
   gradient of the tallied total energy through the SCF.
4. Pressure g_ewald-independence: 0.004-0.07 atm out of ~8778 (<= 1e-5
   relative) for truncation-converged splittings (gewald 0.36 vs 0.40;
   grid-converged, checked at accuracy 1e-8). The 1.8 atm deviation at
   gewald 0.28 is real-space truncation (G*rc = 3.36), not bookkeeping.
   No virial changes were needed: `vtally_force` + fdotr (real) and the
   PR #5059 recip virial (kspace) were already complete.

## Stage 5 (DONE) — init/robustness details

- **SCF solver control from the input script**: `pair_style gcpm[/long]
  enable_polar eps_rf cut_buck [cutcoul] [polar/tol <tol>] [polar/maxiter <n>]`
  (defaults 1.0e-5 / 50, parsed in `PairGCPM::settings()`; inherited by the
  GPU variants). No more rebuild-to-tighten for validation runs:
  `polar/tol 1.0e-12 polar/maxiter 200` reproduces the temporary-build
  results bit-for-bit.
- **Restart settings** now persist `enable_polar`, `eps_rf` (with `enable_rf`
  rederived), `tol`, and `maxiter` (previously NOT saved -- they silently
  reset to constructor defaults on read_restart). Restart roundtrip
  validated bit-identical (pe to 13 digits, which also proves the restored
  tol since the default would differ in the 7th digit). NOTE: this changes
  the restart format for the GCPM styles; old GCPM restart files are not
  readable.
- **g_ewald auto-estimate with near-zero seed dipoles, characterized**: the
  water-box deck (accuracy 1e-4, auto) picks g_ewald = 0.219 / 12^3 grid --
  effectively charge-only values, because tiny induced-dipole seeds are
  invisible to the error model. Measured cost vs a tight reference
  (1e-6, gewald 0.36): max force error 0.117 kcal/mol/A = ~3x the
  charge-only estimate (0.041) -- usable, but optimistic.
  `PairGCPMLong::init_style()` now WARNS when the rms seed dipole is below
  0.05 e*A, recommending `kspace_modify gewald` (g_ewald*cut_coul >= ~3.3)
  and/or `mesh`. Restarts with converged bulk dipoles (~0.28 e*A) do not
  trigger it; weakly-polarized small systems may -- harmless. The example
  deck now pins `kspace_modify gewald 0.30` with an explanatory comment.

## Stage 6 — validation (user-specified acceptance test): DONE, gcpm/long PASSES

**Primary acceptance test (specified by the user 2026-07-16):** use
`examples/PACKAGES/gcpm/data.gcpm` with `in.gcpm` (500 GCPM waters from the
Fortran frame, `fix rigid/nvt/small` at 298 K, dt 0.5, rc 11.220684,
`pair gcpm 1 78.4 rc rc` + per-pair RF) as the reference, and run the
`gcpm/long` counterpart deck **`examples/PACKAGES/gcpm/in.gcpm.long`**
(same data file/thermostat/seed; `pair_style gcpm/long 1 0.0 rc`,
`kspace_style pppm/dipole 0.0001`, `kspace_modify gewald 0.30`, NO
`neigh_modify exclude`). Acceptance criterion: statistical consistency
with pair gcpm in temperature, energy, and pressure.

### Results (run 2026-07-16, 4 MPI ranks, 1000 steps, samples at steps 100-1000)

| quantity | gcpm/long (in.gcpm.long) | gcpm RF (log.16Jul26.gcpm.g++.4) |
|---|---|---|
| Temp [K]      | 291.8 +/- 5.5   | 294.6 +/- 5.9  |
| Press [atm]   | 76 +/- 274      | 82 +/- 286     |
| PotEng [kcal/mol] | -5135.6 +/- 25.5 | +16177 +/- 4193 (spurious, see below) |
| NVE etotal drift (500 steps, fix rigid/small) | +/- 0.07 kcal/mol | +17,000 kcal/mol |
| wall time (1000 steps, 4 ranks) | 64 s | 43 s |

- **Temperature and pressure: statistically consistent** (differences well
  inside one sigma of the fluctuations).
- **Structure: consistent.** RDF first peaks (long / RF / Fortran gofr.dat):
  O-O 2.75/2.60 vs 2.75/2.58 vs 2.79/2.76; intermolecular O-H
  1.85/1.33 vs 1.85/1.36 vs 1.83/1.39; H-H 2.41/1.33 vs 2.41/1.31 vs
  2.42/1.39. Long-vs-RF max |dg| for r > 2 A is 0.12 (gOO), 0.04 (gOH, gHH).
- **SCF cost**: ~11 iterations/step at tol 1e-5; gcpm/long is only ~1.5x the
  RF wall time on this box.
- Reference artifacts saved: `examples/PACKAGES/gcpm/log.16Jul26.gcpm.long.g++.4`
  and `rdf.long.txt` (compare with the user's committed
  `log.16Jul26.gcpm.g++.4` and `rdf.txt`).

### Finding: the RF reference's reported ENERGY is unusable on this deck
(pair gcpm defect, not a gcpm/long problem)

The RF run's PotEng climbs from -4808 to ~+16000 within 100 fs and then
random-walks with sigma ~4000 kcal/mol (reproduced bit-for-bit against the
user's committed reference log), while its temperature, pressure, induced
dipoles (M-site |mu| 0.174 +/- stable over 500 steps), and RDFs all stay
normal. Diagnosis chain:

1. **2x2 cross-evaluation** (each style single-pointed on each run's final
   config): gcpm/long rates BOTH configs normal (-5147 / -5164 kcal/mol);
   RF rates the long-config normal (-4744) but its OWN config +14949.
2. **Term isolation**: plain truncated smeared Coulomb (`gcpm 0 0.0`,
   polar off, RF off) already shows the split (+8597 vs -4538), so neither
   polarization nor the RF term causes it.
3. **Post-processing resum** of both final configs: atom-atom truncated sum
   +8059 vs -5096; adding the RF r^2 term +14751 vs -4997; **adding the
   missing shift constant -5375 vs -5409** (gap collapses from ~19,700 to
   34 kcal/mol); molecular (M-site distance) truncation -5348 vs -5409.

**Root cause** (`PairGCPM::charge_charge()`, src/GCPM/pair_gcpm.cpp:453-455):
the per-pair energy `qq*erfa/r + 0.5*qq*c_rf*r^2` has NO shift constant, so
it does not vanish at the cutoff: E(rc) = qq*(1 + B0/2)*qqrd2e/rc, up to
~66 kcal/mol per M-M pair. The Fortran never sees this because it truncates
by molecule COM-COM distance and GCPM molecules are neutral: the per-pair
constants sum to exactly zero over each molecule pair. With LAMMPS atom-atom
truncation, sites straddle the cutoff individually, so every crossing jumps
the reported energy (and NVE etotal) by O(10) kcal/mol. Forces are only
~2% discontinuous at rc (factor 1-B0, eps=78.4), which is why the RF
TRAJECTORY (structure, dipoles, pressure, temperature) remains essentially
correct -- the RF dynamics wanders across cutoff-shell pair-count
fluctuations that the energy tally, lacking the shift, amplifies ~1000x.

**Fix APPLIED (2026-07-18):** the per-type-pair shift constant
`e_shift_ij = qqrd2e*erf(alpha_ij*rc)/rc + 0.5*c_rf*rc^2` (times qq, bare
part scaled by factor_coul) is subtracted from ecoul in `charge_charge()`
when enable_rf is on, so E(rc) = 0. Implementation: new per-type-pair array
`e_shift_qq[i][j] = qqrd2e*erf(alpha_ij*rc)/rc` computed in `init_one()`
(erf(inf) = 1 covers the zero-width O-O pairs, which carry no charge and
never tally anyway); the RF part is applied inline as
`0.5*qq*c_rf*(r^2 - rc^2)`. The GPU kernels (`lib/gpu/lal_gcpm.cu`, both
plain and fast) apply the identical shift, computed in-kernel from `aij`
and `cut_coulsq` under `EVFLAG && eflag`. Not applied when enable_rf is
off (the bare-truncation debug mode keeps its historical unshifted
values), and gcpm/long is untouched (it overrides `charge_charge()` and
forbids RF). Validation (4 ranks, in.gcpm 1000-step deck vs the committed
log.16Jul26.gcpm.g++.4): temperature and pressure columns BIT-IDENTICAL
at every thermo step (trajectories unchanged); pe now -5120 +/- 30
kcal/mol (was a random walk from -4808 to +13163, sigma 7237), consistent
with gcpm/long's -5136 +/- 25 on the same deck. NVE (rigid/nve/small,
dt 0.5, 500 steps): etotal drift ~0.6 kcal/mol per 250 fs within a
+/-1 kcal/mol band (residual = the documented ~2% RF force discontinuity
at rc + SCF tol 1e-5). GPU runtime cross-check (after the post-suspend
nvidia_uvm reload restored CUDA): gcpm/gpu with `-pk gpu 1 neigh no`
(host neighboring; GPU neighbor builds reject neigh_modify exclude)
matches the CPU pe to ~2e-4 kcal/mol out of -5088 (rel ~4e-8, normal
GPU accumulation-order noise) at steps 0 and 10 of the NVE deck --
without the kernel shift the GPU pe would read -4808 (+280 kcal/mol).

**Acceptance verdict:** gcpm/long passes -- temperature, pressure, and
structure are statistically consistent with the RF reference; its absolute
energy is stable, NVE-conserving, and consistent with the shifted-RF /
molecular-truncation evaluation of the RF trajectory up to the expected
RF-vs-tinfoil systematic offset (~200 kcal/mol on this box). The energy
criterion cannot be scored against the RF style's own reported pe until the
shift fix lands. (2026-07-17: the user accepts the RF style's energy
inconsistency vs the Fortran as a cutoff-convention artifact; the energy
criterion for gcpm/long is instead scored directly against the Fortran
below, and passes.)

### Stage 6b (2026-07-17) — gcpm/long vs the Fortran, single point, same frame

Number-by-number comparison on the exact pwatin frame (the same frame behind
`examples/PACKAGES/gcpm/data.gcpm`; coordinate identity re-verified against a
fresh `frame.dat` -> converter round trip). Fortran side: `md_water_cmp` with
`dump_compare.f` extended to also export per-molecule induced dipoles
(`mu_frame.dat`, e*Ang; unit factor validated by the permanent dipole coming
out at exactly 1.855 D) and the exact molecular COM (`com_frame.dat`).
LAMMPS side: `gcpm/long` single point, PPPM accuracy 1e-6, `gewald 0.30`,
`polar/tol 1e-10` (`in.singlepoint.long` in the session scratchpad).

Headline energies (kcal/mol):

| term | Fortran | gcpm/long (M-site dipole) | gcpm/long (COM dipole) |
|---|---|---|---|
| dispersion (exp-6, no tail) | 1101.464 | 1101.441 | 1101.441 |
| permanent-charge electrostatics | -4773.10 (bare) / -4780.14 (RF+self) | -4778.91 | -4778.91 |
| E_pol | -1535.95 | -1421.20 | -1534.76 |
| total (uconf, no tail) | -5214.63 | -5098.66 | **-5212.23** |

- **The paper (Eq. 3) places the induced dipole at the molecular COM**, as
  the Fortran does; LAMMPS puts it on the M site (~0.2 A away). That
  placement is the ENTIRE 116 kcal/mol total-energy gap: with a 5-site
  variant data file (massless, chargeless type-4 site at the exact Fortran
  COM carrying the dipole flag, polarizability 1.444 and width 0.610; M
  keeps its charge, loses the dipole flag; NO code changes needed), the
  total agrees to **2.4 kcal/mol (0.046%)** and E_pol to 0.08%. The
  residual is the genuine RF-vs-Ewald long-range difference.
- Per-molecule comparison, COM-dipole variant vs Fortran (500 molecules):
  induced dipoles median |dmu|/|mu| 1.8%, cos(angle) 0.9999; net forces
  median 2.7%, cos 0.9998; torques about COM median 4.4%, cos 0.9995.
  With the M-site placement instead: dipoles 7.6% (|mu| mean 0.174 vs
  0.183 e*Ang), forces 15%, torques 16% -- all model difference, not bugs.
  (Compare the old RF-style force comparison in section D: median 72%,
  cos 0.857 -- most of that scatter was the atom-atom cutoff, which Ewald
  removes.)
- **Why the M-site dipoles are ~4.5% weaker (systematic, not
  frame-specific):** isolating placement within LAMMPS (M-site vs COM
  single points, same Ewald code), the per-molecule ratio
  |mu_M|/|mu_COM| = 0.956 +/- 0.043 with 436/500 molecules below 1 and
  direction unchanged (cos 0.999) -- a uniform shift, not outliers. The
  driver is the axial gradient of the local field in the H-bond network:
  the smeared charge field projected on the molecular axis is 2.2% weaker
  at M (0.0892 vs 0.0912 e/Ang^2, bare min-image sum). The two accepting
  H-bond hydrogens (~1.9 A behind the center) actually contribute slightly
  MORE at M; the deficit comes from the farther, opposing shells (> 2.5 A,
  net projection -0.063 vs -0.059). The SCF dipole-dipole feedback then
  roughly doubles the relative deficit (2.2% in E_q -> 4.4% in mu), and
  E_pol follows as ~mu^2 (0.956^2 = 0.914 vs the observed -1421/-1536 =
  0.925). Every molecule in the liquid sees the same axial field profile
  on the 0.2 A scale, so the effect persists for any equilibrated
  liquid-water configuration at this state point (the MD trajectory
  average |mu| 0.174 equals this frame's M-site value); its magnitude
  will vary with density/temperature as the H-bond structure changes.
  Analysis script: `MD_water/MD_water/mu_placement_analysis.py`.
- g_ewald independence at this frame: pe shifts 0.0015 kcal/mol between
  gewald 0.30 and 0.36 (grid re-tuned each time).
- SCF/torque-channel consistency: max residual torque on converged dipoles
  1e-9 (M-site) / 2e-9 (COM) kcal/mol.
- Tooling (all in `MD_water/MD_water/`): `dump_compare.f` (extended to
  write `mu_frame.dat` and `com_frame.dat`), `make_data5.py` (builds the
  5-site COM-dipole data file from `frame.dat` + `com_frame.dat`),
  `in.singlepoint.long` / `in.sp.com` (the two single-point decks; run
  them next to `data.gcpm` / `data.gcpm5`), and `compare_frame.py`
  (per-molecule dipole/force/torque comparison:
  `compare_frame.py <dump> <dipole-type>`).
- **Promoted to the example directory (2026-07-18):** `data.gcpm5` and
  `in.sp.com` now live in `examples/PACKAGES/gcpm/`, with the reference log
  `log.18Jul26.sp.com.g++.4` (4 ranks; pe -5212.225, matching the table
  above; 25 SCF iterations at `polar/tol 1e-10`).

**Stage 6 verdict (final): gcpm/long is validated against the Fortran
reference.** Dispersion matches to 2e-5, permanent-charge electrostatics to
0.12% (molecular truncation vs Ewald), and -- once the dipole is placed at
the COM as the paper prescribes -- polarization energy to 0.08%, total
energy to 0.046%, with per-molecule dipoles/forces/torques matching to a
few percent (RF-vs-Ewald residual). Open modeling decision: whether to keep
the M-site dipole placement (convenient 4-site decks, ~7% weaker induced
dipoles than the published model) or promote the 5-site COM-dipole data
layout, which reproduces the paper exactly with the existing pair style.

## Stage 7 — docs, examples, housekeeping

`doc/src/pair_gcpm.rst` (with `.. versionchanged:: TBD` for the gcpm/long
behavior change), reference logs, `make check`; add the still-untracked
`src/GCPM/pair_gcpm_long.{cpp,h}` (and friends) to git and to
`src/.gitignore`; update the three GCPM `.md` notes. Performance follow-up
(not in plan): the SCF costs 6 FFTs/iteration (~21 iters/step) — FFT-plan
reuse, caching the charge-density FFT, or convergence acceleration are the
targets. `gcpm/gpu` stays RF-only for now.



# Single-point numerical comparison (LAMMPS vs. Fortran)

The sections above compare the *equations/code*. This section records the
**number-by-number** validation: one real configuration is taken through both
codes and the energies/forces are compared directly.

## A. Building and running the Fortran reference

Legacy fixed-form Intel Fortran (a Visual Studio `.vfproj`). On Linux:

```bash
cd MD_water/MD_water
gfortran -O2 -fno-automatic -ffixed-line-length-132 -w \
  main.f init.f force.f Pos_alc.f pre_corrc_average.f \
  diel_cerf_hbond.f [dump_compare.f] -o md_water
./md_water        # reads pwat1.dat (unit 30), param (unit 12), pwatin (unit 40)
```

- **`-fno-automatic` is required**: `force` declares ~200 MB of local arrays
  (`xijt(mnm,mnm)` … with `mnm = 1372`) that Intel allocates statically; gfortran
  defaults to the stack and segfaults without it.
- `-ffixed-line-length-132`: Intel fixed form is 132 columns (gfortran defaults
  to 72); trailing comments past 132 are ignored.
- Inputs: `pwat1.dat` (NC, dt, nsteps, dipole iters/tol, IFLAG, rcut[sigma], …),
  `param` (eps, sigma, gamma, widths, geometry, polarizability), `pwatin` (the
  starting configuration; read when `IFLAG > 1`).

## B. Single-frame comparison tooling (added to `MD_water/MD_water/`)

To compare at one fixed configuration, the Fortran was instrumented to dump the
first frame in LAMMPS "real" units, and a converter builds the LAMMPS data file.

- `dump_compare.f` — new subroutine writing
  - `frame.dat`: box (Angstrom) + per molecule the 4 sites (O,H,H,M) with type,
    lab position (Angstrom) and charge (e);
  - `eforce.dat`: per-term energies (kcal/mol) and per-molecule net force
    (kcal/mol/Angstrom) and torque about the COM (kcal/mol).
- Minimal edits to the originals: `common /cmpene/` in `pwat.inc`; export lines
  in `force.f`; a pre-loop hook in `main.f` that runs `acalc`/`position`/`force`
  at the **exact pwatin configuration** (no predictor step) then `STOP`s.
  (Comment out the hook to run the full MD.)
- `pwatin_to_lammps.py` — converts `frame.dat` to `data.gcpm` (atom_style
  `hybrid full dipole sphere`; the M site, type 3, gets `mu = (0,0,0.01)` so
  `mu[3] != 0` flags it as the induced-dipole site).
- `in.lammps_compare` — the LAMMPS input (parameters + comparison notes).

```bash
./md_water_cmp                                 # -> frame.dat, eforce.dat
python3 pwatin_to_lammps.py frame.dat data.gcpm
lmp -in in.lammps_compare                      # compare thermo to eforce.dat
```

## C. Parameters (from `MD_water/param`, in LAMMPS real units)

| quantity | value |
|---|---|
| epsilon | 110 K = **0.218445 kcal/mol** (O–O exp-6) |
| sigma | **3.69 Å** |
| gamma (Buckingham) | **12.75** |
| M-site / H Gaussian width | **0.610** / **0.455** |
| polarizability | **1.444 Å³** (M site) |
| charges | **qM = −1.2226 e, qH = +0.6113 e** (GCPM, *not* TIP4P −1.04/0.52) |
| reaction-field dielectric | **78.4** (hard-coded in `diel_cerf_hbond.f`) |
| system | 500 molecules, cubic box **24.6554 Å**, cutoff **≈ 11.22 Å** |

The exp-6 dispersion acts only between O sites (type 1); H and M have
`epsilon = 0`, so all cross-term dispersion vanishes under geometric mixing.

## D. Results (500-molecule pwatin frame)

### Energies

| term | Fortran (`eforce.dat`) | LAMMPS `gcpm` | verdict |
|---|---|---|---|
| dispersion (O–O exp-6, no tail) | 1101.46 | 1101.44 | **match to ~5 digits** |
| pure smeared Coulomb (no RF) | −4773.10 | −4586.75 | ~4 % off |
| total config energy (`uconf`, no tail) | −5214.63 | — | offset (see below) |

The dispersion match to ~5 digits validates, end-to-end, the geometry
reconstruction (quaternion → site positions), the reduced→real unit conversions,
the GCPM charges and Gaussian widths, and the exp-6 kernel.

### Per-molecule net force

Per-atom forces cannot match directly (the Fortran puts the induced dipole at the
molecular COM, LAMMPS on the M site), so the comparable quantity is the **net
force per molecule** — the sum of the per-atom forces over each molecule. Across
all 500 molecules:

| metric | value |
|---|---|
| net-force component RMS difference | 3.50 kcal/mol/Å |
| \|F\| mean (Fortran / LAMMPS) | 8.50 / 9.19 |
| \|F\| max (Fortran / LAMMPS) | 30.2 / 29.6 |
| per-molecule \|ΔF\|/\|F\| (median / mean) | ~72 % / ~86 % |
| direction agreement, median cos(angle) | 0.857 (~31° off) |
| Σ net forces (momentum), both codes | ~0 ✓ |

In aggregate the forces agree — similar magnitude distributions, correlated
directions, and momentum conserved in both codes — but they scatter ~70 %
molecule by molecule, far more than the ~5 % energy difference. This is expected:
the net force on a molecule is a **small residual of large, nearly-canceling
pairwise contributions**, so the few-percent interaction-level differences (the
cutoff convention E.1 and the induced-dipole site E.4) are amplified. It is **not
a code bug** — the LAMMPS forces are FD-validated (`F = −dU/dx` to 4–5 digits on a
2-molecule test) and the Fortran forces are self-consistent with its own energy;
the two models differ in cutoff convention and dipole placement, and forces
expose that far more than energies do.

## E. Why the electrostatics (and forces) do not match exactly (model differences, not bugs)

1. **Cutoff convention.** The Fortran truncates *all* interactions between two
   molecules by their **COM–COM** distance (`if (r2ij(i,j) <= rcut2)`), whereas
   LAMMPS truncates per **atom–atom** distance. Dispersion is immune because the
   O site sits essentially at the COM, so the molecular and atomic cutoffs nearly
   coincide — hence its near-exact match. The charge sites (H, M) are offset from
   the COM by ~1 Å, so the two conventions include/exclude different site pairs in
   the cutoff shell. This is the dominant ~4 % effect on the charge–charge energy.
2. **Reaction-field self / intramolecular bookkeeping.** The Fortran sums
   *intermolecular* Coulomb plus *explicit* RF self terms; the LAMMPS per-pair
   reaction field obtains the self terms from *intramolecular* pairs, so neither
   "include" nor "exclude intramolecular" in LAMMPS reproduces it exactly (see
   section 5 above).
3. **Dispersion tail.** Fortran adds an analytic exp-6 tail (`eset`); LAMMPS uses
   `pair_modify shift/tail`. Short-range forces are identical.
4. **Induced-dipole site placement.** The Fortran places the induced dipole at the
   molecular COM (`x0`); LAMMPS places it on the M site (`mu[3] != 0`), ~0.2 Å
   away. Dipole–dipole distances are COM–COM vs M–M, so the polarization energy
   and — especially — the per-molecule forces differ by a real amount (water
   polarization forces are sensitive to the dipole position).

## F. Isolating individual terms

- **Dispersion only** — compare LAMMPS `evdwl` to `E_disp`.
- **Pure smeared Coulomb** — `pair_style gcpm 0 0.0 ${rc} ${rc}` (polar + RF off)
  with `neigh_modify exclude molecule/intra all`, then compare `ecoul` to
  `E_coul_smear` (`dump_compare.f` exports this term separately).
- **Per-molecule force / torque** — sum LAMMPS per-atom forces over each molecule
  and compare to `eforce.dat`. These agree only approximately because the Fortran
  places the induced dipole at the molecular COM while LAMMPS places it on the M
  site (~0.2 Å offset), on top of the cutoff-convention effect.

## G. To reach an exact electrostatic match

The cleanest next step is to make both codes use the **same cutoff convention**
— either add a molecule-distance cutoff mode to the LAMMPS pair style, or switch
the Fortran to atom–atom truncation — and align the RF self / intramolecular
bookkeeping. Not done here; the current tooling is sufficient to diagnose each
term.



# Self-diffusion coefficient: thermostat and mass pitfalls

The GCPM self-diffusion coefficient (Paricaud et al., Table IV: **D = 0.226
Ang^2/ps = 2.26e-5 cm^2/s** at T = 298 K, rho = 0.997 g/cm^3; Tables V/VI cover
supercooled/supercritical states) is a *dynamic* property, so it is sensitive to
two things that structure and energy are not: the per-molecule **mass** and the
**thermostat**. Two mismatches broke the initial LAMMPS-vs-paper comparison.

## 1. Per-atom mass (fixed in the data files)

`fix rigid/small` builds each body's mass, COM, and moment of inertia from the
**per-atom** `rmass` — and because `atom_style ... sphere` sets `rmass_flag`,
`rmass` comes from **column 12 of the data file**, NOT from the `mass` command
(`fix_rigid_small.cpp`: `if (rmass) massone = rmass[i]; else massone =
mass[type[i]]`). For a point particle (diameter/col-11 = 0) column 12 is read
verbatim as the mass (`atom_vec_sphere.cpp` `data_atom_post`: rmass is scaled by
`4/3 pi r^3` only when radius > 0). The original `data.gcpm` / `data.water_box`
had column 12 = 1.0 for every atom, so each water body weighed **4 amu instead of
18.015**. At fixed temperature velocities scale as sqrt(3kT/m), so molecules moved
sqrt(18/4) ~ **2.1x too fast** and D was inflated by ~2.1x. Fixed by writing the
per-type mass into column 12 (O 15.9994, H 1.008, M 1e-100; M stays ~massless as a
virtual site, and `1e-100` avoids the `rmass <= 0` "Invalid density" error).
See also the M-site diameter/DOF note (diameter must be 0 for a point dipole).

## 2. Thermostat: Langevin (LAMMPS input) vs. Evans-Hoover Gaussian (Fortran)

The reference `in.gcpm` used `fix rigid/small molecule langevin 298 298 100.0`.
A Langevin thermostat adds a stochastic drag gamma = 1/tau_damp (here 1/100 fs)
that **directly suppresses diffusion** — wrong tool for a transport coefficient,
and it pulls D in the opposite direction from the mass bug.

The Fortran uses an **Evans-Hoover Gaussian isokinetic thermostat**, which the
paper (p.8) states explicitly:

- Friction coefficient computed every step (`Pos_alc.f:87`):
  `alpha1 = Sum(F.p + tau.omega) / Sum(p^2 + I.omega^2)`
  (translational numerator `Sum F.p`, denominator `Sum p^2`; plus the rotational
  torque.omega / I.omega^2 terms).
- Applied as a deterministic friction in the Gear corrector
  (`pre_corrc_average.f:114-122`): `dp/dt = F - alpha1*p` (and the analogous
  `dL/dt = tau - alpha1*omega` for rotation). This holds the **total
  (translational + rotational) kinetic energy exactly constant** each step, is
  time-reversible, and perturbs the trajectory far less than Langevin.
- A hard velocity rescale to the target T every 4000 steps and at step 20
  (`pre_corrc_average.f:242-258`, `lambdt = sqrt(3*nmol*tstar/psqr)`) mops up
  drift in the isokinetic constraint.
- Integrator: Gear 4th-order predictor-corrector; run params from `pwat1.dat`:
  NC=5 -> 500 molecules (paper diffusion tables use N=256), reduced dt 6e-4 ->
  ~0.98 fs (tau = sigma*sqrt(m/eps) ~ 1.637 ps), 1e6 steps ~ 1 ns, T = 298.15 K,
  rho = 0.997 g/cm^3, cutoff 10 sigma capped at half-box, MSD -> `meansq.dat`,
  D = slope/6 (Einstein).

**Reproducing D in LAMMPS.** Gaussian isokinetic dynamics give the same transport
coefficients as NVE, so the correct surrogate is energy-conserving production:

- Simplest and most faithful: equilibrate with a thermostat, then run production
  in pure NVE (`fix rigid/nve/small molecule`).
- If a thermostat must stay on: `fix rigid/nvt/small molecule temp 298 298 Tdamp`
  (Nose-Hoover, deterministic) with a *loose* `Tdamp` — much closer to isokinetic
  than Langevin. Start at `Tdamp = 100*dt` (~100 fs at dt = 1 fs); for a
  transport measurement use 100-500 fs, or emulate the Fortran's periodic rescale
  with `fix temp/rescale` over NVE. Avoid tight coupling (< ~50 fs), which biases D.
- Match dt ~ 1 fs, rho ~ 0.997 g/cm^3 (data.gcpm is already ~0.998), run ~1 ns
  with a `compute msd`, and take D = slope/6. Compare to 0.226 Ang^2/ps.

**What D measures.** The paper's D (Tables V/VI) is the **translational
center-of-mass** self-diffusion coefficient. The Fortran MSD is built only from
the molecular COM `x0,y0,z0` (`pre_corrc_average.f:284-286`; site positions are
`x0(i)+xsite(i,j)`, so `x0` is the COM, not the O atom), and `D = slope/6` by the
Einstein relation. Rotation never enters D directly -- it only affects the
thermostat and the forces. The LAMMPS analogue is `compute msd` on the rigid-body
COM (or the `fix rigid` COM output), *not* a per-atom MSD.

## 3. How rotation is thermostatted: Evans isokinetic vs Nose-Hoover chains

Both codes model each water as a rigid body with full 3-D rotation (principal-axis
inertia + quaternion orientation), and **both thermostat the rotational
*momentum*, never the quaternion**. The quaternion is a pure kinematic follower:
it is integrated from the angular velocity/momentum, so the coupling chain is
always `thermostat -> rotational momentum -> quaternion`. Damping the rotational
momentum is exactly damping the rotational DOF; the orientation just tracks it.

**Fortran (`MD_water/`) -- Evans-Hoover Gaussian isokinetic.**
- Rotational state = body-frame angular velocity `(wx0,wy0,wz0)` + quaternion
  `(q10..q40)`, propagated by a Gear 4th-order predictor-corrector.
- The quaternion derivative is purely kinematic, `qdot = 1/2 Q(q).omega`
  (`pre_corrc_average.f:123-126`, the `cq1..cq4`); it just tracks `omega`.
- The thermostat is **one** friction `alpha1` shared by translation *and* rotation
  (`Pos_alc.f:78-87`):
  `alpha1 = Sum(F.p + tau.omega) / Sum(p^2 + I.omega^2)` -- numerator = translational
  power `F.p` + rotational power `tau.omega`; denominator = `2 KE_trans + 2 KE_rot`.
  That is a **single Gaussian constraint on the total (trans+rot) kinetic energy**.
- Applied to `omega` via the Euler equation in the corrector
  (`pre_corrc_average.f:117-122`):
  `wdot_x = (tau_x + wy.wz(Iyy-Izz))/Ixx - alpha1.wx0`, with the *identical*
  `alpha1` that damps translation (`pdot_x = fmol - alpha1.px0`, line 114). The
  thermostatted `omega` then feeds the quaternion ODE.
- Plus a periodic hard rescale every 4000 steps (and step 20), *separately* for
  translation (`lambdt`) and rotation (`lambdr`); the quaternion first-derivatives
  `q11..q41` are rebuilt from the rescaled `omega` (`pre_corrc_average.f:242-261`).

**LAMMPS `fix rigid/nvt/small` -- Nose-Hoover chains.**
- Rotational state = quaternion `quat` + angular momentum `angmom`, carried as the
  conjugate quaternion momentum `conjqm`; propagated by the no_squish/Miller
  symplectic rotation (`fix_rigid_nh_small.cpp:492-496`).
- `no_squish_rotate` advances `quat` from `conjqm` -- again the quaternion just
  follows the momentum.
- The thermostat scales the *rotational momentum*: `conjqm *= scale_r`,
  `scale_r = exp(-dtq.eta_dot_r)` (`:484-487`), where `eta_dot_r` comes from a
  **separate rotational NH chain** driven by the rotational DOF count `nf_r` and
  rotational KE `akin_r` (`:360`, `:305`). Translation is scaled independently by
  `scale_t` from its own chain (`nf_t`, `akin_t`).

**Side by side:**

| | Fortran `MD_water` | LAMMPS `rigid/nvt/small` |
|---|---|---|
| Thermostat | Evans Gaussian **isokinetic** (deterministic, KE held exactly, ~NVE transport) | **Nose-Hoover chain** (canonical NVT, KE fluctuates) |
| Trans/rot coupling | **one** friction `alpha1` over **total** KE | **two separate** chains (`nf_t`/`akin_t` vs `nf_r`/`akin_r`) |
| Rotational variable damped | body-frame `omega` | angular momentum (`conjqm`) |
| Rotation propagator | Gear PC on `omega` + quaternion ODE | no_squish (Miller) on `conjqm` |
| Quaternion role | kinematic follower of `omega` (never thermostatted) | kinematic follower of `conjqm` (never thermostatted) |
| Drift cleanup | periodic hard rescale (`lambdt`, `lambdr`) | absorbed continuously by the chain |

**Bottom line for D.** The two thermostats are structurally similar (both damp the
rotational momentum and let the quaternion follow), but the Fortran holds trans+rot
KE *jointly and exactly* (isokinetic ~ NVE dynamics), whereas `rigid/nvt/small`
runs *two independent canonical chains*. That is why NVE production is the faithful
surrogate for reproducing D, and `rigid/nvt/small` with a loose `Tdamp` is the
next-best -- deterministic like Evans, but canonical rather than isokinetic and
split across two chains.

## 4. How D is computed in the `Self-Diffusion-Study` folder (signac workflow)

`src/GCPM/Self-Diffusion-Study/` is a signac/row project that reproduces the
paper's Tables V/VI. How it computes D, and two pitfalls in that pipeline:

**D is a molecular center-of-mass diffusivity (not per-atom).** LAMMPS itself
computes no MSD -- the run (`src/files/in.simple_water`: `pair_style gcpm 1 78.4`,
`fix rigid/small ... langevin`, `timestep 0.5`) only dumps a trajectory
(`dump 2 all dcd 10000 trajectory.dcd`). D is computed in post-processing by
`src/data.py::_calculate_msd` (MDAnalysis): each molecule's position is the
mass-weighted average of its O + 2 H atoms, `np.average([O,H,H],
weights=(16,1.01,1.01))` (`data.py:85-87`; the massless M site, type 3, is
excluded), then a sliding-window MSD over molecules, linear fit of MSD vs. time
between 10 % and 50 % of `max_lag`, `D = slope/6` (3-D Einstein, `data.py:104-120`).
So D is the translational COM coefficient -- consistent with the Fortran (`x0` = COM)
and the paper -- **not** an atom-based MSD.

**PITFALL 1 -- frame-time mismatch (D comes out ~2x too small).** The DCD is dumped
every `10000 * 0.5 fs = 5 ps`, but `data.py:79` hard-codes `dt = 10000/1000 = 10 ps`.
The MSD time axis is stretched 2x, so the fitted slope and hence D are **half** the
true value. This partially *cancels* the separate per-atom-mass bug (col-12 = 1.0 ->
body mass 4 amu instead of 18 -> D inflated ~2.1x, section 1 above), so a
plausible-looking D in this folder can be two compensating errors. Fix `dt` to match
`dump_every * timestep` (here 5 ps), and correct the col-12 masses, before trusting
the numbers.

**PITFALL 2 -- COM built from wrapped coordinates, then unwrapped.** The COM is
averaged from the raw (wrapped) atom positions and only unwrapped afterward at the
COM level (`data.py:85-95`). LAMMPS DCD stores wrapped coordinates, so a molecule
straddling a periodic boundary in a frame yields a corrupted COM (O and its H's
averaged across the box). Usually minor, but a latent bug for rigid molecules that
can sit on a boundary. The O/H pairing (`hs.positions[::2]` / `[1::2]`) also assumes
atoms are ordered O,H,H,M per molecule and returned in matching molecule order --
true for these data files but fragile.

## 5. Recommended recipe to make D consistent with Tables V/VI

Combining sections 1-4, here is the checklist to turn the `Self-Diffusion-Study`
pipeline into a paper-comparable D. The first three items remove the *dynamical
biases*; the last two set expectations for the *residual* gap.

**(a) Correct the per-atom masses (data file).** Column 12 (sphere `rmass`) must be
the real per-type mass: O 15.9994, H 1.008, M 1e-100 (kept ~massless, avoids the
`rmass <= 0` error). Column-12 = 1.0 makes each body 4 amu instead of 18 and inflates
D ~2.1x (section 1). This is mandatory.

**(b) Thermostat -> NVE for production.** Equilibrate with
`fix rigid/nvt/small molecule temp T T Tdamp` (loose `Tdamp`, ~100-500 fs), then run
*production* under NVE with **`fix rigid/nve/small molecule`** (not plain `fix rigid`,
which lumps atoms into few bodies by group; not `langevin`, whose drag suppresses D).
This is the faithful surrogate for the Fortran's Evans isokinetic dynamics
(sections 2-3). Measure D only on the NVE leg.

**(c) Use the correct analysis `dt` = frame spacing (NOT the MD timestep).** In
`data.py`, `dt` scales the MSD time axis and D ~ 1/dt, so it must equal
`dump_every * timestep`. With `dump ... dcd 10000` and `timestep 0.5`, that is
**5 ps per frame** -- not the current hard-coded 10 ps (D 2x too small) and *never*
the 0.5 fs timestep (D 10^4x too big). If you change the dump frequency or timestep,
update `dt` to match the product. Also prefer a finer dump (e.g. every 1000 steps =
0.5 ps) so ~1 ns yields ~2000 frames and a clean linear MSD region instead of ~200.

**(d) Finite-size correction (the dominant residual, ~10-20%).** MD self-diffusion in
PBC is box-size dependent:
`D_inf = D_PBC + 2.837297 * kB*T / (6*pi*eta*L)` (Yeh-Hummer), always making the raw
`D_PBC` *smaller* than the infinite-system value. For N=256 water (~19-20 A box) this
is a ~10-20% upward correction. Check whether Tables V/VI report raw-PBC or
size-corrected D and match the convention (and box size / N) before comparing.

**(e) Statistics.** A single 1 ns trajectory gives D with ~10% scatter (worse for the
low-T supercooled points in Table V, where the diffusive regime sets in late). Use
multiple seeds or longer runs, and fit the MSD only in the linear (diffusive) region.

**Expected outcome.** With (a)-(c) applied and (d)-(e) accounted for, D should agree
with Tables V/VI to within finite-size + statistical error (order ~10-20%), not
bit-for-bit. If (a) and (c) are both wrong in the original folder, note they *partly
cancel* (mass inflates ~2.1x, `dt`=10 ps deflates 2x), so a plausible-looking D there
is two compensating errors, not a validated result.

## 6. Results after the mass + thermostat fixes (2026-07): the remaining
## supercritical bias is a trajectory-unwrapping artifact, not dynamics

With the per-atom masses corrected (section 1) and production switched to
`fix rigid/nvt/small` (section 2), the `Self-Diffusion-Study` workflow
(N = 256 molecules per the signac statepoints) now gives:

| T (K) | rho (g/cm^3) | D_calc (Ang^2/ps) | D_ref (Ang^2/ps) | Error (%) |
|------:|-------------:|------------------:|-----------------:|----------:|
|   273 |        1.000 |          0.085595 |           0.1269 |     -32.5 |
|   313 |        0.996 |          0.267661 |           0.2468 |      +8.5 |
|   333 |        0.983 |          0.380537 |           0.4194 |      -9.3 |
|   343 |        0.978 |          0.464706 |           0.4999 |      -7.0 |
|   673 |        0.800 |          2.860751 |           3.5050 |     -18.4 |
|   673 |        0.600 |          3.278022 |           5.3230 |     -38.4 |
|   673 |        0.400 |          5.424735 |           8.3620 |     -35.1 |
|   673 |        0.200 |          8.300653 |          16.4900 |     -49.7 |
|   673 |        0.100 |         14.488891 |          32.5000 |     -55.4 |
|   873 |        0.800 |          2.994687 |           4.2360 |     -29.3 |
|   873 |        0.600 |          4.062095 |           6.3090 |     -35.6 |
|   873 |        0.400 |          5.497201 |          10.5900 |     -48.1 |
|   873 |        0.200 |          8.588294 |          21.8500 |     -60.7 |
|   873 |        0.100 |         14.514719 |          43.9500 |     -67.0 |

(D_ref are the paper's own GCPM simulation values -- Table VI for the
atmospheric points and Table VII for the supercritical isotherms, both N = 256
NVT MD -- so this comparison is implementation-vs-implementation, with no
model error in the gap. For the record, the pre-fix Langevin-era numbers in
`Self-Diffusion-Study/msd.txt` were -37 to -50% at ambient and -60 to -87%
supercritical, so the mass/thermostat fixes clearly moved things the right way.)

**Reading the table.** The ambient-liquid points (313-343 K) now agree to
within +/- 10%, i.e. within the expected statistics for a single ~1 ns
trajectory (section 5e). The supercooled 273 K point and, especially, the
supercritical isotherms are still systematically low, and the error *grows as
the density drops* (-18% at rho = 0.8 down to -67% at rho = 0.1). That
density trend is not a dynamics problem -- it is the MSD post-processing
failing, specifically the jump-unwrapping step in `data.py`.

**The artifact.** The DCD trajectory stores *wrapped* coordinates and
`_calculate_msd` unwraps them post hoc by minimum-image jump correction
(`jumps = np.round(diff/box)`). That reconstruction is only valid while the
*true* displacement of a molecule between consecutive frames stays well below
L/2. With frames every 5 ps (dump every 10000 steps x 0.5 fs) and the large,
fast supercritical D, the per-frame displacement is comparable to the half
box, the round() picks the wrong image, and every too-large jump is folded
back into [-L/2, L/2]. The measured "MSD" then saturates at the slope of a
random walk whose steps are box-limited, giving an apparent

    D_sat = L^2 / (24 * dt_frame)

(per component the folded jump has variance ~L^2/12, three components, Einstein
factor 6). This ceiling depends only on box size and frame spacing -- not on
temperature, not on the physics. Computing it for each state point (N = 256):

| T (K) | rho | L (Ang) | sigma_1D/frame (Ang) | P(jump > L/2) | D_sat | D_calc | D_ref |
|------:|----:|--------:|---------------------:|--------------:|------:|-------:|------:|
|  673 | 0.8 |   21.2 |                 5.9 |         7% |  3.8 |   2.86 |  3.50 |
|  673 | 0.6 |   23.4 |                 7.3 |        11% |  4.6 |   3.28 |  5.32 |
|  673 | 0.4 |   26.7 |                 9.1 |        14% |  6.0 |   5.42 |  8.36 |
|  673 | 0.2 |   33.7 |                12.8 |        19% |  9.5 |   8.30 | 16.49 |
|  673 | 0.1 |   42.5 |                18.0 |        24% | 15.0 |  14.49 | 32.50 |
|  873 | 0.8 |   21.2 |                 6.5 |        10% |  3.8 |   2.99 |  4.24 |
|  873 | 0.6 |   23.4 |                 7.9 |        14% |  4.6 |   4.06 |  6.31 |
|  873 | 0.4 |   26.7 |                10.3 |        19% |  6.0 |   5.50 | 10.59 |
|  873 | 0.2 |   33.7 |                14.8 |        25% |  9.5 |   8.59 | 21.85 |
|  873 | 0.1 |   42.5 |                21.0 |        31% | 15.0 |  14.51 | 43.95 |

(sigma_1D/frame = sqrt(2 * D_ref * dt_frame) is the true rms one-component
displacement between frames; P is the per-component per-frame probability of
a mis-unwrapped jump.)

Two smoking guns:

1. **D_calc pins to D_sat, not to D_ref, once rho <= 0.4.** At rho = 0.1 the
   ceiling is 15.0 and both isotherms measured ~14.5; at rho = 0.2 the ceiling
   is 9.5 and both measured ~8.4-8.6; at rho = 0.4 the ceiling is 6.0 and both
   measured ~5.4-5.5. The measured values track L^2 (i.e. rho^(-2/3)), exactly
   as the artifact predicts.
2. **D_calc is temperature-independent where the true D is not.** 14.489 vs
   14.515 at 673 vs 873 K (rho = 0.1), while the reference values differ by
   35% (32.5 vs 43.95). A real diffusion measurement cannot produce that; a
   box-size-limited one must.

At rho = 0.6-0.8 the clipping is partial (P ~ 7-14%): D_calc sits below both
D_ref and D_sat, biased low by the fraction of folded jumps. The ambient
points are immune (sigma_1D/frame ~ 1.5 Ang vs L/2 ~ 9.9 Ang), which is why
they came out clean -- and their +/- 10% agreement also confirms the old
frame-time bug (section 4, PITFALL 1) is no longer in play, since that one
would show up as a uniform -50% everywhere.

**The 273 K point (-32.5%) is a different story.** Unwrapping is safe there
(0.9 Ang/frame). This is the supercooled regime: the diffusive MSD regime sets
in late, single-trajectory scatter is at its worst, and the reference itself
is soft -- the paper's Table V (250/273 K isochores) reports D = 0.1182 at
273 K / 1.00 g/cm^3 while its Table VI (atmospheric) reports 0.1269 at the
same nominal state point, an 8% internal spread. Treat this point with longer
runs and multiple seeds (section 5e) before reading anything into it; the
Nose-Hoover-vs-NVE production choice (section 5b) also matters most here.

**(f) Fix: never post-hoc-unwrap wrapped frames.** In order of preference:

1. **Dump unwrapped coordinates** -- `dump ... custom N file id mol type xu yu zu`
   (image-flag-based, exact for any frame spacing) instead of DCD, and build
   the COM from `xu/yu/zu` directly. This also kills PITFALL 2 of section 4
   (COM assembled from wrapped atoms across a boundary) in one stroke, and it
   frees the frame spacing to be chosen purely for statistics.
2. Or compute the MSD inside LAMMPS: `compute com/chunk` on molecules (LAMMPS
   uses image flags internally) + `compute msd/chunk`, or the COM output of
   `fix rigid/small`, with no trajectory post-processing at all.
3. If wrapped DCD must be kept, the frame spacing must satisfy
   `sqrt(2 * D_expected * dt_frame) < L/10` (mis-unwrap probability < 1e-6).
   At 873 K / 0.1 g/cm^3 that means dt_frame < ~0.2 ps, i.e. dump every
   <= 400 steps at 0.5 fs -- 25x finer than the current 10000.

**Changes to `data.py` required by option 1.** Switching the dump to
`xu yu zu` without touching the analysis re-corrupts the data; three edits
must land together:

1. **Delete the jump-correction block** (the `jumps = np.round(diff/box)`
   loop, `data.py:90-95`). Applied to already-unwrapped coordinates it does
   active harm: any genuine per-frame displacement larger than L/2 looks
   like a wrap "jump" and gets folded back into the box -- the same D_sat
   clipping artifact, now self-inflicted on clean data. The whole point of
   `xu/yu/zu` is that LAMMPS unwraps exactly from integer image flags.
2. **Change the trajectory loader** -- a `dump custom` file is not DCD:

   ```python
   u = mda.Universe(
       job.fn("subset.lammps"), job.fn("trajectory.lammpstrj"),
       format="LAMMPSDUMP",
       topology_format="DATA",
       atom_style="id type x y z resid charge dip radius rmas omega torque",
       lammps_coordinate_convention="unwrapped",
   )
   ```

   `lammps_coordinate_convention="unwrapped"` makes MDAnalysis read the
   `xu yu zu` columns (and error out if they are missing -- a useful guard).
   On the LAMMPS side use `dump ... custom N trajectory.lammpstrj id type
   xu yu zu` **plus `dump_modify ... sort id`**: the COM pairing
   (`hs.positions[::2]` / `[1::2]`) relies on atoms arriving in data-file
   order.
3. **Set `dt` to the new frame spacing.** `data.py:79` hard-codes it; it must
   equal `dump_every * timestep / 1000` ps. Since unwrapped coordinates make
   fine spacing safe, dump more often for statistics (e.g. every 1000 steps
   = 0.5 ps) -- but then this line must change in the same commit as the
   dump line, or the unwrap artifact is traded for the old factor-of-N
   time-axis error (section 4, PITFALL 1).

Everything else -- the O+2H mass-weighted COM, the sliding-window MSD, the
10-50% fit window, `D = slope/6` -- carries over unchanged. (Bonus: the
wrapped-COM pitfall of section 4 disappears too, since the COM is now built
from unwrapped atom positions.) Option 2 instead replaces most of `data.py`:
no MDAnalysis or trajectory at all, just read the `msd/chunk` output file
and fit the slope.

**Expected outcome after (f).** The supercritical points should collapse to
the same +/- 10% band as the 313-343 K liquid points, since the reference is
the same code-and-model lineage at the same N. Residual systematic gaps beyond
that would then be worth attributing to real protocol differences (Nose-Hoover
vs Evans isokinetic production, cutoff convention at low density where the
Fortran caps rcut at L/2 with 10 sigma while LAMMPS keeps ~11.2 Ang) -- but
none of those can be assessed until the measurement itself is valid.



