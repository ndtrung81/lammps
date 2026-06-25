# GCPM: `pair_gcpm.cpp` vs. the original Fortran code (`MD_water/`)
This section compares the LAMMPS pair style `src/GCPM/pair_gcpm.cpp` with the
original Fortran reference implementation of the Gaussian Charge Polarizable
Model (GCPM) in `MD_water/MD_water/` (files `force.f`, `init.f`, `main.f`,
`diel_cerf_hbond.f`, `pwat.inc`).

Reference: Paricaud, Predota, Chialvo, Cummings, *J. Chem. Phys.* **122**,
244511 (2005).

PairGCPM (the `gcpm` style) reproduces the Fortran Coulomb method and all the GCPM kernels:
  - smeared real-space Coulomb erf(α_ij·r)/r, no Ewald;
  - the per-pair reaction field is exactly the Fortran ferf bookkeeping;
  - exp-6 Buckingham dispersion;
  - the self-consistent induced-dipole solver with the Fortran warm-start guess.

This is validated by finite-difference F = −dU/dx matching to ~4–5 digits and
clean rigid-NVE conservation — i.e. the forces are self-consistent with the
energy I implemented from `force.f`.

To ensure that the code in `pair_gcpm.cpp` is completely consistent with the Fortran code,
further tests are needed:
  1. Need a direct number-to-number comparison against the actual Fortran
  executable's energies/forces has been done — the validation is finite-difference (FD)
  self-consistency against the equations being transcribed, not a byte-for-byte
  match to Fortran output. That's the one check I'd still recommend before
  claiming full equivalence.
  2. Reaction-field dielectric is a user input (eps_rf); the Fortran hard-codes
  78.4 (calcul_dielectric → DIELW = 78.4). You match it by passing eps_rf 78.4.
  3. Dispersion tail: the Fortran sets ercut = 0 and adds analytic exp-6 tail
  corrections (eset/pset); PairGCPM instead relies on LAMMPS pair_modify
  shift/tail. The short-range force is identical; the long-range dispersion
  correction is handled differently.
  4. Intramolecular exclusion / data model: the Fortran is hard-coded 4-site
  water summing intermolecular only; PairGCPM is generic per-atom and depends on
  your special_coul setup. For rigid water these differences are internal
  (projected out) but affect the absolute PE.

To summarize, the physics kernel is consistent and FD-exact; "completely consistent" in
the strict numerical sense would need a direct comparison run plus matching
the dielectric (78.4) and the dispersion-tail treatment.


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
| Long-range method | Real-space cutoff **+ Onsager reaction field** (`ferf`). No Ewald. | **Ewald / PPPM** (`ewaldflag = pppmflag = 1`, requires a `kspace` style). |
| Smeared Coulomb | `cerf(v)/r` with `v = r/alpha(is,js)`; `cerf` is the Abramowitz–Stegun erf approximation. | `erf(alpha_ij·r)/r` via `MathSpecial::my_erfcx`. |
| Real-space form | full smeared term (no screening subtraction). | GCPM Gaussian **minus** Ewald real-space screening: `falpha - erf(g_ewald·r) + EWALD_F·grij·expm2` (`pair_gcpm_long.cpp:331,337`). |
| Reaction field | **always on**, dielectric `erf` **hard-coded to 78.4** (`diel_cerf_hbond.f:46`); `ferf = 2(erf-1)/((2·erf+1)·rc³)` (`main.f:324`). | **optional** (`enable_rf`, only when `eps_rf > 0`); `eps_rf` is a user input. |

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

### Smeared T tensor (Eqs. 6–7) — identical math
- C++ `compute_induced_efield()` uses plain `erf(r/(2·sigmaM))` (no Ewald):
  `f = erf - (rds + rds·r²/(6s²))·exp`, `g = erf - rds·exp`,
  `T = 3f·r⁻⁵·rr - g·r⁻³·I`.
- Fortran `force.f:210–230` uses the same `fv`/`gv` and tensor, with
  `v = r/alpha` and `cerf`.

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
  earlier bug — "affects the torque"). In C++ this is subsumed into
  `reaction_field_post()` (the `q_k·(R_m^q + ½R_m^p)` site forces).

---

## 5. Reaction field — equivalent formula, different bookkeeping

- **Fortran:** folds `ferf` into every pair's tensor diagonal
  (`txx(i,j)+ferf`, `force.f:232`) and the charge field, plus self terms. For
  neutral molecules the per-pair sum reduces to a per-molecule cavity field
  `R_i = ferf · Σ_{j in cutoff incl. i} d_j`, and a per-pair charge-charge
  energy `0.5·qi·qj·ferf·r²` (`force.f:520`) that makes the Coulomb force go to
  zero smoothly at the cutoff.
- **C++:** implements the per-**molecule** form directly
  (`reaction_field()`): `R_i = c_rf · Σ_j d_j` over molecular cavity centers
  within `cut_coul`, with
  `c_rf = 2·qqrd2e·(eps_rf-1)/((2·eps_rf+1)·rc³)` — exactly the Fortran `ferf`
  with `qqrd2e` baked in. Permanent (`mol_mu` → `mol_Rq`) and induced
  (`mol_p` → `mol_Rp`) contributions are handled in `reaction_field_pre()` /
  `reaction_field_post()`.

The constants match (`c_rf` ≡ `ferf` at `qqrd2e = 1`). The difference is
granularity: Fortran applies the reaction field **per atom pair** (which also
smooths the pairwise cutoff); C++ applies it **per molecule** (an add-on to a
complete Ewald sum, which is already smooth at the cutoff). With Ewald present
this is consistent and energy-conserving; without Ewald, the per-molecule form
does **not** smooth the pairwise smeared-Coulomb cutoff — which is why the
reaction-field `gcpm` style (`pair_gcpm.cpp`) folds it in per pair instead. See
`pair_gcpm_summary.md`.

---

## 6. Charge–charge energy expression

- **Fortran** (`force.f:520`):
  `eqq += 0.5·qi·qj·(erf(v)/r + 0.5·ferf·r²)` — smeared Coulomb **plus** the
  per-pair reaction-field energy.
- **C++** (`charge_charge`): Ewald-split energy `prefactor·(erfa - erf)` for the
  real-space part; the reciprocal part comes from PPPM; the reaction-field
  energy (when enabled) is added in `reaction_field_post()`.

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
`pair_gcpm_long.cpp` replaces the Fortran's **cutoff + reaction field** electrostatics
with **Ewald / PPPM long-range**, keeping the reaction field only as an
optional, user-tunable add-on.

For a head-to-head numerical validation against the Fortran reference, use the
reaction-field style **`gcpm`** (`pair_gcpm.cpp`, no k-space) rather than
**`gcpm/long`** (`pair_gcpm_long.cpp`, Ewald/PPPM), because the two long-range
treatments are otherwise not directly comparable.

Note on class layout: `gcpm` (class `PairGCPM`, reaction field) is the base
class holding the shared GCPM machinery; `gcpm/long` (class `PairGCPMLong`,
Ewald/PPPM) derives from it and overrides only the Coulomb method.



