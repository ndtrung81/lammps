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



