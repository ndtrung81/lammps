# Self-diffusion of GCPM water with pair gcpm (reaction field), NVE production.
#
# Reference: Paricaud et al., J. Chem. Phys. 122, 244511 (2005), Table IV:
#   D = 0.226 Ang^2/ps at T = 298 K, rho = 0.997 g/cm^3.
# The Fortran reference code uses an Evans Gaussian isokinetic thermostat
# (holds total KE constant, ~NVE for transport), so production here is NVE
# after an NVT equilibration.  MSD is computed in LAMMPS (no wrapped-DCD
# post-processing) on the O sites: a rigid site's MSD equals the molecular
# COM MSD plus a constant rotational offset, so the slope (and D = slope/6)
# is unaffected.
#
# Two variants, selecting where the induced dipole lives:
#   COM placement (paper Eq. 3, matches the Fortran):
#     lmp -in in.gcpm.msd  (defaults: data.gcpm5, dipoletype 4, tag com)
#   M-site placement (historical 4-site decks, ~4.5% weaker dipoles):
#     lmp -in in.gcpm.msd -var datafile data.gcpm -var dipoletype 3 -var tag msite

variable datafile    index data.gcpm5
variable dipoletype  index 4        # 4 = COM dipole site (data.gcpm5); 3 = M site (data.gcpm)
variable tag         index com      # suffix for the msd output file
variable T           index 298
variable velseed     index 817324
variable rc          index 11.220684
variable equil_steps index 40000    # 20 ps NVT
variable prod_steps  index 400000   # 200 ps NVE

units real
atom_style hybrid full dipole sphere
read_data ${datafile}

pair_style gcpm 1 78.4 ${rc} ${rc}
pair_coeff 1 1 0.218445 3.69 12.75 0.0   0.000
pair_coeff 2 2 0.0      1.0  12.75 0.0   0.455
if "${dipoletype} == 4" then &
  "pair_coeff 3 3 0.0      1.0  12.75 0.0   0.610" &
  "pair_coeff 4 4 0.0      1.0  12.75 1.444 0.610" &
else &
  "pair_coeff 3 3 0.0      1.0  12.75 1.444 0.610"

neigh_modify exclude molecule/intra all

timestep 0.5

velocity all create $T ${velseed} mom yes rot yes dist gaussian

fix 1 all rigid/nvt/small molecule temp $T $T 100.0

thermo_style custom step temp pe etotal press
thermo 1000

run ${equil_steps}

# --- NVE production with molecular-COM MSD ---
unfix 1
fix 1 all rigid/nve/small molecule

reset_timestep 0

group oxygens type 1
compute msd oxygens msd com yes
fix 3 all ave/time 100 1 100 c_msd[4] file msd.${tag}.out

run ${prod_steps}
