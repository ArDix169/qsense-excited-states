#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o_pes
#SBATCH --nodes=1
# One process per node holding all 192 cores.  Without these, sbatch warns and
# defaults to 192 TASKS of 1 core each -- 192 copies of the batch script rather
# than one script with 192 cores for joblib, and SLURM_CPUS_PER_TASK unset.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --array=0-79
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# H2O PES scan: the two lowest states of every C2v irrep, in both spin
# manifolds, at fixed selection thresholds.
#
#   rdist  = 0.75 .. 3.0 step 0.25          10 geometries
#   irrep  = A1, A2, B1, B2                  4 irreps
#   spin   = singlet (S_by2 0), triplet (2)  2 manifolds
#                                          ---------------
#                                            80 tasks
#
# THRESHOLDS, held constant across the whole scan:
#
#   eps_1 = 1e-6      csf_small_thrsh    (drop a CSF whose |coef| is below this
#                                         in EVERY target root)
#   eps_2 = 0         mp2_ampld_thrsh    (no MP2 pre-screen of ia pairs)
#   eps_3 = 1e-6      ratio * Ethrsh_select_ia = 1.0 * 1e-6
#   eps_l = eps_3^l   chained combinations admitted at (ratio*Ethrsh)^l
#   l_max = 2         combo_order
#
# WHY eps_3 = 1e-6 AND NOT 1.5e-4.  At 1.5e-4 the B2/singlet subspace starves
# past 2.5 A -- n_ucsf collapses to 12 of the sector's 28 CSFs -- and root 1
# lands 10.9 / 21.8 / 20.4 mHa high at 2.5 / 2.75 / 3.0.  Root 0 is unaffected
# (0.11-0.19 mHa): the ground state is carried by the base CSFs, it is the
# excited root that needs the basis extensions.  Measured recovery of root 1:
#
#   eps_3      2.5 A     2.75 A    3.0 A     n_ucsf at 3.0
#   1.5e-4     10.906    21.754    20.447    12
#   1.0e-4      2.964    17.786    20.447    12   <- ratio alone changes nothing
#   1.0e-5      2.842     1.353     0.558    19       at 3.0: no ia pair lies
#   1.0e-6     -0.002     1.354     0.558    19       in the 1e-4..1.5e-4 window
#
# Two orders were needed, not one: 1e-5 rescues 2.75 and 3.0 but leaves 2.5 at
# 2.84 mHa.  A per-geometry threshold would be cheaper but would break the
# "held constant across the scan" claim, so the tightest value is used
# everywhere.
#
# Only the PRODUCT ratio*Ethrsh_select_ia is physically meaningful -- with
# l_include_ia_in_CAS every in-CAS pair is kept regardless of Ethrsh_select_ia
# alone -- so (ratio 1.5, Ethrsh 1e-4) and (ratio 1.5e-4, Ethrsh 1.0) would
# select the same manifold.  The split below is the one asked for.
#
# WHY A2 IS INCLUDED.  STO-3G water has no A2 ORBITAL, so A2 is absent from
# mol.irrep_name -- but A2 STATES are plentiful, since a CSF with a singly
# occupied B1 and a singly occupied B2 carries B1 x B2 = A2.  At 1.0 A the A2
# sector holds 20 singlet and 26 triplet CSFs.  Leaving it out would silently
# drop a quarter of the spectrum.
#
# ACTIVE SPACE.  actmo 1..6 with the 6o8e Hamiltonian is CAS(8e,6o): H2O has
# ONE core orbital (O 1s), unlike H2O2's two.  The stored array is still
# 7 orbitals wide (the frozen core sits at index 0), so actmo_end is
# INCLUSIVE and 6 is the last of seven -- passing 7 crashes inside
# generate_CASCI_space with a shape-(0,) broadcast error.
#
# HAMILTONIAN.  h2o_sto3g_7o10e_* (symmetry=True full space, then
# actmo-restricted to CAS(8e,6o)). CONFIRMED 2026-08-25, empirically, not just
# by argument: a full 80-task rerun on h2o_sto3g_6o8e (symmetry=False) was
# tried and diffed against this sweep's dumps. 0.75-2.5 A agreed to <1 mHa,
# but 2.75 A was wrong by 58-370 mHa in EVERY one of the 8 sectors, and two
# roots at 3.0 A (A2/triplet E1, B1/triplet E1) were off by ~96 mHa -- the
# 6o8e orbital optimizer genuinely drifts onto a different electronic state
# near dissociation. 7o10e does not have this problem at any of the 10
# geometries (see HANDOFF.md for the full diff table). Do not switch this
# back to 6o8e for the PES scan.
#
# FILENAME COLLISION WARNING: the dump name encodes no_states, irrep, ratio,
# S_by2, csf_small_thrsh (T) and combo_order (C) -- but NOT Ethrsh_select_ia or
# Uopt_thrsh.  A rerun differing only in those two overwrites this one.
# Archive QSENSE_ES_dump before changing them.
#
# Subsets, without touching the rest:
#   sbatch --array=0-7  <this>                       # 0.75 A only
#   sbatch --export=ALL,BONDLENGTHS="1.0 1.5" --array=0-15 <this>

set -euo pipefail

module load python/3.11

VENV=""
for cand in "$SCRATCH/qsense_env_tri" "$PROJECT/qsense_env_tri" \
            "$HOME/qsense_env_tri"; do
    if [ -f "$cand/bin/activate" ]; then VENV="$cand"; break; fi
done
if [ -z "$VENV" ]; then
    echo "ERROR: qsense_env_tri not found in \$SCRATCH, \$PROJECT, or \$HOME" >&2
    exit 1
fi
echo "venv: $VENV"
source "$VENV/bin/activate"

WORKDIR="$SCRATCH/Q-SENSE"
cd "$WORKDIR"

# Write straight into the paper-data PES directory rather than the shared
# QSENSE_ES_dump.  The dump name encodes neither Ethrsh_select_ia nor
# Uopt_thrsh (see the collision warning above), so keeping this scan in its own
# directory is what stops an unrelated rerun from clobbering it.
export QSENSE_DUMPDIR="${QSENSE_DUMPDIR:-QSENSE_paper_data/PES/H2O}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NCORE="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-192}}"

# --- fixed thresholds for this scan ---
no_states=2
mp2_ampld_thrsh=0          # eps_2
csf_small_thrsh=1e-6       # eps_1
Ethrsh_select_ia=1e-6      # with ratio -> eps_3 = 1e-6
ratio=1.0
combo_order=2              # l_max
Uopt_thrsh=1e-6

actmo_start=1              # H2O has ONE core orbital
actmo_end=6                # INCLUSIVE index of the last of seven

# --- task decomposition: geometry x irrep x spin ---
if [ -n "${BONDLENGTHS:-}" ]; then
    read -r -a bondlengths <<< "$BONDLENGTHS"
else
    bondlengths=(0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
fi
irreps=(A1 A2 B1 B2)
spins=(0 2)

nspin=${#spins[@]}
nirrep=${#irreps[@]}
per_geom=$(( nirrep * nspin ))

g_idx=$(( SLURM_ARRAY_TASK_ID / per_geom ))
rem=$(( SLURM_ARRAY_TASK_ID % per_geom ))
i_idx=$(( rem / nspin ))
s_idx=$(( rem % nspin ))

if [ "$g_idx" -ge "${#bondlengths[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} is beyond the grid "\
"(${#bondlengths[@]} geometries x ${per_geom} sectors)" >&2
    exit 1
fi

rdist=${bondlengths[$g_idx]}
irrep=${irreps[$i_idx]}
s_by2=${spins[$s_idx]}
if [ "$s_by2" -eq 0 ]; then spin_label=singlet; else spin_label=triplet; fi

hamfile="$WORKDIR/Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_7o10e_phys_spatial_${rdist}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: rdist=${rdist}  irrep=${irrep}  ${spin_label}"
echo "  eps_1(csf_small_thrsh)=${csf_small_thrsh}  eps_2(mp2)=${mp2_ampld_thrsh}"
echo "  eps_3=ratio*Ethrsh=${ratio}*${Ethrsh_select_ia}  l_max(combo_order)=${combo_order}"
echo "  no_states=${no_states}  Uopt_thrsh=${Uopt_thrsh}  cores=${NCORE}"
echo "Hamiltonian: ${hamfile}"

if [ ! -f "$hamfile" ]; then
    echo "ERROR: Hamiltonian not found: $hamfile" >&2
    echo "  generate it with: cd Ham_gen && python3 h2o_sto3g_full.py" >&2
    exit 1
fi

# arg order: 1 hamfile 2 mp2_thrsh 3 nparal 4 actmo_start 5 actmo_end 6 rdist
#   7 Ethrsh_select_ia 8 Uopt_thrsh 9 initial_orb_rot 10 opt_orb
#   11 internal_mo_start 12 internal_mo_end 13 irrep 14 no_states 15 ratio
#   16 combo_order 17 S_by2 18 csf_small_thrsh
python qsense_subspace.py \
    "$hamfile" \
    "$mp2_ampld_thrsh" "$NCORE" "$actmo_start" "$actmo_end" "$rdist" \
    "$Ethrsh_select_ia" "$Uopt_thrsh" False True "$actmo_start" "$actmo_end" \
    "$irrep" "$no_states" "$ratio" "$combo_order" "$s_by2" "$csf_small_thrsh"
