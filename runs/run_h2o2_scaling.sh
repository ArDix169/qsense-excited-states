#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o2_scaling
#SBATCH --nodes=1
# One process per node holding all 192 cores.  Without these, sbatch warns and
# defaults to 192 TASKS of 1 core each -- 192 copies of the batch script rather
# than one script with 192 cores for joblib, and SLURM_CPUS_PER_TASK unset.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Resource scaling with the number of targeted states, at the three geometries
# identified by the hardness diagnostic.
#
#   no_states = 1 .. 5   x   rdist = 1.5, 1.875, 3.0    = 15 tasks
#
# Geometries (see the CISD/CCSD hardness scan):
#   1.5   A  equilibrium  -- FCI ground-state minimum
#   1.875 A  correlated   -- interior maximum of eps_CCSD (1.20 mHa); the
#                           restricted reference goes unstable just beyond it
#                           (<S^2> departs from 0 at 2.0 A)
#   3.0   A  dissociated  -- two-radical limit; four configurations in two
#                           near-degenerate pairs carry 96% of the wavefunction
#
# Sector: A / singlet, matching the earlier n = 1..5 threshold sweep so the two
# are directly comparable.  Override with IRREP / S_BY2 if other sectors are
# wanted (the array size is unchanged; only the sector moves).
#
# PARAMETERS (differ from the PES sweep -- note this deliberately):
#   csf_small_thrsh   = 1e-4   (was 1e-6)
#   Ethrsh_select_ia  = 1e-6   (was 1e-7; overridable via ETHRSH_IA)
#   Uopt_thrsh        = 1e-6   (overridable via UOPT_THRSH) -- also sets the
#                              orbital-rotation eps, since qsense_subspace.py now
#                              passes Uopt_thrsh to
#                              opt_orbitals_for_weighted_n_roots
#   ratio             = 1.0    (unchanged)
#   combo_order       = 3      (unchanged)
#
# WHY THESE VALUES.  The first run of this sweep used 1e-5 for both and was not
# usable: at 1.875 A the ground state came out 8.7 mHa (n=3) and 8.0 mHa (n=4)
# above FCI, while the pre-optimization diagonalisation of the same reachable
# space sat at 0.07 and 0.12 mHa.  Every root degraded, so it was not a
# state-averaging trade-off -- the optimisation simply stopped short.  The PES
# sweep, at Uopt_thrsh = 1e-6, had nothing worse than 1.04 mHa.
#
# The basis-extension threshold is ratio * Ethrsh_select_ia = 1e-6, one order
# tighter than the PES sweep's 1e-7 and one looser than the first attempt here,
# so expect n_ucsf between the two.
#
# FILENAME COLLISION WARNING: the dump name encodes no_states, irrep, ratio,
# S_by2, csf_small_thrsh (T) and combo_order (C) -- but NOT Ethrsh_select_ia or
# Uopt_thrsh.  Runs differing only in those two will overwrite each other.
# T = 1e-04 with the _C3 tag distinguishes this sweep from both the PES sweep
# (T = 1e-06) and the older untagged threshold sweep, so nothing collides today.

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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NCORE="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-192}}"

# --- fixed for this sweep ---
irrep="${IRREP:-A}"
s_by2="${S_BY2:-0}"                            # 0 = singlet, 2 = triplet
# ratio multiplies Ethrsh_select_ia to set the generator/basis-extension cut:
# |improve| < ratio*Ethrsh_select_ia  ->  generator (amplitude ratio locked
# across states), otherwise -> basis extension (independent CI coefficient per
# state).  Driving ratio towards zero forces everything into basis extensions,
# which is the limiting case that tests whether the classifier is demoting
# state-differentiating configurations into generators.  Diagnostic, not a
# production setting: n_ucsf grows and compactness suffers.
ratio="${RATIO:-1.0}"
combo_order=3
csf_small_thrsh="${CSF_SMALL_THRSH:-1e-4}"
Ethrsh_select_ia="${ETHRSH_IA:-1e-6}"
Uopt_thrsh="${UOPT_THRSH:-1e-6}"

# --- map SLURM_ARRAY_TASK_ID -> (no_states, bond length) ---
nstates=(1 2 3 4 5)
# BONDLENGTHS restricts the scan to a subset, so one geometry can be re-run at
# a different threshold without touching the others.  The array range is
# 5*nbond - 1, i.e. 0-4 for a single geometry:
#
#   sbatch --array=0-4 --export=ALL,BONDLENGTHS=3.0,ETHRSH_IA=1e-7 <this script>
#
# The accuracy target is fixed (all targeted roots within 1.6 mHa of FCI); the
# threshold needed to reach it is a property of the geometry, and the subspace
# size at that point is the quantity being reported.
if [ -n "${BONDLENGTHS:-}" ]; then
    read -r -a bondlengths <<< "$BONDLENGTHS"
else
    bondlengths=(1.5 1.875 3.0)
fi
nbond=${#bondlengths[@]}

n_idx=$(( SLURM_ARRAY_TASK_ID / nbond ))       # 0..4
b_idx=$(( SLURM_ARRAY_TASK_ID % nbond ))       # 0..2

no_states=${nstates[$n_idx]}
rdist=${bondlengths[$b_idx]}
if [ "$s_by2" -eq 0 ]; then spin_label=singlet; else spin_label=triplet; fi

hamfile="$WORKDIR/Ham_gen/hamiltonians/ES_Hamiltonians/h2o2_sto3g_12o18e_phys_spatial_${rdist}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: no_states=${no_states}  rdist=${rdist}  irrep=${irrep}  ${spin_label}"
echo "  csf_small_thrsh=${csf_small_thrsh}  Ethrsh_select_ia=${Ethrsh_select_ia}  Uopt_thrsh=${Uopt_thrsh}"
echo "  ratio=${ratio}  combo_order=${combo_order}   cores=${NCORE}"
echo "Hamiltonian: ${hamfile}"

if [ ! -f "$hamfile" ]; then
    echo "ERROR: Hamiltonian not found: $hamfile" >&2
    exit 1
fi

# arg order: 1 hamfile 2 mp2_thrsh 3 nparal 4 actmo_start 5 actmo_end 6 rdist
#   7 Ethrsh_select_ia 8 Uopt_thrsh 9 initial_orb_rot 10 opt_orb
#   11 internal_mo_start 12 internal_mo_end 13 irrep 14 no_states 15 ratio
#   16 combo_order 17 S_by2 18 csf_small_thrsh
python qsense_subspace.py \
    "$hamfile" \
    0 "$NCORE" 2 11 "$rdist" "$Ethrsh_select_ia" "$Uopt_thrsh" False True 2 11 \
    "$irrep" "$no_states" "$ratio" "$combo_order" "$s_by2" "$csf_small_thrsh"
