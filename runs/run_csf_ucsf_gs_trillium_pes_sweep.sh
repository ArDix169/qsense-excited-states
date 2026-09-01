#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=CSF_UCSF_pes
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --array=0-31
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Full PES sweep on the CORRECTED geometry.
#
#   2 spins x 2 irreps x 8 bond lengths = 32 tasks (array indices 0-31)
#   %A = array job id, %a = task index -> one .out/.err per (spin,irrep,bond).
#
# Supersedes the pair of scripts that used to cover this range:
#   run_csf_ucsf_gs_fir_sweep.sh        1.25-2.5  (Fir, 128 cores)
#   run_csf_ucsf_gs_trillium_sweep_275_30.sh   2.75, 3.0
# Running one array instead keeps every geometry on identical parameters and
# identical hardware, which the old split did not.
#
# GEOMETRY.  These Hamiltonians were regenerated after two errors were found in
# the generator's Cartesian construction: `theta` and `tau` both entered as
# their supplements, so the earlier files were at angle(HOO) = 78.1 deg and
# dihedral = 68.5 deg rather than 101.9 / 111.5.  Results from before that fix
# are NOT comparable with these, and the filenames do not distinguish them.
#
# PARAMETERS: identical to the previous PES sweeps, so the only variable that
# has changed is the geometry --
#   Ethrsh_select_ia = 1e-7   (not a knob: l_include_ia_in_CAS keeps every
#                              in-CAS excitation regardless of it)
#   Uopt_thrsh       = 1e-6
#   no_states        = 2
#   ratio            = 1.0
#   combo_order      = 3
#   csf_small_thrsh  = 1e-6
#
# TRILLIUM NOTES (differ from Fir):
#   * account rrg-izmaylov (better FairShare than def-izmaylov, matches $PROJECT)
#   * SciNet schedules WHOLE NODES: request --nodes=1 and use all cores rather
#     than --cpus-per-task / --mem.  The `compute` partition is 192 cores /
#     767 GB per node, so NCORE resolves to 192 -- 50% more parallelism than
#     the 128-core Fir runs.
#   * run from $SCRATCH: $HOME is small and may be read-only from compute
#     nodes, and this job writes .dump/.json into the tree.
#
# WALLTIME.  6 h per task.  The longest Fir task (2.5 A) took 5:38 at 128
# cores; at 192 cores that should land near 3:45, so 6 h has headroom but is
# not generous.  The dissociated geometries (2.75, 3.0) have the largest CSF
# spaces and were previously given 24 h -- if tasks 6/7/14/15/22/23/30/31 come
# back TIMEOUT, those are the ones, and they need resubmitting with more time
# rather than a parameter change.

set -euo pipefail

module load python/3.11

# Trillium venv (qsense_env_tri).  $SCRATCH is searched FIRST and $HOME LAST:
# on SciNet $HOME is READ-ONLY from compute nodes, and `from Sym_C2V import *`
# makes openfermion write an .hdf5 into its own site-packages data dir at
# import time -- which fails with "Invalid cross-device link" / "Permission
# denied" if the venv lives in $HOME.
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

NCORE="${SLURM_CPUS_ON_NODE:-192}"

# --- map SLURM_ARRAY_TASK_ID -> (spin, irrep, bond length) ---
# BONDLENGTHS overrides the default list, so a single geometry can be added
# without editing this file or writing a second script, e.g. the correlated
# geometry identified by the CCSD hardness diagnostic:
#
#   sbatch --array=0-3 --export=ALL,BONDLENGTHS=1.875 <this script>
#
# The index arithmetic below is written against ${#bondlengths[@]}, so the
# array range is 2*2*nbond - 1 for whatever list is supplied.
if [ -n "${BONDLENGTHS:-}" ]; then
    read -r -a bondlengths <<< "$BONDLENGTHS"
else
    bondlengths=(1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
fi
irreps=(A B)
spins=(0 2)                                    # S_by2: 0 = singlet, 2 = triplet

nbond=${#bondlengths[@]}                        # 8
nirrep=${#irreps[@]}                            # 2
ncomb=$(( nbond * nirrep ))                     # 16  (irrep x bond block)

spin_idx=$(( SLURM_ARRAY_TASK_ID / ncomb ))     # 0..1
rem=$(( SLURM_ARRAY_TASK_ID % ncomb ))
irrep_idx=$(( rem / nbond ))                    # 0..1
bond_idx=$(( rem % nbond ))                     # 0..7

rdist=${bondlengths[$bond_idx]}
irrep=${irreps[$irrep_idx]}
s_by2=${spins[$spin_idx]}
if [ "$s_by2" -eq 0 ]; then spin_label=singlet; else spin_label=triplet; fi

hamfile="$WORKDIR/Ham_gen/hamiltonians/ES_Hamiltonians/h2o2_sto3g_12o18e_phys_spatial_${rdist}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: spin=${spin_label}(S_by2=${s_by2})  irrep=${irrep}  rdist=${rdist}"
echo "Hamiltonian: ${hamfile}   cores=${NCORE}"

if [ ! -f "$hamfile" ]; then
    echo "ERROR: Hamiltonian not found: $hamfile" >&2
    exit 1
fi

# arg order: 1 hamfile 2 mp2_thrsh 3 nparal 4 actmo_start 5 actmo_end 6 rdist
#   7 Ethrsh_select_ia 8 Uopt_thrsh 9 initial_orb_rot 10 opt_orb
#   11 internal_mo_start 12 internal_mo_end 13 irrep 14 no_states 15 ratio
#   16 combo_order 17 S_by2 18 csf_small_thrsh
#
# The dump filename carries irrep, no_states, ratio, S_by2, T and C, so no two
# tasks in this array collide.
#
# Uopt_thrsh (arg 8) is overridable so individual tasks can be re-run tighter
# without forking this script:
#
#   sbatch --array=6,12,17 --export=ALL,UOPT_THRSH=1e-8 <this script>
#
# Why: uopt_check.py showed the U optimization RAISING the energy in 3 of the
# 32 tasks -- 1.5 A/triplet root1 (+0.658 mHa), 2.25 B/singlet (+0.260/+0.519)
# and 2.75 A/singlet root0 (+0.261) -- which are precisely the three largest
# deviations from FCI in the whole sweep.  A converged minimization cannot end
# above its own starting point, so those are optimizer failures, not subspace
# error; the other 29 tasks agree with FCI to ~0.1 mHa.
#
# NOTE: the dump/JSON filename does NOT encode Uopt_thrsh, so a re-run
# OVERWRITES the previous result for that task.  Archive first if you want the
# before/after comparison:
#   mkdir -p QSENSE_ES_dump/uopt1e-6 && cp QSENSE_ES_dump/*UCSF_2_*_1.0_S*_T1e-06_C3_*.json QSENSE_ES_dump/uopt1e-6/
UOPT="${UOPT_THRSH:-1e-6}"
echo "  Uopt_thrsh = ${UOPT}"

python CSF_UCSF_GS.py \
    "$hamfile" \
    0 "$NCORE" 2 11 "$rdist" 1e-7 "$UOPT" False True 2 11 "$irrep" 2 1.0 3 "$s_by2" 1e-6
