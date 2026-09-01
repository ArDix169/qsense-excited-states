#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o_ratio_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# 2-D sweep of Ethrsh_select_ia x ratio for the H2O A1-singlet n-scaling study.
#
# WHY TWO AXES.  qsense_subspace.py uses the two thresholds for DIFFERENT things
# (see its own plot labels, lines 2393-2394):
#
#     Ethrsh_select_ia          -> "Generator Selection Threshold"
#     ratio * Ethrsh_select_ia  -> "Basis Extension Threshold"
#
# A pair whose |improvement| exceeds ratio*Ethrsh enters as a BASIS EXTENSION;
# one between Ethrsh and ratio*Ethrsh stays a GENERATOR.  So:
#
#     accuracy  is set by Ethrsh   -- the lower bound, i.e. what gets included
#                                     at all, either way
#     gen/basis split is set by ratio -- how much of that is generators
#
# The previous sweep varied only eps_3 = ratio*Ethrsh at fixed ratio 5.0, which
# cannot separate the two.  It landed on Ethrsh 2e-6 / ratio 5.0: accurate
# (0.899 mHa) but with the generator window squeezed to [2e-6, 1e-5], so almost
# everything became a basis extension and the generators COLLAPSED --
# eps^2 M went to 0.0000 at 1.5 A n=5, fell with increasing n at 3.0 A, and
# CNOT counts dropped as the subspace grew.  All three are the signature of a
# subspace carrying (almost) no generators, and none of them is a usable
# measurement-cost number.
#
# GRID (eps_1 fixed at 1e-3: the earlier sweep showed it has NO effect anywhere
# in 1e-3..1e-5, every row was identical):
#
#   Ethrsh  2e-6, 5e-6, 1e-5, 2e-5      (4)   -- accuracy axis
#   ratio   1, 2, 5, 10, 50             (5)   -- generator/basis split
#                                           4 x 5 = 20 tasks
#
# Each task runs n = 1..5 at rOH = 1.0, 1.5, 3.0 (15 runs) forked on its node.
#
# The dump name carries ratio and csf_small_thrsh but NOT Ethrsh_select_ia, so
# each Ethrsh gets its own directory; ratio distinguishes files within it.
#
# Collect with:
#   python3 hpc/collect_h2o_scaling_ratio_sweep.py

set -uo pipefail

module load python/3.11

VENV=""
for cand in "$SCRATCH/qsense_env_tri" "$PROJECT/qsense_env_tri" \
            "$HOME/qsense_env_tri"; do
    [ -f "$cand/bin/activate" ] && VENV="$cand" && break
done
if [ -z "$VENV" ]; then
    echo "ERROR: qsense_env_tri not found" >&2
    exit 1
fi
source "$VENV/bin/activate"

WORKDIR="$SCRATCH/Q-SENSE"
cd "$WORKDIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NCORE="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-192}}"

mp2_ampld_thrsh=0
csf_small_thrsh="${CSF_SMALL_THRSH:-1e-3}"
combo_order=1
Uopt_thrsh=1e-6
actmo_start=1
actmo_end=6
IRREP=A1
S_BY2=0
HAMTAG=h2o_sto3g_7o10e

read -r -a eth_grid   <<< "${ETH_GRID:-2e-6 5e-6 1e-5 2e-5}"
read -r -a ratio_grid <<< "${RATIO_GRID:-1 2 5 10 50}"
read -r -a nstates    <<< "${NSTATES:-1 2 3 4 5}"
read -r -a bondlengths <<< "${BONDLENGTHS:-1.0 1.5 3.0}"

n_ratio=${#ratio_grid[@]}
i_eth=$(( SLURM_ARRAY_TASK_ID / n_ratio ))
i_ratio=$(( SLURM_ARRAY_TASK_ID % n_ratio ))
if [ "$i_eth" -ge "${#eth_grid[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} beyond grid" >&2; exit 1
fi
Ethrsh_select_ia=${eth_grid[$i_eth]}
# qsense_subspace.py builds the dump name with str(float(ratio)): RATIO=5 would look
# for "_5_" while the file is "_5.0_".  Canonicalise, as the meas-bench runner does.
ratio=$(python3 -c "print(float('${ratio_grid[$i_ratio]}'))")

export QSENSE_DUMPDIR="QSENSE_ES_dump/h2o_scaling_ratio_sweep/eth_${Ethrsh_select_ia}"
mkdir -p "$WORKDIR/$QSENSE_DUMPDIR"

njobs=$(( ${#nstates[@]} * ${#bondlengths[@]} ))
MAXPAR="${MAXPAR:-$njobs}"
[ "$MAXPAR" -gt "$njobs" ] && MAXPAR="$njobs"
per=$(( NCORE / MAXPAR )); [ "$per" -lt 1 ] && per=1

eps3=$(python3 -c "print(f'{${ratio}*${Ethrsh_select_ia}:.1e}')")
echo "================================================================"
echo "  H2O A1 n-scaling  Ethrsh x ratio sweep -- task ${SLURM_ARRAY_TASK_ID}"
echo "  Ethrsh (generator thrsh)      = ${Ethrsh_select_ia}"
echo "  ratio                         = ${ratio}"
echo "  ratio*Ethrsh (basis ext thrsh)= ${eps3}"
echo "  eps_1 = ${csf_small_thrsh}   l_max = ${combo_order}"
echo "  n   : ${nstates[*]}"
echo "  rOH : ${bondlengths[*]}"
echo "  dumps -> ${QSENSE_DUMPDIR}"
echo "================================================================"

running=0
for r in "${bondlengths[@]}"; do
    hamfile="$WORKDIR/Ham_gen/hamiltonians/ES_Hamiltonians/${HAMTAG}_phys_spatial_${r}"
    if [ ! -f "$hamfile" ]; then
        echo "ERROR: Hamiltonian not found: $hamfile" >&2; continue
    fi
    for n in "${nstates[@]}"; do
        while [ "$running" -ge "$MAXPAR" ]; do
            wait -n || true; running=$(( running - 1 ))
        done
        log="$WORKDIR/$QSENSE_DUMPDIR/run_r${ratio}_n${n}_g${r}.log"
        (
            python qsense_subspace.py \
                "$hamfile" \
                "$mp2_ampld_thrsh" "$per" "$actmo_start" "$actmo_end" "$r" \
                "$Ethrsh_select_ia" "$Uopt_thrsh" False True \
                "$actmo_start" "$actmo_end" \
                "$IRREP" "$n" "$ratio" "$combo_order" "$S_BY2" \
                "$csf_small_thrsh"
        ) > "$log" 2>&1 &
        running=$(( running + 1 ))
    done
done

wait
echo "task ${SLURM_ARRAY_TASK_ID} done"
