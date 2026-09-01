#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o_scal_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Threshold sweep for the H2O A1-singlet n-SCALING study (n = 1..5), to replace
# the set currently in QSENSE_paper_data/Scaling/H2O.
#
# WHY THE CURRENT SET IS NOT USABLE.  It runs eps_3 = ratio x Ethrsh = 5.0 x
# 1e-4 = 5e-4, and at that threshold the subspace starves for the higher roots:
#
#   3.0 A, n=4 and n=5:  root3 lands at -74.50011665 while FCI has it at
#                        -74.56052877 -- 60.4 mHa high.  Roots 0-2 are fine
#                        (0.001-0.01 mHa), and n_ucsf is 21 at BOTH n=4 and
#                        n=5, i.e. the subspace did not grow at all when the
#                        4th state was added.  Not a state-tracking artifact:
#                        the nearest FCI root to that energy is still root3.
#   1.0 A, n=3..5:       roots 2 and 3 sit 2.7-4.1 mHa high.
#
# Every set on disk that IS inside chemical accuracy uses eps_3 = 1e-5, fifty
# times tighter.  This sweep finds the LOOSEST eps_3 that still works, so the
# subspaces stay as small as the accuracy bound allows.
#
# GRID (ratio 5.0 and l_max 1 fixed, matching the figure's stated setup):
#
#   eps_1  csf_small_thrsh    1e-3, 1e-4, 1e-5              (3)
#   Eth    Ethrsh_select_ia   2e-6, 4e-6, 1e-5, 2e-5, 4e-5  (5)
#          -> eps_3 = 5 x Eth = 1e-5, 2e-5, 5e-5, 1e-4, 2e-4
#                                                       3 x 5 = 15 tasks
#
# Each task runs n = 1..5 at rOH = 1.0, 1.5, 3.0 (15 runs) forked on its node.
#
# Output goes to its own directory per eps_3, because the dump name encodes
# csf_small_thrsh but NOT Ethrsh_select_ia -- two eps_3 at the same eps_1 would
# otherwise overwrite each other.  QSENSE_DUMPDIR makes qsense_subspace.py honour it.
#
# Collect with:
#   python3 hpc/collect_h2o_scaling_sweep.py

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
ratio=5.0
combo_order=1
Uopt_thrsh=1e-6
actmo_start=1
actmo_end=6
IRREP=A1
S_BY2=0
HAMTAG=h2o_sto3g_7o10e

read -r -a eps1_grid <<< "${EPS1_GRID:-1e-3 1e-4 1e-5}"
read -r -a eth_grid  <<< "${ETH_GRID:-2e-6 4e-6 1e-5 2e-5 4e-5}"
read -r -a nstates   <<< "${NSTATES:-1 2 3 4 5}"
read -r -a bondlengths <<< "${BONDLENGTHS:-1.0 1.5 3.0}"

n_eth=${#eth_grid[@]}
i_eps1=$(( SLURM_ARRAY_TASK_ID / n_eth ))
i_eth=$(( SLURM_ARRAY_TASK_ID % n_eth ))
if [ "$i_eps1" -ge "${#eps1_grid[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} beyond grid" >&2; exit 1
fi
csf_small_thrsh=${eps1_grid[$i_eps1]}
Ethrsh_select_ia=${eth_grid[$i_eth]}

eps3=$(python3 -c "print(f'{${ratio}*${Ethrsh_select_ia}:.0e}')")
export QSENSE_DUMPDIR="QSENSE_ES_dump/h2o_scaling_sweep/eps3_${eps3}"
mkdir -p "$WORKDIR/$QSENSE_DUMPDIR"

njobs=$(( ${#nstates[@]} * ${#bondlengths[@]} ))
MAXPAR="${MAXPAR:-$njobs}"
[ "$MAXPAR" -gt "$njobs" ] && MAXPAR="$njobs"
per=$(( NCORE / MAXPAR )); [ "$per" -lt 1 ] && per=1

echo "================================================================"
echo "  H2O A1 n-scaling threshold sweep -- task ${SLURM_ARRAY_TASK_ID}"
echo "  eps_1 = ${csf_small_thrsh}   Ethrsh = ${Ethrsh_select_ia}"
echo "  eps_3 = ${ratio} x ${Ethrsh_select_ia} = ${eps3}   l_max ${combo_order}"
echo "  n     : ${nstates[*]}"
echo "  rOH   : ${bondlengths[*]}"
echo "  dumps -> ${QSENSE_DUMPDIR}"
echo "  ${NCORE} cores -> ${MAXPAR} concurrent x ${per} each"
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
        log="$WORKDIR/$QSENSE_DUMPDIR/run_n${n}_r${r}.log"
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
