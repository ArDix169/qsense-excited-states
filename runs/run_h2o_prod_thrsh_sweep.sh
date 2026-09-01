#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o_prod_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Threshold sweep for the H2O PRODUCTION sectors, to find the LOOSEST setting
# whose five lowest singlets all stay inside chemical accuracy of FULL FCI.
#
# Why: the `baseline` set (eps_1 1e-3, Ethrsh_ia 1e-6 -> eps_3 5e-6) lands at
# worst |dE| = 0.174 mHa, roughly 10x inside the 1.6 mHa bound, so it is paying
# for accuracy nobody asked for.  `opt` (eps_1 1e-2/5e-2, Ethrsh_ia 1e-4 ->
# eps_3 5e-4) reaches 0.76 mHa, still under half the bound.  The paper's stated
# protocol is the LARGEST threshold that keeps every targeted root inside
# chemical accuracy, so both are tighter than the protocol requires.
#
# GRID (ratio and l_max fixed at the production values 5.0 and 1):
#
#   eps_1  csf_small_thrsh    1e-2, 2e-2, 5e-2, 1e-1, 2e-1     (5)
#   Eth    Ethrsh_select_ia   1e-4, 5e-4, 1e-3                 (3)
#          -> eps_3 = ratio x Eth = 5e-4, 2.5e-3, 5e-3
#                                                          5 x 3 = 15 tasks
#
# Each task runs all 12 (sector x geometry) combinations of one parameter pair,
# forked across the node:  A1:2  A2:1  B1:1  B2:1   x   rOH 1.0, 1.5, 3.0.
#
# OUTPUT DIRECTORY, and why it is not the shared dump dir.  The dump filename
# encodes csf_small_thrsh (as _T<%.0e>) but NOT Ethrsh_select_ia, so two runs
# differing only in Eth write byte-identical names and the second destroys the
# first.  Each Eth therefore gets its own directory, keyed by eps_3, exactly as
# the h2o_A1_nstates/eps3_* convention does.  QSENSE_DUMPDIR (added 2026-08-28)
# is what makes CSF_UCSF_GS.py honour it.
#
#   QSENSE_ES_dump/h2o_prod_sweep/eps3_<eps3>/..._T<eps_1>_C1_..._<r>.dump
#
# Collect with:
#   python3 hpc/collect_h2o_prod_sweep.py

set -uo pipefail

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

# --- fixed at the production values ---
mp2_ampld_thrsh=0
ratio=5.0
combo_order=1
Uopt_thrsh=1e-6
actmo_start=1          # H2O has ONE core orbital
actmo_end=6            # INCLUSIVE index of the last of seven
HAMTAG=h2o_sto3g_7o10e

read -r -a eps1_grid <<< "${EPS1_GRID:-1e-2 2e-2 5e-2 1e-1 2e-1}"
read -r -a eth_grid  <<< "${ETH_GRID:-1e-4 5e-4 1e-3}"
read -r -a bondlengths <<< "${BONDLENGTHS:-1.0 1.5 3.0}"
read -r -a sectors <<< "${SECTORS:-A1:2 A2:1 B1:1 B2:1}"

n_eth=${#eth_grid[@]}
i_eps1=$(( SLURM_ARRAY_TASK_ID / n_eth ))
i_eth=$(( SLURM_ARRAY_TASK_ID % n_eth ))

if [ "$i_eps1" -ge "${#eps1_grid[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} is beyond the grid" >&2
    exit 1
fi

csf_small_thrsh=${eps1_grid[$i_eps1]}
Ethrsh_select_ia=${eth_grid[$i_eth]}

# eps_3 = ratio * Eth, formatted the way the directory name needs it
eps3=$(python3 -c "print(f'{${ratio}*${Ethrsh_select_ia}:.0e}')")
export QSENSE_DUMPDIR="QSENSE_ES_dump/h2o_prod_sweep/eps3_${eps3}"
mkdir -p "$WORKDIR/$QSENSE_DUMPDIR"

njobs=$(( ${#sectors[@]} * ${#bondlengths[@]} ))
MAXPAR="${MAXPAR:-$njobs}"
[ "$MAXPAR" -gt "$njobs" ] && MAXPAR="$njobs"
per=$(( NCORE / MAXPAR ))
[ "$per" -lt 1 ] && per=1

echo "================================================================"
echo "  H2O production threshold sweep -- task ${SLURM_ARRAY_TASK_ID}"
echo "  eps_1 (csf_small_thrsh) = ${csf_small_thrsh}"
echo "  Ethrsh_select_ia        = ${Ethrsh_select_ia}"
echo "  eps_3 = ratio x Eth     = ${ratio} x ${Ethrsh_select_ia} = ${eps3}"
echo "  ratio ${ratio}   l_max ${combo_order}"
echo "  sectors : ${sectors[*]}"
echo "  rOH     : ${bondlengths[*]}"
echo "  dumps -> ${QSENSE_DUMPDIR}"
echo "  node    : ${NCORE} cores -> ${MAXPAR} concurrent x ${per} cores each"
echo "================================================================"

running=0
fail=0
for r in "${bondlengths[@]}"; do
    hamfile="$WORKDIR/Ham_gen/hamiltonians/ES_Hamiltonians/${HAMTAG}_phys_spatial_${r}"
    if [ ! -f "$hamfile" ]; then
        echo "ERROR: Hamiltonian not found: $hamfile" >&2
        fail=$(( fail + 1 ))
        continue
    fi
    for spec in "${sectors[@]}"; do
        irrep="${spec%%:*}"
        no_states="${spec##*:}"

        while [ "$running" -ge "$MAXPAR" ]; do
            wait -n || true
            running=$(( running - 1 ))
        done

        log="$WORKDIR/$QSENSE_DUMPDIR/run_${irrep}_n${no_states}_r${r}.log"
        (
            # arg order: 1 hamfile 2 mp2_thrsh 3 nparal 4 actmo_start
            #   5 actmo_end 6 rdist 7 Ethrsh_select_ia 8 Uopt_thrsh
            #   9 initial_orb_rot 10 opt_orb 11 internal_mo_start
            #   12 internal_mo_end 13 irrep 14 no_states 15 ratio
            #   16 combo_order 17 S_by2 18 csf_small_thrsh
            python CSF_UCSF_GS.py \
                "$hamfile" \
                "$mp2_ampld_thrsh" "$per" "$actmo_start" "$actmo_end" "$r" \
                "$Ethrsh_select_ia" "$Uopt_thrsh" False True \
                "$actmo_start" "$actmo_end" \
                "$irrep" "$no_states" "$ratio" "$combo_order" 0 \
                "$csf_small_thrsh"
        ) > "$log" 2>&1 &
        running=$(( running + 1 ))
    done
done

wait
echo "task ${SLURM_ARRAY_TASK_ID} done (${fail} pre-flight failure(s))"
