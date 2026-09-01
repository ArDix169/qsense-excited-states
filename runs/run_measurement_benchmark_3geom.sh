#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=VO_meas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# VO circuit / measurement benchmark over the Q-SENSE UCSF subspaces at the
# three geometries from the hardness diagnostic, for n = 1..5 targeted states.
#
#   n = 1..5   x   rdist = 1.5, 1.875, 3.0    = 15 tasks
#
# PARAMETER SETS.  These are NOT uniform across geometries -- each is the
# largest selection threshold at which every targeted root lands within
# chemical accuracy (1.6 mHa) of FCI, which is the accuracy protocol the paper
# states.  The dissociated geometry needs a different partition of the ia pairs:
#
#   1.5   A   ratio 1.0  csf 1e-4  Ethrsh_ia 1e-6
#   1.875 A   ratio 1.0  csf 1e-4  Ethrsh_ia 1e-6
#   3.0   A   ratio 0.1  csf 1e-6  Ethrsh_ia 1e-7
#
# Why 3.0 A differs: its three lowest A singlets lie within 1.6 mHa of each
# other and are different orthogonal combinations of ONE shared six-
# configuration set, with weights differing by factors of 33-154.  A generator
# ties the amplitude ratio identically across every state, so those
# configurations have to enter as basis extensions instead -- ratio 0.1 moves
# them there (generators 951 -> 674 at n=3, basis 395 -> 672).  At ratio 1.0 no
# csf_small_thrsh in 1e-3..1e-6 put all five roots inside chemical accuracy.
#
# WALLTIME.  Phase 2 of the benchmark is N(N-1)/2 off-diagonal elements, each
# transpiled at optimization_level=3, where N = n_ucsf.  Its docstring targets
# ~600 states; here N runs 887..2070, so the largest tasks are ~12x the
# workload it was written for (n=5 at 3.0 A: N=2070 -> ~2.1M elements).  6 h is
# a queue-turnaround choice, not a measurement of what the job needs, and the
# n=5 tasks (12, 13, 14) are the ones at risk.  A TIMEOUT there means the task
# needs splitting or a checkpoint -- never a parameter change, since the
# parameter sets are fixed by the accuracy protocol above.
#
# NOTE: OMP_NUM_THREADS=1 is required, not optional -- the script forks a pool
# of NPARAL workers and each would otherwise spawn its own BLAS threads.

set -euo pipefail

module load python/3.11

# The benchmark lives in the seniority tree and imports src.measurement_new
# relative to it, so it must run from there -- but the dumps live in Q-SENSE.
MEASDIR="${MEASDIR:-$SCRATCH/seniority}"
DUMPDIR="${DUMPDIR:-$SCRATCH/Q-SENSE/QSENSE_ES_dump}"

if [ ! -f "$MEASDIR/Measurement_Benchmarking_Circuit_parallel.py" ]; then
    echo "ERROR: benchmark script not found in $MEASDIR" >&2
    exit 1
fi

# Prefer a venv that can import the seniority package; override with MEAS_VENV.
VENV="${MEAS_VENV:-}"
if [ -z "$VENV" ]; then
    for cand in "$SCRATCH/seniority_env_tri" "$SCRATCH/qsense_env_tri" \
                "$PROJECT/qsense_env_tri" "$HOME/qsense_env_tri"; do
        if [ -f "$cand/bin/activate" ]; then VENV="$cand"; break; fi
    done
fi
if [ -z "$VENV" ]; then
    echo "ERROR: no venv found; set MEAS_VENV=/path/to/env" >&2
    exit 1
fi
echo "venv: $VENV"
source "$VENV/bin/activate"

cd "$MEASDIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Something in seniority_env_tri calls MPI_Init at import time (an MPI-linked
# h5py or a transitively pulled mpi4py).  On Trillium that hangs: the run is a
# plain `python`, not mpirun, so Open MPI comes up as a singleton, spawns an
# orted, and blocks trying to open a UD queue pair on mlx5_0 --
#   "Failed to modify UD QP to INIT on mlx5_0: Operation not permitted"
# followed by 14 minutes at 0.1% CPU.  This benchmark parallelizes with
# multiprocessing (fork), never MPI, so confine MPI to shared memory and stop
# it launching a daemon.
export OMPI_MCA_plm=isolated
export OMPI_MCA_ess_singleton_isolated=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_btl_openib_allow_ib=0
export UCX_TLS=self,sm

NCORE="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-192}}"

# --- map SLURM_ARRAY_TASK_ID -> (no_states, geometry + its parameter set) ---
nstates=(1 2 3 4 5)
bondlengths=(1.5 1.875 3.0)
# tag = <ratio>_S0_T<csf_small_thrsh>, matching the dump filename
tags=("1.0_S0_T1e-04" "1.0_S0_T1e-04" "0.1_S0_T1e-06")

nbond=${#bondlengths[@]}
n_idx=$(( SLURM_ARRAY_TASK_ID / nbond ))
b_idx=$(( SLURM_ARRAY_TASK_ID % nbond ))

no_states=${nstates[$n_idx]}
rdist=${bondlengths[$b_idx]}
tag=${tags[$b_idx]}

dump="$DUMPDIR/h2o2_sto3g_12o18e_UCSF_${no_states}_A_${tag}_C3_for_Arjun_${rdist}.dump"
outjson="$DUMPDIR/h2o2_VO_benchmark_n${no_states}_${rdist}.json"

echo "Array task ${SLURM_ARRAY_TASK_ID}: no_states=${no_states}  rdist=${rdist}  tag=${tag}"
echo "  dump:   ${dump}"
echo "  output: ${outjson}"
echo "  cores:  ${NCORE}"

if [ ! -f "$dump" ]; then
    echo "ERROR: dump not found: $dump" >&2
    exit 1
fi

python Measurement_Benchmarking_Circuit_parallel.py "$dump" \
    --molecule h2o2 \
    --bond-length "$rdist" \
    --no-states "$no_states" \
    --nparal "$NCORE" \
    --output-json "$outjson"
