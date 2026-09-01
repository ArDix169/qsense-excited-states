#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=VO_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# VO circuit / measurement benchmark on the PRODUCTION Q-SENSE subspaces --
# the ones that go into the paper's cross-method table.
#
# The five lowest singlets of H2O2 split 3 A + 2 B at every geometry (FCI
# census), so the reported N_basis is the A/n=3 subspace PLUS the B/n=2
# subspace.  Six dumps, one task each:
#
#   A n=3 @ 1.5, 1.875, 3.0      B n=2 @ 1.5, 1.875, 3.0
#
# The task table is written out explicitly rather than derived by index
# arithmetic, because the parameter sets are NOT uniform: 3.0 A uses a
# different (ratio, csf_small_thrsh) than the other two, so no single formula
# maps a task index onto a filename.  See HANDOFF.md for why 3.0 A differs.
#
# WALLTIME.  Phase 2 is N(N-1)/2 off-diagonal elements, each transpiled at
# optimization_level=3, with N = n_ucsf running 1081..1688 here.  The n=5 runs
# of the earlier sweep were the expensive ones and are absent from this set, so
# 6 h should be comfortable -- but it is a guess, not a measurement.
#
# NOTE: OMP_NUM_THREADS=1 is required, not optional -- the benchmark forks a
# pool of NPARAL workers and each would otherwise spawn its own BLAS threads.

set -euo pipefail

module load python/3.11

MEASDIR="${MEASDIR:-$SCRATCH/seniority}"
DUMPDIR="${DUMPDIR:-$SCRATCH/Q-SENSE/QSENSE_ES_dump}"

if [ ! -f "$MEASDIR/Measurement_Benchmarking_Circuit_parallel.py" ]; then
    echo "ERROR: benchmark script not found in $MEASDIR" >&2
    exit 1
fi

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

# Something in seniority_env_tri calls MPI_Init at import.  This is a plain
# `python`, not mpirun, so Open MPI comes up as a singleton, spawns an orted and
# blocks opening a UD queue pair on mlx5_0.  The benchmark parallelizes with
# multiprocessing, never MPI, so confine MPI to shared memory.
export OMPI_MCA_plm=isolated
export OMPI_MCA_ess_singleton_isolated=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_btl_openib_allow_ib=0
export UCX_TLS=self,sm

NCORE="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-192}}"

# --- explicit task table: no_states | irrep | rdist | <ratio>_S<2S>_T<csf> ----
tasks=(
    "3 A 1.5   1.0_S0_T1e-04"
    "3 A 1.875 1.0_S0_T1e-04"
    "3 A 3.0   0.1_S0_T1e-06"
    "2 B 1.5   1.0_S0_T1e-04"
    "2 B 1.875 1.0_S0_T1e-04"
    "2 B 3.0   0.1_S0_T1e-06"
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#tasks[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} beyond table (${#tasks[@]} entries)" >&2
    exit 1
fi
read -r no_states irrep rdist tag <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"

dump="$DUMPDIR/h2o2_sto3g_12o18e_UCSF_${no_states}_${irrep}_${tag}_C3_for_Arjun_${rdist}.dump"
outjson="$DUMPDIR/h2o2_VO_prod_${irrep}${no_states}_${rdist}.json"

echo "Array task ${SLURM_ARRAY_TASK_ID}: no_states=${no_states}  irrep=${irrep}  rdist=${rdist}"
echo "  tag:    ${tag}"
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
