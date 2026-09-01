#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=bench_detbasis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Execute the detbasis notebook once per geometry, for either molecule.
#
#   MOLECULE=h2o2 (default)   rOO = 1.5, 1.875, 3.0    CAS(18e,12o), 24 qubits
#   MOLECULE=h2o              rOH = 1.0, 1.5,   3.0    CAS(10e, 7o), 14 qubits
#
# The notebook already branches on MOLECULE and reads its geometry from $ROO or
# $ROH accordingly, so only the environment changes -- no second notebook and no
# edits.  Override the grid with GEOMS="1.0 1.5".
#
# NO NOTEBOOK EDITS ARE NEEDED.  The notebook already reads its geometry from
# $ROO and builds every output path from $OUTDIR, both environment variables:
#
#   RSCAN  = float(os.environ.get('ROO', 1.875))
#   OUTDIR = os.environ.get('OUTDIR', 'ham')
#
# so giving each task its own OUTDIR namespaces all of its output at once.
#
# WHAT WOULD HAVE COLLIDED.  Three files are written under OUTDIR:
#
#   {MOLECULE}_{RSCAN}_{tag}.pkl        the results -- ALREADY carries the
#                                       geometry, would not have collided
#   circmetrics_{key}.pkl               circuit-metric cache, key is an md5 of
#                                       the generator specs, NOT the geometry
#   pergen_{MOLECULE}_{NCAS}o_l3.pkl    per-generator transpile cache
#
# The last two are deliberately geometry-INDEPENDENT: the notebook's own
# comment says "every point of the O-O scan reuses these entries".  Sharing
# them is correct in a serial scan and dangerous in a parallel one -- pergen is
# rewritten in full every 64 generators with a bare
# `pickle.dump(per, open(percache,'wb'))`, so three tasks writing concurrently
# can leave a truncated file that the next read fails on.  Per-task OUTDIR
# removes the race.
#
# THE COST OF THAT.  Each task now transpiles the generator pool itself instead
# of reusing a neighbour's cache.  If that dominates the runtime, warm a shared
# cache ONCE and let the tasks read it:
#
#   1. run task 1 alone with OUTDIR=ham_shared to populate pergen_*.pkl
#   2. resubmit 0 and 2 with OUTDIR=ham_shared
#
# That is safe only because by then no task is still WRITING new generators.
# Do not start all three against a shared OUTDIR.
#
# CPUS AND MEMORY.  Trillium schedules WHOLE NODES and rejects --mem outright:
#
#   "The --mem=... request is not allowed nor necessary on Trillium; all nodes
#    have the same amount of available memory (745 GiB) and each job gets all
#    the available memory of the node"
#
# So 8 CPUs / 32 GB cannot be requested -- the allocation is 192 cores and
# 745 GiB whichever way it is asked for.  BENCH_N_JOBS follows
# SLURM_CPUS_PER_TASK and therefore becomes 192, which is the right default
# when the node is billed to this job regardless.  To hold the pool smaller
# anyway (memory pressure, or reproducing an earlier 8-worker run):
#
#   sbatch --export=ALL,BENCH_N_JOBS=8 hpc/run_benchmark_array.sh

set -euo pipefail

module load python/3.11

NOTEBOOK="${NOTEBOOK:-Benchmark_Detbasis (3) (1).ipynb}"

WORKDIR="${WORKDIR:-$SCRATCH/Q-SENSE}"
cd "$WORKDIR"

if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: notebook not found: $WORKDIR/$NOTEBOOK" >&2
    echo "--- notebooks here ---" >&2
    ls -1 ./*.ipynb >&2 2>/dev/null || echo "  (none)" >&2
    exit 1
fi

VENV="${BENCH_VENV:-}"
if [ -z "$VENV" ]; then
    for cand in "$SCRATCH/seniority_env_tri" "$SCRATCH/qsense_env_tri" \
                "$PROJECT/seniority_env_tri" "$PROJECT/qsense_env_tri"; do
        if [ -f "$cand/bin/activate" ]; then VENV="$cand"; break; fi
    done
fi
if [ -z "$VENV" ]; then
    echo "ERROR: no venv found. Set BENCH_VENV=/path/to/env" >&2
    exit 1
fi
echo "venv: $VENV"
source "$VENV/bin/activate"

if ! python -c "import nbconvert, nbformat, ipykernel" 2>/dev/null; then
    echo "ERROR: nbconvert / nbformat / ipykernel missing from $VENV" >&2
    echo "  pip install nbconvert nbformat ipykernel" >&2
    exit 1
fi

# --- molecule, geometry grid, and the env var the notebook reads -------------
export MOLECULE="${MOLECULE:-h2o2}"
if [ "$MOLECULE" = "h2o" ]; then
    RVAR=ROH; RTAG=rOH; DEFAULT_GEOMS="1.0 1.5 3.0"
else
    RVAR=ROO; RTAG=rOO; DEFAULT_GEOMS="1.5 1.875 3.0"
fi
read -r -a geoms <<< "${GEOMS:-$DEFAULT_GEOMS}"

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#geoms[@]}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} is beyond the grid "\
"(${#geoms[@]} geometries for ${MOLECULE})" >&2
    exit 1
fi
RVAL="${geoms[${SLURM_ARRAY_TASK_ID:-0}]}"

# the notebook reads ROO for h2o2 and ROH for h2o -- export whichever applies
export "$RVAR=$RVAL"
export OUTDIR="${OUTDIR:-ham_${RTAG}_${RVAL}}"
mkdir -p "$OUTDIR"

# --- threading --------------------------------------------------------------
# BENCH_N_JOBS is the notebook's own pool size (get_n_jobs reads it first).
# The BLAS variables must be 1 or each of those workers spawns its own thread
# team on top of the pool and oversubscribes the allocation.
export BENCH_N_JOBS="${BENCH_N_JOBS:-${SLURM_CPUS_PER_TASK:-192}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Keep OpenMPI from spawning an orted daemon for a non-MPI singleton, which on
# Trillium deadlocks bringing up an InfiniBand HCA the job may not touch.
export OMPI_MCA_ess_singleton_isolated=1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_pml=ob1
export UCX_TLS=self,sm

OUTNB="Benchmark_${RTAG}_${RVAL}.ipynb"

echo "================================================================"
echo "  task ${SLURM_ARRAY_TASK_ID}:  ${RVAR} = ${RVAL}   MOLECULE = ${MOLECULE}"
echo "  notebook : ${NOTEBOOK}"
echo "  outdir   : ${OUTDIR}"
echo "  executed : ${OUTNB}"
echo "  workers  : BENCH_N_JOBS=${BENCH_N_JOBS}  (BLAS threads pinned to 1)"
echo "================================================================"

# --output takes a BASE NAME, not a path -- --output-dir sets the directory.
# timeout=-1 disables the per-cell limit: cells here run for hours and the
# nbconvert default would kill them.
# allow-errors=0 (the default) means a failing cell aborts the task, so a
# broken run is a nonzero exit rather than a saved notebook full of tracebacks.
# `set -e` would abort here on failure, so the exit code is captured with an
# explicit `|| rc=$?` rather than a bare `$?` on the next line -- that form is
# dead code under `set -e` and the summary below would never print on a failed
# run, which is exactly when it is wanted.
rc=0
srun python -m nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$OUTNB" \
    --output-dir "$WORKDIR" \
    "$NOTEBOOK" || rc=$?

echo
if [ "$rc" -eq 0 ]; then
    echo "nbconvert OK -> ${OUTNB}"
else
    echo "nbconvert FAILED (exit ${rc}) -- see bench_detbasis-${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}.err"
fi
echo "--- ${OUTDIR} ---"
ls -lh "$OUTDIR" || true
exit "$rc"
