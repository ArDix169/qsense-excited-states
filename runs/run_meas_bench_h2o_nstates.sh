#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=meas_h2o_n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# VO circuit / measurement benchmark over the H2O CAS(8e,6o) A1-singlet
# n-scaling dumps.
#
#   n = 1..5   x   rOH = 1.0, 1.5, 3.0    = 15 runs
#
# Dumps: h2o_sto3g_7o10e_UCSF_<n>_A1_<ratio>_S<s>_T<tag>_C<l>_for_Arjun_<r>.dump
# (was h2o_sto3g_6o8e_...; default HAMTAG switched 2026-08-25, see HANDOFF.md)
# written by hpc/run_h2o_nstates_local.sh at eps_1 = 1e-3, eps_3 = 1e-5.
#
# WHY THIS IS NOT AN ARRAY JOB.  Trillium schedules whole nodes, and every one
# of these subspaces is tiny -- n_ucsf runs 9..31, so the dominant phase 2 is
# at most 31*30/2 = 465 transpiled off-diagonal elements.  A 15-task array
# would claim 15 nodes and sit in the queue for a workload one node finishes in
# minutes.  Instead all 15 runs are forked side by side on ONE node with
# 192/MAXPAR cores each.  (The H2O2 sweeps ARE arrays because there N = 887..2070
# and phase 2 reaches ~2.1M elements -- a genuine per-node workload.)
#
# Knobs (all overridable with --export=ALL,VAR=...):
#   DUMPDIR   directory holding the .dump files
#   HAMTAG    filename stem, default h2o_sto3g_7o10e (was 6o8e; see HANDOFF.md 2026-08-25)
#   RATIO CSF_TAG COMBO S_BY2 IRREP   the tags baked into the dump name
#   NSTATES BONDLENGTHS               the grid
#   RESDIR    where the per-run JSONs land
#   MAXPAR    concurrent runs on the node
#
# Submit:
#   sbatch hpc/run_meas_bench_h2o_nstates.sh
#   sbatch --export=ALL,COMBO=2 hpc/run_meas_bench_h2o_nstates.sh
#
# Collect afterwards:
#   python3 hpc/collect_meas_h2o_nstates.py $SCRATCH/seniority/results_h2o_nstates

set -uo pipefail

module load python/3.11

MEASDIR="${MEASDIR:-$SCRATCH/seniority}"
BENCH="$MEASDIR/Measurement_Benchmarking_Circuit_parallel.py"
if [ ! -f "$BENCH" ]; then
    echo "ERROR: benchmark script not found: $BENCH" >&2
    exit 1
fi

# Dedicated env: this needs qiskit / qiskit_aer / tequila / networkx, none of
# which are in qsense_env_tri, and installing them there would move numpy and
# scipy underneath the running Q-SENSE sweeps.
VENV="${MEAS_VENV:-}"
if [ -z "$VENV" ]; then
    for cand in "$SCRATCH/seniority_env_tri" "$PROJECT/seniority_env_tri"; do
        if [ -f "$cand/bin/activate" ]; then VENV="$cand"; break; fi
    done
fi
if [ -z "$VENV" ]; then
    echo "ERROR: seniority_env_tri not found in \$SCRATCH or \$PROJECT." >&2
    exit 1
fi
echo "venv: $VENV"
source "$VENV/bin/activate"

cd "$MEASDIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Keep OpenMPI out of the way -- this job does not use MPI.  One of the imported
# packages calls MPI_Init at import time, and outside mpirun that singleton
# spawns an orted daemon which deadlocks bringing up the mlx5_0 HCA that a
# non-srun job may not touch.  Symptom: python at 0% CPU forever, hung in import
# before the first marker prints.
export OMPI_MCA_ess_singleton_isolated=1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_pml=ob1
export UCX_TLS=self,sm
export OMPI_MCA_btl_base_warn_component_unused=0

NCORE="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-192}}"

# --- the grid --------------------------------------------------------------
DUMPDIR="${DUMPDIR:-$SCRATCH/Q-SENSE/QSENSE_ES_dump/h2o_A1_nstates/eps3_1e-05}"
# Keyed to the dump directory's basename (eps3_1e-05, eps3_1e-06, ...).  It has
# to be: the result filename is h2o_VO_benchmark_n<n>_r<r>.json, which carries
# NO threshold, so two runs at different eps_3 write byte-identical names and
# the second silently destroys the first -- the same hazard the dump names have
# with Ethrsh_select_ia.
RESDIR="${RESDIR:-$SCRATCH/seniority/results_h2o_nstates/$(basename "$DUMPDIR")}"
LOGDIR="${LOGDIR:-$RESDIR/logs}"

HAMTAG="${HAMTAG:-h2o_sto3g_7o10e}"
IRREP="${IRREP:-A1}"
RATIO="${RATIO:-1.0}"
# Canonicalise to Python's float repr, same as run_h2o_nstates_local.sh --
# CSF_UCSF_GS.py builds the dump name with str(float(ratio)), so RATIO=5
# looks for "_5_" while the file is "_5.0_" and every lookup silently misses.
# This is what made the first production_opt/production submissions
# (RATIO=5) fail outright: "ERROR: 15 dump(s) missing".
RATIO="$(python3 -c "print(float('${RATIO}'))")"
S_BY2="${S_BY2:-0}"
CSF_TAG="${CSF_TAG:-1e-03}"     # csf_small_thrsh, as CSF_UCSF_GS.py's '%.0e'
COMBO="${COMBO:-1}"             # l_max, appears as _C<l_max>

read -r -a nstates     <<< "${NSTATES:-1 2 3 4 5}"
read -r -a bondlengths <<< "${BONDLENGTHS:-1.0 1.5 3.0}"

# SECTORS overrides IRREP/NSTATES with an explicit "<irrep>:<n>" list, crossed
# with BONDLENGTHS.  The n-scaling study sweeps ONE irrep over many n; a
# production set is the opposite -- several irreps at their own n, chosen from
# the FCI census of the lowest states.  A single (irrep, n) pair cannot express
# that, and running the script once per sector would scatter the results across
# directories the collector cannot reassemble.
#
#   SECTORS="A1:2 A2:1 B1:1 B2:1"   x   3 geometries   =   12 runs
if [ -n "${SECTORS:-}" ]; then
    read -r -a sectors <<< "$SECTORS"
else
    sectors=()
    for _n in "${nstates[@]}"; do sectors+=("${IRREP}:${_n}"); done
fi

mkdir -p "$RESDIR" "$LOGDIR"

njobs=$(( ${#sectors[@]} * ${#bondlengths[@]} ))
MAXPAR="${MAXPAR:-$njobs}"
[ "$MAXPAR" -gt "$njobs" ] && MAXPAR="$njobs"
per=$(( NCORE / MAXPAR ))
[ "$per" -lt 1 ] && per=1

echo "================================================================"
echo "  H2O A1-singlet VO measurement benchmark"
echo "  dumps    : $DUMPDIR"
echo "  pattern  : ${HAMTAG}_UCSF_<n>_<irrep>_${RATIO}_S${S_BY2}_T${CSF_TAG}_C${COMBO}_for_Arjun_<r>.dump"
echo "  sectors  : ${sectors[*]}"
echo "  rOH      : ${bondlengths[*]}"
echo "  results  : $RESDIR"
echo "  node     : ${NCORE} cores -> ${MAXPAR} concurrent x ${per} cores each"
echo "================================================================"

# --- pre-flight: every dump must exist before anything is launched ----------
declare -a DUMPS=() TAGS=() NS=() RS=()
missing=0
for r in "${bondlengths[@]}"; do
    for spec in "${sectors[@]}"; do
        irr="${spec%%:*}"
        n="${spec##*:}"
        d="${DUMPDIR}/${HAMTAG}_UCSF_${n}_${irr}_${RATIO}_S${S_BY2}_T${CSF_TAG}_C${COMBO}_for_Arjun_${r}.dump"
        if [ ! -f "$d" ]; then
            echo "MISSING: $d" >&2
            missing=$(( missing + 1 ))
            continue
        fi
        DUMPS+=("$d")
        # irrep is in the tag: a production set has several sectors at the same
        # n and geometry, which would otherwise overwrite each other's JSON.
        TAGS+=("${irr}_n${n}_r${r}")
        NS+=("$n")
        RS+=("$r")
    done
done

if [ "$missing" -gt 0 ]; then
    echo >&2
    echo "ERROR: ${missing} dump(s) missing -- refusing to run a partial grid." >&2
    echo "--- what IS in ${DUMPDIR} ---" >&2
    ls -1 "$DUMPDIR"/*.dump >&2 2>/dev/null || echo "  (directory empty or absent)" >&2
    echo >&2
    echo "The tags RATIO / CSF_TAG / COMBO / S_BY2 must match the dump names" >&2
    echo "exactly.  Note that Ethrsh_select_ia is NOT in the name -- it is" >&2
    echo "encoded only by which eps3_* directory you point DUMPDIR at." >&2
    exit 1
fi

echo "${#DUMPS[@]} dumps found, launching."
echo

# --- fan out ---------------------------------------------------------------
# Exit status is recorded by each child into its own .rc file rather than read
# back with `wait <pid>`.  The throttle below uses `wait -n`, which reaps an
# ARBITRARY child, so a later `wait <pid>` on an already-reaped pid would fail
# with 127 and be indistinguishable from a genuine benchmark failure.
declare -a TAGRUN=()
running=0
fail=0

t0=$SECONDS
for i in "${!DUMPS[@]}"; do
    while [ "$running" -ge "$MAXPAR" ]; do
        wait -n || true
        running=$(( running - 1 ))
    done

    tag="${TAGS[$i]}"
    log="${LOGDIR}/meas_h2o_${tag}.log"
    rc="${LOGDIR}/meas_h2o_${tag}.rc"
    out="${RESDIR}/h2o_VO_benchmark_${tag}.json"
    rm -f "$rc"

    echo "  launch ${tag}  (${per} cores)  -> ${out}"
    (
        python "$BENCH" "${DUMPS[$i]}" \
            --molecule h2o \
            --bond-length "${RS[$i]}" \
            --no-states "${NS[$i]}" \
            --nparal "$per" \
            --output-json "$out" > "$log" 2>&1
        echo "$?" > "$rc"
    ) &
    TAGRUN+=("$tag")
    running=$(( running + 1 ))
done

echo
echo "waiting for ${#TAGRUN[@]} runs ..."
wait

for tag in "${TAGRUN[@]}"; do
    rc_file="${LOGDIR}/meas_h2o_${tag}.rc"
    rc="$(cat "$rc_file" 2>/dev/null || echo 99)"
    if [ "$rc" = "0" ]; then
        echo "  ok      ${tag}"
    else
        echo "  FAILED  ${tag}  (rc=${rc})  -- ${LOGDIR}/meas_h2o_${tag}.log"
        fail=$(( fail + 1 ))
    fi
done

echo
echo "elapsed: $(( SECONDS - t0 )) s   failures: ${fail} / ${#TAGRUN[@]}"

# --- summary table ---------------------------------------------------------
COLLECT="${SLURM_SUBMIT_DIR:-$SCRATCH/Q-SENSE}/hpc/collect_meas_h2o_nstates.py"
[ -f "$COLLECT" ] || COLLECT="$SCRATCH/Q-SENSE/hpc/collect_meas_h2o_nstates.py"
if [ -f "$COLLECT" ]; then
    echo
    NSTATES="${nstates[*]}" BONDLENGTHS="${bondlengths[*]}" \
        python "$COLLECT" "$RESDIR" || true
fi

exit "$fail"
