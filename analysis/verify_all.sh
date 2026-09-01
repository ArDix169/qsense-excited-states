#!/bin/bash
# Recheck every published table and figure against the files in
# QSENSE_paper_data, in one pass.  Each study is self-contained in its own
# directory, so this reads nothing from the shared QSENSE_ES_dump -- which is
# the point: the shared tree has several runs of the same study at different
# thresholds whose dump names collide, and picking the wrong one is what
# produced the Table III / frozen-core-reference errors in the first place.
#
# Every deviation below is against the FULL FCI (the (irrep, spin) key), NOT
# the frozen-core (irrep, spin, 'cas') sector.  actmo_start restricts which
# excitations the ansatz generates, not the Hamiltonian.  See HANDOFF.md.
#
# Usage (from $SCRATCH/Q-SENSE):
#     bash hpc/verify_all.sh                 # everything
#     bash hpc/verify_all.sh h2o_pes         # one section
#     bash hpc/verify_all.sh 2>&1 | tee verify_all.log

set -uo pipefail

module load python/3.11 2>/dev/null
VENV=""
for cand in "$SCRATCH/qsense_env_tri" "$PROJECT/qsense_env_tri" \
            "$HOME/qsense_env_tri"; do
    [ -f "$cand/bin/activate" ] && VENV="$cand" && break
done
[ -n "$VENV" ] && source "$VENV/bin/activate"

cd "${WORKDIR:-$SCRATCH/Q-SENSE}"

PD=QSENSE_paper_data
WANT="${1:-all}"

section() {
    [ "$WANT" = all ] || [ "$WANT" = "$1" ] && return 0
    return 1
}

banner() {
    echo
    echo "################################################################"
    echo "#  $*"
    echo "################################################################"
}

# ---------------------------------------------------------------- H2O PES ---
if section h2o_pes; then
    banner "H2O PES  ->  fig_pes_h2o.py   ($PD/PES/H2O)"
    DUMPDIR=$PD/PES/H2O python3 hpc/collect_h2o_pes.py 2>&1 | tail -12
fi

# --------------------------------------------------------- H2O production ---
if section h2o_prod; then
    banner "H2O production FINAL  ->  tab:h2o-{energy-errors,sampling,circuits}"
    VARIANT=final RESDIR=$PD/Production/H2O/final_bench \
        python3 hpc/collect_h2o_production.py 2>&1

    banner "H2O production OPT (superseded by final; kept for comparison)"
    VARIANT=opt RESDIR=$PD/Production/H2O/opt_bench \
        python3 hpc/collect_h2o_production.py 2>&1 | grep -E \
        'rOH|Q-SENSE  \|dE\||TOTAL'
fi

# ------------------------------------------------------------ H2O scaling ---
if section h2o_scaling; then
    banner "H2O scaling  ->  fig_scaling_h2o.py   ($PD/Scaling/H2O)"
    # Set replaced 2026-08-29: was ratio 5.0 / eps_1 1e-4 / Ethrsh 1e-4
    # (eps_3 5e-4), which put A1 root3 60.4 mHa above FCI at 3.0 A for n>=4.
    # Now ratio 5.0 / eps_1 1e-3 / Ethrsh 2e-6 -> eps_3 1e-5, worst 0.899 mHa.
    RATIO=5.0 CSF_THRSH=1e-3 LMAX=1 \
        python3 hpc/collect_h2o_nstates.py $PD/Scaling/H2O 2>&1 | tail -22
fi

# -------------------------------------------------------- H2O2 production ---
if section h2o2_prod; then
    banner "H2O2 production  ->  tab:h2o2-{energy-errors,sampling,circuits}"
    DUMPDIR=$PD/Production/H2O2 python3 hpc/collect_h2o2_production.py 2>&1
fi

# ----------------------------------------------------------- H2O2 scaling ---
if section h2o2_scaling; then
    banner "H2O2 scaling  ->  fig_scaling_h2o2.py   ($PD/Scaling/H2O2)"
    REF=Ham_gen/hamiltonians/ES_Hamiltonians/h2o2_sto3g_fci_ref_full.pkl
    # 1.5 and 1.875 share ratio 1.0 / T1e-04; 3.0 uses ratio 0.1 / T1e-06 --
    # the same per-geometry split H2O2 production uses, so it takes two passes.
    for spec in "1.0 1e-4 1.5 1.875" "0.1 1e-6 3.0"; do
        set -- $spec
        rat=$1; csf=$2; shift 2
        echo
        echo "--- ratio $rat  eps_1 $csf  geometries: $* ---"
        REFPATH=$REF HAMTAG=h2o2_sto3g_12o18e IRREP=A LMAX=3 \
            RATIO=$rat CSF_THRSH=$csf BONDLENGTHS="$*" \
            python3 hpc/collect_h2o_nstates.py $PD/Scaling/H2O2 2>&1 | tail -18
    done
fi

# --------------------------------------------------------------- H2O2 PES ---
if section h2o2_pes; then
    banner "H2O2 PES  ->  fig_pes_h2o2.py   ($PD/PES/H2O2)"
    # The reference was extended to the full PES grid 2026-08-29, so this is
    # checkable now; it previously reported as not verifiable.
    DUMPDIR=$PD/PES/H2O2 python3 hpc/collect_h2o2_pes.py 2>&1 | tail -14
fi

echo
echo "done."
