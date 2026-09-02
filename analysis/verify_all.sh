#!/bin/bash
# Re-verify every published number against the archived data, in one pass.
#
# Runs from a clone with no setup beyond the conda environment: paths resolve
# relative to this file, and the data ships in the repository under
# data/QSENSE_paper_release/.
#
# Every deviation is taken against the FULL active-space FCI -- the
# (irrep, spin) key of the reference pickles, NOT the frozen-core
# (irrep, spin, 'cas') sector. Q-SENSE's actmo_start restricts which
# excitations the ansatz generates, not the Hamiltonian, so the full space is
# what the ansatz approximates. Differencing against the frozen-core sector
# understates the error by ~0.15 mHa and can produce a negative deviation,
# which is variationally impossible. See the README.
#
# Usage:
#     bash analysis/verify_all.sh                 # everything
#     bash analysis/verify_all.sh h2o_pes         # one section
#     bash analysis/verify_all.sh 2>&1 | tee verify_all.log
#
# Sections: h2o_pes h2o_prod h2o_scaling h2o2_pes h2o2_prod h2o2_scaling

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

DATA="${QSENSE_DATA:-$ROOT/data/QSENSE_paper_release}"
HAM="$DATA/Hamiltonians"
PY="${PYTHON:-python3}"
WANT="${1:-all}"

if [ ! -d "$DATA" ]; then
    echo "ERROR: no data at $DATA" >&2
    echo "The archive ships with this repository; if it is missing, either the" >&2
    echo "clone is incomplete or QSENSE_DATA points somewhere else." >&2
    exit 1
fi

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

REF_H2O="$HAM/h2o_sto3g_fci_ref_full.pkl"
REF_H2O2="$HAM/h2o2_sto3g_fci_ref_full.pkl"

# ---------------------------------------------------------------- H2O PES ---
if section h2o_pes; then
    banner "H2O PES  ->  figures/fig_pes_h2o.py     expect: max 1.358 mHa, 0 outside"
    DUMPDIR="$DATA/PES/H2O" REFPATH="$REF_H2O" \
        $PY analysis/collect_h2o_pes.py 2>&1 | tail -8
fi

# --------------------------------------------------------- H2O production ---
if section h2o_prod; then
    banner "H2O production  ->  tab:h2o-*           expect: worst 0.84 mHa"
    # The archive holds the `final` set directly, with its benchmark in bench/;
    # the HPC working tree kept several variants side by side instead.
    DUMPDIR="$DATA/Production/H2O" RESDIR="$DATA/Production/H2O/bench" \
        HAMROOT="$DATA/Baselines" REFPATH="$REF_H2O" \
        $PY analysis/collect_h2o_production.py 2>&1
fi

# ------------------------------------------------------------ H2O scaling ---
if section h2o_scaling; then
    banner "H2O scaling  ->  fig_scaling_h2o.py     expect: worst 0.899 mHa"
    # ratio 5.0, eps_1 1e-3, Ethrsh 2e-6 -> eps_3 1e-5, l_max 1.
    # Replaced an eps_3 5e-4 set that put A1 root3 60.4 mHa above FCI at 3.0 A
    # once n reached 4 -- the subspace stopped growing there.
    REFPATH="$REF_H2O" RATIO=5.0 CSF_THRSH=1e-3 LMAX=1 \
        $PY analysis/collect_h2o_nstates.py "$DATA/Scaling/H2O" 2>&1 | tail -8
fi

# --------------------------------------------------------------- H2O2 PES ---
if section h2o2_pes; then
    banner "H2O2 PES  ->  fig_pes_h2o2.py           expect: max 1.042 mHa, 0 outside"
    DUMPDIR="$DATA/PES/H2O2" REFPATH="$REF_H2O2" \
        $PY analysis/collect_h2o2_pes.py 2>&1 | tail -8
fi

# -------------------------------------------------------- H2O2 production ---
if section h2o2_prod; then
    banner "H2O2 production  ->  tab:h2o2-*         expect: reproduces the tables"
    DUMPDIR="$DATA/Production/H2O2" HAMROOT="$DATA/Baselines" \
        $PY analysis/collect_h2o2_production.py 2>&1
fi

# ----------------------------------------------------------- H2O2 scaling ---
if section h2o2_scaling; then
    banner "H2O2 scaling  ->  fig_scaling_h2o2.py   expect: 1.265 / 0.853 mHa"
    # 1.5 and 1.875 A share ratio 1.0 / eps_1 1e-4; 3.0 A uses ratio 0.1 /
    # eps_1 1e-6 -- the same per-geometry split H2O2 production uses, so this
    # takes two passes rather than one.
    for spec in "1.0 1e-4 1.5 1.875" "0.1 1e-6 3.0"; do
        set -- $spec
        rat=$1; csf=$2; shift 2
        echo
        echo "--- ratio $rat  eps_1 $csf  geometries: $* ---"
        REFPATH="$REF_H2O2" HAMTAG=h2o2_sto3g_12o18e IRREP=A LMAX=3 \
            RATIO=$rat CSF_THRSH=$csf BONDLENGTHS="$*" \
            $PY analysis/collect_h2o_nstates.py "$DATA/Scaling/H2O2" 2>&1 | tail -6
    done
fi

echo
echo "done."
