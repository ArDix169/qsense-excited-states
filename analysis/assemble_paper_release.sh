#!/bin/bash
# Assemble everything the paper cites into one self-contained tree, ready to
# archive (Zenodo) and reference from the code repository.
#
# Copies rather than moves: the working directories stay untouched, so a
# mistake here costs nothing and the release can be rebuilt from scratch.
#
# WHAT GOES IN -- only the runs the paper actually uses:
#
#   PES/H2O          80 sectors, eps_1 1e-6 / eps_3 1e-6 / l_max 2
#                   ; max |dE| 1.358 mHa, 0 outside chem acc)
#   PES/H2O2         32 sectors
#   Production/H2O   eps_3 1e-5 at 1.0/1.5, per-geometry eps_1;
#                    worst |dE| 0.84 mHa.
#   Production/H2O2  3 geometries x {A:3, B:2} plus the VO cost JSONs
#   Scaling/H2O      eps_3 1e-5 set, worst |dE| 0.899 mHa.  The eps_3 5e-4 set
#                    is EXCLUDED: it was 60.4 mHa outside chemical accuracy.
#   Scaling/H2O2     3 geometries x n=1..5
#   Hamiltonians/    the phys_spatial files those runs consumed, + FCI refs
#   Benchmarks/      detbasis pickles behind the q-sc-EOM / SG-SSVQE columns
#
# Each data directory gets a PROVENANCE.txt naming the parameters and the
# verification result, because the dump filename encodes neither
# Ethrsh_select_ia nor Uopt_thrsh -- without it the numbers cannot be traced.
#
# Usage (from $SCRATCH/Q-SENSE):
#     bash hpc/assemble_paper_release.sh
#     DEST=/somewhere/else bash hpc/assemble_paper_release.sh

set -uo pipefail

SRC="${SRC:-$SCRATCH/Q-SENSE}"
DEST="${DEST:-$SCRATCH/QSENSE_paper_release}"
PD="$SRC/QSENSE_paper_data"

cd "$SRC"
mkdir -p "$DEST"

note() { printf '%s\n' "$*" ; }
copy_dir() {   # $1 = src dir, $2 = dest dir, $3.. = provenance lines
    local s="$1" d="$2"; shift 2
    if [ ! -d "$s" ]; then note "  MISSING SOURCE: $s"; return 1; fi
    mkdir -p "$d"
    cp -n "$s"/*.dump "$s"/*.json "$d"/ 2>/dev/null
    local n_d n_j
    n_d=$(ls "$d"/*.dump 2>/dev/null | wc -l)
    n_j=$(ls "$d"/*.json 2>/dev/null | wc -l)
    { printf '%s\n' "$@"; echo; echo "files: ${n_d} dump, ${n_j} json"; \
      echo "assembled: $(date -u +%Y-%m-%dT%H:%M:%SZ) from ${s#$SRC/}"; } \
        > "$d/PROVENANCE.txt"
    note "  $(printf '%-34s' "${d#$DEST/}") ${n_d} dump / ${n_j} json"
}

note "=============================================================="
note "  assembling -> $DEST"
note "=============================================================="

note "PES:"
copy_dir "$PD/PES/H2O" "$DEST/PES/H2O" \
  "H2O PES scan, 10 geometries x 4 irreps x 2 spins = 80 sectors." \
  "eps_1 = 1e-6, eps_2 = 0, eps_3 = ratio 1.0 x Ethrsh 1e-6 = 1e-6, l_max = 2." \
  "Hamiltonian h2o_sto3g_7o10e, actmo 1-6 => CAS(8e,6o), 2 states/sector." \
  "Verified vs FULL CAS(10e,7o) FCI: max |dE| = 1.358 mHa, 0 sectors outside" \
  "chemical accuracy (160 roots).  to fill 3 missing" \
  "A1/singlet sectors."

copy_dir "$PD/PES/H2O2" "$DEST/PES/H2O2" \
  "H2O2 PES scan, 8 geometries (1.25-3.0 A) x 2 irreps x 2 spins = 32 sectors." \
  "ratio 1.0, eps_1 1e-6, l_max 3.  Hamiltonian h2o2_sto3g_12o18e." \
  "FCI arrays in figures/fig_pes_h2o2.py verified against the full-space" \
  "reference at the 9 overlapping geometries: agreement 0.00000 mHa."

note "Production:"
copy_dir "$PD/Production/H2O/final" "$DEST/Production/H2O" \
  "H2O production, five lowest singlets = A1:2 + A2:1 + B1:1 + B2:1." \
  "ratio 5.0, eps_2 = 0, l_max = 1, per geometry:" \
  "  1.0 A : eps_1 1e-2, Ethrsh 1e-4  -> eps_3 5e-4" \
  "  1.5 A : eps_1 2e-2, Ethrsh 1e-4  -> eps_3 5e-4" \
  "  3.0 A : eps_1 5e-2, Ethrsh 1e-3  -> eps_3 5e-3" \
  "Verified vs FULL CAS(10e,7o) FCI: worst |dE| = 0.84 mHa (1.5 A, S3)."
copy_dir "$PD/Production/H2O/final_bench" "$DEST/Production/H2O/bench" \
  "VO measurement benchmark for the production/H2O dumps beside this file." \
  "N_basis 64/64/19 and eps^2 M 0.873/2.68/0.190 at 1.0/1.5/3.0 A." \
  "basis_states cross-checked against n_ucsf for all 12 runs: agree."

copy_dir "$PD/Production/H2O2" "$DEST/Production/H2O2" \
  "H2O2 production, five lowest singlets = A:3 + B:2." \
  "1.5 and 1.875 A : ratio 1.0, eps_1 1e-4, l_max 3" \
  "3.0 A           : ratio 0.1, eps_1 1e-6, l_max 3" \
  "Reproduces tab:h2o2-{energy-errors,sampling,circuits} exactly via" \
  "hpc/collect_h2o2_production.py.  Includes the h2o2_VO_prod_*.json cost files."

note "Scaling:"
copy_dir "$PD/Scaling/H2O" "$DEST/Scaling/H2O" \
  "H2O A1-singlet n-scaling, n = 1..5 at rOH 1.0/1.5/3.0." \
  "ratio 5.0, Ethrsh 2e-6 -> eps_3 1e-5, eps_1 1e-3, eps_2 0, l_max 1." \
  "Verified vs FULL CAS(10e,7o) FCI: worst |dE| = 0.899 mHa (1.0 A, n=4," \
  "root3), every root of every n inside chemical accuracy." \
  "Accuracy is set by the product ratio x Ethrsh: <= 1e-5 passes." \
copy_dir "$PD/Scaling/H2O/bench" "$DEST/Scaling/H2O/bench" \
  "VO measurement benchmark for the Scaling/H2O dumps beside this file." \
  "basis_states cross-checked against n_ucsf for all 15 runs: agree." \
  "1.5 A n=5 has eps^2 M exactly 0: that subspace carries no generators" \
  "(0 of 31 basis states have an ia pair), so every matrix element is" \
  "classically evaluable and no quantum measurement is required."

copy_dir "$PD/Scaling/H2O2" "$DEST/Scaling/H2O2" \
  "H2O2 A-singlet n-scaling, n = 1..5 at rOO 1.5/1.875/3.0." \
  "1.5 and 1.875 A : ratio 1.0, eps_1 1e-4, l_max 3" \
  "3.0 A           : ratio 0.1, eps_1 1e-6, l_max 3" \
  "Verified at 1.5/1.875 A: worst |dE| = 1.265 mHa, 0 roots outside chemical" \
  "accuracy.  n_ucsf matches fig_scaling_h2o2.py's 'basis' exactly."

# --- Hamiltonians + FCI references -----------------------------------------
note "Hamiltonians and FCI references:"
H="$DEST/Hamiltonians"
mkdir -p "$H"
cp -n Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_7o10e_phys_spatial_* "$H"/ 2>/dev/null
cp -n Ham_gen/hamiltonians/ES_Hamiltonians/h2o2_sto3g_12o18e_phys_spatial_* "$H"/ 2>/dev/null
cp -n Ham_gen/hamiltonians/ES_Hamiltonians/*fci_ref_full.pkl "$H"/ 2>/dev/null
{
  echo "CASSCF Hamiltonians consumed by every run in this archive, plus the"
  echo "full-space FCI references used for all accuracy checks."
  echo
  echo "h2o_sto3g_7o10e_phys_spatial_<r>    CAS(10e,7o), symmetry=True."
  echo "h2o2_sto3g_12o18e_phys_spatial_<r>  CAS(18e,12o)."
  echo "h2o_sto3g_fci_ref_full.pkl          per-sector FCI, keys (irrep, spin)"
  echo "h2o2_sto3g_fci_ref_full.pkl         and (irrep, spin, 'cas')."
  echo
  echo "USE THE 2-TUPLE KEY.  (irrep, spin) is the FULL active space; the"
  echo "3-tuple ...,'cas' is the frozen-core sector.  Q-SENSE's actmo_start"
  echo "restricts which excitations the ANSATZ generates, not the Hamiltonian,"
  echo "so full-space FCI is what the ansatz approximates.  Differencing"
  echo "against the frozen-core sector understates the error by ~0.15 mHa and"
  echo "can produce a NEGATIVE dE, which is variationally impossible."
  echo
  echo "Generated by Ham_gen/h2o_sto3g_full.py and Ham_gen/h2o2_sto3g_full.py."
  echo "assembled: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$H/PROVENANCE.txt"
note "  $(printf '%-34s' 'Hamiltonians') $(ls "$H" | grep -vc PROVENANCE) files"

# --- detbasis baselines ------------------------------------------------------
note "Baseline (q-sc-EOM / SG-SSVQE) pickles:"
B="$DEST/Baselines"
mkdir -p "$B"
cp -n ham_rOH_*/h2o_*_gen_T2.pkl "$B"/ 2>/dev/null
cp -n ham_rOO_*/h2o2_*_gen_T2.pkl "$B"/ 2>/dev/null
{
  echo "Output of H2O2_Benchmark_detbasis.py (determinant-basis port), which"
  echo "produces the q-sc-EOM and SG-SSVQE columns of the energy-error tables"
  echo "and their resource estimates."
  echo
  echo "Each pickle holds: config, fci_targets, feed, qsceom, ssvqe, collection,"
  echo "resources, stage_m.  'collection' carries the per-state errors."
  echo
  echo "Its FCI is the FULL active space (the script sets NCAS,NELECAS = 7,10"
  echo "for H2O and 12,18 for H2O2), so its errors are already on the same"
  echo "reference as the Q-SENSE ones in this archive -- checked to 0.0000 mHa."
  echo "assembled: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$B/PROVENANCE.txt"
note "  $(printf '%-34s' 'Baselines') $(ls "$B" | grep -vc PROVENANCE) files"

note ""
note "=============================================================="
du -sh "$DEST"
note "--------------------------------------------------------------"
find "$DEST" -mindepth 1 -maxdepth 2 -type d | sort | while read -r d; do
    printf '  %-40s %s\n' "${d#$DEST/}" "$(du -sh "$d" | cut -f1)"
done
note "=============================================================="
note "Only the cited runs are copied; exploratory sweeps and earlier"
note "parameter sets under QSENSE_ES_dump are left out."
