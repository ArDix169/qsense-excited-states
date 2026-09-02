#!/bin/bash
# H2O A1-singlet state-count scaling: n = 1..5 at three geometries, run locally.
#
#   rdist = 1.0 (equilibrium), 1.5 (strongly correlated), 3.0 (dissociated)
#   n     = 1..5 lowest A1 singlet roots
#                                                  -> 15 runs
#
# 1.5 A is the strongly correlated point from Ham_gen/h2o_hardness.py: it is
# where eps_CISD peaks (22.683 mHa) under a stability-followed broken-symmetry
# UHF reference.  NOTE it is NOT the 2.1 A that Choi et al. report -- their
# value sits on the restricted-UHF branch..
#
# THRESHOLD DEFAULTS (held constant across all 15 runs of one invocation).
# These are the values the script uses with no environment set -- keep this
# block in step with the assignments below, it has drifted before:
#   eps_1 = 1e-4     csf_small_thrsh      override with CSF_THRSH
#   eps_2 = 0        mp2_ampld_thrsh      fixed
#   eps_3 = 1e-5     ratio * Ethrsh_ia    override with RATIO and ETHRSH_IA
#   l_max = 1        combo_order          override with LMAX
#
# WHY eps_3 = 1e-5 AND NOT 1e-4.  At 1e-4 the subspace starves at 3.0 A --
# n_ucsf tops out at 24 of the sector's 37 CSFs -- and the upper roots miss by
# 8.8 mHa (n=4 root 2), 32.0 (n=4 root 3) and 60.1 (n=5 root 3).  At 1e-5 the
# same subspace reaches 31-33 and every root lands within 0.275 mHa.  The cost
# is concentrated where it is needed: +34% subspace at 3.0 A, under +10% at
# equilibrium.
#
# FILENAME COLLISION.  Ethrsh_select_ia is NOT in the dump name, so these runs
# produce filenames identical to the PES sweep's (same ratio, T, C) at n=2.
# The script therefore stashes any pre-existing same-named file, moves its own
# output into OUTDIR, and restores the stash -- so the PES data survives.
#
# Usage:
#   ./hpc/run_h2o_nstates_local.sh              # 4 concurrent jobs
#   ./hpc/run_h2o_nstates_local.sh 8            # 8 concurrent
#   OUTDIR=... BONDLENGTHS="1.5" ./hpc/run_h2o_nstates_local.sh
#   CSF_THRSH=1e-3 ./hpc/run_h2o_nstates_local.sh    # vary eps_1
#   RATIO=5 ETHRSH_IA=1e-6 ./hpc/run_h2o_nstates_local.sh   # eps_3 = 5e-6
#   LMAX=3 ./hpc/run_h2o_nstates_local.sh                   # vary l_max
#   IRREPS="A2 B1 B2" NSTATES=1 ./hpc/run_h2o_nstates_local.sh  # other sectors

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-/usr/local/bin/python3}"          # must be the one with pyscf
HAM="Ham_gen/hamiltonians/ES_Hamiltonians"
JOBS="${1:-4}"

read -r -a bondlengths <<< "${BONDLENGTHS:-1.0 1.5 3.0}"
read -r -a nstates     <<< "${NSTATES:-1 2 3 4 5}"

# --- fixed parameters ---------------------------------------------------
# The Hamiltonian stem appears in the input path AND in every dump name.  It
# used to be spelled out at each site, which is how hpc/collect_h2o_nstates.py
# came to look for 7o10e files while this script wrote 6o8e ones -- every lookup
# missed and the summary reported 0 runs present.  One variable, forwarded to
# the collector below, so they cannot drift again.
# 7o10e by default: 6o8e's symmetry=False orbitals land on the wrong
# electronic state near dissociation.
ham_tag="${HAMTAG:-h2o_sto3g_7o10e}"
# IRREPS may name several.  The irrep IS in the dump name, so sectors coexist
# in one directory and the whole set can be swept in one invocation.
read -r -a irreps <<< "${IRREPS:-A1}"
s_by2="${S_BY2:-0}"     # 0 = singlet, 2 = triplet
mp2_ampld_thrsh=0       # eps_2
csf_small_thrsh="${CSF_THRSH:-1e-4}"    # eps_1
# The dump name carries eps_1 as '_T<%.0e>', so runs at different eps_1 coexist
# in one directory.  DERIVE the tag -- it was hardcoded 'T1e-06', which would
# make any other eps_1 write under the wrong name and collect nothing.
#
# HAZARD: %.0e is ONE significant figure, so nearby thresholds collide --
# 1.5e-2 and 1e-2 BOTH format to "1e-02".  Whichever runs second silently
# overwrites the other's file, with no error.  This bit a full production
# sweep: 1.0 A's A1 sector and 1.5 A's A2/B1 sectors were assembled from a
# 1.5e-2 run mislabeled as 1e-2, and the mistake was only caught by
# cross-checking against a fresh independent computation.  NEVER sweep a
# CSF_THRSH value alongside a neighbour that rounds to the same one-sig-fig
# tag (e.g. 1e-2 and 1.5e-2 together) without widening this to %.1e -- and if
# you do widen it, update collect_h2o_nstates.py, collect_h2o_sectors.py and
# run_meas_bench_h2o_nstates.sh's tag construction to match, or they will
# silently find nothing.
csf_tag="$(printf '%.0e' "$csf_small_thrsh")"
Ethrsh_select_ia="${ETHRSH_IA:-1e-5}"   # with ratio -> eps_3
ratio="${RATIO:-1.0}"
# Canonicalise to Python's float repr.  qsense_subspace.py builds the dump name
# with str(float(argv)), so RATIO=5 becomes "5.0" there while bash keeps "5" --
# the move step then looks for a file that does not exist and silently leaves
# the output behind in QSENSE_ES_dump.
ratio="$("${PY}" -c "print(float('${ratio}'))")"

# eps_3 keys the output directory.  It has to: `ratio` is in the dump name but
# Ethrsh_select_ia is NOT, so (ratio 5, Ethrsh 1e-6) and (ratio 5, Ethrsh 1e-5)
# write byte-identical filenames despite being 5e-6 and 5e-5.  Without an
# eps_3-keyed directory the second run silently destroys the first.
eps3="$("${PY}" -c "print(f'{${ratio}*${Ethrsh_select_ia}:.0e}')")"
# Deferred to here, not to the top: the default path contains $eps3, which is
# not known until ratio and Ethrsh_select_ia have been read and canonicalised.
# Directory is keyed by the irreps swept AND by eps_3.  The A1-only default
# keeps the historical path so existing runs stay where they were.
irrep_tag="$(IFS=-; echo "${irreps[*]}")"
OUTDIR="${OUTDIR:-QSENSE_ES_dump/h2o_${irrep_tag}_nstates/eps3_${eps3}}"
LOGDIR="${LOGDIR:-$OUTDIR/logs}"
# eps_3 = ratio * Ethrsh_select_ia and ONLY THE PRODUCT is physical --
# l_include_ia_in_CAS keeps every in-CAS pair regardless of Ethrsh alone.
# The split still matters for bookkeeping: `ratio` IS in the dump name and
# Ethrsh is NOT, so (5, 1e-6) and (1, 5e-6) give the same physics under
# different filenames, while (1, 1e-5) and (1, 1e-6) would COLLIDE.
combo_order="${LMAX:-1}"   # l_max -- chained combinations up to order l_max,
                           # admitted at (ratio*Ethrsh)^l.  Unlike Ethrsh this
                           # IS in the dump name (_C<l_max>), so runs at
                           # different l_max coexist in one directory.
Uopt_thrsh=1e-6
actmo_start=1           # H2O has ONE core orbital (O 1s)
actmo_end=6             # INCLUSIVE index of the last of seven

mkdir -p "$OUTDIR" "$LOGDIR"
STASH="$(mktemp -d)"

# RESTORE, then remove.  This trap used to be a bare `rm -rf "$STASH"`, which
# permanently DELETED the stashed pre-existing dumps whenever the script did not
# reach its restore step.  Reproduced on an isolated model of this same
# structure (stash -> background children -> parent blocked in `wait`): SIGTERM
# or SIGHUP -- closing the terminal, losing an ssh session -- fires the EXIT
# trap while the parent sits in `wait`, and the stashed files are gone for good.
# On a clean finish the restore below has already emptied $STASH, so this is a
# no-op there.
restore_stash () {
    if [ -n "${STASH:-}" ] && [ -n "$(ls -A "$STASH" 2>/dev/null)" ]; then
        mv "$STASH"/* QSENSE_ES_dump/ 2>/dev/null \
            && echo "restored stashed file(s) from $STASH"
    fi
    rm -rf "$STASH"
}
trap restore_stash EXIT
trap 'echo; echo "interrupted -- restoring stashed dumps" >&2; exit 130' INT TERM HUP

dumpname () {   # $1 = n, $2 = rdist, $3 = irrep
    local irrep="$3"
    echo "${ham_tag}_UCSF_${1}_${irrep}_${ratio}_S${s_by2}_T${csf_tag}_C${combo_order}_for_Arjun_${2}"
}

echo "output   -> $OUTDIR"
echo "eps_1=$csf_small_thrsh (tag $csf_tag)  eps_2=$mp2_ampld_thrsh  "\
"eps_3=$ratio*$Ethrsh_select_ia=$eps3  l_max=$combo_order"
echo "irreps: ${irreps[*]}  S_by2=$s_by2   geometries: ${bondlengths[*]}"
echo "n: ${nstates[*]}   concurrency: $JOBS"

# stash any pre-existing dumps that these runs would overwrite
for r in "${bondlengths[@]}"; do for n in "${nstates[@]}"; do
  for irr in "${irreps[@]}"; do
    for ext in dump json; do
        f="QSENSE_ES_dump/$(dumpname "$n" "$r" "$irr").$ext"
        [ -f "$f" ] && mv "$f" "$STASH/"
    done
  done
done; done
echo "stashed $(ls "$STASH" 2>/dev/null | wc -l | tr -d ' ') pre-existing file(s)"

fail=0
run_one () {
    local r="$1" n="$2" irrep="$3"
    "$PY" -u qsense_subspace.py "$HAM/${ham_tag}_phys_spatial_${r}" \
        "$mp2_ampld_thrsh" 1 "$actmo_start" "$actmo_end" "$r" \
        "$Ethrsh_select_ia" "$Uopt_thrsh" False True "$actmo_start" "$actmo_end" \
        "$irrep" "$n" "$ratio" "$combo_order" "$s_by2" "$csf_small_thrsh" \
        > "$LOGDIR/${irrep}_n${n}_r${r}.out" 2>&1
    echo "$? irrep=${irrep} n=${n} r=${r}" >> "$LOGDIR/status.txt"
}

: > "$LOGDIR/status.txt"
k=0
for r in "${bondlengths[@]}"; do
    if [ ! -f "$HAM/${ham_tag}_phys_spatial_${r}" ]; then
        echo "ERROR: no Hamiltonian for rOH=${r}" >&2
        echo "  generate it: cd Ham_gen && $PY h2o_sto3g_full.py ${r}" >&2
        fail=1; continue
    fi
    for n in "${nstates[@]}"; do
      for irr in "${irreps[@]}"; do
        run_one "$r" "$n" "$irr" &
        k=$((k+1)); if (( k % JOBS == 0 )); then wait; fi
      done
    done
done
wait

# collect this run's output, then put the stashed files back
for r in "${bondlengths[@]}"; do for n in "${nstates[@]}"; do
  for irr in "${irreps[@]}"; do
    for ext in dump json; do
        f="QSENSE_ES_dump/$(dumpname "$n" "$r" "$irr").$ext"
        [ -f "$f" ] && mv "$f" "$OUTDIR/"
    done
  done
done; done
restore_stash

bad=$(awk '$1!=0' "$LOGDIR/status.txt" | wc -l | tr -d ' ')
echo "done: $(wc -l < "$LOGDIR/status.txt" | tr -d ' ') runs, ${bad} nonzero exit(s)"

# --- summary: n_states vs n_ucsf vs dE ----------------------------------
# Delegated to the collector rather than reimplemented here, so the table the
# run prints and the table you get later from the same JSONs cannot drift.
echo
# One collector run per irrep -- the report is written for a single sector.
for irr in "${irreps[@]}"; do
    RATIO="$ratio" CSF_THRSH="$csf_small_thrsh" LMAX="$combo_order" \
    HAMTAG="$ham_tag" NSTATES="${nstates[*]}" BONDLENGTHS="${bondlengths[*]}" \
    IRREP="$irr" S_BY2="$s_by2" \
        "$PY" "$REPO/hpc/collect_h2o_nstates.py" "$OUTDIR" || true
done
[ "$bad" -eq 0 ] && [ "$fail" -eq 0 ]
