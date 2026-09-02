"""Collect the H2O A1-singlet state-count scan: energies and subspace growth.

Reads the JSONs written by hpc/run_h2o_nstates_local.sh (n = 1..5 at each
geometry, eps_1 = 1e-6, eps_2 = 0, eps_3 = 1e-5, l_max = 2) and prints

  1. dE per root against the FULL CAS(10e,7o) FCI sector of the SAME orbitals.
     actmo_start = 1 restricts which excitations the ansatz generates; it does
     not reduce the Hamiltonian, so the full-space FCI is the correct
     reference, not the frozen-core [CAS(8e,6o)] one.
  2. subspace size vs n, with the sector dimension for context
  3. a copy-paste dict laid out like fig_scaling_h2o2.py's `data`

Usage (from the repo root):
    python3 hpc/collect_h2o_nstates.py
    python3 hpc/collect_h2o_nstates.py QSENSE_ES_dump/h2o_A1_nstates/eps3_1e-4
"""
import json
import os
import pickle
import sys

import numpy as np

# positional arg wins, then $DUMPDIR, then the default -- the sibling collectors
# are all env-driven, and a DUMPDIR= that silently did nothing here read as "0/15
# runs present" rather than as a usage error.
DUMPDIR = (sys.argv[1] if len(sys.argv) > 1
           else os.environ.get('DUMPDIR',
                               'QSENSE_ES_dump/h2o_A1_nstates/eps3_1e-05'))
# overridable so the same collector serves H2O2's n-scaling set, whose
# reference pickle and HAMTAG/IRREP/tag conventions differ
REFPATH = os.environ.get(
    'REFPATH', 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl')

GEOMS = [g for g in (os.environ.get('BONDLENGTHS') or '1.0 1.5 3.0').split()]
NSTATES = [int(x) for x in (os.environ.get('NSTATES') or '1 2 3 4 5').split()]
# ONE sector per invocation.  Both the irrep and S_by2 are in the dump name, so
# a directory can hold several -- the runner calls this once per irrep rather
# than the report looping, which would tangle the FCI-reference lookups and the
# copy-paste block below.
IRREP = os.environ.get('IRREP', 'A1')
S_BY2 = int(os.environ.get('S_BY2', 0))
SPIN = 'singlet' if S_BY2 == 0 else 'triplet'
# l_max IS in the dump name as _C<l_max>, so it selects the file
COMBO = int(os.environ.get('LMAX', 2))

# Hamiltonian stem.  MUST match what the runner used -- 6o8e and 7o10e produce
# different dump names for otherwise identical settings, and a mismatch here
# looks exactly like "no runs completed" rather than like a lookup failure.
HAMTAG = os.environ.get('HAMTAG', 'h2o_sto3g_7o10e')
# `ratio` is in the dump name, Ethrsh_select_ia is not -- so this selects the
# file, but eps_3 = RATIO * Ethrsh cannot be recovered from the name alone.
RATIO = float(os.environ.get('RATIO', 1.0))
# must match the run: the dump name encodes eps_1 as _T<%.0e>
CSF_THRSH = float(os.environ.get('CSF_THRSH', 1e-6))
CHEM_ACC = 1.6e-3


def load(n, r):
    p = os.path.join(DUMPDIR,
                     f'{HAMTAG}_UCSF_{n}_{IRREP}_{RATIO}_S{S_BY2}'
                     f'_T{CSF_THRSH:.0e}_C{COMBO}_for_Arjun_{r}.json')
    return json.load(open(p)) if os.path.exists(p) else None


ref = pickle.load(open(REFPATH, 'rb')) if os.path.exists(REFPATH) else {}
rk = {round(float(k), 3): v for k, v in ref.items()}

print('=' * 78)
print(f'  H2O  {IRREP}/{SPIN}   n = {NSTATES}   eps_1 = {CSF_THRSH:.0e}   '
      f'l_max = {COMBO}   ham = {HAMTAG}')
print(f'  source: {DUMPDIR}')
print('  dE vs the full CAS(10e,7o) FCI sector of the same orbitals, mHa.  * > 1.6')
print('=' * 78)

worst, bad, missing, empty_ref = 0.0, [], [], []
for r in GEOMS:
    key = round(float(r), 3)
    R = rk.get(key, {}).get((IRREP, SPIN))
    # An EMPTY reference sector is not "no deviations" -- it is "nothing to
    # compare against", and silently reporting 0.000 mHa / 0 roots outside
    # chemical accuracy reads as a pass.  This is exactly how H2O2 A/singlet at
    # 3.0 A (Davidson returned a batch-level converged=False and every root was
    # discarded) looked like a clean result.  Say so instead.
    if R is not None and len(R) == 0:
        empty_ref.append((r, IRREP, SPIN))
        R = None
    dim = rk.get(key, {}).get('csf_dimensions', {}).get((IRREP, SPIN))
    print(f'\nrOH = {r} A' + (f'     (sector = {dim} CSFs)' if dim else ''))
    print(f'  {"n":>2} {"n_csf":>6} {"n_ucsf":>7} {"n_bas_ia":>9}   ' +
          ' '.join(f'{"dE" + str(k):>9}' for k in range(5)))
    for n in NSTATES:
        d = load(n, r)
        if d is None:
            missing.append((r, n)); print(f'  {n:>2}   (missing)'); continue
        E = d['output_energy']; cells = []
        for k in range(5):
            if R is not None and k < len(E) and k < len(R):
                dv = (E[k] - R[k]) * 1e3
                worst = max(worst, abs(dv))
                if abs(dv) > CHEM_ACC * 1e3:
                    bad.append((r, n, k, dv))
                cells.append(f'{dv:>8.3f}{"*" if abs(dv) > CHEM_ACC*1e3 else " "}')
            else:
                cells.append(f'{"":>9}')
        print(f'  {n:>2} {d["n_csf"]:>6} {d["n_ucsf"]:>7} {d["n_basis_ia"]:>9}   '
              + ' '.join(cells))

print('\n' + '=' * 78)
print('  SUBSPACE SIZE  n_ucsf  vs number of targeted states')
print('=' * 78)
print(f'  {"rOH":>7} ' + ' '.join(f'{"n=" + str(n):>6}' for n in NSTATES)
      + f' {f"n{NSTATES[0]}->n{NSTATES[-1]}":>8} {"per state":>10}')
for r in GEOMS:
    v = [load(n, r) for n in NSTATES]
    if any(x is None for x in v):
        print(f'  {r:>7}   (incomplete)'); continue
    u = [x['n_ucsf'] for x in v]
    print(f'  {r:>7} ' + ' '.join(f'{x:>6}' for x in u)
          # A single-n run has no slope -- len(u)-1 is 0 and dividing blows up.
          + (f' {u[-1]-u[0]:>+8} {(u[-1]-u[0])/(len(u)-1):>+10.2f}'
             if len(u) > 1 else f' {"--":>8} {"--":>10}'))

print('\n' + '=' * 78)
print('  copy-paste (fig_scaling_h2o2.py `data` layout)')
print('=' * 78)
print(f'n_states = {NSTATES}')
print('data = {')
for r in GEOMS:
    v = [load(n, r) for n in NSTATES]
    if any(x is None for x in v):
        continue
    dim = rk.get(round(float(r), 3), {}).get('csf_dimensions', {}).get((IRREP, SPIN))
    u = [x['n_ucsf'] for x in v]
    b = [x['n_basis_ia'] for x in v]
    frac = [round(x / dim, 3) for x in u] if dim else None
    print(f"    {r}: {{'n_ucsf': {u}, 'n_basis_ia': {b}, 'frac_sector': {frac}}},")
print('}')

print('\n' + '=' * 78)
n_have = len(GEOMS) * len(NSTATES) - len(missing)
print(f'  {n_have} / {len(GEOMS)*len(NSTATES)} runs present')
if missing:
    print(f'  missing: {missing}')
if empty_ref:
    print(f'  *** {len(empty_ref)} EMPTY reference sector(s): {empty_ref}')
    print('      no deviation was computed for these -- NOT a pass.')
    print('      regenerate the reference pickle before citing them.')
if rk:
    print(f'  worst |dE| = {worst:.3f} mHa'
          + ('  (over the sectors that HAD a reference)' if empty_ref else ''))
    print(f'  {len(bad)} root(s) outside chemical accuracy'
          + (': ' + ', '.join(f'r={r} n={n} root{k} ({dv:+.2f})'
                              for r, n, k, dv in bad) if bad else ''))
else:
    print(f'  no FCI reference at {REFPATH} -- deviations not computed')
