"""Pick the LOOSEST H2O production thresholds still inside chemical accuracy.

Reads the sweep written by hpc/run_h2o_prod_thrsh_sweep.sh and, for every
(eps_1, eps_3) pair, reports the worst |dE| over the five lowest singlets at
each geometry together with the subspace size that produced it.

The five states are A1:2 + A2:1 + B1:1 + B2:1, pooled and ordered by energy,
matching tab:h2o-energy-errors.  Deviations are against the FULL CAS(10e,7o)
FCI, NOT the frozen-core (irrep, spin, 'cas') sector -- actmo_start restricts
which excitations the ansatz generates, not the Hamiltonian, so full-space FCI
is what the ansatz is approximating.  See HANDOFF.md.

The winner is chosen PER GEOMETRY, because that is what the existing production
sets already do (`opt` runs eps_1 = 1e-2 at 1.0/1.5 A and 5e-2 at 3.0 A) and
because the dissociated geometry behaves differently from the bound ones.

Usage (from $SCRATCH/Q-SENSE):
    python3 hpc/collect_h2o_prod_sweep.py
    SWEEPROOT=QSENSE_ES_dump/h2o_prod_sweep python3 hpc/collect_h2o_prod_sweep.py
"""
import glob
import json
import os
import pickle
import re

SWEEPROOT = os.environ.get('SWEEPROOT', 'QSENSE_ES_dump/h2o_prod_sweep')
REFPATH = 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl'
GEOMS = (os.environ.get('BONDLENGTHS') or '1.0 1.5 3.0').split()
SECTORS = [('A1', 2), ('A2', 1), ('B1', 1), ('B2', 1)]
HAMTAG = 'h2o_sto3g_7o10e'
RATIO = '5.0'
COMBO = '1'
CHEM_ACC = 1.6

ref = pickle.load(open(REFPATH, 'rb'))
rk = {round(float(k), 3): v for k, v in ref.items()}


def sweep_cells():
    """{(eps3, eps1): dirpath} discovered from the sweep tree."""
    out = {}
    for d in sorted(glob.glob(os.path.join(SWEEPROOT, 'eps3_*'))):
        eps3 = os.path.basename(d)[len('eps3_'):]
        for p in glob.glob(os.path.join(d, f'{HAMTAG}_UCSF_*_for_Arjun_*.json')):
            m = re.search(r'_S0_T([0-9.e+-]+)_C', os.path.basename(p))
            if m:
                out[(eps3, m.group(1))] = d
    return out


def spectrum(d, r, eps1):
    """Pooled five lowest singlets: (worst |dE| mHa, N_basis, n_missing).

    eps1 is the _T<tag> as it appears in the filename and MUST be pinned: one
    sweep directory holds every eps_1 at a fixed eps_3, so globbing over T and
    taking the first hit silently reads the same file for every row (which made
    the whole grid identical, and read '1e-01' throughout because that sorts
    before '1e-02').
    """
    rec = rk.get(round(float(r), 3), {})
    states, nbasis, missing = [], 0, 0
    for irrep, n in SECTORS:
        pat = os.path.join(
            d, f'{HAMTAG}_UCSF_{n}_{irrep}_{RATIO}_S0_T{eps1}'
               f'_C{COMBO}_for_Arjun_{r}.json')
        hits = sorted(glob.glob(pat))
        R = rec.get((irrep, 'singlet'))
        if not hits or R is None:
            missing += 1
            continue
        j = json.load(open(hits[0]))
        nbasis += j.get('n_ucsf', 0)
        E = j['output_energy']
        for k in range(min(n, len(E), len(R))):
            states.append((E[k], abs(E[k] - R[k]) * 1e3))
    if not states:
        return None, 0, missing
    states.sort()
    return max(d_ for _, d_ in states), nbasis, missing


cells = sweep_cells()
if not cells:
    raise SystemExit(f'no sweep results under {SWEEPROOT} -- '
                     f'run hpc/run_h2o_prod_thrsh_sweep.sh first')


def eps_key(s):
    try:
        return float(s)
    except ValueError:
        return float('inf')


eps3s = sorted({k[0] for k in cells}, key=eps_key)
eps1s = sorted({k[1] for k in cells}, key=eps_key)

for r in GEOMS:
    print()
    print('=' * 78)
    print(f'  rOH = {r} A   -- worst |dE| (mHa) over the five lowest singlets')
    print(f'  vs FULL CAS(10e,7o) FCI.  * = outside chemical accuracy '
          f'({CHEM_ACC} mHa).')
    print('=' * 78)
    # kept out of the f-string: a backslash inside an f-string expression is a
    # SyntaxError before Python 3.12, and Trillium's module is 3.11
    corner = 'eps_1 \\ eps_3'
    print(f'  {corner:>16} ' + ' '.join(f'{e:>16}' for e in eps3s))
    print('  ' + '-' * (16 + 17 * len(eps3s)))
    for e1 in eps1s:
        row = []
        for e3 in eps3s:
            d = cells.get((e3, e1))
            if d is None:
                row.append(f'{"--":>16}')
                continue
            worst, nb, miss = spectrum(d, r, e1)
            if worst is None:
                row.append(f'{"(missing)":>16}')
                continue
            mark = '*' if worst > CHEM_ACC else ' '
            tag = f'{worst:.3f}{mark}N={nb}'
            if miss:
                tag += f'!{miss}'
            row.append(f'{tag:>16}')
        print(f'  {e1:>16} ' + ' '.join(row))

    # loosest safe cell: largest eps_1 x eps_3 product still inside the bound,
    # tie-broken by the smaller subspace, which is what the cost actually tracks
    best = None
    for e3 in eps3s:
        for e1 in eps1s:
            d = cells.get((e3, e1))
            if d is None:
                continue
            worst, nb, miss = spectrum(d, r, e1)
            if worst is None or miss or worst > CHEM_ACC:
                continue
            score = (eps_key(e1) * eps_key(e3), -nb)
            if best is None or score > best[0]:
                best = (score, e1, e3, worst, nb)
    if best is None:
        print('\n  no setting in this grid stays inside chemical accuracy')
    else:
        _, e1, e3, worst, nb = best
        print(f'\n  LOOSEST SAFE: eps_1 = {e1}   eps_3 = {e3}   '
              f'worst |dE| = {worst:.3f} mHa   N_basis = {nb}')
