"""Pick the loosest H2O A1 n-scaling thresholds with ALL n=1..5 roots converged.

Reads runs/run_h2o_scaling_thrsh_sweep.sh's output and, for every (eps_1, eps_3)
pair, reports the worst |dE| over every root of every n at every geometry,
together with the n=5 subspace size -- the quantity fig_scaling_h2o.py plots.

The criterion is strict on purpose: the set replacing QSENSE_paper_data/Scaling/H2O
must have EVERY root of EVERY n inside chemical accuracy, not just the low ones.
The set it replaces passed on roots 0-2 and failed by 60 mHa on root 3 at 3.0 A,
because n_ucsf stopped growing when the 4th state was added.

Deviations are against the FULL CAS(10e,7o) FCI, not the frozen-core sector --
actmo_start restricts the ansatz, not the Hamiltonian..

Usage (from the repository root):
    python3 analysis/collect_h2o_scaling_sweep.py
"""
import glob
import json
import os
import pickle
import re

SWEEPROOT = os.environ.get('SWEEPROOT', 'QSENSE_ES_dump/h2o_scaling_sweep')
REFPATH = 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl'
GEOMS = (os.environ.get('BONDLENGTHS') or '1.0 1.5 3.0').split()
NSTATES = [int(x) for x in (os.environ.get('NSTATES') or '1 2 3 4 5').split()]
HAMTAG = 'h2o_sto3g_7o10e'
IRREP, RATIO, COMBO = 'A1', '5.0', '1'
CHEM_ACC = 1.6

ref = pickle.load(open(REFPATH, 'rb'))
rk = {round(float(k), 3): v for k, v in ref.items()}


def cells():
    out = {}
    for d in sorted(glob.glob(os.path.join(SWEEPROOT, 'eps3_*'))):
        eps3 = os.path.basename(d)[len('eps3_'):]
        for p in glob.glob(os.path.join(d, f'{HAMTAG}_UCSF_*_for_Arjun_*.json')):
            m = re.search(r'_S0_T([0-9.e+-]+)_C', os.path.basename(p))
            if m:
                out[(eps3, m.group(1))] = d
    return out


def evaluate(d, eps1):
    """(worst |dE| mHa, {geom: n_ucsf at max n}, n_missing, worst_locator)."""
    worst, loc, missing, sizes = 0.0, None, 0, {}
    for r in GEOMS:
        R = rk.get(round(float(r), 3), {}).get((IRREP, 'singlet'))
        if R is None or len(R) == 0:
            missing += len(NSTATES)
            continue
        for n in NSTATES:
            # eps_1 pinned: one directory holds every eps_1 at a fixed eps_3,
            # and globbing over T silently reads the same file for every row.
            p = os.path.join(
                d, f'{HAMTAG}_UCSF_{n}_{IRREP}_{RATIO}_S0_T{eps1}'
                   f'_C{COMBO}_for_Arjun_{r}.json')
            if not os.path.exists(p):
                missing += 1
                continue
            j = json.load(open(p))
            if n == max(NSTATES):
                sizes[r] = j['n_ucsf']
            for k, e in enumerate(j['output_energy']):
                if k < len(R):
                    dv = abs(e - R[k]) * 1e3
                    if dv > worst:
                        worst, loc = dv, f'r={r} n={n} root{k}'
    return worst, sizes, missing, loc


C = cells()
if not C:
    raise SystemExit(f'no sweep output under {SWEEPROOT} -- '
                     f'run runs/run_h2o_scaling_thrsh_sweep.sh first')


def key(s):
    try:
        return float(s)
    except ValueError:
        return float('inf')


eps3s = sorted({k[0] for k in C}, key=key)
eps1s = sorted({k[1] for k in C}, key=key)

print('=' * 96)
print('  H2O A1-singlet n-scaling sweep -- worst |dE| (mHa) over ALL n and roots')
print(f'  vs FULL CAS(10e,7o) FCI.  * = some root outside chemical accuracy '
      f'({CHEM_ACC} mHa).')
print('  N is n_ucsf at n=5, summed convention per geometry 1.0/1.5/3.0.')
print('=' * 96)
corner = 'eps_1 \\ eps_3'
print(f'  {corner:>14} ' + ' '.join(f'{e:>26}' for e in eps3s))
print('  ' + '-' * (14 + 27 * len(eps3s)))

results = []
for e1 in eps1s:
    row = []
    for e3 in eps3s:
        d = C.get((e3, e1))
        if d is None:
            row.append(f'{"--":>26}')
            continue
        worst, sizes, missing, loc = evaluate(d, e1)
        if missing:
            row.append(f'{f"({missing} missing)":>26}')
            continue
        mark = '*' if worst > CHEM_ACC else ' '
        sz = '/'.join(str(sizes.get(g, '?')) for g in GEOMS)
        row.append(f'{f"{worst:.3f}{mark}N={sz}":>26}')
        results.append((worst, sizes, e1, e3, loc))
    print(f'  {e1:>14} ' + ' '.join(row))

safe = [x for x in results if x[0] <= CHEM_ACC]
print()
if not safe:
    print('  NO setting in this grid keeps every root inside chemical accuracy.')
    if results:
        b = min(results, key=lambda x: x[0])
        print(f'  closest: eps_1={b[2]} eps_3={b[3]}  worst {b[0]:.3f} mHa ({b[4]})')
        print('  -> tighten the grid (smaller eps_3 and/or eps_1) and rerun.')
else:
    # smallest total subspace among the safe cells: n_ucsf is what the figure
    # plots, and eps^2 M tracks it far better than threshold looseness does
    # (the production sweep showed a looser cell can COST more).
    best = min(safe, key=lambda x: sum(x[1].values()))
    worst, sizes, e1, e3, loc = best
    print(f'  {len(safe)} safe cell(s).  SMALLEST SUBSPACE:')
    print(f'    eps_1 = {e1}   eps_3 = {e3}')
    print(f'    worst |dE| = {worst:.3f} mHa ({loc})')
    print(f'    n_ucsf at n=5 = ' +
          ', '.join(f'{g}:{sizes.get(g)}' for g in GEOMS))
    print()
    print('  all safe cells, by total n=5 subspace:')
    for w, s, a, b, l in sorted(safe, key=lambda x: sum(x[1].values())):
        print(f'    eps_1={a:>6} eps_3={b:>8}  worst {w:>6.3f} mHa  '
              f'N={"/".join(str(s.get(g, "?")) for g in GEOMS)}')
    print()
    print('  NOTE: the winner still needs its VO measurement benchmark before')
    print('  fig_scaling_h2o.py can use it -- BASIS, the COST directory, and the')
    print('  docstring parameters must all describe the same run.')
