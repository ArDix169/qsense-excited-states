"""Pick H2O A1 n-scaling thresholds that are accurate AND carry real generators.

Reads hpc/run_h2o_scaling_ratio_sweep.sh and reports, for every
(Ethrsh, ratio) cell, THREE things that have to be sound together:

  worst |dE|   over every root of every n, vs FULL CAS(10e,7o) FCI
  n_basis_ia   ia pairs above ratio*Ethrsh -- the BASIS EXTENSION count
  generators   n_valid_ia - n_basis_ia     -- pairs left as GENERATORS

The previous single-axis sweep optimised the first alone and produced a cell
with essentially no generators, whose eps^2 M was 0.0000 at 1.5 A n=5 and FELL
with increasing n at 3.0 A.  A cell is only usable if it is inside chemical
accuracy AND keeps a non-trivial generator count at every (n, geometry) --
otherwise the measurement-cost and circuit numbers describe a degenerate
subspace rather than the method.

Usage (from $SCRATCH/Q-SENSE):
    python3 hpc/collect_h2o_scaling_ratio_sweep.py
"""
import glob
import json
import os
import pickle
import re

ROOT = os.environ.get('SWEEPROOT', 'QSENSE_ES_dump/h2o_scaling_ratio_sweep')
REFPATH = 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl'
GEOMS = (os.environ.get('BONDLENGTHS') or '1.0 1.5 3.0').split()
NSTATES = [int(x) for x in (os.environ.get('NSTATES') or '1 2 3 4 5').split()]
HAMTAG = 'h2o_sto3g_7o10e'
IRREP, COMBO = 'A1', '1'
CHEM_ACC = 1.6
MIN_GEN = int(os.environ.get('MIN_GEN', 1))   # generators required everywhere

ref = pickle.load(open(REFPATH, 'rb'))
rk = {round(float(k), 3): v for k, v in ref.items()}


def cells():
    """{(eth, ratio): dir} from the sweep tree."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, 'eth_*'))):
        eth = os.path.basename(d)[len('eth_'):]
        for p in glob.glob(os.path.join(d, f'{HAMTAG}_UCSF_*_for_Arjun_*.json')):
            m = re.search(rf'_UCSF_\d+_{IRREP}_([\d.]+)_S0_T', os.path.basename(p))
            if m:
                out[(eth, m.group(1))] = d
    return out


def evaluate(d, ratio):
    worst, loc, missing = 0.0, None, 0
    min_gen = None
    sizes = {}
    for r in GEOMS:
        R = rk.get(round(float(r), 3), {}).get((IRREP, 'singlet'))
        if R is None or len(R) == 0:
            missing += len(NSTATES)
            continue
        for n in NSTATES:
            pat = os.path.join(
                d, f'{HAMTAG}_UCSF_{n}_{IRREP}_{ratio}_S0_T*_C{COMBO}'
                   f'_for_Arjun_{r}.json')
            hits = sorted(glob.glob(pat))
            if not hits:
                missing += 1
                continue
            j = json.load(open(hits[0]))
            if n == max(NSTATES):
                sizes[r] = j['n_ucsf']
            gen = (j.get('n_valid_ia') or 0) - (j.get('n_basis_ia') or 0)
            min_gen = gen if min_gen is None else min(min_gen, gen)
            for k, e in enumerate(j['output_energy']):
                if k < len(R):
                    dv = abs(e - R[k]) * 1e3
                    if dv > worst:
                        worst, loc = dv, f'r={r} n={n} root{k}'
    return worst, sizes, (min_gen if min_gen is not None else 0), missing, loc


C = cells()
if not C:
    raise SystemExit(f'no sweep output under {ROOT} -- run the sweep first')


def fkey(s):
    try:
        return float(s)
    except ValueError:
        return float('inf')


eths = sorted({k[0] for k in C}, key=fkey)
ratios = sorted({k[1] for k in C}, key=fkey)

print('=' * 104)
print('  H2O A1 n-scaling:  Ethrsh (generator thrsh)  x  ratio (gen/basis split)')
print(f'  cell = worst |dE| mHa / min generators over all (n, geom) / n_ucsf at n=5')
print(f'  * = a root outside chemical accuracy ({CHEM_ACC} mHa);'
      f'  ! = generators fell below {MIN_GEN}')
print('=' * 104)
corner = 'Ethrsh \\ ratio'
print(f'  {corner:>14} ' + ' '.join(f'{x:>17}' for x in ratios))
print('  ' + '-' * (14 + 18 * len(ratios)))

good = []
for eth in eths:
    row = []
    for rt in ratios:
        d = C.get((eth, rt))
        if d is None:
            row.append(f'{"--":>17}')
            continue
        worst, sizes, gen, missing, loc = evaluate(d, rt)
        if missing:
            row.append(f'{f"({missing} miss)":>17}')
            continue
        m1 = '*' if worst > CHEM_ACC else ''
        m2 = '!' if gen < MIN_GEN else ''
        row.append(f'{f"{worst:.2f}{m1}/g{gen}{m2}":>17}')
        if worst <= CHEM_ACC and gen >= MIN_GEN:
            good.append((sum(sizes.values()), worst, gen, eth, rt, sizes, loc))
    print(f'  {eth:>14} ' + ' '.join(row))

print()
if not good:
    print('  NO cell is both inside chemical accuracy and generator-carrying.')
    print('  Widen the grid: smaller Ethrsh for accuracy, larger ratio for generators.')
else:
    print(f'  {len(good)} usable cell(s), smallest total n=5 subspace first:')
    for tot, worst, gen, eth, rt, sizes, loc in sorted(good):
        print(f'    Ethrsh={eth:>7}  ratio={rt:>5}  worst {worst:>6.3f} mHa ({loc})'
              f'  min gen={gen:>4}  N=' +
              '/'.join(str(sizes.get(g, '?')) for g in GEOMS))
    print()
    print('  Pick from these by eps^2 M, not by subspace size -- the production')
    print('  sweep showed a smaller subspace can cost MORE.  Run the VO benchmark')
    print('  on the top candidates before committing.')
