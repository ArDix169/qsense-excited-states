"""Join the H2O production sectors' ACCURACY and MEASUREMENT COST in one table.

The two live in different files and neither alone is enough to judge a sector:

  accuracy      the Q-SENSE run JSON written beside each .dump (output_energy),
                differenced against the FULL CAS(10e,7o) FCI sector of the SAME
                orbitals in h2o_sto3g_fci_ref_full.pkl -- not the frozen-core
                [CAS(8e,6o)] sector: Q-SENSE's actmo_start only restricts the
                ansatz's excitation manifold, it does not reduce the
                Hamiltonian, so the full-space FCI is the correct reference. and analysis/collect_h2o2_production.py, which
                already used the full FCI and reproduces the manuscript's
                H2O2 table exactly.
  cost          the VO benchmark JSON in RESDIR (sampling_cost, cx_counts, ...)

A sector is only interesting if it is BOTH inside chemical accuracy and cheap,
and a table showing one without the other invites quoting a cost that was never
converged -- or dismissing a subspace that was.

Usage:
    SECTORS="A1:2 A2:1 B1:1 B2:1" \\
    DUMPDIR=QSENSE_ES_dump/h2o_production \\
    RESDIR=results_h2o_sectors \\
    python3 analysis/collect_h2o_sectors.py
"""
import json
import os
import pickle

CHEM_ACC = 1.6e-3

DUMPDIR = os.environ.get('DUMPDIR', 'QSENSE_ES_dump/h2o_production')
RESDIR = os.environ.get('RESDIR', '')
GEOMS = (os.environ.get('BONDLENGTHS') or '1.0 1.5 3.0').split()
SECTORS = (os.environ.get('SECTORS') or 'A1:2 A2:1 B1:1 B2:1').split()
TASKS = [(s.split(':')[0], int(s.split(':')[1])) for s in SECTORS]

HAMTAG = os.environ.get('HAMTAG', 'h2o_sto3g_7o10e')
RATIO = os.environ.get('RATIO', '5.0')
CSF_TAG = os.environ.get('CSF_TAG', '1e-03')
COMBO = os.environ.get('COMBO', '1')
S_BY2 = os.environ.get('S_BY2', '0')
SPIN = 'singlet' if S_BY2 == '0' else 'triplet'

REFPATH = os.environ.get(
    'REFPATH', 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl')
ref = pickle.load(open(REFPATH, 'rb')) if os.path.exists(REFPATH) else {}


def qsense_json(irrep, n, r):
    return os.path.join(
        DUMPDIR, f'{HAMTAG}_UCSF_{n}_{irrep}_{RATIO}_S{S_BY2}_T{CSF_TAG}'
                 f'_C{COMBO}_for_Arjun_{r}.json')


def bench_json(irrep, n, r):
    return os.path.join(RESDIR, f'h2o_VO_benchmark_{irrep}_n{n}_r{r}.json')


def fci_roots(r, irrep):
    rec = ref.get(round(float(r), 3), {})
    v = rec.get((irrep, SPIN))
    return list(v) if v is not None else None


hdr = (f"{'rOH':>6} {'irrep':>6} {'n':>2} {'basis':>6} {'eps^2 M':>12} "
       f"{'gen avg':>8} {'worst dE':>10}  roots (mHa)")
print()
print('=' * 96)
print(f'  H2O production sectors -- accuracy and measurement cost')
print(f'  dumps: {DUMPDIR}')
print(f'  cost : {RESDIR or "(RESDIR unset -- accuracy only)"}')
print('=' * 96)
print(hdr)
print('-' * len(hdr))

rows, bad = [], []
for r in GEOMS:
    for irrep, n in TASKS:
        qp, bp = qsense_json(irrep, n, r), bench_json(irrep, n, r)
        if not os.path.exists(qp):
            print(f'{r:>6} {irrep:>6} {n:>2}   (no Q-SENSE JSON)')
            continue
        q = json.load(open(qp))
        E = list(q['output_energy'])
        R = fci_roots(r, irrep)
        dev = [E[k] - R[k] for k in range(min(len(E), len(R)))] if R else []
        worst = max(dev, key=abs) if dev else float('nan')

        cost = gen = None
        if RESDIR and os.path.exists(bp):
            b = json.load(open(bp))
            cost, gen = b['sampling_cost'], b['generators']['avg']

        cstr = f'{cost:>12.4e}' if cost is not None else f'{"--":>12}'
        gstr = f'{gen:>8.2f}' if gen is not None else f'{"--":>8}'
        mark = '*' if dev and abs(worst) > CHEM_ACC else ' '
        roots = ' '.join(f'{d * 1e3:+.3f}' for d in dev) if dev else '--'
        print(f'{r:>6} {irrep:>6} {n:>2} {q["n_ucsf"]:>6} {cstr} {gstr} '
              f'{worst * 1e3:>+9.3f}{mark}  {roots}')
        rows.append(dict(r=r, irrep=irrep, n=n, basis=q['n_ucsf'],
                         cost=cost, gen=gen, worst=worst))
        if dev and abs(worst) > CHEM_ACC:
            bad.append((r, irrep, n))
    print()

print(f'  * marks |dE| above chemical accuracy ({CHEM_ACC * 1e3:.1f} mHa)')
if rows:
    w = [abs(x['worst']) for x in rows if x['worst'] == x['worst']]
    if w:
        print(f'  worst |dE| over all sectors = {max(w) * 1e3:.3f} mHa'
              + (f'   -- {len(bad)} sector(s) outside: {bad}' if bad
                 else '   -- every sector inside chemical accuracy'))

    print()
    print('  per-geometry totals (the targeted spectrum as a whole):')
    for r in GEOMS:
        v = [x for x in rows if x['r'] == r]
        if len(v) != len(TASKS):
            continue
        have_cost = all(x['cost'] is not None for x in v)
        tot = f"{sum(x['cost'] for x in v):.4e}" if have_cost else '--'
        wv = max(abs(x['worst']) for x in v) * 1e3
        print(f'    {r:>6} A   N_basis = {sum(x["basis"] for x in v):>4}   '
              f'total eps^2 M = {tot:>12}   worst dE = {wv:.3f} mHa')
