"""Collect the H2O2 PES scan and check it against the full-space FCI.

The H2O2 analogue of collect_h2o_pes.py, which did not exist -- the H2O2 PES
figure's Q-SENSE arrays had never been verified against the dumps, only its FCI
arrays had (those agree with h2o2_sto3g_fci_ref_full.pkl to 0.00000 mHa over
the 9 overlapping geometries).

Grid: rOO 1.25..3.0 step 0.25 (8 geometries) x irrep A/B x singlet/triplet,
two states per sector = 32 runs, at ratio 1.0 / eps_1 1e-6 / l_max 3.

Deviations are against the FULL CAS(18e,12o) FCI -- the (irrep, spin) key, NOT
the frozen-core (irrep, spin, 'cas') sector.  actmo_start restricts which
excitations the ansatz generates, not the Hamiltonian, so full-space FCI is
what the ansatz approximates..

Usage (from the repository root):
    DUMPDIR=QSENSE_paper_data/PES/H2O2 python3 analysis/collect_h2o2_pes.py
    python3 analysis/collect_h2o2_pes.py --csv h2o2_pes.csv
"""
import json
import os
import pickle
import sys

HA2EV = 27.211386245988
CHEM_ACC = 1.6e-3

DUMPDIR = os.environ.get('DUMPDIR', 'QSENSE_paper_data/PES/H2O2')
REFPATH = os.environ.get(
    'REFPATH', 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o2_sto3g_fci_ref_full.pkl')

GEOMS = [g for g in (os.environ.get('BONDLENGTHS')
                     or '1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0').split()]
IRREPS = ['A', 'B']
SPINS = [(0, 'singlet'), (2, 'triplet')]

HAM_TAG = 'h2o2_sto3g_12o18e'
NO_STATES = 2
RATIO = os.environ.get('RATIO', '1.0')
CSF_THRSH = float(os.environ.get('CSF_THRSH', 1e-6))
COMBO = os.environ.get('LMAX', '3')


def path_for(irrep, s_by2, r):
    return os.path.join(
        DUMPDIR, f'{HAM_TAG}_UCSF_{NO_STATES}_{irrep}_{RATIO}'
                 f'_S{s_by2}_T{CSF_THRSH:.0e}_C{COMBO}_for_Arjun_{r}.json')


def load(irrep, s_by2, r):
    p = path_for(irrep, s_by2, r)
    return json.load(open(p)) if os.path.exists(p) else None


ref = pickle.load(open(REFPATH, 'rb')) if os.path.exists(REFPATH) else {}
rk = {round(float(k), 3): v for k, v in ref.items()}

rows, missing, empty_ref = [], [], []

print('=' * 84)
print('  H2O2 PES scan -- all energies (Ha)')
print(f'  ratio {RATIO}   eps_1 {CSF_THRSH:.0e}   l_max {COMBO}   '
      f'{NO_STATES} states/sector')
print(f'  source: {DUMPDIR}')
print('=' * 84)

for r in GEOMS:
    rec = rk.get(round(float(r), 3), {})
    print(f'\nrOO = {r} A')
    hdr = (f'  {"sector":<16} {"E0":>16} {"E1":>16} {"n_csf":>7} {"n_ucsf":>8}'
           f' {"dE0":>9} {"dE1":>9}')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for irrep in IRREPS:
        for s_by2, spin in SPINS:
            d = load(irrep, s_by2, r)
            if d is None:
                missing.append((r, irrep, spin))
                print(f'  {irrep + "/" + spin:<16} {"(missing)":>16}')
                continue
            E = list(d['output_energy']) + [float('nan')] * 2
            R = rec.get((irrep, spin))
            # An EMPTY reference sector is "nothing to compare against", not a
            # pass -- this is exactly how H2O2 A/singlet at 3.0 A once read as
            # 0.000 mHa with 0 roots outside chemical accuracy.
            if R is not None and len(R) == 0:
                empty_ref.append((r, irrep, spin))
                R = None
            dev = [None, None]
            cells = []
            for k in range(2):
                if R is not None and k < len(R):
                    dev[k] = E[k] - R[k]
                    mark = '*' if abs(dev[k]) > CHEM_ACC else ' '
                    cells.append(f'{dev[k] * 1e3:>8.3f}{mark}')
                else:
                    cells.append(f'{"--":>9}')
            print(f'  {irrep + "/" + spin:<16} {E[0]:>16.9f} {E[1]:>16.9f} '
                  f'{d.get("n_csf", -1):>7} {d.get("n_ucsf", -1):>8} '
                  + ' '.join(cells))
            rows.append(dict(rdist=r, irrep=irrep, spin=spin, E0=E[0], E1=E[1],
                             n_csf=d.get('n_csf'), n_ucsf=d.get('n_ucsf'),
                             dE0=dev[0], dE1=dev[1]))

print('\n' + '=' * 84)
print('  Vertical excitations (eV) from the lowest state at each geometry')
print('=' * 84)
by = {(x['rdist'], x['irrep'], x['spin']): x for x in rows}
print(f'{"rOO":>6} {"ground":>14} ' +
      ' '.join(f'{i + "/" + s[0]:>10}' for i in IRREPS for s in ('s', 't')))
for r in GEOMS:
    present = [x for x in rows if x['rdist'] == r]
    if not present:
        continue
    gmin = min(present, key=lambda x: x['E0'])
    cells = []
    for irrep in IRREPS:
        for _s, spin in SPINS:
            x = by.get((r, irrep, spin))
            cells.append(f'{(x["E0"] - gmin["E0"]) * HA2EV:>10.4f}'
                         if x else f'{"--":>10}')
    print(f'{r:>6} {gmin["irrep"] + "/" + gmin["spin"]:>14} ' + ' '.join(cells))

print('\n' + '=' * 84)
total = len(GEOMS) * len(IRREPS) * len(SPINS)
print(f'  {len(rows)} / {total} sectors present')
if missing:
    print(f'  missing: {missing}')
if empty_ref:
    print(f'  *** {len(empty_ref)} EMPTY reference sector(s): {empty_ref}')
    print('      no deviation computed for these -- NOT a pass.')
devs = [abs(v) for x in rows for v in (x['dE0'], x['dE1']) if v is not None]
if devs:
    bad = [x for x in rows
           if any(v is not None and abs(v) > CHEM_ACC
                  for v in (x['dE0'], x['dE1']))]
    print(f'  max |dE| = {max(devs) * 1e3:.3f} mHa over {len(devs)} roots')
    print(f'  {len(bad)} sector(s) outside chemical accuracy'
          + (': ' + ', '.join(f'{b["rdist"]}/{b["irrep"]}/{b["spin"]}'
                              for b in bad) if bad else ''))
else:
    print(f'  no FCI reference at {REFPATH} -- deviations not computed')

if '--csv' in sys.argv:
    import csv
    out = sys.argv[sys.argv.index('--csv') + 1]
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n  wrote {out}')
