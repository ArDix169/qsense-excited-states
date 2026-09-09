"""Collect the H2O PES scan and print every energy.

Reads the JSONs that qsense_subspace.py writes next to each .dump, for the grid

    rdist 0.75..3.0 step 0.25   x   irrep A1/A2/B1/B2   x   singlet/triplet

at the fixed thresholds of runs/run_h2o_pes.sh (eps_1 = 1e-6,
eps_2 = 0, eps_3 = 1e-6, l_max = 2, two states per sector).

Prints three things:

  1. every energy, grouped by geometry, with the subspace size that produced it
  2. the same energies as a PES table, one row per geometry
  3. deviations from FCI when the reference file is present -- that is
     Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl, written
     by Ham_gen/h2o_sto3g_full.py. Uses the FULL CAS(10e,7o) sectors, not the
     frozen-core [CAS(8e,6o)] ones: Q-SENSE's actmo_start only restricts which
     excitations the ansatz generates, it does not reduce the Hamiltonian
     itself, so the Hamiltonian Q-SENSE is actually diagonalizing against is
     the full-space one. Comparing to the frozen-core sector would answer a
     narrower question (how well the ansatz solves its own restricted
     manifold) instead of the true total error. Confirmed against
     analysis/collect_h2o2_production.py, which already used the full FCI (from
     the detbasis run's own fci_targets) and reproduces the manuscript's
     H2O2 energy-error table to 3 decimal places --.

Usage (from the repository root):
    python3 analysis/collect_h2o_pes.py
    python3 analysis/collect_h2o_pes.py --csv h2o_pes.csv
"""
import json
import os
import pickle
import sys

HA2EV = 27.211386245988
CHEM_ACC = 1.6e-3           # Ha

DUMPDIR = os.environ.get('DUMPDIR', 'QSENSE_ES_dump')
# Env-overridable so the collector works from a clone (where the reference
# lives under data/QSENSE_paper_release/Hamiltonians/) as well as from the
# HPC working tree.
REFPATH = os.environ.get(
    'REFPATH', 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl')

GEOMS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
IRREPS = ['A1', 'A2', 'B1', 'B2']
SPINS = [(0, 'singlet'), (2, 'triplet')]

# Must match the batch script.  These appear in the dump filename, so a
# mismatch here silently finds nothing rather than finding the wrong run.
NO_STATES = 2
RATIO = 1.0                 # eps_3 = RATIO * Ethrsh_select_ia = 1.0 * 1e-6
CSF_THRSH = 1e-6
COMBO = 2

# NOTE: Ethrsh_select_ia is NOT in the dump filename, so this pattern cannot
# tell a 1e-6 run from a 1e-4 one at the same ratio.  If both exist, whichever
# was written last is what gets read.


HAM_TAG = 'h2o_sto3g_7o10e'


def path_for(irrep, s_by2, rdist):
    # 7o10e uses symmetry=True orbitals.  A symmetry=False (6o8e) Hamiltonian
    # drifts onto a different electronic state near dissociation, so it must
    # not be substituted here.
    name = (f'{HAM_TAG}_UCSF_{NO_STATES}_{irrep}_{RATIO}'
            f'_S{s_by2}_T{CSF_THRSH:.0e}_C{COMBO}_for_Arjun_{rdist}.json')
    return os.path.join(DUMPDIR, name)


def load(irrep, s_by2, rdist):
    p = path_for(irrep, s_by2, rdist)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_reference():
    """{(rdist, irrep, spin): [E0, E1]} from the full-space sectors, or None."""
    if not os.path.exists(REFPATH):
        return None
    with open(REFPATH, 'rb') as f:
        allref = pickle.load(f)
    out = {}
    for r, rec in allref.items():
        for key, val in rec.items():
            # the full-space (un-frozen) sectors are keyed (irrep, spin)
            if isinstance(key, tuple) and len(key) == 2:
                out[(round(float(r), 3), key[0], key[1])] = list(val)
    return out or None


ref = load_reference()
rows = []
missing = []

print('=' * 78)
print('  H2O PES scan -- all energies (Ha)')
print('  eps_1 = 1e-6   eps_2 = 0   eps_3 = 1e-6   l_max = 2   2 states/sector')
print('=' * 78)

for r in GEOMS:
    print(f'\nrOH = {r} A')
    hdr = (f'  {"sector":<16} {"E0":>16} {"E1":>16} {"n_csf":>6} {"n_ucsf":>7}')
    if ref:
        hdr += f' {"dE0":>9} {"dE1":>9}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for irrep in IRREPS:
        for s_by2, spin in SPINS:
            d = load(irrep, s_by2, r)
            if d is None:
                missing.append((r, irrep, spin))
                print(f'  {irrep + "/" + spin:<16} {"(pending)":>16}')
                continue
            E = list(d['output_energy'])
            E += [float('nan')] * (2 - len(E))
            line = (f'  {irrep + "/" + spin:<16} {E[0]:>16.9f} {E[1]:>16.9f} '
                    f'{d.get("n_csf", -1):>6} {d.get("n_ucsf", -1):>7}')
            dev = [None, None]
            if ref:
                key = (round(float(r), 3), irrep, spin)
                if key in ref:
                    R = ref[key]
                    for k in range(min(2, len(R))):
                        dev[k] = E[k] - R[k]
                    fmt = []
                    for k in range(2):
                        if dev[k] is None:
                            fmt.append(f'{"--":>9}')
                        else:
                            mark = '*' if abs(dev[k]) > CHEM_ACC else ' '
                            fmt.append(f'{dev[k] * 1e3:>8.3f}{mark}')
                    line += ' ' + ' '.join(fmt)
                else:
                    line += f' {"--":>9} {"--":>9}'
            print(line)
            rows.append(dict(rdist=r, irrep=irrep, spin=spin,
                             E0=E[0], E1=E[1],
                             n_csf=d.get('n_csf'), n_ucsf=d.get('n_ucsf'),
                             dE0=dev[0], dE1=dev[1]))

if ref:
    print('\n  dE in mHa vs the full CAS(10e,7o) FCI sector of the SAME orbitals.')
    print(f'  * marks |dE| above chemical accuracy ({CHEM_ACC * 1e3:.1f} mHa).')

# ---- PES table: one row per geometry -------------------------------------
print('\n' + '=' * 78)
print('  PES table (Ha) -- E0 of each sector')
print('=' * 78)
cols = [f'{i}/{s[0]}' for i in IRREPS for s in ('s', 't')]
print(f'{"rOH":>6} ' + ' '.join(f'{c:>14}' for c in cols))
by = {(x['rdist'], x['irrep'], x['spin']): x for x in rows}
for r in GEOMS:
    cells = []
    for irrep in IRREPS:
        for s_by2, spin in SPINS:
            x = by.get((r, irrep, spin))
            cells.append(f'{x["E0"]:>14.8f}' if x else f'{"--":>14}')
    print(f'{r:>6} ' + ' '.join(cells))

# ---- vertical excitations from the global ground state --------------------
print('\n' + '=' * 78)
print('  Vertical excitations (eV) from the lowest state at each geometry')
print('=' * 78)
print(f'{"rOH":>6} {"ground sector":>16} ' +
      ' '.join(f'{i + "/" + s[0]:>10}' for i in IRREPS for s in ('s', 't')))
for r in GEOMS:
    present = [x for x in rows if x['rdist'] == r]
    if not present:
        continue
    gmin = min(present, key=lambda x: x['E0'])
    cells = []
    for irrep in IRREPS:
        for s_by2, spin in SPINS:
            x = by.get((r, irrep, spin))
            cells.append(f'{(x["E0"] - gmin["E0"]) * HA2EV:>10.4f}' if x
                         else f'{"--":>10}')
    print(f'{r:>6} {gmin["irrep"] + "/" + gmin["spin"]:>16} ' + ' '.join(cells))

# ---- summary --------------------------------------------------------------
print('\n' + '=' * 78)
done = len(rows)
print(f'  {done} / {len(GEOMS) * len(IRREPS) * len(SPINS)} sectors present')
if missing:
    print(f'  missing: {missing}')
if ref:
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
    print('  generate it with: cd Ham_gen && python3 h2o_sto3g_full.py')

if '--csv' in sys.argv:
    out = sys.argv[sys.argv.index('--csv') + 1]
    import csv
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n  wrote {out}')
