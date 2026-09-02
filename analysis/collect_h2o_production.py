"""H2O five-lowest-singlet cross-method table: Q-SENSE vs q-sc-EOM vs SG-SSVQE.

The H2O analogue of collect_h2o2_production.py, which did not exist -- the H2O
table was assembled by hand.  Joins three sources:

  1. Q-SENSE energies   QSENSE_paper_data/Production/H2O/<variant>/
                        h2o_sto3g_7o10e_UCSF_<n>_<irrep>_..._<r>.json
                        (output_energy).  Thresholds vary per geometry in the
                        `opt` variant, so the files are GLOBBED, not named.
  2. FCI reference      Ham_gen/.../h2o_sto3g_fci_ref_full.pkl, the FULL
                        CAS(10e,7o) (irrep, 'singlet') key -- NOT the
                        frozen-core (irrep, 'singlet', 'cas') sector:
                        actmo_start restricts the ansatz, not the Hamiltonian,
                        so full-space FCI is what the ansatz approximates.
  3. q-sc-EOM/SG-SSVQE  ham_rOH_<r>/h2o_<r>_gen_T2.pkl from the detbasis run.
                        Its own `collection` errors are already against the
                        full space (that script sets NCAS,NELECAS = 7,10).

The five lowest singlets split A1:2 + A2:1 + B1:1 + B2:1 at these geometries,
so states are pooled across sectors and ordered by energy, matching how the
H2O2 table reports them.

Usage (from $SCRATCH/Q-SENSE):
    python3 hpc/collect_h2o_production.py
    VARIANT=opt python3 hpc/collect_h2o_production.py
"""
import glob
import json
import os
import pickle

VARIANT = os.environ.get('VARIANT', 'baseline')
DUMPDIR = os.environ.get(
    'DUMPDIR', f'QSENSE_paper_data/Production/H2O/{VARIANT}')
HAMROOT = os.environ.get('HAMROOT', '.')
# where the VO measurement benchmark wrote h2o_VO_benchmark_<irrep>_n<n>_r<r>.json;
# unset means energies only, no resource block
RESDIR = os.environ.get('RESDIR', '')
GEOMS = (os.environ.get('BONDLENGTHS') or '1.0 1.5 3.0').split()
# Env-overridable so the collector works from a clone (where the reference
# lives under data/QSENSE_paper_release/Hamiltonians/) as well as from the
# HPC working tree.
REFPATH = os.environ.get(
    'REFPATH', 'Ham_gen/hamiltonians/ES_Hamiltonians/h2o_sto3g_fci_ref_full.pkl')

# (irrep, roots kept) -- the five lowest singlets at these geometries
SECTORS = [('A1', 2), ('A2', 1), ('B1', 1), ('B2', 1)]
HAMTAG = 'h2o_sto3g_7o10e'
CHEM_ACC = 1.6

ref = pickle.load(open(REFPATH, 'rb'))
rk = {round(float(k), 3): v for k, v in ref.items()}


def qsense(irrep, n, r):
    """output_energy for one sector, globbing over the threshold tags."""
    pat = os.path.join(DUMPDIR,
                       f'{HAMTAG}_UCSF_{n}_{irrep}_*_S0_*_for_Arjun_{r}.json')
    hits = sorted(glob.glob(pat))
    if not hits:
        return None, None
    return json.load(open(hits[0]))['output_energy'], os.path.basename(hits[0])


def detbasis(r):
    for pat in (os.path.join(HAMROOT, f'ham_rOH_{r}', f'h2o_{r}_gen_T2.pkl'),
                os.path.join(HAMROOT, '**', f'h2o_{r}_gen_T2.pkl')):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return pickle.load(open(hits[0], 'rb'))
    return None


def fmt(d):
    """Table convention: <0.01, two decimals under 10, else 3 sig figs."""
    if d is None:
        return '--'
    a = abs(d)
    if a < 0.01:
        return '< 0.01'
    if a < 10:
        return f'{a:.2f}'
    return f'{a:.3g}'


print(f'variant: {VARIANT}   dumps: {DUMPDIR}')
print('dE vs FULL CAS(10e,7o) FCI, mHa.  * = outside chemical accuracy.')

for r in GEOMS:
    rec = rk.get(round(float(r), 3), {})
    D = detbasis(r)

    # pool the sectors into one energy-ordered spectrum
    states, missing = [], []
    for irrep, n in SECTORS:
        E, fname = qsense(irrep, n, r)
        R = rec.get((irrep, 'singlet'))
        if E is None or R is None:
            missing.append(irrep)
            continue
        for k in range(min(n, len(E), len(R))):
            states.append({'irrep': irrep, 'root': k,
                           'E': E[k], 'dE': (E[k] - R[k]) * 1e3})
    states.sort(key=lambda s: s['E'])

    print()
    print('=' * 88)
    print(f'  H2O  rOH = {r} A   -- five lowest singlets')
    print('=' * 88)
    if missing:
        print(f'  MISSING sectors: {missing}')

    print(f"  {'state':>6} {'irrep':>6} {'E (Ha)':>16} {'E_FCI (Ha)':>16} "
          f"{'|dE| mHa':>10}")
    print('  ' + '-' * 60)
    for i, s in enumerate(states):
        R = rec[(s['irrep'], 'singlet')][s['root']]
        mark = '*' if abs(s['dE']) > CHEM_ACC else ' '
        print(f"  S{i:<5} {s['irrep']:>6} {s['E']:>16.8f} {R:>16.8f} "
              f"{fmt(s['dE']):>10}{mark}")

    qs_row = [fmt(s['dE']) for s in states]
    print(f"\n  Q-SENSE  |dE| row : {'  '.join(f'{v:>7}' for v in qs_row)}")

    # ---- resources: tab:h2o-sampling and tab:h2o-circuits -------------------
    # Combined exactly as collect_h2o2_production.py does: N_basis and eps^2 M
    # are SUMS (both subspaces must be prepared and measured, and different
    # bases mean additive shot budgets), circuits take the max over sectors and
    # an element-count-weighted mean -- a plain mean of means would weight a
    # 9-state sector equally with a 25-state one.
    if RESDIR:
        rows, miss_b = [], []
        for irrep, n in SECTORS:
            p = os.path.join(RESDIR, f'h2o_VO_benchmark_{irrep}_n{n}_r{r}.json')
            if not os.path.exists(p):
                miss_b.append(f'{irrep}:{n}')
                continue
            rows.append((irrep, n, json.load(open(p))))
        if miss_b:
            print(f'  (benchmark missing for {miss_b})')
        if rows:
            print(f"\n  {'sector':>7} {'N':>5} {'eps^2 M':>12} {'CX avg':>8} "
                  f"{'CX max':>7} {'D avg':>7} {'D max':>7}")
            print('  ' + '-' * 56)
            tot_c = tot_n = 0.0
            wsum = wcx = wdp = 0.0
            mcx = mdp = 0
            for irrep, n, b in rows:
                N = b['basis_states']
                cx, dp = b['cx_counts'], b['depth']
                print(f"  {irrep + str(n):>7} {N:>5} {b['sampling_cost']:>12.4e} "
                      f"{cx['mean']:>8.1f} {cx['max']:>7} "
                      f"{dp['mean']:>7.1f} {dp['max']:>7}")
                tot_c += b['sampling_cost']
                tot_n += N
                w = N * N
                wsum += w
                wcx += w * cx['mean']
                wdp += w * dp['mean']
                mcx = max(mcx, cx['max'])
                mdp = max(mdp, dp['max'])
            print('  ' + '-' * 56)
            print(f"  {'TOTAL':>7} {int(tot_n):>5} {tot_c:>12.4e} "
                  f"{wcx / wsum:>8.1f} {mcx:>7} {wdp / wsum:>7.1f} {mdp:>7}")

    if D is None:
        print('  (no detbasis pkl -- q-sc-EOM / SG-SSVQE unavailable)')
        continue
    col = D['collection']
    q_row = [fmt(c.get('qsceom_err_mHa')) for c in col]
    s_row = [fmt(c.get('ssvqe_err_mHa')) for c in col]
    print(f"  q-sc-EOM |dE| row : {'  '.join(f'{v:>7}' for v in q_row)}")
    print(f"  SG-SSVQE |dE| row : {'  '.join(f'{v:>7}' for v in s_row)}")

    # the detbasis FCI must agree with the pickle's full-space roots, or the
    # two halves of the table are quoting different references
    dev = []
    for c in col:
        rr = rec.get((c['irrep'], 'singlet'))
        if rr is None:
            continue
        dev.append(min(abs(c['E_fci'] - x) * 1e3 for x in rr))
    if dev:
        print(f"  [ref check] detbasis E_fci vs pickle full-space: "
              f"max {max(dev):.4f} mHa")
