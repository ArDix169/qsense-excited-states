"""H2O2 five-lowest-singlet cross-method table: accuracy, subspace, cost, circuits.

Joins THREE sources, none of which is sufficient alone:

  1. Q-SENSE energies   QSENSE_ES_dump/h2o2_..._UCSF_<n>_<irrep>_..._C3_..._<r>.json
                        (output_energy), written beside each dump
  2. Q-SENSE cost       QSENSE_ES_dump/h2o2_VO_prod_<irrep><n>_<r>.json
                        (sampling_cost, cx_counts, depth, basis_states)
  3. FCI + baselines    ham_rOO_<r>/h2o2_<r>_gen_T2.pkl from the detbasis
                        notebook -- fci_targets gives the reference spectrum,
                        collection gives per-state q-sc-EOM and SG-SSVQE errors,
                        stage_m gives their eps^2 M

The FCI reference for H2O2 comes from the detbasis run rather than a stored
pkl, because there is no h2o2 analogue of h2o_sto3g_fci_ref_full.pkl -- the B
sector references at 1.875 and 3.0 A did not exist until those runs finished.

The five states split 3 A + 2 B at every geometry (FCI census), so Q-SENSE
targets A/n=3 and B/n=2 and the reported subspace is the SUM of the two.

Usage (from $SCRATCH/Q-SENSE):
    python3 hpc/collect_h2o2_production.py
    DUMPDIR=... HAMROOT=... python3 hpc/collect_h2o2_production.py
"""
import glob
import json
import os
import pickle

DUMPDIR = os.environ.get('DUMPDIR', 'QSENSE_ES_dump')
HAMROOT = os.environ.get('HAMROOT', '.')
GEOMS = (os.environ.get('GEOMS') or '1.5 1.875 3.0').split()

# per-geometry Q-SENSE thresholds, as in run_measurement_benchmark_production.sh
TAGS = {'1.5': '1.0_S0_T1e-04', '1.875': '1.0_S0_T1e-04', '3.0': '0.1_S0_T1e-06'}
SECTORS = [('A', 3), ('B', 2)]


def detbasis(r):
    hits = glob.glob(os.path.join(HAMROOT, f'ham_rOO_{r}', f'h2o2_{r}_gen_T2.pkl'))
    if not hits:
        hits = glob.glob(os.path.join(HAMROOT, '**', f'h2o2_{r}_gen_T2.pkl'),
                         recursive=True)
    return pickle.load(open(hits[0], 'rb')) if hits else None


def qsense_energies(r, irrep, n):
    p = os.path.join(DUMPDIR, f'h2o2_sto3g_12o18e_UCSF_{n}_{irrep}_{TAGS[r]}'
                              f'_C3_for_Arjun_{r}.json')
    return json.load(open(p))['output_energy'] if os.path.exists(p) else None


def qsense_cost(r, irrep, n):
    p = os.path.join(DUMPDIR, f'h2o2_VO_prod_{irrep}{n}_{r}.json')
    return json.load(open(p)) if os.path.exists(p) else None


for r in GEOMS:
    D = detbasis(r)
    print()
    print('=' * 92)
    print(f'  H2O2  rOO = {r} A   -- five lowest singlets')
    print('=' * 92)
    if D is None:
        print('  no detbasis pickle -- FCI reference and baselines unavailable')
        continue

    # FCI spectrum, and the per-irrep ordering Q-SENSE roots map onto
    fci = [(E, nm) for E, nm, _ in D['fci_targets']]
    by_irrep = {}
    for E, nm in fci:
        by_irrep.setdefault(nm, []).append(E)

    # ---- accuracy, per state -------------------------------------------
    print(f"  {'state':>6} {'irrep':>5} {'E_FCI (Ha)':>15} "
          f"{'Q-SENSE':>10} {'q-sc-EOM':>10} {'SG-SSVQE':>10}   |dE| mHa")
    print('  ' + '-' * 74)
    qs_err = {}
    for irrep, n in SECTORS:
        E = qsense_energies(r, irrep, n)
        if E is None:
            continue
        for k in range(min(n, len(by_irrep.get(irrep, [])))):
            qs_err[(irrep, k)] = abs(E[k] - by_irrep[irrep][k]) * 1e3
    coll = {c['state']: c for c in D.get('collection', [])}
    seen = {}
    for idx, (E_fci, nm) in enumerate(fci):
        k = seen.get(nm, 0); seen[nm] = k + 1
        q = qs_err.get((nm, k))
        c = coll.get(idx, {})
        f = lambda v: f'{v:>10.3f}' if isinstance(v, (int, float)) else f'{"--":>10}'
        print(f'  {"S"+str(idx):>6} {nm:>5} {E_fci:>15.8f} '
              f'{f(q)} {f(c.get("qsceom_err_mHa"))} {f(c.get("ssvqe_err_mHa"))}')

    # ---- subspace, cost, circuits ---------------------------------------
    print()
    print(f"  {'method':<12} {'N_basis':>8} {'eps^2 M':>13} "
          f"{'CNOT avg':>9} {'CNOT max':>9} {'depth avg':>10} {'depth max':>10}")
    print('  ' + '-' * 76)

    nb = cost = 0
    cxs, dps, wts = [], [], []
    cxm = dpm = 0
    ok = True
    for irrep, n in SECTORS:
        c = qsense_cost(r, irrep, n)
        if c is None:
            ok = False
            continue
        N = c['basis_states']
        nb += N
        cost += c['sampling_cost']
        # combined average weighted by N^2 -- each sector's mean runs over its
        # own N^2 matrix elements, so a plain mean would over-weight the small
        # sector.  Same convention as the H2O production table.
        wts.append(N * N)
        cxs.append(c['cx_counts']['mean'])
        dps.append(c['depth']['mean'])
        cxm = max(cxm, c['cx_counts']['max'])
        dpm = max(dpm, c['depth']['max'])
    if ok and wts:
        W = sum(wts)
        cxa = sum(m * w for m, w in zip(cxs, wts)) / W
        dpa = sum(m * w for m, w in zip(dps, wts)) / W
        print(f'  {"Q-SENSE":<12} {nb:>8} {cost:>13.4e} '
              f'{cxa:>9.1f} {cxm:>9} {dpa:>10.1f} {dpm:>10}')
    else:
        print(f'  {"Q-SENSE":<12} (VO_prod JSONs incomplete)')

    sm = D.get('stage_m') or {}
    res = D.get('resources') or {}
    cfg = D.get('config', {})

    # N_basis for the baselines is NOT n_params.  n_params is the ansatz
    # parameter count (1110 here) and has nothing to do with a subspace
    # dimension -- quoting it in this column overstates both methods by more
    # than two orders of magnitude.
    #
    #   q-sc-EOM  has a genuine subspace: the EOM excitation manifold, summed
    #             over irrep blocks (qsceom_ranks).  This is the quantity that
    #             gives 44 in the H2O table.
    #   SG-SSVQE  has NO subspace.  It is variational over NSTATE target states,
    #             one circuit each, so the comparable entry is the state count.
    ranks = D.get('qsceom_ranks') or {}
    nb_base = {'qsceom': sum(ranks.values()) if ranks else None,
               'ssvqe': cfg.get('nstate')}
    for lbl, key in (('q-sc-EOM', 'qsceom'), ('SG-SSVQE', 'ssvqe')):
        e2 = sm.get(f'{key}_eps2M')
        if e2 is None:
            continue
        nb_s = f'{nb_base[key]:>8}' if nb_base[key] is not None else f'{"--":>8}'
        print(f'  {lbl:<12} {nb_s} {e2:>13.4e} '
              f'{res.get("cnot", "--"):>9} {res.get("cnot", "--"):>9} '
              f'{res.get("depth", "--"):>10} {res.get("depth", "--"):>10}')
    if ranks:
        print(f'  q-sc-EOM ranks per irrep: {dict(sorted(ranks.items()))}')
    print('  (SG-SSVQE N_basis is the TARGET STATE COUNT -- it has no subspace '
          'expansion.\n   Baseline circuit figures are UPPER BOUNDS from '
          'resources, so avg = max.)')
