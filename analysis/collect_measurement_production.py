"""Collect the production VO benchmark and assemble the cross-method table row.

Reads the JSONs written by run_measurement_benchmark_production.sh and combines
the A (n=3) and B (n=2) sectors at each geometry, because the five lowest
singlets of H2O2 split 3 A + 2 B and the reported N_basis is the SUM over
irreps -- matching the H2O table's definition, where q-sc-EOM's 44 is likewise
a global count.

How the per-sector numbers combine:

  N_basis   sum.  Both subspaces must be prepared and measured.
  eps^2 M   sum.  Different subspace bases, so the shot budgets are additive.
  CNOT/depth max over sectors for the max, and an element-count-weighted mean
            for the average -- a plain mean of two means would weight a
            1081-state sector equally with a 1688-state one.

Usage (from the repository root):
    python3 analysis/collect_measurement_production.py
"""
import json
import os
import sys

DUMPDIR = sys.argv[1] if len(sys.argv) > 1 else 'QSENSE_ES_dump'
GEOMS = [('1.5', 'eq.'), ('1.875', 'corr.'), ('3.0', 'diss.')]
SECTORS = [('A', 3), ('B', 2)]


def load(irrep, n, r):
    p = os.path.join(DUMPDIR, f'h2o2_VO_prod_{irrep}{n}_{r}.json')
    return json.load(open(p)) if os.path.exists(p) else None


print("Per sector")
hdr = (f"{'r':>7} {'sec':>4} {'N':>6} {'eps^2 M':>12} {'CX avg':>8} "
       f"{'CX max':>7} {'D avg':>7} {'D max':>7} {'gen avg':>8}")
print(hdr)
print('-' * len(hdr))
for r, _lab in GEOMS:
    for irrep, n in SECTORS:
        d = load(irrep, n, r)
        if d is None:
            print(f'{r:>7} {irrep+str(n):>4}   (pending)')
            continue
        print(f"{r:>7} {irrep+str(n):>4} {d['basis_states']:>6} "
              f"{d['sampling_cost']:>12.4e} {d['cx_counts']['mean']:>8.1f} "
              f"{d['cx_counts']['max']:>7d} {d['depth']['mean']:>7.1f} "
              f"{d['depth']['max']:>7d} {d['generators']['avg']:>8.2f}")

print("\nCombined over irreps  (this is the Table II row)")
hdr2 = (f"{'r':>7} {'N_basis':>8} {'eps^2 M':>12} {'CX avg':>8} {'CX max':>7} "
        f"{'D avg':>7} {'D max':>7}")
print(hdr2)
print('-' * len(hdr2))
for r, lab in GEOMS:
    ds = [load(irrep, n, r) for irrep, n in SECTORS]
    if any(d is None for d in ds):
        print(f'{r:>7}   (incomplete)')
        continue
    N = sum(d['basis_states'] for d in ds)
    cost = sum(d['sampling_cost'] for d in ds)
    # element counts weight the averages: N diagonal + N(N-1)/2 off-diagonal
    wts = [d['basis_states'] * (d['basis_states'] + 1) / 2 for d in ds]
    wsum = sum(wts)
    cx_avg = sum(w * d['cx_counts']['mean'] for w, d in zip(wts, ds)) / wsum
    d_avg = sum(w * d['depth']['mean'] for w, d in zip(wts, ds)) / wsum
    print(f"{r:>7} {N:>8} {cost:>12.4e} {cx_avg:>8.1f} "
          f"{max(d['cx_counts']['max'] for d in ds):>7d} {d_avg:>7.1f} "
          f"{max(d['depth']['max'] for d in ds):>7d}")

print("""
  N_basis is summed over irreps, matching the H2O table where q-sc-EOM's 44 is
  a global count.  eps^2 M is at eps = 1.6 mHa and is eps-independent, so it
  compares directly across methods and systems.  CNOT and depth are per
  measured matrix element and, per the seniority benchmark, count STATE
  PREPARATION ONLY -- the measurement-basis rotations are built there and
  discarded.  The qubit-space baselines report CX_sa + readout, so state which
  convention the table uses.""")
