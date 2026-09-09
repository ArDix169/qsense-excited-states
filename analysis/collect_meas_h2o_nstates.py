"""Collect the H2O A1-singlet VO measurement benchmark across n and geometry.

Reads the JSONs written by runs/run_meas_bench_h2o_nstates.sh --

    <RESDIR>/h2o_VO_benchmark_n<n>_r<rOH>.json

-- and prints, per (geometry, n): the subspace dimension, the sampling cost
eps^2 M, the CNOT and depth statistics, and the generators-per-state average.

The last of those is the quantity the H2O2 result turned on: measurement cost
tracks RETAINED QUANTUM STRUCTURE, not subspace dimension, so a larger subspace
with fewer generators per state can be cheaper to measure than a smaller one.

Usage:
    python3 analysis/collect_meas_h2o_nstates.py results_h2o_nstates
    NSTATES="1 3 5" BONDLENGTHS="1.0 3.0" python3 analysis/collect_meas_h2o_nstates.py <dir>
"""
import json
import os
import sys

RESDIR = sys.argv[1] if len(sys.argv) > 1 else 'results_h2o_nstates'

NSTATES = [int(x) for x in os.environ.get('NSTATES', '1 2 3 4 5').split()]
GEOMS = os.environ.get('BONDLENGTHS', '1.0 1.5 3.0').split()

# SECTORS switches to the production layout: an explicit "<irrep>:<n>" list,
# whose results are named h2o_VO_benchmark_<irrep>_n<n>_r<r>.json.  Without it
# the n-scaling layout is assumed, h2o_VO_benchmark_n<n>_r<r>.json, which is
# what a single-irrep sweep writes.
SECTORS = os.environ.get('SECTORS', '').split()
if SECTORS:
    TASKS = [(s.split(':')[0], int(s.split(':')[1])) for s in SECTORS]
else:
    TASKS = [(None, n) for n in NSTATES]


def result_path(resdir, irrep, n, r):
    tag = f'{irrep}_n{n}_r{r}' if irrep else f'n{n}_r{r}'
    return os.path.join(resdir, f'h2o_VO_benchmark_{tag}.json')

# Optional "% of the sector" column.  Not defaulted, because the A1/singlet CSF
# count of CAS(8e,6o) is not something to assert from memory -- the full singlet
# space is 105 by Weyl-Paldus, but its split over A1/A2/B1/B2 depends on the
# orbital irreps at each geometry.  Set CAS_DIM to turn the column on.
CAS_DIM = int(os.environ['CAS_DIM']) if os.environ.get('CAS_DIM') else None

_pct_hdr = f"{'%sec':>6} " if CAS_DIM else ''
_sec_hdr = f"{'irrep':>6} " if SECTORS else ''
hdr = (f"{'rOH':>6} " + _sec_hdr + f"{'n':>2} {'basis':>6} " + _pct_hdr + f"{'eps^2 M':>13} "
       f"{'CNOT avg':>10} {'CNOT max':>9} {'depth avg':>10} {'depth max':>10} "
       f"{'gen avg':>8} {'qubits':>7}")
print()
print('=' * len(hdr))
print('  H2O CAS(8e,6o) A1 singlet -- VO measurement / circuit benchmark')
print(f'  results: {RESDIR}')
print('=' * len(hdr))
print(hdr)
print('-' * len(hdr))

rows = []
missing = []
for r in GEOMS:
    for irrep, n in TASKS:
        p = result_path(RESDIR, irrep, n, r)
        sec = f'{irrep:>6} ' if SECTORS else ''
        if not os.path.exists(p):
            missing.append((r, irrep, n))
            print(f'{r:>6} ' + sec + f'{n:>2}   (pending)')
            continue
        with open(p) as f:
            d = json.load(f)
        N = d['basis_states']
        cost = d['sampling_cost']
        cx = d['cx_counts']
        dp = d['depth']
        gen = d['generators']
        pct = f'{100.0 * N / CAS_DIM:>5.1f}% ' if CAS_DIM else ''
        print(f'{r:>6} ' + sec + f'{n:>2} {N:>6} ' + pct + f'{cost:>13.4e} '
              f'{cx["mean"]:>10.1f} {cx["max"]:>9} {dp["mean"]:>10.1f} '
              f'{dp["max"]:>10} {gen["avg"]:>8.2f} {d["n_qubits"]:>7}')
        rows.append(dict(rdist=r, irrep=irrep, n=n, basis=N, cost=cost,
                         cx_mean=cx['mean'], cx_max=cx['max'],
                         depth_mean=dp['mean'], depth_max=dp['max'],
                         gen_avg=gen['avg'],
                         energies=d.get('energies', [])))
    print()

print(f'  {len(rows)} / {len(GEOMS) * len(TASKS)} runs present')

if SECTORS and rows:
    # A production set is summed, not compared: the reported N_basis and cost
    # are the TOTAL over the sectors making up the targeted spectrum.
    print()
    print('  per-geometry totals over all sectors:')
    for r in GEOMS:
        v = [x for x in rows if x['rdist'] == r]
        if len(v) != len(TASKS):
            print(f'    {r:>6} A   incomplete ({len(v)}/{len(TASKS)})')
            continue
        parts = ', '.join(f'{x["irrep"]}:{x["basis"]}' for x in v)
        print(f'    {r:>6} A   N_basis = {sum(x["basis"] for x in v):>4}   '
              f'total eps^2 M = {sum(x["cost"] for x in v):.4e}   ({parts})')
if missing:
    print(f'  missing: {missing}')

# ---- monotonicity check ---------------------------------------------------
# The H2O2 table claimed every series non-increasing in n except one depth max.
# Say what is actually true here rather than assuming the same holds.
if rows:
    print()
    print('  behaviour in n, per geometry (cost / CNOT max / depth max):')
    for r in GEOMS:
        ser = [x for x in rows if x['rdist'] == r]
        ser.sort(key=lambda x: x['n'])
        if len(ser) < 2:
            continue
        def trend(key):
            v = [x[key] for x in ser]
            if all(b <= a for a, b in zip(v, v[1:])):
                return 'non-increasing'
            if all(b >= a for a, b in zip(v, v[1:])):
                return 'non-decreasing'
            return 'non-monotone'
        print(f'    {r:>6} A   cost {trend("cost"):<14} '
              f'CNOT max {trend("cx_max"):<14} depth max {trend("depth_max")}')

# ---- copy-paste block for the figure --------------------------------------
if rows:
    print()
    print('  data block for figures/fig_scaling_h2o.py:')
    print('    data = {')
    for r in GEOMS:
        ser = sorted([x for x in rows if x['rdist'] == r], key=lambda x: x['n'])
        if not ser:
            continue
        ns = [x['n'] for x in ser]
        print(f"        {r}: dict(n={ns},")
        print(f"                 basis={[x['basis'] for x in ser]},")
        costs = [float('%.4e' % x['cost']) for x in ser]
        print(f"                 cost={costs}),")
    print('    }')
