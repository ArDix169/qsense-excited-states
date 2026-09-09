"""Collect the VO measurement/circuit benchmark across the three geometries.

Reads the JSONs written by run_measurement_benchmark_3geom.sh and reports, per
(geometry, no_states): the sampling cost (eps^2 M), the CNOT statistics, the
depth statistics, and the subspace dimension.

The parameter sets differ by geometry -- 1.5 and 1.875 A use ratio 1.0 /
csf 1e-4 / Ethrsh_ia 1e-6, while 3.0 A uses ratio 0.1 / csf 1e-6 /
Ethrsh_ia 1e-7 -- because at dissociation the three lowest A singlets are
different orthogonal combinations of one shared configuration set and need
basis extensions rather than generators to separate them.  Each is the largest
threshold meeting the 1.6 mHa accuracy target, so the subspace sizes remain
comparable as "smallest subspace reaching chemical accuracy".

Usage (from the repository root):
    python3 analysis/collect_measurement_benchmark.py
    python3 analysis/collect_measurement_benchmark.py /path/to/dump/dir
"""
import json
import os
import sys

DUMPDIR = sys.argv[1] if len(sys.argv) > 1 else 'QSENSE_ES_dump'
GEOMS = ['1.5', '1.875', '3.0']
NSTATES = [1, 2, 3, 4, 5]

# A-sector singlet dimension of the space Q-SENSE can REACH.
#
# EXACT, not an estimate.  Q-SENSE holds the lowest two orbitals doubly occupied
# and allows no excitations out of them, so its subspace lives in 14e/10o, not
# the full 18e/12o.  With 5 A and 5 B active orbitals that is 4,950 singlet CSFs
# (Weyl-Paldus), splitting 2,500 A / 2,450 B by the determinant count
# N_CSF(singlet,g) = N_det(Ms=0,g) - N_det(Ms=1,g).
#
# DO NOT use the full-space figure here.  18e/12o gives 15,730 singlets, 7,910
# in A, and quoting against that would flatter the method with a denominator its
# ansatz cannot access -- 26% instead of 83% at n=5.  The FCI REFERENCE is
# full-space; the frozen-occupancy restriction is part of the approximation being
# validated by the energy errors, not part of this denominator.
CAS_DIM = 2500

hdr = (f"{'r':>7} {'n':>2} {'basis':>6} {'%CAS':>6} {'eps^2 M':>13} "
       f"{'CNOT avg':>10} {'CNOT max':>9} {'depth avg':>10} {'depth max':>10} "
       f"{'gen avg':>8}")
print(hdr)
print('-' * len(hdr))

missing = []
for r in GEOMS:
    for n in NSTATES:
        p = os.path.join(DUMPDIR, f'h2o2_VO_benchmark_n{n}_{r}.json')
        if not os.path.exists(p):
            missing.append((r, n))
            print(f'{r:>7} {n:>2}   (pending)')
            continue
        d = json.load(open(p))
        N = d['basis_states']
        print(f"{r:>7} {n:>2} {N:>6} {100*N/CAS_DIM:>5.1f}% "
              f"{d['sampling_cost']:>13.4e} "
              f"{d['cx_counts']['mean']:>10.1f} {d['cx_counts']['max']:>9d} "
              f"{d['depth']['mean']:>10.1f} {d['depth']['max']:>10d} "
              f"{d['generators']['avg']:>8.1f}")
    print()

if missing:
    print(f'MISSING: {missing}')

print("""
  eps^2 M is the weighted sampling cost: the shot budget times the square of the
  target standard deviation, so it is independent of the accuracy demanded and
  can be compared directly across subspace sizes and against the other methods.
  CNOT/depth are per matrix element -- N diagonal plus N(N-1)/2 off-diagonal --
  so 'max' is the deepest single circuit that must run and 'avg' is the mean
  over all measured elements.""")
