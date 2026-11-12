"""Measure the Trotter-seam overestimate in the reported baseline circuit counts.

The benchmark transpiles ONE Trotter step of the sa-UCCGSD ansatz and multiplies
by the Trotter depth, so cancellation across the single seam between steps is not
exploited and the reported count is a slight overestimate.  This script builds the
same generator pool and the same circuits, transpiles one step and all steps
jointly, and prints the difference.

It reproduces the archived cnot_joint as a validation before reporting the gap.

  python baselines/trotter_seam.py --molecule h2o  --geom 1.0
  python baselines/trotter_seam.py --molecule h2o2 --geom 1.5
"""
import argparse, functools, os, pickle, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASEDIR = os.path.join(REPO, 'data', 'QSENSE_paper_release', 'Baselines')

ap = argparse.ArgumentParser()
ap.add_argument('--molecule', default='h2o', choices=['h2o', 'h2o2'])
ap.add_argument('--geom', default='1.0')
ap.add_argument('--trotter', type=int, default=2)
ap.add_argument('--opt-level', type=int, default=3)
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

# Everything comes from the archived baseline pickle: it stores the exact
# orbsym_id array the benchmark used, so no symmetry recomputation is needed
# and the h2o/h2o2 Hamiltonians need not carry matching metadata.
ref = os.path.join(BASEDIR, f"{a.molecule}_{a.geom}_gen_T{a.trotter}.pkl")
if not os.path.exists(ref):
    raise SystemExit(f"no archived baseline pickle at {ref}")
arch = pickle.load(open(ref, 'rb'))
cfg, res = arch['config'], arch.get('resources', {})
NCAS, EXC_FREEZE = cfg['ncas'], cfg['exc_freeze']
sym, NSO = cfg['orbsym_id'], 2 * cfg['ncas']
tot = lambda idx: functools.reduce(lambda x, y: x ^ y, [sym[o] for o in idx]) == 0

pool = list(range(EXC_FREEZE, NCAS))
pairs = [(p, q) for ip, p in enumerate(pool) for q in pool[:ip]]
specs = [('S', p, q) for (p, q) in pairs if tot([p, q])]
specs += [('D', p, q, r, s) for X, (p, q) in enumerate(pairs)
          for (r, s) in pairs[X:] if tot([p, q, r, s])]

print(f"{a.molecule} {a.geom} A   NCAS {NCAS}  NSO {NSO}  "
      f"point group {cfg['point_group']}  exc_freeze {EXC_FREEZE}")
print(f"generators: {sum(1 for s in specs if s[0]=='S')} singles + "
      f"{sum(1 for s in specs if s[0]=='D')} doubles = {len(specs)}")
archived = (res.get('n_generators'), res.get('cnot_joint'), res.get('depth_joint'))
print(f"archived:   n_generators {archived[0]}  cnot_joint {archived[1]}  "
      f"depth_joint {archived[2]}")
assert archived[0] == len(specs), (
    f"generator pool mismatch: built {len(specs)}, archive says {archived[0]}")


def Eop(p, q):
    return FermionicOp({f"+_{p} -_{q}": 1.0,
                        f"+_{p+NCAS} -_{q+NCAS}": 1.0},
                       num_spin_orbitals=NSO)


def build(nrep):
    qc = QuantumCircuit(NSO)
    for rep in range(nrep):
        for i, s in enumerate(specs):
            traw = (Eop(s[1], s[2]) if s[0] == 'S'
                    else (Eop(s[1], s[2]) @ Eop(s[3], s[4])).normal_order())
            P = JordanWignerMapper().map(1j * (traw - traw.adjoint()))
            qc.append(PauliEvolutionGate(P, time=0.1 + 1e-3 * i + 0.01 * rep),
                      range(NSO))
    return qc


out = {}
for nrep, tag in ((1, 'one step'), (a.trotter, f'{a.trotter} steps joint')):
    t0 = time.time()
    tq = transpile(build(nrep).decompose(reps=3), basis_gates=['u3', 'cx'],
                   optimization_level=a.opt_level, seed_transpiler=a.seed)
    out[nrep] = (tq.num_nonlocal_gates(), tq.depth())
    print(f"  {tag:<16} CNOT {out[nrep][0]:>8}  depth {out[nrep][1]:>8}"
          f"   [{time.time()-t0:.0f}s]", flush=True)

cx1, dp1 = out[1]
cxN, dpN = out[a.trotter]
conv_cx, conv_dp = a.trotter * cx1, a.trotter * dp1
print(f"\n  reported convention (one step x {a.trotter}):  "
      f"CNOT {conv_cx}  depth {conv_dp}")
print(f"  joint over all {a.trotter} steps:            "
      f"CNOT {cxN}  depth {dpN}")
print(f"  SEAM OVERESTIMATE:  CNOT {conv_cx-cxN} "
      f"({100*(conv_cx-cxN)/max(conv_cx,1):.3f}%)   "
      f"depth {conv_dp-dpN} ({100*(conv_dp-dpN)/max(conv_dp,1):.3f}%)")
if archived and archived[1]:
    agree = "MATCHES archive" if conv_cx == archived[1] else \
            f"differs from archive by {100*abs(conv_cx-archived[1])/archived[1]:.1f}%"
    print(f"  validation: one step x {a.trotter} = {conv_cx} vs "
          f"archived cnot_joint {archived[1]}  -> {agree}")
