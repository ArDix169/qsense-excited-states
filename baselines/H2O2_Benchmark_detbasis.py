#!/usr/bin/env python3
"""
================================================================================
 H2O2 (and H2O) CAS/STO-3G -- VQE / q-sc-EOM / SS-SSSA-VQE benchmark
 DETERMINANT-BASIS PORT
================================================================================

 WHAT CHANGED vs the H2O qubit-space pipeline -- and what did NOT
 ----------------------------------------------------------------
 The method is unchanged: same CAS, same full-valence excitation space, same
 generators, same ansatz ordering, same optimizers.  Only the REPRESENTATION
 changed, because H2O2/STO-3G is CAS(18e,12o) = 24 qubits and the full Fock
 space (2^24 = 16.8 M, complex128) does not fit.

   (1) States live in the physical (N, Sz) determinant sector, not the full
       Fock space.  Every operator in the pipeline conserves N and Sz, so this
       is a change of basis, NOT an approximation:
           H2O2  CAS(18e,12o):  2^24 = 16,777,216  ->  C(12,9)^2 =  48,400
           H2O   CAS(10e,7o) :  2^14 =     16,384  ->  C(7,5)^2  =     441
   (2) float64 throughout.  In the determinant basis H is real symmetric and
       every generator G = tau^dag - tau is real ANTISYMMETRIC, so the state
       never leaves R.  (Verified: max|G + G^T| = 0 exactly.)
   (3) H is applied via PySCF's FCI contraction -- never formed as a matrix.
   (4) exp(theta G) uses the ACTUAL eigenvalue set, not integers.  G
       antisymmetric => eigenvalues {0, +-i mu_k}; with K distinct mu the
       exponential is exactly a degree-2K polynomial in G.  The original code
       assumed mu_k = k, which is false for generalized doubles (they have
       mu = sqrt2, sqrt3, 2 sqrt2) and silently fell back to Krylov for 25 of
       44 H2O generators.  Detecting mu by Lanczos on -G^2 makes every
       generator closed-form: 10x faster, and exact.

 Verified against the qubit-space implementation on H2O CAS(10e,7o):
   HF energy           agree to 1.3e-13
   FCI ground state    agree to 1.6e-13
   sa-UCCSD E(theta)   agree to 1.6e-13 for random theta, 2 Trotter steps

 NOT PORTED: Stage M (sorted insertion + KKT shot allocation).  It builds every
 fragment operator as a matrix in the full 2^(2M) space, which is exactly the
 thing that does not fit at 24 qubits.  It needs the same projection treatment;
 until then this script produces the ACCURACY tables, not the sampling ones.
================================================================================
"""
import os, sys, time, pickle
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from math import sin, cos, radians
from scipy.optimize import minimize

from pyscf import gto, scf, mcscf, fci, ao2mo, symm
from pyscf.fci import cistring, direct_spin1, spin_op

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False

# ============================== configuration ================================
MOLECULE = os.environ.get('MOLECULE', 'h2o2')       # 'h2o2' or 'h2o'
BASIS    = 'sto3g'

if MOLECULE == 'h2o2':
    RSCAN      = float(os.environ.get('ROO', 1.5))   # scanned O-O distance
    ROH_FIXED  = 0.9697
    ANG_HOO    = 101.9
    DIH_HOOH   = 111.5
    NCAS, NELECAS = 12, 18       # full STO-3G space -> 24 qubits
    EXC_FREEZE = 2               # both O 1s doubly occupied, never excited
    RLABEL     = 'rOO'
else:
    RSCAN      = float(os.environ.get('ROH', 1.0))   # scanned O-H distance
    ANG_HOH    = 104.5
    NCAS, NELECAS = 7, 10        # full STO-3G space -> 14 qubits
    EXC_FREEZE = 1               # O 1s doubly occupied, never excited
    RLABEL     = 'rOH'

NSTATE              = 5
SAUCCSD_TROTTER     = 2
SSVQE_TROTTER       = 2
SAUCCSD_SPATIAL_SYM = True       # keep only totally-symmetric excitations
GENERALIZED_POOL    = True       # all p>q over the excitable orbitals
SAUCCSD_N_STARTS    = 2
RUN_SSVQE           = True
SSVQE_N_STARTS      = 3
# Match the qubit-space notebook exactly.  These were maxiter=2000 / gtol=1e-5,
# i.e. a third fewer iterations and a gradient tolerance an order of magnitude
# looser than the reference run -- which showed up as SS-SSSA-VQE excited-state
# energies worse than the notebook's even though this script's ground-state
# feed is converged HARDER than the notebook's (ftol 1e-10/gtol 1e-6 here vs
# 1e-8/1e-5 there).  The sector ansatz is deeper and state-averaged, so it
# needs the larger budget, not the smaller one.
SSVQE_MAXITER       = 3000
SSVQE_FTOL          = 1e-6
SSVQE_GTOL          = 1e-6
SSVQE_PRUNE_ON_REFS = False      # UNSOUND for a product-form ansatz: generator
                                 # k acts on the partially evolved state, not on
                                 # the bare reference.  Keep False.
RUN_RESOURCES       = os.environ.get('RUN_RESOURCES', '1') == '1'
OUTDIR              = os.environ.get('OUTDIR', 'ham')
os.makedirs(OUTDIR, exist_ok=True)

NSO = 2 * NCAS
NP_ = (NELECAS // 2, NELECAS // 2)


def get_n_jobs():
    if not _HAVE_JOBLIB:
        return 1
    for v in ('BENCH_N_JOBS', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        if os.environ.get(v):
            try:
                return max(1, int(os.environ[v]))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


N_JOBS = get_n_jobs()


def pmap(func, items, n_jobs=N_JOBS, desc=None):
    items = list(items)
    if not items:
        return []
    n = max(1, min(n_jobs, len(items)))
    if desc:
        print(f"    [parallel] {desc}: {len(items)} task(s), {n} worker(s)",
              flush=True)
    if n == 1 or not _HAVE_JOBLIB or len(items) == 1:
        return [func(x) for x in items]
    return Parallel(n_jobs=n, backend='loky')(delayed(func)(x) for x in items)


def banner(s):
    print(f"\n{'='*76}\n{s}\n{'='*76}", flush=True)


# ====================== determinant-basis operator layer =====================
class DetBasis:
    """The (N, Sz) determinant sector.  CI vectors are the PySCF (na, nb) array
    flattened C-order, so idx = ia*nb + ib and a spin-summed one-body operator
    is  E_pq = kron(A_pq, I_nb) + kron(I_na, A_pq)."""

    def __init__(self, norb, nelec):
        self.norb, self.nelec = norb, tuple(nelec)
        na_e, nb_e = self.nelec
        self.strsa = cistring.make_strings(range(norb), na_e)
        self.strsb = cistring.make_strings(range(norb), nb_e)
        self.na, self.nb = len(self.strsa), len(self.strsb)
        self.dim = self.na * self.nb
        self.link_a = cistring.gen_linkstr_index(range(norb), na_e)
        self.link_b = cistring.gen_linkstr_index(range(norb), nb_e)
        self._epq = {}

    def _one_body_spin(self, p, q, alpha):
        link = self.link_a if alpha else self.link_b
        n = self.na if alpha else self.nb
        rows, cols, vals = [], [], []
        for str0 in range(n):
            for a, i, str1, sign in link[str0]:
                if a == p and i == q:
                    rows.append(str1); cols.append(str0); vals.append(float(sign))
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def Epq(self, p, q):
        """Spin-summed E_pq, cached (the doubles are products of these)."""
        if (p, q) not in self._epq:
            Aa = self._one_body_spin(p, q, True)
            Ab = self._one_body_spin(p, q, False)
            self._epq[(p, q)] = (
                sp.kron(Aa, sp.identity(self.nb, format='csr'), format='csr')
                + sp.kron(sp.identity(self.na, format='csr'), Ab, format='csr')
            ).tocsr()
        return self._epq[(p, q)]

    def hf_vector(self):
        na_e, nb_e = self.nelec
        ia = int(np.where(self.strsa == int('0b' + '1' * na_e, 2))[0][0])
        ib = int(np.where(self.strsb == int('0b' + '1' * nb_e, 2))[0][0])
        v = np.zeros(self.dim)
        v[ia * self.nb + ib] = 1.0
        return v

    def det_irrep(self, ia, ib, orbsym):
        """XOR of the irreps of the singly occupied orbitals."""
        sa, sb = int(self.strsa[ia]), int(self.strsb[ib])
        irr = 0
        for o in range(self.norb):
            if ((sa >> o) & 1) ^ ((sb >> o) & 1):
                irr ^= int(orbsym[o])
        return irr

    def vec_irrep(self, v, orbsym):
        ia, ib = np.unravel_index(int(np.argmax(np.abs(v))), (self.na, self.nb))
        return self.det_irrep(ia, ib, orbsym)

    def s_squared(self, v):
        c = v.reshape(self.na, self.nb)
        return float(spin_op.spin_square0(c, self.norb, self.nelec)[0])


def make_generator(basis, spec):
    """G = tau^dag - tau (real antisymmetric), or None if it vanishes.

    Matches the qubit pipeline's convention exactly:
        op = 1j (tau - tau^dag);  P = map(op);  G = 1j P  =>  G = tau^dag - tau
    """
    if spec[0] == 'S':
        tau = basis.Epq(spec[1], spec[2])
    else:
        tau = basis.Epq(spec[1], spec[2]) @ basis.Epq(spec[3], spec[4])
    G = (tau.T - tau).tocsr()
    G.eliminate_zeros()
    return None if G.nnz == 0 else G


def detect_mu(G, Kmax=10, tol=1e-8, nprobe=2, seed=0):
    """Distinct nonzero |mu| of antisymmetric G, by Lanczos on -G^2.

    -G^2 is symmetric PSD with eigenvalues {0, mu_k^2}.  If only K distinct
    nonzero values exist, Lanczos breaks down after K steps.  Returns None only
    if the Krylov space has not closed by Kmax (then we fall back to Taylor)."""
    n = G.shape[0]
    rng = np.random.default_rng(seed)
    best, closed = None, False
    for probe in range(nprobe):
        v = rng.normal(size=n); v /= np.linalg.norm(v)
        Q, alpha, beta, brk = [v], [], [], False
        for k in range(Kmax):
            w = -(G @ (G @ Q[-1]))
            a = float(Q[-1] @ w); alpha.append(a)
            w = w - a * Q[-1] - (beta[-1] * Q[-2] if k else 0.0)
            for q in Q:
                w = w - (q @ w) * q
            b = float(np.linalg.norm(w))
            if b < tol:
                brk = True
                break
            beta.append(b); Q.append(w / b)
        ev = np.linalg.eigvalsh(np.diag(alpha) + np.diag(beta, 1)
                                + np.diag(beta, -1))
        mu = np.sqrt(np.maximum(ev, 0.0))
        mu = mu[mu > 1e-7]
        if best is None or len(mu) > len(best):
            best, closed = mu, brk
    if best is None or len(best) == 0 or not closed:
        return None
    return np.unique(np.round(best, 9))


def op_norm(G):
    """Spectral radius of antisymmetric G (for the Taylor fallback)."""
    try:
        return float(sla.eigsh(-(G @ G), k=1, which='LA',
                               return_eigenvectors=False, maxiter=5000)[0] ** 0.5)
    except Exception:
        return float(abs(G).sum(axis=1).max())


def exp_apply(info, G, t, v):
    """exp(tG) v.  info = (mu array | None, spectral norm).

    With K distinct |mu|, exp(tG) is EXACTLY a degree-2K polynomial in G, whose
    coefficients follow from matching e^{i mu t} at every eigenvalue; that
    splits into two KxK real systems (cosines -> even, sines -> odd).  Costs 2K
    matvecs.  If the Krylov space did not close, fall back to scaling-and-
    squaring Taylor with the norm computed once (never re-estimated)."""
    if abs(t) <= 1e-15:
        return v
    mus, nrm = info
    if mus is None:
        s = max(1, int(np.ceil(abs(t) * nrm / 0.5)))
        ts = t / s
        for _ in range(s):
            term = out = v
            for k in range(1, 30):
                term = (ts / k) * (G @ term)
                out = out + term
                if np.linalg.norm(term) < 1e-14 * np.linalg.norm(out):
                    break
            v = out
        return v
    K = len(mus)
    Ae = np.array([[(-1.0) ** k * m ** (2 * k) for k in range(1, K + 1)]
                   for m in mus])
    ae = np.linalg.solve(Ae, np.cos(mus * t) - 1.0)
    Ao = np.array([[(-1.0) ** k * m ** (2 * k + 1) for k in range(K)]
                   for m in mus])
    ao = np.linalg.solve(Ao, np.sin(mus * t))
    a = np.zeros(2 * K + 1); a[0] = 1.0; a[2::2] = ae; a[1::2] = ao
    out = a[0] * v; w = v
    for j in range(1, 2 * K + 1):
        w = G @ w
        out = out + a[j] * w
    return out


class DetHamiltonian:
    """H|v> via PySCF's FCI contraction -- H is never formed."""

    def __init__(self, basis, h1e, eri, ecore=0.0):
        self.b, self.ecore = basis, ecore
        self.h2e = direct_spin1.absorb_h1e(h1e, eri, basis.norb, basis.nelec, .5)

    def matvec(self, v):
        c = v.reshape(self.b.na, self.b.nb)
        return direct_spin1.contract_2e(self.h2e, c, self.b.norb,
                                        self.b.nelec).ravel()

    def apply_cols(self, X):
        return np.column_stack([self.matvec(X[:, j]) for j in range(X.shape[1])])

    def expect(self, v):
        return float(v @ self.matvec(v)) + self.ecore


# ========================= STAGE 1: Hamiltonian + FCI ========================
banner(f"STAGE 1: CASSCF Hamiltonian + FCI singlet references  "
       f"[{MOLECULE}, {RLABEL} = {RSCAN}]")

if MOLECULE == 'h2o2':
    _th, _ta = radians(ANG_HOO), radians(DIH_HOOH)
    _half = (np.pi - _ta) / 2.0
    geo = [['O', [-RSCAN / 2.0, 0.0, 0.0]],
           ['O', [RSCAN / 2.0, 0.0, 0.0]],
           ['H', [-RSCAN / 2.0 + ROH_FIXED * cos(_th),
                  ROH_FIXED * sin(_th) * cos(_half),
                  -ROH_FIXED * sin(_th) * sin(_half)]],
           ['H', [RSCAN / 2.0 - ROH_FIXED * cos(_th),
                  -ROH_FIXED * sin(_th) * cos(_half),
                  -ROH_FIXED * sin(_th) * sin(_half)]]]
else:
    _h = radians(ANG_HOH / 2)
    geo = [['O', [0, 0, 0]],
           ['H', [-sin(_h) * RSCAN, 0, cos(_h) * RSCAN]],
           ['H', [sin(_h) * RSCAN, 0, cos(_h) * RSCAN]]]

mol = gto.Mole(atom=geo, basis=BASIS, symmetry=True, verbose=0); mol.build()
mf = scf.RHF(mol); mf.kernel()
print(f"{MOLECULE} {RLABEL}={RSCAN} A  {BASIS}   RHF E = {mf.e_tot:.10f}")

# Orbitals EXACTLY as the Ham_gen driver produces them: ONE CASSCF over the
# full space, then sort the MOs by orbital energy.  (No second kernel: for
# ncas == nao the CASSCF energy is FCI and orbital rotation is redundant, but
# the ORBITALS decide which two are excluded by EXC_FREEZE and what the ansatz
# sees, so this must match the Hamiltonian generator.)
mc2 = mcscf.CASSCF(mf, NCAS, NELECAS); mc2.fix_spin_(ss=0); mc2.kernel()
print(f"CASSCF E = {mc2.e_tot:.10f}  converged = {mc2.converged}")
if not mc2.converged:
    print("  *** WARNING: CASSCF did NOT converge -- everything below is suspect")

_idx = mc2.mo_energy.argsort()
mo_sorted = mc2.mo_coeff[:, _idx]
h1e_cas = mo_sorted.T @ (mol.intor('int1e_kin') + mol.intor('int1e_nuc')) @ mo_sorted
g_cas = ao2mo.restore(1, ao2mo.kernel(mol, mo_sorted), NCAS)
ecore = mf.energy_nuc()

active_mo = mo_sorted
orbsym_name = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, active_mo))
orbsym_id = [int(i) for i in
             symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, active_mo)]
id2name = dict(zip([int(i) for i in mol.irrep_id], list(mol.irrep_name)))


def irrep_name(iid):
    if iid in id2name:
        return id2name[iid]
    try:
        return symm.irrep_id2name(mol.groupname, iid)
    except Exception:
        return str(iid)


B = DetBasis(NCAS, NP_)
H = DetHamiltonian(B, h1e_cas, g_cas, ecore)
hf_vec = B.hf_vector()
print(f"point group: {mol.groupname}   active irreps: "
      f"{[f'{n}({i})' for n, i in zip(orbsym_name, orbsym_id)]}")
print(f"active space: {NCAS} spatial -> {NSO} qubits;  determinant sector "
      f"{B.na} x {B.nb} = {B.dim}  (Fock space would be {2**NSO})")
print(f"HF energy = {H.expect(hf_vec):.10f}")

# --- FCI singlets ------------------------------------------------------------
solver = fci.direct_spin0.FCI(); solver.nroots = min(16, B.dim)
e_all, civecs = solver.kernel(h1e_cas, g_cas, NCAS, NELECAS)
e_all = np.atleast_1d(e_all)
if not isinstance(civecs, list):
    civecs = [civecs]

singlets = []
print(f"\n{'root':>4} {'E':>16} {'<S^2>':>8} {'w_HF':>8}  keep  irrep")
print("-" * 58)
hf_ia = int(np.where(B.strsa == int('0b' + '1' * NP_[0], 2))[0][0])
hf_ib = int(np.where(B.strsb == int('0b' + '1' * NP_[1], 2))[0][0])
for i, (e, ci) in enumerate(zip(e_all, civecs)):
    ss = solver.spin_square(ci, NCAS, NELECAS)[0]
    keep = abs(ss) < 1e-6
    v = np.asarray(ci).ravel()
    iid = B.vec_irrep(v, orbsym_id)
    whf = float(np.asarray(ci)[hf_ia, hf_ib] ** 2)
    print(f"{i:>4} {e+ecore:16.8f} {ss:8.4f} {whf:8.4f}  "
          f"{'Y' if keep else 'n':>4}  {irrep_name(iid):>5}")
    if keep:
        singlets.append({'E': e + ecore, 'v': v / np.linalg.norm(v),
                         'irrep': iid, 'w_hf': whf})
    if len(singlets) >= NSTATE:
        break

if len(singlets) < NSTATE:
    raise RuntimeError(f"only {len(singlets)} singlets found, need {NSTATE}; "
                       f"raise solver.nroots")
singlets = singlets[:NSTATE]
e_ref = np.array([s['E'] for s in singlets])
fci_vecs = [s['v'] for s in singlets]
E0 = e_ref[0]
print(f"\nVERIFY S0 vs CASSCF: {abs(E0 - mc2.e_tot):.2e}  "
      f"{'OK' if abs(E0-mc2.e_tot) < 1e-7 else 'FAIL'}")

# ====================== STAGE M PREP: JW operator + bitstrings ================
# Stage M (sorted insertion + KKT shot allocation) is the ONE stage that cannot
# live purely in the determinant sector: a commuting Pauli GROUP does not
# conserve (N, S_z), so O|psi> leaves the sector even though H and every
# generator preserve it.  stage_m_detbasis works on the reachable support of
# O|psi> (<= k * 48,400 entries) instead of the 2^24 dense space.
#
# Nothing else in this script needs the qubit Hamiltonian; it is built here
# solely to be grouped into commuting fragments.
#
# CONVENTION.  qiskit-nature's mappers use BLOCKED spin-orbital ordering
# (alpha -> p, beta -> norb + p), NOT OpenFermion's interleaved (2p, 2p+1).
# Getting this wrong does not raise -- it silently produces plausible, wrong
# sigmas -- so verify_bitstrings checks <v|H_qubit|v> against the FCI energies
# before anything downstream runs.  That check needs no dense operator and so
# works at 24 qubits exactly as it does at 14.
RUN_STAGE_M = os.environ.get('RUN_STAGE_M', '1') == '1'
qop = None
if RUN_STAGE_M:
    banner("STAGE M PREP: Jordan-Wigner operator + determinant bitstrings")
    from qiskit_nature.second_q.operators import ElectronicIntegrals
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from stage_m_detbasis import sector_bitstrings, verify_bitstrings

    _t0 = time.time()
    _ei = ElectronicIntegrals.from_raw_integrals(h1e_cas, g_cas)
    _fop = ElectronicEnergy(_ei).second_q_op()
    qop = JordanWignerMapper().map(_fop)
    # second_q_op() does NOT fold in ElectronicEnergy's `constants`, which is
    # why the qubit-space notebook adds ecore by hand to every eigenvalue.
    # Carry it as an identity term instead, so qop is the total-energy operator
    # and <v|qop|v> is directly comparable with the FCI energies.  Sorted
    # insertion splits the identity back out as a variance-free coefficient, so
    # this does not change any sigma.
    from qiskit.quantum_info import SparsePauliOp as _SPO
    qop = (qop + _SPO('I' * qop.num_qubits, [ecore])).simplify()
    print(f"  JW: {qop.num_qubits} qubits, {len(qop)} Pauli terms "
          f"(incl. ecore = {ecore:.8f} as identity)  [{time.time()-_t0:.1f}s]")

    det_bits = sector_bitstrings(B.strsa, B.strsb, NCAS,
                                 so_index=lambda p, s: p + NCAS * s)
    print(f"  determinant bitstrings: {len(det_bits)} "
          f"(sector is {100*len(det_bits)/2**NSO:.3f}% of Fock space)")

    _t0 = time.time()
    # atol 1e-6, not the default 1e-8.  This guard exists to catch a BITSTRING
    # CONVENTION mismatch (a wrong so_index), which puts <v|H|v> O(1) Ha away
    # from the FCI energy -- 1e-6 still catches that by five orders of
    # magnitude.  1e-8 additionally fails on loosely converged FCI EIGENVECTORS,
    # which is what happens at dissociation: at rOO = 3.0 A the four lowest
    # H2O2 singlets span 1.65 mHa, Davidson converges their energies but not
    # their vectors, and max |diff| came out 3.8e-8 -- with the deviation
    # tracking the degeneracy exactly (3.8e-8, 1.1e-10, 8.2e-9, 3.0e-8 on the
    # clustered roots against 3.6e-9 on the separated one).  That is numerical
    # noise 40,000x below chemical accuracy, not a convention error.
    _err = verify_bitstrings(qop, np.column_stack(fci_vecs), det_bits, e_ref,
                             atol=1e-6, label='JW <v|H|v>')
    print(f"  [{time.time()-_t0:.1f}s]")
    if _err > 1e-6:
        raise RuntimeError(
            "JW bitstring convention does not reproduce the FCI energies -- "
            "check so_index against the mapper's spin-orbital ordering. "
            "Stage M would produce silently wrong sigmas.")

# ============================ FCI CENSUS =====================================
banner("FCI CENSUS: targeting plan")
fci_targets = [(s['E'], irrep_name(s['irrep']), s['irrep']) for s in singlets]
targets_by_irrep = {}
for k, (_E, _nm, _id) in enumerate(fci_targets):
    targets_by_irrep.setdefault(_id, []).append(k)
print(f"  {'state':>6} {'E (Ha)':>16} {'dE (eV)':>10} {'irrep':>6} {'w_HF':>8}")
print("  " + "-" * 50)
for k, (E, nm, _) in enumerate(fci_targets):
    print(f"  S{k:<5} {E:16.8f} {(E-E0)*27.2114:10.4f} {nm:>6} "
          f"{singlets[k]['w_hf']:8.4f}")
print("\n  sectors (states per irrep):")
for iid, ks in targets_by_irrep.items():
    print(f"    {irrep_name(iid):>4}: {len(ks)} root(s) -> "
          f"{['S'+str(k) for k in ks]}")

# ============================ generator pool =================================
banner("GENERATOR POOL (shared by the q-sc-EOM feed and the SS-SSSA-VQE sectors)")

nocc = NELECAS // 2
occ_x = list(range(EXC_FREEZE, nocc))
vir_x = list(range(nocc, NCAS))


def a_total(idxs):
    x = 0
    for o in idxs:
        x ^= orbsym_id[o]
    return x == 0


if GENERALIZED_POOL:
    pool = list(range(EXC_FREEZE, NCAS))
    exc_pairs = [(p, q) for ip, p in enumerate(pool) for q in pool[:ip]]
    POOL_TAG = 'generalized sa-UCCGSD'
else:
    exc_pairs = [(a, i) for a in vir_x for i in occ_x]
    POOL_TAG = 'occ->virt sa-UCCSD'

specs = [('S', p, q) for (p, q) in exc_pairs
         if not (SAUCCSD_SPATIAL_SYM and not a_total([p, q]))]
specs += [('D', p, q, r, s)
          for X, (p, q) in enumerate(exc_pairs) for (r, s) in exc_pairs[X:]
          if not (SAUCCSD_SPATIAL_SYM and not a_total([p, q, r, s]))]
print(f"  pool: {POOL_TAG}, excitable orbitals {EXC_FREEZE}..{NCAS-1}")
print(f"  candidate specs: {sum(1 for s in specs if s[0]=='S')} singles + "
      f"{sum(1 for s in specs if s[0]=='D')} doubles = {len(specs)}")

t0 = time.time()
for (p, q) in exc_pairs:                      # warm the E_pq cache serially
    B.Epq(p, q); B.Epq(q, p)
gens_1, specs_1 = [], []
for spec in specs:
    g = make_generator(B, spec)
    if g is not None:
        gens_1.append(g); specs_1.append(spec)
print(f"  built {len(gens_1)} non-null generators  [{time.time()-t0:.1f}s]")

t0 = time.time()
info_1 = [(detect_mu(g), 0.0) for g in gens_1]
info_1 = [(mu, 0.0 if mu is not None else op_norm(g))
          for (mu, _), g in zip(info_1, gens_1)]
from collections import Counter
_kc = Counter(len(mu) if mu is not None else -1 for mu, _ in info_1)
print("  exponentials: " + ", ".join(
    (f"K={k} ({2*k} matvecs): {v}" if k > 0 else f"Taylor fallback: {v}")
    for k, v in sorted(_kc.items(), reverse=True)) +
    f"   [{time.time()-t0:.1f}s]")

GENS = gens_1 * SAUCCSD_TROTTER
INFO = info_1 * SAUCCSD_TROTTER
NPAR = len(GENS)
print(f"  ansatz: {len(gens_1)}/step x {SAUCCSD_TROTTER} step(s) = {NPAR} parameters")


# ------------------------------ ansatz engine --------------------------------
def forward(theta, ref, gens=None, info=None):
    gens = GENS if gens is None else gens
    info = INFO if info is None else info
    v = ref
    for k in range(len(gens)):
        if abs(theta[k]) > 1e-15:
            v = exp_apply(info[k], gens[k], theta[k], v)
    return v


def energy_grad(theta, refs, weights, gens=None, info=None):
    """Weighted energy and its analytic reverse-sweep gradient.

    The references are propagated TOGETHER as a (dim, n_ref) block, so each
    generator application is one sparse matrix-matrix product instead of n_ref
    separate matvecs.  That matters because the biggest sector here carries 4
    references and dominates the run.  refs/weights of length 1 recovers the
    plain ground-state case."""
    gens = GENS if gens is None else gens
    info = INFO if info is None else info
    n = len(gens)
    R = np.column_stack(refs)                       # (dim, n_ref)
    w = np.asarray(weights, dtype=float)
    psis = [R]
    V = R
    for k in range(n):
        if abs(theta[k]) > 1e-15:
            V = exp_apply(info[k], gens[k], theta[k], V)
        psis.append(V)
    Psi = psis[-1]
    HPsi = H.apply_cols(Psi)
    F = float(np.einsum('A,iA,iA->', w, Psi, HPsi))
    Lam = HPsi * w[None, :]
    grad = np.zeros(n)
    for k in range(n - 1, -1, -1):
        grad[k] = 2.0 * float(np.einsum('iA,iA->', Lam, gens[k] @ psis[k + 1]))
        if abs(theta[k]) > 1e-15:
            Lam = exp_apply(info[k], gens[k], -theta[k], Lam)
    return F + H.ecore * float(w.sum()), grad


# ===================== STAGE 3b: sa-UCCSD ground state =======================
# ------------------------- optimizer progress reporting ----------------------
# L-BFGS-B prints nothing until it returns, and at 1.7 s per evaluation a
# healthy run is silent for an hour -- indistinguishable from a hang.  This
# wraps the objective so the callback can report the CURRENT energy without
# re-evaluating it (the callback only receives xk, and recomputing f there
# would double the cost of the whole optimization).
#
# PROGRESS_EVERY = 0 disables.  Note that under pmap with n_jobs > 1 these
# prints come from loky workers and may be buffered or interleaved; run with
# BENCH_N_JOBS=1 if you need a clean single-stream trace.
PROGRESS_EVERY = int(os.environ.get('PROGRESS_EVERY', 25))


def _tracked(fun, label, every=None):
    """Return (wrapped_objective, callback) reporting progress every `every`
    iterations.  The wrapper caches the last f so the callback is free."""
    every = PROGRESS_EVERY if every is None else every
    st = {'f': None, 'nfev': 0, 'nit': 0, 't0': time.time()}

    def wrapped(t):
        F, G = fun(t)
        st['f'] = F
        st['nfev'] += 1
        return F, G

    def cb(xk, *_):
        st['nit'] += 1
        if every and (st['nit'] == 1 or st['nit'] % every == 0):
            print(f"      [{label}] iter {st['nit']:5d}  F = {st['f']:.10f}  "
                  f"nfev = {st['nfev']:5d}  t = {time.time() - st['t0']:7.1f}s",
                  flush=True)

    return wrapped, (cb if every else None)


# --------------------------- gradient cost probe -----------------------------
# One energy_grad call is a forward sweep of NPAR generator applications plus a
# reverse sweep of the same, so the optimizer's wall time is (this number) x
# (iterations) x (starts).  At 24 qubits the vectors are 48,400 long instead of
# H2O's 441, so a setting that is comfortable there can be a week here -- print
# the cost and the implied budget BEFORE committing to the optimization rather
# than discovering it from a job that never finishes.
#
# theta = 0 is not representative: `abs(theta[k]) > 1e-15` skips every
# exponential, so it measures only the H application.  The random-theta timing
# is the one that matters.  Set TIME_GRAD=0 to skip (costs 2 evaluations).
if os.environ.get('TIME_GRAD', '1') == '1':
    banner("GRADIENT COST PROBE")
    _tg = time.time()
    energy_grad(np.zeros(NPAR), [hf_vec], [1.0])
    _d0 = time.time() - _tg

    _xr = np.random.default_rng(0).normal(0.0, 0.05, NPAR)
    _tg = time.time()
    energy_grad(_xr, [hf_vec], [1.0])
    _d1 = time.time() - _tg

    print(f"  dim = {len(hf_vec)}   NPAR = {NPAR}   generators = {len(GENS)}")
    print(f"  energy_grad at theta=0      : {_d0:8.3f} s  (exponentials skipped)")
    print(f"  energy_grad at random theta : {_d1:8.3f} s  <-- representative")
    for _lab, _it, _ns in (('sa-UCCSD', 2000, SAUCCSD_N_STARTS),
                           ('SSVQE/sector', SSVQE_MAXITER, SSVQE_N_STARTS)):
        print(f"    {_lab:<14}: {_it} iters x {_ns} start(s) "
              f"-> {_d1 * _it * _ns / 3600:8.2f} h serial, "
              f"{_d1 * _it * _ns / 3600 / max(1, min(N_JOBS, _ns)):8.2f} h "
              f"with {min(N_JOBS, _ns)} worker(s)")
    print("  NOTE: L-BFGS rarely runs to maxiter, so these are upper bounds; "
          "scale by the\n        fraction of maxiter actually used "
          "(H2O typically converges well short of it).")
    sys.stdout.flush()

banner("STAGE 3b: sa-UCCSD ground state (q-sc-EOM feed)")
t0 = time.time()


def _sa_start(s):
    x0 = (np.zeros(NPAR) if s == 0
          else np.random.default_rng(200 + s).normal(0, 0.02 * s, NPAR))
    _f, _cb = _tracked(lambda t: energy_grad(t, [hf_vec], [1.0]),
                       f'sa-UCCSD s{s}')
    r = minimize(_f, x0, jac=True, method='L-BFGS-B', callback=_cb,
                 options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6})
    return s, r.x, float(r.fun), int(r.nit), int(r.status)


res = pmap(_sa_start, range(SAUCCSD_N_STARTS), desc='sa-UCCSD starts')
res.sort(key=lambda t: t[2])
for s, _x, f, nit, st in sorted(res, key=lambda t: t[0]):
    print(f"    start {s}: E = {f:.10f}  err = {(f-E0)*1000:+9.4f} mHa  "
          f"({nit} iters, status {st}){'  BEST' if f == res[0][2] else ''}")
theta_sa = res[0][1]
psi_sa = forward(theta_sa, hf_vec)
psi_sa /= np.linalg.norm(psi_sa)
e_feed = H.expect(psi_sa)
print(f"\n  E        = {e_feed:.10f}")
print(f"  FCI S0   = {E0:.10f}")
print(f"  error    = {(e_feed-E0)*1000:+.4f} mHa  "
      f"({'CHEM ACC' if abs(e_feed-E0) < 1.6e-3 else 'not chem acc'})")
print(f"  <S^2>    = {B.s_squared(psi_sa):.3e}   t = {time.time()-t0:.0f}s")


def U_apply(v):
    return forward(theta_sa, v)


# ==================== STAGE 4e: per-irrep q-sc-EOM ===========================
banner("STAGE 4e: symmetry-adapted per-irrep q-sc-EOM  (feed = sa-UCCSD)")

occ_e, vir_e = list(range(EXC_FREEZE, nocc)), list(range(nocc, NCAS))
man_specs, man_irr = [], []
for a in vir_e:
    for i in occ_e:
        man_specs.append(('S', a, i))
        man_irr.append(orbsym_id[a] ^ orbsym_id[i])
_ep = [(a, i) for a in vir_e for i in occ_e]
for X, (a, i) in enumerate(_ep):
    for (b, j) in _ep[X:]:
        man_specs.append(('D', a, i, b, j))
        man_irr.append(orbsym_id[a] ^ orbsym_id[i] ^ orbsym_id[b] ^ orbsym_id[j])


def _man_vec(spec):
    if spec[0] == 'S':
        return B.Epq(spec[1], spec[2]) @ hf_vec
    return B.Epq(spec[1], spec[2]) @ (B.Epq(spec[3], spec[4]) @ hf_vec)


man_vec = [_man_vec(s) for s in man_specs]
print(f"  manifold: {len(man_specs)} spin-adapted operators; per irrep "
      f"{ {irrep_name(g): sum(1 for x in man_irr if x == g) for g in sorted(set(man_irr))} }")

excited_by_irrep = {}
for k in range(1, len(fci_targets)):
    excited_by_irrep.setdefault(fci_targets[k][2], []).append(k)

qsceom_blocks, qsceom_by_irrep, qsceom_ranks = {}, {}, {}
for iid in sorted(set(man_irr)):
    cols = [w / np.linalg.norm(w) for m, w in enumerate(man_vec)
            if man_irr[m] == iid and np.linalg.norm(w) > 1e-10]
    if not cols:
        continue
    # SVD, not QR: unpivoted QR does not order |diag(R)| by magnitude, so
    # thresholding it mis-counts the rank and retains arbitrary basis-completing
    # columns that carry no irrep label.
    Usvd, sv, _ = np.linalg.svd(np.column_stack(cols), full_matrices=False)
    rank = int(np.sum(sv > max(1e-10, 1e-12 * sv[0])))
    Bg = Usvd[:, :rank]
    t0 = time.time()
    UB = np.column_stack(pmap(U_apply, [Bg[:, j].copy() for j in range(rank)],
                              desc=f'U_apply irrep {irrep_name(iid)} (rank {rank})'))
    M = UB.T @ H.apply_cols(UB) + (H.ecore - e_feed) * np.eye(rank)
    M = 0.5 * (M + M.T)
    ev, evc = np.linalg.eigh(M)
    kept = []
    for j in range(len(ev)):
        if ev[j] <= 1e-8:
            continue
        pk = UB @ evc[:, j]
        nk = np.linalg.norm(pk)
        if nk < 1e-10:
            continue
        if abs(B.s_squared(pk / nk)) < 1e-4:
            kept.append((float(ev[j]), evc[:, j].copy()))
    kept.sort(key=lambda t: t[0])
    qsceom_by_irrep[iid] = [e for e, _ in kept]
    qsceom_ranks[iid] = rank
    qsceom_blocks[iid] = {
        'UB': UB,
        'evecs': (np.column_stack([v for _, v in kept]) if kept
                  else np.zeros((rank, 0))),
        'evals': np.array([e for e, _ in kept])}
    print(f"    {irrep_name(iid):>4}: rank {rank}, {len(kept)} singlet root(s) "
          f"[{time.time()-t0:.0f}s]")

# --- subspace representability: the CEILING on q-sc-EOM accuracy -------------
# The trial space is span{U tau_mu|HF>}; Rayleigh-Ritz gives the best energies
# IN that space, so once the matrix elements are exact the only error left is
# that the exact state is not in the space.  This separates "manifold too small"
# from feed error, mislabelling, or convergence.
print("\n  FCI weight inside the q-sc-EOM subspace:")
subspace_overlaps = {}
for k in range(1, NSTATE):
    iid = fci_targets[k][2]
    if iid not in qsceom_blocks:
        continue
    w = float(np.linalg.norm(qsceom_blocks[iid]['UB'].T @ fci_vecs[k]) ** 2)
    subspace_overlaps[k] = {'irrep': fci_targets[k][1], 'weight': w}
    print(f"    S{k} ({fci_targets[k][1]}): ||P_sub |E_k>||^2 = {w:.4f}")

# --- results, matched to FCI states BY OVERLAP, not by energy order ----------
# Energy-order matching is unreliable as soon as a block starts missing states.
print(f"\n  {'state':>6} {'irrep':>6} {'E_tot (Ha)':>16} {'FCI (Ha)':>16} "
      f"{'err (mHa)':>11} {'ovlp':>7}  flag")
print("  " + "-" * 78)
qsceom_rows = []
for k in range(1, NSTATE):
    E_fci, nm, iid = fci_targets[k]
    blk = qsceom_blocks.get(iid)
    best = None
    if blk is not None and blk['evecs'].shape[1]:
        for r in range(blk['evecs'].shape[1]):
            psi = blk['UB'] @ blk['evecs'][:, r]
            psi /= np.linalg.norm(psi)
            ov = float(np.dot(fci_vecs[k], psi) ** 2)
            if best is None or ov > best[0]:
                best = (ov, r)
    if best is None:
        print(f"  S{k:<5} {nm:>6} {'--':>16} {E_fci:16.8f} {'MISSING':>11} "
              f"{'--':>7}  (no root)")
        qsceom_rows.append({'state': k, 'irrep': nm, 'E': None, 'E_fci': E_fci,
                            'err_mHa': None, 'err_exc_mHa': None,
                            'match_overlap': None, 'root': None})
        continue
    ov, r = best
    Etot = e_feed + blk['evals'][r]
    err = (Etot - E_fci) * 1000
    err_x = (blk['evals'][r] - (E_fci - E0)) * 1000
    print(f"  S{k:<5} {nm:>6} {Etot:16.8f} {E_fci:16.8f} {err:11.4f} "
          f"{ov:7.3f}  {'CHEM ACC' if abs(err) < 1.6 else 'not chem acc'}")
    qsceom_rows.append({'state': k, 'irrep': nm, 'E': Etot, 'E_fci': E_fci,
                        'err_mHa': err, 'err_exc_mHa': err_x,
                        'match_overlap': ov, 'root': r})

# ==================== STAGE 6: symmetry-adapted SS-SSSA-VQE =====================
ssvqe_out, ssvqe_rows = None, []
if RUN_SSVQE:
    banner(f"STAGE 6: symmetry-adapted weighted SS-VQE ({POOL_TAG}, per sector)")
    orb_e = np.real(np.diag(h1e_cas))

    def norm_apply(vec):
        n = np.linalg.norm(vec)
        return (vec / n) if n > 1e-10 else None

    def sector_refs(iid, n_ref):
        """Orthonormal singlet references of spatial irrep iid.
        E_pq is spin-summed, so E_ai|HF> is ALREADY the singlet CG combination
        -- the spin adaptation comes free from the operator algebra."""
        cands = []
        if iid == 0:
            cands.append(('HF', hf_vec))
        singles = sorted(((a, i) for a in vir_e for i in occ_e
                          if (orbsym_id[a] ^ orbsym_id[i]) == iid),
                         key=lambda ai: orb_e[ai[0]] - orb_e[ai[1]])
        for a, i in singles:
            v = norm_apply(B.Epq(a, i) @ hf_vec)
            if v is not None:
                cands.append((f'S {i}->{a}', v))
        dbl = []
        for X, (a, i) in enumerate(_ep):
            for (b, j) in _ep[X:]:
                if (orbsym_id[a] ^ orbsym_id[i] ^ orbsym_id[b] ^ orbsym_id[j]) == iid:
                    dbl.append(((orb_e[a]-orb_e[i]) + (orb_e[b]-orb_e[j]),
                                a, i, b, j))
        for _, a, i, b, j in sorted(dbl, key=lambda t: t[0]):
            v = norm_apply(B.Epq(a, i) @ (B.Epq(b, j) @ hf_vec))
            if v is not None:
                cands.append((f'D {i}{j}->{a}{b}', v))
        refs, labels = [], []
        for lab, v in cands:
            w = v.copy()
            for u in refs:
                w -= (u @ w) * u
            n = np.linalg.norm(w)
            if n < 1e-8:
                continue
            refs.append(w / n); labels.append(lab)
            if len(refs) == n_ref:
                break
        return refs, labels

    gens_R = gens_1 * SSVQE_TROTTER
    info_R = info_1 * SSVQE_TROTTER
    NPAR_R = len(gens_R)
    assert NPAR_R == NPAR or SSVQE_TROTTER != SAUCCSD_TROTTER, \
        "feed and sector ansatz should match when the Trotter depths agree"
    print(f"  sector ansatz: {NPAR_R} parameters "
          f"({len(gens_1)}/step x {SSVQE_TROTTER})")

    ssvqe_sectors = []
    # Kept for STAGE M: the converged sector states are the basis over which the
    # SS-VQE shot allocation is solved.  (The qubit-space notebook adds the same
    # line via si_stage_edits.py.)
    ssvqe_state_vectors = {}
    t0 = time.time()
    for iid, ks in targets_by_irrep.items():
        n_ref = len(ks)
        nm = irrep_name(iid)
        refs, labels = sector_refs(iid, n_ref)
        if len(refs) < n_ref:
            print(f"\n    *** {nm}: only {len(refs)} orthogonal references "
                  f"reachable, need {n_ref} -- targeting the lowest {len(refs)}")
            if not refs:
                continue
            ks, n_ref = ks[:len(refs)], len(refs)
        weights = np.array([n_ref - A for A in range(n_ref)], dtype=float)
        weights /= weights.sum()
        print(f"\n    --- sector {nm} ({iid}): {n_ref} root(s) ---")
        print(f"        references: " +
              ",  ".join(f"{l} (w={w:.3f})" for l, w in zip(labels, weights)))
        # overlap of each reference with its FCI target: a reference with poor
        # overlap cannot be rescued by depth or optimizer budget
        for A, k in enumerate(ks):
            print(f"        <ref {A} | FCI S{k}> = "
                  f"{abs(float(refs[A] @ fci_vecs[k])):.3f}")

        def _start(s, refs=refs, weights=weights, iid=iid):
            x0 = (np.zeros(NPAR_R) if s == 0
                  else np.random.default_rng(900 + 13 * int(iid) + s)
                       .normal(0, 0.05, NPAR_R))
            _f, _cb = _tracked(
                lambda t: energy_grad(t, refs, weights, gens_R, info_R),
                f'SSVQE {nm} s{s}')
            r = minimize(_f, x0, jac=True, method='L-BFGS-B', callback=_cb,
                         options={'maxiter': SSVQE_MAXITER, 'ftol': SSVQE_FTOL,
                                  'gtol': SSVQE_GTOL})
            return s, r.x, float(r.fun), int(r.nit), int(r.status), \
                float(np.max(np.abs(r.jac)))

        out = pmap(_start, range(SSVQE_N_STARTS), desc=f'SS-VQE {nm} starts')
        out.sort(key=lambda t: t[2])
        for s, _x, f, nit, st, gj in sorted(out, key=lambda t: t[0]):
            print(f"        [start{s}] F = {f:.8f}  ({nit} iters, status {st}, "
                  f"max|grad| = {gj:.2e})")
        best_x = out[0][1]

        states = []
        for ref in refs:
            psi = forward(best_x, ref, gens_R, info_R)
            psi /= np.linalg.norm(psi)
            states.append((H.expect(psi), B.s_squared(psi), psi))
        states.sort(key=lambda t: t[0])
        ssvqe_state_vectors[int(iid)] = np.column_stack([s[2] for s in states])
        max_ov = max((abs(float(states[a][2] @ states[b][2]))
                      for a in range(len(states))
                      for b in range(a + 1, len(states))), default=0.0)
        print(f"        max |<chi_A|chi_B>| = {max_ov:.2e} (structural)")
        for A, (k, (eA, s2A, _)) in enumerate(zip(ks, states)):
            err = (eA - fci_targets[k][0]) * 1000
            print(f"        RESULT S{k} {nm:>3} (root {A}): E = {eA:.8f}  "
                  f"err = {err:+9.4f} mHa  <S^2> = {s2A:6.3f}  "
                  f"{'CHEM ACC' if abs(err) < 1.6 else 'not chem acc'}")
            ssvqe_rows.append({'state': k, 'irrep': nm, 'sector_root': A,
                               'E': eA, 'E_fci': fci_targets[k][0],
                               'err_mHa': err, 's2': s2A})
        ssvqe_sectors.append({'irrep': nm, 'irrep_id': int(iid),
                              'n_roots': n_ref, 'n_params': NPAR_R,
                              'refs': labels, 'weights': weights.tolist(),
                              'max_overlap': max_ov})
    print(f"\n  [stage 6 total {time.time()-t0:.0f}s]")
    ssvqe_out = {'n_qubits': NSO, 'ansatz': POOL_TAG, 'trotter': SSVQE_TROTTER,
                 'pruned': SSVQE_PRUNE_ON_REFS, 'full_pool': NPAR_R,
                 'sectors': ssvqe_sectors, 'states': ssvqe_rows}

# ============================ circuit resources ==============================
# The circuit is built ONLY for gate counting; it is never simulated.  It is
# geometry-independent given the generator LABELS, so it is cached on disk and
# reused across geometries (the transpile is the slow part at 24 qubits).
resources = None
if RUN_RESOURCES:
    banner("CIRCUIT RESOURCES (unbound product-form circuit)")
    import hashlib, json
    key = hashlib.md5(json.dumps(
        [list(map(int, s[1:])) + [s[0]] for s in specs_1] +
        [NSO, SAUCCSD_TROTTER]).encode()).hexdigest()[:16]
    cache = os.path.join(OUTDIR, f'circmetrics_{key}.pkl')
    if os.path.exists(cache):
        resources = pickle.load(open(cache, 'rb'))
        print(f"  loaded from cache {cache}")
    else:
        from qiskit import transpile, QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit_nature.second_q.operators import FermionicOp
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        mapper = JordanWignerMapper()

        def Eop(p, q):
            return FermionicOp({f"+_{p} -_{q}": 1.0,
                                f"+_{p+NCAS} -_{q+NCAS}": 1.0},
                               num_spin_orbitals=NSO)

        t0 = time.time()
        qc = QuantumCircuit(NSO)
        idx = 0
        for _ in range(SAUCCSD_TROTTER):
            for spec in specs_1:
                traw = (Eop(spec[1], spec[2]) if spec[0] == 'S'
                        else (Eop(spec[1], spec[2]) @ Eop(spec[3], spec[4])
                              ).normal_order())
                P = mapper.map(1j * (traw - traw.adjoint()))
                qc.append(PauliEvolutionGate(P, time=Parameter(f"p{idx}")),
                          range(NSO))
                idx += 1
        print(f"  circuit built ({idx} evolution gates) [{time.time()-t0:.0f}s]"
              f" -- transpiling, this is the slow step at {NSO} qubits...",
              flush=True)
        t0 = time.time()
        tq = transpile(qc.decompose(reps=3), basis_gates=['u3', 'cx'],
                       optimization_level=3, seed_transpiler=42)
        resources = {'cnot': tq.num_nonlocal_gates(), 'depth': tq.depth(),
                     'size': tq.size(), 'params': qc.num_parameters,
                     'n_qubits': NSO}
        # --- joint transpile of one Trotter step -------------------------
        # REPRODUCIBILITY: the paper quotes the joint figures, so the shipping
        # code has to produce them.  Stored ALONGSIDE the per-generator sum,
        # never replacing it -- the sum is cached and resumable (flushed every
        # CHUNK generators), the joint pass is all-or-nothing, and keeping both
        # lets the bound and the measurement be reported together.
        #
        # Measured on the real H2O2 pool (555 generators): the per-generator
        # sum overestimates CNOTs by 1.4% and depth by 13%.  CNOTs are nearly
        # additive; depth is not, because generators on disjoint qubits run
        # concurrently in a joint circuit but are serialised by the sum.
        # Parameter binding does not affect either count (checked explicitly),
        # so bound angles are used here for a single-pass transpile.
        if os.environ.get('RES_JOINT', '1') == '1':
            try:
                _t1 = time.time()
                _qc = QuantumCircuit(NSO)
                for _i, _s in enumerate(specs_1):
                    _traw = (Eop(_s[1], _s[2]) if _s[0] == 'S'
                             else (Eop(_s[1], _s[2]) @ Eop(_s[3], _s[4])
                                   ).normal_order())
                    _P = JordanWignerMapper().map(1j * (_traw - _traw.adjoint()))
                    _qc.append(PauliEvolutionGate(_P, time=0.1 + 1e-3 * _i),
                               range(NSO))
                _tq = transpile(_qc.decompose(reps=3), basis_gates=['u3', 'cx'],
                                optimization_level=RES_OPT_LEVEL,
                                seed_transpiler=42)
                resources['cnot_joint'] = SAUCCSD_TROTTER * int(_tq.num_nonlocal_gates())
                resources['depth_joint'] = SAUCCSD_TROTTER * int(_tq.depth())
                resources['joint_estimator'] = (
                    'ONE transpile of the full per-Trotter-step circuit at '
                    f'optimization_level={RES_OPT_LEVEL}, times the Trotter '
                    'depth.  Measured, not a bound, except that the seam '
                    'BETWEEN Trotter steps is still not exploited.')
                print(f"  joint transpile: CNOT {resources['cnot_joint']}  "
                      f"depth {resources['depth_joint']}  "
                      f"(bound/actual "
                      f"{resources['cnot']/max(resources['cnot_joint'],1):.3f}x cx, "
                      f"{resources['depth']/max(resources['depth_joint'],1):.3f}x depth)"
                      f"  [{time.time()-_t1:.0f}s]", flush=True)
            except Exception as _e:
                # never let this kill a run -- the per-generator numbers stand
                print(f"  joint transpile SKIPPED: {type(_e).__name__}: {_e}",
                      flush=True)

        pickle.dump(resources, open(cache, 'wb'))
        print(f"  transpiled [{time.time()-t0:.0f}s] -> cached {cache}")
    # BACKFILL the joint transpile onto a CACHED resources dict.
    #
    # The joint block lives inside the `else` above, so a run that loads
    # circmetrics_<key>.pkl from cache skips it and the pickle comes out with no
    # cnot_joint/depth_joint at all -- which is exactly what happened when H2O2
    # was rerun over its existing cache while H2O (new hash key, no cache)
    # picked the fields up.  This clause fills them in either way and rewrites
    # the cache, so no rerun is needed to obtain them a second time.
    if (resources is not None and 'cnot_joint' not in resources
            and os.environ.get('RES_JOINT', '1') == '1'):
        try:
            import time as _time
            from qiskit import transpile as _transpile, QuantumCircuit as _QC
            from qiskit.circuit.library import PauliEvolutionGate as _PEG
            from qiskit_nature.second_q.operators import FermionicOp as _FOp
            from qiskit_nature.second_q.mappers import JordanWignerMapper as _JW

            def _Eop(p_, q_):
                return _FOp({f"+_{p_} -_{q_}": 1.0,
                             f"+_{p_ + NCAS} -_{q_ + NCAS}": 1.0},
                            num_spin_orbitals=NSO)

            _t1 = _time.time()
            _qc = _QC(NSO)
            for _i, _sp in enumerate(specs_1):
                _traw = (_Eop(_sp[1], _sp[2]) if _sp[0] == 'S'
                         else (_Eop(_sp[1], _sp[2]) @ _Eop(_sp[3], _sp[4])
                               ).normal_order())
                _P = _JW().map(1j * (_traw - _traw.adjoint()))
                _qc.append(_PEG(_P, time=0.1 + 1e-3 * _i), range(NSO))
            _tq = _transpile(_qc.decompose(reps=3), basis_gates=['u3', 'cx'],
                             optimization_level=resources.get('opt_level', 3),
                             seed_transpiler=42)
            resources['cnot_joint'] = SAUCCSD_TROTTER * int(_tq.num_nonlocal_gates())
            resources['depth_joint'] = SAUCCSD_TROTTER * int(_tq.depth())
            resources['joint_estimator'] = (
                'ONE transpile of the full per-Trotter-step circuit, times the '
                'Trotter depth.  Measured, not a bound, except that the seam '
                'BETWEEN Trotter steps is not exploited.')
            pickle.dump(resources, open(cache, 'wb'))
            print(f"  joint transpile (backfilled): CNOT {resources['cnot_joint']}  "
                  f"depth {resources['depth_joint']}  "
                  f"[{_time.time()-_t1:.0f}s]", flush=True)
        except Exception as _e:
            print(f"  joint backfill SKIPPED: {type(_e).__name__}: {_e}", flush=True)

    print(f"  CNOT = {resources['cnot']}   depth = {resources['depth']}   "
          f"params = {resources['params']}")

# ============ STAGE M: sorted insertion + KKT shot allocation ================
# Determinant-basis counterpart of the qubit-space notebook's STAGE M.  The
# allocation code (sorted_insertion, kkt_shot_allocation, eigvec_weights, ...)
# is imported VERBATIM from stage_m_core so the two representations run
# identical logic; only the sigma builder differs, because a commuting Pauli
# group is the one operator here that does not preserve (N, S_z).
#
# ACCEPTANCE TEST: at MOLECULE=h2o this must reproduce the notebook's numbers
# (eps^2 M = 419 / 1211 / 266 for q-sc-EOM and 83.0 / 44.1 / 4.90 for SS-SSSA-VQE
# at rOH = 1.0 / 2.125 / 3.0).  Until it does, do NOT trust the H2O2 output --
# the whole point of lifting the core verbatim is that agreement there
# validates the port.
stage_m = None
if RUN_STAGE_M and qop is not None:
    banner("STAGE M: Sorted Insertion (FC) + KKT-optimal shot allocation")
    from stage_m_core import (sorted_insertion, eigvec_weights, pack_sigma,
                              kkt_shot_allocation, SI_EPS_HA)
    from stage_m_detbasis import sigma_matrix_detbasis

    _t0 = time.time()
    op_H, groups_H, cH, idc_H = sorted_insertion(qop)
    print(f"  sorted insertion: {len(groups_H)} fully-commuting group(s) from "
          f"{len(cH)} Pauli term(s); identity coeff = {idc_H:.6f} Ha "
          f"(variance-free)  [{time.time()-_t0:.1f}s]")
    _e2 = SI_EPS_HA ** 2
    stage_m = {'eps': SI_EPS_HA, 'n_groups': len(groups_H),
               'identity_coeff': float(idc_H)}

    # --- q-sc-EOM: coupled allocation per irrep block ------------------------
    # Only the TARGETED roots are held to eps.  Each block keeps every singlet
    # root it finds (often ~10), but the table reports NSTATE-1 excited states,
    # so constraining all of them would charge for energies nobody reports --
    # it inflated the budget 8x before this was fixed.  Within a block the
    # allocation is genuinely coupled: one matrix element feeds every root.
    print("  q-sc-EOM (targeted roots only, coupled within each block):")
    _q_tot = 0.0
    for _iid, _blk in sorted(qsceom_blocks.items()):
        _tgt = excited_by_irrep.get(_iid, [])
        if not _tgt or _blk['evecs'].shape[1] == 0:
            continue
        _ns = min(len(_tgt), _blk['evecs'].shape[1])
        _C, _UB = _blk['evecs'][:, :_ns], _blk['UB']
        _t = time.time()
        _S = sigma_matrix_detbasis(op_H, groups_H, cH, _UB, det_bits)
        _sig = pack_sigma(_S)
        _W, _, _ = eigvec_weights(_C)
        _lam, _M, _F, _act, _info = kkt_shot_allocation(_W, _sig, SI_EPS_HA)
        _m = float(_M.sum())
        _q_tot += _m
        print(f"    {irrep_name(_iid):>4}: rank {_UB.shape[1]:>3}, "
              f"{_ns} targeted root(s), {len(_sig)} element(s), "
              f"sigma_max = {_S.max():.4f}, shots = {_m:.3e}  "
              f"[{time.time()-_t:.0f}s]")
    stage_m['qsceom_shots'] = _q_tot
    stage_m['qsceom_eps2M'] = _e2 * _q_tot

    # --- SS-SSSA-VQE: separable per root, coupled only through the shared S0 ----
    # Each root is measured in its own state, so sigma is just S[A, A] and the
    # per-root problems are separable.  But the REPORTED quantities are
    # excitation energies dE_k = E_k - E_0, and every one of them contains S0 --
    # a star coupling, with closed form (sqrt(R0) + sqrt(sum_k Rk))^2 / eps^2.
    # Charging total energies instead misses that and understates the budget
    # (it was 1.75x low before this was fixed).
    _s_tot = None
    if RUN_SSVQE and ssvqe_out is not None and ssvqe_state_vectors:
        print("  SS-SSSA-VQE (separable per root; S0 shared by every difference):")
        _by_state = {}
        for _iid, _Vs in ssvqe_state_vectors.items():
            _S = sigma_matrix_detbasis(op_H, groups_H, cH, _Vs, det_bits)
            for _A in range(_Vs.shape[1]):
                _row = next((r for r in ssvqe_rows
                             if r['irrep'] == irrep_name(_iid)
                             and r['sector_root'] == _A), None)
                if _row is not None:
                    _by_state[_row['state']] = (float(_S[_A, _A]),
                                                f"{irrep_name(_iid)}[{_A}]")
        # order by GLOBAL state index so index 0 really is S0 -- the star
        # coupling below assumes it
        _ks = sorted(_by_state)
        _sig_root = np.array([_by_state[k][0] for k in _ks])
        for _k, _s in zip(_ks, _sig_root):
            print(f"    S{_k} ({_by_state[_k][1]:>7}): sigma = {_s:.4f}   "
                  f"shots @ eps = {_s**2/_e2:.3e}")
        _m_tot = float(np.sum(_sig_root ** 2) / _e2)
        _K = len(_sig_root) - 1
        if _K >= 1 and _ks[0] == 0:
            _Wx = np.zeros((_K, len(_sig_root)))
            for _k in range(_K):
                _Wx[_k, 0] = 1.0                    # shared ground state
                _Wx[_k, _k + 1] = 1.0
            _lx, _Mx, _Fx, _ax, _ix = kkt_shot_allocation(_Wx, _sig_root,
                                                          SI_EPS_HA)
            _s_tot = float(_Mx.sum())
            _cf = ((np.sqrt(_sig_root[0] ** 2)
                    + np.sqrt(np.sum(_sig_root[1:] ** 2))) ** 2 / _e2)
            print(f"    excitation-energy budget: KKT {_s_tot:.6e}  vs "
                  f"closed form {_cf:.6e}  "
                  f"({'PASS' if abs(_cf-_s_tot)/max(_cf,1e-300) < 1e-6 else '*** FAIL ***'})")
            print(f"    S0 takes {_Mx[0]/_s_tot*100:.1f}% of the budget; "
                  f"total-energy charge would be {_m_tot:.3e} "
                  f"({_s_tot/max(_m_tot,1e-300):.2f}x)")
        else:
            _s_tot = _m_tot
            print(f"    (no S0 in the SS-VQE set -- charging total energies)")
        stage_m['ssvqe_shots'] = _s_tot
        stage_m['ssvqe_eps2M'] = _e2 * _s_tot
        stage_m['ssvqe_shots_totals'] = _m_tot

    print(f"\n  {'method':<12} {'shots':>13} {'eps^2 M (Ha^2)':>16}")
    print("  " + "-" * 43)
    print(f"  {'q-sc-EOM':<12} {_q_tot:>13.4e} {_e2*_q_tot:>16.4e}")
    if _s_tot is not None:
        print(f"  {'SS-SSSA-VQE':<12} {_s_tot:>13.4e} {_e2*_s_tot:>16.4e}")
    print(f"\n  eps = {SI_EPS_HA:.1e} Ha on every reported energy; eps^2 M is "
          f"eps-independent\n  and so comparable across methods and systems.")

# =============================== COLLECTION ==================================
banner("COLLECTION: excited states per irrep sector")
_q = {r['state']: r for r in qsceom_rows}
_s = {r['state']: r for r in ssvqe_rows}
print(f"  {'state':>5} {'irrep':>5} {'FCI (Ha)':>16} {'q-scEOM err':>12} "
      f"{'SSVQE err':>11} {'sub. ovlp':>10}")
print("  " + "-" * 66)
collection = []
for k in range(NSTATE):
    E_fci, nm, iid = fci_targets[k]
    if k == 0:
        qerr = (e_feed - E_fci) * 1000       # feed, not an EOM root
        qov = None
    else:
        qerr = _q.get(k, {}).get('err_mHa')
        qov = subspace_overlaps.get(k, {}).get('weight')
    serr = _s.get(k, {}).get('err_mHa')
    print(f"  S{k:<4} {nm:>5} {E_fci:16.8f} "
          f"{('--' if qerr is None else f'{qerr:.3f}'):>12} "
          f"{('--' if serr is None else f'{serr:.3f}'):>11} "
          f"{('--' if qov is None else f'{qov:.4f}'):>10}")
    collection.append({'state': k, 'irrep': nm, 'E_fci': E_fci,
                       'qsceom_err_mHa': qerr, 'ssvqe_err_mHa': serr,
                       'subspace_overlap': qov})
print(f"\n  S0 for q-sc-EOM is the sa-UCCSD feed, not an EOM root; every "
      f"excited-state error carries it.")

# ================================= save ======================================
cfg = {'molecule': MOLECULE, RLABEL: RSCAN, 'basis': BASIS,
       'ncas': NCAS, 'nelecas': NELECAS, 'exc_freeze': EXC_FREEZE,
       'nstate': NSTATE, 'generalized_pool': GENERALIZED_POOL,
       'sauccsd_trotter': SAUCCSD_TROTTER, 'ssvqe_trotter': SSVQE_TROTTER,
       'ssvqe_pruned': SSVQE_PRUNE_ON_REFS, 'spatial_sym': SAUCCSD_SPATIAL_SYM,
       'point_group': mol.groupname, 'orbsym_id': orbsym_id,
       'casscf_converged': bool(mc2.converged), 'det_dim': B.dim,
       'n_params': NPAR}
out = {'config': cfg, 'e_ref': e_ref,
       'fci_targets': [(E, nm, int(i)) for E, nm, i in fci_targets],
       'targets_by_irrep': {int(k): v for k, v in targets_by_irrep.items()},
       'feed': {'E': e_feed, 'err_mHa': (e_feed - E0) * 1000,
                'theta': theta_sa, 'n_params': NPAR},
       'qsceom': qsceom_rows,
       'qsceom_ranks': {int(k): v for k, v in qsceom_ranks.items()},
       'qsceom_evals': {int(k): v['evals'] for k, v in qsceom_blocks.items()},
       'subspace_overlaps': subspace_overlaps,
       'ssvqe': ssvqe_out, 'collection': collection, 'resources': resources,
       'stage_m': stage_m}
tag = ('gen' if GENERALIZED_POOL else 'ov') + f"_T{SAUCCSD_TROTTER}"
fname = f'{OUTDIR}/{MOLECULE}_{RSCAN}_{tag}.pkl'
with open(fname, 'wb') as f:
    pickle.dump(out, f)
print(f"\nsaved -> {fname}")
