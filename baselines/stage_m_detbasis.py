"""STAGE M in the determinant representation -- sigma matrices at 24 qubits.

WHY THIS EXISTS
---------------
The qubit-space pipeline computes each commuting group's Gram matrices as

    O = SparsePauliOp(op.paulis[g], c[g]).to_matrix(sparse=True)
    Y = O @ V;   A = V^dag Y;   G = Y^dag Y

which materializes a 2^n x 2^n operator.  At H2O's 14 qubits that is 16,384^2
and fine; at H2O2's 24 qubits it is 16.8M x 16.8M and hopeless.  This module
computes the SAME A and G without ever forming O.

THE KEY FACT
------------
Unlike H and the UCC generators, a commuting Pauli group does NOT conserve
(N, S_z): O|psi> leaves the determinant sector, so -- unlike every other stage
of the determinant-basis port -- we cannot simply stay in the 48,400-dim space.

But we do not need dense Fock vectors either.  A Pauli string acts on a
computational basis state as a bit flip plus a phase, so for a group with k
terms the image O|psi> is supported on at most

    k * n_det   entries        (48,400 * k for H2O2, vs 16,777,216 dense)

Working on that support is ~170x cheaper than dense at k ~ 150, and the
alternative of expanding O^2 into k^2 in-sector Pauli products is ~150x worse
than that again.  So: build OV on its reachable support, then contract.

CONVENTION
----------
Qiskit's Pauli is  P = (-i)^phase * Z^z X^x  (symplectic (x, z) + phase), so on
a computational basis state |b>:

    P|b> = (-i)^phase * (-1)^popcount(z & (b ^ x)) * |b ^ x>

Bitstrings are packed with spin-orbital i in bit i, matching the Jordan-Wigner
ordering used to build the qubit Hamiltonian.  `sector_bitstrings` builds them
from PySCF's (alpha, beta) determinant strings; PASS THE SAME so_index MAPPING
USED FOR THE JW TRANSFORM -- if the two disagree the sigmas are silently wrong,
which is why `selftest_against_dense` exists.

The r x r Gram trick from the qubit pipeline is unchanged: memory stays O(r^2)
per group rather than O(r^2 * dim).
"""
import multiprocessing as mp
import os

import numpy as np

__all__ = ['sector_bitstrings', 'group_grams', 'sigma_matrix_detbasis',
           'sector_expectation', 'verify_bitstrings', 'selftest_against_dense']


# ----------------------------------------------------------------------------
# bit utilities
# ----------------------------------------------------------------------------
if hasattr(np, 'bitwise_count'):                      # numpy >= 2.0
    def _popcount(a):
        return np.bitwise_count(a)
else:
    _POP8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    def _popcount(a):
        """popcount of int64 array, byte-table lookup."""
        v = np.ascontiguousarray(a).view(np.uint8).reshape(*a.shape, 8)
        return _POP8[v].sum(axis=-1)


def _symplectic(op):
    """(x_int, z_int, phase) with P = (-i)^phase * Z^z X^x, qubit i in bit i.

    NOTE the phase.  PauliList.phase is the CANONICAL phase, which already has
    the Y factors divided out -- it reads 0 for 'XY' and 'YY' alike.  The
    exponent that goes with Z^z X^x is the internal one, related by

        _phase = (phase + n_Y) mod 4,     n_Y = #qubits with x & z

    Using .phase directly costs a factor (-i)^n_Y per term and silently
    corrupts every sigma; this is what selftest_against_dense catches.
    """
    xs = np.asarray(op.paulis.x, dtype=np.int64)
    zs = np.asarray(op.paulis.z, dtype=np.int64)
    w = (np.int64(1) << np.arange(xs.shape[1], dtype=np.int64))
    n_y = (xs & zs).sum(axis=1)
    phase = (np.asarray(op.paulis.phase, dtype=int) + n_y) % 4
    return xs @ w, zs @ w, phase


def sector_bitstrings(strs_a, strs_b, norb, so_index=None):
    """Pack PySCF determinant strings into Jordan-Wigner bitstrings.

    Parameters
    ----------
    strs_a, strs_b : (na,), (nb,) int arrays
        PySCF cistring occupation bitmasks, orbital p in bit p.
    norb : int
        Number of spatial orbitals in the CAS.
    so_index : callable (p, sigma) -> int, optional
        Spin-orbital index for spatial orbital p and spin sigma in {0: alpha,
        1: beta}.  Default is the OpenFermion interleaved convention
        (alpha -> 2p, beta -> 2p+1).  Override for a blocked convention
        (alpha -> p, beta -> norb + p).

    Returns
    -------
    bits : (na*nb,) int64
        Bitstring per determinant, ordered as PySCF's ci.ravel(), i.e.
        determinant (ia, ib) at index ia*nb + ib.
    """
    if so_index is None:
        def so_index(p, s):
            return 2 * p + s

    na, nb = len(strs_a), len(strs_b)
    amap = np.zeros(na, dtype=np.int64)
    bmap = np.zeros(nb, dtype=np.int64)
    for p in range(norb):
        ba, bb = np.int64(1) << so_index(p, 0), np.int64(1) << so_index(p, 1)
        amap |= np.where((np.asarray(strs_a) >> p) & 1, ba, 0)
        bmap |= np.where((np.asarray(strs_b) >> p) & 1, bb, 0)
    return (amap[:, None] | bmap[None, :]).ravel()


# ----------------------------------------------------------------------------
# per-group Gram matrices
# ----------------------------------------------------------------------------
def group_grams(px, pz, pph, pc, V, bits, bits_sorted, bits_order,
                chunk=1 << 19):
    """A = V^dag O V  and  G = (OV)^dag (OV)  for one commuting group.

    px, pz : (k,) int64      symplectic X and Z masks of the group's Paulis
    pph    : (k,) int        Qiskit phase exponents (P = (-i)^pph Z^z X^x)
    pc     : (k,) complex    coefficients
    V      : (n_det, r)      reference block, columns in the determinant sector
    bits   : (n_det,) int64  bitstring of each determinant
    bits_sorted, bits_order  np.sort(bits), np.argsort(bits)  -- precomputed
                             once by the caller and reused for every group
    chunk  : int             support entries per pass; controls peak memory
                             (chunk * r * 16 bytes)

    The image entries are indexed e = j*n_det + d for Pauli j and determinant d,
    so sorting by target bitstring and reducing within runs accumulates OV
    without ever allocating the full (k*n_det, r) block.
    """
    n_det, r = V.shape
    k = len(pc)

    # target bitstring of every (Pauli, determinant) pair, then group equal ones
    #
    # Two narrowings of the sort.  Measured in ISOLATION at H2O2 scale
    # (n_det = 48,400, k = 200):
    #
    #   int64 + stable     6.35 s   77.4 MB      <- what this used to be
    #   int64 + quicksort  1.23 s   77.4 MB       5.2x
    #   int32 + quicksort  1.03 s   38.7 MB       6.1x
    #
    # But DO NOT mistake this for the hot spot.  Measured inside this function
    # (k = 100, n_det = 20,000, r = 20) the sort is only 8.9% of the runtime --
    # the other 91% is the chunked accumulation below, dominated by the
    # V[d, :] gather and add.reduceat.  These two narrowings are worth ~1.1x
    # overall, not the 6x the isolated numbers suggest.  They are kept because
    # halving the allocation also halves each fork worker's working set, and
    # sigma_matrix_detbasis was bandwidth-bound rather than compute-bound.
    #
    # int32 is safe whenever the bitstrings fit in 31 bits -- 24 qubits for
    # H2O2, 14 for H2O.  Wider registers fall back to int64 automatically.
    #
    # quicksort rather than 'stable' is safe because starts, uniq and sec_row
    # are all built from the sorted VALUES, which are identical under either
    # kind (verified explicitly).  Stability fixes only the order of entries
    # WITHIN a run of equal targets, and those entries are summed by the
    # add.reduceat below -- so the sole effect is floating-point rounding of
    # that sum, not which entries are grouped together.
    _wide = max(int(bits.max()), int(px.max()))
    _dt = np.int32 if 0 <= _wide < (1 << 31) else np.int64
    Y = (bits.astype(_dt, copy=False)[None, :]
         ^ px.astype(_dt, copy=False)[:, None]).ravel()
    order = np.argsort(Y, kind='quicksort')
    Ys = Y[order]
    del Y
    starts = np.flatnonzero(np.r_[True, Ys[1:] != Ys[:-1]])
    uniq = Ys[starts].astype(bits.dtype, copy=False)
    del Ys

    coef = pc * ((-1j) ** np.asarray(pph))             # fold Qiskit's phase in

    A = np.zeros((r, r), dtype=complex)
    G = np.zeros((r, r), dtype=complex)

    # which support rows land back inside the sector -> those feed A
    hit = np.searchsorted(bits_sorted, uniq)
    np.clip(hit, 0, n_det - 1, out=hit)
    in_sec = bits_sorted[hit] == uniq
    sec_row = np.where(in_sec, bits_order[hit], -1)

    nseg = len(starts)
    seg = 0
    while seg < nseg:
        seg_hi = min(nseg, seg + max(1, chunk // max(1, k)))
        lo = starts[seg]
        hi = starts[seg_hi] if seg_hi < nseg else len(order)
        ent = order[lo:hi]
        j, d = ent // n_det, ent % n_det
        sgn = 1.0 - 2.0 * (_popcount(pz[j] & (bits[d] ^ px[j])) & 1)
        vals = (coef[j] * sgn)[:, None] * V[d, :]
        rel = starts[seg:seg_hi] - lo
        OVc = np.add.reduceat(vals, rel, axis=0)       # (seg_hi-seg, r)
        del vals

        G += OVc.conj().T @ OVc
        m = sec_row[seg:seg_hi] >= 0
        if m.any():
            A += V[sec_row[seg:seg_hi][m], :].conj().T @ OVc[m]
        seg = seg_hi

    return A, G


def _group_contribution(A, G, r, iu):
    """One group's contribution to S, from its Gram matrices.

    Factored out so the serial and parallel paths cannot drift: both call this
    with the same A, G and sum the results in the same order.
    """
    dA, dG = np.real(np.diag(A)), np.real(np.diag(G))
    var = np.zeros((r, r))
    np.fill_diagonal(var, dG - dA ** 2)

    # OFF-DIAGONALS: HADAMARD TEST, matching the Q-SENSE side.
    #
    # The subspace off-diagonal Re<mu|O|nu> can be estimated two ways and the
    # variance differs, so the two methods being compared must use the same one.
    # Q-SENSE's VO benchmark uses an ancilla-controlled construction
    # (get_parallelswap_subcircuit with control_qubit_pos=NQ, variance taken over
    # NQ+1 qubits), so this does too.
    #
    # The MEAN-DIAGONAL SHIFT is essential and was missing from the first
    # version of this: Q-SENSE subtracts s = (H_ii + H_jj)/2 before building its
    # augmented operator (_offdiag_task: ij_shift, H_rot - ij_shift), so the
    # baseline must too.  Without it the baseline measures raw H, whose second
    # moment is ~E^2 (about 5600 Ha^2 for H2O), while Q-SENSE measures only the
    # fluctuation about the mean -- an asymmetry worth 40-270x in the baseline's
    # disfavour.  The shift leaves the off-diagonal itself unchanged because the
    # states are orthonormal.
    #
    #   |Psi>       = (|0>|mu> + |1>|nu>)/sqrt(2)      on 1 + n qubits
    #   s           = (A_mumu + A_nunu)/2
    #   O'          = O - s ,   O~ = X (x) O'
    #   <O~>        = Re<mu|O|nu>                       unchanged by the shift
    #   <O'^2>_mu   = G_mumu - 2 s A_mumu + s^2
    #   <O~^2>      = (G_mumu + G_nunu)/2 - s^2         using A_mumu + A_nunu = 2s
    #   Var         = (G_mumu + G_nunu)/2 - s^2 - (Re A_munu)^2
    #
    # WAS the paired-reference (superposition) estimator, which measures
    # (|mu> + |nu>)/sqrt(2) with no ancilla:
    #     m1 = (A_mumu + A_nunu)/2 + Re A_munu
    #     m2 = (G_mumu + G_nunu)/2 + Re G_munu
    #     Var = m2 - m1^2
    # That is a legitimate estimator but a DIFFERENT one -- it folds the two
    # diagonal expectations into the measured quantity, so its variance is not
    # comparable to an ancilla-based Hadamard test.  Using it for the baselines
    # while Q-SENSE used the swap construction made the eps^2 M comparison
    # apples-to-oranges.
    #
    # Non-negative by construction: |<mu|O|nu>| <= min(||O|mu>||, ||O|nu>||),
    # so (Re A)^2 <= min(G_mumu, G_nunu) <= (G_mumu + G_nunu)/2.
    shift = 0.5 * (dA[:, None] + dA[None, :])          # (H_mumu + H_nunu)/2
    var[iu] = (0.5 * (dG[:, None] + dG[None, :])
               - shift ** 2 - np.real(A) ** 2)[iu]
    return np.sqrt(np.maximum(var, 0.0))


def _default_nparal():
    """Worker count, matching the drivers' get_n_jobs() precedence exactly.

    BENCH_N_JOBS first: H2O2_Benchmark_detbasis.py documents BENCH_N_JOBS=1 as
    the way to get a clean single-stream trace, and that has to silence THIS
    pool too or the instruction is a lie.  SLURM_CPUS_ON_NODE is included for
    the same reason -- a job that sets only that one would otherwise get
    os.cpu_count() here and the allocation's count everywhere else.
    """
    for v in ('BENCH_N_JOBS', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        if os.environ.get(v):
            try:
                return max(1, int(os.environ[v]))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


# Payload for forked workers.  Set in the parent BEFORE the pool is created, so
# every worker inherits V, bits and the symplectic arrays by copy-on-write
# rather than having them pickled once per task -- V alone is n_det x r, which
# at H2O2 scale would dominate the transfer if it were sent per group.
_SIGMA_PAYLOAD = {}


def _sigma_group_task(gi):
    d = _SIGMA_PAYLOAD
    g = d['groups'][gi]
    A, G = group_grams(d['x'][g], d['z'][g], d['ph'][g], d['c'][g],
                       d['V'], d['bits'], d['bits_sorted'], d['bits_order'],
                       chunk=d['chunk'])
    return gi, _group_contribution(A, G, d['r'], d['iu'])


def sigma_matrix_detbasis(op, groups, c, V, bits, chunk=1 << 19, progress=None,
                          nparal=None):
    """Drop-in replacement for the qubit pipeline's sigma builder.

    Returns S (r x r): S[mu, mu] is sigma of reference |psi_mu>, and S[mu, nu]
    for mu < nu is sigma of the paired reference (|psi_mu> + |psi_nu>)/sqrt2,
    each summed over groups -- exactly the quantity the KKT allocation consumes.

    op     : SparsePauliOp   full qubit Hamiltonian
    groups : list[array]     index sets from sorted_insertion
    c      : (nterm,) complex coefficients aligned with op.paulis
    nparal : int or None     worker processes over the group loop.  None uses
                             _default_nparal(), which follows the SAME env
                             precedence as the benchmark drivers' get_n_jobs().
                             Pass 1 to force the serial path.

    PARALLELISM.  The group loop is the whole cost of this function and the
    groups are independent, so it is run over a fork pool.

    Do not expect linear scaling.  The cost inside group_grams is ~91% the
    chunked V[d, :] gather and add.reduceat, which are memory-bandwidth bound,
    not compute bound -- measured 1.4-2.1x on 8 cores, and 8 workers can be
    SLOWER than 4 once they contend for the memory bus.  A 192-core node has
    more aggregate bandwidth and may do better, but measure it rather than
    assuming.  The real fix is to replace the gather/reduceat with a single
    sparse CSR product (OV = P @ V), which shrinks each worker's working set
    enough for the pool to scale; that has NOT been done.

    Results are accumulated in GROUP ORDER, not completion order, so the sum is
    bitwise identical to the serial path rather than merely close.

    Set OMP_NUM_THREADS=1 (and MKL/OPENBLAS) in the caller.  Each worker
    otherwise spawns its own BLAS threads for the contraction and oversubscribes
    the node.
    """
    r = V.shape[1]
    S = np.zeros((r, r))
    iu = np.triu_indices(r, 1)

    bits_order = np.argsort(bits, kind='stable')
    bits_sorted = bits[bits_order]

    x_int, z_int, phase = _symplectic(op)
    c = np.asarray(c)
    ngroup = len(groups)

    if nparal is None:
        nparal = _default_nparal()
    nparal = max(1, min(int(nparal), ngroup))

    # fork is required: the workers inherit V and bits instead of receiving
    # them.  Anything else (spawn, threads) either re-pickles the payload per
    # task or serialises on the GIL, and both erase the gain.
    ctx = None
    if nparal > 1:
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = None

    if ctx is None:
        for gi, g in enumerate(groups):
            A, G = group_grams(x_int[g], z_int[g], phase[g], c[g],
                               V, bits, bits_sorted, bits_order, chunk=chunk)
            S += _group_contribution(A, G, r, iu)
            if progress and (gi + 1) % progress == 0:
                print(f"      [sigma] group {gi+1}/{ngroup}", flush=True)
        return S

    _SIGMA_PAYLOAD.update(groups=groups, x=x_int, z=z_int, ph=phase, c=c,
                          V=V, bits=bits, bits_sorted=bits_sorted,
                          bits_order=bits_order, chunk=chunk, r=r, iu=iu)

    # Largest groups first.  Cost is superlinear in group size, so leaving a
    # 200-term group until last strands it as a tail while the pool idles.
    order = sorted(range(ngroup), key=lambda i: -len(groups[i]))

    parts = [None] * ngroup
    done = 0
    try:
        with ctx.Pool(nparal) as pool:
            for gi, contrib in pool.imap_unordered(_sigma_group_task, order,
                                                   chunksize=1):
                parts[gi] = contrib
                done += 1
                if progress and done % progress == 0:
                    print(f"      [sigma] group {done}/{ngroup} "
                          f"({nparal} workers)", flush=True)
    finally:
        _SIGMA_PAYLOAD.clear()

    for contrib in parts:                       # group order -> deterministic
        S += contrib
    return S


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def sector_expectation(op, V, bits):
    """diag(V^dag O V) for the FULL operator, using only in-sector terms.

    Cheaper than group_grams because no G is needed: a Pauli contributes to
    <v|O|v> only where b and b ^ x are BOTH determinants, so this never leaves
    the sector and costs one searchsorted per term instead of a support build.
    """
    order = np.argsort(bits, kind='stable')
    bs = bits[order]
    x_int, z_int, phase = _symplectic(op)
    coef = np.asarray(op.coeffs) * ((-1j) ** phase)

    out = np.zeros(V.shape[1], dtype=complex)
    for j in range(len(coef)):
        y = bits ^ x_int[j]
        hit = np.searchsorted(bs, y)
        np.clip(hit, 0, len(bs) - 1, out=hit)
        ok = bs[hit] == y
        if not ok.any():
            continue
        src = np.flatnonzero(ok)
        dst = order[hit[src]]
        sgn = 1.0 - 2.0 * (_popcount(z_int[j] & y[src]) & 1)
        out += coef[j] * np.einsum('ik,i,ik->k', V[src].conj(), sgn, V[dst])
    return out


def verify_bitstrings(op, V, bits, expected, atol=1e-8, label='qubit H'):
    """Check the bitstring convention at ANY qubit count.

    <v|H_qubit|v> for an eigenvector v must reproduce its FCI energy.  If
    so_index does not match the mapper's spin-orbital ordering the energies come
    out wrong (usually wildly), so this catches a convention mismatch without
    needing the 2^n dense operator -- which is the whole point at 24 qubits.

    `expected` are total energies; the qubit operator built from
    ElectronicEnergy(constants={'core': ecore}) already carries the core term.
    """
    got = np.real(sector_expectation(op, V, bits))
    exp = np.asarray(expected, dtype=float)
    d = np.abs(got - exp)
    print(f"  {label} vs FCI: max |diff| = {d.max():.3e}  "
          f"({'PASS' if d.max() < atol else 'FAIL'})")
    for i, (g, e) in enumerate(zip(got, exp)):
        print(f"    state {i}: <v|H|v> = {g:16.10f}   FCI = {e:16.10f}   "
              f"diff = {g - e:+.2e}")
    return d.max()


def selftest_against_dense(op, groups, c, V, bits, nq, atol=1e-9):
    """Compare against the dense qubit-space path on a small case.

    Only run where 2^nq is small (H2O's 14 qubits).  Embeds V into the full
    Fock space, forms each group operator explicitly, and checks A and G --
    which validates the bitstring convention as well as the arithmetic, and is
    the ONLY thing that catches an so_index mismatch.
    """
    from qiskit.quantum_info import SparsePauliOp
    dim, r = 1 << nq, V.shape[1]
    Vf = np.zeros((dim, r), dtype=complex)
    Vf[bits, :] = V

    bits_order = np.argsort(bits, kind='stable')
    bits_sorted = bits[bits_order]
    x_int, z_int, phase = _symplectic(op)

    worst = 0.0
    for g in groups:
        O = SparsePauliOp(op.paulis[g], np.asarray(c)[g]).to_matrix(sparse=True)
        Yf = O @ Vf
        A_ref, G_ref = Vf.conj().T @ Yf, Yf.conj().T @ Yf
        A, G = group_grams(x_int[g], z_int[g], phase[g], np.asarray(c)[g],
                           V, bits, bits_sorted, bits_order)
        worst = max(worst, np.max(np.abs(A - A_ref)), np.max(np.abs(G - G_ref)))
    print(f"  selftest: max |delta| over {len(groups)} groups = {worst:.3e} "
          f"({'PASS' if worst < atol else 'FAIL'})")
    return worst
