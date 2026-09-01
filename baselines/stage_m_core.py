"""Stage M core: Sorted Insertion (fully commuting) + KKT-optimal shot allocation.

Lifted VERBATIM from the H2O qubit-space notebook (H2O_Benchmarking cell 2,
lines 1390-1657) so the determinant-basis port and the qubit-space original run
the SAME allocation code.  Do not edit here to fix a determinant-basis problem:
if these two ever disagree, the H2O cross-check that validates the port stops
meaning anything.

Only the module scaffolding is new -- the SI_* knobs became module-level
defaults (they were notebook globals) and numpy/qiskit imports were added.

The one function NOT lifted is `block_sigma`, which materialized each group as
a 2^n x 2^n operator.  Its replacement is stage_m_detbasis.sigma_matrix_detbasis,
which works on the reachable support of O|psi> instead.  Both produce the same
r x r sigma matrix; only the sigma builder differs between representations.
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp, PauliList

__all__ = ['sorted_insertion', 'fc_diagonalizer', 'eigvec_weights',
           'pack_sigma', 'pair_labels', 'kkt_shot_allocation']

SI_EPS_HA = 1.6e-3      # target standard error on each REPORTED energy
SI_VERIFY = True        # check each FC diagonalizer really maps its group to Z
SI_TOL    = 1e-9        # KKT residual tolerance, RELATIVE to eps^2
SI_DTOL   = 1e-8        # multiplier below this FRACTION of the largest active
                        # one counts as collapsed -> slack


# ======================= sorted insertion (fully commuting) ==================
def _si_symplectic(sp_op):
    """(op, x, z, c, is_id) for a SparsePauliOp, simplified and phase-free.

    SparsePauliOp keeps its PauliList phase-free (phases live in coeffs), so x/z
    alone fix the commutation structure.  H is Hermitian => c is real.
    """
    op = sp_op.simplify()
    x = np.asarray(op.paulis.x, dtype=bool)
    z = np.asarray(op.paulis.z, dtype=bool)
    c = np.asarray(op.coeffs)
    if np.max(np.abs(np.imag(c))) > 1e-10:
        print(f"    *** WARNING: non-Hermitian input, max |Im c| = "
              f"{np.max(np.abs(np.imag(c))):.2e} -- taking the real part ***")
    c = np.real(c)
    is_id = ~(x.any(axis=1) | z.any(axis=1))
    return op, x, z, c, is_id


def sorted_insertion(sp_op, tol=1e-12):
    """Fully-commuting Sorted Insertion.

    Returns (op, groups, c, id_coeff); `groups` are index lists into the
    SIMPLIFIED operator.  The identity term is never grouped -- it is a
    constant with zero variance -- and its coefficient is returned separately.

    Commutation is checked on the symplectic form, vectorized over a whole
    group: the per-qubit anticommutation indicator of two Paulis is
        a_q = (x1_q & z2_q) XOR (z1_q & x2_q)
    and they fully commute iff sum_q a_q is EVEN (the two sign flips cancel).
    """
    op, x, z, c, is_id = _si_symplectic(sp_op)
    id_coeff = float(c[is_id].sum()) if is_id.any() else 0.0

    idx = [i for i in range(len(c)) if not is_id[i] and abs(c[i]) > tol]
    idx.sort(key=lambda i: -abs(c[i]))                      # the "sorted" step
    nq, T = x.shape[1], len(idx)

    groups, gx, gz, gn = [], [], [], []
    for i in idx:
        xi, zi = x[i], z[i]
        placed = False
        for g in range(len(groups)):
            m = gn[g]
            a = (gx[g][:m] & zi) ^ (gz[g][:m] & xi)         # (m, nq) bool
            if np.all(a.sum(axis=1) % 2 == 0):
                groups[g].append(i)
                gx[g][m] = xi; gz[g][m] = zi; gn[g] = m + 1
                placed = True
                break
        if not placed:
            bx = np.zeros((T, nq), dtype=bool); bz = np.zeros((T, nq), dtype=bool)
            bx[0] = xi; bz[0] = zi
            groups.append([i]); gx.append(bx); gz.append(bz); gn.append(1)
    return op, groups, c, id_coeff


# --------------------------- FC diagonalizing Cliffords ---------------------
def _gf2_independent(x, z, gidx):
    """Row-echelon over GF(2) on the group's [x|z] rows.

    Only an INDEPENDENT generating set may go to the stabilizer synthesizer: a
    dependent set can multiply out to -I under the all-plus signs used here,
    which is not a valid stabilizer group.  Dropping dependent rows is free --
    they are products of the kept ones, so the same Clifford diagonalizes them.
    """
    basis, keep = [], []
    for i in gidx:
        w = np.concatenate([x[i], z[i]]).astype(np.uint8)
        for b in basis:
            if w[int(np.argmax(b))]:
                w = w ^ b
        if w.any():
            basis.append(w); keep.append(i)
    return keep


def fc_diagonalizer(op, x, z, gidx, verify=SI_VERIFY):
    """Basis change D with D P D^dag Z-type for every P in the group.

    synth_circuit_from_stabilizers returns C with C|0> the stabilizer state of
    the generators, i.e. C Z_k C^dag = g_k, so the readout circuit is D = C^dag.
    Signs are classical post-processing: <P> = sign * <Z_S>, and both are
    recorded so the block is a usable recipe, not only a gate count.
    """
    keep = _gf2_independent(x, z, gidx)
    labels = ['+' + op.paulis[i].to_label().lstrip('+-i') for i in keep]
    qc = synth_circuit_from_stabilizers(labels, allow_redundant=True,
                                        allow_underconstrained=True)
    D = qc.inverse()
    recipe, ok = {}, True
    if verify:
        Cd = Clifford(qc).adjoint()
        for i in gidx:
            q = Pauli(op.paulis[i].to_label().lstrip('+-i')).evolve(Cd, frame='s')
            if q.x.any():                       # not diagonal -> synthesis failed
                ok = False
                break
            recipe[i] = (-1 if q.to_label().startswith('-') else 1,
                         tuple(np.where(q.z)[0]))
    return D, recipe, ok, len(keep)


# ============================ sigma over a block basis ======================
def eigvec_weights(C):
    """w_(i,p) over the upper-triangle index p, from eigenvectors C (r x nstate).

    w = c^4 on the diagonal, 4 c^2 c^2 off it -- the coefficients multiplying
    sigma^2/M in the first-order mean-square error.
    """
    r, ns = C.shape
    di = np.arange(r)
    iu = np.triu_indices(r, 1)
    W = np.zeros((ns, r * (r + 1) // 2))
    # pair index: diagonal entries first, then the strict upper triangle
    for i in range(ns):
        ci2 = np.real(C[:, i] * np.conj(C[:, i]))
        W[i, :r] = ci2 ** 2
        W[i, r:] = 4.0 * ci2[iu[0]] * ci2[iu[1]]
    return W, di, iu


def pack_sigma(S):
    """Flatten a sigma matrix to the same upper-triangle order as the weights."""
    r = S.shape[0]
    iu = np.triu_indices(r, 1)
    return np.concatenate([np.diag(S), S[iu]])

def pair_labels(r):
    iu = np.triu_indices(r, 1)
    return [(m, m) for m in range(r)] + list(zip(iu[0].tolist(), iu[1].tolist()))


# =================== Algorithm 2: active-set damped Newton ==================
def kkt_shot_allocation(W, sig, eps, tol=SI_TOL, dtol=SI_DTOL,
                        max_outer=50, max_inner=200, verbose=False):
    """Minimize sum_p M_p subject to  sum_p W[i,p] sig[p]^2 / M_p <= eps^2  for
    every state i, via the KKT system

        M*_p        = sig_p sqrt( (lambda . W)_p )
        F_i(lambda) = sum_p W[i,p] sig_p (lambda . W)_p^(-1/2)  =  eps^2

    Returns (lam, M_star, F, active, info).  Exactly reproduces the closed form
    M = (sum_p sig_p sqrt(W_p))^2 / eps^2 when a single state is active.
    """
    ns, npair = W.shape
    e2 = eps ** 2
    tau = tol * e2                    # residuals are compared against eps^2
    lam = np.zeros(ns)
    active = np.ones(ns, dtype=bool)

    # --- seed (step 2): treat each state as if it shared the elements evenly
    tot_w = W.sum(axis=0)
    safe = tot_w > 0
    gj = np.zeros(ns)
    for j in range(ns):
        gj[j] = np.sum(W[j, safe] * sig[safe] / np.sqrt(tot_w[safe]))
    lam = (gj / e2) ** 2

    def _d(l):
        return np.maximum(l @ W, 1e-300)          # (lambda . w)_p, floored

    def _F(l):
        dd = _d(l)
        return W @ (sig / np.sqrt(dd))

    info = {'outer': 0, 'inner': 0, 'lstsq': 0, 'converged': False}
    for _outer in range(max_outer):
        info['outer'] += 1
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        for _inner in range(max_inner):          # damped Newton on the active set
            info['inner'] += 1
            dd = _d(lam)
            F = W @ (sig / np.sqrt(dd))
            r_ = F[idx] - e2
            if np.max(np.abs(r_)) < tau:
                break
            # J_jk = -1/2 sum_p W_jp d_p^(-3/2) W_kp sig_p
            coef = sig / dd ** 1.5
            J = -0.5 * (W[idx] * coef) @ W[idx].T
            if np.linalg.cond(J) > 1e10:
                delta, *_ = np.linalg.lstsq(J, r_, rcond=None)
                info['lstsq'] += 1
            else:
                delta = np.linalg.solve(J, r_)
            # fraction-to-the-boundary, then backtracking on the residual
            pos = delta > 0
            alpha = 1.0
            if np.any(pos):
                alpha = min(1.0, 0.99 * np.min(lam[idx][pos] / delta[pos]))
            base = np.max(np.abs(r_))
            accepted = False
            for _bt in range(60):
                trial = lam.copy()
                trial[idx] = np.maximum(lam[idx] - alpha * delta, 0.0)
                if np.max(np.abs(_F(trial)[idx] - e2)) < base:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:                      # no descent direction left
                break
            lam = trial
        # --- active-set update (steps 11-13)
        F = _F(lam)
        new_active = active.copy()
        scale = np.max(lam[active]) if active.any() else 0.0
        collapsed = np.where(active)[0][lam[active] <= dtol * max(scale, 1e-300)]
        new_active[collapsed] = False                                  # collapsed
        new_active[(~active) & (F > e2 + tau)] = True                  # binding
        if np.array_equal(new_active, active):
            info['converged'] = (np.max(np.abs(_F(lam)[active] - e2)) < tau
                                 if active.any() else True)
            break
        # an inactive state carries lambda = 0 exactly, or its stale value keeps
        # contributing to (lambda . w) and biases every M_p
        lam[~new_active] = 0.0
        active = new_active

    dd = _d(lam)
    M_star = sig * np.sqrt(dd)
    F = _F(lam)
    info['max_violation'] = float(np.max(F - e2) / e2)
    if info['max_violation'] > 1e-6:
        info['converged'] = False
    return lam, M_star, F, active, info


def _closed_form_single(W_row, sig, eps):
    """(sum_p sig_p sqrt(W_p))^2 / eps^2 -- the one-active-state solution."""
    return float(np.sum(sig * np.sqrt(W_row)) ** 2 / eps ** 2)
