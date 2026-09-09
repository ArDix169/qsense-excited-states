import numpy as np
from openfermion import get_sparse_operator

def expectation(Op, State):
    return (State @ Op @ State.conj().T)[0,0]

def matrix_element(Op, Bra, Ket):
    return (Bra @ Op @ Ket.conj().T)[0,0]

def variance_of_operator(Op, State):
    """
    computes the variance of a Hermitian operator <psi|H^2|psi> - <psi|H|psi>^2
    """
    first  = (State @ Op) @ (Op @ State.conj().T)
    second = (State @ Op @ State.conj().T) ** 2
    return first - second

def variance_of_general_operator(Op, State):
    """
    computes the variance of a general non-Hermitian operator<psi|H^t H|psi> - <psi|H^t|psi><psi|H|psi>

    note that qubit Hamiltonians with complex coefficients are not Hermitian. But a QWC or FC Hamiltonian can be measured independent of
    what the coefficients are
    """
    first  = (State @ Op.conjugate().transpose()) @ (Op @ State.conj().T)
    second = (State @ Op @ State.conj().T)
    third  = (State @ Op.conjugate().transpose() @ State.conj().T) 
    return first - (second * third)

def variance_of_decomp(decomp, State, N, general=False):
    if not general:
        var_list = [variance_of_operator(get_sparse_operator(Op, N), State) for Op in decomp]
    else:
        var_list = [variance_of_general_operator(get_sparse_operator(Op, N), State) for Op in decomp]
    var_list = np.array([x.toarray()[0,0] for x in var_list])
    return np.sum((var_list)**(1/2))**2

def sampling_cost(c, sig_matrix):
    Nstates = len(c)
    cost    = 0
    for i in range(Nstates):
        for j in range(Nstates):
            if i == j:
                cost += c[i] ** 2 * sig_matrix[i,i]
            elif i > j:
                cost += 2 * np.abs(c[i] * c[j]) * sig_matrix[i,j]
    return cost ** 2

def solve_lambdas(n_states, w, sigma, eps, tol=1e-8, verbose=True):
    """
    Solve the per-state variance KKT system

        F_i(lambda) <= eps^2,   lambda_i >= 0,   lambda_i (F_i - eps^2) = 0

    via an active-set Newton method.

    A state is BINDING  if its constraint is tight (F_i = eps^2, lambda_i > 0).
    A state is SLACK    if it's below target for free (F_i < eps^2, lambda_i = 0):
                        the shots demanded by the binding states already drive
                        its variance under eps^2, so it costs nothing extra.

    Parameters
    ----------
    w     : (n_states, n_elem)  weights w^(j)_e
    sigma : (n_elem,)           sigma_e per matrix element
    eps   : float               per-state target std (variance target = eps**2)

    Returns
    -------
    lam      : (n_states,)  multipliers (0 for slack states)
    active   : (n_states,)  bool mask, True = binding
    """
    n = w.shape[0]
    eps2 = eps ** 2

    # Floor for d = lam @ w.  It CANNOT be 1e-300, which is what this was:
    # d itself is representable there, but the Jacobian needs d^(3/2), and
    # (1e-300)^(3/2) = 1e-450 underflows to exactly 0.0 in float64.  w / d32
    # then yields inf, inf * 0 in the matmul yields nan, np.linalg.cond(J) of a
    # nan matrix is nan, `nan > 1e10` is False, so the ill-conditioned branch is
    # skipped and np.linalg.solve is handed a nan matrix -- LAPACK prints
    # "On entry to DLASCL, parameter number 4 had an illegal value" and numpy
    # reports the misleading "Singular matrix".
    #
    # d hits the floor whenever some matrix element has zero weight in EVERY
    # state, which happens once the subspace spans its whole CSF sector.
    # 1e-200 is the largest round power of ten whose 3/2 power (1e-300) is still
    # a normal float64, so it protects d32 as well as d12.
    D_FLOOR = 1e-200

    def variances(lam):
        d = np.maximum(lam @ w, D_FLOOR)
        return (w * sigma / np.sqrt(d)).sum(axis=1)

    # ------------------------------------------------------------------ #
    #  inner solve: F_i = eps^2 on the active block, lambda = 0 elsewhere #
    # ------------------------------------------------------------------ #

    def newton_on_active(active, max_iter=500):
        wsum = np.maximum(w.sum(axis=0), D_FLOOR)
        g = (w * sigma / np.sqrt(wsum)).sum(axis=1)  # per-state scale
        lam = np.zeros(n)
        lam[active] = (g[active] / eps2) ** 2  # seed active only

        last_rnorm, last_alpha, it_used = np.inf, 1.0, 0
        for it in range(max_iter):
            it_used = it
            lam[~active] = 0.0
            d = np.maximum(lam @ w, D_FLOOR)
            d12 = np.sqrt(d)
            d32 = np.maximum(d * d12, D_FLOOR)
            F = (w * sigma / d12).sum(axis=1)
            r = (F - eps2)[active]

            last_rnorm = np.max(np.abs(r)) if active.any() else 0.0
            if last_rnorm < tol:
                break

            Jfull = -0.5 * (w / d32) @ (w * sigma).T
            J = Jfull[np.ix_(active, active)]
            # A non-finite cond means J is not usable at all.  Treated as
            # ill-conditioned rather than falling through to solve(), because
            # `nan > 1e10` is False and would silently take the WRONG branch.
            try:
                cond = np.linalg.cond(J)
            except np.linalg.LinAlgError:
                cond = np.inf
            if not np.isfinite(cond) or cond > 1e10:
                step = np.linalg.lstsq(J, r, rcond=1e-12)[0]
            else:
                step = np.linalg.solve(J, r)

            delta = np.zeros(n)
            delta[active] = step

            # fraction-to-the-boundary over active states only
            alpha = 1.0
            pos = (delta > 0) & active
            if pos.any():
                alpha = min(alpha, 0.99 * np.min(lam[pos] / delta[pos]))

            # backtrack on the active residual
            rnorm = last_rnorm
            for _ in range(200):
                lam_try = lam.copy()
                lam_try[active] = np.maximum(lam[active] - alpha * delta[active], 0.0)
                d_try = np.maximum(lam_try @ w, D_FLOOR)
                F_try = (w * sigma / np.sqrt(d_try)).sum(axis=1)
                if np.max(np.abs((F_try - eps2)[active])) < rnorm:
                    break
                alpha *= 0.5
            last_alpha = alpha

            lam[active] = np.maximum(lam[active] - alpha * delta[active], 0.0)

        if verbose:
            print(f"    [inner] iters={it_used:4d}  max|r_active|={last_rnorm:.3e}  "
                  f"alpha_final={last_alpha:.2e}  "
                  f"{'CONVERGED' if last_rnorm < tol else 'STALLED'}")
        return lam

    # ------------------------------------------------------------------ #
    #  outer active-set loop                                             #
    # ------------------------------------------------------------------ #

    active = np.ones(n, dtype=bool)  # start: assume all states bind
    if verbose:
        print(f"[active-set] start: assume all {n} states binding")

    lam = newton_on_active(active)
    for outer in range(2 * n + 5):
        F = variances(lam)
        r = F - eps2

        big = max(lam.max(), 1.0)
        # (a) binding state whose multiplier collapsed -> really SLACK, drop
        drop = active & (lam <= 1e-8 * big)
        # (b) slack state now ABOVE target -> must BIND, add
        add = (~active) & (r > tol)

        if verbose:
            print(f"[active-set] pass {outer}: "
                  f"binding={active.astype(int)}  "
                  f"max|r|={np.max(np.abs(r)):.3e}")
            if drop.any():
                print(f"             DROP states {np.where(drop)[0].tolist()} "
                      f"(lambda collapsed -> slack)")
            if add.any():
                print(f"             ADD  states {np.where(add)[0].tolist()} "
                      f"(F > eps^2 -> binding)")

        if not drop.any() and not add.any():
            if verbose:
                print(f"[active-set] converged after {outer} update(s)")
            break

        active = (active & ~drop) | add
        lam = newton_on_active(active)

    # ------------------------------------------------------------------ #
    #  final KKT report                                                  #
    # ------------------------------------------------------------------ #

    F = variances(lam)
    r = F - eps2
    if verbose:
        print("\n  ---------- KKT solution ----------")
        for i in range(n):
            tag = "BINDING" if active[i] else "slack  "
            print(f"   state {i}: {tag}  lambda={lam[i]:.4e}  "
                  f"F={F[i]:.4e}  F-eps^2={r[i]:+.3e}")
        primal_ok = np.all(r <= tol)  # every state within budget
        dual_ok = np.all(lam >= -tol)  # multipliers non-negative
        comp_ok = np.all(np.abs(lam * r) <= 1e-4 * max(big, 1.0))  # slackness
        print(f"   primal feasible (all F<=eps^2): {primal_ok}")
        print(f"   dual feasible   (all lambda>=0): {dual_ok}")
        print(f"   complementary slackness        : {comp_ok}")
        if primal_ok and dual_ok and comp_ok:
            print("   ==> KKT satisfied: optimal allocation ✓")
        else:
            print("   ==> KKT NOT satisfied — see violations above")
        print("  ----------------------------------\n")

    j = np.argmax(lam)  # the runaway state (largest multiplier)
    d = lam @ w  # total shot-density per element
    contrib = w[j] * sigma / np.sqrt(d)  # state j's variance contribution per element

    order = np.argsort(contrib)[::-1]  # elements sorted by how much they drive F_j
    print(f"runaway state j = {j}, lambda_j = {lam[j]:.3e}")
    print(f"F_j (total) = {contrib.sum():.3e},  eps^2 = {eps ** 2:.3e}\n")

    print(f"{'elem':>6} {'contrib':>11} {'w_j':>11} {'sigma':>10} {'d':>11} {'w_others_max':>13}")
    for e in order[:8]:
        if n_states == 1:
            w_others = 0.0
        else:
            w_others = np.delete(w[:, e], j).max()
        print(f"{e:6d} {contrib[e]:11.3e} {w[j, e]:11.3e} {sigma[e]:10.3e} {d[e]:11.3e} {w_others:13.3e}")
    return np.maximum(lam, 0.0)

def optimal_allocation(lam, w, sigma, integer=False):
    """
    Compute optimal shot allocation M_e for each matrix element.
    M_e = sigma_e * sqrt(lambda . w_e)

    Returned CONTINUOUS by default.  It used to be `np.round(M_e).astype(int)`
    unconditionally, which silently destroyed the total whenever the optimal
    per-element allocation fell below half a shot: every such element became
    exactly 0 and dropped out of the sum.

    That is not a rare corner.  The optimum spreads M_tot over ~N^2/2 elements,
    so mean M_e = M_tot / (N^2/2) is small for any well-conditioned subspace.
    Measured on the H2O A1 singlet runs, mean M_e is 2-3 at n=3,4 and BELOW 0.5
    for the cheapest points -- those reported a sampling cost of exactly 0.0,
    with a converged active set, KKT satisfied and healthy multipliers.  The
    H2O2 runs are in the same regime with mean M_e of 0.001-0.14, so their
    totals came from the handful of elements that happened to clear 0.5.

    M_e is an expected shot count entering eps^2 * sum(M_e).  Rounding it is
    not a physical constraint, it is a lossy discretization of the very
    quantity being reported.  Pass integer=True only for a per-element schedule
    that must be handed to hardware, and even then round the ALLOCATION, never
    the cost.
    """
    M_e = sigma * np.sqrt(lam @ w)
    return np.round(M_e).astype(int) if integer else M_e


def weighted_sampling_cost(no_states, c, sig_matrix, eps=1.6e-3):

    # Parameters
    # ----------
    # no_states : int
    #     Number of target states. Must equal c.shape[1] (the number of
    #     eigenvector columns).
    # c : (n_dim, no_states) array, complex allowed
    #     Eigenvectors of the subspace Hamiltonian as columns; column j is the
    #     eigenvector of target state j in the CSF/basis-state basis. A 1-D
    #     array is promoted to a single column. Should be normalized.
    # sig_matrix : (n_dim, n_dim) array
    #     Standard deviation of the estimator for each Hamiltonian matrix
    #     element; sig_matrix[mu, nu] = sigma of H_{mu,nu}. Absolute value is
    #     taken internally.
    # eps : float, optional
    #     Per-state target energy standard deviation (variance target is
    #     eps**2). Default 1.6e-3 Ha = chemical accuracy.

    c      = np.array(c)
    sig_matrix = np.abs(sig_matrix)
    sigma  = np.real(np.array(sig_matrix)).reshape(-1)

    if c.ndim == 1:
        c = c[:, np.newaxis]

    n_dim    = c.shape[0]
    n_elem   = n_dim ** 2
    n_states = no_states

    # build weight matrix w — shape (n_states, n_elem)
    w_3d = np.zeros((n_states, n_dim, n_dim))
    for j in range(n_states):
        cj = c[:, j]           # complex eigenvector for state j
        for mu in range(n_dim):
            for nu in range(n_dim):
                if mu == nu:
                    w_3d[j, mu, nu] = np.abs(cj[mu])**4
                else:
                    w_3d[j, mu, nu] = 4 * np.abs(cj[mu])**2 * np.abs(cj[nu])**2

    w = w_3d.reshape(n_states, n_elem)

    sigma = sig_matrix.reshape(-1)

    lam   = solve_lambdas(no_states, w, sigma, eps, verbose=True)
    # Continuous: eps^2 * sum(M_e) is the reported cost, and rounding M_e
    # before summing zeroes every element below half a shot.  See
    # optimal_allocation's docstring.
    M     = optimal_allocation(lam, w, sigma)
    M_2d = M.reshape(n_dim, n_dim)
    M_tot = np.triu(M_2d).sum()

    print("eps:", eps)
    print("c norm:", np.linalg.norm(c))
    print("sigma norm:", np.linalg.norm(sig_matrix))
    print("w min/max:", w.min(), w.max())
    print("lambda:", lam)
    top_n = min(50, len(M))
    top_idx = np.argsort(M)[::-1][:top_n]

    print("\nTop {} matrix elements by M:".format(top_n))
    print(f"{'Rank':>4} {'(μ,ν)':>10} {'M':>14} {'sigma':>14}", end="")
    for j in range(no_states):
        print(f" {'w[{j}]':>14}", end="")
    print()

    for rank, idx in enumerate(top_idx, 1):
        mu, nu = np.unravel_index(idx, (n_dim, n_dim))
        print(f"{rank:4d} ({mu:2d},{nu:2d}) "
              f"{M[idx]:14.6e} "
              f"{sigma[idx]:14.6e}", end="")
        for j in range(no_states):
            print(f" {w[j, idx]:14.6e}", end="")
        print()
    print("M sum:", M.sum())

    cost = eps ** 2 * M_tot

    # Zero cost has TWO causes and only one of them is a defect.
    #
    # LEGITIMATE: sigma is identically zero because every matrix element is
    # fully classically evaluable.  The benchmark returns sig = 0 by design when
    # a state carries no generators (NQ == 0), and a subspace of bare CSFs needs
    # no quantum measurement at all -- zero shots is the correct answer, not a
    # failure.  Measured: H2O A1 singlet, 1.5 A, n=5 has 0 generators across all
    # 31 states (n=3 has 6, n=4 has 3), so sigma norm is exactly 0.0.
    #
    # DEFECT: sigma is nonzero but the cost still came out zero or non-finite --
    # a collapsed allocation, an unconverged active set, or integer rounding.
    # That must never reach a figure.
    if not np.isfinite(cost) or cost <= 0.0:
        if np.all(sigma <= 0.0):
            print('  NOTE: sigma is identically zero -- every matrix element is '
                  'classically evaluable, so this subspace requires no quantum '
                  'measurement.  Sampling cost is legitimately 0.')
            return 0.0
        raise ValueError(
            f'weighted_sampling_cost: non-physical cost {cost!r} with NONZERO '
            f'sigma (norm={np.linalg.norm(sigma):.6e}, M_tot={M_tot!r}, '
            f'lambda={lam!r}).  A subspace with variance cannot be measured '
            f'with zero shots -- do not report this value.')
    return cost