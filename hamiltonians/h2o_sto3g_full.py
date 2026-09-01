print("Starting script...", flush=True)

import os

# Must precede numpy/pyscf import: one BLAS thread per worker process,
# otherwise N processes x N threads oversubscribes the machine.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
import pyscf as ps
from pyscf import symm, mcscf, fci, ao2mo, lib
from math import sin, cos, radians, comb
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

lib.num_threads(1)

# =====================================================================
# H2O port of h2o2_sto3g.py (the parallel full-space FCI driver).
#
# Everything structural is unchanged.  What differs, and why:
#
#   geometry     H2O has one internal coordinate scanned (rOH) and a fixed
#                HOH angle, versus H2O2's rOO / theta / tau.  Taken verbatim
#                from h2o_sto3g.py so the orbitals match the existing files.
#
#   full space   nao = 7, nelectron = 10  ->  7o10e, the analogue of H2O2's
#                12o18e.  h2o_sto3g.py already ran this CASSCF under the
#                comment "Full CI calculation" but never pickled it.
#
#   point group  C2v, not C2.  The dimension counting is group-agnostic: it
#                works off mol.irrep_id and XOR, which is the correct direct
#                product for every abelian group in PySCF's id convention
#                (B1 ^ B2 = 2 ^ 3 = 1 = A2).  STO-3G water spans only A1, B1
#                and B2, so A2 simply never appears in mol.irrep_name and the
#                sector loops skip it on their own.
#
#   NCORE = 1    H2O has ONE core orbital (O 1s), not two.  This must match
#                actmo_start in the qsense_subspace.py invocation or the "CAS"
#                reference below is not the reference Q-SENSE is converging
#                to.  With NCORE = 1 the sub-CAS is CAS(8e,6o), which is
#                exactly the existing h2o_sto3g_6o8e_phys_spatial_* space.
#
#   mo_sym       the pickle carries the CASSCF orbital irreps as a trailing
#                8th element, so qsense_subspace.py reads them instead of using a
#                hardcoded per-geometry table.  See h2o_sto3g_sweep.py.
#
# =====================================================================

# =====================================================================
# Spin / spatial symmetry sector dimensions
#
# The determinant count in an (irrep, Ms) sector is NOT the number of
# physically distinct states: an Ms=0 block contains the Sz=0 component
# of every S = 0, 1, 2, ... multiplet at once.  The number of spin
# eigenfunctions (CSFs) of total spin S in spatial irrep G follows from
# differencing consecutive Ms blocks,
#
#     N_CSF(S, G) = N_det(Ms = S, G) - N_det(Ms = S+1, G),
#
# since a spin-S multiplet contributes exactly one dimension to each Ms
# with |Ms| <= S.  This is exact and needs no GUGA machinery.
# =====================================================================

SPIN_LABEL = {0: 'singlet', 1: 'triplet', 2: 'quintet', 3: 'septet',
              4: 'nonet'}


def irrep_id_to_name(mol):
    """{irrep_id: irrep_name} for the FULL point group.

    NOT `dict(zip(mol.irrep_id, mol.irrep_name))` -- those list only the irreps
    SPANNED BY THE BASIS.  STO-3G water has no A2 orbital, so A2 is absent from
    mol.irrep_name, but A2 STATES exist in abundance: any determinant with a
    singly occupied B1 and a singly occupied B2 carries B1 x B2 = 2 ^ 3 = 1 =
    A2.  Using the molecule's list drops 20 singlet and 26 triplet A2 states at
    rOH = 1.0 and makes the Weyl-Paldus check fail 85 vs 105.

    H2O2 never exposed this: C2 has only {A: 0, B: 1} and XOR is closed on the
    two, so the spanned list and the full group coincide.
    """
    table = symm.param.IRREP_ID_TABLE.get(mol.groupname)
    if table is None:                      # non-abelian or unlisted: best effort
        return dict(zip(mol.irrep_id, mol.irrep_name))
    return {v: k for k, v in table.items()}


def string_irrep_counts(orb_syms, nocc):
    """Occupation strings of length `nocc` binned by spatial irrep id.

    `orb_syms` is the list of integer irrep ids of the active orbitals.
    For the abelian groups PySCF supports, the direct product of two
    irreps is the XOR of their ids, so a doubly occupied orbital always
    contributes the totally symmetric irrep (s ^ s = 0) and drops out.

    Dynamic programming over orbitals: O(norb * nocc * nirrep) instead of
    enumerating C(norb, nocc) strings explicitly.
    """
    if nocc < 0 or nocc > len(orb_syms):
        return {}

    dp = [{} for _ in range(nocc + 1)]
    dp[0][0] = 1

    for s in orb_syms:
        # descending k so dp[k-1] is still the pre-orbital table
        for k in range(min(nocc, len(dp) - 1), 0, -1):
            for sym, cnt in dp[k - 1].items():
                key = sym ^ s
                dp[k][key] = dp[k].get(key, 0) + cnt

    return dp[nocc]


def count_determinants_by_irrep(norb, nelec, orbsym, ms, ncore=1):
    """Determinants with Sz = ms, resolved by spatial irrep id.

    The lowest `ncore` orbitals are taken as doubly occupied.  This is a
    dimension count only -- the Hamiltonian is untouched.
    """
    active = [int(orbsym[i]) for i in range(ncore, norb)]
    ne_act = nelec - 2 * ncore

    na = ne_act // 2 + ms
    nb = ne_act // 2 - ms

    if nb < 0 or na > len(active):
        return {}

    a = string_irrep_counts(active, na)
    b = string_irrep_counts(active, nb)

    out = {}
    for sa, ca in a.items():
        for sb, cb in b.items():
            key = sa ^ sb
            out[key] = out.get(key, 0) + ca * cb
    return out


def count_csf_sectors(mol, norb, nelec, orbsym, ncore=1, max_s=1):
    """CSF dimensions keyed by (irrep_name, spin_label).

    Also returns the underlying Ms-block determinant counts so the two
    can be logged side by side.
    """
    id2name = irrep_id_to_name(mol)
    all_ids = sorted(id2name)

    det = {m: count_determinants_by_irrep(norb, nelec, orbsym, m, ncore)
           for m in range(0, max_s + 2)}

    csf_dims = {}
    det_dims = {}
    for s in range(0, max_s + 1):
        for ir in all_ids:
            key = (id2name[ir], SPIN_LABEL[s])
            csf_dims[key] = det[s].get(ir, 0) - det[s + 1].get(ir, 0)
            det_dims[key] = det[s].get(ir, 0)

    return csf_dims, det_dims


def weyl_dimension(nact, ne_act, S):
    """Weyl-Paldus formula: total CSFs of spin S, all spatial irreps."""
    n_half = ne_act // 2
    if n_half - S < 0 or n_half + S + 1 > nact + 1:
        return 0
    return ((2 * S + 1) * comb(nact + 1, n_half - S)
            * comb(nact + 1, n_half + S + 1)) // (nact + 1)


def active_space_dimension(norb, nelec, ncore=1):
    """Total Ms=0 determinant dimension with `ncore` frozen doubly occ."""
    nact = norb - ncore
    ne_act = nelec - 2 * ncore
    na = nb = ne_act // 2
    return comb(nact, na) * comb(nact, nb)


# =====================================
# Fixed internal geometry (only the O-H distance rOH is scanned)
# Both O-H bonds are stretched together, so the molecule stays C2v.
# =====================================
theta_h2o = radians(104.5 / 2)   # half the H-O-H angle

basis_set = 'sto3g'
moltag    = 'h2o'

OUTDIR = 'hamiltonians/ES_Hamiltonians'

HA2EV = 27.211386245988

NROOTS       = 5     # roots kept per (irrep, spin) sector
NROOTS_SOLVE = 8     # roots Davidson actually solves for
S2_TOL       = 1e-3

NSD_PRINT = 50      # determinants printed per eigenstate
SD_TOL    = 1e-3    # |coef| cutoff

# H2O has a single core orbital (O 1s).  Must equal actmo_start in the
# qsense_subspace.py invocation.
NCORE = 1

# Above this an MO is a genuine irrep mixture and no single label is correct.
PURITY_TOL = 1e-6


def stable_argsort(mo_energy, orbsym, tol=1.0e-3):
    """Energy ordering with ties broken by irrep id, so it is machine-independent.

    A plain `mo_energy.argsort()` is not reproducible here.  With ncas == nao
    the CASSCF energy is invariant under ANY orbital rotation, so `mo_energy`
    carries no physics -- it is whatever PySCF's canonicalization produced.  At
    rOH = 3.0 the MO 3/4 pair is separated by 2.2e-4 Ha, and a different PySCF
    build orders them the other way: locally B2 then A1, on Vesta A1 then B2.

    Nothing physical changes -- permuting active orbitals leaves every FCI root
    identical to 2.8e-14 Ha, verified -- but the two machines then disagree on
    what "orbital 3" means, which makes index-level comparison of dumps
    meaningless.  So: cluster orbitals whose adjacent energies differ by less
    than `tol`, and inside a cluster sort by irrep id.

    tol = 1 mHa is comfortably above the ambiguous gaps (2.2e-4 at 3.0 A) and
    far below the real ones (5e-2 at equilibrium), so it only ever reorders
    pairs that were arbitrary to begin with.
    """
    order = list(np.argsort(mo_energy, kind='stable'))
    e = np.asarray(mo_energy)[order]

    out, start = [], 0
    for k in range(1, len(order) + 1):
        if k == len(order) or (e[k] - e[k - 1]) >= tol:
            cluster = order[start:k]
            out.extend(sorted(cluster, key=lambda i: (int(orbsym[i]), i)))
            start = k
    return np.asarray(out)


def sd_expansion(c, norb, nelec, nmax=NSD_PRINT, tol=SD_TOL):
    """Top `nmax` Slater determinants of a CI vector, as printable lines.

    pyscf's large_ci returns (coeff, occ_a, occ_b) with the occupied spatial
    orbital indices per spin; the occupation string and SOMO/DOMO split follow
    from the intersection and symmetric difference of those two sets.
    """
    rows = fci.addons.large_ci(np.asarray(c), norb, nelec,
                               tol=tol, return_strs=False)
    rows = sorted(rows, key=lambda t: -abs(t[0]))[:nmax]

    lines = []
    for coef, oa, ob in rows:
        sa, sb = set(int(x) for x in oa), set(int(x) for x in ob)
        occ = ' '.join(('1' if p in sa else '0') + ('1' if p in sb else '0')
                       for p in range(norb))
        domo = sorted(sa & sb)
        somo = sorted(sa ^ sb)
        lines.append(f'   coef = {coef:+.6f}   |{occ}>   '
                     f'SOMOs: {somo if somo else ["(no-somo)"]} | '
                     f'DOMOs: {domo}')
    return lines


def irrep_weights(mol, mo, s_ao):
    """Fraction of each MO's norm lying in each irrep subspace.

    PySCF's `symm_orb` blocks are symmetry-adapted AO combinations, NOT
    orthonormal, so the projector needs the metric explicitly:

        w_b = (Xb^T S c)^T (Xb^T S Xb)^-1 (Xb^T S c) / (c^T S c)

    Skipping the (Xb^T S Xb)^-1 gives weights that do not sum to one and
    silently misassign near-degenerate orbitals.
    """
    out = np.zeros((mo.shape[1], len(mol.symm_orb)))
    for b, Xb in enumerate(mol.symm_orb):
        XtS = Xb.T @ s_ao
        M = XtS @ Xb
        proj = XtS @ mo
        out[:, b] = np.einsum('ij,ij->j', proj, np.linalg.solve(M, proj))
    norms = np.einsum('ij,ij->j', mo, s_ao @ mo)
    return out / norms[:, None]


def run_fci(mol, h1e, eri, orbsym, norb, nelec, irrep, ss, label,
            E_nuc, nroots=NROOTS, nroots_solve=NROOTS_SOLVE, csf_dim=None):
    """`nroots` lowest spin-clean FCI states of a given spatial irrep and spin.

    Each sector is solved in its own Ms = S determinant block, so no spin
    penalty is needed and no root is spent on a wrong-spin state.
    Energies are <c|H|c> + E_nuc.
    """
    log = [f'\nFCI  irrep {irrep}  {label}:']

    if csf_dim is not None:
        if csf_dim <= 0:
            log.append('  sector is empty (0 CSFs) - skipped.')
            return np.array([]), '\n'.join(log)
        if nroots_solve > csf_dim:
            log.append(f'  capping roots: sector holds only {csf_dim} CSF(s).')
            nroots_solve = csf_dim
        nroots = min(nroots, csf_dim)

    if ss == 0:
        # Ms=0 CI matrix obeys C^T = (-1)^S C, so the symmetric-only
        # solver admits S = 0, 2, 4... and excludes triplets exactly.
        solver = fci.direct_spin0_symm.FCI(mol)
        nelec_sec = nelec
    else:
        # Ms=1 block: contains every S >= 1 multiplet and no singlets.
        # Energies match the Ms=0 triplets since H is spin-free.
        ne = sum(nelec) if hasattr(nelec, '__len__') else nelec
        solver = fci.direct_spin1_symm.FCI(mol)
        nelec_sec = (ne // 2 + 1, ne // 2 - 1)

    solver.orbsym        = orbsym
    solver.wfnsym        = irrep
    solver.conv_tol      = 1e-10
    solver.lindep        = 1e-14
    solver.max_space     = max(40, 6 * nroots_solve)
    solver.max_cycle     = 2000
    solver.pspace_size   = 1600
    solver.davidson_only = True
    # no fix_spin_ - the Ms block already removed the contaminants

    log.append(f'  Ms = S block: nelec = {nelec_sec}')
    print(f"[{irrep} {label}] Starting Davidson ({nroots_solve} roots)",
          flush=True)

    e, civ = solver.kernel(h1e, eri, norb, nelec_sec, nroots=nroots_solve)

    civ = list(civ) if isinstance(civ, (list, tuple)) else [civ]
    e   = np.atleast_1d(e)

    conv = np.atleast_1d(solver.converged)
    if conv.size == 1 and len(e) > 1:
        conv = np.repeat(conv, len(e))

    print(f"[{irrep} {label}] Davidson finished ({len(e)} roots)", flush=True)

    # direct_spin1 handles any (na, nb) and makes no symmetry assumption
    clean = fci.direct_spin1.FCI(mol)
    h2e   = clean.absorb_h1e(h1e, eri, norb, nelec_sec, 0.5)

    keep = []
    for i, c in enumerate(civ):
        c   = np.asarray(c)
        nrm = float(np.dot(c.ravel(), c.ravel()))
        if nrm < 1e-10:
            continue

        hc       = clean.contract_2e(h2e, c, norb, nelec_sec)
        E_var    = float(np.dot(c.ravel(), hc.ravel()) / nrm) + E_nuc
        s2, mult = solver.spin_square(c, norb, nelec_sec)
        pen      = (e[i] + E_nuc) - E_var

        # No penalty operator now, so E_dav and E_var must agree exactly
        # at convergence; any gap is an unconverged root.
        ok_conv = bool(conv[i]) and abs(pen) < 1e-7
        ok_spin = abs(s2 - ss) < S2_TOL

        flag = ''
        if not ok_spin:
            flag = '   <-- SPIN CONTAMINATED (discarded)'
        elif not ok_conv:
            flag = '   <-- NOT CONVERGED (discarded)'

        log.append(f'  {i:2d}:  E_var = {E_var: .10f}   '
                   f'E_dav = {e[i] + E_nuc: .10f}   '
                   f'resid = {pen: .2e}   <S^2> = {s2: .6f}{flag}')

        if ok_conv and ok_spin:
            keep.append((E_var, c / np.sqrt(nrm)))

    keep.sort(key=lambda r: r[0])
    keep = keep[:nroots]

    E = np.array([r[0] for r in keep])

    #     # ---- SD expansion of the retained eigenstates ----------------------
    # for k, (E_k, c_k) in enumerate(keep):
    #     log.append(f'\n  Eigenstate {k}, Energy = {E_k:.8f} Ha')
    #     log.append(f'  {"-" * 50}')
    #     log.append(f'   Significant SDs (|coef| > {SD_TOL:g}), '
    #                f'top {NSD_PRINT}:')
    #     log.extend(sd_expansion(c_k, norb, nelec_sec))

    if len(E) < nroots:
        log.append(f'  INSUFFICIENT: {len(E)}/{nroots} usable roots from '
                   f'{nroots_solve} solved.')

    if len(E) >= 2 and (E[1] - E[0]) * 1000 < 2.0:
        log.append(f'  NOTE: roots 0/1 separated by only '
                   f'{(E[1] - E[0]) * 1000:.3f} mEh '
                   f'- root labels are not reliable.')

    return E, '\n'.join(log)


def compute_geometry(rOH):
    """One bond length: RHF -> CASSCF -> Hamiltonian pickle -> FCI by sector."""
    print(f"Worker started {rOH}", flush=True)

    log = ['\n' + '=' * 70,
           f'  Bond length rOH = {rOH} Angstrom',
           '=' * 70]

    # O at the origin, both H in the xz plane, C2 axis along z.  Already the
    # standard C2v orientation, so symmetry=True does not reorient anything.
    x = sin(theta_h2o) * rOH
    z = cos(theta_h2o) * rOH

    mol = ps.gto.Mole(atom=[['O', [0.0, 0.0, 0.0]],
                            ['H', [-x, 0.0, z]],
                            ['H', [ x, 0.0, z]]],
                      basis=basis_set,
                      symmetry=True,
                      verbose=0)
    mol.build()

    mf = ps.scf.RHF(mol)
    mf.kernel()

    # =====================================
    # CASSCF over the full orbital space -> CASCI == FCI
    #
    # With ncas == nao the orbital gradient vanishes identically -- the energy
    # is invariant under any orbital rotation -- so what actually fixes the
    # orbitals is PySCF's canonicalization at convergence.  Same as H2O2.
    # =====================================
    ncas    = mol.nao_nr()      # 7
    nelecas = mol.nelectron     # 10

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.fix_spin_(ss=0)
    mc.kernel()

    # Irreps of the UNSORTED orbitals -- stable_argsort needs them as the
    # tie-break key, so they have to be labelled before the reordering.
    orbsym_raw = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                     mc.mo_coeff)

    idx  = stable_argsort(mc.mo_energy, orbsym_raw)
    mo   = mc.mo_coeff[:, idx]
    orbe = mc.mo_energy[idx]

    # =====================================
    # Integrals and transformation to CASSCF MOs
    # =====================================
    h1e_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    s_ao   = mol.intor('int1e_ovlp')
    g_ao   = mol.intor('int2e')
    E_nuc  = mf.energy_nuc()

    h1e_cmo = mo.T @ h1e_ao @ mo
    g_cmo   = 0.5 * np.einsum('psqr,pa,qb,rc,sd->abcd', g_ao, mo, mo, mo, mo)

    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)

    # =====================================
    # MO irreps -> printed AND embedded in the pickle
    # =====================================
    w = irrep_weights(mol, mo, s_ao)
    best = w.argmax(axis=1)
    mo_sym = {i: mol.irrep_name[b] for i, b in enumerate(best)}
    contamination = (1.0 - w[np.arange(len(best)), best]).tolist()
    pure = bool(max(contamination) <= PURITY_TOL)

    # "energy-ordered", not "energy-sorted": inside a sub-mHa cluster the order
    # is by irrep id, so e is not strictly monotonic there.  See stable_argsort.
    log.append(f'\nCASSCF MO irreps (energy-ordered, the Hamiltonian basis):')
    for i in range(ncas):
        tag = 'core' if i < NCORE else ('occ' if i < nelecas // 2 else 'virt')
        log.append(f'  MO {i:2d}:  {mo_sym[i]:>4}   e = {orbe[i]: .6f} Ha   {tag}')
    log.append(f'  mo_sym = {mo_sym}')
    if not pure:
        bad = [i for i, c in enumerate(contamination) if c > PURITY_TOL]
        log.append(f'  *** NOT IRREP-PURE: max contamination '
                   f'{max(contamination):.2e} on MO(s) {bad}.  No single label '
                   f'is correct for these; qsense_subspace.py will refuse the file.')

    sym_rec = {'point_group': mol.groupname,
               'irrep_name': list(mol.irrep_name),
               'mo_sym': mo_sym,
               'orbsym_id': [int(v) for v in orbsym],
               'contamination': contamination,
               'irrep_weights': w.tolist(),
               'pure': pure,
               'rOH': rOH, 'ncas': ncas, 'nelecas': nelecas,
               'basis': basis_set}

    # =====================================
    # Save CAS Hamiltonian (the one QSENSE consumes)
    #
    # 8 elements, not 7: the irrep record is APPENDED so readers that slice
    # rec[:7] keep working on both old and new files.
    # =====================================
    os.makedirs(OUTDIR, exist_ok=True)
    fname = (f'{OUTDIR}/'
             f'{moltag}_{basis_set}_{ncas}o{nelecas}e_phys_spatial_{rOH}')
    with open(fname, 'wb') as f:
        pickle.dump([E_nuc, h1e_cmo, g_cmo, orbe,
                     mol.nelectron, ncas, nelecas, sym_rec], f)
    log.append(f'\nwrote {fname}  (8 elements)')

    # =====================================
    # FCI resolved by spatial irrep and spin
    # =====================================
    eri = ao2mo.kernel(mol, mo)

    # =====================================
    # Sector-resolved CSF dimensions
    # =====================================
    nact   = ncas - NCORE            # 6
    ne_act = mol.nelectron - 2 * NCORE   # 8

    csf_dims, det_dims = count_csf_sectors(
        mol, ncas, mol.nelectron, orbsym, ncore=NCORE, max_s=1)

    log.append('\nSector-resolved dimensions '
               '(CSFs, with Ms=S determinant block for reference):')
    for key in csf_dims:
        log.append(f'  {key[0]:>3s}/{key[1]:<8s} : '
                   f'{csf_dims[key]:>10d} CSF   '
                   f'({det_dims[key]:>10d} dets)')

    # Cross-check against the Weyl-Paldus dimension formula
    for s, lbl in ((0, 'singlet'), (1, 'triplet')):
        got  = sum(v for k, v in csf_dims.items() if k[1] == lbl)
        want = weyl_dimension(nact, ne_act, s)
        tag  = 'ok' if got == want else '*** MISMATCH ***'
        log.append(f'  Weyl-Paldus check {lbl:<8s}: '
                   f'sum over irreps = {got}, formula = {want}   {tag}')

    log.append(f'\nPoint group: {mol.groupname}   '
               f'irreps spanned by orbitals: {mol.irrep_name}   '
               f'full group: {sorted(irrep_id_to_name(mol).values())}')
    log.append(f'norb = {ncas}   nelec = {mol.nelec}')

    # =====================================
    # QSENSE-compatible total determinant dimension (Ms = 0)
    # Lowest NCORE orbitals constrained doubly occupied
    # (dimension only, Hamiltonian unchanged)
    # =====================================
    fci_dim = active_space_dimension(ncas, mol.nelectron, ncore=NCORE)

    log.append('\nQSENSE-compatible active-space FCI dimension:')
    log.append(f'  inactive orbitals = {NCORE}')
    log.append(f'  active orbitals   = {nact}')
    log.append(f'  active electrons  = {ne_act}')
    log.append(f'  determinant dimension (Ms=0) = {fci_dim}')

    out = {
        'fci_dimension':      fci_dim,
        'csf_dimensions':     csf_dims,
        'sector_dimensions':  det_dims,
        'ncore':              NCORE,
        'nactive_orbitals':   nact,
        'nactive_electrons':  ne_act,
        'mo_sym':             mo_sym,
        'point_group':        mol.groupname,
        'irrep_pure':         pure,
    }

    id2name = irrep_id_to_name(mol)
    for irrep in [id2name[i] for i in sorted(id2name)]:
        for ss, spin in ((0, 'singlet'), (2, 'triplet')):
            E, txt = run_fci(mol, h1e_cmo, eri, orbsym, ncas, mol.nelec,
                             irrep, ss, spin, E_nuc,
                             csf_dim=csf_dims.get((irrep, spin)))
            out[(irrep, spin)] = E
            log.append(txt)

    # ---- reference in Q-SENSE's OWN active space, CAS(8e,6o), ncore = 1 ----
    # The full-space FCI above is the exact answer for 7o10e, but Q-SENSE runs
    # with actmo_start=NCORE, so comparing against it folds the frozen-core
    # truncation into what looks like subspace error.  This is the reference
    # that isolates the subspace.
    cas_tag = f'CAS({ne_act}e,{nact}o)'
    cas = mcscf.CASCI(mf, nact, ne_act)
    cas.mo_coeff = mo
    h1_act, ecore_act = cas.get_h1eff()
    eri_act = ao2mo.restore(1, cas.get_h2eff(mo[:, NCORE:NCORE + nact]), nact)
    orbsym_act = np.asarray(orbsym)[NCORE:NCORE + nact]

    for irrep in [id2name[i] for i in sorted(id2name)]:
        for ss, spin in ((0, 'singlet'), (2, 'triplet')):
            S = 0 if ss == 0 else 1
            nelec_act = (ne_act // 2 + S, ne_act // 2 - S)
            E, txt = run_fci(mol, h1_act, eri_act, orbsym_act, nact, nelec_act,
                             irrep, ss, f'{spin} [{cas_tag}]', ecore_act,
                             csf_dim=csf_dims.get((irrep, spin)))
            out[(irrep, spin, 'cas')] = E
            log.append(txt)

    # Sanity check: overall FCI ground must match the CASSCF total energy
    _found = [arr[0] for arr in out.values()
              if isinstance(arr, np.ndarray) and len(arr)]
    if not _found:
        log.append('  ERROR: no spin-clean roots in ANY sector')
        return rOH, out, '\n'.join(log)
    E0 = min(_found)

    log.append(f'\nCASSCF E_tot        = {mc.e_tot: .8f} Ha')
    log.append(f'FCI ground (all sym)= {E0: .8f} Ha')
    log.append(f'  difference        = {abs(E0 - mc.e_tot): .2e} Ha')

    log.append('\nExcitation energies relative to the overall ground state (eV):')
    for irrep in [id2name[i] for i in sorted(id2name)]:
        for spin in ('singlet', 'triplet'):
            for i, ei in enumerate(out[(irrep, spin)]):
                log.append(f'  {irrep}/{spin} {i}:  {(ei - E0) * HA2EV: 8.4f} eV')

    return rOH, out, '\n'.join(log)


# =====================================
# Driver
# =====================================
if __name__ == '__main__':

    print("Entered main", flush=True)

    # Default 0.75 -> 3.0 A in 0.25 steps, matching the grid already on disk.
    # Distances may be given on the command line to add or redo single points,
    # e.g. `python3 h2o_sto3g_full.py 2.125`.  Safe to do since the FCI
    # reference is merged rather than overwritten (see the end of this block).
    import sys
    _args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if _args:
        bond_lengths = [float(a) for a in _args]
    else:
        bond_lengths = np.round(np.arange(0.75, 3.01, 0.25), 3)

    n_workers = min(20, len(bond_lengths))
    print(f'Running {len(bond_lengths)} geometries on {n_workers} workers.',
          flush=True)

    fci_all = {}
    failed  = []

    print("Creating process pool", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        print("Pool created", flush=True)

        futures = {}

        for r in bond_lengths:
            print(f"Submitting {r}", flush=True)
            futures[ex.submit(compute_geometry, float(r))] = float(r)

        print("Finished submitting", flush=True)

        for fut in as_completed(futures):
            r = futures[fut]
            try:
                rOH, out, log = fut.result()
                fci_all[rOH] = out
                print(log, flush=True)
            except Exception as exc:
                failed.append(r)
                print(f'\n  rOH = {r} FAILED: {exc!r}', flush=True)

    # ---- mo_sym summary across the grid --------------------------------
    print('\n' + '=' * 70)
    print('  mo_sym across the grid')
    print('=' * 70)
    for r in sorted(fci_all):
        s = fci_all[r]
        labels = [s['mo_sym'][i] for i in range(len(s['mo_sym']))]
        flag = '' if s['irrep_pure'] else '   <-- NOT IRREP-PURE'
        print(f'  {r:>5} A  {s["point_group"]:>4}  {labels}{flag}')

    # MERGE, do not overwrite.  fci_all holds only the geometries THIS run
    # covered, so a subset run (e.g. a single new bond length) would otherwise
    # wipe every other geometry out of the shared reference file.
    ref_name = f'{OUTDIR}/{moltag}_{basis_set}_fci_ref_full.pkl'
    merged = {}
    if os.path.exists(ref_name):
        with open(ref_name, 'rb') as f:
            merged = pickle.load(f)
        print(f'\nmerging into existing reference with '
              f'{len(merged)} geometries: {sorted(merged)}')
    merged.update(fci_all)
    with open(ref_name, 'wb') as f:
        pickle.dump(merged, f)
    print(f'FCI references -> {ref_name}  ({len(merged)} geometries)')
    if failed:
        print(f'FAILED geometries: {failed}')

    print('\nDone.')
