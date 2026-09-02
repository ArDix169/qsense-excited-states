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
# Full-space CAS(18e,12o) FCI reference generator for H2O2, plus the
# frozen-core CAS(14e,10o) sub-reference (ncore=2).
# =====================================================================

SPIN_LABEL = {0: 'singlet', 1: 'triplet', 2: 'quintet', 3: 'septet',
              4: 'nonet'}


def irrep_id_to_name(mol):
    return dict(zip(mol.irrep_id, mol.irrep_name))


def string_irrep_counts(orb_syms, nocc):
    if nocc < 0 or nocc > len(orb_syms):
        return {}

    dp = [{} for _ in range(nocc + 1)]
    dp[0][0] = 1

    for s in orb_syms:
        for k in range(min(nocc, len(dp) - 1), 0, -1):
            for sym, cnt in dp[k - 1].items():
                key = sym ^ s
                dp[k][key] = dp[k].get(key, 0) + cnt

    return dp[nocc]


def count_determinants_by_irrep(norb, nelec, orbsym, ms, ncore=2):
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


def count_csf_sectors(mol, norb, nelec, orbsym, ncore=2, max_s=1):
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
    n_half = ne_act // 2
    if n_half - S < 0 or n_half + S + 1 > nact + 1:
        return 0
    return ((2 * S + 1) * comb(nact + 1, n_half - S)
            * comb(nact + 1, n_half + S + 1)) // (nact + 1)


def active_space_dimension(norb, nelec, ncore=2):
    nact = norb - ncore
    ne_act = nelec - 2 * ncore
    na = nb = ne_act // 2
    return comb(nact, na) * comb(nact, nb)


rOH   = 0.9697
theta = radians(101.9)
tau   = radians(111.5)

basis_set = 'sto3g'
moltag    = 'h2o2'

OUTDIR = 'hamiltonians/ES_Hamiltonians'

HA2EV = 27.211386245988

NROOTS       = 5
NROOTS_SOLVE = 8
S2_TOL       = 1e-3

NSD_PRINT = 50
SD_TOL    = 1e-3


def run_fci(mol, h1e, eri, orbsym, norb, nelec, irrep, ss, label,
            E_nuc, nroots=NROOTS, nroots_solve=NROOTS_SOLVE, csf_dim=None):
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
        solver = fci.direct_spin0_symm.FCI(mol)
        nelec_sec = nelec
    else:
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

    if len(E) < nroots:
        log.append(f'  INSUFFICIENT: {len(E)}/{nroots} usable roots from '
                   f'{nroots_solve} solved.')
        # PySCF reports ONE converged flag for the whole Davidson batch and
        # np.repeat spreads it over every root, so a single hard root (usually
        # the last, most-degenerate one requested) discards the whole set even
        # when the others have residuals ~1e-9.  Asking for fewer roots lets the
        # batch converge cleanly.  This is what left H2O2 A/singlet EMPTY at
        # 3.0 A -- and an empty sector silently reads as "no deviation" in the
        # collectors, so it must be fixed here rather than worked around.
        if nroots_solve > nroots:
            retry = max(nroots, nroots_solve - 2)
            log.append(f'  RETRYING with nroots_solve = {retry} '
                       f'(batch converged flag discarded every root).')
            return run_fci(mol, h1e, eri, orbsym, norb, nelec, irrep, ss,
                           label + ' [retry]', E_nuc, nroots=nroots,
                           nroots_solve=retry, csf_dim=csf_dim)

    if len(E) >= 2 and (E[1] - E[0]) * 1000 < 2.0:
        log.append(f'  NOTE: roots 0/1 separated by only '
                   f'{(E[1] - E[0]) * 1000:.3f} mEh '
                   f'- root labels are not reliable.')

    return E, '\n'.join(log)


def compute_geometry(rOO):
    print(f"Worker started {rOO}", flush=True)

    log = ['\n' + '=' * 70,
           f'  Bond length rOO = {rOO} Angstrom',
           '=' * 70]

    O1 = np.array([-rOO / 2.0, 0.0, 0.0])
    O2 = np.array([ rOO / 2.0, 0.0, 0.0])

    half = (np.pi - tau) / 2.0

    H1 = np.array([
        -rOO / 2.0 + rOH * cos(theta),
         rOH * sin(theta) * cos(half),
        -rOH * sin(theta) * sin(half)
    ])

    H2 = np.array([
         rOO / 2.0 - rOH * cos(theta),
        -rOH * sin(theta) * cos(half),
        -rOH * sin(theta) * sin(half)
    ])

    mol = ps.gto.Mole(atom=[['O', O1], ['O', O2], ['H', H1], ['H', H2]],
                      basis=basis_set,
                      symmetry=True,
                      verbose=0)
    mol.build()

    mf = ps.scf.RHF(mol)
    mf.kernel()

    ncas    = mol.nao_nr()
    nelecas = mol.nelectron

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.fix_spin_(ss=0)
    mc.kernel()

    idx  = mc.mo_energy.argsort()
    mo   = mc.mo_coeff[:, idx]
    orbe = mc.mo_energy[idx]

    h1e_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    g_ao   = mol.intor('int2e')
    E_nuc  = mf.energy_nuc()

    h1e_cmo = mo.T @ h1e_ao @ mo
    g_cmo   = 0.5 * np.einsum('psqr,pa,qb,rc,sd->abcd', g_ao, mo, mo, mo, mo)

    os.makedirs(OUTDIR, exist_ok=True)
    fname = (f'{OUTDIR}/'
             f'{moltag}_{basis_set}_{ncas}o{nelecas}e_phys_spatial_{rOO}')
    with open(fname, 'wb') as f:
        pickle.dump([E_nuc, h1e_cmo, g_cmo, orbe,
                     mol.nelectron, ncas, nelecas], f)

    eri    = ao2mo.kernel(mol, mo)
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)

    NCORE = 2
    nact   = ncas - NCORE
    ne_act = mol.nelectron - 2 * NCORE

    csf_dims, det_dims = count_csf_sectors(
        mol, ncas, mol.nelectron, orbsym, ncore=NCORE, max_s=1)

    log.append('\nSector-resolved dimensions '
               '(CSFs, with Ms=S determinant block for reference):')
    for key in csf_dims:
        log.append(f'  {key[0]:>3s}/{key[1]:<8s} : '
                   f'{csf_dims[key]:>10d} CSF   '
                   f'({det_dims[key]:>10d} dets)')

    for s, lbl in ((0, 'singlet'), (1, 'triplet')):
        got  = sum(v for k, v in csf_dims.items() if k[1] == lbl)
        want = weyl_dimension(nact, ne_act, s)
        tag  = 'ok' if got == want else '*** MISMATCH ***'
        log.append(f'  Weyl-Paldus check {lbl:<8s}: '
                   f'sum over irreps = {got}, formula = {want}   {tag}')

    log.append(f'\nPoint group: {mol.groupname}   irreps: {mol.irrep_name}')
    log.append(f'norb = {ncas}   nelec = {mol.nelec}')

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
    }

    id2name = irrep_id_to_name(mol)
    for irrep in [id2name[i] for i in sorted(id2name)]:
        for ss, spin in ((0, 'singlet'), (2, 'triplet')):
            E, txt = run_fci(mol, h1e_cmo, eri, orbsym, ncas, mol.nelec,
                             irrep, ss, spin, E_nuc,
                             csf_dim=csf_dims.get((irrep, spin)))
            out[(irrep, spin)] = E
            log.append(txt)

    # ---- reference in Q-SENSE's OWN active space, CAS(14e,10o), ncore = 2 ----
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
                             irrep, ss, spin + ' [CAS(14e,10o)]', ecore_act,
                             csf_dim=csf_dims.get((irrep, spin)))
            out[(irrep, spin, 'cas')] = E
            log.append(txt)

    _found = [arr[0] for arr in out.values()
              if isinstance(arr, np.ndarray) and len(arr)]
    if not _found:
        log.append('  ERROR: no spin-clean roots in ANY sector')
        return rOO, out, '\n'.join(log)
    E0 = min(_found)

    log.append(f'\nCASSCF E_tot        = {mc.e_tot: .8f} Ha')
    log.append(f'FCI ground (all sym)= {E0: .8f} Ha')
    log.append(f'  difference        = {abs(E0 - mc.e_tot): .2e} Ha')

    log.append('\nExcitation energies relative to the overall ground state (eV):')
    for irrep in [id2name[i] for i in sorted(id2name)]:
        for spin in ('singlet', 'triplet'):
            for i, ei in enumerate(out[(irrep, spin)]):
                log.append(f'  {irrep}/{spin} {i}:  {(ei - E0) * HA2EV: 8.4f} eV')

    return rOO, out, '\n'.join(log)


if __name__ == '__main__':

    print("Entered main", flush=True)

    # the PES scan grid (1.25-3.0, matching QSENSE_paper_data/PES/H2O2)
    # PLUS 1.875, which only the production/scaling sets use.  One pickle
    # then serves every H2O2 check.
    bond_lengths = [1.25, 1.5, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0]

    n_workers = min(20, len(bond_lengths))

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
                rOO, out, log = fut.result()
                fci_all[rOO] = out
                print(log, flush=True)
            except Exception as exc:
                failed.append(r)
                print(f'\n  rOO = {r} FAILED: {exc!r}', flush=True)

    ref_name = f'{OUTDIR}/{moltag}_{basis_set}_fci_ref_full.pkl'
    with open(ref_name, 'wb') as f:
        pickle.dump(fci_all, f)
    print(f'\nFCI references -> {ref_name}')
    if failed:
        print(f'FAILED geometries: {failed}')

    print('\nDone.')
