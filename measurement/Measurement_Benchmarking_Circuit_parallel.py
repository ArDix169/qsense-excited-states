### benchmark circuits and measurement cost - VO  (parallel)
#
# Parallel version of Measurement_Benchmarking_Circuit.py, for large basis sets
# (~600 states).  The three hot loops are independent per element and are run
# over a process pool:
#
#   phase 0  overlap matrix        N(N+1)/2 elements   (Hermitian: half skipped)
#   phase 1  Hsub/sig diagonal     N elements
#   phase 2  Hsub/sig off-diagonal N(N-1)/2 elements   <-- dominates at N=600
#
# Workers inherit the read-only setup (Hqub, factorized states, CSFs, ...) by
# fork, so nothing large is pickled per task; only small tuples come back.
# Phase 2's pool is created AFTER phase 1 fills Hsub, so workers inherit the
# finished diagonal.
#
# Usage:
#   python Measurement_Benchmarking_Circuit_parallel.py <dump> \
#       [--molecule h2o2] [--bond-length 2.25] [--no-states 2] [--nparal 128]
# nparal defaults to $SLURM_CPUS_PER_TASK, else os.cpu_count().
# See --help for the rest.
#
# NOTE: set OMP_NUM_THREADS=1 (and MKL/OPENBLAS) in the job script, otherwise
# each worker spawns its own BLAS threads and oversubscribes the node.

import os
import sys
import time
import json
import pickle
import argparse
import multiprocessing as mp

import numpy as np

from src.measurement_new.utils_basic import (
    copy_hamiltonian
)
from src.measurement_new.utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from src.measurement_new.utils_states import (
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    tz_state_seniority_config,
    compress_state,
    decompress_state,
    create_composite_state
)
from src.measurement_new.utils_m1_seniority import (
    project_out_seniority_symmetries,
    seniority_rotate_hamiltonian,
    project_out_seniority_symmetries_prerotated
)
from src.measurement_new.utils_m2_factorize import (
    expand_tensor_product,
    expand_tensor_product_for_incomplete_qubit_set,
    get_indices_mapping_2_wvn_vo,
    factorize_state,
    evaluate_fully_classical_factors,
    obtain_coarse_dicts,
    QC_assignment_from_qubit_labels
)
from src.measurement_new.utils_m3_swap import (
    XorY_augment
)
from src.measurement_new.utils_m4_partitioning import (
    sorted_insertion_decomposition
)
from src.measurement_new.utils_results import (
    variance_of_decomp,
    sampling_cost, weighted_sampling_cost
)
from openfermion import (
    get_sparse_operator,
    jordan_wigner,
    count_qubits
)

from src.circuits.circuits_csf import get_csfs_from_dump, get_Uext_csfs_from_dump
from src.circuits.circuits_swap import get_parallelswap_subcircuit
from src.circuits.utils_circuit import get_decomp_circuits_estimators, show_state, verify_circuit_state
from qiskit import QuantumCircuit, transpile

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

_DEFAULT_DUMP = ('/Users/arjundixit/PycharmProjects/Q-SENSE/QSENSE_ES_dump/'
                 'h2o_sto3g_6o8e_UCSF_2_A1_1.5_for_Arjun_2.5.dump')

_p = argparse.ArgumentParser(
    description='Parallel VO circuit/measurement benchmark over Q-SENSE basis states.')
_p.add_argument('dump', nargs='?', default=_DEFAULT_DUMP,
                help='path to the Q-SENSE .dump of basis states')
_p.add_argument('--molecule', default='h2o', help='label used in the results banner')
_p.add_argument('--bond-length', default='2.5', help='label used in the results banner')
_p.add_argument('--no-states', type=int, default=2,
                help='number of lowest eigenpairs to report / weight the sampling cost')
_p.add_argument('--nparal', type=int, default=None,
                help='pool size (default: $SLURM_CPUS_PER_TASK, else os.cpu_count())')
_p.add_argument('--check-brute-force', action='store_true',
                help='cross-check Hsub[i,i] against <i|H|i> using the full 2^Nqubits '
                     'operator (infeasible beyond ~16 qubits)')
_p.add_argument('--verbose-states', action='store_true',
                help='dump per-state circuit debug info (very large at N=600)')
_p.add_argument('--output-json', default=None,
                help='where to write the results JSON '
                     '(default: <dump base>_VO_benchmark.json next to the dump)')
_args = _p.parse_args()

NPARAL = _args.nparal or int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or os.cpu_count()

# brute-force <i|H|i> cross-check builds the full 2^Nqubits sparse Hamiltonian.
# Fine for small systems, hopeless at 24 qubits -- off unless asked for.
CHECK_BRUTE_FORCE = _args.check_brute_force or os.environ.get('CHECK_BRUTE_FORCE', '0') == '1'

# per-state circuit debug dumps (huge at N=600) -- off unless asked for.
VERBOSE_STATES = _args.verbose_states or os.environ.get('VERBOSE_STATES', '0') == '1'

tol = 1e-5

# ----------------------------------------------------------------------------
# load Q-SENSE basis states  (serial setup; workers inherit all of this)
# ----------------------------------------------------------------------------

molecule = _args.molecule
bond_length = _args.bond_length
no_states = _args.no_states
filename = _args.dump

_T0 = time.time()


def _step(msg):
    """Timestamped progress line for the (single-threaded) setup phase."""
    print(f'[setup {time.time() - _T0:8.1f}s] {msg}', flush=True)


_step(f'loading dump: {filename}')
with open(filename, 'rb') as f:
    (
        CSF_tz_states,
        W_amplitudes,
        list_list_theta_CSF,
        list_sym_CSF_vec,
        list_UCSF_tz,
        UCSF_tz_states,
        somos_list, list_list_extra_vecs,
        psi_GS_UCSF_smik,
        list_orb_rot,
        x_orbrot,
        Enuc,
        obt_spatial,
        tbt_spatial
    ) = pickle.load(f)
_step(f'dump loaded: {len(CSF_tz_states)} CSF groups')

CSFs = get_csfs_from_dump(filename, verify_states=False, verbose=False)
_step(f'get_csfs_from_dump: {len(CSFs)} CSFs')

# rotate orbitals and obtain Hamiltonian operator

if len(list_orb_rot) != 0:
    obt, tbt = orthogonal_transform_obt_tbt(x_orbrot, list_orb_rot, obt_spatial, tbt_spatial)
else:
    obt = obt_phys_spatial_to_spin(obt_spatial)
    tbt = tbt_phys_spatial_to_spin(tbt_spatial)
_step('orbital rotation done')

Hfer = make_short_H_ferm_op(Enuc, obt, tbt)
_step('fermionic H built')
Hqub = jordan_wigner(Hfer)
_step(f'jordan_wigner done: {len(Hqub.terms)} qubit terms')

Nqubits = obt.shape[0]
Norb = Nqubits // 2
dim = 2 ** Nqubits
_step(f'Nqubits = {Nqubits}, dim = 2^{Nqubits} = {dim}')

# process information so that we can taper and factorize the Q-SENSE states

# build parallel lists like the Uext code does
UCSF_tz_states_full = []
CSF_tz_states_full = []
W_amplitudes_full = []

ucsf_idx = 0
for i in range(len(CSF_tz_states)):
    n_states_in_group = 1 + len(list_list_extra_vecs[i])  # ref + extra
    for j in range(n_states_in_group):
        UCSF_tz_states_full.append(UCSF_tz_states[ucsf_idx])
        CSF_tz_states_full.append(CSF_tz_states[i])
        W_amplitudes_full.append(W_amplitudes[i])
        ucsf_idx += 1

# now use the full lists
Nstates = len(UCSF_tz_states_full)
_step(f'Nstates = {Nstates}')
configs = [tz_state_seniority_config(tz_state) for tz_state in UCSF_tz_states_full]
UCSF_information = [get_indices_mapping_2_wvn_vo(CSF_tz_states_full[i], W_amplitudes_full[i], Norb)
                    for i in range(Nstates)]
_step('configs + UCSF_information done')

SW_list = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'W']) for i in range(Nstates)]
SV_list = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'V']) for i in range(Nstates)]
SN_list = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'N']) for i in range(Nstates)]
state_type_list = [UCSF_information[i][1] for i in range(Nstates)]

_step(f'building {Nstates} statevectors at dim {dim} ...')
statevectors = []
for _i, _tz in enumerate(UCSF_tz_states_full):
    statevectors.append(convert_TZ_format_to_sparse_format(dim, _tz))
    if (_i + 1) % 50 == 0 or _i + 1 == Nstates:
        _step(f'  statevectors {_i + 1}/{Nstates}')

# NOTE: psi.toarray() materializes a dense length-`dim` vector per state
# (dim = 2^Nqubits), so this loop is the memory/time hot spot of the setup.
def _compress_state_sparse(psi_csr, nq):
    """Vectorized, sparse-input equivalent of utils_states.compress_state.

    compress_state() densifies to 2**nq and walks every element in Python,
    formatting a binary string per element -- ~1.7e7 iterations per state here.
    The states are CSR with a handful of nonzeros, so instead we map only the
    nonzero indices.

    The tapering keeps bin(i)[1::2] of the MSB-first 2N-bit string, which in
    integer terms is "gather the even-numbered bits": result bit k = source
    bit 2k.  Verified against compress_state for random indices.
    """
    n_out = nq // 2
    psi_t = np.zeros(2 ** n_out)          # float64, matching compress_state

    idx = psi_csr.indices.astype(np.int64)
    vals = psi_csr.data
    keep = np.abs(vals) > 1e-12           # same cutoff compress_state applies
    idx, vals = idx[keep], vals[keep]

    out_idx = np.zeros_like(idx)
    for k in range(n_out):
        out_idx |= ((idx >> (2 * k)) & 1) << k

    psi_t[out_idx] = np.real(vals)
    return psi_t


_step('compressing (tapering) statevectors ...')
tapered_statevectors = []
for _i, _psi in enumerate(statevectors):
    tapered_statevectors.append(convert_dense_format_to_sparse_format(_compress_state_sparse(_psi, Nqubits)))
    if (_i + 1) % 100 == 0 or _i + 1 == Nstates:
        _step(f'  tapered {_i + 1}/{Nstates}')

def _describe_state(i):
    """Diagnostics for a state that failed to factorize.

    NOTE: somos_list is indexed by original CSF *group*, while i indexes the
    expanded per-state list, so it is only safe to index when the two happen to
    line up -- guard it rather than raising IndexError inside the handler.
    """
    lines = [f'  i = {i}',
             f'    state_type : {state_type_list[i]}',
             f'    SW={SW_list[i]}  SV={SV_list[i]}  SN={SN_list[i]}',
             f'    config     : {configs[i]}',
             f'    nnz(tapered): {tapered_statevectors[i].nnz}',
             f'    W_amplitudes (generators): {W_amplitudes_full[i]}']
    if i < len(somos_list):
        lines.append(f'    somo       : {somos_list[i]}')
    else:
        lines.append(f'    somo       : <unavailable: len(somos_list)={len(somos_list)} <= i>')
    return '\n'.join(lines)


_step('factorizing tapered statevectors ...')
factorized_tapered_statevectors = [None] * Nstates
_factorize_failures = []
for i in range(Nstates):
    try:
        factorized_tapered_statevectors[i] = \
            factorize_state(tapered_statevectors[i], SW_list[i], SV_list[i],
                            SN_list[i], state_type_list[i])
    except AssertionError:
        _factorize_failures.append(i)

if _factorize_failures:
    print(f'\nFACTORIZE FAILED for {len(_factorize_failures)} of {Nstates} states:',
          flush=True)
    print(f'  indices: {_factorize_failures}', flush=True)
    for i in _factorize_failures:
        print(_describe_state(i), flush=True)
    # A state whose W/V blocks are both empty is being treated as a full product
    # state over single orbitals; that only reconstructs if it is genuinely
    # unentangled, which an open-shell CSF is not.
    _all_n = [i for i in _factorize_failures
              if len(SW_list[i]) == 0 and len(SV_list[i]) == 0]
    if _all_n:
        print(f'\n  {len(_all_n)} of these have empty SW and SV (all orbitals in the '
              f'N block), i.e. labelled fully separable: {_all_n}', flush=True)
    raise SystemExit('factorization failed; see diagnostics above')

_step('factorization done')

# ---------------------------------------------------------------------------
# Hoist the Clifford rotation out of the per-task work.
#
# project_out_seniority_symmetries() = rotate(H, Nqubits) then taper(., v, w).
# The rotation does not depend on the seniority configs, so all 2e5 tasks were
# recomputing the same thing -- and it is the expensive half (openfermion
# deepcopies the term dict on every QubitOperator multiplication).  Compute it
# once here; workers inherit it through fork.
#
# The off-diagonal tasks taper (Hqub - ij_shift) with a scalar ij_shift.  The
# rotation is linear and fixes the identity, so rotate(H - c) == rotate(H) - c
# and the shift can be applied to the pre-rotated operator instead.
# ---------------------------------------------------------------------------
_step('rotating Hamiltonian (once, hoisted out of the task loops) ...')
H_rot = seniority_rotate_hamiltonian(Hqub, Nqubits)
_step(f'rotation done: {len(H_rot.terms)} terms')

# Verify the hoist reproduces the original routine before relying on it.
_v, _w = configs[0], configs[1 % Nstates]
for _cfg_a, _cfg_b in ((configs[0], configs[0]), (_v, _w)):
    _ref = project_out_seniority_symmetries(Hqub, Nqubits, _cfg_a, _cfg_b)
    _new = project_out_seniority_symmetries_prerotated(H_rot, _cfg_a, _cfg_b)
    _diff = _ref - _new
    _diff.compress()
    assert not _diff.terms, f'hoisted rotation mismatch: {len(_diff.terms)} residual terms'
# and once with a scalar shift, exercising rotate(H - c) == rotate(H) - c
_shift = 0.37
_ref = project_out_seniority_symmetries(Hqub - _shift, Nqubits, _v, _w)
_new = project_out_seniority_symmetries_prerotated(H_rot - _shift, _v, _w)
_diff = _ref - _new
_diff.compress()
assert not _diff.terms, f'hoisted rotation mismatch under shift: {len(_diff.terms)} terms'
_step('hoisted rotation verified against project_out_seniority_symmetries')

print(f'Nstates = {Nstates}, Nqubits = {Nqubits}, nparal = {NPARAL}', flush=True)
print(f'  diagonal tasks     : {Nstates}', flush=True)
print(f'  off-diagonal tasks : {Nstates * (Nstates - 1) // 2}', flush=True)

# results (filled by the phases below; workers inherit these read-only)
Sovlp = np.zeros([Nstates, Nstates], dtype=np.complex128)
Hsub = np.zeros([Nstates, Nstates], dtype=np.complex128)
sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def get_quantum_indices(bra_f, ket_f, bra_labels, ket_labels):
    quantum_indices = []

    join_partition, coarse_dict_bra, coarse_dict_ket = obtain_coarse_dicts(bra_f, ket_f)
    QC_assignment_dict = QC_assignment_from_qubit_labels(bra_labels, ket_labels, join_partition)

    for k, v in QC_assignment_dict.items():
        if v == 'Q':
            for idx in k:
                quantum_indices.append(idx)
    return quantum_indices


def _chunksize(ntasks, nworkers):
    """Enough tasks per dispatch to amortize IPC, small enough to stay balanced."""
    return max(1, min(256, ntasks // (nworkers * 8) or 1))


def _run_pool(func, tasks, nworkers, label, report_every=60.0):
    """imap_unordered with progress reporting. Returns list of results.

    Progress is reported on a TIME interval, not every N tasks: the phases here
    differ by three orders of magnitude in task count (638 vs ~2e5), so a
    percentage-based interval either spams the log or goes silent for an hour.
    """
    ntasks = len(tasks)
    if ntasks == 0:
        return []
    t0 = time.time()
    t_last = t0
    out = []
    # fork: workers inherit current module globals, nothing big is pickled
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=nworkers) as pool:
        for n, res in enumerate(pool.imap_unordered(func, tasks,
                                                    chunksize=_chunksize(ntasks, nworkers)), 1):
            out.append(res)
            now = time.time()
            if now - t_last >= report_every or n == ntasks:
                t_last = now
                el = now - t0
                rate = n / el if el > 0 else 0.0
                eta = (ntasks - n) / rate if rate > 0 else 0.0
                print(f'  [{label}] {n}/{ntasks}  {el:7.1f}s elapsed  '
                      f'{rate:8.1f} task/s  ETA {eta/60:7.1f} min', flush=True)
    return out


# ----------------------------------------------------------------------------
# phase 0: overlap matrix (Hermitian -> compute i >= j only)
# ----------------------------------------------------------------------------

def _ovlp_task(ij):
    i, j = ij
    return (i, j, (statevectors[i] @ statevectors[j].conj().T)[0, 0])


ovlp_pairs = [(i, j) for i in range(Nstates) for j in range(i + 1)]
for i, j, v in _run_pool(_ovlp_task, ovlp_pairs, NPARAL, 'overlap'):
    Sovlp[i, j] = v
    Sovlp[j, i] = np.conj(v)

print("Overlap matrix diagonal:", np.diag(Sovlp).real, flush=True)


# ----------------------------------------------------------------------------
# phase 1: diagonal elements
# ----------------------------------------------------------------------------

def _diag_task(i):
    ket_f = factorized_tapered_statevectors[i]
    ket_labels = UCSF_information[i][0]
    ket_config = configs[i]

    Htapered = project_out_seniority_symmetries_prerotated(H_rot, ket_config, ket_config)
    HQ, ketQ, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)
    quantum_indices = get_quantum_indices(ket_f, ket_f, ket_labels, ket_labels)

    if NQ == 0 or np.sum(np.abs(list(HQ.terms.values()))) <= tol:
        return (i, HQ.constant, 0.0, 0, 0)

    HQsparse = get_sparse_operator(HQ)
    ketQ = convert_dense_format_to_sparse_format(ketQ)
    h_ii = (ketQ @ HQsparse @ ketQ.T)[0, 0]

    HQ -= HQ.constant
    HQ.compress()
    decomp = sorted_insertion_decomposition(HQ, 'fc')
    var_metric = variance_of_decomp(decomp, ketQ, NQ, general=True)
    sig_ii = np.sqrt(var_metric)

    ### build circuits!
    decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ, NQ, methodtag='fc')
    csf_circ = CSFs[i].get_tapered_full_circuit(quantum_indices)

    if VERBOSE_STATES:
        print(f"CSF index i={i}")
        print(f"CSF t_vec: {CSFs[i].t_vec}")
        print(f"CSF orbitals: {CSFs[i].orbitals}")
        print(f"quantum_indices: {quantum_indices}")
        print(f"circuit state: {show_state(csf_circ)}")
        print(f"ketQ: {ketQ}")

    assert verify_circuit_state(csf_circ, ketQ)
    csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)

    return (i, h_ii, sig_ii, csf_circ_t.num_nonlocal_gates(), csf_circ_t.depth())


circuit_cx_counts = []
circuit_depth = []

for i, h_ii, sig_ii, cx, dep in _run_pool(_diag_task, list(range(Nstates)), NPARAL, 'diag'):
    Hsub[i, i] = h_ii
    sig_matrix[i, i] = sig_ii
    circuit_cx_counts.append(cx)
    circuit_depth.append(dep)


# ----------------------------------------------------------------------------
# phase 2: off-diagonal elements
# Hsub diagonal is now filled, so workers forked below inherit it.
# ----------------------------------------------------------------------------

def _offdiag_task(ij):
    i, j = ij

    ij_shift = 0.5 * (Hsub[i, i] + Hsub[j, j])

    bra_f = factorized_tapered_statevectors[i]
    bra_labels = UCSF_information[i][0]
    bra_config = configs[i]

    ket_f = factorized_tapered_statevectors[j]
    ket_labels = UCSF_information[j][0]
    ket_config = configs[j]

    Htapered = project_out_seniority_symmetries_prerotated(H_rot - ij_shift, bra_config, ket_config)
    HQ, braQ, ketQ, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)
    quantum_qubits = get_quantum_indices(bra_f, ket_f, bra_labels, ket_labels)

    if NQ == 0 or np.sum(np.abs(list(HQ.terms.values()))) <= tol:
        return (i, j, HQ.constant, 0.0, 0, 0)

    HQsparse = get_sparse_operator(HQ, NQ)
    braQ = convert_dense_format_to_sparse_format(braQ)
    ketQ = convert_dense_format_to_sparse_format(ketQ)
    h_ij = (braQ @ HQsparse @ ketQ.T)[0, 0]

    comp = create_composite_state(braQ, ketQ, NQ)
    HQ_aug = XorY_augment(HQ, NQ)
    decomp = sorted_insertion_decomposition(HQ_aug, 'fc')
    var_metric = variance_of_decomp(decomp, comp, NQ + 1, general=True)
    sig_ij = np.sqrt(var_metric)

    ### build circuits!
    decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ_aug, NQ + 1, methodtag='fc')
    csf_circ = get_parallelswap_subcircuit(CSFs[i], CSFs[j], quantum_qubits=quantum_qubits,
                                           control_qubit_pos=NQ)
    # assert verify_circuit_state(csf_circ, comp, truncate_bitstrings=list(range(NQ+1)))
    csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)

    return (i, j, h_ij, sig_ij, csf_circ_t.num_nonlocal_gates(), csf_circ_t.depth())


offdiag_pairs = [(i, j) for i in range(Nstates) for j in range(i)]

for i, j, h_ij, sig_ij, cx, dep in _run_pool(_offdiag_task, offdiag_pairs, NPARAL, 'offdiag'):
    Hsub[i, j] = h_ij
    Hsub[j, i] = h_ij
    sig_matrix[i, j] = sig_ij
    sig_matrix[j, i] = sig_ij
    # off-diagonal used twice
    circuit_cx_counts.append(cx)
    circuit_depth.append(dep)
    circuit_cx_counts.append(cx)
    circuit_depth.append(dep)


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------

if CHECK_BRUTE_FORCE:
    # builds the full 2^Nqubits x 2^Nqubits operator -- only viable for small Nqubits
    Hqub_sparse_full = get_sparse_operator(Hqub, Nqubits)
    print("\n[tapering check]  i :   Hsub[i,i] (tapered)      <i|H|i> (brute)          diff")
    for i in range(Nstates):
        psi_i = statevectors[i]
        e_brute = (psi_i @ Hqub_sparse_full @ psi_i.conj().T)[0, 0]
        print(f"  {i:2d} : {Hsub[i, i].real:+.10f}    {e_brute.real:+.10f}    {abs(Hsub[i, i] - e_brute):.2e}")

vals, vecs = np.linalg.eigh(Hsub)
vals = vals[:no_states]
c = vecs[:, :no_states]

cost = weighted_sampling_cost(no_states, c, sig_matrix, 1.6e-3)

generator_counts = [len(W_amplitudes_full[i]) for i in range(Nstates) if len(W_amplitudes_full[i]) > 0]

total_generators = sum(generator_counts)
avg_generators = total_generators / Nstates
max_generators = max(generator_counts) if len(generator_counts) > 0 else 0

print(f'''
    Final Results:
        Method              : {'VO'}
        Molecule            : {molecule}
        Bond Length         : {bond_length}
        Energies            : {vals}
        Sampling Cost       : {cost:.12e}

        Basis States        : {Nstates}
        Avg Generators      : {avg_generators:.2f}
        Max Generators      : {max_generators}
        Total Generators    : {total_generators}
''')

print('CX counts:\nMean: {}, Max: {}, Total: {}'.format(np.mean(circuit_cx_counts), np.max(circuit_cx_counts),
                                                        np.sum(circuit_cx_counts)))
print('Depth:\nMean: {}, Max: {}, Total: {}'.format(np.mean(circuit_depth), np.max(circuit_depth), np.sum(circuit_depth)))

# ----------------------------------------------------------------------------
# persist the same results to JSON so they are recoverable without parsing stdout
# ----------------------------------------------------------------------------

output_data = {
    'method': 'VO',
    'molecule': molecule,
    'bond_length': bond_length,
    'dump': filename,
    'no_states': no_states,
    'energies': [float(v) for v in vals],
    'sampling_cost': float(cost),
    'basis_states': int(Nstates),
    'n_qubits': int(Nqubits),
    'generators': {
        'avg': float(avg_generators),
        'max': int(max_generators),
        'total': int(total_generators),
    },
    'cx_counts': {
        'mean': float(np.mean(circuit_cx_counts)),
        'max': int(np.max(circuit_cx_counts)),
        'total': int(np.sum(circuit_cx_counts)),
    },
    'depth': {
        'mean': float(np.mean(circuit_depth)),
        'max': int(np.max(circuit_depth)),
        'total': int(np.sum(circuit_depth)),
    },
    'nparal': int(NPARAL),
}

if _args.output_json:
    json_save_path = _args.output_json
else:
    # default: alongside the dump, same base name, .json extension
    json_save_path = os.path.splitext(filename)[0] + '_VO_benchmark.json'

os.makedirs(os.path.dirname(os.path.abspath(json_save_path)), exist_ok=True)
with open(json_save_path, 'w') as jf:
    json.dump(output_data, jf, indent=2)
print(f'\nResults saved to: {json_save_path}')
