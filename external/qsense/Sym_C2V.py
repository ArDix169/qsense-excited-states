import numpy as np
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf
from openfermion import s_squared_operator
from scipy.sparse.linalg import eigsh
from openfermion import count_qubits
from openfermion import jordan_wigner
from openfermion import get_sparse_operator, QubitOperator, FermionOperator, load_operator, get_fermion_operator
from openfermion import s_squared_operator, number_operator, sz_operator
from openfermion import is_hermitian
from openfermionpyscf import generate_molecular_hamiltonian
from openfermion.chem import MolecularData
from pyscf import gto, scf



#C2 symmetry

r = 1.2
basis = 'ccpvdz'
nact_e = 6
nact_o = 6

# Generate C2 Hamiltonian:
geometry = [('C', (0, 0, 0)), ('C', (0, 0, r))]
h_interac = generate_molecular_hamiltonian(geometry, basis = basis,
    multiplicity = 1, n_active_electrons=nact_e, n_active_orbitals=nact_o)

h_ferm = get_fermion_operator(h_interac)

n_so = count_qubits(h_ferm)
s_squared_ferm = s_squared_operator(n_so // 2)
number_ferm = number_operator(n_so)
sz_ferm = sz_operator(n_so // 2)

# --- D2h irreps multiplication table ---
D2h_mult = {
    ('Ag','Ag'):'Ag', ('Ag','B1g'):'B1g', ('Ag','B2g'):'B2g', ('Ag','B3g'):'B3g',
    ('Ag','Au'):'Au', ('Ag','B1u'):'B1u', ('Ag','B2u'):'B2u', ('Ag','B3u'):'B3u',

    ('B1g','Ag'):'B1g', ('B1g','B1g'):'Ag', ('B1g','B2g'):'B3g', ('B1g','B3g'):'B2g',
    ('B1g','Au'):'B1u', ('B1g','B1u'):'Au', ('B1g','B2u'):'B3u', ('B1g','B3u'):'B2u',

    ('B2g','Ag'):'B2g', ('B2g','B1g'):'B3g', ('B2g','B2g'):'Ag', ('B2g','B3g'):'B1g',
    ('B2g','Au'):'B2u', ('B2g','B1u'):'B3u', ('B2g','B2u'):'Au', ('B2g','B3u'):'B1u',

    ('B3g','Ag'):'B3g', ('B3g','B1g'):'B2g', ('B3g','B2g'):'B1g', ('B3g','B3g'):'Ag',
    ('B3g','Au'):'B3u', ('B3g','B1u'):'B2u', ('B3g','B2u'):'B1u', ('B3g','B3u'):'Au',

    ('Au','Ag'):'Au', ('Au','B1g'):'B1u', ('Au','B2g'):'B2u', ('Au','B3g'):'B3u',
    ('Au','Au'):'Ag', ('Au','B1u'):'B1g', ('Au','B2u'):'B2g', ('Au','B3u'):'B3g',

    ('B1u','Ag'):'B1u', ('B1u','B1g'):'Au', ('B1u','B2g'):'B3u', ('B1u','B3g'):'B2u',
    ('B1u','Au'):'B1g', ('B1u','B1u'):'Ag', ('B1u','B2u'):'B3g', ('B1u','B3u'):'B2g',

    ('B2u','Ag'):'B2u', ('B2u','B1g'):'B3u', ('B2u','B2g'):'Au', ('B2u','B3g'):'B1u',
    ('B2u','Au'):'B2g', ('B2u','B1u'):'B3g', ('B2u','B2u'):'Ag', ('B2u','B3u'):'B1g',

    ('B3u','Ag'):'B3u', ('B3u','B1g'):'B2u', ('B3u','B2g'):'B1u', ('B3u','B3g'):'Au',
    ('B3u','Au'):'B3g', ('B3u','B1u'):'B2g', ('B3u','B2u'):'B1g', ('B3u','B3u'):'Ag',
}


def classify_state_exact(psi, mo_irreps, mult_table):
    n_qubits = int(np.log2(len(psi)))
    n_spatial = len(mo_irreps)

    assert n_qubits == 2 * n_spatial, \
        f"Mismatch: n_qubits={n_qubits}, 2*n_spatial={2*n_spatial}"

    weights = {'A1': 0.0, 'A2': 0.0, 'B1': 0.0, 'B2': 0.0}

    for idx, amp in enumerate(psi):
        if abs(amp) < 1e-12:
            continue

        # OpenFermion JW: qubit 0 = MSB in binary representation
        # Do NOT reverse — keep big-endian so index 0 = qubit 0
        bin_str = format(idx, f'0{n_qubits}b')  # no [::-1]
        bitstring = np.array(list(bin_str), dtype=int)

        ir = determinant_irrep(bitstring, mo_irreps, mult_table)
        weights[ir] += abs(amp)**2

    total = sum(weights.values())
    if total > 1e-12:
        for k in weights:
            weights[k] /= total

    dominant_irrep = max(weights, key=lambda x: weights[x])
    return dominant_irrep, weights


def determinant_irrep(bitstring, mo_irreps, mult_table):
    n_spatial = len(mo_irreps)
    singly_occ = []

    # JW interleaved: spatial orbital p → qubits 2p (up) and 2p+1 (down)
    for p in range(n_spatial):
        occ_p = bitstring[2*p] + bitstring[2*p + 1]
        if occ_p == 1:
            singly_occ.append(p)

    if len(singly_occ) == 0:
        return 'A1'

    irrep = mo_irreps[singly_occ[0]]
    for p in singly_occ[1:]:
        irrep = mult_table[(irrep, mo_irreps[p])]

    return irrep

def sym():
    multiplicity = 1
    charge = 0
    nact_e = 6
    nact_o = 6

    molecule = MolecularData(
        geometry=geometry, basis=basis,
        multiplicity=multiplicity, charge=charge, description="C2"
    )
    molecule = run_pyscf(molecule, run_scf=True, run_fci=False)

    mol_sym = gto.M(
        atom=geometry, basis=basis,
        charge=charge, spin=multiplicity - 1, symmetry="D2h")

    mf_sym = scf.RHF(mol_sym).run()

    n_occ = mol_sym.nelectron // 2
    active_start = n_occ - (nact_e // 2)
    active_end = active_start + nact_o
    all_irreps = mf_sym.get_orbsym()
    active_irreps_int = all_irreps[active_start:active_end]

    ids   = mol_sym.irrep_id
    names = mol_sym.irrep_name
    id_to_label = {ids[k]: names[k] for k in range(len(ids))}
    mo_irreps = [id_to_label.get(i, f'Unknown_{i}') for i in active_irreps_int]

    if any('Unknown' in ir for ir in mo_irreps):
        print("WARNING: unresolved irreps:", mo_irreps)

    print("Active MO irreps:", mo_irreps)
    mo_sym = {}
    for i, ir in enumerate(mo_irreps):
        mo_sym[str(i)] = ir
    return mo_sym, D2h_mult, names


def build_h2o_geometry():
    bond_length = 1.5
    angle_rad = np.deg2rad(107.6)

    z = bond_length * np.cos(angle_rad / 2)
    x = bond_length * np.sin(angle_rad / 2)

    H1 = (x, 0.0, z)
    H2 = (-x, 0.0, z)

    H1 = tuple(float(f"{c:.12f}") for c in H1)
    H2 = tuple(float(f"{c:.12f}") for c in H2)

    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", H1),
        ("H", H2),
    ]

    return geometry

# --- C2v irreps multiplication table ---
C2V_MULT = {
    ('A1','A1'):'A1', ('A1','A2'):'A2', ('A1','B1'):'B1', ('A1','B2'):'B2',
    ('A2','A1'):'A2', ('A2','A2'):'A1', ('A2','B1'):'B2', ('A2','B2'):'B1',
    ('B1','A1'):'B1', ('B1','A2'):'B2', ('B1','B1'):'A1', ('B1','B2'):'A2',
    ('B2','A1'):'B2', ('B2','A2'):'B1', ('B2','B1'):'A2', ('B2','B2'):'A1',
}

def classify_state_exact(psi, mo_irreps, mult_table):
    n_qubits = int(np.log2(len(psi)))
    n_spatial = len(mo_irreps)

    assert n_qubits == 2 * n_spatial, \
        f"Mismatch: n_qubits={n_qubits}, 2*n_spatial={2*n_spatial}"

    weights = {'A1': 0.0, 'A2': 0.0, 'B1': 0.0, 'B2': 0.0}

    for idx, amp in enumerate(psi):
        if abs(amp) < 1e-12:
            continue

        # OpenFermion JW: qubit 0 = MSB in binary representation
        # Do NOT reverse — keep big-endian so index 0 = qubit 0
        bin_str = format(idx, f'0{n_qubits}b')  # no [::-1]
        bitstring = np.array(list(bin_str), dtype=int)

        ir = determinant_irrep(bitstring, mo_irreps, mult_table)
        weights[ir] += abs(amp)**2

    total = sum(weights.values())
    if total > 1e-12:
        for k in weights:
            weights[k] /= total

    dominant_irrep = max(weights, key=lambda x: weights[x])
    return dominant_irrep, weights


def determinant_irrep(bitstring, mo_irreps, mult_table):
    n_spatial = len(mo_irreps)
    singly_occ = []

    # JW interleaved: spatial orbital p → qubits 2p (up) and 2p+1 (down)
    for p in range(n_spatial):
        occ_p = bitstring[2*p] + bitstring[2*p + 1]
        if occ_p == 1:
            singly_occ.append(p)

    if len(singly_occ) == 0:
        return 'A1'

    irrep = mo_irreps[singly_occ[0]]
    for p in singly_occ[1:]:
        irrep = mult_table[(irrep, mo_irreps[p])]

    return irrep

def sym():
    # Define the molecule
    geometry = build_h2o_geometry()
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    description = "H2O"

    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        description=description,
    )

    molecule = run_pyscf(
        molecule,
        run_scf=True,
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=True
    )

    print("Electrons:", molecule.n_electrons)
    print("Orbitals:", molecule.n_orbitals)
    print("Qubits:", molecule.n_qubits)

    # ---- 2) Separate PySCF run with symmetry=True to get MO irreps ----
    # geometry here can be a list of (atom, (x,y,z)), which PySCF understands
    mol_sym = gto.M(
        atom=geometry,
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,  # 2S = multiplicity-1
        symmetry="C2v"
    )

    mf_sym = scf.RHF(mol_sym).run()

    mo_irreps = mf_sym.get_orbsym()

    print("Raw PySCF orb sym (ints):", mf_sym.get_orbsym())

    # They may be integers; convert if needed
    if isinstance(mo_irreps[0], (int, np.integer)):
        ids   = mol_sym.irrep_id      # e.g. [0,2,3]
        names = mol_sym.irrep_name    # e.g. ['A1','B1','B2']

        # Build a dict: integer_irrep → string_label
        id_to_label = {ids[k]: names[k] for k in range(len(ids))}

        # Convert MO irreps into labels
        mo_irreps = [id_to_label[i] for i in mo_irreps]

    print("\nMO irreps from PySCF:")
    mo_sym = {}
    for i, ir in enumerate(mo_irreps):
        mo_sym[str(i)] = ir
    c2v_mult = {('A1', 'A1'): 'A1', ('A1', 'A2'): 'A2', ('A1', 'B1'): 'B1', ('A1', 'B2'): 'B2',
                ('A2', 'A1'): 'A2', ('A2', 'A2'): 'A1', ('A2', 'B1'): 'B2', ('A2', 'B2'): 'B1',
                ('B1', 'A1'): 'B1', ('B1', 'A2'): 'B2', ('B1', 'B1'): 'A1', ('B1', 'B2'): 'A2',
                ('B2', 'A1'): 'B2', ('B2', 'A2'): 'B1', ('B2', 'B1'): 'A2', ('B2', 'B2'): 'A1', }
    names = ['A1', 'A2', 'B1', 'B2']
    return mo_sym, c2v_mult, names