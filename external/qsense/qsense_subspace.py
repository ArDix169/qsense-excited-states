import sys, scipy, os
#sys.path.append('../')
from scipy.sparse import csr_matrix
import numpy as np
import pickle, copy
import json
from openfermion import (FermionOperator, hermitian_conjugated, normal_ordered,get_ground_state)
from itertools import combinations
import matplotlib.pyplot as plt
from util_CSF_and_UCSF import *
from Sym_C2V import *
from ferm_utils import (get_on_idx, op_action_tz,op_action_tz_remove_0coef,braket_tz)

import os
print(os.getcwd())

#Parameters section
l_remove_CSF_with_0_Hmat_with_CSF0 = False
l_remove_UCSF_with_0_Hmat_with_UCSF0 = False
l_remove_CSF_with_small_amplitudes = True
l_remove_CSF_with_GS_small_amplitudes = True
l_remove_CSF_chain_0_Hmat_with_CSF0 = False
l_opt_orb = True
l_opt_orb_GS = False
l_initial_orb_rot = False
l_select_ia_based_on_CSF_E0 = True
l_select_ia_based_on_CSF_Eavg = True
l_select_ia_based_on_AVO1by1 = False
l_pairex_within_actmo = False
l_use_decomp_genmat = True
l_include_ia_in_CAS = True
l_no_gen_groupping = True
l_no_sym = True
l_dooh_sym = False
l_no_gen_groupping = l_no_sym

mp2_ampld_thrsh = float(sys.argv[2])
nparal = int(sys.argv[3])
if nparal <= 0:
    print(f'Non positive number of processes detected, nparal: {nparal}. Bombing out!')
    sys.exit()
list_seniority = []

actmo_start  = int(sys.argv[4])
actmo_end    = int(sys.argv[5])
print("actmo_start:", int(sys.argv[4]))
print("actmo_end:", int(sys.argv[5]))
# Spin sector: 2*S (0 = singlet, 2 = triplet, ...). Optional 17th positional
# arg; defaults to singlet so pre-existing 16-arg invocations still work.
S_by2 = int(sys.argv[17]) if len(sys.argv) > 17 else 0
# CSF small-amplitude filter threshold: a CSF is dropped if |coef| < this in
# every target eigenstate. Optional 18th positional arg; defaults to 1e-6 so
# pre-existing invocations are unchanged.
csf_small_thrsh = float(sys.argv[18]) if len(sys.argv) > 18 else 1e-6
rdist = float(sys.argv[6])

small_orbrot_angle = 0.01
# Convergence test for the orbital-optimization loop, applied to the STATE-
# AVERAGED energy.  At the old fixed 1e-4 a step that moves the average by
# 0.09 mHa terminates the loop while an individual root is still moving by
# no_states * 0.09 mHa -- which is outside chemical accuracy for no_states >= 2
# and was the reason the 3.0 A / n=3 run stopped with root 2 several mHa high
# while its average was still improving.  Override with ORBOPT_CONV.
orbopt_conv_thrsh = float(os.environ.get('ORBOPT_CONV', 1.0e-4))
Ethrsh_select_ia = float(sys.argv[7])
Uopt_thrsh = float(sys.argv[8])
text_initial_orb_rot = sys.argv[9]
if text_initial_orb_rot == 'True':
    l_initial_orb_rot = True
text_opt_orb = sys.argv[10]
if text_opt_orb == 'False':
    l_opt_orb = False

#actmos are for creating all possible SOMOs.
#internal mos are for defining pair excited CSFs to be linearly combined, instead of Uext perturbed.
internal_mo_start = int(sys.argv[11])
internal_mo_end   = int(sys.argv[12])
irrep_label_choice = sys.argv[13]
no_states = int(sys.argv[14])
ratio = float(sys.argv[15])
combo_order = int(sys.argv[16])

# Core orbitals below actmo_start are frozen — exclude them from all ia_pair selections.
list_mo_exclud = list(range(actmo_start))

#Parameters section done
filnam_pyscf_phys_spatial = sys.argv[1]

#Print out the key parameters
print(f'\nHamiltonian read from {filnam_pyscf_phys_spatial}')
print(f'MP2 amplitude threshold to pre-screen pair excitations: {mp2_ampld_thrsh}')
if l_select_ia_based_on_CSF_E0 and l_select_ia_based_on_AVO1by1:
    print('Two schemes of selecting ia pairs are invoked. This cannot be the case. Choose one only.')
    print('Either l_select_ia_based_on_CSF_E0 = True of l_select_ia_based_on_AVO1by1 = True')
    print('Bombing out!')
    sys.exit()
if l_select_ia_based_on_CSF_E0:
    print(f'ia pairs are selected based on their capabilities in lowering E0')
elif l_select_ia_based_on_AVO1by1:
    print(f'ia pairs are selected based on the AVO 1x1 scheme')
else:
    print(f'Bombing out! No ia selection scheme is chosen.')
    sys.exit()
print(f'Active orbitals from and including {actmo_start} to {actmo_end}')
if l_opt_orb:
    print(f'Orbital optimization will be carried out with the convergence threshold of {orbopt_conv_thrsh}')
else:
    print('No orbital optimization will be carried out')
if l_initial_orb_rot:
    print(f'Initial orbital rotation is to be carried out')
print(f'Energy threshold for selecting ia pairs in E0 lowering scheme: {Ethrsh_select_ia}')
print(f'Convergence threshold for U optimization including pair excitation within CAS: {Uopt_thrsh}')
if nparal > 1:
    print(f'The calculation will be done in parallel with {nparal} processes.')
if len(list_seniority) == 0:
    print(f'There is no restriction on seniority selection')
else:
    print(f'Only CSFs of the following seniority are selected: {list_seniority}')

with open(filnam_pyscf_phys_spatial,'rb') as f:
    _ham_rec = pickle.load(f)
# Slice rather than unpack.  Hamiltonians written by Ham_gen/h2o_sto3g_sweep.py
# carry an 8th element holding the CASSCF orbital irreps, so a fixed 7-tuple
# unpack would raise on them.  Older files stop at 7 and give ham_sym = None.
Enuc, obt_spatial, tbt_spatial, orbene, nelec, nactmo, nactel = _ham_rec[:7] #This is for the CASSCF dump file
#Enuc, obt_spatial, tbt_spatial, orbene, nelec                 = _ham_rec[:5] #This is for the rhf dump file
ham_sym = _ham_rec[7] if len(_ham_rec) > 7 else None
if l_use_decomp_genmat:
    print('Analytical U matrices with decomposed generating matrices will be used')
else:
    print('Numerical U matrices from scipy.linalg.expm will be used')
if l_remove_CSF_chain_0_Hmat_with_CSF0:
    print('Original CSFs that do not appear in the same eigenstates with CSF0 will be removed')
    print('They do not have the same IRREPs with CSF0')

print(f'Target Irrep label: {irrep_label_choice}')
print(f'Spin sector S_by2 = {S_by2} (2*S; 0=singlet, 2=triplet), S = {S_by2/2.0}')
print(f'Number of target states: {no_states}')

print('\nThe following orbitals will be excluded from reference CSF construction')
print(list_mo_exclud)
print('\nThe following orbitals will be excluded from internal excitations')
print(list_mo_exclud)

n_spatialmo = obt_spatial.shape[0]
n_spinmo = 2*n_spatialmo

# irrep_labels = ['A1', 'A2', 'B1', 'B2']
irrep_labels = ['A', 'B']

# if rdist == 1.5 or rdist == 1.75 or rdist == 2.0 or rdist == 2.25 or rdist == 2.5:
#     mo_sym = {0: 'A1', 1: 'A1', 2: 'B2', 3: 'B1', 4: 'A1', 5: 'A1', 6: 'B1'}
# elif rdist == 0.75 or rdist == 1.0:
#     mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'A1', 4: 'B2', 5: 'A1', 6: 'B1'}
# elif rdist == 1.25:
#     mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'B2', 4: 'A1', 5: 'A1', 6: 'B1'}

# Orbital irreps.  PREFERRED SOURCE is the Hamiltonian file itself: the irreps
# are a property of the orbitals the integrals were built from, so shipping them
# separately invites exactly the drift that happened here.  Everything below the
# `ham_sym` branch is a fallback for files written before that was added.
if ham_sym is not None:
    mo_sym = dict(ham_sym['mo_sym'])
    # irrep_labels is NOT set from set(mo_sym.values()) -- that is the set of
    # irreps carried by ORBITALS, and states can span irreps no orbital has.
    # STO-3G water has no A2 orbital but 14 A2 CSFs, so deriving the labels
    # here made A2 an "Invalid Irrep Choice".  It comes from the point group
    # below instead.
    print(f"Orbital irreps read from the Hamiltonian file "
          f"({ham_sym['point_group']}, {len(mo_sym)} MOs)")
    if not ham_sym.get('pure', True):
        worst = max(ham_sym['contamination'])
        bad = [i for i, c in enumerate(ham_sym['contamination']) if c > 1e-6]
        raise SystemExit(
            f'\nERROR: the orbitals in {filnam_pyscf_phys_spatial} are not '
            f'irrep-pure.\n'
            f'  max contamination {worst:.2e} on MO(s) {bad}\n'
            f'  Those orbitals are genuine mixtures of two irreps, so no single\n'
            f'  label is correct and every CSF built on them would be assigned a\n'
            f'  symmetry it does not have.  Symmetry-constrain the CASSCF at this\n'
            f'  geometry before using it, or drop the geometry.')

# CASSCF energy-sorted orbital irreps per O-O distance (from Ham_gen/h2o2_sto3g.py).
# The A/B ordering shifts with bond length due to level crossings (core O(1s)
# pair, the frontier sigma/sigma* LUMO, and an inner-valence pair).
elif 'h2o2' in filnam_pyscf_phys_spatial:
    _pg_fallback = 'C2'
    if rdist == 1.0:
        mo_sym = {0: 'A', 1: 'B', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'B', 8: 'A', 9: 'A', 10: 'B', 11: 'B'}
    elif rdist == 1.25:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'A', 8: 'B', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 1.5:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'A', 8: 'B', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 1.75:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'B', 8: 'A', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 1.875:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'B', 8: 'A', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 2.0:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'B', 8: 'A', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 2.25:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'B', 8: 'A', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 2.5:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'A', 6: 'A', 7: 'B', 8: 'A', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 2.75:
        mo_sym = {0: 'B', 1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'A', 6: 'A', 7: 'B', 8: 'A', 9: 'B', 10: 'A', 11: 'B'}
    elif rdist == 3.0:
        mo_sym = {0: 'A', 1: 'B', 2: 'A', 3: 'B', 4: 'B', 5: 'A', 6: 'A', 7: 'B', 8: 'A', 9: 'B', 10: 'A', 11: 'B'}

# CASSCF orbital irreps per O-H distance, from Ham_gen/h2o_sto3g_full.py.
# H2O is C2v with seven orbitals; the h2o2 block above is C2 with twelve.
#
# Two crossings move the labels: B2/B1 swap between 1.25 and 1.5 A, and the
# frontier pair reorders past 2.5 A as the molecule dissociates.  Where two
# orbitals sit within ~1 mHa the order is a convention, not a measurement --
# see stable_argsort in the generator -- so this table is only valid for
# Hamiltonians produced by the same run that produced it.  Prefer the embedded
# record: any file from h2o_sto3g_full.py carries its own labels and takes the
# branch above, which cannot drift out of step with the integrals.
elif 'h2o' in filnam_pyscf_phys_spatial:
    _pg_fallback = 'C2v'
    irrep_labels = ['A1', 'A2', 'B1', 'B2']
    if rdist == 0.75:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B2', 3: 'A1', 4: 'B1', 5: 'A1', 6: 'B2'}
    elif rdist == 1.0:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B2', 3: 'A1', 4: 'B1', 5: 'A1', 6: 'B2'}
    elif rdist == 1.25:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B2', 3: 'B1', 4: 'A1', 5: 'A1', 6: 'B2'}
    elif rdist == 1.5:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'B2', 4: 'A1', 5: 'A1', 6: 'B2'}
    elif rdist == 1.75:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'B2', 4: 'A1', 5: 'A1', 6: 'B2'}
    elif rdist == 2.0:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'B2', 4: 'A1', 5: 'A1', 6: 'B2'}
    elif rdist == 2.25:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'B2', 4: 'A1', 5: 'A1', 6: 'B2'}
    elif rdist == 2.5:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'B2', 4: 'A1', 5: 'A1', 6: 'B2'}
    elif rdist == 2.75:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'A1', 4: 'A1', 5: 'B2', 6: 'B2'}
    elif rdist == 3.0:
        mo_sym = {0: 'A1', 1: 'A1', 2: 'B1', 3: 'A1', 4: 'B2', 5: 'A1', 6: 'B2'}

if 'mo_sym' not in dir():
    raise SystemExit(
        f'\nERROR: no orbital irreps for rdist={rdist} in '
        f'{filnam_pyscf_phys_spatial}.\n'
        f'  Add the geometry to the table above, or regenerate the Hamiltonian\n'
        f'  with a generator that embeds the irreps.')
if len(mo_sym) != n_spatialmo:
    raise SystemExit(
        f'\nERROR: irrep table has {len(mo_sym)} entries but the Hamiltonian '
        f'has {n_spatialmo} spatial MOs.')

dinfh_degenerate_groups = []
# Build partner lookup
degenerate_partners = {}
for group in dinfh_degenerate_groups:
    for orb in group:
        degenerate_partners[orb] = [o for o in group if o != orb]

# irrep_labels = ['Ag', 'B1g', 'B1u', 'B2g', 'B2u', 'Au', 'B3u', 'B3g']
# d2h_mo_sym = {0:"Ag", 1:"Au", 2:"Ag", 3:"Au", 4:"B3u", 5:"B2u", 6:"Ag", 7:"B3g", 8:"B2g", 9:"Au"}
# d2h_table = {('Ag',  'Ag' ): 'Ag',   ('Ag',  'B1g'): 'B1g',  ('Ag',  'B2g'): 'B2g',  ('Ag',  'B3g'): 'B3g',
#     ('Ag',  'Au' ): 'Au',   ('Ag',  'B1u'): 'B1u',  ('Ag',  'B2u'): 'B2u',  ('Ag',  'B3u'): 'B3u',
#
#     ('B1g', 'Ag' ): 'B1g',  ('B1g', 'B1g'): 'Ag',   ('B1g', 'B2g'): 'B3g',  ('B1g', 'B3g'): 'B2g',
#     ('B1g', 'Au' ): 'B1u',  ('B1g', 'B1u'): 'Au',   ('B1g', 'B2u'): 'B3u',  ('B1g', 'B3u'): 'B2u',
#
#     ('B2g', 'Ag' ): 'B2g',  ('B2g', 'B1g'): 'B3g',  ('B2g', 'B2g'): 'Ag',   ('B2g', 'B3g'): 'B1g',
#     ('B2g', 'Au' ): 'B2u',  ('B2g', 'B1u'): 'B3u',  ('B2g', 'B2u'): 'Au',   ('B2g', 'B3u'): 'B1u',
#
#     ('B3g', 'Ag' ): 'B3g',  ('B3g', 'B1g'): 'B2g',  ('B3g', 'B2g'): 'B1g',  ('B3g', 'B3g'): 'Ag',
#     ('B3g', 'Au' ): 'B3u',  ('B3g', 'B1u'): 'B2u',  ('B3g', 'B2u'): 'B1u',  ('B3g', 'B3u'): 'Au',
#
#     ('Au',  'Ag' ): 'Au',   ('Au',  'B1g'): 'B1u',  ('Au',  'B2g'): 'B2u',  ('Au',  'B3g'): 'B3u',
#     ('Au',  'Au' ): 'Ag',   ('Au',  'B1u'): 'B1g',  ('Au',  'B2u'): 'B2g',  ('Au',  'B3u'): 'B3g',
#
#     ('B1u', 'Ag' ): 'B1u',  ('B1u', 'B1g'): 'Au',   ('B1u', 'B2g'): 'B3u',  ('B1u', 'B3g'): 'B2u',
#     ('B1u', 'Au' ): 'B1g',  ('B1u', 'B1u'): 'Ag',   ('B1u', 'B2u'): 'B3g',  ('B1u', 'B3u'): 'B2g',
#
#     ('B2u', 'Ag' ): 'B2u',  ('B2u', 'B1g'): 'B3u',  ('B2u', 'B2g'): 'Au',   ('B2u', 'B3g'): 'B1u',
#     ('B2u', 'Au' ): 'B2g',  ('B2u', 'B1u'): 'B3g',  ('B2u', 'B2u'): 'Ag',   ('B2u', 'B3u'): 'B1g',
#
#     ('B3u', 'Ag' ): 'B3u',  ('B3u', 'B1g'): 'B2u',  ('B3u', 'B2g'): 'B1u',  ('B3u', 'B3g'): 'Au',
#     ('B3u', 'Au' ): 'B3g',  ('B3u', 'B1u'): 'B2g',  ('B3u', 'B2u'): 'B1g',  ('B3u', 'B3u'): 'Ag',}
#
# mult_table = {('A1g',  'A1g' ): 'A1g',  ('A1g',  'A1u' ): 'A1u',  ('A1g',  'A2g' ): 'A2g',  ('A1g',  'A2u' ): 'A2u',  ('A1g',  'E1gx'): 'E1gx', ('A1g',  'E1gy'): 'E1gy', ('A1g',  'E1ux'): 'E1ux', ('A1g',  'E1uy'): 'E1uy',('A1u',  'A1g' ): 'A1u',  ('A1u',  'A1u' ): 'A1g',  ('A1u',  'A2g' ): 'A2u',  ('A1u',  'A2u' ): 'A2g',  ('A1u',  'E1gx'): 'E1ux', ('A1u',  'E1gy'): 'E1uy', ('A1u',  'E1ux'): 'E1gx', ('A1u',  'E1uy'): 'E1gy',
#      ('A2g',  'A1g' ): 'A2g',  ('A2g',  'A1u' ): 'A2u',  ('A2g',  'A2g' ): 'A1g',  ('A2g',  'A2u' ): 'A1u',  ('A2g',  'E1gx'): 'E1gy', ('A2g',  'E1gy'): 'E1gx', ('A2g',  'E1ux'): 'E1uy', ('A2g',  'E1uy'): 'E1ux',
#      ('A2u',  'A1g' ): 'A2u',  ('A2u',  'A1u' ): 'A2g',  ('A2u',  'A2g' ): 'A1u',  ('A2u',  'A2u' ): 'A1g',  ('A2u',  'E1gx'): 'E1uy', ('A2u',  'E1gy'): 'E1ux', ('A2u',  'E1ux'): 'E1gy', ('A2u',  'E1uy'): 'E1gx',
#      ('E1gx', 'A1g' ): 'E1gx', ('E1gx', 'A1u' ): 'E1ux', ('E1gx', 'A2g' ): 'E1gy', ('E1gx', 'A2u' ): 'E1uy', ('E1gx', 'E1gx'): 'A1g',  ('E1gx', 'E1gy'): 'A2g',  ('E1gx', 'E1ux'): 'A1u',  ('E1gx', 'E1uy'): 'A2u',
#      ('E1gy', 'A1g' ): 'E1gy', ('E1gy', 'A1u' ): 'E1uy', ('E1gy', 'A2g' ): 'E1gx', ('E1gy', 'A2u' ): 'E1ux', ('E1gy', 'E1gx'): 'A2g',  ('E1gy', 'E1gy'): 'A1g',  ('E1gy', 'E1ux'): 'A2u',  ('E1gy', 'E1uy'): 'A1u',
#      ('E1ux', 'A1g' ): 'E1ux', ('E1ux', 'A1u' ): 'E1gx', ('E1ux', 'A2g' ): 'E1uy', ('E1ux', 'A2u' ): 'E1gy', ('E1ux', 'E1gx'): 'A1u',  ('E1ux', 'E1gy'): 'A2u',  ('E1ux', 'E1ux'): 'A1g',  ('E1ux', 'E1uy'): 'A2g',
#      ('E1uy', 'A1g' ): 'E1uy', ('E1uy', 'A1u' ): 'E1gy', ('E1uy', 'A2g' ): 'E1ux', ('E1uy', 'A2u' ): 'E1gx', ('E1uy', 'E1gx'): 'A2u',  ('E1uy', 'E1gy'): 'A1u',  ('E1uy', 'E1ux'): 'A2g',  ('E1uy', 'E1uy'): 'A1g',}
# mult_table = {
#     ('A1', 'A1'): 'A1', ('A1', 'A2'): 'A2', ('A1', 'B1'): 'B1', ('A1', 'B2'): 'B2',
#     ('A2', 'A1'): 'A2', ('A2', 'A2'): 'A1', ('A2', 'B1'): 'B2', ('A2', 'B2'): 'B1',
#     ('B1', 'A1'): 'B1', ('B1', 'A2'): 'B2', ('B1', 'B1'): 'A1', ('B1', 'B2'): 'A2',
#     ('B2', 'A1'): 'B2', ('B2', 'A2'): 'B1', ('B2', 'B1'): 'A2', ('B2', 'B2'): 'A1',
# }
# Direct-product table, chosen by point group rather than hardcoded.  H2O2 is C2
# (A/B) and H2O is C2v (A1/A2/B1/B2), so a single fixed table cannot serve both:
# with the C2 table above, the first ('A1','B2') lookup raises KeyError.
_MULT_TABLES = {
    'C2': {('A', 'A'): 'A', ('A', 'B'): 'B',
           ('B', 'A'): 'B', ('B', 'B'): 'A'},
    'C2v': {
        ('A1', 'A1'): 'A1', ('A1', 'A2'): 'A2', ('A1', 'B1'): 'B1', ('A1', 'B2'): 'B2',
        ('A2', 'A1'): 'A2', ('A2', 'A2'): 'A1', ('A2', 'B1'): 'B2', ('A2', 'B2'): 'B1',
        ('B1', 'A1'): 'B1', ('B1', 'A2'): 'B2', ('B1', 'B1'): 'A1', ('B1', 'B2'): 'A2',
        ('B2', 'A1'): 'B2', ('B2', 'A2'): 'B1', ('B2', 'B1'): 'A2', ('B2', 'B2'): 'A1',
    },
}
# `_pg_fallback` is set by whichever hardcoded branch ran.  It must NOT default
# to C2: an h2o file taking the fallback path is C2v, and the C2 table has no
# ('A1','B2') entry, so the first CSF symmetry product would KeyError.
_point_group = (ham_sym['point_group'] if ham_sym is not None
                else globals().get('_pg_fallback', 'C2'))
if _point_group not in _MULT_TABLES:
    raise SystemExit(f'\nERROR: no direct-product table for point group '
                     f'{_point_group}. Add one to _MULT_TABLES.')
mult_table = _MULT_TABLES[_point_group]

# A label in mo_sym with no row in the table would fail deep inside the CSF
# symmetry recursion, where the traceback says nothing useful.
_unknown = sorted({s for s in mo_sym.values()
                   if (s, s) not in mult_table})
if _unknown:
    raise SystemExit(f'\nERROR: orbital irreps {_unknown} are not in the '
                     f'{_point_group} direct-product table.')

# The point group is the authority on which irreps a STATE may carry, so the
# selectable labels come from the product table rather than from the orbitals.
irrep_labels = sorted({k[0] for k in mult_table})

print(f"point group: {_point_group}   irrep labels: {irrep_labels}")
print("mo_sym:", mo_sym)

# if l_dooh_sym:
#     print('Degenerate partners:', degenerate_partners)
#
#     mo_ml  = {0:0, 1:0, 2:0, 3:0, 4:+1, 5:-1, 6:0, 7:+1, 8:-1, 9:0}
#     mo_par = {0:+1, 1:-1, 2:+1, 3:-1, 4:-1, 5:-1, 6:+1, 7:+1, 8:+1, 9:-1}
#
#     def csf_symmetry_dooh(somo, dmo, mo_ml, mo_par, mo_sym, mult_table):
#         # DOMOs are paired ; only SOMOs determine the D∞h irrep
#         ML = sum(mo_ml.get(o, 0) for o in somo) + 2*sum(mo_ml.get(o, 0) for o in dmo)
#         par = 1
#         for o in somo:          # DOMOs contribute mo_par²=+1, only SOMOs matter
#             par *= mo_par.get(o, 1)
#         par_label = 'g' if par > 0 else 'u'
#         if ML == 0:
#             # Multiply only SOMO D∞h irreps; DOMOs contribute A1g and don't change the product
#             if not somo:
#                 return 'A1g'
#             result = mo_sym[somo[0]]
#             for o in somo[1:]:
#                 result = mult_table[(result, mo_sym[o])]
#             return result
#         elif ML == +1: return f'E1{par_label}x'
#         elif ML == -1: return f'E1{par_label}y'
#         elif ML == +2: return f'E2{par_label}x'
#         elif ML == -2: return f'E2{par_label}y'
#         elif ML == +3: return f'E3{par_label}x'
#         elif ML == -3: return f'E3{par_label}y'
#         else:          return f'ML={ML}{par_label}'
#
#     dooh_irrep_labels = ['A1g', 'A2g', 'E1gx', 'E1gy', 'E2gx', 'E2gy',
#         'A1u', 'A2u', 'E1ux', 'E1uy', 'E2ux', 'E2uy',]
#     dooh_degen_pairs = {'E1gx': ['E1gx', 'E1gy'], 'E1gy': ['E1gx', 'E1gy'],
#         'E1ux': ['E1ux', 'E1uy'], 'E1uy': ['E1ux', 'E1uy'],
#         'E2gx': ['E2gx', 'E2gy'], 'E2gy': ['E2gx', 'E2gy'],
#         'E2ux': ['E2ux', 'E2uy'], 'E2uy': ['E2ux', 'E2uy'],}
#
#     def get_target_dooh_labels(irrep_choice):
#         return dooh_degen_pairs.get(irrep_choice, [irrep_choice])
#
#     print(f'\nD∞h symmetry adaptation enabled')
#     print(f'mo_ml:  {mo_ml}')
#     print(f'mo_par: {mo_par}')

#If read-in orbital energies contain doubly degeneracy, axial symmetry is detected.
#Symmetry-adapted CSFs will be generated.
list_degmo = []
l_axial_sym = False
for iorb in range(n_spatialmo-1):
    if np.isclose(orbene[iorb],orbene[iorb+1]):
        l_axial_sym = True
        list_degmo.append([iorb,iorb+1])

if l_axial_sym:
    print('Axial symmetry detected, with the following pairs of doubly degenerate orbitals:')
    for degmo_pair in list_degmo:
        print(degmo_pair)

if l_no_sym:
    print(f'\nAlthough axial symmetri is detected, no symmetry consideration is applied, given the hard put-in l_no_sym: {l_no_sym}')
    l_axial_sym = False

list_orb_rot_filnam = 'list_orb_rot_' + str(rdist)

list_orb_rot = []
if os.path.isfile(list_orb_rot_filnam):
    f = open(list_orb_rot_filnam, 'r')
    file_data = f.readlines()
    line_list_orb_rot_start = -1
    line_list_orb_rot_end   = -1
    line_ini_orb_rot_start = -1
    line_ini_orb_rot_end   = -1
    for iline in range(len(file_data)):
       #if 'list_orb_rot_start' in file_data[iline] : line_list_orb_rot_start = iline + 1
        if file_data[iline].startswith("list_orb_rot_start"): line_list_orb_rot_start = iline + 1
        if file_data[iline].startswith("list_orb_rot_end"): line_list_orb_rot_end = iline - 1
        if file_data[iline].startswith("list_initial_rotation_start"): line_ini_orb_rot_start = iline + 1
        if file_data[iline].startswith("list_initial_rotation_end"): line_ini_orb_rot_end = iline - 1
    print(f'line_list_orb_rot_start: {line_list_orb_rot_start}')
    print(f'line_list_orb_rot_end  : {line_list_orb_rot_end}')
    if line_list_orb_rot_start >= line_list_orb_rot_end or line_list_orb_rot_start <= 0 or line_list_orb_rot_end <= 0:
        print(f'\n Symmetry-preserving orbital pair rotations are allowed')
        for imo in range(n_spatialmo):
            for jmo in range(imo + 1, n_spatialmo):
                if mo_sym[imo] == mo_sym[jmo]:
                    list_orb_rot.append([imo, jmo])
    else:
        if line_list_orb_rot_start >= line_list_orb_rot_end or line_list_orb_rot_start <= 0 or line_list_orb_rot_end <= 0:
            print(f'\n Symmetry-preserving orbital pair rotations are allowed')
            for imo in range(n_spatialmo):
                for jmo in range(imo + 1, n_spatialmo):
                    if mo_sym[imo] == mo_sym[jmo]:
                        list_orb_rot.append([imo, jmo])
    if l_initial_orb_rot:
        print(f'\nInitialize orbital rotation')
        if line_ini_orb_rot_start >= line_ini_orb_rot_end or line_ini_orb_rot_start <= 0 or line_ini_orb_rot_end <=0:
            print(f'Check {list_orb_rot_filnam}. The initial orbital rotations cannot be read. Bombing out!')
            sys.exit()
        list_initial_rotation = []
        x_orbrot_0 = np.zeros(len(list_orb_rot))
        for iline in range(line_ini_orb_rot_start,line_ini_orb_rot_end+1):
            orb1, orb2, x_rot = int(file_data[iline].split()[0]),int(file_data[iline].split()[1]),float(file_data[iline].split()[2])
            list_initial_rotation.append([[orb1,orb2],x_rot])
       #Symmetrize the rotations for degenerate pairs
       #same_xrot_thrsh = 1.0e-4
       #for iitem, item in enumerate(list_initial_rotation):
       #    [[iorb1,iorb2],x_rot_i] = item
       #    for jtem in list_initial_rotation[iitem+1:]:
       #        [[jorb1,jorb2],x_rot_j] = jtem
       #        if ([iorb1,jorb1] in list_degmo or [jorb1,iorb1] in list_degmo) and \
       #          ([iorb2,jorb2] in list_degmo or [jorb2,iorb2] in list_degmo):
       #            print(f'Rotations to be symmetrized: {item,jtem}')
       #            if abs(x_rot_i - x_rot_j) > same_xrot_thrsh:
       #                print(f'Original read-in rotational angles differ > {same_xrot_thrsh}')
       #                print(f'Please check. Bombing out!')
       #                sys.exit()
       #            x_rot_sym = 0.5*(x_rot_i + x_rot_j)
       #            item[1] = x_rot_sym
       #            jtem[1] = x_rot_sym
       #print(f'\nlist_initial_rotation: {list_initial_rotation}')
        for item in list_initial_rotation:
            if item[0] in list_orb_rot:
               #print(f'Found item[0]')
                x_ind = list_orb_rot.index(item[0])
                x_orbrot_0[x_ind] = item[1]

        if l_axial_sym or l_dooh_sym:
            symmetrize_xorbrot(list_orb_rot,x_orbrot_0,list_degmo)
        for ii, orbpair in enumerate(list_orb_rot):
            if not np.isclose(x_orbrot_0[ii],0.0):
                print(orbpair,x_orbrot_0[ii])
    f.close()
else:
    print(f'\n Symmetry-preserving orbital pair rotations are allowed')
    n = len(mo_sym)
    for imo in range(n):
        for jmo in range(imo + 1, n):
            if mo_sym[imo] == mo_sym[jmo]:
                list_orb_rot.append([imo, jmo])

    print(f'\n All orbital pair rotations are added to the full list')
    list_orb_ind = []
    list_orb_rot_full = []
    for imo in range(n_spatialmo):
        list_orb_ind.append(imo)
    for iimo, imo in enumerate(list_orb_ind):
        for jmo in list_orb_ind[iimo + 1:]:
            list_orb_rot_full.append([imo, jmo])

print(f'\nList of orbitals that can be mixed:')
for item in list_orb_rot:
    print(item, mo_sym[item[0]], mo_sym[item[1]])

if l_dooh_sym:
    def build_linked_orbrot_pairs(dinfh_degenerate_groups):
        partner = {}
        for a, b in dinfh_degenerate_groups:
            partner[a] = b
            partner[b] = a

        degen_set = set()

        for a, b in dinfh_degenerate_groups:
            degen_set.add(a)
            degen_set.add(b)

        degen_list = list(degen_set)

        def sym_pair(a):
            return partner.get(a, a)

        def act(edge):
            a, b = edge
            return (sym_pair(a), sym_pair(b))

        def canon(edge):
            return tuple(sorted(edge))

        result = []
        seen = set()

        for p in degen_list:
            for q in degen_list:

                if q == p:
                    continue

                if q != sym_pair(p):

                    # canonical key for uniqueness (must be hashable → use tuple here internally)
                    key = tuple(sorted([p, q]))

                    if key not in seen:
                        seen.add(key)
                        result.append(sorted([p, q]))

        result = list(result)

        return result

    linked_orbrot_pairs = build_linked_orbrot_pairs(dinfh_degenerate_groups)
    print(f'\nLinked orbital rotation pairs (must share angle): {linked_orbrot_pairs}')
    list_orb_rot += linked_orbrot_pairs
    for pair in linked_orbrot_pairs:
        print(f' {pair[0]} links {pair[1]}')

if l_initial_orb_rot:
    obt, tbt = orthogonal_transform_obt_tbt(x_orbrot_0,list_orb_rot,obt_spatial,tbt_spatial)
else:
    obt = obt_phys_spatial_to_spin(obt_spatial)
    tbt = tbt_phys_spatial_to_spin(tbt_spatial)

list_act_orb = []
for iorb in range(actmo_start,actmo_end+1):
    list_act_orb.append(iorb)

homo = nelec // 2 - 1
lumo = nelec //2

#Parameters section
nactorb = actmo_end - actmo_start + 1 #Here, nactorb is allowed to be different from the read-in nactmo
nelact = nelec - (actmo_start)*2 #nelact is allowed to be different from the read-in nactel
#Parameters section done

mp2_ampld_list, mp2_Ecorr_list, ia_pair_list = prepare_mp2_amplitudes_actmo(actmo_end,actmo_start,orbene,tbt,l_no_sym=l_no_sym,debug=True)
print(f'List of pairs of occupied and unoccupied spatial orbitals:')
print(ia_pair_list)
print(f'List of MP2 amplitudes:')
print(mp2_ampld_list)
print(f'List of MP2 E corr.:')
print(mp2_Ecorr_list)

mp2_ampld_list_full, mp2_Ecorr_list_full, ia_pair_list_full = prepare_mp2_amplitudes_actmo(n_spatialmo-1, 0,orbene,tbt,l_no_sym=l_no_sym,debug=True)  # ← full virtual space,               # ← from frozen coreorbene, tbt, ...)

sorted_mp2_ampld, sorted_ia_pair, sorted_mp2_Ecorr = sort_mp2_amplitudes(mp2_ampld_list,mp2_Ecorr_list,ia_pair_list,False)


sorted_mp2_ampld_full, sorted_ia_pair_full, sorted_mp2_Ecorr_full = sort_mp2_amplitudes(mp2_ampld_list_full,mp2_Ecorr_list_full, ia_pair_list_full,False)

print(sorted_mp2_ampld)
print(sorted_ia_pair)
print(sorted_mp2_Ecorr)

remove_mp2_amplitudes(sorted_mp2_ampld,sorted_ia_pair,sorted_mp2_Ecorr,mp2_ampld_thrsh,list_mo_exclud=list_mo_exclud)

remove_mp2_amplitudes(sorted_mp2_ampld_full,sorted_ia_pair_full,sorted_mp2_Ecorr_full,mp2_ampld_thrsh,list_mo_exclud=list_mo_exclud)

print('\nAfter removal of small amplitudes:')
print(sorted_mp2_ampld)
print(sorted_ia_pair)
print(sorted_mp2_Ecorr)

group_mp2_ampld, group_mp2_iapair, group_mp2_Ecorr = group_mp2_amplitudes(sorted_mp2_ampld,sorted_ia_pair,sorted_mp2_Ecorr,l_no_groupping=l_no_gen_groupping)

group_mp2_ampld_full, group_mp2_iapair_full, group_mp2_Ecorr_full = group_mp2_amplitudes(sorted_mp2_ampld_full,sorted_ia_pair_full,sorted_mp2_Ecorr_full,l_no_groupping=l_no_gen_groupping)

print(f'Grouped MP2 amplitudes, orbital pairs, and Ecorr.')
for iampld,ampld in enumerate(group_mp2_ampld):
    print(ampld,group_mp2_iapair[iampld],group_mp2_Ecorr[iampld])

list_CSF, list_SOMO_DMO = generate_CASCI_space(n_spatialmo,nelec,nactorb,nelact,S_by2,list_seniority,list_mo_exclud=list_mo_exclud,l_pair_ex=l_pairex_within_actmo,debug=False)

if l_axial_sym and not l_pairex_within_actmo:
    create_missing_axial_sym_CSFs(list_CSF,list_SOMO_DMO,list_degmo)

list_CSF_full = list_CSF.copy()

active_irrep_labels = dooh_irrep_labels if l_dooh_sym else irrep_labels
csf_dict = {label: [] for label in active_irrep_labels}
print(f'SOMO and DMO of original CSF before screening')
for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
    somo = SOMO_DMO[0]
    dmo  = SOMO_DMO[1]
    if l_dooh_sym:
        csf_sym = csf_symmetry_dooh(somo, dmo, mo_ml, mo_par, mo_sym, mult_table)
        ML  = sum(mo_ml.get(o, 0) for o in somo) + 2*sum(mo_ml.get(o, 0) for o in dmo)
        par = 1
        for o in somo: par *= mo_par.get(o, 1)  # DOMOs contribute mo_par²=+1
        print(f'CSF Basis {ii}, SOMO: {somo}, DMO: {dmo}, Dooh Label: {csf_sym}, ML={ML}, par={par}')
        SOMO_DMO.extend([csf_sym, ML, par])
    else:
        csf_sym = csf_symmetry_from_somos(somo, dmo, mo_sym, mult_table)
        print(f'CSF Basis {ii}, SOMO: {somo}, DMO: {dmo}, Irrep Label: {csf_sym}')
        SOMO_DMO.append(csf_sym)
    for sym_label in active_irrep_labels:
        if csf_sym == sym_label:
            csf_dict[sym_label].append(ii)

for key in csf_dict:
    globals()[f"{key}_csfs"] = csf_dict[key]

if l_axial_sym and not l_pairex_within_actmo:
    print(f'Number of CSFs generated by generate_CASCI_space and with sym partners: {len(list_CSF)}')
else:
    print(f'Number of CSFs generated by generate_CASCI_space: {len(list_CSF)}')

list_SOMO_DMO_original = list_SOMO_DMO.copy()

if l_dooh_sym:
    a1g_seed_indices = [i for i, s in enumerate(list_SOMO_DMO_original)if s[2] == 'A1g']
    if len(a1g_seed_indices) == 0:
        print('No formula-A1g CSFs found in list_CSF_full. Bombing out!')
        sys.exit()
    print(f'\nA1g seed CSFs (formula label, Shell 0): {len(a1g_seed_indices)} CSFs')
    for _si in a1g_seed_indices:
        _sm = list_SOMO_DMO_original[_si]
        _somo, _dmo = _sm[0], _sm[1]
        _label = _sm[2] if len(_sm) > 2 else '?'
        _ml    = _sm[3] if len(_sm) > 3 else '?'
        _sen   = len(_somo)
        print(f'  idx={_si:4d}  {_label:6s}  ML={_ml!s:4s}  sen={_sen}  '
              f'SOMO={_somo}  DMO={_dmo}')

    a1g_seed_csfs  = [list_CSF_full[i] for i in a1g_seed_indices]
    coupling_threshold = 1e-8
    n_full = len(list_CSF_full)

    include_set = set(a1g_seed_indices)   # seeds always kept
    for i, csf in enumerate(list_CSF_full):
        if i in include_set:
            continue
        for seed in a1g_seed_csfs:
            helm = Helm_between_CSFs(Enuc, obt, tbt, seed, csf)
            if abs(helm) > coupling_threshold:
                include_set.add(i)
                print(f'  CSF {i:4d}: coupled to A1g seed '
                      f'(|H|={abs(helm):.3e})  '
                      f'SOMO={list_SOMO_DMO_original[i][0]}  '
                      f'DMO={list_SOMO_DMO_original[i][1]}  '
                      f'label={list_SOMO_DMO_original[i][2]}')
                break

    E_diag   = [Helm_between_CSFs(Enuc, obt, tbt, c, c) for c in list_CSF_full]
    assigned = [False] * n_full
    degen_groups = []
    for i in range(n_full):
        if assigned[i]:
            continue
        grp = [i]
        for j in range(i + 1, n_full):
            if not assigned[j] and abs(E_diag[i] - E_diag[j]) < 1e-8:
                grp.append(j)
        for k in grp:
            assigned[k] = True
        degen_groups.append(grp)

    for grp in degen_groups:
        if any(k in include_set for k in grp):
            include_set.update(grp)

    indices = sorted(include_set)
    if len(indices) == 0:
        print('No CSFs couple to A1g seeds. Bombing out!')
        sys.exit()

    screen_list_CSF = [list_CSF_full[i] for i in indices]
    list_SOMO_DMO   = [list_SOMO_DMO_original[i] for i in indices]

    n_a1g  = sum(1 for i in indices if list_SOMO_DMO_original[i][2] == 'A1g')
    n_other = len(indices) - n_a1g
    print(f'l_dooh_sym coupling filter: {len(indices)} CSFs retained '
          f'({n_a1g} formula-A1g + {n_other} non-A1g with A1g character via real-DOMO) '
          f'out of {len(list_CSF_full)} total')
    for i in indices:
        sm = list_SOMO_DMO_original[i]
        print(f'  CSF {i:3d}: {sm[2]:6s}  ML={sm[3]:+d}  SOMO={sm[0]}  DMO={sm[1]}')
    list_CSF = screen_list_CSF
else:
    for key in active_irrep_labels:
        if irrep_label_choice == key:
            indices = csf_dict[key]
            if len(indices) != 0:
                screen_list_CSF = [list_CSF[i] for i in indices]
                list_SOMO_DMO   = [list_SOMO_DMO_original[i] for i in indices]
            break
    # Exit cleanly rather than falling through to an unbound name.
    if irrep_label_choice not in active_irrep_labels:
        raise SystemExit(f'\nERROR: irrep "{irrep_label_choice}" is not one of '
                         f'{active_irrep_labels} for point group {_point_group}.')
    if 'screen_list_CSF' not in dir():
        raise SystemExit(
            f'\nERROR: no CSFs of irrep "{irrep_label_choice}" and S_by2={S_by2} '
            f'survived screening.\n  The sector is empty at this geometry and '
            f'threshold -- nothing to target.')
    list_CSF = screen_list_CSF

print("\nIrrep labels of CSFs in list_CSF:")

for i, SOMO_DMO in enumerate(list_SOMO_DMO):
    somo = SOMO_DMO[0]
    dmo  = SOMO_DMO[1]
    irrep = SOMO_DMO[2]

    print(f"CSF {i}: irrep = {irrep}, SOMO = {somo}, DMO = {dmo}")

# print(f"CSF {i}: {list_CSF[i]}" for i in range(len(list_CSF)))

if l_remove_CSF_with_small_amplitudes:
    Hmat_CSF = construct_Hmat_CSFs_paral_triu(list_CSF, Enuc, obt, tbt, nparal)
    n_lowest = no_states      # number of lowest states to inspect
    small = csf_small_thrsh
    evals, evecs = np.linalg.eigh(Hmat_CSF)
    print(Hmat_CSF)
    print("Eigenstates of H_CSF: ", evals)
    print_eigen_solution(evals, evecs)
    print("Eigenvalues of H_CSF: ", evals)
    psi_low = evecs[:, :n_lowest]
    list_l_remove_CSF = [False] * len(list_CSF)
    for iCSF in range(len(list_CSF)):
        keep = False
        for istate in range(n_lowest):
            if abs(psi_low[iCSF, istate]) >= small:
                keep = True
                break
        if not keep:
            list_l_remove_CSF[iCSF] = True
    n_removed = sum(list_l_remove_CSF)
    print(f"Removing {n_removed} CSFs out of {len(list_CSF)}")

    for iCSF in range(len(list_CSF) - 1, -1, -1):
        if list_l_remove_CSF[iCSF]:
            del list_CSF[iCSF]
            del list_SOMO_DMO[iCSF]

print(f'SOMOs and DMOs of the CSFs:')
for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
    E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_CSF[ii],list_CSF[ii])
    print(f'CSF Basis {ii}, CSF: {list_CSF[ii]}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}, E: {E_CSF}')

if l_axial_sym:
    list_CSF, list_SOMO_DMO, list_sym_sign, list_sym_CSF = reorder_list_CSF_for_sym(list_CSF,list_SOMO_DMO,Enuc,obt,tbt,nparal,debug=True)
    print(f'SOMOs and DMOs of the sym-reordered CSFs:')
    for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_CSF[ii],list_CSF[ii])
        print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}, E: {E_CSF}')
    print(f'# of sym-adapted CSFs: {len(list_sym_CSF)}')
else:
    list_sym_sign = [] #Define trivial sym sign list

list_UCSF = []
list_list_pair_ex_space = []
list_list_genmat = []
list_list_theta = []
list_list_ia = []

if l_select_ia_based_on_CSF_Eavg:
    print(f'Selecting ia pairs based on Eavg of CSF space')
    list_ex_space = copy.deepcopy(list_CSF)
    list_list_pair_ex_space, list_list_ia, list_list_genmat, list_sym_CSF_vec, \
    list_improvement, list_improvement_ia, list_list_extra_vecs, shared_unitary_groups, \
    list_list_symadapted_ex_space, list_list_frozen_positions = \
    select_ia_pairs_for_CSF_Eavg_with_sym_CSF(list_CSF, list_SOMO_DMO, group_mp2_iapair, Enuc, obt, tbt,
        l_include_ia_in_CAS, actmo_start, actmo_end, ratio, no_states, degenerate_partners, combo_order, l_axial_sym=l_axial_sym,
        list_sym_sign=list_sym_sign, Ethrsh=Ethrsh_select_ia, list_mo_exclud=list_mo_exclud, nparal=nparal,
        mo_ml=mo_ml if l_dooh_sym else None,dinfh_degenerate_groups=dinfh_degenerate_groups if l_dooh_sym else None, debug=False)
    print('\nlist_list_pair_ex_space:')
    print(list_list_pair_ex_space)
    print(f'len(list_list_pair_ex_space): {len(list_list_pair_ex_space)}')
    for i, ex_space in enumerate(list_list_pair_ex_space):
        print(f' group {i}: {len(ex_space)} CSFs')
    print('\nlist_list_symadapted_ex_space (symmetry-adapted combined excitations):')
    for i, ex_space in enumerate(list_list_symadapted_ex_space):
        print(f' group {i}: {len(ex_space)} symmetry-adapted entries')
        for j in range(len(ex_space)):
            print(ex_space[j])
    for iCSF_group in range(len(list_list_ia)):
        list_list_theta.append([0.0]*len(list_list_ia[iCSF_group]))
else:
    print(f'\nOptimizing flexible U for individual CSF')
    for iCSF, CSF in enumerate(list_CSF):
        print(f'Seeking U space for CSF {iCSF}')
        list_pair_ex_space, list_ia_included, list_ia_ampld, list_genmat = \
          select_ia_pairs_for_one_CSF(CSF,group_mp2_iapair,group_mp2_ampld,Enuc,obt,tbt,Ethrsh=Ethrsh_select_ia,debug=False)
        list_list_pair_ex_space.append(list_pair_ex_space)
        list_list_ia.append(list_ia_included)
        list_list_theta.append(list_ia_ampld)
        list_list_genmat.append(list_genmat)
        ndim = len(list_pair_ex_space)
        print(f'\nDimension of the pair ex space: {ndim}')

# list_list_pairex_CSF_vec, list_list_pairex_CSF = pick_pairex_within_CAS(list_list_pair_ex_space,list_sym_CSF_vec,internal_mo_start,internal_mo_end,l_axial_sym,Enuc,obt,tbt,nparal,list_mo_exclud=list_mo_exclud)

#convert list_list_genmat to list_list_group_genmat. Genmat within each group has the same eigenvalue
if l_use_decomp_genmat:
    list_list_decomp_genmat = groupping_list_list_genmat(list_list_genmat,False)
    for ii in range(len(list_list_decomp_genmat)):
        list_genmat = list_list_genmat[ii]
        list_decomp_genmat = list_list_decomp_genmat[ii]
        assert len(list_genmat) == len(list_decomp_genmat)
        n_theta = len(list_genmat)
        if n_theta == 0: continue
        x_random = np.random.uniform(low=-0.5, high=-0.5, size = n_theta)
        if nparal != 1:
            compare_num_anl_Umat(x_random,list_genmat,list_decomp_genmat,nparal)
else:
    list_list_decomp_genmat = []

#list_list_ia_noCAS, list_list_genmat_noCAS, list_list_theta_noCAS, list_list_decomp_genmat_noCAS = \remove_pairex_in_CAS(internal_mo_start,internal_mo_end,list_list_ia,list_list_genmat,list_list_theta,l_use_decomp_genmat,list_list_decomp_genmat)

# print(f'\nlist_list_ia_noCAS:')
# for iref,item in enumerate(list_list_ia_noCAS):
#     print(f'Group: {iref}, {item}')
# print(f'\nlist_list_theta_noCAS:')
# for iref,item in enumerate(list_list_theta_noCAS):
#     print(f'Group: {iref}, {item}')
#
# list_list_ia_internal, list_list_ia_external = separate_ia_pairs_internal_external(list_list_ia,internal_mo_start,internal_mo_end,list_mo_exclud=list_mo_exclud,debug=True)

# ── Assemble the full ex space (H basis) with dedup ─────────────────────────
# Accepted states carry an "alive" flag so a later, BIGGER uniform combination
# can retroactively evict its smaller pieces (bigger-combination wins).
seen = set()
_accepted_entries = []   # dicts: g, slot, csf, sds (frozenset), coef, alive

_ext_groups = set()
for _par_d, _members_d in shared_unitary_groups.items():
    for _m_d in _members_d:
        if _m_d != _par_d:
            _ext_groups.add(_m_d)
print(f'Basis-extension groups (SD-priority dedup): {sorted(_ext_groups)}')
_refgrp_sd_owner = {}   # SD idx → ref group that first claimed it

def _is_uniform(coefs):
    _exp = round(1.0 / np.sqrt(len(coefs)), 8)
    return all(abs(abs(c) - _exp) < 1e-6 for c in coefs)

for iCSF_group in range(len(list_list_symadapted_ex_space)):
    print(f'Group {iCSF_group} to full ex CSF space')
    for slot, csf in enumerate(list_list_symadapted_ex_space[iCSF_group]):
        # Canonical signature: sort (idx, coef) pairs by SD index so that states
        # with the same SDs stored in different orders hash to the same key.
        _pairs = sorted(zip((int(x) for x in csf[1]), (float(c) for c in csf[2])), key=lambda p: p[0])
        _canon_idx  = tuple(p[0] for p in _pairs)
        _canon_coef = tuple(round(p[1], 8) for p in _pairs)
        sig     = (_canon_idx,  _canon_coef)
        sig_neg = (_canon_idx, tuple(-c for c in _canon_coef))
        if sig in seen or sig_neg in seen:
            print(f'  [dedup] Dropping state (duplicate or -1 multiple): '
                  f'SD-idx={_canon_idx}  coefs={_canon_coef}')
            continue
        _this_sds = frozenset(_canon_idx)
        _drop_reason = None
        # Case A — current state is a proper SUBSET of an accepted uniform
        # state: the bigger combination is already in, drop the piece.
        for _e in _accepted_entries:
            if _e['alive'] and _this_sds < _e['sds'] and _is_uniform(_e['coef']):
                _drop_reason = ('subset', _e['sds'])
                break
        # Case A2 — current state is a uniform superposition whose SD set is a
        # proper SUPERSET of accepted state(s): the bigger combination wins —
        # evict the smaller pieces and accept the current state.
        if _drop_reason is None and _is_uniform(_canon_coef):
            _evict = [_e for _e in _accepted_entries
                      if _e['alive'] and _e['sds'] < _this_sds]
            for _e in _evict:
                _e['alive'] = False
                print(f'  [bigger-set dedup] Evicting previously accepted state '
                      f'group {_e["g"]} slot {_e["slot"]} '
                      f'SD-idx={sorted(_e["sds"])} '
                      f'(subset of bigger uniform state SD-idx={sorted(_this_sds)})')
        # Case B — this state is uniform and its SD-set is the disjoint union
        # of accepted alive uniform states — adds no new basis function.
        if _drop_reason is None and _is_uniform(_canon_coef):
            _sub_accepted = [_e['sds'] for _e in _accepted_entries
                             if _e['alive'] and _e['sds'] < _this_sds
                             and _is_uniform(_e['coef'])]
            _covered = frozenset()
            for _s in _sub_accepted:
                if not (_s & _covered):   # keep only disjoint subsets
                    _covered |= _s
            if _covered == _this_sds:
                _drop_reason = ('union', _covered)
        # Case C — SD-priority: a state from a basis-extension group that
        # shares any SD with an accepted state of a reference group is dropped
        # (the refCSF group's representation of that SD wins).
        if _drop_reason is None and iCSF_group in _ext_groups:
            _clash = {k: _refgrp_sd_owner[k] for k in _canon_idx
                      if k in _refgrp_sd_owner}
            if _clash:
                _drop_reason = ('sd_clash', _clash)
        if _drop_reason is not None:
            _case, _reason_set = _drop_reason
            if _case == 'subset':
                print(f'  [sub-component dedup] Dropping state SD-idx={sorted(_this_sds)} '
                      f'(component of accepted uniform state SD-idx={sorted(_reason_set)})')
            elif _case == 'sd_clash':
                print(f'  [SD-priority dedup] Dropping ext-group {iCSF_group} state '
                      f'SD-idx={sorted(_this_sds)}: SD(s) '
                      f'{sorted(_reason_set.keys())} already claimed by refCSF '
                      f'group(s) {sorted(set(_reason_set.values()))}')
            else:
                print(f'  [sub-component dedup] Dropping state SD-idx={sorted(_this_sds)} '
                      f'(redundant: union of accepted states covers {sorted(_reason_set)})')
            continue
        print(f'Adding {csf} to list_list_symadapted_ex_space')
        seen.add(sig)
        seen.add(sig_neg)   # also block the -1 multiple
        if iCSF_group not in _ext_groups:
            for _k_own in _canon_idx:
                _refgrp_sd_owner.setdefault(_k_own, iCSF_group)
        _accepted_entries.append({'g': iCSF_group, 'slot': slot, 'csf': csf,
                                  'sds': _this_sds, 'coef': _canon_coef,
                                  'alive': True})

# Assemble the flat list from surviving entries, in (group, slot) order
list_all_CSFs_ex_space = []
list_UCSF_subspace_start_end = []
_state_to_slot = {}   # flat_idx → (group, slot_in_symadapted_ex_space)
istart = 0
for iCSF_group in range(len(list_list_pair_ex_space)):
    unique_new = []
    for _e in _accepted_entries:
        if _e['alive'] and _e['g'] == iCSF_group:
            _state_to_slot[istart + len(unique_new)] = (iCSF_group, _e['slot'])
            unique_new.append(_e['csf'])
    iend = istart + len(unique_new)
    list_all_CSFs_ex_space += unique_new
    list_UCSF_subspace_start_end.append([istart, iend])
    print([istart, iend])
    istart = iend

print(f'Full ex CSF space with {sum(len(list_list_pair_ex_space[i]) for i in range(len(list_list_pair_ex_space)))} terms')
for i, group in enumerate(list_list_pair_ex_space):
    print(f'Group {i} contains {len(group)} terms: {group}')

_state_to_group = {}
for _g, (_gs, _ge) in enumerate(list_UCSF_subspace_start_end):
    for _k in range(_gs, _ge):
        _state_to_group[_k] = _g

n_ex = len(list_all_CSFs_ex_space)
S_ov = np.zeros((n_ex, n_ex))
for _i in range(n_ex):
    _idx_i  = list_all_CSFs_ex_space[_i][1]
    _coef_i = list_all_CSFs_ex_space[_i][2]
    _sd_map = {}
    for k, c in zip(_idx_i, _coef_i):
        _sd_map[int(k)] = _sd_map.get(int(k), 0.0) + float(c)
    for _j in range(_i, n_ex):
        _idx_j  = list_all_CSFs_ex_space[_j][1]
        _coef_j = list_all_CSFs_ex_space[_j][2]
        _ov = sum(_sd_map[int(k)] * float(c) for k, c in zip(_idx_j, _coef_j)
                  if int(k) in _sd_map)
        S_ov[_i, _j] = S_ov[_j, _i] = _ov

print('\n===  Full ex space overlap matrix check ===')
print(f'  diagonal range  : [{S_ov.diagonal().min():.6f}, {S_ov.diagonal().max():.6f}]  (all should be 1.0)')
_off = S_ov - np.eye(n_ex)
print(f'  max |off-diag|  : {np.abs(_off).max():.6f}  (all should be 0.0)')
_evals = np.linalg.eigvalsh(S_ov)
print(f'  S eigenvalues   : min={_evals.min():.6f}  max={_evals.max():.6f}  (all should be 1.0)')
_n_zero = int(np.sum(_evals < 1e-6))
if _n_zero:
    print(f'  WARNING: {_n_zero} near-zero eigenvalue(s) → linearly dependent / repeated states')
def _sd_pair_moves(sd_occ, ref_occ):
    """Spatial-orbital pair moves turning ref_occ into sd_occ: (sources, destinations)."""
    n_sp = len(ref_occ) // 2
    occ_s = [int(sd_occ[2 * o]) + int(sd_occ[2 * o + 1]) for o in range(n_sp)]
    ref_s = [int(ref_occ[2 * o]) + int(ref_occ[2 * o + 1]) for o in range(n_sp)]
    src = [o for o in range(n_sp) if ref_s[o] > occ_s[o]]
    dst = [o for o in range(n_sp) if occ_s[o] > ref_s[o]]
    return src, dst

_thr = 0.01
_n_nonorth_fullex = 0
for _i in range(n_ex):
    for _j in range(_i + 1, n_ex):
        if abs(S_ov[_i, _j]) > _thr:
            _n_nonorth_fullex += 1
            _gi = _state_to_group.get(_i)
            _gj = _state_to_group.get(_j)
            _sds_i = [[int(x) for x in sd] for sd in list_all_CSFs_ex_space[_i][0]]
            _sds_j = [[int(x) for x in sd] for sd in list_all_CSFs_ex_space[_j][0]]
            print(f'  NON-ORTH [{_i},{_j}] S={S_ov[_i,_j]:.6f}'
                  f'  (state {_i} ← group {_gi}, state {_j} ← group {_gj})')
            # conflicting SDs: shared SD indices whose coefficient products
            # build up the nonzero overlap
            _mi_c, _mj_c = {}, {}
            for k, c in zip(list_all_CSFs_ex_space[_i][1], list_all_CSFs_ex_space[_i][2]):
                _mi_c[int(k)] = _mi_c.get(int(k), 0.0) + float(c)
            for k, c in zip(list_all_CSFs_ex_space[_j][1], list_all_CSFs_ex_space[_j][2]):
                _mj_c[int(k)] = _mj_c.get(int(k), 0.0) + float(c)
            _sd_occ_c = {}
            for _st_c in (list_all_CSFs_ex_space[_i], list_all_CSFs_ex_space[_j]):
                for _a_c, _k_c in zip(_st_c[0], _st_c[1]):
                    _sd_occ_c[int(_k_c)] = _a_c
            print(f'    Conflicting SDs (shared between state {_i} and state {_j}):')
            for _k_c in sorted(set(_mi_c) & set(_mj_c)):
                _prod = _mi_c[_k_c] * _mj_c[_k_c]
                if abs(_prod) < 1e-10:
                    continue
                print(f'      idx={_k_c}  c_i={_mi_c[_k_c]:+.4f}  c_j={_mj_c[_k_c]:+.4f}  '
                      f'c_i*c_j={_prod:+.6f}  occ={[int(x) for x in _sd_occ_c[_k_c]]}')
            for _k_idx, _k_grp, _sds_k, _csf_k in (
                    (_i, _gi, _sds_i, list_all_CSFs_ex_space[_i]),
                    (_j, _gj, _sds_j, list_all_CSFs_ex_space[_j])):
                _sk = sorted(int(x) for x in _csf_k[1])
                print(f'    state[{_k_idx}] (group {_k_grp}): SD-idx={_sk}  coefs={np.round(_csf_k[2],4).tolist()}')
                for _sd, _c in zip(_sds_k, _csf_k[2]):
                    print(f'      coef={_c:+.4f}  SD={_sd}')
                if _k_grp is not None:
                    _bref = list_list_symadapted_ex_space[_k_grp][0]
                    _bref_idx  = sorted(int(x) for x in _bref[1])
                    _bref_sds  = [[int(x) for x in sd] for sd in _bref[0]]
                    print(f'    baserefCSF[group {_k_grp}]: SD-idx={_bref_idx}  coefs={np.round(_bref[2],4).tolist()}')
                    for _sd, _c in zip(_bref_sds, _bref[2]):
                        print(f'      coef={_c:+.4f}  SD={_sd}')
                    _slot_info = _state_to_slot.get(_k_idx)
                    if _slot_info is not None:
                        _, _slot = _slot_info
                        if _slot == 0:
                            _ia_label = 'refCSF (slot 0)'
                        else:
                            _ia_slot = _slot - 1   # slot 0 = refCSF, slots 1+ = generators
                            _ia_list = list_list_ia[_k_grp] if _k_grp < len(list_list_ia) else []
                            if _ia_slot < len(_ia_list):
                                _ia_label = f'generator slot {_slot}: {_ia_list[_ia_slot]}'
                            else:
                                _ia_label = (f'slot {_slot}: chain/extension state '
                                             f'(see creation trace below)')
                        print(f'    generator[group {_k_grp}]: {_ia_label}')
                    # creation trace: orbital pair moves of each SD relative to the
                    # group's base reference — this IS the generator chain that
                    # created the SD, regardless of slot bookkeeping.
                    _ref_sds_tr = [[int(x) for x in sd] for sd in _bref[0]]
                    print(f'    creation trace (pair moves vs base refCSF of group {_k_grp}):')
                    for _sd_tr, _c_tr in zip(_sds_k, _csf_k[2]):
                        _best_tr = None
                        for _rsd_tr in _ref_sds_tr:
                            _src_tr, _dst_tr = _sd_pair_moves(_sd_tr, _rsd_tr)
                            if _best_tr is None or len(_src_tr) < len(_best_tr[0]):
                                _best_tr = (_src_tr, _dst_tr)
                        _src_tr, _dst_tr = _best_tr
                        if not _src_tr:
                            print(f'      coef={_c_tr:+.4f}: = base reference SD (no pair moves)')
                        else:
                            print(f'      coef={_c_tr:+.4f}: l={len(_src_tr)} chain, '
                                  f'pairs moved from orbitals {_src_tr} to {_dst_tr}')
if _n_nonorth_fullex:
    print(f'\n  {_n_nonorth_fullex} non-orthogonal pair(s) in the full ex space — '
          f'the H eigensolve below assumes an orthonormal basis. Bombing out!')
    sys.exit()
print('=== end overlap check ===\n')

#Hmat_full_space, list_list_rdm1, list_list_rdm2 =  construct_Hmat_CSFs(list_all_CSFs_ex_space,Enuc,obt,tbt,lrdm=True)
# The RDMs from this call were never used — the orbital-optimization loop below
# (and the UCSF section later) rebuild list_list_rdm1/list_list_rdm2 from scratch
Hmat_full_space = construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space,Enuc,obt,tbt,nparal=nparal)
print(f'\nDimension of the unique accumulated excited CSF space: {len(list_all_CSFs_ex_space)}')
print(f'\nAccumulated excited CSF space: {list_all_CSFs_ex_space}')

print(f'\nAccumulated excited space CSF 0: {list_list_symadapted_ex_space[0]}')
#print('\nHamiltonian matrix of the whole ex space of kept CSFs')
#print_matrix(Hmat_full_space)
Hmat_full_space_sparse = csr_matrix(Hmat_full_space)
E_GS_full, psi_GS_full = get_ground_state(Hmat_full_space_sparse)
Energies = get_lowest_n_eigen(Hmat_full_space_sparse, no_states)
Evecs_full = get_lowest_n_eigvec(Hmat_full_space_sparse, no_states)
print(f'\nEnergies of the full ex space of all CSFs: {Energies}')
print(f'\nEigenstates:')
small = 0.001
for j in range(no_states):
    print(f'State {j}:')
    for ii in range(len(Evecs_full[j])):
        if abs(Evecs_full[j][ii]) > small: print(ii,Evecs_full[j][ii])

print(f'Dimension of ex space: {len(list_all_CSFs_ex_space)}')
print(f'E_GS_full: {E_GS_full}')

if l_opt_orb and not l_dooh_sym:
    print('\nOrbital optimization starts')

    if not l_initial_orb_rot:
        x_orbrot_0 = np.zeros(len(list_orb_rot))

    improve = 100
    nround = -1

    nroots = no_states
    weights = np.array([1/no_states] * no_states)

    E_avg_prev = None

    while abs(improve) > orbopt_conv_thrsh:
        nround += 1

        Hmat_UCSF_opt, list_list_rdm1, list_list_rdm2 = construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space,Enuc, obt, tbt, nparal,lrdm=True)

        evals = get_lowest_n_eigen(Hmat_UCSF_opt, nroots)
        E_avg = np.dot(weights, evals)

        if E_avg_prev is None:
            E_avg_prev = E_avg

        x_orbrot, _, obt_opt, tbt_opt = opt_orbitals_for_weighted_n_roots_of_CSF_space(
                list_all_CSFs_ex_space, nroots, Enuc, obt_spatial, tbt_spatial, list_orb_rot,
                list_list_rdm1, list_list_rdm2, l_dooh_sym, list_degmo, x_orbrot_0, nparal,
                eps=Uopt_thrsh, normalize_weights=True, debug=False,
                linked_orbrot_pairs=linked_orbrot_pairs if l_dooh_sym else None)

        obt = obt_opt
        tbt = tbt_opt
        x_orbrot_0 = x_orbrot

        Hmat_after = construct_Hmat_CSFs_paral_triu(
            list_all_CSFs_ex_space,
            Enuc, obt, tbt, nparal, lrdm=False
        )

        evals_after = get_lowest_n_eigen(Hmat_after, nroots)
        E_avg_new = np.dot(weights, evals_after)

        improve = abs(E_avg_new - E_avg_prev)
        print(f'Round {nround}: ΔE_avg = {improve}')
        print('Changed E: ', evals_after)

        E_avg_prev = E_avg_new

    print('Orbital optimization converged')

    obt = obt_opt
    tbt = tbt_opt
    x_orbrot_0 = x_orbrot

else:
    x_orbrot = np.zeros(len(list_orb_rot))

Hmat_full_space = construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space, Enuc, obt, tbt, nparal, lrdm=False)
Energies = get_lowest_n_eigen(Hmat_full_space, no_states)
print(f'\nEnergies of the full ex space before U optimization and after orbital rotation: {Energies}')

E_GS_full, psi_GS_full = get_ground_state(csr_matrix(Hmat_full_space))
print(f'E_GS_full after orbital rotation: {E_GS_full}')

print('\n=== Orbital degeneracy diagnostic (post-rotation) ===')
_psi2 = np.array(psi_GS_full) ** 2          # |c_i|² for each CSF
_rdm_occ = np.zeros(obt.shape[0])           # spin-orbital occupations
for _i, _csf in enumerate(list_all_CSFs_ex_space):
    _sds, _coefs = _csf[0], _csf[2]
    for _sd, _c in zip(_sds, _coefs):
        for _s in range(len(_sd)):
            _rdm_occ[_s] += _psi2[_i] * float(_c)**2 * float(_sd[_s])

print(f'  {"Group":<12} {"orb_p":>6} {"orb_q":>6}  '
      f'{"ε(p)":>12} {"ε(q)":>12} {"Δε":>10}  '
      f'{"mix|obt[2p,2q]|":>16}  '
      f'{"occ(p)":>10} {"occ(q)":>10} {"Δocc":>10}  status')
for _grp in dinfh_degenerate_groups:
    _p, _q = _grp[0], _grp[1]
    _ep    = obt[2*_p, 2*_p]
    _eq    = obt[2*_q, 2*_q]
    _de    = abs(_ep - _eq)
    _mix   = abs(obt[2*_p, 2*_q])
    # occupation = alpha + beta for each spatial orbital
    _occ_p = _rdm_occ[2*_p] + _rdm_occ[2*_p+1]
    _occ_q = _rdm_occ[2*_q] + _rdm_occ[2*_q+1]
    _docc  = abs(_occ_p - _occ_q)
    _ok    = _de < 1e-6 and _mix < 1e-6 and _docc < 1e-4
    _flag  = 'OK' if _ok else 'BROKEN'
    print(f'  {mo_sym[_p]}/{mo_sym[_q]:<8} {_p:>6} {_q:>6}  '
          f'{_ep:>12.8f} {_eq:>12.8f} {_de:>10.2e}  '
          f'{_mix:>16.2e}  '
          f'{_occ_p:>10.6f} {_occ_q:>10.6f} {_docc:>10.2e}  {_flag}')
print('=== end degeneracy diagnostic ===\n')

# #Get the symmetry-adapted basis vectors for all CSFs in the excited space
# if l_axial_sym or l_dooh_sym:
#     list_ECSF_all = []
#     ndim_full = len(list_all_CSFs_ex_space)
#     for iCSF in range(ndim_full):
#         list_ECSF_all.append(Hmat_full_space[iCSF,iCSF])
#
#     list_sym_CSF_vec_all_space = []
#     CSF_already_considered = []
#     degen_thrsh = 1.0e-12
#     for iCSF in range(ndim_full):
#         if iCSF in CSF_already_considered: continue
#         CSF_already_considered.append(iCSF)
#         list_degen_pair = []
#         for jCSF in range(iCSF+1,ndim_full):
#             if abs(list_ECSF_all[iCSF] - list_ECSF_all[jCSF]) < degen_thrsh and np.isclose(abs(psi_GS_full[iCSF]),abs(psi_GS_full[jCSF])):
#                 list_degen_pair.append(jCSF)
#         sym_CSF_vec = np.zeros(ndim_full)
#         if len(list_degen_pair) == 0:
#             sym_CSF_vec[iCSF] = 1.0
#             list_sym_CSF_vec_all_space.append(sym_CSF_vec)
#        #elif len(list_degen_pair) > 1:
#        #    print(f'More than one CSF is found dgenerate with CSF{iCSF} with {list_ECSF_all[iCSF], psi_GS_full[iCSF]}')
#        #    print(list_all_CSFs_ex_space[iCSF])
#        #    for jCSF in list_degen_pair:
#        #        print(f'CSF{jCSF}, {list_ECSF_all[jCSF], psi_GS_full[jCSF]}')
#        #        print(list_all_CSFs_ex_space[jCSF])
#        #    print('Bombing out')
#        #    sys.exit()
#         else:
#             sym_CSF_vec[iCSF] = psi_GS_full[iCSF]
#             for jCSF in list_degen_pair:
#                 sym_CSF_vec[jCSF] = psi_GS_full[jCSF]
#                 CSF_already_considered.append(jCSF)
#             norm_vec = np.linalg.norm(sym_CSF_vec)
#             sym_CSF_vec = sym_CSF_vec / norm_vec
#             list_sym_CSF_vec_all_space.append(sym_CSF_vec)
#
#     n_sym_CSF_vec_total = len(list_sym_CSF_vec_all_space)
#     print(f'\nTotal number of sym-adapted CSFs in the whole ex space: {n_sym_CSF_vec_total}')
#     Umat_sym_adapted = np.zeros([ndim_full,n_sym_CSF_vec_total])
#     for icol in range(n_sym_CSF_vec_total):
#         Umat_sym_adapted[:,icol] = list_sym_CSF_vec_all_space[icol]
#
#     mat_should_be_1 = Umat_sym_adapted.transpose()@Umat_sym_adapted
#     resid_mat = mat_should_be_1 - np.eye(n_sym_CSF_vec_total)
#     assert np.isclose(np.linalg.norm(resid_mat),0.0)
#    #print('\nIndices of non-zero elements in resid_mat:')
#    #print(np.where(abs(resid_mat) > 1.0e-6 ))
#     Hmat_CSF_sym_adapted = Umat_sym_adapted.transpose()@Hmat_full_space@Umat_sym_adapted
#     Hmat_CSF_sym_sparse = csr_matrix(Hmat_CSF_sym_adapted)
#     E_GS_sym_adapted, psi_GS_sym_adapted = get_ground_state(Hmat_CSF_sym_sparse)
#     if abs(E_GS_sym_adapted - E_GS_full) > degen_thrsh*100:
#         print(f'Too large difference between E_GS using sym-adapted CSFs or not: {E_GS_sym_adapted,E_GS_full}')
#         print('Bombing out!')
#         sys.exit()

#Get the external rotational amplitudes one by one as the initial guess for later
#get_ext_ampld_1by1(list_CSF,list_SOMO_DMO,list_sym_sign,list_list_ia_internal,list_list_ia_external,l_axial_sym,nparal)

#
# if l_use_decomp_genmat:
#     list_list_genmat_to_use = list_list_decomp_genmat_noCAS
# else:
#     list_list_genmat_to_use = list_list_genmat_noCAS

#Get the mp2 amplitudes for the external ia pairs
# list_list_mp2_ampld_for_Uext = []
# for iref in range(len(list_list_ia_noCAS)):
#     print(f'Ref {iref}:')
#     list_mp2_ampld_for_Uext = []
#     for ia_pairs in list_list_ia_noCAS[iref]:
#         for pair in ia_pairs:
#             ampld_taken = sorted_mp2_ampld[sorted_ia_pair.index(pair)]
#
#         print(f'pairs: {ia_pairs}, mp2 ampld: {ampld_taken}')
#         list_mp2_ampld_for_Uext.append(ampld_taken)
#     list_list_mp2_ampld_for_Uext.append(list_mp2_ampld_for_Uext)

# Umat_full_Uext_mp2, Vmat_full_make_Uvec_full, list_list_refCSF, list_list_Uext_mp2_CSF = make_Uvec_full(list_list_ia_noCAS,list_list_mp2_ampld_for_Uext,l_use_decomp_genmat,list_list_genmat_to_use,list_UCSF_subspace_start_end,list_list_pairex_CSF_vec,list_list_pair_ex_space,debug=True)
#
# Hmat_Uext_mp2 = Umat_full_Uext_mp2.transpose()@Hmat_full_space@Umat_full_Uext_mp2
# E0_Uext_mp2, psi_Uext_mp2 = get_ground_state(csr_matrix(Hmat_Uext_mp2))
# H_mp2_sparse = csr_matrix(Hmat_Uext_mp2)
# eigenvalues = np.linalg.eigvalsh(H_mp2_sparse.toarray())
# print(f'Eigenvalues:', eigenvalues)
# print(f'\nE0_Uext_mp2: {E0_Uext_mp2}')
# print(f'psi_Uext_mp2: {psi_Uext_mp2}')

#Confirm E0_Uext_mp2 from the actual Uext mp2 CSFs
#list_all_Uext_mp2_CSF = []
#for item in list_list_Uext_mp2_CSF:
#    list_all_Uext_mp2_CSF += item
#
#Hmat_Uext_mp2_CSF = construct_Hmat_CSFs(list_all_Uext_mp2_CSF,Enuc,obt,tbt)
#E0_Uext_mp2_CSF, psi_Uext_mp2_CSF = get_ground_state(csr_matrix(Hmat_Uext_mp2_CSF))
#print(f'\nE0_Uext_mp2_CSF: {E0_Uext_mp2_CSF}')
#print(f'psi_Uext_mp2_CSF: {psi_Uext_mp2_CSF}')
#
#
# list_list_theta_opt_noCAS, list_list_Uvec_noCAS, Umat_total_noCAS, Vmat_total_inCAS = opt_U_overlap_with_noEX_in_CAS(list_list_ia_noCAS,list_list_theta_noCAS,list_list_genmat_to_use,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_list_pairex_CSF_vec,l_use_decomp_genmat,conv=1.0e-5,nparal=nparal,ldisp=True,debug=True)
#
# print(f'\nlist_list_theta_opt_noCAS')
# for ii, item in enumerate(list_list_theta_opt_noCAS):
#     print(ii,item)
#
# np.isclose(np.sum(abs(Vmat_full_make_Uvec_full - Vmat_total_inCAS)),0.0)
# print(f'The Vmat_full from make_Uvec_full and opt_U_overlap_with_noEX_in_CAS are consistent')
#
# Hmat_inCAS = Vmat_total_inCAS.transpose()@Hmat_full_space@Vmat_total_inCAS
# E0_inCAS, psi_GS_inCAS = get_ground_state(csr_matrix(Hmat_inCAS))
# print(f'\nE0 of in CAS basis: {E0_inCAS}')
# print(f'psi_GS_inCAS: {psi_GS_inCAS}')
#
# print(Umat_total_noCAS.shape,Hmat_full_space.shape)
# Hmat_UnoCAS = Umat_total_noCAS.transpose()@Hmat_full_space@Umat_total_noCAS
# print('\nHamiltonian matrix of the U states without CAS pair excitation\n')
# print_matrix(Hmat_UnoCAS)
#
# E0_UnoCAS, psi_GS_UnoCAS = get_ground_state(csr_matrix(Hmat_UnoCAS))
# print(f'E0 of UnoCAS basis: {E0_UnoCAS}, diff. from E0 of full space by {E0_UnoCAS - E_GS_full}')
# print(f'psi_GS: {psi_GS_UnoCAS}')
#
# #Calculate overlap between the ground state of full ex space and the ground state of the reduced space
# #without pair excitations within CAS
# print(f'\nOverlap between ground state of full ex space and noCASex space {psi_GS_full.transpose()@Umat_total_noCAS@psi_GS_UnoCAS}')
# #Overlap = 0.0
# #n_Uvec_total = Umat_total_noCAS.shape[1]
# #for ivec in range(n_Uvec_total):
# #    np.dot(Umat_total_noCAS[:,ivec],)
#
# print(f'Optimized thetas for Uext excitations:')
# list_list_Uext_mp2_ampld = []
# list_list_Uext_opt_ampld = []
# for irefCSF in range(len(list_list_theta_opt_noCAS)):
#     list_Uext_mp2_ampld = []
#     list_Uext_opt_ampld = []
#     list_theta_opt_noCAS = list_list_theta_opt_noCAS[irefCSF]
#     list_ia_noCAS = list_list_ia_noCAS[irefCSF]
#     if len(list_theta_opt_noCAS) == 0:
#         list_list_Uext_mp2_ampld.append(list_Uext_mp2_ampld)
#         list_list_Uext_opt_ampld.append(list_Uext_opt_ampld)
#         continue
#     n_train_U = len(list_theta_opt_noCAS) // len(list_ia_noCAS)
#     print(f'There are {n_train_U} trains of U')
#     for i_train in range(n_train_U):
#         print(f'\nTrain {i_train+1:}, iCSF_group: {irefCSF}')
#         for ipairs in range(len(list_ia_noCAS)):
#             pairs = list_ia_noCAS[ipairs]
#             theta_opt = list_theta_opt_noCAS[i_train*len(list_ia_noCAS)+ipairs]
#             mp2_idx = group_mp2_iapair.index(pairs)
#             print(f'pairs: {pairs}, Opt ampld: {theta_opt}, MP2 ampld: {group_mp2_ampld[mp2_idx]}')
#             if i_train == 0:
#                 list_Uext_mp2_ampld.append([pairs,group_mp2_ampld[mp2_idx]])
#             list_Uext_opt_ampld.append([pairs,theta_opt])
#
#     list_list_Uext_mp2_ampld.append(list_Uext_mp2_ampld)
#     list_list_Uext_opt_ampld.append(list_Uext_opt_ampld)
#     print(f'Updated {irefCSF}-th Uext amplds. {len(list_list_Uext_mp2_ampld),len(list_list_Uext_opt_ampld)}')
#
# #assert len(list_list_Uext_mp2_ampld) == len(list_list_refCSF)
# #assert len(list_list_Uext_opt_ampld) == len(list_list_refCSF)
# if len(list_list_Uext_mp2_ampld) != len(list_list_refCSF):
#     print(f'length of list_list_Uext_mp2_ampld: {len(list_list_Uext_mp2_ampld)}')
#     print(f'length of list_list_refCSF: {len(list_list_refCSF)}')
#     print('Inconsistent. Bombing out!')
#     sys.exit()
# if len(list_list_Uext_opt_ampld) != len(list_list_refCSF):
#     print(f'length of list_list_Uext_opt_ampld: {len(list_list_Uext_opt_ampld)}')
#     print(f'length of list_list_refCSF: {len(list_list_refCSF)}')
#     print('Inconsistent. Bombing out!')
#     sys.exit()
#
# for iref in range(len(list_list_Uext_mp2_ampld)):
#     list_refCSF = list_list_refCSF[iref]
#     list_Uext_mp2_ampld = list_list_Uext_mp2_ampld[iref]
#     list_Uext_opt_ampld = list_list_Uext_opt_ampld[iref]
#     print(f'MP2 amplitudes for external excitations for reference group {iref}:')
#     for item in list_Uext_mp2_ampld:
#         print(item)
#     print(f'Optimized amplitudes for external excitations for reference group {iref}:')
#     for item in list_Uext_opt_ampld:
#         print(item)
#     print(f'The following CSFs have the same SOMOs and same singlet coupling pathway. They share the external amplitudes above')
#     for item in list_refCSF:
#         print(item)
#
# print('list of refCSF')
# print(list_refCSF)
#
# save_filename = 'Uext_CSF_for_Praveen_Smik_' + str(rdist) + '.dump'
# with open(save_filename, 'wb') as f:
#     pickle.dump([list_list_refCSF,list_list_Uext_mp2_CSF,list_list_Uext_mp2_ampld,list_list_Uext_opt_ampld,\
#       list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)
#
# #save the input data for opt_U_overlap, which usually takes long time.
# save_filename = 'CSF_UCSF_GS_before_opt_U_overlap' + str(rdist) + '.dump'
# with open(save_filename, 'wb') as f:
#     pickle.dump([list_list_ia,list_list_theta,list_list_genmat,list_list_decomp_genmat,list_UCSF_subspace_start_end,\
#       psi_GS_full,Uopt_thrsh,list_sym_CSF_vec],f)

#list_list_theta_opt, list_Uvec_opt = opt_U_overlap(list_list_ia,list_list_theta,list_list_genmat,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_sym_CSF_vec,l_use_decomp_genmat=False,ldisp=True,debug=True)
#if l_use_decomp_genmat:
    #list_list_theta_opt, list_Uvec_opt, list_Umat_opt = opt_U_overlap(list_list_ia,list_list_theta,list_list_decomp_genmat,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_sym_CSF_vec,nparal,l_use_decomp_genmat,ldisp=True,debug=True)
#else:
    #list_list_theta_opt, list_Uvec_opt, list_Umat_opt = opt_U_overlap(list_list_ia,list_list_theta,list_list_genmat,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_sym_CSF_vec,nparal,l_use_decomp_genmat,ldisp=True,debug=True)

#Ensure consistency among list_Umat_opt, list_Uvec_opt, and list_sym_CSF_vec
#for isymCSF in range(len(list_sym_CSF_vec)):
    #temp_norm = np.linalg.norm(list_Uvec_opt[isymCSF]-list_Umat_opt[isymCSF]@list_sym_CSF_vec[isymCSF])
    #assert np.isclose(temp_norm,0.0)

list_list_theta_opt = [[0.0] * len(ia) for ia in list_list_ia]

# Build frozen_theta_map: {(group_idx, theta_pos): π/4}
# Ref-degenerate pairs have θ=π/4 exactly (tan(2θ)→∞ when ΔE=0), so no optimization needed.
frozen_theta_map = {}
if 'list_list_frozen_positions' in dir():
    for _g, _frozen_pos in enumerate(list_list_frozen_positions):
        for _pos in _frozen_pos:
            if _pos >= len(list_list_theta_opt[_g]):
                # Position out of range → this pair is already treated explicitly; skip
                print(f'  [frozen-theta] group {_g} pos {_pos} out of range '
                      f'(len={len(list_list_theta_opt[_g])}), pair treated explicitly')
                continue
            frozen_theta_map[(_g, _pos)] = float(np.pi / 4)
            list_list_theta_opt[_g][_pos] = float(np.pi / 4)   # warm-start at π/4
if frozen_theta_map:
    print(f'\n[frozen-theta] Freezing {len(frozen_theta_map)} θ parameters at π/4:')
    for (g, pos), val in sorted(frozen_theta_map.items()):
        print(f'  group {g}, position {pos}: θ = {val:.6f}')

print("Opt Theta list before optimizing: ", list_list_theta_opt)
print(sum(len(list_list_theta) for list_list_theta in list_list_theta_opt))

if l_use_decomp_genmat:
    list_list_genmat_to_use = copy.deepcopy(list_list_decomp_genmat)
    list_list_theta_opt, list_Uvec_opt, list_Umat_opt, list_UCSF_opt = opt_multi_train_UCSF(list_list_pair_ex_space,
    list_list_theta_opt,list_list_genmat_to_use,l_use_decomp_genmat, Enuc, obt, tbt, nparal, list_sym_CSF_vec, shared_unitary_groups, no_states, list_list_extra_vecs=list_list_extra_vecs,
    frozen_theta_map=frozen_theta_map)
else:
    list_list_genmat_to_use = copy.deepcopy(list_list_genmat)
    list_list_theta_opt, list_Uvec_opt, list_Umat_opt, list_UCSF_opt = opt_multi_train_UCSF(list_list_pair_ex_space,
    list_list_theta_opt,list_list_genmat_to_use,l_use_decomp_genmat, Enuc, obt, tbt, nparal, list_sym_CSF_vec, shared_unitary_groups, no_states, list_list_extra_vecs=list_list_extra_vecs,
    frozen_theta_map=frozen_theta_map)

print("Opt Theta list after optimizing: ", list_list_theta_opt)
print(sum(len(list_list_theta) for list_list_theta in list_list_theta_opt))

# list_UCSF = []
# list_CSF_ref = []
# for iCSF_group in range(len(list_list_pair_ex_space)):
#     list_theta = list_list_theta_opt[iCSF_group]
#     print(f'iCSF_group {iCSF_group}, {len(list_theta), len(list_list_genmat[iCSF_group])}')
#     list_pair_ex_space = list_list_pair_ex_space[iCSF_group]
#     Uvec = list_Uvec_opt[iCSF_group]
#     [istart,iend] = list_UCSF_subspace_start_end[iCSF_group]
#     refvec = copy.deepcopy(psi_GS_full[istart:iend])
#     refvec = refvec / np.linalg.norm(refvec)
#     UCSF = make_UCSF_state(list_pair_ex_space,Uvec)
#     list_UCSF.append(UCSF)
#     refCSF = make_UCSF_state(list_pair_ex_space,refvec)
#     list_CSF_ref.append(refCSF)
#     n_unique_ia_pair = len(list_list_genmat[iCSF_group])
#     if n_unique_ia_pair == 0:
#         print(f'\nNo ia pairs for CSF group {iCSF_group}')
#         continue
#     print(f'\nia pairs of UCSF{iCSF_group}:')
#     n_U_trains = len(list_theta) // n_unique_ia_pair
#     if n_U_trains != 1:
#         list_list_genmat_to_use[iCSF_group] *= n_U_trains
#     print(f'# of U trains: {n_U_trains}')
#     for i_train in range(n_U_trains):
#         for ipair in range(n_unique_ia_pair):
#             mp2_idx = group_mp2_iapair.index(list_list_ia[iCSF_group][ipair])
#             print(f'pairs: {list_list_ia[iCSF_group][ipair]}, Opt ampld: {list_theta[ipair+i_train*n_unique_ia_pair]},\
#               MP2 ampld: {group_mp2_ampld[mp2_idx]}')
#         print()

print(len(list_UCSF))

list_UCSF = list_UCSF_opt

if l_axial_sym:
    assert len(list_UCSF) == len(list_sym_CSF)

#Prepare CSFs and UCSFs for Smik and Praveen
if l_axial_sym:
    list_UCSF_sym_components = []
    list_list_theta_opt_CSF_comp = []
    list_list_ia_CSF_comp = []
    for iCSF_group in range(len(list_list_pair_ex_space)):
        list_pair_ex_space = list_list_pair_ex_space[iCSF_group]
        ndim_iCSF_group = len(list_pair_ex_space)
        sym_CSF_vec = list_sym_CSF_vec[iCSF_group]
        list_non0_ind = np.where(sym_CSF_vec != 0.0)[0]
        if len(list_non0_ind) == 1:
            list_UCSF_sym_components.append(list_UCSF[iCSF_group])
            list_list_ia_CSF_comp.append(list_list_ia[iCSF_group])
            list_list_theta_opt_CSF_comp.append(list_list_theta_opt[iCSF_group])
        else:
            for ind in list_non0_ind:
                comp_CSF_vec = np.zeros(ndim_iCSF_group)
                comp_CSF_vec[ind] = 1.0
                UCSF_comp_vec = list_Umat_opt[iCSF_group]@comp_CSF_vec
                UCSF_comp = make_UCSF_state(list_pair_ex_space,UCSF_comp_vec)
                list_UCSF_sym_components.append(UCSF_comp)
            list_list_ia_CSF_comp.append(list_list_ia[iCSF_group])
            list_list_ia_CSF_comp.append(list_list_ia[iCSF_group])
            list_list_theta_opt_CSF_comp.append(list_list_theta_opt[iCSF_group])
            list_list_theta_opt_CSF_comp.append(list_list_theta_opt[iCSF_group])

else:
    list_UCSF_sym_components = list_UCSF
    list_list_theta_opt_CSF_comp = list_list_theta_opt
    list_list_ia_CSF_comp = list_list_ia

print(list_list_theta_opt)

#Hmat_refCSF = construct_Hmat_CSFs(list_CSF_ref,Enuc,obt,tbt)
# Hmat_refCSF = construct_Hmat_CSFs_paral_triu(list_CSF_ref,Enuc,obt,tbt,nparal)
# print('\nHmat of ref. CSF')
# print_matrix(Hmat_refCSF)
# Hmat_refCSF_sparse = csr_matrix(Hmat_refCSF)
# E_GS_refCSF,psi_GS_refCSF = get_ground_state(Hmat_refCSF_sparse)
#print(f'\nE0 of ref CSF space: {E_GS_refCSF}')
#print('\nGround state:')
#print(psi_GS_refCSF)
#assert np.isclose(E_GS_refCSF,E_GS_full)

Hmat_CSF = construct_Hmat_CSFs_paral_triu(list_CSF,Enuc,obt,tbt,nparal)
print('\nHmat of CSF')
print_matrix(Hmat_CSF)
Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
eigval, eigvec = scipy.sparse.linalg.eigsh(Hmat_CSF_sparse, k=no_states, which="SA")
idx = np.argsort(eigval)
eigval = eigval[idx]
eigvec = eigvec[:, idx]
print(f"\nComputed {no_states} lowest eigenpairs of CSF (which='SA'):")
print_eigen_solution(eigval, eigvec)

print('UCSF list')
print(len(list_UCSF))

Hmat_UCSF = construct_Hmat_CSFs_paral_triu(list_UCSF,Enuc,obt,tbt,nparal)
print('\nHmat of UCSF')
print_matrix(Hmat_UCSF)
Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF)

E_GS_UCSF, psi_GS_UCSF = get_ground_state(Hmat_UCSF_sparse)
print(f'\nE0 of UCSF space of all CSFs: {E_GS_UCSF}')
print(f'Error from U opt: {E_GS_UCSF - E_GS_full}')
print('\nGround state:')
print(psi_GS_UCSF)

print('\nReference ground state:')
print(psi_GS_full)

print('\n' + '='*72)
print('WAVEFUNCTION RECONCILIATION: UCSF vs full-ex CSF space')
print('='*72)

def _expand_wfn_to_SD(basis_states, coeffs):
    """
    Expand a wavefunction  Ψ = Σ_i coeffs[i] * basis_states[i]
    into a flat dict  {SD_key: total_coefficient}.
    Each basis state is [onlist, idx_list, coef_vec].
    SD_key = tuple of int occupations (the onlist entry).
    """
    sd_dict = {}
    for i, state in enumerate(basis_states):
        c_state = float(coeffs[i])
        if abs(c_state) < 1e-15:
            continue
        onlist, _, coef_vec = state[0], state[1], state[2]
        for sd, c_sd in zip(onlist, coef_vec):
            key = tuple(int(x) for x in sd)
            sd_dict[key] = sd_dict.get(key, 0.0) + c_state * float(c_sd)
    return sd_dict

# All no_states eigenvectors of the UCSF Hamiltonian (full-ex ones are in Evecs_full)
_eigval_ucsf_rec, _eigvec_ucsf_rec = scipy.sparse.linalg.eigsh(
    Hmat_UCSF_sparse, k=no_states, which="SA")
_order_rec = np.argsort(_eigval_ucsf_rec)
_eigval_ucsf_rec = _eigval_ucsf_rec[_order_rec]
_eigvec_ucsf_rec = _eigvec_ucsf_rec[:, _order_rec]

def _sd_coef_in_psi(psi_vec, ex_space, target_key):
    """Sum psi[i] * c_sd over all basis CSFs where target_key appears."""
    total = 0.0
    for _bi, _bcsf in enumerate(ex_space):
        if _bi >= len(psi_vec): break
        _c_psi = float(psi_vec[_bi])
        if abs(_c_psi) < 1e-14: continue
        for _sd, _csd in zip(_bcsf[0], _bcsf[2]):
            if tuple(int(x) for x in _sd) == target_key:
                total += _c_psi * float(_csd)
    return total

# ── State-assignment overlap matrix ──────────────────────────────────────────
# <full-ex_m | UCSF_n> over the SD basis.  Diagonal ≈ ±1: states match one-to-
# one.  A large off-diagonal entry means the UCSF spectrum reordered or mixed
# the states; a row with NO large entry means the full-ex state m is simply
# not representable in the UCSF span (span deficiency).
_sd_full_all = [_expand_wfn_to_SD(list_all_CSFs_ex_space, Evecs_full[m])
                for m in range(no_states)]
_sd_ucsf_all = [_expand_wfn_to_SD(list_UCSF, _eigvec_ucsf_rec[:, n])
                for n in range(no_states)]
print('\n=== State-assignment overlap matrix <full-ex_m | UCSF_n> ===')
print('            ' + '  '.join(f' UCSF {n}  ' for n in range(no_states)))
for m in range(no_states):
    _row = []
    for n in range(no_states):
        _ov_mn = sum(_sd_full_all[m][k] * _sd_ucsf_all[n].get(k, 0.0)
                     for k in _sd_full_all[m])
        _row.append(_ov_mn)
    print(f'  full-ex {m}: ' + '  '.join(f'{v:+.4f}' for v in _row))
    _best_n = int(np.argmax([abs(v) for v in _row]))
    if abs(_row[_best_n]) < 0.9:
        print(f'    ⚠ full-ex state {m}: best UCSF overlap is only '
              f'|{_row[_best_n]:.4f}| (column {_best_n}) — state poorly '
              f'represented in the UCSF span')
    elif _best_n != m:
        print(f'    ⚠ full-ex state {m} maps to UCSF state {_best_n} '
              f'(reordered)')

# Projection of each full-ex state onto the FULL UCSF span (all columns, not
# just the no_states lowest eigenstates).  Distinguishes:
#   fraction ≈ 1   → state is representable; it sits in HIGHER UCSF roots
#                    (reordering / theta-optimization problem)
#   fraction << 1  → state is NOT representable: span deficiency
#                    (missing extension/chain directions in the basis)
_sd_cols = [_expand_wfn_to_SD([_col], [1.0]) for _col in list_UCSF]
print(f'\n  Fraction of each full-ex state inside the full UCSF span '
      f'({len(list_UCSF)} columns):')
for m in range(no_states):
    _b_proj = [sum(_sd_full_all[m][k] * _sd_c.get(k, 0.0)
                   for k in _sd_full_all[m]) for _sd_c in _sd_cols]
    _frac = float(sum(v * v for v in _b_proj))
    if _frac > 0.95:
        _verdict = ''
    elif _frac > 0.8:
        _verdict = '   <- partially representable'
    else:
        _verdict = '   <- missing weight is OUTSIDE the span (basis deficiency)'
    print(f'    full-ex {m}: {_frac:.4f}{_verdict}')
    _big_cols = sorted(range(len(_b_proj)), key=lambda c: -abs(_b_proj[c]))[:5]
    print(f'      dominant columns: ' + ', '.join(
        f'col {c} ({_b_proj[c]:+.4f})' for c in _big_cols))

for _ist in range(no_states):
    print('\n' + '='*72)
    print(f'WAVEFUNCTION RECONCILIATION (state {_ist}): UCSF vs full-ex CSF space')
    print(f'  E_full-ex = {Energies[_ist]:.10f}   E_UCSF = {_eigval_ucsf_rec[_ist]:.10f}   '
          f'dE = {_eigval_ucsf_rec[_ist] - Energies[_ist]:+.3e}')
    _degen_with = [m for m in range(no_states)
                   if m != _ist and abs(Energies[m] - Energies[_ist]) < 1e-8]
    if _degen_with:
        print(f'  NOTE: state {_ist} is degenerate with state(s) {_degen_with} — '
              f'SD amplitudes are only defined up to a rotation within the manifold')
    print('='*72)

    _sd_full = _expand_wfn_to_SD(list_all_CSFs_ex_space, Evecs_full[_ist])
    _sd_ucsf = _expand_wfn_to_SD(list_UCSF, _eigvec_ucsf_rec[:, _ist])

    _all_sds = set(_sd_full) | set(_sd_ucsf)
    _thresh  = 5e-3   # only report SDs with |coef| > this in at least one wfn

    _dom_key = max(_sd_full, key=lambda k: abs(_sd_full[k]))
    if np.sign(_sd_full[_dom_key]) != np.sign(_sd_ucsf.get(_dom_key, 1.0)):
        for key in _sd_ucsf:
            _sd_ucsf[key] = -_sd_ucsf[key]

    print(f'\nSD basis comparison (threshold |coef| > {_thresh}):')
    print(f'{"SD occupation":<50}  {"full-ex":>10}  {"UCSF":>10}  {"diff":>10}  note')
    print('-'*100)
    _missing_from_ucsf = []
    _wrong_amplitude   = []
    for _key in sorted(_all_sds, key=lambda k: -abs(_sd_full.get(k, 0.0))):
        _c_full = _sd_full.get(_key, 0.0)
        _c_ucsf = _sd_ucsf.get(_key, 0.0)
        _diff   =_c_ucsf - _c_full
        if abs(_c_full) < _thresh and abs(_c_ucsf) < _thresh:
            continue
        _note = ''
        if abs(_c_full) > _thresh and abs(_c_ucsf) < _thresh:
            _note = '← MISSING in UCSF'
            _missing_from_ucsf.append((_key, _c_full))
        elif abs(_c_ucsf) > _thresh and abs(_c_full) < _thresh:
            _note = '← SPURIOUS in UCSF'
        elif abs(_diff) > _thresh * 0.5:
            _note = '← AMPLITUDE MISMATCH'
            _wrong_amplitude.append((_key, _c_full, _c_ucsf))
        _sd_str = '|' + ' '.join(str(x) for x in _key) + '>'
        print(f'{_sd_str:<50}  {_c_full:>10.5f}  {_c_ucsf:>10.5f}  {_diff:>10.5f}  {_note}')

    if _missing_from_ucsf:
        print(f'\n{"─"*72}')
        print(f'Generator trace for {len(_missing_from_ucsf)} SD(s) missing from UCSF '
              f'(state {_ist}):')
        for (_missing_key, _missing_coef) in _missing_from_ucsf:
            _missing_occ = list(_missing_key)
            print(f'\n  Missing SD (coef={_missing_coef:.4f}): |{"".join(str(x) for x in _missing_occ)}>')
            # First: is the SD carried by any UCSF COLUMN (after rotation)?
            # This distinguishes span deficiency (SD in no column → CI cannot
            # reach it) from CI choice (SD in a column whose CI coefficient
            # is small for this state).
            _in_cols = []
            for _c, _ucsf_col in enumerate(list_UCSF):
                for _sd_c, _csd_c in zip(_ucsf_col[0], _ucsf_col[2]):
                    if tuple(int(x) for x in _sd_c) == tuple(_missing_key):
                        _in_cols.append((_c, float(_csd_c)))
                        break
            if _in_cols:
                print(f'    → SD carried by UCSF column(s):')
                for _c, _v in _in_cols:
                    _ci_c = float(_eigvec_ucsf_rec[_c, _ist])
                    print(f'        col {_c}: coef-in-column={_v:+.5f}  '
                          f'CI coef (state {_ist})={_ci_c:+.5f}  '
                          f'contribution={_v * _ci_c:+.6f}')
                print(f'      → direction EXISTS in the span; the CI chose not '
                      f'to use it (check H couplings / theta of those columns)')
            else:
                print(f'    → SD is in NO UCSF column: span deficiency — '
                      f'no CI coefficient can recover it for any state')
            # Search all UCSF excitation spaces for this SD
            _found_in = []
            for _g, _ex_space in enumerate(list_list_pair_ex_space):
                for _cidx, _csf in enumerate(_ex_space):
                    _occ_list = _csf[1]   # idx_list field
                    _sd_key_g = tuple(int(x) for x in _occ_list) if _occ_list else ()
                    if _sd_key_g == tuple(_missing_key):
                        _found_in.append((_g, _cidx))
                    # Also check via onlist
                    for _sd in _csf[0]:
                        if tuple(int(x) for x in _sd) == tuple(_missing_key):
                            if (_g, _cidx) not in _found_in:
                                _found_in.append((_g, _cidx))
            if _found_in:
                print(f'    → SD IS in ex_space at: {_found_in}  (but coefficient is zero after U-opt)')
                for (_g, _cidx) in _found_in[:3]:
                    _is_ref_pos = (_cidx == 0)
                    if _is_ref_pos:
                        print(f'      Group {_g} position 0 = REFERENCE CSF.')
                        print(f'      This SD is in the *reference* — U is rotating amplitude AWAY from it.')
                        print(f'      Diagnosis: large thetas are over-rotating the reference into excited states,')
                        print(f'      depleting SDs that only live in the reference sector.')
                    else:
                        print(f'      Group {_g} position {_cidx} = excited CSF.')
                        print(f'      Diagnosis: the generator for ia_pair at position {_cidx-1} has theta≈0,')
                        print(f'      so it is not rotating the reference into this excited CSF.')
                    if _g < len(list_list_ia) and _g < len(list_list_theta_opt):
                        _thetas_g = list_list_theta_opt[_g]
                        _ia_g     = list_list_ia[_g]
                        # ── Print the base reference CSF (the sym_CSF_vec starting state) ──
                        _sym_vec_g = list_sym_CSF_vec[_g] if _g < len(list_sym_CSF_vec) else None
                        _ex_sp_g   = list_list_pair_ex_space[_g] if _g < len(list_list_pair_ex_space) else None
                        if _sym_vec_g is not None and _ex_sp_g is not None:
                            print(f'      Group {_g} base reference (sym_CSF_vec):')
                            _sv = np.array(_sym_vec_g).flatten()
                            _nonzero_ref = [(k, float(_sv[k])) for k in range(min(len(_sv), len(_ex_sp_g)))
                                            if abs(_sv[k]) > 1e-6]
                            for _k, _wt in _nonzero_ref:
                                _ref_csf = _ex_sp_g[_k]
                                # csf[0]=onlist (occupation vectors), csf[1]=idx_list (integers), csf[2]=coef_vec
                                _sds    = _ref_csf[0]
                                _ccoefs = _ref_csf[2]
                                if len(_sds) == 1:
                                    _occ_str = ''.join(str(int(x)) for x in _sds[0])
                                    print(f'        [{_k}] wt={_wt:+.6f}  |{_occ_str}>')
                                else:
                                    print(f'        [{_k}] wt={_wt:+.6f}  (multi-SD CSF, {len(_sds)} SDs):')
                                    for _sd, _cc in zip(_sds, _ccoefs):
                                        _occ_str = ''.join(str(int(x)) for x in _sd)
                                        print(f'          coef={float(_cc):+.4f}  |{_occ_str}>')
                            if not _nonzero_ref:
                                print(f'        (all weights < 1e-6)')
                        print(f'      Group {_g} ia_pairs: {_ia_g}')
                        print(f'      Group {_g} opt thetas: {_thetas_g}')
                        # ── Step-by-step coefficient trace through U chain ──────────────
                        _genmats_g = list_list_genmat[_g] if _g < len(list_list_genmat) else None
                        if _sym_vec_g is not None and _ex_sp_g is not None and _genmats_g is not None:
                            print(f'      --- Coefficient trace through U-chain (group {_g}) ---')
                            _psi_trace = np.array(_sym_vec_g, dtype=float).flatten()
                            _c0 = _sd_coef_in_psi(_psi_trace, _ex_sp_g, tuple(_missing_key))
                            print(f'        Start (sym_CSF_vec):        coef = {_c0:+.6f}')
                            for _ki, (_theta_k, _gm_k, _ia_k) in enumerate(
                                    zip(_thetas_g, _genmats_g, _ia_g)):
                                if abs(_theta_k) < 1e-12:
                                    print(f'        After θ[{_ki}]={_theta_k:.4f} ({_ia_k}): skipped (θ≈0)')
                                    continue
                                import scipy.sparse.linalg as _spla
                                _G_k = _gm_k if scipy.sparse.issparse(_gm_k) else csr_matrix(np.asarray(_gm_k, dtype=float))
                                _psi_trace = _spla.expm_multiply(_theta_k * _G_k, _psi_trace)
                                _ck = _sd_coef_in_psi(_psi_trace, _ex_sp_g, tuple(_missing_key))
                                print(f'        After θ[{_ki}]={_theta_k:.4f} ({_ia_k}): coef = {_ck:+.6f}')
                            print(f'        Final coef = {_sd_coef_in_psi(_psi_trace, _ex_sp_g, tuple(_missing_key)):+.6f}  '
                                  f'(expected from full-CI: {_missing_coef:+.6f})')
                        else:
                            print(f'      (sym_CSF_vec / genmat not available for group {_g} — cannot trace)')
            else:
                print(f'      SD is NOT in any UCSF excitation space.')
                print(f'      Diagnosis: this SD cannot be generated by any selected ia_pair.')
                print(f'      It may be a cross-term (product of ≥2 generators via BCH expansion),')
                print(f'      or the ia_pair that generates it was not selected.')

print(f'\n{"─"*72}')
print('Optimised θ values per group:')
for _g, _thetas in enumerate(list_list_theta_opt):
    _ia = list_list_ia[_g] if _g < len(list_list_ia) else '?'
    _frozen = list_list_frozen_positions[_g] if 'list_list_frozen_positions' in dir() and _g < len(list_list_frozen_positions) else []
    _theta_str = ', '.join(
        f'θ[{_p}]={_t:.4f}{"*" if _p in _frozen else ""}'
        for _p, _t in enumerate(_thetas)
    )
    print(f'  Group {_g}: {_theta_str}   ia={_ia}')
print('  (* = frozen at π/4)')
print('='*72 + '\n')

#Ensure the sym components of UCSFs give the same ground state
Hmat_UCSF_symcomp = construct_Hmat_CSFs(list_UCSF_sym_components,Enuc,obt,tbt)
Hmat_UCSF_symcomp = construct_Hmat_CSFs_paral_triu(list_UCSF_sym_components,Enuc,obt,tbt,nparal)
Hmat_UCSF_symcomp_sparse = csr_matrix(Hmat_UCSF_symcomp)
E_GS_UCSF_symcomp, psi_GS_UCSF_symcomp = get_ground_state(Hmat_UCSF_symcomp)
assert np.isclose(E_GS_UCSF_symcomp,E_GS_UCSF)

# list_list_somo_UCSF_symcomp = []
# for UCSF_symcomp in list_UCSF_sym_components:
#     list_list_somo_UCSF_symcomp.append(get_SOMO_in_CSF(UCSF_symcomp))
#    #print(list_list_somo_UCSF_symcomp[-1])
#
list_list_somo_UCSF_symcomp = []
for csf_state in list_CSF:
    SD = csf_state[0][0]          # first Slater determinant
    orb_occ = SD[::2] + SD[1::2]  # sum alpha + beta for each orbital
    somo = [i for i in range(len(orb_occ)) if orb_occ[i] == 1]  # singly occupied = sum of 1
    list_list_somo_UCSF_symcomp.append(somo)

# save_filename = 'CSF_UCSF_GS_optU_optorb_flexibleU_matchstate_E0select_ia_' + str(rdist) + '.dump'
# if l_axial_sym:
#     with open(save_filename, 'wb') as f:
#         pickle.dump([list_CSF,list_sym_CSF_vec,list_sym_CSF,list_UCSF,list_list_ia,list_list_theta_opt,list_list_genmat,\
#           list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)
# else:
#     with open(save_filename, 'wb') as f:
#         pickle.dump([list_CSF,list_UCSF,list_list_ia,list_list_theta_opt,list_list_genmat,\
#           list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)


# save_filename = 'UCSF_sym_comp_for_Praveen_Smik_' + str(rdist) + '.dump'
# with open(save_filename, 'wb') as f:
#     pickle.dump([list_CSF,list_list_ia_CSF_comp,list_list_theta_opt_CSF_comp,list_sym_CSF_vec,list_UCSF,list_UCSF_sym_components,\
#       list_list_somo_UCSF_symcomp,psi_GS_UCSF_symcomp,list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)

# save_filename = 'data_for_opt_U_for_GS' + str(rdist) + '.dump'
# with open(save_filename, 'wb') as f:
#     pickle.dump([list_list_pair_ex_space,Hmat_full_space,list_list_genmat_to_use,l_use_decomp_genmat,list_list_theta_opt,\
#       list_UCSF_subspace_start_end,Enuc,obt,tbt,obt_spatial,tbt_spatial,list_orb_rot,l_axial_sym,list_degmo,x_orbrot],f)

Hmat_full_space_updated = Hmat_full_space
list_vec_theta_ini = list_list_theta_opt
E_old = E_GS_UCSF
x_orbrot_0 = x_orbrot

improve = 100.0
nround = 0
while(abs(improve) < orbopt_conv_thrsh):
    nround += 1
    print(f'\nRound {nround} opt for both ia amplitudes and orbitals for UCSF')
    #Opt amplitudes for E0 of UCSF. Ideally, this shall converge immediately
    list_vec_theta_opt, list_UCSF_opt = opt_U_for_GS_of_Hmat_v2(list_list_pair_ex_space,Hmat_full_space_updated,list_list_genmat_to_use,l_use_decomp_genmat,list_vec_theta_ini,list_UCSF_subspace_start_end,nparal,ldisp=True,debug=True)

    Hmat_UCSF_opt, list_list_rdm1, list_list_rdm2 = construct_Hmat_CSFs_paral_triu(list_UCSF_opt,Enuc,obt,tbt,nparal,lrdm=True)
    E_GS_UCSF_opt, psi_GS_UCSF_opt = get_ground_state(csr_matrix(Hmat_UCSF_opt))
    print(f'E_GS_UCSF_opt: {E_GS_UCSF_opt} for checking')
    if not l_opt_orb:
        print(f'This is the final energy. No orbital optimization iteration is needed')
        break

    #Opt orbitals for the UCSF from optimizing E0
    x_orbrot_UCSF, E_orbopt_UCSF, obt_opt, tbt_opt = opt_orbtials_for_GS_of_CSF_space(list_UCSF_opt,psi_GS_UCSF_opt,Enuc,obt_spatial,tbt_spatial,list_orb_rot,list_list_rdm1,list_list_rdm2,l_axial_sym,list_degmo,x_orbrot_0,nparal,debug=False)
    improve = E_orbopt_UCSF - E_old
    print(f'E_new: {E_orbopt_UCSF} vs. E_old: {E_old}, improve: {improve}, conv. thrsh: {orbopt_conv_thrsh}')

    Hmat_full_space_updated =  construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space,Enuc,obt_opt,tbt_opt,nparal=nparal)
    E_old = E_orbopt_UCSF
    print(f'Updated E_old: {E_old} ')
    list_vec_theta_ini = list_vec_theta_opt
    x_orbrot_0 = x_orbrot_UCSF

#list_list_theta_opt, list_Uvec_opt, list_Umat_opt, list_UCSF_opt = opt_theta_for_energy_UCSF(list_list_pair_ex_space,list_list_theta_opt,list_list_genmat_to_use,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_sym_CSF_vec,nparal,l_use_decomp_genmat,Hmat_full_space, no_states, [1/no_states] * no_states)

Hmat_UCSF_opt, list_list_rdm1, list_list_rdm2 = construct_Hmat_CSFs_paral_triu(list_UCSF_opt, Enuc, obt, tbt, nparal,lrdm=True)
evals_init = get_lowest_n_eigen(Hmat_UCSF_opt, no_states)
weights = np.array([1/no_states] * no_states)
E_old = float(np.dot(weights, evals_init))
print(f"Initial E_old: {E_old}")

eigval_full, eigvec_full = scipy.sparse.linalg.eigsh(csr_matrix(Hmat_full_space_updated), k=no_states, which="SA")
idx = np.argsort(eigval_full)
eigval_full = eigval_full[idx]
eigvec_full = eigvec_full[:, idx]
print(f"\nComputed {no_states} lowest eigenpairs of FULL expanded CSF space:")
print_eigen_solution(eigval_full, eigvec_full)

while abs(improve) > orbopt_conv_thrsh:
    nround += 1
    print(f"\nRound {nround}")

    list_list_theta_opt, list_Uvec_opt, list_Umat_opt, list_UCSF_opt = opt_multi_train_UCSF(list_list_pair_ex_space,list_list_theta_opt,list_list_genmat_to_use,l_use_decomp_genmat,Enuc, obt, tbt,
    nparal, list_sym_CSF_vec, shared_unitary_groups, no_states, list_list_extra_vecs=list_list_extra_vecs, frozen_theta_map=frozen_theta_map)

    # ── UCSF orthogonality check after theta optimization ──────────────────────
    _n_ucsf = len(list_UCSF_opt)
    _S_ucsf = np.zeros((_n_ucsf, _n_ucsf))
    for _ui in range(_n_ucsf):
        _mi = {}
        for k, c in zip(list_UCSF_opt[_ui][1], list_UCSF_opt[_ui][2]):
            _mi[int(k)] = _mi.get(int(k), 0.0) + float(c)
        for _uj in range(_ui, _n_ucsf):
            _mj = {}
            for k, c in zip(list_UCSF_opt[_uj][1], list_UCSF_opt[_uj][2]):
                _mj[int(k)] = _mj.get(int(k), 0.0) + float(c)
            _ov = sum(_mi[k] * _mj[k] for k in _mi if k in _mj)
            _S_ucsf[_ui, _uj] = _S_ucsf[_uj, _ui] = _ov
    _off_ucsf = _S_ucsf - np.eye(_n_ucsf)
    _max_off  = np.abs(_off_ucsf).max()
    _diag_min = _S_ucsf.diagonal().min()
    _diag_max = _S_ucsf.diagonal().max()
    print(f'\n=== UCSF orthogonality check (post theta-opt, round {nround}) ===')
    print(f'  diagonal: [{_diag_min:.6f}, {_diag_max:.6f}]  (all should be 1.0)')
    print(f'  max |off-diag|: {_max_off:.6f}  (all should be 0.0)')
    _bad_ucsf = [(i, j, _S_ucsf[i, j]) for i in range(_n_ucsf)
                 for j in range(i+1, _n_ucsf) if abs(_S_ucsf[i, j]) > 1e-6]
    # build UCSF-column → group map.  Column order in opt_multi_train_UCSF is:
    # for each group g: one rotated-reference column (if sym_CSF_vec[g] != 0),
    # then one column per extra vec in list_list_extra_vecs[g].
    _ucsf_to_group = {}
    _col = 0
    for _ug in range(len(list_sym_CSF_vec)):
        if np.any(np.asarray(list_sym_CSF_vec[_ug]) != 0.0):
            _ucsf_to_group[_col] = _ug
            _col += 1
        for _ in list_list_extra_vecs[_ug]:
            _ucsf_to_group[_col] = _ug
            _col += 1

    def _print_ucsf_state(idx):
        st = list_UCSF_opt[idx]
        grp = _ucsf_to_group.get(idx, '?')
        sd_idx = sorted(int(x) for x in st[1])
        coefs  = [round(float(c), 4) for c in st[2]]
        print(f'    UCSF[{idx}] (group {grp}): SD-idx={sd_idx}  coefs={coefs}')
        for _sd_arr, _sd_coef in zip(st[0], st[2]):
            print(f'      coef={float(_sd_coef):+.4f}  occ={[int(x) for x in _sd_arr]}')
        # base CSF of the group: sym_CSF_vec-weighted combination of its basis states
        if grp == '?':
            return
        _bsym = np.asarray(list_sym_CSF_vec[grp])
        _bspace = list_list_pair_ex_space[grp]
        _bvd, _bsd = {}, {}
        for _bslot in np.nonzero(_bsym)[0]:
            _bst = _bspace[_bslot]
            for _ba, _bk, _bc in zip(_bst[0], _bst[1], _bst[2]):
                _bvd[int(_bk)] = _bvd.get(int(_bk), 0.0) + float(_bsym[_bslot]) * float(_bc)
                _bsd[int(_bk)] = _ba
        print(f'      base CSF of group {grp} (sym_CSF_vec slots '
              f'{np.nonzero(_bsym)[0].tolist()}):')
        for _bk in sorted(_bvd):
            if abs(_bvd[_bk]) < 1e-8:
                continue
            print(f'        idx={_bk}  coef={_bvd[_bk]:+.6f}  '
                  f'occ={[int(x) for x in _bsd[_bk]]}')

    if _bad_ucsf:
        print(f'  NON-ORTHOGONAL pairs (|S|>1e-6):')
        _printed = set()
        for _bi, _bj, _bv in _bad_ucsf:
            print(f'    S[{_bi},{_bj}] = {_bv:+.6f}')
            for _bk in (_bi, _bj):
                if _bk not in _printed:
                    _print_ucsf_state(_bk)
                    _printed.add(_bk)
        print(f' Non orthogonal UCSF. Bombing out!')
        sys.exit()
    elif _max_off > 1e-10:
        print(f'  UCSF states NOT exactly orthogonal: residual overlaps up to '
              f'{_max_off:.2e} (below bomb threshold 1e-6)')
        _res_pairs = sorted(((i, j, _S_ucsf[i, j]) for i in range(_n_ucsf)
                             for j in range(i+1, _n_ucsf)
                             if abs(_S_ucsf[i, j]) > 1e-10),
                            key=lambda t: -abs(t[2]))
        for _ri, _rj, _rv in _res_pairs[:10]:
            print(f'    S[{_ri},{_rj}] = {_rv:+.3e}  '
                  f'(group {_ucsf_to_group.get(_ri, "?")} vs '
                  f'group {_ucsf_to_group.get(_rj, "?")})')
    else:
        print('  All UCSF states are orthogonal (to numerical precision).')

    Hmat_UCSF_opt, list_list_rdm1, list_list_rdm2 = construct_Hmat_CSFs_paral_triu(list_UCSF_opt, Enuc, obt, tbt, nparal, lrdm=True)

    evals_before = get_lowest_n_eigen(Hmat_UCSF_opt, no_states)
    print("Lowest eigenvalues BEFORE orb opt:", evals_before)

    if not l_opt_orb:
        break

    obt_prev, tbt_prev = obt, tbt   # for the revert guard below

    x_orbrot_UCSF, Eavg_new, obt_opt, tbt_opt = opt_orbitals_for_weighted_n_roots_of_CSF_space(
        list_UCSF_opt, no_states, Enuc, obt_spatial, tbt_spatial, list_orb_rot,
        list_list_rdm1, list_list_rdm2,
        l_dooh_sym, list_degmo,
        x_orbrot_0, nparal, eps=Uopt_thrsh,
        normalize_weights=True, debug=False)

    if l_dooh_sym and list_degmo:
        print('\n  [orb-rot symmetry check] degenerate-partner rotation angles:')

        def _partner(_o):
            for _pair in list_degmo:
                if _o == _pair[0]:
                    return _pair[1]
                if _o == _pair[1]:
                    return _pair[0]
            return _o            # σ-type / non-degenerate → its own partner

        _rot_pos = {(int(a), int(b)): k for k, (a, b) in enumerate(list_orb_rot)}
        _n_asym = 0
        _seen = set()
        _checked_any = False
        for _ip, (_i1, _i2) in enumerate(list_orb_rot):
            _img = (_partner(int(_i1)), _partner(int(_i2)))
            if _img == (int(_i1), int(_i2)):
                continue                       # self-partnered: nothing to compare
            _jp = _rot_pos.get(_img)
            if _jp is None or _jp == _ip:
                continue                       # partner rotation not parameterized
            _key = frozenset((_ip, _jp))
            if _key in _seen:
                continue
            _seen.add(_key)
            _checked_any = True
            _dx = abs(x_orbrot_UCSF[_ip] - x_orbrot_UCSF[_jp])
            _flag = '' if _dx < 1e-6 else '  *** SYMMETRY BROKEN ***'
            print(f'    rot{list_orb_rot[_ip]} = {x_orbrot_UCSF[_ip]:+.8f}  vs  '
                  f'rot{list_orb_rot[_jp]} = {x_orbrot_UCSF[_jp]:+.8f}  '
                  f'|diff| = {_dx:.2e}{_flag}')
            if _dx >= 1e-6:
                _n_asym += 1
        if not _checked_any:
            print('    (no degenerate-partner rotation pairs to check)')
        elif _n_asym == 0:
            print('    all degenerate-partner rotations equal — symmetry preserving')
        else:
            print(f'    WARNING: {_n_asym} symmetry-broken rotation pair(s)')

    improve = Eavg_new - E_old
    print(f"Eavg_new: {Eavg_new}  Eavg_old: {E_old}  ΔE: {improve:+.6e}")

    if improve > 1e-10:
        print(f" orbital optimization INCREASED energy by {improve:.3e} Ha — "
              f"reverting to previous integrals (round rejected)")
        obt, tbt = obt_prev, tbt_prev
        Hmat_UCSF_after = Hmat_UCSF_opt   # already built with the previous integrals
        Eavg_new = E_old
        improve = 0.0
    else:
        obt, tbt = obt_opt, tbt_opt
        Hmat_UCSF_after = construct_Hmat_CSFs_paral_triu(list_UCSF_opt, Enuc, obt, tbt, nparal)
        x_orbrot_0 = x_orbrot_UCSF

    evals_after = get_lowest_n_eigen(Hmat_UCSF_after, no_states)
    print("Lowest eigenvalues AFTER orb opt:", evals_after)

    Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF_after)
    eigval, eigvec = scipy.sparse.linalg.eigsh(Hmat_UCSF_sparse, k=no_states, which="SA")
    idx = np.argsort(eigval)
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    print(f"\nComputed {no_states} lowest eigenpairs of UCSF (which='SA'):")
    print_eigen_solution(eigval, eigvec)

    E_old = Eavg_new

print(f"\nComputed {no_states} lowest eigenpairs of UCSF (which='SA'):")
print_eigen_solution(eigval, eigvec)

list_CSF_dump = [make_UCSF_state(list_list_pair_ex_space[g], list_sym_CSF_vec[g])
                 for g in range(len(list_list_pair_ex_space))]

list_UCSF                    = list_UCSF_opt
list_UCSF_sym_components      = list_UCSF_opt
list_list_theta_opt_CSF_comp = list_list_theta_opt
list_list_ia_CSF_comp        = list_list_ia

x_orbrot = x_orbrot_0
_obt_chk, _tbt_chk = orthogonal_transform_obt_tbt(x_orbrot, list_orb_rot, obt_spatial, tbt_spatial)

import os
import pickle
import sys
import json

# 1. Define the destination directory, relative to this script's location.
#    QSENSE_DUMPDIR overrides it (absolute, or relative to this script) so a run
#    can write straight into e.g. QSENSE_paper_data/PES/H2O instead of the
#    shared QSENSE_ES_dump -- the dump name encodes neither Ethrsh_select_ia nor
#    Uopt_thrsh, so runs that differ only in those silently overwrite each other.
_dumpdir = os.environ.get('QSENSE_DUMPDIR', 'QSENSE_ES_dump')
target_dir = (_dumpdir if os.path.isabs(_dumpdir) else
              os.path.join(os.path.dirname(os.path.abspath(__file__)), _dumpdir))

# 2. Safety check: Create the folder automatically if it doesn't exist yet
os.makedirs(target_dir, exist_ok=True)

# 3. Construct your dynamic filename exactly as before.
#    csf_small_thrsh is part of the name ('_T1e-06') so a threshold sweep at fixed
#    (no_states, irrep, ratio, S_by2, rdist) does not overwrite itself.
#    '%.0e' keeps it filesystem-safe and uniform: 1e-06 / 1e-05 / 1e-04 / 1e-03.
#    '_C<combo_order>' is included for the same reason: it appears nowhere else
#    in the name, so a sweep over the combination order would otherwise
#    overwrite itself.  (Ethrsh_select_ia is deliberately NOT in the name: with
#    l_include_ia_in_CAS every in-CAS excitation is kept regardless of it, so it
#    is not a knob this study varies.)
#    The molecule/space tag is DERIVED from the input Hamiltonian, not
#    hardcoded, so an H2O run cannot write under an h2o2 name.  Splitting on
#    '_phys_spatial' gives 'h2o2_sto3g_12o18e' and 'h2o_sto3g_7o10e'.
_ham_tag = os.path.basename(filnam_pyscf_phys_spatial).split('_phys_spatial')[0]

save_filename = (_ham_tag + '_UCSF_' + str(no_states) + '_'
                 + str(irrep_label_choice) + '_' + str(ratio)
                 + '_S' + str(S_by2)
                 + '_T' + f'{csf_small_thrsh:.0e}'
                 + '_C' + str(combo_order)
                 + '_for_Arjun_' + str(rdist) + '.dump')

# 4. Join them to form the absolute path destination
full_save_path = os.path.join(target_dir, save_filename)

# 5. Open and dump your data
with open(full_save_path, 'wb') as f:
    pickle.dump([list_CSF_dump, list_list_ia_CSF_comp, list_list_theta_opt_CSF_comp, list_sym_CSF_vec, list_UCSF, list_UCSF_sym_components,\
      list_list_somo_UCSF_symcomp, list_list_extra_vecs, psi_GS_UCSF_symcomp, list_orb_rot, x_orbrot, Enuc, obt_spatial, tbt_spatial], f)

print(f'File saved to: {full_save_path}')

print('\n\nUCSF Eigenstates expressed in Slater Determinants')
print('='*60)

small_coef = 0.001
output_energy = eigval

for istate in range(no_states):
    print(f'\nEigenstate {istate}, Energy = {eigval[istate]:.8f} Ha')
    print('-'*50)

    sd_dict = {}

    for iucsf, coef_ucsf in enumerate(eigvec[:, istate]):
        if abs(coef_ucsf) < 1e-12:
            continue
        UCSF = list_UCSF_opt[iucsf]
        onlist  = UCSF[0]
        idxlist = UCSF[1]
        coefvec = UCSF[2]

        for iSD in range(len(onlist)):
            sd_idx  = idxlist[iSD]
            sd_coef = coef_ucsf * coefvec[iSD]
            if sd_idx in sd_dict:
                sd_dict[sd_idx][1] += sd_coef
            else:
                sd_dict[sd_idx] = [onlist[iSD], sd_coef]

    sorted_sds = sorted(sd_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

    norm = sum(v[1]**2 for _, v in sorted_sds)
    print(f' Norm of SD expansion: {norm:.6f}')

    print(f'  Significant SDs (|coef| > {small_coef}):')
    for sd_idx, (onvec, coef) in sorted_sds:
        if abs(coef) < small_coef:
            break
        n_spatial = len(onvec) // 2
        bit_string = ''
        somos, domos = [], []
        for imo in range(n_spatial):
            a = int(onvec[2 * imo])
            b = int(onvec[2 * imo + 1])
            bit_string += str(a) + str(b) + ' '
            if a == 1 and b == 1:
                domos.append(str(imo))
            elif a != b:  # a+b == 1
                somos.append(str(imo))
        somo_str = str(somos) if somos else "['(no-somo)']"
        domo_str = str(domos) if domos else "['(no-domo)']"
        print(f'   coef = {coef:+.6f}   |{bit_string.strip()}>   '
              f'SOMOs: {somo_str} | DOMOs: {domo_str}')

print('\n\nFull CSF space eigenstates expressed in Slater Determinants')
print('='*60)

# Recompute the full-space eigenpairs in the FINAL optimized orbital basis
_Hmat_full_final = csr_matrix(construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space, Enuc, obt, tbt, nparal))
Energies   = get_lowest_n_eigen(_Hmat_full_final, no_states)
Evecs_full = get_lowest_n_eigvec(_Hmat_full_final, no_states)

for istate in range(no_states):
    print(f'\nEigenstate {istate}, Energy = {Energies[istate]:.8f} Ha')
    print('-'*50)

    sd_dict = {}
    if (l_axial_sym or l_dooh_sym) and 'Umat_sym_adapted' in dir():
        coefs_sym = Umat_sym_adapted.T @ Evecs_full[istate]
        for isym, coef_sym in enumerate(coefs_sym):
            if abs(coef_sym) < 1e-12:
                continue
            for icsf in range(len(list_all_CSFs_ex_space)):
                w = Umat_sym_adapted[icsf, isym]
                if abs(w) < 1e-12:
                    continue
                CSF = list_all_CSFs_ex_space[icsf]
                for iSD in range(len(CSF[0])):
                    sd_idx  = CSF[1][iSD]
                    sd_coef = coef_sym * w * CSF[2][iSD]
                    if sd_idx in sd_dict:
                        sd_dict[sd_idx][1] += sd_coef
                    else:
                        sd_dict[sd_idx] = [CSF[0][iSD], sd_coef]
    else:
        for icsf, coef_csf in enumerate(Evecs_full[istate]):
            if abs(coef_csf) < 1e-12:
                continue
            CSF = list_all_CSFs_ex_space[icsf]
            onlist  = CSF[0]
            idxlist = CSF[1]
            coefvec = CSF[2]

            for iSD in range(len(onlist)):
                sd_idx  = idxlist[iSD]
                sd_coef = coef_csf * coefvec[iSD]
                if sd_idx in sd_dict:
                    sd_dict[sd_idx][1] += sd_coef
                else:
                    sd_dict[sd_idx] = [onlist[iSD], sd_coef]

    sorted_sds = sorted(sd_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

    norm = sum(v[1]**2 for _, v in sorted_sds)
    print(f' Norm of SD expansion: {norm:.6f}')

    print(f'  Significant SDs (|coef| > {small_coef}):')
    for sd_idx, (onvec, coef) in sorted_sds:
        if abs(coef) < small_coef:
            break
        n_spatial = len(onvec) // 2
        bit_string = ''
        somos, domos = [], []
        for imo in range(n_spatial):
            a = int(onvec[2 * imo])
            b = int(onvec[2 * imo + 1])
            bit_string += str(a) + str(b) + ' '
            if a == 1 and b == 1:
                domos.append(str(imo))
            elif a != b:  # a+b == 1
                somos.append(str(imo))
        somo_str = str(somos) if somos else "['(no-somo)']"
        domo_str = str(domos) if domos else "['(no-domo)']"
        print(f'   coef = {coef:+.6f}   |{bit_string.strip()}>   '
              f'SOMOs: {somo_str} | DOMOs: {domo_str}')

#Hmat_full_space = construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space, Enuc, obt, tbt, nparal)
#Umat_full_Uext_mp2, Vmat_full_make_Uvec_full, list_list_refCSF, list_list_Uext_mp2_CSF = make_Uvec_full(list_list_ia_noCAS,list_list_mp2_ampld_for_Uext,l_use_decomp_genmat,list_list_genmat_to_use,list_UCSF_subspace_start_end,list_list_pairex_CSF_vec,list_list_pair_ex_space,debug=True)
#Hmat_Uext_mp2 = Umat_full_Uext_mp2.transpose()@Hmat_full_space@Umat_full_Uext_mp2
#E0_Uext_mp2, psi_Uext_mp2 = get_ground_state(csr_matrix(Hmat_Uext_mp2))
#H_mp2_sparse = csr_matrix(Hmat_Uext_mp2)
#eigenvalues = np.linalg.eigvalsh(H_mp2_sparse.toarray())
#print(f'Eigenvalues:', eigenvalues)
#print(f'\nE0_Uext_mp2: {E0_Uext_mp2}')
#print(f'psi_Uext_mp2: {psi_Uext_mp2}')

#Hmat_UCSF = construct_Hmat_CSFs(list_UCSF,Enuc,obt,tbt)
# Hmat_UCSF = construct_Hmat_CSFs_paral_triu(list_UCSF,Enuc,obt,tbt,nparal)
# print('\nHmat of UCSF')
# print_matrix(Hmat_UCSF)
# Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF)
# eigval, eigvec = scipy.sparse.linalg.eigsh(Hmat_UCSF_sparse, k=no_states, which="SA")
# idx = np.argsort(eigval)
# eigval = eigval[idx]
# eigvec = eigvec[:, idx]
# print(f"\nComputed {no_states} lowest eigenpairs of UCSF (which='SA'):")
# print_eigen_solution(eigval, eigvec)

count = sum(1 for sublist in list_improvement for x in sublist)
print(f'Total no. of valid ia pairs: ', count)

count = sum(1 for sublist in list_improvement for x in sublist if abs(x) > ratio * Ethrsh_select_ia)
print(f'Total no. of basis excitation ia pairs: ', count)

print("\nIrrep labels of CSFs in list_CSF:")

for i, SOMO_DMO in enumerate(list_SOMO_DMO):
    somo = SOMO_DMO[0]
    dmo  = SOMO_DMO[1]
    irrep = SOMO_DMO[2]

    print(f"CSF {i}: irrep = {irrep}, SOMO = {somo}, DMO = {dmo}, {list_CSF[i]}")

print('\n=== Base state of each UCSF group ===')
for _bg in range(len(list_list_pair_ex_space)):
    _bsym = np.asarray(list_sym_CSF_vec[_bg])
    _bspace = list_list_pair_ex_space[_bg]
    print(f'Group {_bg}: {len(_bspace)} states, '
          f'sym_CSF_vec nonzero slots = {np.nonzero(_bsym)[0].tolist()}')
    # physical base state = sum of sym_CSF_vec components over the group's slots
    _bvd = {}
    _bsd = {}
    for _bslot in np.nonzero(_bsym)[0]:
        _bst = _bspace[_bslot]
        for _ba, _bk, _bc in zip(_bst[0], _bst[1], _bst[2]):
            _bvd[int(_bk)] = _bvd.get(int(_bk), 0.0) + float(_bsym[_bslot]) * float(_bc)
            _bsd[int(_bk)] = _ba
    for _bk in sorted(_bvd):
        if abs(_bvd[_bk]) < 1e-8:
            continue
        print(f'    idx={_bk}  coef={_bvd[_bk]:+.6f}  '
              f'occ={[int(x) for x in _bsd[_bk]]}')

# Resource counts, recorded alongside the energies so a threshold / state-count
# study can read subspace size straight from the JSON instead of scraping these
# numbers back out of the SLURM .out files.  These are the quantities that drive
# the circuit cost in the measurement benchmark: the UCSF subspace dimension
# sets how many diagonal + off-diagonal matrix elements must be measured, and
# the ia-pair counts set the generator (and hence gate) count.
n_valid_ia = sum(1 for sublist in list_improvement for x in sublist)
n_basis_ia = sum(1 for sublist in list_improvement for x in sublist
                 if abs(x) > ratio * Ethrsh_select_ia)

output_data = {
    "output_energy": list(output_energy),  # Ensure it's a standard list, not a numpy array
    "irrep_label_choice": irrep_label_choice,
    "rdist": rdist,
    "no_states": no_states,
    "ratio": ratio,
    "S_by2": S_by2,
    "csf_small_thrsh": csf_small_thrsh,
    # --- resource counts ---
    "n_csf": len(list_CSF),                        # CSFs surviving the filters
    "n_ucsf": len(list_UCSF),                      # UCSF subspace dimension
    "n_ucsf_groups": len(list_list_pair_ex_space),  # UCSF groups (base states)
    "n_group_states": [len(g) for g in list_list_pair_ex_space],
    "n_valid_ia": n_valid_ia,                      # all valid ia pairs
    "n_basis_ia": n_basis_ia,                      # ia pairs above ratio*Ethrsh
    "Ethrsh_select_ia": Ethrsh_select_ia,
    "combo_order": combo_order,
}

# Also persist the energies to a JSON file next to the .dump (same base name),
# so the results are recoverable without parsing stdout.
json_save_path = os.path.join(target_dir, save_filename.replace('.dump', '.json'))
with open(json_save_path, 'w') as jf:
    json.dump(output_data, jf, indent=2)
print(f'Energies saved to: {json_save_path}')

y_values = [list_improvement[item] for item in range(len(list_improvement))]
x_axis = [x for x in range(len(list_improvement))]

x_plot = []
y_plot = []

for i, sublist in enumerate(y_values):
    for val in sublist:
        x_plot.append(int(i))
        y_plot.append(10**3 * abs(val))

plt.figure(figsize=(10, 6))
plt.scatter(x_plot, y_plot, s=20)
plt.yscale('log')

plt.title('CSF Subspace Excitation Energy Difference (mHa) vs CSF index')
plt.xticks(range(len(x_axis)))
plt.xlabel('CSF Index')
plt.ylabel('CSF Subspace Excitation Energy Difference (mHa)')
plt.axhline(y=10**3 * ratio * Ethrsh_select_ia, color='red', linestyle='--', label='Basis Extension Threshold')
plt.axhline(y=10**3 * Ethrsh_select_ia, color='orange', linestyle='--', label='Generator Selection Threshold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

#plt.show()
    
# Print ONLY this JSON block to stdout and exit cleanly
print(json.dumps(output_data))
sys.exit(0)

