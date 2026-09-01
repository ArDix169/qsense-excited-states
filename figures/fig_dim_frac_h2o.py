"""Subspace size and compression vs bond length, H2O.

NOT the sister of fig_scaling_h2o2.py -- fig_scaling_h2o.py is, and it plots
the same eps^2 M / basis-states panels against the number of targeted states.
This figure is kept because it shows something that one does not: the UCSF
dimension and its fraction of the full CSF sector across ALL EIGHT irrep/spin
sectors, swept over bond length.

  top     n_ucsf, the UCSF subspace dimension actually diagonalised -- this is
          what sets how many matrix elements a measurement would need
  bottom  n_ucsf as a fraction of the full CSF sector dimension, from the
          Weyl-Paldus-verified census in h2o_sto3g_fci_ref_full.pkl.  Below 1.0
          the subspace is a genuine compression; at 1.0 it spans the sector and
          Q-SENSE is reproducing CASCI exactly.

Straight segments, no interpolation -- the series are not smooth in r and a
spline would invent structure, the same reason fig_scaling_h2o2.py uses them.

Source: the 80 sector runs of hpc/run_h2o_pes.sh, eps_1 = 1e-6,
eps_2 = 0, eps_3 = 1e-6, l_max = 2, two states per sector.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

from qsense_style import paper_style, COL_W, IRREP_C2V, SUB_LS

paper_style()

r_values = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

E_QSENSE = {
    'singlet': {
        'A1': {1: [-74.78578683, -75.01977735, -74.96750585, -74.87341038, -74.80092164, -74.76198525, -74.74627695, -74.7405949, -74.73851188, -74.73773919],
               2: [-73.97369939, -74.45625702, -74.56127154, -74.59014739, -74.61854646, -74.63340052, -74.63942615, -74.64163055, -74.64222771, -74.64238135]},
        'A2': {1: [-73.94159527, -74.52641242, -74.68809714, -74.7257577, -74.73289686, -74.73527404, -74.73662407, -74.73717135, -74.73731842, -74.73733747],
               2: [-73.10256036, -73.99925398, -74.38192199, -74.54464639, -74.60964947, -74.63230416, -74.63936668, -74.64151862, -74.64216944, -74.6423597]},
        'B1': {1: [-74.06118413, -74.60611789, -74.74785646, -74.76463843, -74.75353913, -74.74434383, -74.74019052, -74.73849791, -74.73779033, -74.73749786],
               2: [-73.14153936, -74.02445842, -74.39379895, -74.5477072, -74.60941036, -74.63175465, -74.63907821, -74.64140167, -74.64212744, -74.64234565]},
        'B2': {1: [-73.80528172, -74.36900645, -74.54161861, -74.60265925, -74.62662804, -74.63644539, -74.64034144, -74.64176953, -74.64221703, -74.64235098],
               2: [-73.67113174, -74.24003091, -74.37850588, -74.36494015, -74.30452002, -74.2381409, -74.18322285, -74.14591358, -74.12070517, -74.10383042]},
    },
    'triplet': {
        'A1': {1: [-74.03108174, -74.56166488, -74.71196702, -74.74535803, -74.74650865, -74.74249613, -74.73974827, -74.73837461, -74.7377503, -74.73748293],
               2: [-73.67459913, -74.31558697, -74.5245785, -74.59746119, -74.62522487, -74.63613494, -74.64028527, -74.64175891, -74.64223933, -74.64238115]},
        'A2': {1: [-73.97975373, -74.56307426, -74.7188035, -74.74779073, -74.74626493, -74.74195038, -74.73948175, -74.7382834, -74.73772088, -74.73747457],
               2: [-73.16800053, -74.06448625, -74.44564983, -74.61292078, -74.68622476, -74.71739019, -74.72992923, -74.73470429, -74.73643685, -74.73703595]},
        'B1': {1: [-74.13705555, -74.66227195, -74.78325033, -74.78293312, -74.76091932, -74.74676171, -74.74094793, -74.7387458, -74.73787256, -74.73752412],
               2: [-73.24615231, -74.11028281, -74.47021814, -74.62391122, -74.69072722, -74.71916034, -74.73060965, -74.73495939, -74.73652861, -74.73706719]},
        'B2': {1: [-73.91213583, -74.48895053, -74.6655504, -74.71336318, -74.72498095, -74.73090056, -74.73464499, -74.73636933, -74.73701313, -74.73723069],
               2: [-73.80162705, -74.38198562, -74.53912666, -74.58810694, -74.61858553, -74.633984, -74.63995727, -74.6417311, -74.64222669, -74.64238291]},
    },
}

# E_FCI is the FULL CAS(10e,7o) FCI, not the frozen-core CAS(8e,6o) sector.
# actmo_start=1 restricts which excitations the ANSATZ generates; it does not
# reduce the Hamiltonian, so full-space FCI is what the ansatz approximates.
# Corrected 2026-08-29 (was the CAS(8e,6o) sector, as fig_pes_h2o.py also was).
E_FCI = {
    'singlet': {
        'A1': {1: [-74.78590188, -75.0198548, -74.96755511, -74.87343609, -74.80092773, -74.76198843, -74.74627745, -74.74059587, -74.73851248, -74.73773982],
               2: [-73.97382676, -74.45636535, -74.56135241, -74.59018955, -74.61855176, -74.63340174, -74.63942795, -74.64163194, -74.64222797, -74.64238142]},
        'A2': {1: [-73.94164436, -74.52644218, -74.68811646, -74.72576812, -74.73290027, -74.73527502, -74.73662532, -74.73717181, -74.73731859, -74.73733753],
               2: [-73.10260744, -73.99927086, -74.38192912, -74.54465089, -74.60965242, -74.63230523, -74.63936751, -74.64151896, -74.64216957, -74.64235974]},
        'B1': {1: [-74.06125759, -74.60616319, -74.74787926, -74.76464804, -74.75354282, -74.74434597, -74.74019247, -74.73849857, -74.73779095, -74.73749797],
               2: [-73.14157112, -74.02447307, -74.3938053, -74.54770964, -74.60941107, -74.63175493, -74.6390787, -74.64140225, -74.64212754, -74.64234568]},
        'B2': {1: [-73.80535418, -74.36905402, -74.54165276, -74.60267748, -74.62663796, -74.63645026, -74.64034634, -74.64177021, -74.64224128, -74.64238155],
               2: [-73.67127008, -74.24012815, -74.37858272, -74.36500994, -74.30456969, -74.2381703, -74.18323981, -74.14591955, -74.12206352, -74.104391]},
    },
    'triplet': {
        'A1': {1: [-74.031172, -74.56172716, -74.71200332, -74.74537488, -74.74651451, -74.74249823, -74.73974876, -74.73837567, -74.73775063, -74.73748366],
               2: [-73.67466129, -74.31562459, -74.52460249, -74.59747448, -74.62522973, -74.63613674, -74.64028569, -74.64175943, -74.64223951, -74.64238129]},
        'A2': {1: [-73.97984688, -74.56312622, -74.71882959, -74.7478019, -74.74626875, -74.74195153, -74.73948349, -74.73828398, -74.73772211, -74.73747476],
               2: [-73.16804261, -74.06450785, -74.44565944, -74.61292478, -74.68622604, -74.71739049, -74.72992951, -74.73470531, -74.73643759, -74.73703604]},
        'B1': {1: [-74.13712366, -74.66231822, -74.78327829, -74.78294638, -74.76092395, -74.74676306, -74.74094957, -74.73874682, -74.73787363, -74.73752441],
               2: [-73.24617887, -74.11029738, -74.47022512, -74.62391428, -74.69072837, -74.71916063, -74.7306099, -74.73495989, -74.73652906, -74.73706726]},
        'B2': {1: [-73.91223226, -74.489005, -74.66558135, -74.71337825, -74.72499137, -74.73090204, -74.73464593, -74.73637076, -74.73701746, -74.73723092],
               2: [-73.80172876, -74.38207279, -74.53919027, -74.58813081, -74.61861209, -74.63398496, -74.63995821, -74.64173188, -74.6422414, -74.64238298]},
    },
}

N_UCSF = {
    'singlet': {
        'A1': [35, 35, 35, 35, 35, 35, 35, 35, 35, 32],
        'A2': [20, 20, 20, 20, 20, 20, 20, 20, 20, 19],
        'B1': [20, 20, 20, 20, 20, 19, 20, 17, 16, 16],
        'B2': [27, 28, 26, 26, 26, 26, 26, 25, 20, 19],
    },
    'triplet': {
        'A1': [24, 24, 24, 24, 24, 23, 23, 21, 21, 19],
        'A2': [26, 26, 26, 26, 26, 26, 26, 24, 24, 24],
        'B1': [24, 24, 23, 23, 24, 24, 24, 20, 19, 19],
        'B2': [28, 29, 29, 28, 28, 28, 28, 24, 22, 23],
    },
}

N_CSF = {
    'singlet': {
        'A1': [11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        'A2': [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
        'B1': [11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        'B2': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    },
    'triplet': {
        'A1': [13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
        'A2': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        'B1': [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
        'B2': [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    },
}

SECTOR_DIM = {
    'singlet': {
        'A1': 37,
        'A2': 20,
        'B1': 20,
        'B2': 28,
    },
    'triplet': {
        'A1': 25,
        'A2': 26,
        'B1': 24,
        'B2': 30,
    },
}

irrep_order = ['A1', 'A2', 'B1', 'B2']
spins = ['singlet', 'triplet']
r = np.array(r_values)

fig = plt.figure(figsize=(COL_W, 4.4))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08,
                       left=0.22, right=0.97, top=0.86, bottom=0.12)

ax_dim = fig.add_subplot(gs[0])
ax_frac = fig.add_subplot(gs[1], sharex=ax_dim)

for irr in irrep_order:
    c = IRREP_C2V[irr]
    for spin in spins:
        n = np.array(N_UCSF[spin][irr], dtype=float)
        # Spin is encoded by linestyle, reusing SUB_LS so the dashing means the
        # same thing it does in the PES figure's root encoding is NOT implied --
        # this is a different axis, hence the explicit legend below.
        ls = '-' if spin == 'singlet' else '--'
        ax_dim.plot(r, n, ls, color=c, marker='o' if spin == 'singlet' else 's',
                    markersize=3, markeredgecolor='white', markeredgewidth=0.4)
        ax_frac.plot(r, n / SECTOR_DIM[spin][irr], ls, color=c,
                     marker='o' if spin == 'singlet' else 's',
                     markersize=3, markeredgecolor='white', markeredgewidth=0.4)

ax_dim.set_ylabel('UCSF Subspace Dim.')
ax_dim.grid(True)
plt.setp(ax_dim.get_xticklabels(), visible=False)

ax_frac.set_ylabel('Fraction of Sector')
ax_frac.grid(True)
ax_frac.axhline(1.0, color='0.4', linestyle=':', linewidth=0.6, zorder=1)
ax_frac.set_ylim(0, 1.15)
ax_frac.xaxis.set_major_locator(MultipleLocator(0.5))
ax_frac.set_xlabel(r'O$-$H Bond Length (\AA) in $\mathrm{H}_2\mathrm{O}$')
ax_frac.set_xlim(0.65, 3.10)

irrep_h = [Line2D([], [], color=IRREP_C2V[i], linestyle='-', label=i)
           for i in irrep_order]
spin_h = [Line2D([], [], color='0.35', linestyle='-', marker='o',
                 markersize=3, label='singlet'),
          Line2D([], [], color='0.35', linestyle='--', marker='s',
                 markersize=3, label='triplet')]

pos = ax_dim.get_position()
leg = fig.legend(handles=irrep_h + spin_h, ncol=3, loc='lower center',
                 bbox_to_anchor=(pos.x0, 0.885, pos.width, 0.04),
                 bbox_transform=fig.transFigure)
leg.get_frame().set_linewidth(0.6)

# NO tight_layout / bbox_inches='tight': the GridSpec margins ARE the layout,
# and either would override them and change the final width.
plt.savefig('h2o_dim_frac_col.pdf')
plt.savefig('h2o_dim_frac_col.png')
