"""H2O PES by C2v irrep and spin, with the Q-SENSE error against FCI.

Sister figure to fig_pes_h2o2.py -- same import of qsense_style, same stacked
singlet/triplet layout, same error subpanels, so the two sit on a page as one
system.  Three things differ, all forced by the molecule:

  four irreps, not two.  C2v spans A1/A2/B1/B2 where C2 spanned A/B, so the
  legend gains a row and the palette comes from IRREP_C2V.  A1 and B1 reuse
  H2O2's A and B hues exactly.

  A2 is included even though STO-3G water has NO A2 ORBITAL.  A2 states are
  plentiful -- a CSF with a singly occupied B1 and a singly occupied B2 carries
  B1 x B2 = A2 -- and the sector holds 20 singlet and 26 triplet CSFs at 1.0 A.

  the FCI reference is the FULL CAS(10e,7o) space, not the frozen-core
  CAS(8e,6o) sector.  Q-SENSE's actmo_start=1 restricts which excitations the
  ANSATZ generates; it does not reduce the Hamiltonian, which still spans the
  full space.  So the full-space FCI is the energy the ansatz is actually
  approximating, and differencing against the frozen-core sector would answer
  the narrower question of how well the ansatz solves its own restricted
  manifold.

Numbers are inlined rather than read from QSENSE_ES_dump so the figure can be
regenerated without the dumps, matching the h2o2 convention.
"""
import os as _os
_OUT = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (LogLocator, NullLocator, NullFormatter,
                               LogFormatterMathtext, MultipleLocator)
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

from qsense_style import (paper_style, COL_W, IRREP_C2V, STATE_MARKER,
                          SUB_LS, FCI_KW, CHEM_ACC_KW, MARKER_KW)

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

irrep_order = ['A1', 'A2', 'B1', 'B2']
MS, MEW = MARKER_KW['markersize'], MARKER_KW['markeredgewidth']
marker_kw = dict(markersize=MS, markeredgecolor='white', markeredgewidth=MEW)
r = np.array(r_values)


def smooth(x, y, n=300):
    """Spline only the FCI reference, which is smooth by construction."""
    if len(x) < 4:
        return x, y
    cs = CubicSpline(x, y, bc_type='natural')
    xn = np.linspace(x.min(), x.max(), n)
    return xn, cs(xn)


def plot_spin(ax_pes, ax_err, spin):
    for irr in irrep_order:
        c = IRREP_C2V[irr]
        for s in (1, 2):
            yq = np.array(E_QSENSE[spin][irr][s])
            yf = np.array(E_FCI[spin][irr][s])

            xs, ys = smooth(r, yf)
            ax_pes.plot(xs, ys, **FCI_KW)
            ax_pes.plot(r, yq, linestyle='none', color=c,
                        marker=STATE_MARKER[s], markerfacecolor=c, **marker_kw)

            # |dE| against the matching CAS sector, same grid -- no
            # interpolation needed, unlike H2O2 where the FCI grid was denser.
            ax_err.plot(r, np.abs(yq - yf) * 1e3, color=c,
                        marker=STATE_MARKER[s], markerfacecolor=c,
                        linestyle=SUB_LS[s], **marker_kw)

    ax_pes.set_ylabel('Energy (Ha)')
    ax_pes.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_pes.grid(True)
    plt.setp(ax_pes.get_xticklabels(), visible=False)

    ax_err.set_ylabel(r'$|\Delta E|$ (mHa)')
    ax_err.set_yscale('log')
    ax_err.axhline(y=1.6, **CHEM_ACC_KW)
    ax_err.grid(True, which='both')
    ax_err.set_ylim(1e-5, 1e1)
    ax_err.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax_err.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_err.yaxis.set_minor_locator(NullLocator())
    ax_err.yaxis.set_minor_formatter(NullFormatter())


fig = plt.figure(figsize=(COL_W, 6.5))
gs = gridspec.GridSpec(5, 1, height_ratios=[3.2, 1.5, 0.45, 3.2, 1.5],
                       hspace=0.08, left=0.22, right=0.97, top=0.83, bottom=0.10)

ax_s_pes = fig.add_subplot(gs[0, 0])
ax_s_err = fig.add_subplot(gs[1, 0], sharex=ax_s_pes)
ax_t_pes = fig.add_subplot(gs[3, 0])
ax_t_err = fig.add_subplot(gs[4, 0], sharex=ax_t_pes)

plot_spin(ax_s_pes, ax_s_err, 'singlet')
plot_spin(ax_t_pes, ax_t_err, 'triplet')

ax_s_pes.set_ylim(-75.10, -73.00)
ax_t_pes.set_ylim(-74.85, -73.00)

for ax in (ax_s_pes, ax_s_err, ax_t_pes, ax_t_err):
    ax.set_xlim(0.65, 3.10)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))



plt.setp(ax_s_err.get_xticklabels(), visible=False)

irrep_h = [Line2D([], [], color=IRREP_C2V[i], marker='o', linestyle='none',
                  label=i, **marker_kw) for i in irrep_order]
state_h = [Line2D([], [], color='0.45', marker=STATE_MARKER[s],
                  linestyle='none', label=f'root {s}', **marker_kw)
           for s in (1, 2)]
fci_h = [Line2D([], [], color='black', linestyle='--', linewidth=0.8,
                dashes=(3.5, 1.8), label='FCI')]
chem_h = [Line2D([], [], label='Chem. Acc.', **CHEM_ACC_KW)]

pos = ax_s_pes.get_position()
leg = fig.legend(handles=irrep_h + state_h + fci_h + chem_h, ncol=4,
                 loc='lower center',
                 bbox_to_anchor=(pos.x0, 0.865, pos.width, 0.04),
                 bbox_transform=fig.transFigure,
                 frameon=True, framealpha=1.0, edgecolor='0.6',
                 fancybox=False, borderpad=0.35,
                 columnspacing=0.7, handletextpad=0.25)
leg.get_frame().set_linewidth(0.6)

ax_s_pes.set_title('Singlet', pad=3)
ax_t_pes.set_title('Triplet', pad=3)

fig.supxlabel(r'O$-$H Bond Length (\AA)', x=0.595, y=0.04)
fig.align_ylabels([ax_s_pes, ax_s_err, ax_t_pes, ax_t_err])

plt.savefig(_os.path.join(_OUT, 'h2o_stacked.pdf'))
plt.savefig(_os.path.join(_OUT, 'h2o_stacked.png'))   # convenience preview; the PDF is the deliverable
