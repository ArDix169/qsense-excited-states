"""Sampling cost and subspace size vs number of targeted states, H2O2.

Companion to fig_pes_h2o2.py -- both import qsense_style so typography, line
weights, grid and legend framing are identical.

CNOT and depth are deliberately absent: they vary by under 15% across the study
and belong in a table.  Straight segments, no interpolation -- with five
x-values and a non-monotonic series (3.0 A spikes at n=3, collapses at n=4) a
spline invents overshoot that is not in the data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from qsense_style import paper_style, shade, COL_W, C_COST, C_BASIS, GEOM

# --- smooth interpolation ---------------------------------------------------
# PCHIP, not a natural cubic spline.  Both are piecewise cubics, but PCHIP is
# SHAPE-PRESERVING: it never overshoots the data range and never invents a local
# extremum between two points.  A natural cubic through these series does both --
# measured on the H2O2 3.0 A costs (which spike at n=3 and collapse at n=4) it
# rings well outside the measured range, drawing structure between integer n
# that no run supports.  Markers stay on the true values; only the connecting
# line is interpolated.
#
# Cost is interpolated in LOG space so the curve is smooth as drawn on the log
# axis rather than smooth in linear units and kinked on screen.
def smooth(x, y, log=False, npts=300):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return x, y
    yy = np.log10(y) if log else y
    xs = np.linspace(x.min(), x.max(), npts)
    ys = PchipInterpolator(x, yy)(xs)
    return xs, (10.0 ** ys if log else ys)



paper_style()

n_states = [1, 2, 3, 4, 5]

# eps^2 M from the VO measurement benchmark.  M_e is summed continuously; do
# not round it to integer shots first, which would zero every element below
# half a shot.
data = {
    1.5: {
        'samp_cost': [1.9432e-02, 4.8154e-02, 1.9747e-01, 3.5119e-01, 5.1573e-01],
        'basis':     [887, 1207, 1364, 1547, 1653],
    },
    1.875: {
        'samp_cost': [4.6823e-03, 1.8539e-02, 7.1849e-02, 2.5819e-01, 4.3276e-01],
        'basis':     [949, 1054, 1133, 1211, 1510],
    },
    3.0: {
        'samp_cost': [2.0229e-03, 2.2194e-03, 6.7738e-02, 9.6638e-03, 1.9371e-02],
        'basis':     [1136, 1588, 1688, 1902, 2070],
    },
}


fig = plt.figure(figsize=(COL_W, 4.4))

# Margins match fig_pes_h2o2.py so the two figures align on the page.
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08,
                       left=0.22, right=0.97, top=0.88, bottom=0.12)

ax_cost = fig.add_subplot(gs[0])
ax_basis = fig.add_subplot(gs[1], sharex=ax_cost)

for r, d in data.items():
    g = GEOM[r]
    cc, cb = shade(C_COST, g['weight']), shade(C_BASIS, g['weight'])

    xs, ys = smooth(n_states, d['samp_cost'], log=True)
    ax_cost.plot(xs, ys, '-', color=cc)
    ax_cost.plot(n_states, d['samp_cost'], linestyle='none',
                 marker=g['marker'], color=cc)

    xs, ys = smooth(n_states, d['basis'])
    ax_basis.plot(xs, ys, '-', color=cb)
    ax_basis.plot(n_states, d['basis'], linestyle='none',
                  marker=g['marker'], color=cb)

ax_cost.set_yscale('log')
ax_cost.set_ylabel(r'Sampling Cost $(\epsilon^2M)$')
# headroom below: the two 3.0 A points at ~1.7e-3 otherwise sit on the axis
ax_cost.set_ylim(1e-3, 1.0)
ax_cost.grid(True, which='both')
plt.setp(ax_cost.get_xticklabels(), visible=False)

ax_basis.set_ylabel('Basis States')
ax_basis.grid(True)
ax_basis.set_xticks(n_states)
ax_basis.set_xlabel(r'Number of States Targeted in $\mathrm{H}_2\mathrm{O}_2$')

# Geometry encoding is identical in both panels -> one neutral-gray legend,
# anchored over the top panel exactly as the PES figure's legend is.
CG = '0.25'
pos = ax_cost.get_position()
leg = fig.legend(handles=[Line2D([], [], color=shade(CG, GEOM[r]['weight']),
                                 linestyle='-', marker=GEOM[r]['marker'],
                                 label=GEOM[r]['label']) for r in data],
                 ncol=3, loc='lower center',
                 bbox_to_anchor=(pos.x0 - 0.025, 0.905, pos.width, 0.04),
                 bbox_transform=fig.transFigure)
leg.get_frame().set_linewidth(0.6)

# NO tight_layout / bbox_inches='tight': the GridSpec margins are the layout,
# and either would override them and change the final width.
plt.savefig('h2o2_cost_basis_col.pdf')
plt.savefig('h2o2_cost_basis_col.png')
plt.show()
