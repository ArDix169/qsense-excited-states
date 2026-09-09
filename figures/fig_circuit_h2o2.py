"""Per-circuit resources vs number of targeted states, H2O2.

Companion to fig_scaling_h2o2.py -- same column width, margins, geometry
encoding (marker + shade) and PCHIP curves.  Quantity is carried by colour:
C_CNOT for gate count, C_DEPTH for depth.

ONLY THE AVERAGES ARE PLOTTED; the maxima are in tab_h2o2_circuit_max.tex,
matching the H2O treatment.  They do carry a distinct message -- the averages
fall 3.1x with n (CNOT, 30.3 -> 9.8) while the maxima fall only 1.15x
(136 -> 118), so the typical circuit shrinks substantially while the worst one
barely moves -- but on a shared axis the near-flat maxima sit an order of
magnitude above and force a log scale that flattens the trend the averages
carry.  The table says it in five rows without that cost.

WHAT THE AVERAGE IS OVER.  Measurement_Benchmarking_Circuit_parallel.py pools
N^2 values, N = n_ucsf: N diagonal entries (CSF state-preparation circuits) and
N(N-1) off-diagonal entries (parallel-swap circuits, each appended twice), so
the mean is dominated by the swap circuits.  A matrix element that is
classically evaluable contributes a literal ZERO via the `NQ == 0` branch, the
same branch that sets sigma = 0.  So the averages track the classical fraction
as well as circuit complexity -- which is exactly why the maxima are shown
beside them.

Source: analysis/collect_measurement_benchmark.py over SLURM 2173726, rerun under
the fixed optimal_allocation.  Per-geometry thresholds: ratio 1.0 / csf 1e-4 at
1.5 and 1.875 A, ratio 0.1 / csf 1e-6 at 3.0 A.
"""
import os as _os
_OUT = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator

from qsense_style import paper_style, shade, COL_W, C_CNOT, C_DEPTH, GEOM

paper_style()

n_states = [1, 2, 3, 4, 5]

CX_AVG = {1.5:   [30.3, 23.3, 21.3, 17.2, 15.4],
          1.875: [26.4, 26.5, 24.6, 23.2, 19.1],
          3.0:   [25.0, 16.5, 15.8, 12.5,  9.8]}
CX_MAX = {1.5:   [136, 126, 124, 120, 120],
          1.875: [132, 132, 126, 121, 121],
          3.0:   [129, 123, 120, 120, 118]}
DP_AVG = {1.5:   [33.1, 26.5, 23.7, 19.6, 17.6],
          1.875: [29.8, 29.0, 26.8, 25.8, 21.8],
          3.0:   [28.2, 19.0, 18.0, 14.5, 11.5]}
DP_MAX = {1.5:   [139, 135, 134, 128, 126],
          1.875: [130, 130, 128, 129, 131],
          3.0:   [138, 128, 126, 126, 123]}


def smooth(x, y, npts=300):
    """PCHIP: cubic but shape-preserving -- no overshoot, no invented extrema
    between integer n.  Same interpolant as the other scaling figures."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return x, y
    xs = np.linspace(x.min(), x.max(), npts)
    return xs, PchipInterpolator(x, y)(xs)


fig = plt.figure(figsize=(COL_W, 4.6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08,
                       left=0.20, right=0.97, top=0.86, bottom=0.115)
ax_cx = fig.add_subplot(gs[0])
ax_dp = fig.add_subplot(gs[1], sharex=ax_cx)

for r in CX_AVG:
    g = GEOM[r]
    for ax, avg, base in ((ax_cx, CX_AVG, C_CNOT), (ax_dp, DP_AVG, C_DEPTH)):
        col = shade(base, g['weight'])
        xs, ys = smooth(n_states, avg[r])
        ax.plot(xs, ys, '-', color=col)
        ax.plot(n_states, avg[r], linestyle='none', marker=g['marker'],
                color=col)

# Linear, now that the maxima are gone: the averages span 9.8-33.1, which a
# linear axis renders faithfully.  A log scale was only needed to hold them in
# the same panel as the ~130 maxima.
for ax, lab, hi in ((ax_cx, 'Average CNOTs', 34), (ax_dp, 'Average Depth', 37)):
    ax.set_ylabel(lab)
    ax.grid(True)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(5, hi)

plt.setp(ax_cx.get_xticklabels(), visible=False)
ax_dp.set_xticks(n_states)
ax_dp.set_xlim(0.8, 5.2)
ax_dp.set_xlabel(r'Number of States Targeted in $\mathrm{H}_2\mathrm{O}_2$')

CG = '0.25'
geom_h = [Line2D([], [], color=shade(CG, GEOM[r]['weight']), linestyle='-',
                 marker=GEOM[r]['marker'], label=GEOM[r]['label'])
          for r in CX_AVG]
pos = ax_cx.get_position()
leg = fig.legend(handles=geom_h, ncol=3, loc='lower center',
                 bbox_to_anchor=(pos.x0, pos.y1 + 0.012, pos.width, 0.04),
                 bbox_transform=fig.transFigure)
leg.get_frame().set_linewidth(0.6)

# NO tight_layout / bbox_inches='tight': the GridSpec margins ARE the layout.
plt.savefig(_os.path.join(_OUT, 'h2o2_circuit_col.pdf'))
plt.savefig(_os.path.join(_OUT, 'h2o2_circuit_col.png'))
print('wrote h2o2_circuit_col.pdf / .png')
