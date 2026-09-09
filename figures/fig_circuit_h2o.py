"""Per-circuit resources vs number of targeted states, H2O.

Companion to fig_scaling_h2o.py -- same column width, same margins, same
geometry encoding (marker + shade), same PCHIP curves, so the two stack on the
page as one system.  Quantity is carried by colour: C_CNOT for gate count,
C_DEPTH for depth.

WHY THE AVERAGES ARE PLOTTED AND THE MAXIMA ARE TABLED.  Excluding the
classical-zero point, the averages span 7.9x (CNOT, 3.9-30.7) and 7.2x (depth,
5.3-38.4), while the maxima span 1.07x (61-65) and 1.07x (73-78).  The maxima
are pinned by the single deepest CSF preparation and barely move with n, so
plotting them would add six near-flat lines saying nothing a table cannot --
they go to tab:h2o-circuit-max.  The averages fall steeply because more and
more elements become classically evaluable and contribute a literal zero:
the subspace is not getting simpler circuits, it is getting FEWER of them.

WHY ALL FIVE n ARE SHOWN.  Several series wobble rather than fall
monotonically -- e.g. at 1.0 A the CNOT average rises 12.2 -> 12.3 from n=1 to
n=2 before falling to 3.9, and at 3.0 A it rises 12.8 -> 13.2 at n=3 -- so an
endpoint-only summary would misstate the shape.

WHAT THE AVERAGE IS OVER -- this is not obvious and the axis label cannot say
it.  Measurement_Benchmarking_Circuit_parallel.py pools N^2 values, N = n_ucsf:

  N        diagonal entries, one CSF state-preparation circuit per basis state
  N(N-1)   off-diagonal entries, one parallel-swap circuit per pair, appended
           TWICE ("off-diagonal used twice")

so the mean is dominated by the swap circuits at (N-1)/N of the pool.  Crucially
an element that is classically evaluable contributes a literal ZERO, via the
same `NQ == 0` branch that sets sigma = 0.  The averages therefore track the
CLASSICAL FRACTION as much as circuit complexity, while the maxima barely
move.  Fewer circuits, not simpler ones.

Source: QSENSE_paper_data/Scaling/H2O with its bench/ subdirectory, run
CAS(8e,6o), A1 singlet, h2o_sto3g_7o10e Hamiltonian, eps_1 = 1e-3,
eps_2 = 0, eps_3 = 1e-5 (ratio 5.0 x Ethrsh_ia 2e-6), l_max = 1, uniform across
all three geometries.

Worst |dE| = 0.899 mHa over every root of every n, against the full
CAS(10e,7o) FCI.  Same run as fig_scaling_h2o.py.
"""
import os as _os
_OUT = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator

from qsense_style import paper_style, shade, COL_W, C_CNOT, C_DEPTH, GEOM_H2O

paper_style()

n_states = [1, 2, 3, 4, 5]
geoms = [1.0, 1.5, 3.0]

# From the VO measurement benchmark over QSENSE_paper_data/Scaling/H2O
# (eps_1=1e-3, eps_2=0, eps_3=1e-5 [ratio 5.0 x Ethrsh_ia 2e-6], l_max=1),
#
# AVG is plotted.  MAX is NOT -- it is nearly flat in n (see the docstring) and
# goes to tab:h2o-circuit-max instead; the dicts stay here as that table's
# single source, so the two cannot drift apart.
CX_AVG = {
    1.0: [12.2, 12.3, 7.9, 3.9, 3.9],
    1.5: [19.6, 7.9, 7.9, 3.9, 0.0],
    3.0: [30.7, 12.8, 13.2, 13.2, 8.8],
}
CX_MAX = {
    1.0: [65, 63, 63, 63, 63],
    1.5: [65, 63, 63, 63, 0],
    3.0: [61, 61, 63, 65, 61],
}
DP_AVG = {
    1.0: [15.7, 16.1, 10.6, 5.3, 5.3],
    1.5: [24.4, 10.8, 10.8, 5.4, 0.0],
    3.0: [38.4, 16.7, 17.0, 16.9, 11.6],
}
DP_MAX = {
    1.0: [78, 78, 78, 78, 78],
    1.5: [78, 78, 78, 78, 0],
    3.0: [73, 78, 78, 78, 78],
}

# 1.5 A / n=5 is 0 because that subspace carries NO GENERATORS (0 of 31 basis
# states have an ia pair): every element is classically evaluable, so there is
# no circuit to count.  Plotted as a hollow marker with no line segment, the
# same convention fig_scaling_h2o.py uses for its zero cost -- joining it with
# a line would read as "the circuits got shallow" rather than "there are none".
CLASSICAL_ZERO = {1.5: 5}


def smooth(x, y, npts=300):
    """PCHIP: a cubic, but shape-preserving -- no overshoot, no invented
    extrema between integer n.  Same interpolant as fig_scaling_h2o.py."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return x, y
    xs = np.linspace(x.min(), x.max(), npts)
    return xs, PchipInterpolator(x, y)(xs)


fig = plt.figure(figsize=(COL_W, 4.4))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08,
                       left=0.22, right=0.97, top=0.88, bottom=0.12)
ax_cx = fig.add_subplot(gs[0])
ax_dp = fig.add_subplot(gs[1], sharex=ax_cx)

def _split(r, series):
    """(n, y) with the classical-zero point removed, plus that point alone.

    A zero here is not a small circuit, it is the ABSENCE of one, so it must not
    be interpolated through -- PCHIP would draw a plunge to zero that reads as a
    depth trend."""
    zn = CLASSICAL_ZERO.get(r)
    keep = [(n, v) for n, v in zip(n_states, series[r]) if n != zn]
    drop = [(n, v) for n, v in zip(n_states, series[r]) if n == zn]
    return keep, drop


for r in geoms:
    g = GEOM_H2O[r]
    for ax, data, base in ((ax_cx, CX_AVG, C_CNOT), (ax_dp, DP_AVG, C_DEPTH)):
        col = shade(base, g['weight'])
        keep, drop = _split(r, data)
        xk = [n for n, _ in keep]
        yk = [v for _, v in keep]
        xs, ys = smooth(xk, yk)
        ax.plot(xs, ys, '-', color=col)
        ax.plot(xk, yk, linestyle='none', marker=g['marker'], color=col)
        for n, _ in drop:                          # classical zero, hollow, no line
            ax.plot([n], [0], linestyle='none', marker=g['marker'],
                    color=col, markerfacecolor='white')

# "Average CNOTs", not "CNOTs per Element": the mean is over all N^2 matrix
# elements INCLUDING the classically-evaluable ones, which contribute exactly
# zero (the NQ == 0 branch of _diag_task/_offdiag_task returns cx = depth = 0).
# A per-element label implies a per-circuit cost, when most of the pool is
# zeros -- the caption carries the definition instead.  See the note below.
ax_cx.set_ylabel('Average CNOTs')
ax_cx.grid(True)
ax_cx.yaxis.set_major_locator(MultipleLocator(5))
ax_cx.set_yticks([t for t in ax_cx.get_yticks() if t >= 0])
ax_cx.set_ylim(-1.4, 34)
plt.setp(ax_cx.get_xticklabels(), visible=False)

ax_dp.set_ylabel('Average Depth')
ax_dp.grid(True)
ax_dp.yaxis.set_major_locator(MultipleLocator(5))
ax_dp.set_yticks([t for t in ax_dp.get_yticks() if t >= 0])
ax_dp.set_ylim(-1.7, 42)
ax_dp.set_xticks(n_states)
ax_dp.set_xlim(0.8, 5.2)
ax_dp.set_xlabel(r'Number of States Targeted in $\mathrm{H}_2\mathrm{O}$')

# Geometry encoding is identical in both panels -> one neutral-gray legend,
# anchored over the top panel exactly as the other figures' legends are.
CG = '0.25'
pos = ax_cx.get_position()
leg = fig.legend(handles=[Line2D([], [], color=shade(CG, GEOM_H2O[r]['weight']),
                                 linestyle='-', marker=GEOM_H2O[r]['marker'],
                                 label=GEOM_H2O[r]['label']) for r in geoms],
                 ncol=3, loc='lower center',
                 bbox_to_anchor=(pos.x0, pos.y1 + 0.012, pos.width, 0.04),
                 bbox_transform=fig.transFigure)
leg.get_frame().set_linewidth(0.6)

# NO tight_layout / bbox_inches='tight': the GridSpec margins ARE the layout.
plt.savefig(_os.path.join(_OUT, 'h2o_circuit_col.pdf'))
plt.savefig(_os.path.join(_OUT, 'h2o_circuit_col.png'))
print('wrote h2o_circuit_col.pdf / .png')
