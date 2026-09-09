"""Sampling cost and subspace size vs number of targeted states, H2O.

Direct sister of fig_scaling_h2o2.py -- same two panels (eps^2 M on top, basis
states below), same x-axis (number of targeted states), same margins, so the
pair aligns on the page and reads as one encoding across the two molecules.

Only the geometries differ.  H2O2's are 1.5 / 1.875 / 3.0 A; H2O's equilibrium
is 1.0 A and its strongly correlated point is 1.5 A, which is where the
hardness diagnostic peaks (eps_CISD 22.683 mHa over the 0.75-3.0 A scan).
Hence GEOM_H2O rather than GEOM, with markers and weights matched
position-for-position.

Subspace: CAS(8e,6o), irrep A1, singlet, eps_1 = 1e-3, eps_2 = 0, eps_3 = 1e-5
(ratio 5.0 x Ethrsh_ia 2e-6), l_max = 1.

Accuracy is set by the product ratio x Ethrsh, the basis-extension threshold:
it passes at <= 1e-5 and fails sharply above, so 1e-5 is the loosest accurate
choice -- and loosest is what we want, since a smaller product only grows the
subspace. eps_1 has no effect in 1e-3..1e-5.

Worst |dE| = 0.899 mHa over every root of every n, against the full
CAS(10e,7o) FCI. BASIS, COST and GENERATORS below come from one run.

WHERE THE COST COMES FROM.  eps^2 M is produced by the VO measurement benchmark
(Measurement_Benchmarking_Circuit_parallel.py), run on Trillium via
runs/run_meas_bench_h2o_nstates.sh and gathered by analysis/collect_meas_h2o_nstates.py.
This script READS those JSONs -- it does not carry cost numbers inline, because
a hardcoded cost table silently goes stale the moment the thresholds move.
Point RESULTS at the collected directory:

    python3 fig_scaling_h2o.py /path/to/results_h2o_nstates
    RESULTS=/path/to/results_h2o_nstates python3 fig_scaling_h2o.py

The basis sizes ARE inline, as a cross-check: they come from the Q-SENSE JSONs
written beside each dump (n_ucsf) and must agree with the benchmark's
'basis_states'.  A mismatch means the benchmark ran on a different dump than the
one these numbers came from, and the script says so rather than plotting it.

Straight segments, no interpolation -- same reason as the H2O2 figure: with
five x-values and non-monotonic series a spline invents overshoot.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator

from qsense_style import paper_style, shade, COL_W, C_COST, C_BASIS, GEOM_H2O

paper_style()

n_states = [1, 2, 3, 4, 5]

# geoms is every geometry the study covers -- the data tables below are keyed by
# it and the loader validates against it.
geoms = [1.0, 1.5, 3.0]

# All three are drawn, though 3.0 A's costs run
# 1.2e-13 to 7.2e-8 and stretching the axis down there compressed the other two
# into the top fifth of the panel.  It is now shown as UPPER LIMITS at
# CENSOR_AT -- the standard convention for censored data -- which keeps the
# geometry in the comparison without spending twelve decades of axis on
# resolving values that carry no operational meaning.
PLOT_GEOMS = [1.0, 1.5, 3.0]

# Censoring threshold: ONE SHOT.
#
# Cost = eps^2 * M_tot, so M_tot < 1 is exactly cost < eps^2 = 2.56e-6 Ha^2.
# Below this line the ENTIRE subspace needs less than a single measurement, so
# the precise value is operationally meaningless -- 1e-8 and 1e-13 both say
# "no measurement required", and resolving between them costs five decades of
# axis to communicate nothing.
#
# Derived, not chosen.  A round 1e-6 would sit within a factor of 2.6 of this
# and look the same, but eps^2 is defensible under questioning: it is the cost
# of one shot at the stated target accuracy.
CENSOR_AT = 1.6e-3 ** 2

# n_ucsf from the Q-SENSE run JSONs, QSENSE_ES_dump/h2o_A1_nstates_7o10e_opt,
# files h2o_sto3g_7o10e_UCSF_<n>_A1_5.0_S0_T1e-04_C1_for_Arjun_<r>.json.
BASIS = {
    1.0: [19, 28, 29, 30, 30],
    1.5: [25, 29, 29, 30, 31],
    3.0: [11, 25, 25, 25, 27],
}

# Generators per geometry: mean ia pairs per basis state, from the VO benchmark
# ('generators'.avg).  Totals are 7/9/6/3/3, 14/6/6/3/0 and 14/10/10/10/8 --
# the 1.5 A n=5 zero is what makes that subspace fully classical, see below.
GENERATORS = {
    1.0: [0.37, 0.32, 0.21, 0.10, 0.10],
    1.5: [0.56, 0.21, 0.21, 0.10, 0.00],
    3.0: [1.27, 0.40, 0.40, 0.40, 0.30],
}

# eps^2 M from the VO measurement benchmark on Trillium over
# QSENSE_paper_data/Scaling/H2O (h2o_sto3g_7o10e, eps_1=1e-3, eps_2=0,
# eps_3=1e-5 [ratio 5.0 x Ethrsh_ia 2e-6], l_max=1), results in that
# directory's bench/ subdirectory.  The benchmark's basis_states was checked
# against n_ucsf for all 15 runs: they agree, so the cost and basis panels
# describe the SAME dumps.
#
# A cost of exactly 0.0 is a REAL RESULT, not a gap -- see CLASSICAL_ZERO below.
COST = {
    1.0: [3.1266e-04, 2.2112e-02, 1.3214e-02, 2.5468e-03, 1.9509e-02],
    1.5: [5.4409e-01, 5.5172e-03, 5.9639e-03, 6.7181e-03, 0.0000e+00],
    3.0: [2.3104e-04, 1.4544e-04, 3.3755e-06, 5.7873e-06, 3.5921e-07],
}

# 1.5 A / n=5 costs exactly zero because that subspace carries NO GENERATORS --
# verified directly in the dump: 0 of 31 basis states have any ia pair, against
# 3 of 30 at n=4 and 6 of 29 at n=3.  With no generators every matrix element is
# fully classically evaluable, the benchmark returns sigma = 0 by design
# (Measurement_Benchmarking_Circuit_parallel.py, the `NQ == 0` branch), and the
# subspace needs no quantum measurement at all.
#
# It cannot be drawn on a log axis, and dropping it would delete the strongest
# instance of the paper's own claim -- that cost tracks retained quantum
# structure, not dimension.  It is therefore drawn as an OPEN marker on the
# axis floor with a downward tick, visually distinct from every measured point.
# Implemented in the plotting loop below: exact zeros and sub-one-shot values
# alike are drawn as open markers AT the one-shot threshold (2.56e-6 Ha^2).
CLASSICAL_ZERO = 'open marker at the one-shot threshold'


RESULTS = (sys.argv[1] if len(sys.argv) > 1
           else os.environ.get('RESULTS', 'results_h2o_nstates'))


def load_costs():
    """{r: [eps^2 M per n]} from the VO benchmark JSONs, overriding COST.

    Falls back to the inline COST table when RESULTS is absent, so the figure
    is reproducible from this file alone.  When the JSONs ARE present they win,
    and any disagreement with COST is reported -- that is the signal the inline
    table has gone stale against a newer run.
    """
    if not os.path.isdir(RESULTS):
        print(f'fig_scaling_h2o: no benchmark directory at {RESULTS!r}, '
              f'using the inline COST table (see its provenance comment).')
        return {r: list(COST[r]) for r in geoms}

    cost, missing, mismatch = {}, [], []
    for r in geoms:
        series = []
        for i, n in enumerate(n_states):
            p = os.path.join(RESULTS, f'h2o_VO_benchmark_n{n}_r{r}.json')
            if not os.path.exists(p):
                missing.append(p)
                series.append(np.nan)
                continue
            with open(p) as f:
                d = json.load(f)
            series.append(float(d['sampling_cost']))
            if int(d['basis_states']) != BASIS[r][i]:
                mismatch.append(
                    f'  r={r} n={n}: benchmark basis_states={d["basis_states"]} '
                    f'but BASIS says {BASIS[r][i]}')
        cost[r] = series

    if mismatch:
        raise SystemExit(
            'fig_scaling_h2o: the benchmark ran on a different subspace than '
            'BASIS describes --\n' + '\n'.join(mismatch) +
            '\nEither the dumps were regenerated at new thresholds or RESULTS '
            'points at the wrong run.  Refusing to plot a mixed figure.')

    if missing and len(missing) == len(geoms) * len(n_states):
        print(f'fig_scaling_h2o: {RESULTS!r} holds no benchmark JSONs, '
              f'using the inline COST table.')
        return {r: list(COST[r]) for r in geoms}

    if missing:
        raise SystemExit(
            f'fig_scaling_h2o: {len(missing)} of {len(geoms) * len(n_states)} '
            f'benchmark JSONs not found under {RESULTS!r}.\n'
            f'  first missing: {missing[0]}\n\n'
            'eps^2 M comes from the VO measurement benchmark, which has to run '
            'before this figure can be drawn:\n'
            '  sbatch runs/run_meas_bench_h2o_nstates.sh\n'
            'then rerun with RESULTS pointing at results_h2o_nstates.')

    return cost


COST = load_costs()

# BROKEN_AXIS splits the cost panel in two log sub-panels.  The series span
# twelve decades -- 1.2e-13 at 3.0 A against 0.585 at 1.5 A -- so on one axis
# the two expensive geometries collapse into the top fifth of the panel and
# their n-dependence is unreadable.  The break restores it without distorting
# anything: both halves stay logarithmic and the gap is still visible as a gap.
# Set to False for the single-axis version, where the empty middle of the panel
# IS the message and nothing needs explaining.
BROKEN_AXIS = os.environ.get('BROKEN_AXIS', '0') != '0'


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

# Axis range is set by the UNCENSORED points only.  Including the censored
# values is what forced the twelve-decade axis in the first place.
_shown = [v for r in PLOT_GEOMS for v in COST[r]
          if np.isfinite(v) and v >= CENSOR_AT]
DLO, DHI = min(_shown), max(_shown)

fig = plt.figure(figsize=(COL_W, 4.4))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08,
                       left=0.22, right=0.97, top=0.88, bottom=0.12)
ax_cost = fig.add_subplot(gs[0])
ax_basis = fig.add_subplot(gs[1], sharex=ax_cost)

_any_censored = any(np.isfinite(v) and v < CENSOR_AT
                    for r in PLOT_GEOMS for v in COST[r])
# Both bounds track the data by a fixed log-space margin (0.15 dex, ~1.4x) --
# NOT rounded to the nearest power of ten, which is what was leaving so much
# empty space on both ends (e.g. DHI=0.737 was rounding up to HI=10, a full
# decade of headroom for a 1.4x margin). LO still anchors to CENSOR_AT when
# something IS censored, so the threshold line/marker stays on-axis.
LO = (10 ** (np.log10(CENSOR_AT) - 0.45) if _any_censored
      else 10 ** (np.log10(DLO) - 0.3))
HI = 10 ** (np.log10(DHI) + 0.3)

has_censored = False
for r in PLOT_GEOMS:
    g = GEOM_H2O[r]
    col = shade(C_COST, g['weight'])
    cost = np.asarray(COST[r], dtype=float)
    shown = cost >= CENSOR_AT           # measured; plotted at its true value
    # An EXACT zero is not a small number -- it is the absence of any quantum
    # measurement (no generators => every element classically evaluable => the
    # benchmark's NQ == 0 branch returns sigma = 0).  Drawing it as a "< 1 shot"
    # upper limit would claim we merely failed to resolve it, so it gets its own
    # marker at the axis floor.  Log axes cannot show zero any other way.
    classical = cost == 0.0
    cens = ~shown & ~classical          # below one shot; plotted as a limit

    # Measured points, smooth curve through them.  A censored point is NOT
    # interpolated through -- its value is unknown at this resolution, so a
    # line into it would draw a number that was never measured.
    if shown.sum() >= 3:
        xs, ys = smooth(np.asarray(n_states)[shown], cost[shown], log=True)
        ax_cost.plot(xs, ys, '-', color=col)
    elif shown.sum() == 2:
        ax_cost.plot(np.asarray(n_states)[shown], cost[shown], '-', color=col)
    if shown.any():
        ax_cost.plot(np.asarray(n_states)[shown], cost[shown], linestyle='none',
                     marker=g['marker'], color=col)

    # Everything below one shot -- whether a small measured value or an exact
    # zero -- is drawn AT the threshold as an open marker, never at its own
    # value.  Two reasons: a log axis cannot place zero at all, and eps^2 M
    # below the threshold corresponds to M < 1 in the continuous allocation
    # model, i.e. a shot count the model cannot deliver.  Open (unfilled) says
    # "not a measured value"; the geometry marker is kept so the series is
    # still identifiable.  The caption states the cutoff.
    below = cens | classical
    if below.any():
        has_censored = True
        ax_cost.plot(np.asarray(n_states)[below],
                     np.full(below.sum(), CENSOR_AT), linestyle='none',
                     marker=g['marker'], color=col, markerfacecolor='white',
                     markeredgewidth=1.0, markersize=5.5, zorder=7)

    cb = shade(C_BASIS, g['weight'])
    xs, ys = smooth(n_states, BASIS[r])
    ax_basis.plot(xs, ys, '-', color=cb)
    ax_basis.plot(n_states, BASIS[r], linestyle='none', marker=g['marker'],
                  color=cb)

ax_cost.set_yscale('log')
ax_cost.grid(True, which='both')
ax_cost.set_ylim(LO, HI)
ax_cost.set_ylabel(r'Sampling Cost $(\epsilon^2M)$')
plt.setp(ax_cost.get_xticklabels(), visible=False)

if has_censored:
    ax_cost.axhline(CENSOR_AT, color='0.35', linestyle='--', linewidth=0.8,
                    dashes=(4, 2), zorder=2)
    # NO in-plot label.  "<1 shot" is a shot count, but this axis is eps^2 M in
    # Ha^2 -- writing it here labels the axis with the wrong units.  The
    # threshold line stays as a visual reference and the caption defines it.

CG = '0.25'
pos_ = ax_cost.get_position()
handles = [Line2D([], [], color=shade(CG, GEOM_H2O[r]['weight']),
                  linestyle='-', marker=GEOM_H2O[r]['marker'],
                  label=GEOM_H2O[r]['label']) for r in PLOT_GEOMS]
leg = fig.legend(handles=handles, ncol=3, loc='lower center',
                 bbox_to_anchor=(pos_.x0, pos_.y1 + 0.012, pos_.width, 0.04),
                 bbox_transform=fig.transFigure)
leg.get_frame().set_linewidth(0.6)

ax_basis.set_ylabel('Basis States')
# Ticks every 2 states.  The auto-locator put them at 17.5, 20.0, 22.5 ... --
# half-integer labels on a count of basis states, which cannot take those
# values.  MultipleLocator(2) also keeps the labels integral.
ax_basis.yaxis.set_major_locator(MultipleLocator(2))
ax_basis.grid(True)
ax_basis.set_xticks(n_states)
ax_basis.set_xlabel(r'Number of States Targeted in $\mathrm{H}_2\mathrm{O}$')
ax_basis.set_xlim(0.8, 5.2)

CG = '0.25'
pos_ = ax_cost.get_position()
leg = fig.legend(handles=[Line2D([], [], color=shade(CG, GEOM_H2O[r]['weight']),
                                 linestyle='-', marker=GEOM_H2O[r]['marker'],
                                 label=GEOM_H2O[r]['label'])
                          for r in PLOT_GEOMS],
                 ncol=3, loc='lower center',
                 bbox_to_anchor=(pos_.x0, pos_.y1 + 0.012, pos_.width, 0.04),
                 bbox_transform=fig.transFigure)
leg.get_frame().set_linewidth(0.6)

# NO tight_layout / bbox_inches='tight': the GridSpec margins ARE the layout,
# and either would override them and change the final width.
plt.savefig('h2o_cost_basis_col.pdf')
plt.savefig('h2o_cost_basis_col.png')
print('wrote h2o_cost_basis_col.pdf / .png')
