import os as _os
_OUT = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (FuncFormatter, LogLocator, NullLocator,
                               NullFormatter, LogFormatterMathtext, MultipleLocator)
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline, interp1d

from qsense_style import (paper_style, COL_W, IRREP, STATE_MARKER,
                          SUB_LS, FCI_KW, CHEM_ACC_KW, MARKER_KW)

paper_style()

# ==============================================================================
# Bond grids
# ==============================================================================
# NOTE: the FCI arrays below hold 17 points, so the grid must run 1.125 -> 3.125.

r_fci_values    = np.round(np.arange(1.125, 3.25, 0.125), 3)
r_qsense_values = np.array([1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])

N_FCI = len(r_fci_values)
N_Q   = len(r_qsense_values)

MS, MEW = MARKER_KW['markersize'], MARKER_KW['markeredgewidth']
fci_kw = dict(FCI_KW)

irrep_order  = ['A', 'B']
irrep_color  = IRREP
irrep_size   = {'A': MS, 'B': MS}
irrep_z      = {'A': 2, 'B': 3}
sub_ls       = SUB_LS
state_marker = STATE_MARKER

def blank(n):
    return {irr: {1: [None] * n, 2: [None] * n} for irr in irrep_order}

singlet_E_qsense = blank(N_Q)
singlet_E_fci    = blank(N_FCI)
triplet_E_qsense = blank(N_Q)
triplet_E_fci    = blank(N_FCI)

# ==============================================================================
#  QSENSE -- SINGLET (S_by2 = 0)
# ==============================================================================

singlet_E_qsense['A'][1] = [-148.81682968, -148.86601190, -148.83255641,
                            -148.80107515, -148.78494425, -148.77818330,
                            -148.77555582, -148.77473473354377]
singlet_E_qsense['A'][2] = [-148.37008394, -148.62540310, -148.72319071,
                            -148.75858918, -148.77043630, -148.77367826,
                            -148.77427881, -148.77427249037115]
singlet_E_qsense['B'][1] = [-148.37029973, -148.59124502, -148.71091797,
                            -148.75529068, -148.76950771, -148.77370786,
                            -148.77446272, -148.77449142]
singlet_E_qsense['B'][2] = [-148.30278957, -148.42824123, -148.52149675,
                            -148.54533512, -148.55065389, -148.55209253,
                            -148.55188479, -148.55153812]

# ==============================================================================
#  QSENSE -- TRIPLET (S_by2 = 2)
# ==============================================================================

triplet_E_qsense['A'][1] = [-148.42994683, -148.66699377, -148.74288444,
                            -148.76609867, -148.77301718, -148.77454630,
                            -148.77464713, -148.77451721]
triplet_E_qsense['A'][2] = [-148.38612357, -148.43806618, -148.51568383,
                            -148.54418237, -148.55091469, -148.55198289,
                            -148.55182162, -148.55151718]
triplet_E_qsense['B'][1] = [-148.41434118, -148.63558034, -148.72983770,
                            -148.76213733, -148.77223357, -148.77471566,
                            -148.77504415, -148.77491665]
triplet_E_qsense['B'][2] = [-148.37553093, -148.57395367, -148.70656823,
                            -148.75322092, -148.76841802, -148.77281562,
                            -148.77392358, -148.77416826]

# ==============================================================================
#  FCI
# ==============================================================================
singlet_E_fci['A'][1] = [-148.69571554, -148.81694966, -148.86158527, -148.86610522, -148.85205033, -148.83262419, -148.81483563, -148.80113382, -148.79147362, -148.78499607, -148.78082026, -148.77823845, -148.77671446, -148.77585456, -148.77538047, -148.77511079, -148.77494407]

singlet_E_fci['A'][2] = [-148.26494354, -148.37026114, -148.52278516, -148.62549498, -148.68654906, -148.72325866, -148.74543085, -148.75864498, -148.76627597, -148.77048355, -148.77266923, -148.77371923, -148.77416671, -148.77431673, -148.77433939, -148.77432404, -148.77430711]

singlet_E_fci['B'][1] = [-148.26673331, -148.37048483, -148.47529167, -148.59134210, -148.66510298, -148.71102166, -148.73891098, -148.75537596, -148.76476052, -148.76987679, -148.77251466, -148.77378141, -148.77433267, -148.77453536, -148.77458205, -148.77456700, -148.77453279]

singlet_E_fci['B'][2] = [-148.16663223, -148.30301187, -148.40355068, -148.42839959, -148.49019044, -148.52166759, -148.53751849, -148.54550483, -148.54947357, -148.55133064, -148.55207645, -148.55226453, -148.55220049, -148.55204371, -148.55186964, -148.55170905, -148.55157123]


triplet_E_fci['A'][1] = [-148.31292642, -148.43011779, -148.57744237, -148.66718458, -148.71623242, -148.74298414, -148.75784920, -148.76618832, -148.77075715, -148.77311203, -148.77421000, -148.77464295, -148.77475707, -148.77473814, -148.77467617, -148.77460950, -148.77455150]

triplet_E_fci['A'][2] = [-148.26297476, -148.38636419, -148.43682622, -148.43910510, -148.47905788, -148.51584532, -148.53487772, -148.54439051, -148.54898101, -148.55106914, -148.55190572, -148.55214166, -148.55211243, -148.55198341, -148.55183039, -148.55168461, -148.55155657]

triplet_E_fci['B'][1] = [-148.31167954, -148.41458231, -148.53472669, -148.63572504, -148.69506200, -148.72993390, -148.75039623, -148.76221183, -148.76880325, -148.77229184, -148.77400892, -148.77476886, -148.77504410, -148.77509193, -148.77504419, -148.77496437, -148.77488064]

triplet_E_fci['B'][2] = [-148.26151487, -148.37579567, -148.44778051, -148.57408432, -148.65689954, -148.70664623, -148.73612478, -148.75328847, -148.76306111, -148.76847237, -148.77137064, -148.77286765, -148.77361452, -148.77397625, -148.77414681, -148.77422508, -148.77426025]
# ==============================================================================



def compute_dE(E_q, E_f, r_q, r_f):
    dE = {irr: {s: [] for s in (1, 2)} for irr in irrep_order}
    for irr in irrep_order:
        for s in (1, 2):
            xq, yq = [], []
            for r, e in zip(r_q, E_q[irr][s]):
                if e is not None:
                    xq.append(r); yq.append(e)
            xq = np.array(xq); yq = np.array(yq)
            xf, yf = [], []
            for r, e in zip(r_f, E_f[irr][s]):
                if e is not None:
                    xf.append(r); yf.append(e)
            xf = np.array(xf); yf = np.array(yf)
            if len(xf) < 2 or len(xq) < 1:
                dE[irr][s] = [None] * len(r_q); continue
            f_interp = interp1d(xf, yf, kind="cubic", bounds_error=False,
                                fill_value="extrapolate")
            yf_on_q = f_interp(xq)
            dE[irr][s] = np.abs(yq - yf_on_q) * 1000.0
    return dE

singlet_dE = compute_dE(singlet_E_qsense, singlet_E_fci, r_qsense_values, r_fci_values)
triplet_dE = compute_dE(triplet_E_qsense, triplet_E_fci, r_qsense_values, r_fci_values)


def extract_xy(rvals, yvals):
    xs, ys = [], []
    for x, y in zip(rvals, yvals):
        if y is not None:
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def smooth_curve(x, y, n=300):
    if len(x) < 4:
        return x, y
    cs = CubicSpline(x, y, bc_type='natural')
    x_new = np.linspace(x.min(), x.max(), n)
    return x_new, cs(x_new)


# ==============================================================================
# FIGURE LAYOUT
# ==============================================================================

# JCTC single-column maximum is 240 pt; REVTeX's 246 pt is over spec.
fig = plt.figure(figsize=(COL_W, 6.5))

gs = gridspec.GridSpec(5, 1, height_ratios=[3.2, 1.5, 0.45, 3.2, 1.5],
                       hspace=0.08,left=0.22, right=0.97, top=0.85, bottom=0.10)

ax_s_pes = fig.add_subplot(gs[0, 0])
ax_s_err = fig.add_subplot(gs[1, 0], sharex=ax_s_pes)
ax_t_pes = fig.add_subplot(gs[3, 0])
ax_t_err = fig.add_subplot(gs[4, 0], sharex=ax_t_pes)

marker_kw = dict(
    markersize=MS,
    markeredgecolor='white',
    markeredgewidth=MEW
)


def plot_spin(ax_pes, ax_err, E_q, E_f, dE, r_q, r_f):
    for irr in irrep_order:
        c = irrep_color[irr]
        for s in (1, 2):
            ls = sub_ls[s]
            m  = state_marker[s]

            xf, yf = extract_xy(r_f, E_f[irr][s])
            if len(xf) > 0:
                xs, ys = smooth_curve(xf, yf)
                ax_pes.plot(xs, ys, **fci_kw)

            xq, yq = extract_xy(r_q, E_q[irr][s])
            if len(xq) > 0:
                pkw = {**marker_kw,
                      'markersize': irrep_size[irr],
                      'zorder': irrep_z[irr]}
                ax_pes.plot(xq, yq, linestyle='none', color=c, marker=m,
                            markerfacecolor=c, **pkw)

            xe, ye = extract_xy(r_q, dE[irr][s])
            if len(xe) > 0:
                ekw = {**marker_kw, 'linestyle': ls}
                ax_err.plot(xe, ye, color=c, marker=m, markerfacecolor=c, **ekw)

    ax_pes.set_ylabel('Energy (Ha)')
    ax_pes.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_pes.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_pes.grid(True)
    plt.setp(ax_pes.get_xticklabels(), visible=False)

    ax_err.xaxis.set_major_locator(MultipleLocator(0.5))

    ax_err.set_ylabel(r'$|\Delta E|$ (mHa)')
    ax_err.grid(True, which='both')

    ax_err.set_yscale('log')
    ax_err.axhline(y=1.6, label='Chemical Accuracy', **CHEM_ACC_KW)
    ax_err.set_ylim(1e-2, 5e0)
    ax_err.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax_err.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_err.yaxis.set_minor_locator(NullLocator())
    ax_err.yaxis.set_minor_formatter(NullFormatter())


plot_spin(ax_s_pes, ax_s_err, singlet_E_qsense, singlet_E_fci,
          singlet_dE, r_qsense_values, r_fci_values)
plot_spin(ax_t_pes, ax_t_err, triplet_E_qsense, triplet_E_fci,
          triplet_dE, r_qsense_values, r_fci_values)

# ==============================================================================
# SHARED LEGEND (built once, anchored over the top panel)
# ==============================================================================

irrep_h = [Line2D([], [], color=irrep_color[irr], marker='o', linestyle='none',
                  label=rf'{irr}',
                  **{**marker_kw, 'markersize': irrep_size[irr]}) for irr in irrep_order]
state_h = [Line2D([], [], color='0.45', marker=state_marker[s], linestyle='none',
                  label=f'root {s}', **marker_kw) for s in (1, 2)]
fci_h   = [Line2D([], [], color='Black', linestyle='--',
                  linewidth=0.8, dashes=(3.5, 1.8), label='FCI')]
chem_h  = [Line2D([], [], label='Chem. Acc.', **CHEM_ACC_KW)]

pos = ax_s_pes.get_position()

spacer_h = [Line2D([], [], linestyle='none', label=' ')] * 2

leg = fig.legend(handles=irrep_h + state_h + spacer_h + fci_h + chem_h, ncol=4,
                 loc='lower center',
                 bbox_to_anchor=(pos.x0, 0.895, pos.width, 0.04),
                 bbox_transform=fig.transFigure,
                 frameon=True, framealpha=1.0, edgecolor='0.6',
                 fancybox=False, borderpad=0.35,
                 columnspacing=0.7, handletextpad=0.25)

leg.get_frame().set_linewidth(0.6)

# ==============================================================================
# AXIS LIMITS + INSETS
# ==============================================================================


def draw_series(ax, E_q, E_f, r_q, r_f):
    """Bare PES redraw used for the inset axes."""
    for irr in irrep_order:
        c = irrep_color[irr]
        for s in (1, 2):
            xf, yf = extract_xy(r_f, E_f[irr][s])
            if len(xf):
                xs, ys = smooth_curve(xf, yf)
                ax.plot(xs, ys, **fci_kw)

            xq, yq = extract_xy(r_q, E_q[irr][s])
            if len(xq):
                pkw = {**marker_kw,
                       'markersize': irrep_size[irr],
                       'zorder': irrep_z[irr]}
                ax.plot(xq, yq, linestyle='none', color=c, marker=state_marker[s],
                        markerfacecolor=c, **pkw)


def style_inset(axins):
    axins.xaxis.set_major_locator(MultipleLocator(0.25))
    axins.yaxis.set_major_locator(MultipleLocator(0.002))
    axins.yaxis.set_major_formatter(FuncFormatter(lambda v, p: rf'${v:.3f}$'))
    axins.tick_params(labelsize=6, width=0.5, length=1.8, pad=1.5)
    axins.grid(True, linestyle=':', color='0.8', linewidth=0.5)
    for sp in axins.spines.values():
        sp.set_linewidth(0.6)


ax_s_pes.set_ylim(-148.95, -148.2)
ax_t_pes.set_ylim(-148.8, -148.2)

axins_s = ax_s_pes.inset_axes([0.56, 0.65, 0.40, 0.30])
draw_series(axins_s, singlet_E_qsense, singlet_E_fci,
            r_qsense_values, r_fci_values)
axins_s.set_xlim(2.45, 3.05)
axins_s.set_ylim(-148.779, -148.7715)
style_inset(axins_s)

axins_t = ax_t_pes.inset_axes([0.56, 0.62, 0.40, 0.30])
draw_series(axins_t, triplet_E_qsense, triplet_E_fci,
            r_qsense_values, r_fci_values)
axins_t.set_xlim(2.45, 3.05)
axins_t.set_ylim(-148.7755, -148.7715)
style_inset(axins_t)

for ax in (ax_s_pes, ax_s_err, ax_t_pes, ax_t_err):
    ax.set_xlim(1.1, 3.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax_s_pes.axvspan(2.45, 3.05, 0.21, 0.25, color='0.75', alpha=0.6, zorder=0)
ax_t_pes.axvspan(2.45, 3.05, 0.02, 0.07, color='0.75', alpha=0.6, zorder=0)

plt.setp(ax_s_err.get_xticklabels(), visible=False)

for ax in (ax_s_pes, ax_s_err, ax_t_pes, ax_t_err):
    ax.tick_params(axis='y', labelsize=10)

ax_s_pes.set_title('Singlet', pad=3)
ax_t_pes.set_title('Triplet', pad=3)

fig.supxlabel(r'O$-$O Bond Length (\AA)', x=0.595, y=0.04)
fig.align_ylabels([ax_s_pes, ax_s_err, ax_t_pes, ax_t_err])

plt.savefig(_os.path.join(_OUT, 'h2o2_stacked.pdf'))
plt.savefig(_os.path.join(_OUT, 'h2o2_stacked.png'))   # convenience preview; the PDF is the deliverable
