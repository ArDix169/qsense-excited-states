"""Shared figure style for the Q-SENSE paper.

Both figures import this so typography, line weights, grids and legend framing
are set in ONE place.  Previously each plotting cell carried its own rcParams
block with different values (font 10 vs 8, axes.linewidth 0.5 vs 0.7), which is
why the two figures read as different styles.

Sizes follow the JCTC graphics specification, not REVTeX:
  single column   <= 240 pt (3.33 in)      <- was 246 pt, which is over
  double column   300-504 pt (4.167-7 in)
  max depth       660 pt incl. caption (12 pt per caption line)
  lettering       >= 4.5 pt final
  line weights    >= 0.5 pt
  resolution      >= 300 dpi colour, 600 grayscale, 1200 B/W line art
"""
import os
import shutil
import numpy as np
import matplotlib as mpl
from matplotlib.colors import to_rgb

pt = 1. / 72.27
COL_W = 240. * pt          # JCTC single-column maximum
WIDE_W = 504. * pt         # JCTC double-column maximum

# A GUI-launched interpreter (PyCharm, Jupyter from the Dock) does not inherit
# the login shell's PATH, so `latex` and `kpsewhich` are missing even when TeX
# is installed.  Without kpsewhich matplotlib runs latex successfully but cannot
# resolve the fonts the resulting dvi references, and fails with the misleading
#     FileNotFoundError: ... searched for a file named 'cmr10.tfm' in your
#     texmf tree, but could not find it
# while the file is in fact present.  Prepending the directory fixes both.
_TEX_DIRS = ('/Library/TeX/texbin',
             '/usr/local/texlive/2024/bin/universal-darwin',
             '/opt/homebrew/bin', '/usr/local/bin', '/usr/bin')


def _ensure_tex_on_path():
    """True if `latex` is reachable, adding known TeX directories if needed."""
    if shutil.which('latex'):
        return True
    for d in _TEX_DIRS:
        if os.path.isfile(os.path.join(d, 'latex')):
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
            return True
    return False


# JCTC names two fonts.  For manuscript text: "The fonts 'Times' and 'Symbol'
# produce the best results."  For figure lettering (Appendix 2): "Helvetica or
# Arial fonts work well."  Neither is mandatory -- the hard requirements are
# >= 4.5 pt lettering and legibility at final size -- but Computer Modern
# matches neither and looks unlike the typeset page.  'times' is the default
# here because these axis labels carry a lot of math, and sfmath renders
# expressions such as $\epsilon^2 M$ sans-serif, which reads oddly.
_FONTS = {
    'times': ('serif', ['Times', 'Times New Roman'],
              r'\usepackage{amsmath}\usepackage{amssymb}'
              r'\usepackage{newtxtext}\usepackage{newtxmath}'),
    'helvetica': ('sans-serif', ['Helvetica', 'Arial'],
                  r'\usepackage{amsmath}\usepackage{amssymb}'
                  r'\usepackage{helvet}'
                  r'\renewcommand{\familydefault}{\sfdefault}'
                  r'\usepackage{sfmath}'),
    'cm': ('serif', ['Computer Modern Roman'],
           r'\usepackage{amsmath}\usepackage{amssymb}'),
}


def paper_style(usetex=True, font='cm'):
    """Apply the shared rcParams.

    font : 'cm' (default), 'times' (JCTC body-text convention), or 'helvetica'
           (their figure-lettering suggestion).
    usetex : False for quick previews.

    DEFAULT IS 'cm'.  newtxtext composes \\AA by stacking a ring accent over A,
    which sits visibly high and reads badly in 'O-O Bond Length (\\AA)'.
    Computer Modern has a proper glyph.  JCTC recommends Times for text and
    Helvetica/Arial for figure lettering, but neither is mandatory -- the hard
    requirements are >= 4.5 pt lettering and legibility at final size, both of
    which CM meets.

    If no TeX installation can be found, falls back to mathtext rather than
    failing at savefig time.  That fallback is close but NOT identical -- do
    not submit figures produced that way.
    """
    if usetex and not _ensure_tex_on_path():
        print('qsense_style: no LaTeX found on PATH, falling back to '
              'text.usetex=False (figures will differ slightly)')
        usetex = False
    family, names, preamble = _FONTS[font]
    mpl.rcParams.update({
        'text.usetex': usetex,
        'font.family': family,
        'font.serif': names if family == 'serif' else ['Times'],
        'font.sans-serif': names if family == 'sans-serif' else ['Helvetica'],
        'text.latex.preamble': preamble,
        'mathtext.fontset': 'stix' if font == 'times' else 'cm',
        'font.size': 8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'lines.linewidth': 1.4,
        'lines.markersize': 4.2,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'grid.linestyle': ':', 'grid.color': '0.85',
        'grid.linewidth': 0.5, 'grid.alpha': 1.0,
        'legend.frameon': True, 'legend.fancybox': False,
        'legend.edgecolor': '0.6', 'legend.framealpha': 1.0,
        'legend.borderpad': 0.35, 'legend.handletextpad': 0.25,
        'legend.columnspacing': 0.7, 'legend.handlelength': 1.4,
        'savefig.dpi': 650,
    })


def shade(color, w):
    """Blend `color` toward white.  w = 1 keeps it, w = 0 is white.

    Used instead of alpha for the lighter series: alpha composites the line
    with the dotted gridlines beneath it, which washes it out.  A solid light
    shade keeps the same visual ordering without losing contrast in print.
    """
    return tuple(1.0 - w * (1.0 - np.array(to_rgb(color))))


# Paul Tol 'bright', safe under the common colour-vision deficiencies.
# Irrep encoding belongs to the PES figures -- nothing else may reuse these two.
IRREP = {'A': '#0077BB', 'B': '#EE7733'}
STATE_MARKER = {1: 'o', 2: 's'}

# C2v needs four irreps where C2 needed two.  Chosen by SIMULATING protanopia,
# deuteranopia and tritanopia and maximising the worst-case CIELAB separation
# subject to WCAG's 3:1 non-text contrast on white.
#
# The obvious grouping -- A-family cool, B-family warm, with A1/B1 reusing
# H2O2's A/B hues -- gave '#EE7733' and '#CC3311' for B1/B2.  Those are adjacent
# warm hues that collapse under deuteranopia: worst-case dE 17.2, and minimum
# contrast 2.21, below spec.  This set scores 26.9 (1.56x better) at contrast
# 3.87.  A1 still keeps H2O2's A hue.
#
# Pairwise worst-case dE: A1/A2 26.9, B1/B2 33.0, every cross-family pair >= 35.5.
# The two tightest pairs are WITHIN a family, so a misread stays inside A or
# inside B rather than crossing the symmetry label.
IRREP_C2V = {'A1': '#0077BB',   # blue      (= IRREP['A'])
             'A2': '#332288',   # indigo
             'B1': '#D55E00',   # vermillion
             'B2': '#882255'}   # wine

SUB_LS = {1: '-', 2: '--'}

# Quantity encoding for the scaling figure, same palette family.
C_COST, C_BASIS = '#009988', '#AA3377'

# Circuit-resource figure.  Paul Tol's blue/orange high-contrast pair, which
# stays separable under protanopia, deuteranopia and tritanopia.  These are
# QUANTITY colours like C_COST/C_BASIS -- geometry is still carried by marker
# and shade, exactly as in the scaling figure, so the two read as one system.
C_CNOT, C_DEPTH = '#0077BB', '#EE7733'

# Geometry encoding, wherever geometries appear as series.
GEOM = {
    1.5:   {'label': r'1.5 \AA{} (eq.)',     'marker': 'o', 'weight': 1.00},
    1.875: {'label': r'1.875 \AA{} (corr.)', 'marker': 's', 'weight': 0.78},
    3.0:   {'label': r'3.0 \AA{} (diss.)',   'marker': 'v', 'weight': 0.58},
}

# H2O's three geometries are NOT H2O2's, so GEOM cannot be reused: 1.0 A is
# equilibrium here, and the hardness diagnostic puts the strongly correlated
# point at 1.5 A (eps_CISD 22.683 mHa, the scan maximum over 0.75-3.0 A).
# Markers and weights match GEOM position-for-position so the two scaling
# figures read as the same encoding across molecules.
GEOM_H2O = {
    1.0: {'label': r'1.0 \AA{} (eq.)',     'marker': 'o', 'weight': 1.00},
    1.5: {'label': r'1.5 \AA{} (corr.)',   'marker': 's', 'weight': 0.78},
    3.0: {'label': r'3.0 \AA{} (diss.)',   'marker': 'v', 'weight': 0.58},
}

FCI_KW = dict(color='black', linestyle='--', linewidth=0.8,
              dashes=(3.5, 1.8), alpha=1, zorder=1)
CHEM_ACC_KW = dict(color='red', linestyle='--', linewidth=0.9, zorder=5)
MARKER_KW = dict(markersize=4, markeredgecolor='white', markeredgewidth=0.4)
