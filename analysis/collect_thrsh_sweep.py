"""
Collect and tabulate the CSF-threshold sweep at fixed geometry.

Reads the JSONs written by CSF_UCSF_GS.py into QSENSE_ES_dump/ for a single
(rdist, irrep, S_by2) and lays them out as a no_states x csf_small_thrsh grid.

The point of the sweep is to see how much the CSF small-amplitude cutoff
(csf_small_thrsh) costs you in accuracy.  The tightest threshold keeps the most
CSFs, so it is used as the reference and every looser threshold is reported as a
deviation from it in mHa.  A threshold is "safe" for a given root if that
deviation stays inside chemical accuracy (1.6 mHa).

Usage (on Trillium, from $SCRATCH/Q-SENSE):
    python hpc/collect_thrsh_sweep.py                    # defaults: r=1.5, A, singlet
    python hpc/collect_thrsh_sweep.py --rdist 2.75 --irrep B --s-by2 2
"""

import argparse
import glob
import json
import os

CHEM_ACC_MHA = 1.6


def load(dump_dir, rdist, irrep, s_by2):
    """Return {(no_states, thrsh): [energies]} for the requested sector.

    Older JSONs predate the csf_small_thrsh / S_by2 fields, so both are
    recovered from the filename ('..._S0_T1e-04_for_Arjun_1.5.json') whenever
    they are absent from the file contents.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(dump_dir, '*_for_Arjun_*.json'))):
        with open(path) as f:
            d = json.load(f)
        base = os.path.basename(path)

        thrsh = d.get('csf_small_thrsh')
        if thrsh is None:
            if '_T' not in base:
                continue                      # pre-patch file, threshold unknown
            thrsh = float(base.split('_T')[1].split('_')[0])
        spin = d.get('S_by2')
        if spin is None:
            spin = 2 if '_S2_' in base else 0

        if abs(float(d['rdist']) - rdist) > 1e-9:
            continue
        if str(d['irrep_label_choice']) != irrep or int(spin) != int(s_by2):
            continue

        out[(int(d['no_states']), float(thrsh))] = list(d['output_energy'])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dump-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'QSENSE_ES_dump'))
    p.add_argument('--rdist', type=float, default=1.5)
    p.add_argument('--irrep', default='A')
    p.add_argument('--s-by2', type=int, default=0)
    args = p.parse_args()

    data = load(args.dump_dir, args.rdist, args.irrep, args.s_by2)
    if not data:
        raise SystemExit(f'No JSONs matched r={args.rdist} irrep={args.irrep} '
                         f'S_by2={args.s_by2} in {args.dump_dir}')

    ns = sorted({k[0] for k in data})
    thrshs = sorted({k[1] for k in data}, reverse=True)   # loosest -> tightest
    ref_thrsh = thrshs[-1]                                # tightest = reference
    spin = 'singlet' if args.s_by2 == 0 else 'triplet'

    print(f'\nQ-SENSE CSF threshold sweep   r = {args.rdist} A   '
          f'irrep {args.irrep}   {spin}')
    print(f'{len(data)} runs   reference threshold = {ref_thrsh:.0e}\n')

    # --- absolute energies ---
    print('=' * 78)
    print('ABSOLUTE ENERGIES (Ha)')
    print('=' * 78)
    for n in ns:
        print(f'\n  no_states = {n}')
        header = '    root  ' + ''.join(f'{t:>16.0e}' for t in thrshs)
        print(header)
        print('    ' + '-' * (len(header) - 4))
        nroot = max((len(data[(n, t)]) for t in thrshs if (n, t) in data),
                    default=0)
        for r in range(nroot):
            cells = ''
            for t in thrshs:
                e = data.get((n, t))
                cells += f'{e[r]:>16.8f}' if e and r < len(e) else f'{"--":>16}'
            print(f'    {r:>4}  {cells}')

    # --- deviation from the tightest threshold ---
    print('\n' + '=' * 78)
    print(f'DEVIATION FROM T = {ref_thrsh:.0e}  (mHa)   * = outside '
          f'{CHEM_ACC_MHA} mHa chemical accuracy')
    print('=' * 78)
    for n in ns:
        ref = data.get((n, ref_thrsh))
        if ref is None:
            print(f'\n  no_states = {n}: reference run missing, skipped')
            continue
        print(f'\n  no_states = {n}')
        loose = [t for t in thrshs if t != ref_thrsh]
        header = '    root  ' + ''.join(f'{t:>14.0e}' for t in loose)
        print(header)
        print('    ' + '-' * (len(header) - 4))
        for r in range(len(ref)):
            cells = ''
            for t in loose:
                e = data.get((n, t))
                if not e or r >= len(e):
                    cells += f'{"--":>14}'
                    continue
                d_mha = (e[r] - ref[r]) * 1e3
                mark = '*' if abs(d_mha) > CHEM_ACC_MHA else ' '
                cells += f'{d_mha:>13.3f}{mark}'
            print(f'    {r:>4}  {cells}')

    # --- largest deviation per threshold, across all roots and all n ---
    print('\n' + '=' * 78)
    print(f'MAX |deviation| from T = {ref_thrsh:.0e} over all roots and all n')
    print('=' * 78)
    for t in thrshs:
        if t == ref_thrsh:
            continue
        worst, where = 0.0, None
        for n in ns:
            ref, e = data.get((n, ref_thrsh)), data.get((n, t))
            if not ref or not e:
                continue
            for r in range(min(len(ref), len(e))):
                d_mha = abs(e[r] - ref[r]) * 1e3
                if d_mha > worst:
                    worst, where = d_mha, (n, r)
        if where is None:
            print(f'  T = {t:.0e}:  no comparable runs')
        else:
            verdict = 'OK' if worst <= CHEM_ACC_MHA else 'exceeds chem acc'
            print(f'  T = {t:.0e}:  {worst:9.3f} mHa   '
                  f'(n={where[0]}, root {where[1]})   {verdict}')
    print()


if __name__ == '__main__':
    main()
