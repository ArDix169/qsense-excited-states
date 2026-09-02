# Data

The archived dumps, benchmark output, Hamiltonians and FCI references ship in
this repository under `QSENSE_paper_release/`. No download step is needed:

    bash analysis/verify_all.sh

reads them directly and re-derives every number in the paper.

## Contents

    PES/H2O             80 sectors   10 geometries x 4 irreps x 2 spins
    PES/H2O2            32 sectors    8 geometries x 2 irreps x 2 spins
    Production/H2O      12 runs      + bench/ (VO measurement benchmark)
    Production/H2O2      6 runs      + the h2o2_VO_prod_*.json cost files
    Scaling/H2O         15 runs      + bench/
    Scaling/H2O2        15 runs      + bench/
    Hamiltonians/       CASSCF Hamiltonians + both full-space FCI references
    Baselines/          detbasis pickles behind the q-sc-EOM / SG-SSVQE columns

408 files, 224 MB checked out (~17 MB compressed, which is what a clone costs).

Every directory carries a `PROVENANCE.txt` giving the thresholds that produced
it and its verification result. That is not decoration: the dump filename
encodes neither `Ethrsh_select_ia` nor `Uopt_thrsh`, so two runs differing only
in those write identical filenames. Reading the numbers without the provenance
file is how a superseded set was cited once already.

## Reference pickles

`Hamiltonians/*_fci_ref_full.pkl` key each sector twice:

    (irrep, spin)           FULL active space   <- use this one
    (irrep, spin, 'cas')    frozen-core sector

Use the 2-tuple. See the FCI reference section of the top-level README for why.
