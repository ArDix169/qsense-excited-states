# Data

The dumps, benchmark output, Hamiltonians and FCI references are archived on
Zenodo, not in this repository: the archive is ~224 MB and git is the wrong
place for it.

    bash data/fetch.sh

unpacks it here as `QSENSE_paper_release/`, which is what `analysis/verify_all.sh`
and the tier-1 commands in the top-level README expect.

## What the archive contains

    PES/H2O             80 sectors   10 geometries x 4 irreps x 2 spins
    PES/H2O2            32 sectors    8 geometries x 2 irreps x 2 spins
    Production/H2O      12 runs      + bench/ (VO measurement benchmark)
    Production/H2O2      6 runs      + the h2o2_VO_prod_*.json cost files
    Scaling/H2O         15 runs      + bench/
    Scaling/H2O2        15 runs      + bench/
    Hamiltonians/       CASSCF Hamiltonians + both full-space FCI references
    Baselines/          detbasis pickles behind the q-sc-EOM / SG-SSVQE columns

Every directory carries a `PROVENANCE.txt` giving the thresholds that produced
it and the verification result. This is not decoration: the dump filename
encodes neither `Ethrsh_select_ia` nor `Uopt_thrsh`, so two runs differing only
in those write identical filenames. Reading the numbers without the provenance
file is how the wrong set ended up cited once already.
