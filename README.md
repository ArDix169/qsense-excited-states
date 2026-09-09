# Q-SENSE for excited states — data and analysis

Code and data accompanying *Symmetry-Preserving Seniority-Based Orthogonal
Quantum Subspace Expansion for Electronic Excited States*.

Q-SENSE targets several excited states at once by building a compact subspace
of unitary-rotated CSFs. This repository holds everything needed to check or
regenerate the paper's numbers for H₂O and H₂O₂ in STO-3G, and to compare them
against q-sc-EOM and SS-SSSA-VQE.

## Reproduction tiers

Pick the cheapest tier that answers your question. Tier 1 re-derives every
number in the paper and takes minutes.

| tier | what you get | cost | needs |
|------|--------------|------|-------|
| 0 | every figure redrawn | seconds | this repo |
| 1 | **every number re-verified against FCI** | minutes | this repo |
| 2 | subspace dumps regenerated | HPC-days | + SLURM |
| 3 | Hamiltonians and FCI references regenerated | HPC-days | + PySCF |

### Tier 0 — figures

```bash
conda env create -f environment/qsense.yml && conda activate qsense
python figures/fig_pes_h2o.py          # → figures/h2o_stacked.pdf
python figures/fig_scaling_h2o.py
python figures/fig_circuit_h2o.py
python figures/fig_dim_frac_h2o.py
python figures/fig_pes_h2o2.py
python figures/fig_scaling_h2o2.py
python figures/fig_circuit_h2o2.py
```

The figures carry their numbers inline, so tier 0 does not read the data at
all. Each script's docstring records the run that produced its numbers, the
thresholds, and the verification result.

### Tier 1 — verify every number

```bash
bash analysis/verify_all.sh 2>&1 | tee verify_all.log
```

The data ships in the repository under `data/QSENSE_paper_release/`, so there
is nothing to download. A clone costs ~24 MB; the checked-out data is 224 MB.

This differences every Q-SENSE energy against full-space FCI, across all seven
studies, and reports the worst deviation and any root outside chemical
accuracy (1.6 mHa). Expected result — the state of the paper as published:

| study | worst \|ΔE\| (mHa) | outside chem. acc. |
|---|---|---|
| H₂O PES (80 sectors) | 1.358 | 0 |
| H₂O production | 0.84 | 0 |
| H₂O scaling | 0.899 | 0 |
| H₂O₂ PES (32 sectors) | 1.042 | 0 |
| H₂O₂ production | reproduces the tables exactly | 0 |
| H₂O₂ scaling, 1.5 / 1.875 Å | 1.265 | 0 |
| H₂O₂ scaling, 3.0 Å | 0.853 | 0 |

Individual studies: `bash analysis/verify_all.sh h2o_pes` (also `h2o_prod`,
`h2o_scaling`, `h2o2_prod`, `h2o2_scaling`, `h2o2_pes`).

### Tiers 2–3 — regenerate from scratch

Requires SLURM. `runs/` holds the batch scripts; each header documents its grid, thresholds,
and why they were chosen. Hamiltonians and FCI references come first:

```bash
python hamiltonians/h2o_sto3g_full.py          # 0.75–3.0 Å, writes both the
python hamiltonians/h2o2_sto3g_full.py         # Hamiltonians and the FCI refs
sbatch runs/run_h2o_pes.sh         # 80-task array
```

## The FCI reference convention

**Deviations are taken against the FULL active space, not the frozen-core
sector.** The reference pickles key both: `(irrep, spin)` is the full space,
`(irrep, spin, 'cas')` is frozen-core. Use the 2-tuple.

Q-SENSE's `actmo_start` restricts which excitations the *ansatz* generates; it
does not reduce the Hamiltonian, which still spans the full space. So the
full-space FCI is the energy the ansatz is actually approximating. Differencing
against the frozen-core sector answers a narrower question — how well the
ansatz solves its own restricted manifold — and understates the error by
~0.15 mHa. It can also produce a *negative* deviation, which is variationally
impossible and is the fastest signal that the wrong reference is in use.

## Layout

```
qsense/       the subspace construction itself
measurement/  the VO measurement benchmark
baselines/    q-sc-EOM and SS-SSSA-VQE (determinant-basis benchmark)
hamiltonians/ CASSCF Hamiltonian and full-space FCI reference generators
runs/         SLURM batch scripts (thresholds and grids documented in-header)
analysis/     collectors that rebuild each table from raw output + verify_all.sh
figures/      one script per figure; numbers inline, provenance in the docstring
tables/       generated LaTeX
data/         the archived dumps, benchmarks and FCI references
environment/  two conda environments, deliberately separate
```

Two environments because the measurement benchmark needs qiskit and tequila,
and installing those beside the Q-SENSE stack moves numpy and scipy underneath
running subspace sweeps.

## Method code

`qsense/` builds the subspace and `measurement/` costs it out. Tiers 0 and 1
touch neither — the archived data and the figures' inlined numbers verify every
claim in the paper on their own — so read this section only if you are
regenerating results.

```
qsense/
  qsense_subspace.py     entry point; the scripts in runs/ invoke this
  util_CSF_and_UCSF.py   CSF/UCSF machinery, generator selection, optimizers
  Sym_C2V.py             point-group tables
  ferm_utils.py          fermionic operator helpers

measurement/
  Measurement_Benchmarking_Circuit_parallel.py   entry point
  src/circuits/          CSF state preparation, controlled parallel-swap
  src/measurement_new/   sorted insertion, FC diagonalizers, KKT allocation
```

`qsense_subspace.py` writes its dumps to `QSENSE_ES_dump` unless
`QSENSE_DUMPDIR` says otherwise. Set it. The dump filename encodes
`no_states`, `irrep`, `ratio`, `S_by2`, `csf_small_thrsh` and `combo_order`
but *not* `Ethrsh_select_ia`, so two runs differing only in that threshold
write the same name and the second destroys the first — give each its own
directory.

`measurement/src` is a namespace package with no `__init__.py`, so run the
benchmark from `measurement/`.

The two need different stacks — qiskit and tequila for the benchmark, PySCF and
OpenFermion for the subspace code — and installing them together moves numpy
and scipy underneath running sweeps. `environment/` ships one file for each.

## Citing

Archived at [10.5281/zenodo.22268021](https://doi.org/10.5281/zenodo.22268021)
— the concept DOI, which always resolves to the latest version. Version `v1.0`
specifically is [10.5281/zenodo.22268022](https://doi.org/10.5281/zenodo.22268022).

See `CITATION.cff` for the accompanying paper.
