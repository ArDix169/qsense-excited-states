# Q-SENSE for excited states — data and analysis

Code and data accompanying *[paper title]*.

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
| 2 | subspace dumps regenerated | HPC-days | + `external/qsense` |
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

Requires SLURM and the upstream Q-SENSE implementation (see *External code*).
`runs/` holds the batch scripts; each header documents its grid, thresholds,
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
figures/      one script per figure; numbers inline, provenance in the docstring
analysis/     collectors that rebuild each table from raw output + verify_all.sh
runs/         SLURM batch scripts (thresholds and grids documented in-header)
hamiltonians/ CASSCF Hamiltonian and full-space FCI reference generators
baselines/    q-sc-EOM and SS-SSSA-VQE (determinant-basis benchmark)
tables/       generated LaTeX
data/         the archived dumps, benchmarks and FCI references
environment/  two conda environments, deliberately separate
```

Two environments because the measurement benchmark needs qiskit and tequila,
and installing those beside the Q-SENSE stack moves numpy and scipy underneath
running subspace sweeps.

## External code

Tiers 2–3 additionally require the upstream implementations, added as
submodules under `external/`:

- **Q-SENSE** — the subspace construction (`qsense_subspace.py`)
- **seniority** — the VO measurement benchmark

The scripts in `runs/` invoke `qsense_subspace.py`. Upstream this file is still
named `CSF_UCSF_GS.py`; the rename is proposed in `external/UPSTREAM_PATCH.md`
along with a compatibility shim, and the old name said "GS" (ground state)
while the code targets excited states.

Tiers 0 and 1 do not use them: the archived data and the inlined figure numbers
are sufficient to verify every claim in the paper.

## Citing

Archived at [10.5281/zenodo.22268021](https://doi.org/10.5281/zenodo.22268021)
— the concept DOI, which always resolves to the latest version. Version `v1.0`
specifically is [10.5281/zenodo.22268022](https://doi.org/10.5281/zenodo.22268022).

See `CITATION.cff` for the accompanying paper.
