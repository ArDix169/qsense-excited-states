# External code

The upstream implementations, vendored here with the authors' permission so
that tiers 2–3 need nothing beyond this repository.

```
qsense/       subspace construction   (upstream: toby1998/Q-SENSE)
measurement/  VO measurement benchmark (upstream: Praveen91299/seniority)
```

Each directory holds the entry point and the modules it imports, nothing else —
the import closure was taken from the entry point, so both trees run as they
stand.

## qsense/

| file | |
|---|---|
| `qsense_subspace.py` | entry point; `runs/` invokes this |
| `util_CSF_and_UCSF.py` | CSF and UCSF machinery, generator selection, optimizers |
| `Sym_C2V.py` | point-group tables |
| `ferm_utils.py` | fermionic operator helpers |

Upstream the entry point is still named `CSF_UCSF_GS.py`. It is renamed here
because `GS` reads as *ground state* while the code targets excited states;
`external/UPSTREAM_PATCH.md` proposes the same rename upstream. The copy here
also honours `QSENSE_DUMPDIR`, without which two runs differing only in
`Ethrsh_select_ia` write the same filename and the second destroys the first.

Both differences are to the file's name and its output path. Neither touches
the numerics, so this copy reproduces the published dumps.

## measurement/

`Measurement_Benchmarking_Circuit_parallel.py` plus its `src/` closure —
`circuits/` for CSF state preparation and the controlled parallel-swap
circuits, `measurement_new/` for sorted insertion, the fully-commuting
diagonalizers and the KKT shot allocation. Unmodified from upstream.

There are no `__init__.py` files, upstream or here: `src` is a namespace
package. Run the benchmark from this directory so the relative imports resolve.

## Environment

`measurement/` needs qiskit and tequila; `qsense/` needs the PySCF/OpenFermion
stack. Install them separately — see `environment/`, which ships one file for
each, deliberately not merged.
