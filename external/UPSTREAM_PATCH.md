# Changes proposed upstream

Two changes to `toby1998/Q-SENSE`. **Both are already applied in the vendored
copy under `external/qsense/`**, so nothing here blocks reproduction — they are
recorded so the upstream repository can pick them up, and so the differences
between the vendored copy and upstream are documented rather than silent.

Both are small and neither alters any numerical result.

---

## 1. Honour an output-directory environment variable

**Why.** The dump filename encodes `no_states`, `irrep`, `ratio`, `S_by2`,
`csf_small_thrsh` and `combo_order` — but *not* `Ethrsh_select_ia` and *not*
`Uopt_thrsh`. Two runs differing only in those write byte-identical filenames,
and the second silently destroys the first. The script's own comment says as
much.

That is not hypothetical. It is why a threshold sweep has to give each
`Ethrsh` its own directory, and it is the mechanism behind a set of superseded
runs being mistaken for current ones during this paper's preparation.

**Change.** In `CSF_UCSF_GS.py`, where `target_dir` is defined:

```python
# before
target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'QSENSE_ES_dump')

# after
_dumpdir = os.environ.get('QSENSE_DUMPDIR', 'QSENSE_ES_dump')
target_dir = (_dumpdir if os.path.isabs(_dumpdir) else
              os.path.join(os.path.dirname(os.path.abspath(__file__)), _dumpdir))
```

Default behaviour is unchanged when `QSENSE_DUMPDIR` is unset, so no existing
script breaks.

---

## 2. Rename `CSF_UCSF_GS.py` → `qsense_subspace.py`

**Why.** `GS` reads as *ground state*, but the file's purpose — and its only
use in this paper — is targeting excited states. The new name says what it
builds without the `CSF_UCSF` jargon.

**Change.** `git mv CSF_UCSF_GS.py qsense_subspace.py`, then add a deprecation
shim at the old path so nothing downstream breaks:

```python
# CSF_UCSF_GS.py -- deprecated alias for qsense_subspace.py.
#
# The old name said "GS" (ground state) while the code targets excited states.
# Kept so existing batch scripts and notebooks keep working; remove after the
# next release.
import runpy
import warnings

warnings.warn(
    "CSF_UCSF_GS.py has been renamed to qsense_subspace.py; "
    "update your scripts, this alias will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
runpy.run_module("qsense_subspace", run_name="__main__", alter_sys=True)
```

Other files in the upstream repo that reference the old name by string
(`Ham_gen/h2o_sto3g_full.py`, `Ham_gen/h2o_sto3g_sweep.py`, `HANDOFF.md`, and
several `hpc/run_*.sh`) mention it in comments and can be updated at leisure —
the shim keeps them working either way.

---

## 3. Add a LICENSE

Neither `toby1998/Q-SENSE` nor `Praveen91299/seniority` currently has a license
file. The code vendored here is redistributed with the authors' permission, so
this repository is covered, but a reader who goes to either upstream repository
still finds code with no stated reuse rights.

MIT or BSD-3-Clause is conventional for work of this kind, and matches the
license on this repository.

---

## Keeping the copies in step

The vendored trees are the entry point plus its import closure, taken once. If
upstream changes, re-take the closure rather than patching files individually —
`external/README.md` lists what each tree contains.

Tagging upstream is still worth doing, so the vendored state has a name to
point at:

```bash
git tag -a v1.0-paper -m "State accompanying the Q-SENSE excited-states paper"
git push origin v1.0-paper
```
