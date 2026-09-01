# External code

Tiers 2–3 need the upstream implementations. They are **not** vendored here:
neither repository currently carries a license, so redistribution is not
permitted (see `UPSTREAM_PATCH.md`, item 3).

Once both are licensed and tagged, add them as submodules:

    git submodule add -b v1.0-paper https://github.com/toby1998/Q-SENSE.git    qsense
    git submodule add -b v1.0-paper https://github.com/Praveen91299/seniority.git measurement

Tiers 0 and 1 do not use anything in this directory. Every number in the paper
can be verified from the Zenodo archive and the figures' inlined values alone,
so a reader who skips this step can still check the results.
