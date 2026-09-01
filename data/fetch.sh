#!/bin/bash
# Fetch and unpack the Zenodo data archive into data/QSENSE_paper_release/.
#
# The checksum is verified before unpacking, not after: a truncated download
# that still unpacks would leave a partial tree that the collectors read as
# "sectors missing" rather than as a broken download.

set -euo pipefail

# TODO: fill in after the Zenodo deposition is published
ZENODO_URL="${ZENODO_URL:-https://zenodo.org/records/PLACEHOLDER/files/QSENSE_paper_release.tar.gz}"
EXPECTED_SHA256="${EXPECTED_SHA256:-PLACEHOLDER}"

cd "$(dirname "$0")"
TARBALL=QSENSE_paper_release.tar.gz

if [ -d QSENSE_paper_release ]; then
    echo "data/QSENSE_paper_release/ already present -- nothing to do."
    echo "Delete it first if you want a clean re-fetch."
    exit 0
fi

if [ "$EXPECTED_SHA256" = PLACEHOLDER ]; then
    echo "ERROR: this script has not been pointed at a published archive yet." >&2
    echo "Set ZENODO_URL and EXPECTED_SHA256, or edit the defaults above." >&2
    exit 1
fi

[ -f "$TARBALL" ] || { echo "downloading $ZENODO_URL"; curl -fL -o "$TARBALL" "$ZENODO_URL"; }

echo "verifying checksum"
if command -v sha256sum >/dev/null; then
    echo "${EXPECTED_SHA256}  ${TARBALL}" | sha256sum -c -
else
    got=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)
    [ "$got" = "$EXPECTED_SHA256" ] || { echo "CHECKSUM MISMATCH: $got" >&2; exit 1; }
    echo "${TARBALL}: OK"
fi

tar xzf "$TARBALL"
echo
echo "unpacked:"
find QSENSE_paper_release -maxdepth 2 -type d | sort | sed 's/^/  /'
echo
echo "next:  bash analysis/verify_all.sh"
