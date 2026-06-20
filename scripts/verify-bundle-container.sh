#!/bin/bash
# verify-bundle-container.sh — confirms the python-free release bundle (the
# tar.gz attached to the GitHub release) is intact and loadable on a clean
# distro. Extracts it in a Debian 13 container, checks the expected files are
# present, and ctypes-loads the main library + resolves OrtGetApiBase to prove
# the packaged copy isn't truncated/corrupted. The .so files are byte-identical
# to the wheel that verify-trixie-ort.sh already runs inference with, so this
# gate covers the PACKAGING step, not the libraries themselves.
#   usage: bash scripts/verify-bundle-container.sh <bundle.tar.gz>
set -euo pipefail
BUNDLE="${1:?usage: verify-bundle-container.sh <bundle.tar.gz>}"
STAGE="$(mktemp -d -p "${TMPDIR:-/tmp}" bundlechk.XXXX)"
trap 'rm -rf "$STAGE"' EXIT
cp "$BUNDLE" "$STAGE/bundle.tar.gz"

# The load check lives in its own file (mounted into the container) to avoid
# fragile heredoc/quote nesting inside `podman run bash -c`.
cat > "$STAGE/check.py" <<'PY'
import ctypes, glob, sys
d = sys.argv[1]
so = glob.glob(d + "/libonnxruntime.so.1.*")[0]
lib = ctypes.CDLL(so)
lib.OrtGetApiBase.restype = ctypes.c_void_p
assert lib.OrtGetApiBase(), "OrtGetApiBase returned NULL"
print("LOADED_OK:", so.split("/")[-1])
PY

podman run --rm -v "$STAGE":/stage:Z docker.io/library/debian:trixie bash -c '
  set -e
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null && apt-get install -y -qq python3 >/dev/null 2>&1
  mkdir -p /b && tar xzf /stage/bundle.tar.gz -C /b
  D=$(echo /b/ort-gpu-*-linux-x86_64)
  for f in libonnxruntime.so.1.26.0 libonnxruntime_providers_shared.so \
           libonnxruntime_providers_cuda.so LICENSE PROVENANCE.txt; do
    test -e "$D/$f" || { echo "FAIL: $f missing from bundle"; exit 1; }
  done
  echo "all expected files present"
  python3 /stage/check.py "$D"
  echo "VERDICT: BUNDLE_OK"
'
