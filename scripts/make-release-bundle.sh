#!/bin/bash
# make-release-bundle.sh — packages the three GPU ONNX Runtime libraries from
# the OFFICIAL PyPI wheel into a tar.gz for the GitHub Release, for users
# without python3. Provenance (wheel name + sha256) is embedded so anyone can
# reproduce the bundle from PyPI and diff it. MIT license text ships alongside.
#   usage: bash scripts/make-release-bundle.sh <wheel-path> <out-dir>
set -euo pipefail
WHEEL="${1:?usage: make-release-bundle.sh <wheel> <outdir>}"
OUT="${2:?usage: make-release-bundle.sh <wheel> <outdir>}"
VER="$(basename "$WHEEL" | sed -E 's/onnxruntime_gpu-([0-9.]+)-.*/\1/')"
D="$(mktemp -d -p "${TMPDIR:-/tmp}" bundle.XXXX)"
trap 'rm -rf "$D"' EXIT
NAME="ort-gpu-$VER-linux-x86_64"

# The MIT license ships INSIDE the package (onnxruntime/LICENSE), not in
# dist-info (verified against the 1.26.0 wheel layout).
unzip -o -q "$WHEEL" -d "$D/w" "onnxruntime/capi/*" "onnxruntime/LICENSE"
mkdir -p "$D/$NAME"
cp "$D/w/onnxruntime/capi/libonnxruntime.so."*                "$D/$NAME/"
cp "$D/w/onnxruntime/capi/libonnxruntime_providers_shared.so" "$D/$NAME/"
cp "$D/w/onnxruntime/capi/libonnxruntime_providers_cuda.so"   "$D/$NAME/"
cp "$D/w/onnxruntime/LICENSE" "$D/$NAME/LICENSE" \
	|| { echo "FAIL: onnxruntime/LICENSE missing from wheel — refusing to bundle without it"; exit 1; }
{
	echo "Provenance: extracted unmodified from the official PyPI wheel"
	echo "wheel: $(basename "$WHEEL")"
	echo "sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
	echo "source: https://pypi.org/project/onnxruntime-gpu/$VER/"
} > "$D/$NAME/PROVENANCE.txt"

mkdir -p "$OUT"
tar -C "$D" -czf "$OUT/$NAME.tar.gz" "$NAME"
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
ls -la "$OUT/$NAME.tar.gz" "$OUT/$NAME.tar.gz.sha256"
