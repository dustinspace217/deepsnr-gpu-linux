#!/bin/bash
# install-deepsnr-gpu.sh
# -----------------------------------------------------------------------------
# Enable CUDA acceleration for the current ONNX-backed DeepSNR PixInsight module
# on Linux by supplying the official onnxruntime-gpu CUDA provider library that
# StarNet's Linux package omits. Does NOT touch the signed DeepSNR module; it
# only swaps the ONNX Runtime libraries the module loads -- which is exactly the
# mechanism StarNet documents for Windows.
#
# Two phases, selected automatically by whether you are root:
#   * as your normal user -> fetch + stage the matching GPU libraries from PyPI
#   * with sudo           -> install the staged libraries into PixInsight
#
# Usage:
#   bash install-deepsnr-gpu.sh        # phase 1: fetch (no root)
#   sudo bash install-deepsnr-gpu.sh   # phase 2: install (root)
#
# See README.md for requirements (CUDA 12.x + cuDNN 9.x on PixInsight's path).
# Needs only python3 + unzip — pip is deliberately NOT used, so PEP 668
# ("externally managed environment") distros like Debian 12+/Ubuntu 23.04+
# work without venvs or --break-system-packages.
set -euo pipefail

# Resolve this script's directory so the fetch helper is found regardless of
# the caller's CWD (the helper lives in scripts/ next to this installer).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PI_LIB="${PI_LIB:-/opt/PixInsight/bin/lib}"      # where PI keeps its bundled libs
STAGE="${STAGE:-/var/tmp/deepsnr-gpu-stage}"     # shared between the two phases
MAIN="$PI_LIB/libonnxruntime.so.1"

[ -e "$MAIN" ] || { echo "ERROR: $MAIN not found -- is DeepSNR installed at $PI_LIB?" >&2; exit 1; }

# The CUDA provider must match the main library's onnxruntime version EXACTLY
# (their internal ABI is not stable across versions). We read the version string
# the vendor compiled into the bundled library and fetch precisely that build.
detect_version() {
	strings "$MAIN" | grep -m1 -oE '^[12]\.[0-9]+\.[0-9]+$' || true
}

if [ "$(id -u)" -ne 0 ]; then
	# ----- Phase 1: fetch + stage (runs as the normal user) -------------------
	VER="${ORT_VERSION:-$(detect_version)}"
	[ -n "$VER" ] || { echo "ERROR: could not detect ORT version; set ORT_VERSION=x.y.z" >&2; exit 1; }
	echo "[*] Bundled ONNX Runtime version: $VER"
	rm -rf "$STAGE"; mkdir -p "$STAGE"
	echo "[*] Downloading onnxruntime-gpu==$VER from PyPI (stdlib fetch, no pip) ..."
	WHEEL="$(python3 "$SCRIPT_DIR/scripts/fetch_ort_wheel.py" "$VER" "$STAGE")"
	echo "[*] Extracting ..."
	unzip -o -q "$WHEEL" -d "$STAGE/x"
	[ -e "$STAGE/x/onnxruntime/capi/libonnxruntime_providers_cuda.so" ] \
		|| { echo "ERROR: CUDA provider not found in wheel" >&2; exit 1; }
	echo "[*] Staged to $STAGE"
	echo "[*] Next:  sudo bash $0"
	exit 0
fi

# ----- Phase 2: install (runs as root) ----------------------------------------
CAPI="$STAGE/x/onnxruntime/capi"
[ -d "$CAPI" ] || { echo "ERROR: nothing staged at $CAPI. Run 'bash $0' as your user first." >&2; exit 1; }
VER="${ORT_VERSION:-$(detect_version)}"

# Back up the CPU-only libraries once (idempotent across re-runs and PI updates:
# after an update restores the CPU lib, the backup already exists and is kept).
for f in libonnxruntime.so.1 libonnxruntime_providers_shared.so; do
	if [ -e "$PI_LIB/$f" ] && [ ! -e "$PI_LIB/$f.cpu-backup" ]; then
		cp -a "$PI_LIB/$f" "$PI_LIB/$f.cpu-backup"
		echo "[*] Backed up $f -> $f.cpu-backup"
	fi
done

# Install the GPU trio: main lib + shared bridge (overwrite), CUDA provider (add).
# The wheel names the main lib libonnxruntime.so.<version>; PI loads the soname
# libonnxruntime.so.1, so we install it under that name.
install -m 0644 "$CAPI/libonnxruntime.so.$VER"             "$PI_LIB/libonnxruntime.so.1"
install -m 0644 "$CAPI/libonnxruntime_providers_shared.so" "$PI_LIB/libonnxruntime_providers_shared.so"
install -m 0644 "$CAPI/libonnxruntime_providers_cuda.so"   "$PI_LIB/libonnxruntime_providers_cuda.so"

echo "[*] Installed GPU ONNX Runtime $VER into $PI_LIB"
echo "[*] Launch PixInsight, run DeepSNR, and look for:"
echo "      Backend: ONNX Runtime (CUDA execution provider)"
