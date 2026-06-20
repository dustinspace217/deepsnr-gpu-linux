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
#   sudo bash install-deepsnr-gpu.sh   # phase 2: install (run via sudo)
#
# Security model: phase 1 runs unprivileged and writes a PRIVATE (mode 700)
# staging dir owned by you; the download is integrity-checked (sha256 from
# PyPI). Phase 2 runs as root and REFUSES to proceed unless the staging dir is
# owned by the user who invoked sudo and is not group/other-writable -- so a
# different local user cannot pre-plant a malicious .so for root to install.
# Trust root for the download itself is PyPI's TLS + per-file sha256 (see
# README "Security"); this is not independent provenance.
#
# Needs only python3 + unzip + strings -- pip is deliberately NOT used, so PEP
# 668 ("externally managed environment") distros like Debian 12+/Ubuntu 23.04+
# work without venvs. See README.md for runtime requirements (CUDA 12.x +
# cuDNN 9.x on PixInsight's path).
set -euo pipefail

# Resolve this script's directory so the fetch helper is found regardless of
# the caller's CWD (the helper lives in scripts/ next to this installer).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PI_LIB="${PI_LIB:-/opt/PixInsight/bin/lib}"      # where PI keeps its bundled libs
STAGE="${STAGE:-/var/tmp/deepsnr-gpu-stage}"     # handoff dir between the two phases
MAIN="$PI_LIB/libonnxruntime.so.1"

[ -e "$MAIN" ] || { echo "ERROR: $MAIN not found -- is DeepSNR installed at $PI_LIB?" >&2; exit 1; }

# The CUDA provider must match the main library's onnxruntime version EXACTLY
# (their internal ABI is not stable across versions). We read the version string
# the vendor compiled into the bundled library. NOT anchored to a whole line:
# ORT may embed the version inside a longer string (e.g. "onnxruntime 1.26.0"),
# so we match the version substring anywhere strings reports it.
detect_version() {
	strings "$MAIN" | grep -m1 -oE '[12]\.[0-9]+\.[0-9]+' || true
}

if [ "$(id -u)" -ne 0 ]; then
	# ----- Phase 1: fetch + stage (runs as the normal user) -------------------
	# Preflight: fail early and clearly if a required tool is missing, rather
	# than mid-run with bash's raw "command not found".
	for t in strings unzip python3; do
		command -v "$t" >/dev/null 2>&1 || { echo "ERROR: required tool '$t' not found on PATH." >&2; exit 1; }
	done

	VER="${ORT_VERSION:-$(detect_version)}"
	[ -n "$VER" ] || { echo "ERROR: could not detect ORT version; set ORT_VERSION=x.y.z" >&2; exit 1; }
	echo "[*] Bundled ONNX Runtime version: $VER"

	# Refuse to reuse a staging dir we don't own (someone else may have squatted
	# the path). Then create it fresh and PRIVATE so no other user can write to
	# the files phase 2 will install as root.
	me="$(id -un)"
	if [ -e "$STAGE" ]; then
		owner="$(stat -c %U "$STAGE" 2>/dev/null || echo '?')"
		[ "$owner" = "$me" ] || { echo "ERROR: $STAGE exists but is owned by '$owner', not you ('$me'). Refusing (possible tampering). Remove it or set STAGE=<a private path>." >&2; exit 1; }
	fi
	rm -rf "$STAGE"
	mkdir -p "$STAGE"
	chmod 700 "$STAGE"

	echo "[*] Downloading onnxruntime-gpu==$VER from PyPI (stdlib fetch, no pip) ..."
	WHEEL="$(python3 "$SCRIPT_DIR/scripts/fetch_ort_wheel.py" "$VER" "$STAGE")"
	echo "[*] Extracting ..."
	unzip -o -q "$WHEEL" -d "$STAGE/x"
	[ -e "$STAGE/x/onnxruntime/capi/libonnxruntime_providers_cuda.so" ] \
		|| { echo "ERROR: CUDA provider not found in wheel" >&2; exit 1; }

	# Persist the version so phase 2 uses EXACTLY what was staged -- never
	# re-detects (a PixInsight update between phases could otherwise restore the
	# CPU lib and make phase 2 pick a different version than the staged files).
	printf '%s\n' "$VER" > "$STAGE/version"

	echo "[*] Staged to $STAGE"
	echo "[*] Next:  sudo bash $0"
	exit 0
fi

# ----- Phase 2: install (runs as root) ----------------------------------------
# Trust gate: only proceed if the staging dir belongs to the user who invoked
# sudo (or to root, for the run-everything-as-root case) and is not writable by
# group/other. This is what prevents a different local user from planting a
# payload in a shared dir for root to install.
[ -d "$STAGE" ] || { echo "ERROR: nothing staged at $STAGE -- run 'bash $0' as your user first." >&2; exit 1; }
owner="$(stat -c %U "$STAGE" 2>/dev/null || echo '?')"
expect="${SUDO_USER:-root}"
if [ "$owner" != "$expect" ] && [ "$owner" != "root" ]; then
	echo "ERROR: $STAGE is owned by '$owner', not the invoking user '$expect'. Refusing (run phase 1 as yourself, then phase 2 via sudo)." >&2
	exit 1
fi
# -perm /022 matches if group-write OR other-write is set; empty result = safe.
[ -z "$(find "$STAGE" -maxdepth 0 -perm /022)" ] || { echo "ERROR: $STAGE is group/other-writable -- refusing (possible tampering)." >&2; exit 1; }

VER="$(cat "$STAGE/version" 2>/dev/null || true)"
[ -n "$VER" ] || { echo "ERROR: staged version file missing -- re-run phase 1." >&2; exit 1; }

CAPI="$STAGE/x/onnxruntime/capi"
[ -e "$CAPI/libonnxruntime.so.$VER" ] || { echo "ERROR: staged libraries don't match version $VER -- re-run phase 1." >&2; exit 1; }

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
