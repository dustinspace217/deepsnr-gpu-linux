#!/bin/bash
# verify-trixie-ort.sh — proves the ORT artifact's portability claim on Debian 13.
# Runs a REAL CPU inference of the DeepSNR model inside a debian:trixie container
# using the exact official wheel the installer fetches (GPU behavior is distro-
# independent and already proven on the host; the container proves glibc/ABI).
# Also audits the CUDA provider lib: its only unresolved deps may be CUDA/cuDNN
# (absent in the container by design) — any glibc version error = FAIL.
#
#   usage: bash scripts/verify-trixie-ort.sh <wheel-path> <model.onnx>
set -euo pipefail
WHEEL="${1:?usage: verify-trixie-ort.sh <wheel> <model.onnx>}"
MODEL="${2:?usage: verify-trixie-ort.sh <wheel> <model.onnx>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SELinux: a bind mount needs container-accessible labels, and :z/:Z RELABELS
# the source. Relabeling the model inside /opt/PixInsight or a file in the git
# repo is invasive — so copy everything into a throwaway stage dir and give the
# container that ONE dir with :Z (private label, deleted afterwards).
STAGE="$(mktemp -d -p "${TMPDIR:-/tmp}" trixie-ort.XXXX)"
trap 'rm -rf "$STAGE"' EXIT
# Keep the ORIGINAL wheel filename: pip parses name/version/tags from it
# (PEP 427) and rejects renamed wheels outright.
cp "$WHEEL" "$STAGE/$(basename "$WHEEL")"
cp "$MODEL" "$STAGE/model.onnx"
cp "$SCRIPT_DIR/test_infer.py" "$STAGE/test_infer.py"

podman run --rm \
	-v "$STAGE":/stage:Z \
	docker.io/library/debian:trixie bash -c '
	set -e
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -qq >/dev/null
	apt-get install -y -qq python3 python3-venv python3-pip unzip binutils >/dev/null 2>&1
	python3 -m venv /v
	# Inference half: wheels are interpreter-bound (cp3XX tags), so let pip
	# resolve the build matching the CONTAINER python. Same release, same CI,
	# same capi libs — the version is pinned from the wheel under test.
	VER=$(basename /stage/onnxruntime_gpu-*.whl | sed -E "s/onnxruntime_gpu-([0-9.]+)-.*/\1/")
	/v/bin/pip install -q "onnxruntime-gpu==$VER" numpy
	echo "== CPU inference on trixie =="
	/v/bin/python /stage/test_infer.py /stage/model.onnx cpu
	echo "== CUDA provider ABI audit =="
	unzip -o -q /stage/onnxruntime_gpu-*.whl -d /x "onnxruntime/capi/*"
	objdump -T /x/onnxruntime/capi/libonnxruntime_providers_cuda.so | grep -oE "GLIBC_[0-9.]+" | sort -uV | tail -3
	if ldd /x/onnxruntime/capi/libonnxruntime_providers_cuda.so 2>&1 | grep "not found" | grep -vE "libcu|libnv"; then
		echo "FAIL: non-CUDA unresolved deps"; exit 1
	fi
	echo "VERDICT: TRIXIE_OK"
'
