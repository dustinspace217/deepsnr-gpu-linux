#!/bin/bash
# inspect-ort-arch.sh
# Show which GPU architectures an ONNX Runtime CUDA provider library was built
# for. Use this to check whether a given onnxruntime-gpu build natively supports
# your card's compute capability (e.g. sm_120 for RTX 50-series / Blackwell).
#
#   usage: bash inspect-ort-arch.sh <libonnxruntime_providers_cuda.so>
#   (set CUOBJ=/usr/local/cuda-12.8/bin/cuobjdump if cuobjdump isn't on PATH)
#
# How to read the output:
#   * "native SASS" cubins run directly on exactly those archs (no JIT).
#   * PTX is forward-compatible: the driver can JIT a compute_NN PTX onto any
#     newer arch. A kernel that is SASS-only (no PTX) for archs <= your card's
#     CC and has no usable PTX cannot run -> cudaErrorNoKernelImageForDevice.
set -euo pipefail
LIB="${1:?usage: inspect-ort-arch.sh <libonnxruntime_providers_cuda.so>}"
CUOBJ="${CUOBJ:-cuobjdump}"

echo "== native SASS cubins (run directly, no JIT) =="
"$CUOBJ" --list-elf "$LIB" | grep -oE 'sm_[0-9]+' | sort -u

echo "== PTX (forward-JITable to newer archs) =="
"$CUOBJ" --list-ptx "$LIB" | grep -oE 'compute_[0-9]+|sm_[0-9]+' | sort -u
