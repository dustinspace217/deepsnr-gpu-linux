#!/usr/bin/env python3
"""Smoke-test a DeepSNR ONNX model on the GPU via ONNX Runtime's CUDA provider.

Loads the model, registers the CUDA execution provider (CPU as fallback) — the
same way DeepSNR's PixInsight module does — and runs one inference on a dummy
input shaped from the model's own input signature. If an ORT-native kernel has
no usable image for your GPU, run() raises cudaErrorNoKernelImageForDevice; if
it returns, the build is usable on your card.

    usage:  python3 test_infer.py /opt/PixInsight/bin/deepsnr/DeepSNR_weights_v2.onnx [cpu]

Pass "cpu" as the second argument to skip the CUDA provider entirely — used by
the container-based portability verification, where no GPU stack exists.

Run it with the CUDA runtime on the loader path, e.g.:
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 python3 test_infer.py <model.onnx>
"""
import sys
import traceback
import numpy as np
import onnxruntime as ort

if len(sys.argv) < 2:
    sys.exit("usage: test_infer.py <model.onnx>")
model = sys.argv[1]

print("ort_version:", ort.__version__)
print("available_providers:", ort.get_available_providers())

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
if len(sys.argv) > 2 and sys.argv[2] == "cpu":
    providers = ["CPUExecutionProvider"]

so = ort.SessionOptions()
so.log_severity_level = 2  # surface "node placed on CPU" fallbacks
try:
    sess = ort.InferenceSession(model, so, providers=providers)
except Exception as e:
    print("SESSION_CREATE_FAILED:", repr(e))
    traceback.print_exc()
    sys.exit(2)

print("session_providers:", sess.get_providers())
for i in sess.get_inputs():
    print("input:", i.name, i.shape, i.type)


def concretize(shape):
    """Honor the model's static dims; fill symbolic/dynamic dims (e.g. batch) with 1."""
    return [d if isinstance(d, int) and d > 0 else 1 for d in shape]


feed = {}
for i in sess.get_inputs():
    shp = concretize(i.shape)
    feed[i.name] = np.random.rand(*shp).astype(np.float32)
    print("feeding:", i.name, shp)

try:
    out = sess.run(None, feed)
    print("RUN_OK output_shapes:", [o.shape for o in out])
    if "CUDAExecutionProvider" in sess.get_providers():
        print("VERDICT: RUNS_ON_GPU")
    else:
        print("VERDICT: RAN_ON_CPU_ONLY")
except Exception as e:
    print("RUN_FAILED:", repr(e))
    traceback.print_exc()
    sys.exit(3)
