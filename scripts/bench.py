#!/usr/bin/env python3
"""Benchmark a DeepSNR ONNX model: GPU (CUDA EP) vs CPU, per inference.

Reports the first GPU run separately (it includes any one-time PTX->SASS JIT),
the warm GPU average, the CPU average, and the warm speedup.

    usage:  python3 bench.py /opt/PixInsight/bin/deepsnr/DeepSNR_weights_v2.onnx

Run with the CUDA runtime on the loader path:
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 python3 bench.py <model.onnx>
"""
import sys
import time
import numpy as np
import onnxruntime as ort

if len(sys.argv) < 2:
    sys.exit("usage: bench.py <model.onnx>")
model = sys.argv[1]


def concretize(shape):
    return [d if isinstance(d, int) and d > 0 else 1 for d in shape]


def make_feed(sess):
    feed = {}
    for i in sess.get_inputs():
        feed[i.name] = np.random.rand(*concretize(i.shape)).astype(np.float32)
    return feed


def bench(providers, n):
    so = ort.SessionOptions()
    so.log_severity_level = 3  # quiet
    s = ort.InferenceSession(model, so, providers=providers)
    feed = make_feed(s)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        s.run(None, feed)
        times.append(time.perf_counter() - t)
    return s.get_providers(), times


cp, ct = bench(["CUDAExecutionProvider", "CPUExecutionProvider"], 7)
print("cuda_providers:", cp)
print("cuda_first_run_s: %.3f" % ct[0])
warm = ct[1:]
cuda_warm = sum(warm) / len(warm)
print("cuda_warm_avg_s: %.3f" % cuda_warm)

_, pt = bench(["CPUExecutionProvider"], 3)
cpu_avg = sum(pt) / len(pt)
print("cpu_avg_s: %.3f" % cpu_avg)

print("WARM_SPEEDUP: %.1fx" % (cpu_avg / cuda_warm))
