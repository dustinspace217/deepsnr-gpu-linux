# DeepSNR GPU Acceleration on Linux (NVIDIA, incl. Blackwell / RTX 50‑series)

Re-enable CUDA acceleration for the current **ONNX‑backed DeepSNR** PixInsight
module on Linux — including **Blackwell `sm_120`** cards (RTX 5080 / 5090) —
with **no fork, no patch, and no source build**.

## Background

DeepSNR (and StarNet2) recently retired their TensorFlow backend and moved to
**ONNX Runtime (ORT)**. On Linux, StarNet ships a **CPU‑only** ORT with each
module and documents NVIDIA CUDA acceleration **only for Windows**. The result:
on Linux, DeepSNR runs on the CPU and reports something like
*"ONNX Runtime CUDA execution provider unavailable; using CPU."*

But the Linux module (`/opt/PixInsight/bin/DeepSNR-pxm.so`) **already contains
the full CUDA path** — it calls `AppendExecutionProvider_CUDA_V2`, enumerates
CUDA devices, and falls back to CPU *only because the CUDA provider library is
missing from the package*:

```
$ ls /opt/PixInsight/bin/lib/libonnxruntime*
libonnxruntime.so.1
libonnxruntime_providers_shared.so      # the provider-loader bridge is present…
#                                       # …but libonnxruntime_providers_cuda.so is NOT
```

So the fix isn't a fork — it's supplying the missing GPU libraries. This is
**exactly the library swap StarNet documents for Windows**, applied to Linux.

## Why it works on Blackwell without a source build

`sm_120` (Blackwell) is a well‑known pain point: official `onnxruntime-gpu`
builds ship native kernels only up to `sm_90` (verify with
`scripts/inspect-ort-arch.sh`). Yet the official **1.26.0** build runs DeepSNR's
model on an RTX 5080 anyway, because:

1. The heavy convolutions dispatch to **cuDNN 9.x**, which *does* carry Blackwell
   kernels.
2. The remaining ORT‑native element‑wise kernels are JIT‑compiled from
   `compute_90` **PTX** to `sm_120` by the driver — a one‑time ~0.6 s cost, not
   the multi‑minute freeze seen with the old TensorFlow path.

Net: the stock libraries are enough. (Contrast with the TensorFlow‑based RC Astro
tools, which *do* require a source build for `sm_120`.)

## Measured results

RTX 5080 + Ryzen 9 9950X3D, DeepSNR `v2` model:

| Path | per 512×512×3 tile |
|------|--------------------|
| CPU (ORT CPU EP)        | 3.867 s |
| GPU (ORT CUDA EP, warm) | 0.028 s |
| **Speedup**             | **~136×** (core inference) |

Full image in PixInsight (6252×4176, mono, 126 tiles): **11.6 s** end‑to‑end,
with the process console reporting `Backend: ONNX Runtime (CUDA execution provider)`.

## Requirements

- PixInsight on Linux x64 with a current **ONNX‑backed** DeepSNR (≥ 1.2.1).
- An NVIDIA GPU and a driver new enough for your card (Blackwell needs a recent one).
- **CUDA 12.x runtime + cuDNN 9.x** reachable on PixInsight's library path (see
  below). *Use CUDA 12.x, not 13.x — ORT 1.26 is built against CUDA 12.*
- `python3`, `pip`, `unzip`, and `strings` (binutils). `cuobjdump` (from the CUDA
  toolkit) is needed only for the optional architecture inspection.

## Install

```
# 1. Fetch the matching official onnxruntime-gpu libraries (as your normal user):
bash install-deepsnr-gpu.sh

# 2. Install them into PixInsight (needs root):
sudo bash install-deepsnr-gpu.sh
```

The script auto‑detects the ORT version baked into your bundled
`libonnxruntime.so.1` and fetches exactly that, so the main library and the CUDA
provider stay ABI‑matched. It backs up the CPU‑only libraries to `*.cpu-backup`
before replacing them, and never touches the signed `DeepSNR-pxm.so`.

### Make sure PixInsight can see CUDA

PixInsight's launcher resets `LD_LIBRARY_PATH`, so your CUDA libraries must be
added to it. Edit `/opt/PixInsight/bin/PixInsight.sh` and prepend your CUDA path
to the `LD_LIBRARY_PATH=` line, e.g.:

```
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$dirname/lib:$dirname
```

That directory must contain `libcudart.so.12`, `libcudnn.so.9` (cuDNN 9.x for
CUDA 12), `libcublas.so.12`, etc.

## Verify

Run DeepSNR on any image; the process console should show:

```
Backend: ONNX Runtime (CUDA execution provider)
```

and `nvidia-smi` should show PixInsight using the GPU. If you instead see
*"CUDA execution provider unavailable; using CPU,"* the CUDA runtime isn't on
PixInsight's path — re‑check the launcher edit.

## Surviving PixInsight updates

A PixInsight update restores the bundled CPU‑only ORT libraries (and rewrites the
launcher), reverting this. Just re‑run the install after each update. If you also
run the TensorFlow‑based RC Astro tools on GPU, you can fold both into one helper
you run after every update.

## Reverting

```
cd /opt/PixInsight/bin/lib
sudo cp -a libonnxruntime.so.1.cpu-backup libonnxruntime.so.1
sudo cp -a libonnxruntime_providers_shared.so.cpu-backup libonnxruntime_providers_shared.so
sudo rm -f libonnxruntime_providers_cuda.so
```

## Reproduce the validation

- `scripts/inspect-ort-arch.sh <libonnxruntime_providers_cuda.so>` — list the
  SASS/PTX architectures a provider build contains.
- `scripts/test_infer.py <model.onnx>` — load DeepSNR's model and run one
  inference on the GPU (smoke test).
- `scripts/bench.py <model.onnx>` — GPU‑vs‑CPU timing on the model.

## Caveats / disclaimer

- **Unsupported by StarNet on Linux.** StarNet documents CUDA only for Windows.
  This uses their own mechanism on Linux; it works, but it is not vendor‑supported.
  Use at your own risk.
- It modifies files inside a commercial application's install directory. The
  installer makes `*.cpu-backup` copies; the restore path is above.
- **No StarNet code, libraries, or model weights are redistributed here** — only
  instructions and glue. The ONNX Runtime libraries come straight from the
  official `onnxruntime-gpu` PyPI package (MIT‑licensed, by Microsoft).
- If a future DeepSNR bumps its bundled ORT past 1.26, just re‑run the installer —
  it matches whatever version your module ships.

## License

MIT — see [`LICENSE`](LICENSE). ONNX Runtime is © Microsoft (MIT). PixInsight,
DeepSNR, and StarNet are the property of their respective owners; this project is
not affiliated with or endorsed by them.
