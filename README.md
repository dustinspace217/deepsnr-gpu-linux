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
- An **NVIDIA GPU** and a driver new enough for your card (Blackwell needs a
  recent one). AMD is out of scope at the vendor level: the signed DeepSNR
  module registers ONNX Runtime's **CUDA** execution provider specifically, so
  no drop-in library can route it to ROCm — without an NVIDIA GPU the module
  keeps running on CPU exactly as shipped.
- **CUDA 12.x runtime + cuDNN 9.x** reachable on PixInsight's library path (see
  below). *Use CUDA 12.x, not 13.x — ORT 1.26 is built against CUDA 12.*
- `python3` and `unzip` — **no pip**: the installer fetches from PyPI with the
  Python standard library, so PEP 668 "externally managed" distros (Debian 12+,
  Ubuntu 23.04+) work without venvs. `strings` (binutils) for version
  auto-detection; `cuobjdump` only for the optional architecture inspection.

### Works across distros

Any x86_64 Linux with **glibc ≥ 2.27** works — the libraries come from the
official `manylinux`-tagged ONNX Runtime wheel (Debian 10+/13 ✓, Ubuntu
20.04+ ✓, RHEL/Alma/Rocky 8+ ✓, Fedora ✓, Arch ✓; musl/Alpine ✗). Verified by
running a real CPU inference of the DeepSNR model inside a **Debian 13**
container plus a glibc-symbol audit of the exact shipped libraries
(`scripts/verify-trixie-ort.sh`); GPU end-to-end verified on Fedora 44 +
RTX 5080. On Debian/Ubuntu, install CUDA 12.x + cuDNN 9 per NVIDIA's official
instructions (apt installs typically land CUDA under `/usr/local/cuda-12.x`
and cuDNN in `/usr/lib/x86_64-linux-gnu`); point the `PixInsight.sh` edit at
whichever directories hold `libcudart.so.12` and `libcudnn.so.9`.

### No python3? Use the release bundle

If you can't run the installer, [Releases](../../releases) carries
`ort-gpu-1.26.0-linux-x86_64.tar.gz` — the three libraries extracted unmodified
from the official PyPI wheel (provenance + wheel sha256 inside, MIT license
included). Verify the bundle's `.sha256`, extract, and copy the three `.so`
files into `/opt/PixInsight/bin/lib` yourself (back up the originals first, as
the installer would).

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

### Security model

The download's integrity is checked against the **sha256 published by PyPI** for
that exact file. Note this is *not* independent provenance — the hash arrives over
the same TLS channel as the file, so it catches corruption and a mirror that
tampers with the file but not the metadata; it trusts PyPI's TLS and PyPI itself.
If you want a stronger guarantee, download the bundle from this repo's
[Releases](../../releases) and check it against the `.sha256` committed here.

The two‑phase split crosses a privilege boundary, so the installer is careful
about it: phase 1 (unprivileged) stages into a **private, mode‑700 directory it
owns**, and phase 2 (`sudo`) **refuses to run** unless that directory is owned by
the user who invoked `sudo` and is not group/ or other‑writable. That prevents
another local user from pre‑planting a malicious `.so` in a shared temp directory
for root to install. The root phase installs only the libraries staged from the
verified wheel — it never touches anything outside `/opt/PixInsight/bin/lib`.

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

## RC Astro tools (NoiseXTerminator / StarXTerminator / BlurXTerminator)

Those use **TensorFlow** today, not ONNX, so the libraries here don't accelerate
them yet. The developer has confirmed a move to **ONNX+CUDA on Linux** in a future
version — at which point this same install will cover them too. For GPU
acceleration of the TensorFlow-based RC Astro tools **right now**, see the
companion project: **[pixinsight-blackwell-tensorflow](https://github.com/dustinspace217/pixinsight-blackwell-tensorflow)**.

## License

MIT — see [`LICENSE`](LICENSE). ONNX Runtime is © Microsoft (MIT). PixInsight,
DeepSNR, and StarNet are the property of their respective owners; this project is
not affiliated with or endorsed by them.
