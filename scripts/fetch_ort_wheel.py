#!/usr/bin/env python3
"""Fetch the official onnxruntime-gpu wheel from PyPI — stdlib only, no pip.

Why not pip: Debian 12+/Ubuntu 23.04+ mark the system Python externally
managed (PEP 668), which can refuse pip operations outside a venv. This
fetcher needs nothing beyond python3 itself, so the installer works the same
on every distro.

    usage: fetch_ort_wheel.py <version> <dest_dir>
Prints the downloaded wheel's path on stdout (the installer captures it).
"""
import hashlib
import json
import sys
import urllib.request

MAX_WHEEL_BYTES = 1_500_000_000  # sanity cap (~1.5 GB) — bounds the download loop

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: fetch_ort_wheel.py <version> <dest_dir>")
    ver, dest = sys.argv[1], sys.argv[2]

    # PyPI's JSON API lists every file of a release with URL + sha256.
    meta_url = f"https://pypi.org/pypi/onnxruntime-gpu/{ver}/json"
    try:
        with urllib.request.urlopen(meta_url, timeout=30) as r:
            meta = json.load(r)
    except Exception as e:
        sys.exit(f"ERROR: PyPI metadata fetch failed for {ver}: {e}")

    # The capi .so files are identical across cp3XX tags (they don't link
    # libpython), so any manylinux x86_64 wheel works; sort for determinism.
    # Free-threaded ("t"-suffixed ABI, e.g. cp314t) wheels are EXCLUDED: the
    # standard-build wheel is the canonical artifact, and the exotic variant
    # also can't be pip-installed by the container-based verification.
    def is_freethreaded(name):
        return any(part.endswith("t") and part.startswith("cp")
                   for part in name.split("-"))
    cands = sorted(
        (u for u in meta["urls"]
         if u["filename"].endswith(".whl")
         and "manylinux" in u["filename"]
         and "x86_64" in u["filename"]
         and not is_freethreaded(u["filename"].removesuffix(".whl"))),
        key=lambda u: u["filename"])
    if not cands:
        sys.exit(f"ERROR: no manylinux x86_64 wheel for onnxruntime-gpu {ver}")
    pick = cands[-1]

    out = f"{dest}/{pick['filename']}"
    h = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(pick["url"], timeout=60) as r, open(out, "wb") as f:
        while True:  # bounded by MAX_WHEEL_BYTES below
            chunk = r.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_WHEEL_BYTES:
                sys.exit("ERROR: wheel exceeds sanity cap")
            h.update(chunk)
            f.write(chunk)

    # Integrity gate: PyPI publishes the sha256 per file; a mismatch means a
    # truncated or tampered download — never hand a bad wheel to the installer.
    want = pick["digests"]["sha256"]
    if h.hexdigest() != want:
        sys.exit(f"ERROR: sha256 mismatch (got {h.hexdigest()}, want {want})")
    print(out)

if __name__ == "__main__":
    main()
