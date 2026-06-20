#!/bin/bash
# verify-install-container.sh — exercises the FULL two-phase install flow inside
# a Debian 13 container (unprivileged phase 1 + `sudo` phase 2) against a FAKE
# PixInsight lib dir, so the root-install path is validated end-to-end without
# touching the host's /opt or needing host sudo. Covers: version detection +
# staging (phase 1), the actual library install + one-time backup (phase 2),
# idempotency on re-run, and the privilege gate that must REFUSE a staging dir
# owned by another user.
#   usage: bash scripts/verify-install-container.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# SELinux-safe: copy only what the container needs into a throwaway dir mounted
# :Z (never bind-mount + relabel the repo itself).
WORK="$(mktemp -d -p "${TMPDIR:-/tmp}" deepsnr-itest.XXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src/scripts"
cp "$REPO/install-deepsnr-gpu.sh" "$WORK/src/"
cp "$REPO/scripts/fetch_ort_wheel.py" "$WORK/src/scripts/"

cat > "$WORK/inner.sh" <<'INNER'
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq python3 unzip binutils sudo >/dev/null 2>&1

# Fake PixInsight lib dir. The stub MAIN embeds "1.26.0" so phase-1 version
# detection (strings|grep) finds it, exactly like a real bundled library.
mkdir -p /fakepi
printf 'this is a stub libonnxruntime 1.26.0 placeholder\n' > /fakepi/libonnxruntime.so.1
printf 'stub shared provider\n' > /fakepi/libonnxruntime_providers_shared.so

useradd -m tester
echo 'tester ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/tester
useradd -m mallory
cp -r /work/src /home/tester/deepsnr
chown -R tester:tester /home/tester/deepsnr

run_tester() { su - tester -c "cd ~/deepsnr && $*"; }

echo "== PHASE 1 (unprivileged) =="
run_tester "PI_LIB=/fakepi STAGE=/home/tester/stage bash install-deepsnr-gpu.sh"
test -f /home/tester/stage/version
echo "staged version: $(cat /home/tester/stage/version)"

echo "== PHASE 2 (sudo) =="
run_tester "sudo env PI_LIB=/fakepi STAGE=/home/tester/stage bash install-deepsnr-gpu.sh"

echo "== assertions =="
for f in libonnxruntime.so.1 libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
  test -e "/fakepi/$f" || { echo "FAIL: /fakepi/$f missing"; exit 1; }
done
test -e /fakepi/libonnxruntime.so.1.cpu-backup || { echo "FAIL: cpu-backup missing"; exit 1; }
sz=$(stat -c %s /fakepi/libonnxruntime.so.1)
[ "$sz" -gt 1000000 ] || { echo "FAIL: main lib not replaced by GPU build (size $sz)"; exit 1; }
echo "INSTALL_OK (main lib now $sz bytes; CPU stub backed up)"

echo "== idempotency (re-run phase 2) =="
run_tester "sudo env PI_LIB=/fakepi STAGE=/home/tester/stage bash install-deepsnr-gpu.sh"
bsz=$(stat -c %s /fakepi/libonnxruntime.so.1.cpu-backup)
[ "$bsz" -lt 1000 ] || { echo "FAIL: cpu-backup got clobbered (size $bsz) -- backup not idempotent"; exit 1; }
echo "IDEMPOTENT_OK (original CPU backup preserved at $bsz bytes)"

echo "== security gate: stage owned by another user must be refused =="
chown -R mallory:mallory /home/tester/stage
if run_tester "sudo env PI_LIB=/fakepi STAGE=/home/tester/stage bash install-deepsnr-gpu.sh" 2>/tmp/neg.txt; then
  echo "FAIL: phase 2 accepted a stage owned by another user"; exit 1
fi
test -s /tmp/neg.txt && grep -qi "refusing" /tmp/neg.txt && echo "REFUSAL_OK (foreign-owned stage rejected)"
echo "VERDICT: INSTALL_FLOW_OK"
INNER

podman run --rm -v "$WORK":/work:Z docker.io/library/debian:trixie bash /work/inner.sh
