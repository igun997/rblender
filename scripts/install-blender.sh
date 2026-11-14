#!/usr/bin/env bash
# Install official headless-capable Blender into $BLENDER_HOME (default ~/blender).
# Idempotent. Works on Debian/Ubuntu x86_64 sandboxes (instabox, Daytona, any VPS).
#
#   BLENDER_VERSION=4.5.16 bash scripts/install-blender.sh
#
# Needs: curl, xz, and (once) apt + root/sudo for the shared libraries Blender links against.
set -euo pipefail

VERSION="${BLENDER_VERSION:-4.5.16}"
SERIES="${VERSION%.*}"
HOME_DIR="${BLENDER_HOME:-$HOME/blender}"
BASE="https://download.blender.org/release/Blender${SERIES}"
TARBALL="blender-${VERSION}-linux-x64.tar.xz"

if [ "$(uname -m)" != "x86_64" ]; then
  echo "install-blender: only linux x86_64 is supported (got $(uname -m))" >&2
  exit 1
fi

if [ -x "$HOME_DIR/blender" ] && "$HOME_DIR/blender" -b --version 2>/dev/null | grep -q "Blender ${VERSION}"; then
  echo "install-blender: Blender ${VERSION} already installed at $HOME_DIR"
  exit 0
fi

# 1. system libraries (X11/GL stubs Blender links against even in background mode, EGL/Mesa for
#    EEVEE, xvfb as a fallback display, ffmpeg-free: Blender bundles its own encoders)
SUDO=""
if [ "$(id -u)" != "0" ]; then
  if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then SUDO="sudo -n"; else
    echo "install-blender: need root or passwordless sudo to install libraries" >&2; exit 1; fi
fi
APT="$SUDO env DEBIAN_FRONTEND=noninteractive apt-get"
$APT update -qq
$APT install -y -qq --no-install-recommends \
  ca-certificates curl xz-utils \
  libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libxext6 \
  libgl1 libegl1 libglx-mesa0 libgl1-mesa-dri libgbm1 libsm6 libdecor-0-0 \
  xvfb xauth >/dev/null

# 2. download + verify + extract
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "install-blender: downloading ${TARBALL}"
curl -fsSL --retry 3 -o "$TMP/$TARBALL" "$BASE/$TARBALL"
curl -fsSL --retry 3 -o "$TMP/sums" "$BASE/blender-${VERSION}.sha256"
( cd "$TMP" && grep " ${TARBALL}\$" sums | sha256sum -c - )

rm -rf "$HOME_DIR"; mkdir -p "$HOME_DIR"
tar -xJf "$TMP/$TARBALL" -C "$HOME_DIR" --strip-components=1

"$HOME_DIR/blender" -b --version | head -1
echo "install-blender: ok -> $HOME_DIR/blender"
