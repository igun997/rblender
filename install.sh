#!/bin/sh
# Installs a prebuilt rblender binary (no Rust toolchain needed).
#   curl -fsSL https://raw.githubusercontent.com/igun997/rblender/main/install.sh | sh
# rblender drives Blender through rcargo; install rcargo too if it is missing.
set -eu
repo="igun997/rblender"
os=$(uname -s); arch=$(uname -m)
case "$os-$arch" in
  Linux-x86_64)  target=x86_64-unknown-linux-gnu ;;
  Linux-aarch64|Linux-arm64) target=aarch64-unknown-linux-gnu ;;
  Darwin-arm64)  target=aarch64-apple-darwin ;;
  Darwin-x86_64) target=x86_64-apple-darwin ;;
  *) echo "rblender: no prebuilt binary for $os $arch. Build from source: cargo install --git https://github.com/$repo" >&2; exit 1 ;;
esac
dest="${RBLENDER_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$dest"
url="https://github.com/$repo/releases/latest/download/rblender-$target.tar.gz"
echo "Downloading $url"
curl -fsSL "$url" | tar -xz -C "$dest" rblender
chmod +x "$dest/rblender"
echo "Installed $dest/rblender"
case ":$PATH:" in *":$dest:"*) ;; *) echo "Add $dest to your PATH:  export PATH=\"$dest:\$PATH\"" ;; esac
if ! command -v rcargo >/dev/null 2>&1; then
  echo "rcargo is required (it provides the remote machine). Install it:"
  echo "  curl -fsSL https://raw.githubusercontent.com/igun997/rcargo/main/install.sh | sh"
fi
echo "Next: go to a folder for your 3D project and run:  rblender init"
