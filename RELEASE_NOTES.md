## rblender 0.1.0 - first release

Headless Blender on a remote sandbox (instabox, Daytona, any SSH server) with zero setup, built on [rcargo](https://github.com/igun997/rcargo).

- One command provisions Blender 4.5 LTS on the remote machine on first use (about 20 s), then `rblender run|eval|exec|demo|rig` run Blender there and pull results back.
- `rblender verify` proves artifacts are real: decoded PNGs are not blank, glb has skin and animation, mp4/fbx/usd/blend headers are valid.
- Evidence (see the README): textured and animated scene, rigged walk cycle, and a full fan-made character (Vestia Zeta): 77 meshes, 64k triangles, 123 bones, 5 s animation, glb re-imported in a fresh Blender.
- CI installs real Blender on a runner and re-creates the evidence on every push.

Install:

```sh
curl -fsSL https://raw.githubusercontent.com/igun997/rblender/main/install.sh | sh
```

Assets: `rblender-<target>.tar.gz` for Linux x86_64/aarch64 and macOS arm64/x86_64, plus `SHA256SUMS` and `install.sh`.
