---
name: rblender
description: Use for ANY Blender task (modeling, texturing, rigging, animation, rendering, export to glb/fbx/obj/usd/stl) or when asked to set up remote Blender. Blender runs headless on a remote machine through the `rblender` CLI (built on rcargo); never install or run Blender on the user's computer. Also use when an rblender or rcargo command prints "SETUP REQUIRED".
---

# rblender: headless Blender on a remote machine

Blender is installed on first use on the user's rcargo instance (instabox, Daytona or an SSH server); nothing is installed locally. Run everything from the project root (the folder with `Cargo.toml` and `rcargo.toml`; `rblender init` creates them).

## Commands

| Command | Use |
|---|---|
| `rblender run scenes/x.py [ARGS]` | run a Blender Python script (`bpy` available) from this project |
| `rblender eval '<python>'` | inline Python inside Blender |
| `rblender exec -- -b file.blend -a` | raw Blender command line |
| `rblender demo` / `rblender rig` | built-in examples: texturing, animation, render, export; armature, skin weights, walk cycle |
| `rblender pull [DIR]` | copy remote `out/` to `DIR` (default `./out`) |
| `rblender verify [DIR] [--skin] [--anim]` | prove artifacts are real (PNG not blank, glb has skin/animation, mp4/fbx/usd headers) |
| `rblender doctor`, `rblender setup`, `rblender down` | check, pre-install, pause the machine |

Flags: `--no-pull`, `--keep` (do not wipe `out/`), `--xvfb`, `--verbose`. Choose a machine with `-i NAME` (see `rcargo instances`).

## Writing scripts

```python
import bpy, rb            # rb ships with rblender
rb.clean()                # empty scene
...                       # model, UV unwrap, materials, armature, keyframes
rb.render_still("a.png", engine="CYCLES")   # CYCLES | EEVEE | WORKBENCH, CPU only
rb.render_anim("a.mp4", frames=(1, 48))     # .mp4 (H.264) or PNG sequence
rb.export("model.glb", export_skins=True)   # glb gltf fbx obj usd usdc usda stl ply abc blend
rb.report()
```

Everything written through `rb.out("name.ext")` lands in remote `out/` and is pulled to `./out` (wiped each run unless `--keep`). Sibling modules in the project can be imported. Scripts that export animation to glTF must bind the action slot: `rig.animation_data.action_slot = action.slots[0]` when authoring actions through `action.fcurves`.

## Workflow

1. Write or edit the script, `rblender run ...`, fix Python errors from the traceback.
2. Look at renders (read the PNGs) and run `rblender verify out --skin --anim` before claiming success.
3. Long renders: use `--quick`-style options in your script, then the full run.

## When something is missing

If a command prints **SETUP REQUIRED** (exit 3) follow the rcargo skill: run steps not marked `[USER]` yourself, ask the user for `[USER]` steps (logins, keys), never invent or ask for credentials in chat, then verify with `rblender doctor`.

Never delete the remote machine or workspaces on your own initiative (`rcargo remote destroy|rm|gc --all`); offer and wait for an explicit yes. `rblender down` only pauses it.

Rust builds of rblender itself go through rcargo, never local `cargo build/test`.
