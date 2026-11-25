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

## Precise modeling (spec, measure, correct)

For anything where proportions matter, do not eyeball renders. `import prec` (ships with rblender, see `docs/PRECISION.md`):

1. **Spec first**: write stations/landmarks and target dimensions as numbers (metres) in the script.
2. **Build from tables**, not guesses: loft sections with clean quads and grid UVs; place parts by computation (`prec.arc_positions`, `prec.ray` onto the measured surface, solve arc length by bisection).
3. **Sculpt as declared moves**: `prec.solve_forces` + `prec.field` (Pixar's Regularized Kelvinlets) lands surface points on exact displacements while pins hold the rest still. `bpy.ops.sculpt.brush_stroke` does NOT work headless.
4. **Close the loop**: measure what you built (`prec.ray`, `prec.topology`, `prec.symmetry`), correct the parameter (secant update), repeat until inside tolerance. Keep the best iterate. Never let two loops own the same number.
5. **Against a reference**: absolute pixel to metre mapping, no registration. 2D: `prec.silhouette(objs, center, ppm, res, path, view="side|front|top")` + `prec.iou` + `prec.overlay_png`. 3D mesh: `prec.target_mesh`, `prec.chamfer` (accuracy = model to target, completeness = target to model, always on sampled SURFACE points, not vertices), `prec.fit_coarse_to_fine` (robust median pulls, roughness budget).
6. **Model one half and add a Mirror modifier** (`prec.loft_half`, `prec.add_mirror`): exact symmetry. A side view says nothing about width; say so.
7. **Metrics lie, look at the render**: a crumpled surface can score well on IoU and Chamfer. Also report `prec.roughness` (mean dihedral angle) and compare with the target's. Reject floating parts. Keep third-party references out of the repo; commit derived numbers only.
8. **Report**: `prec.Report` prints a pass/fail table; fail loudly, report the real numbers, never loosen a limit after seeing the result.
9. **Bake**: high-poly (voxel remesh + displacement) onto the low-poly UVs with `prec.bake_pass`, then `prec.pbr_material`. Check the glb with `rblender verify out --pbr`.

Worked examples: `characters/dragon_head.py` (`rblender run characters/dragon_head.py -- --quick`) and `characters/dragon_ref3d.py` (3D target, see `docs/PRECISION.md`).
Gotchas: set `image.colorspace_settings.name` BEFORE `pixels.foreach_set`; brush radius must exceed local vertex spacing; a triangulated quad mesh is not mirror-symmetric as a surface; a tube whose rings follow a sloping path bloats the silhouette (`prec.tube(axis=...)`); an interpolating Kelvinlet solve follows noisy targets (`prec.solve_stable`, median handles).
