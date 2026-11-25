# Precise modeling with an LLM in headless Blender

Research notes behind `py/prec.py`, `characters/dragon_head.py` (2D reference) and `characters/dragon_ref3d.py`
(3D target). Everything marked "measured" was run on a 4 vCPU instabox box with Blender 4.5.16, no GPU, no display.

## The problem

An LLM that writes `bpy` code in one pass produces toy-like shapes: wrong proportions, floating parts,
no way to say "move this ridge 8 mm". Vision models are poor at judging millimetres. Three things help:

1. **Numbers as ground truth.** Specify landmarks and dimensions, measure the mesh, compare. The
   [blender-modeler](https://github.com/calebrisc/blender-modeler) skill makes the same argument ("vision
   models are bad at spotting these, so the agent gets numbers"), and the practitioner write-up
   [Building an AI Agent That Models in Blender](https://akshaynathr.wordpress.com/2026/07/18/building-an-ai-agent-that-models-in-blender-what-worked-what-didnt/)
   reaches the same conclusion: the look-and-fix loop is the product, a judge that demands photorealism
   always says no, and a structured scene diff beats another vision call.
2. **Edit, do not regenerate.** LL3M ([arXiv 2508.08228](https://arxiv.org/abs/2508.08228)) shows that giving the
   coding agent the existing script keeps the asset and improves it; without it the agent rewrites a
   different asset.
3. **Compute placement, do not guess it.** Procedura ([arXiv 2608.26238](https://arxiv.org/abs/2608.26238)) solves
   each part's placement from typed mates and admits a part only after compile, mate and connectivity
   checks pass; an independent critic diagnoses one fix at a time. It targets hard-surface objects.

ViSculpt ([arXiv 2608.24169](https://arxiv.org/abs/2608.24169)) edits existing meshes through the Blender GUI. That
needs a window; this repo stays headless.

## Pixar's contribution: Regularized Kelvinlets

I could not find a Pixar paper about LLMs and modeling. The relevant Pixar work is a sculpting paper:
*Regularized Kelvinlets: Sculpting Brushes based on Fundamental Solutions of Elasticity*
(de Goes and James, Pixar Technical Memo #17-03, SIGGRAPH 2017,
[PDF](https://research.pixar.com/docs/2017.SiggraphPapers.GJ.pdf)). Its brushes are closed-form, so they need no
solver, no volume mesh and no GUI, and they support pointwise constraints with one linear solve. That
makes a sculpt move a precise, declarative operation, which is what an LLM needs.

What `prec.py` implements, read from the paper:

| Piece | Paper | `prec.py` |
|---|---|---|
| Regularized Kelvinlet kernel | Eq. 6: `K = (a-b)/r_e I + b/r_e^3 r r^T + a e^2/(2 r_e^3) I`, `a = 1/(4 pi mu)`, `b = a/(4(1-nu))` | `kernel()` |
| Brush parameterised by tip displacement | Eq. 7: `u(r) = c e K(r) u`, `c = 2/(3a-2b)` | `tip_gain()`, `grab()` |
| Multi-scale (fast falloff) | Sec. 5: bi-scale, tri-scale with `e_i+1 = 1.1 e_i` and weights cancelling the leading terms | `_scales()` |
| Pointwise constraints | Sec. 5/7: one linear solve for the forces | `solve_forces()` |

Not implemented: the twist, scale and pinch (locally affine) brushes. The grab brush with constraints
is enough for everything in the dragon head.

Poisson ratio is 0.45. Pins (zero-displacement handles, Poisson-disk sampled and mirrored) hold the rest of
the surface still, so an edit stays local.

## What works headless (measured)

| Operation | Result |
|---|---|
| `bpy.ops.sculpt.brush_stroke` | **fails**: `poll() failed, context is incorrect` (needs a window), hence Kelvinlets |
| Sculpt mode enter, dyntopo toggle | ok |
| `object.voxel_remesh`, Remesh modifier | ok (162 to 3012 verts at 0.08 m on a unit icosphere) |
| Multires subdivide, Decimate, Shrinkwrap, Displace | ok |
| Cycles bake NORMAL (selected to active), AO, DIFFUSE colour | ok, CPU |
| Smart UV project, BVH ray cast and nearest, Geometry Nodes via Python | ok |
| Texture-paint mode enter | ok (strokes are not tested; paint by writing pixel buffers with numpy) |

Probe script: `scenes/probe_caps.py`.

## Pipeline: dragon head from a 2D reference (`characters/dragon_head.py`)

Reference: the side view of the dragon skull in Monika Zagrobelna's "How to Draw a Realistic Dragon Head in 3D
Space" (Envato Tuts+, 2013; images `cdn.tutsplus.com/vector/uploads/2013/08/dragonhead_1-1_skull.png` and `..._1-3_skull_scheme_simplifeid.png`). The image is
third-party, so only numbers are committed: its binary silhouette (run-length coded) and hand-measured landmarks,
`characters/dragon_ref/ref_silhouette.json`.

```
reference outline + braincase ellipse  ->  stations (top, bottom, half-width, exponent)
   ->  HALF loft (x >= 0) + Mirror modifier          symmetric by construction
   ->  declared surface moves  ->  Regularized Kelvinlet constraint solve + pins, mirrored handles
   ->  eye protrusion measured, secant update (3 iterations)
   ->  silhouette-driven station correction against the reference outline (3 iterations)
   ->  horn crest tube from the outline residual, teeth by computation
   ->  high-poly (voxel remesh + Worley scales)  ->  bake NORMAL / AO / albedo onto the loft UVs  ->  glTF PBR
   ->  verify: topology, spec, bake fidelity, absolute-mapping IoU against the reference
```

The pixel to metre mapping is absolute (700 px per metre), so the IoU is measured with **no registration fitting**.

| Measured (full run, 245 s) | Value | Limit set beforehand |
|---|---|---|
| IoU vs the reference side view | **0.981** | >= 0.93 |
| top / bottom outline error, mean | 0.52 mm / 1.30 mm | <= 4 mm |
| top / bottom outline error, p95 | 1.43 mm / 1.43 mm | <= 12 mm |
| spec checks | 46 / 46 | |
| `rblender verify --pbr` | 25 / 25 files; glb has 3 baseColor, 2 normal, 2 occlusion maps | |
| low / high poly | 46,072 / 152,160 triangles | |
| low vs high surface distance | mean 0.36 mm, p99 0.92 mm | <= 0.8 / 2.0 mm |
| texel density | 3.79 px/mm mean (p5 2.75, p95 5.12) | |

Proof images are in `proof/dragon/`; the red/cyan overlay `dragon_ref_overlay.png` shows reference-only and
model-only pixels. The silhouette loop converged as 2.48, 0.39, 0.27, 0.23 px mean top outline error.

What the measuring loop found that I would not have seen by eye (each is fixed in the code):

- **The first version did not resemble the reference at all** (long low snout, no braincase, no horn crest): it
  scored well against its own spec. A spec you wrote yourself only proves self-consistency; get a reference.
- **Two loops fighting over one number.** The brow height target came from the reference outline, but I measured it
  3D at a lateral offset where the surface is 20 mm lower. The secant loop pushed the brow to 60 mm and the next
  iteration exploded. The silhouette loop already owns that height, so the brow is now a fixed edit and the height is
  verified from the silhouette.
- **A tube whose rings are perpendicular to a sloping path bloats the silhouette** by about `r * sin(slope)`. The
  horn base tilted 61 degrees and sprouted a 40 px block. `prec.tube(axis=...)` keeps the rings in planes
  perpendicular to an axis.
- **An outline loop must ignore columns the model does not cover yet.** The rounded jaw tip left uncovered columns;
  the loop read them as -36 px error and pulled a station 31 px down. Clamp the step and mask the columns.
- **Thin or close handles make the Kelvinlet solve ill-conditioned.** `solve_stable()` retries with Tikhonov
  regularisation until the field on the mesh stays finite and bounded.

Design decisions worth copying:

- **The low-poly is the loft itself.** Its quads follow the head and its UVs are a clean grid, so no retopology or
  auto-unwrap is needed and the bake has uniform texel density.
- **Model one half, let Blender mirror it.** A Mirror modifier is exact by construction and replaced my manual
  mirrored anchors and averaged fields. It gives left to right only: a side view says nothing about width, so the
  width profile (and the superellipse exponent) stay an assumption.
- **Keep the best iterate.** A correction that makes the error worse is discarded.
- **Brush radius must exceed local vertex spacing**, or the move cannot be represented.
- **Colorspace before pixels.** Setting `image.colorspace_settings.name` after `pixels.foreach_set` silently zeroes
  the buffer in Blender 4.5.

## Practice against a 3D target (`characters/dragon_ref3d.py`)

Target: *Red Dragon* by GadgetHamster ([Sketchfab](https://sketchfab.com/3d-models/red-dragon-77e89ab9b88a444cb51339c2ed3959a3),
CC-BY-4.0), a rigged full-body dragon. The glb is not in the repo. `characters/dragon_ref/prep_glb.py` reads its raw
bind-pose positions and **skin weights** with numpy and splits the head exactly (upper, jaw, horns, tongue, teeth)
instead of guessing from geometry. The target is only ever measured:

```
python3 characters/dragon_ref/prep_glb.py refs3d/red_dragon.glb refs3d/red_dragon_head.npz
rblender run characters/dragon_ref3d.py -- refs3d/red_dragon_head.npz [--fit N] [--direction closest|normal] [--no-render]
```

1. **loft**: stations (top, bottom, half-width, exponent) cut from slabs of each part; the exponent makes the
   superellipse area equal the slab's cross-section area (closed occupancy grid, not a convex hull, which fills the
   valleys between spikes). The open jaw gets its own frame along its principal axis. Horns and eyes are tubes and
   spheres from a principal curve.
2. **fit**: Regularized Kelvinlet pulls toward the target, coarse to fine (handle spacing 50, 35, 25 mm, 3 iterations
   each, `prec.fit_coarse_to_fine`). Each handle takes the median pull of its 8 nearest handles, the solve is
   regularised, and an iterate rougher than 1.5x the target's mean dihedral angle is never kept: accuracy is pursued
   inside a smoothness budget. Horns are not fitted (the tube is smooth by construction; fitting it twisted it).
3. **residual features**: target vertices still >12 mm from the model are grouped by mesh connectivity; each elongated
   group whose root touches the model (<= 15 mm) becomes a tapered tube with a buried root (6 in the default run).

Scores are in metres against the target, with limits fixed before the first run (fractions of the 890 mm extent):
accuracy = model to target surface distance, completeness = target to model, IoU on side/front/top silhouettes with
absolute mapping, roughness = mean dihedral angle.

| Stage | accuracy mean / p95 | completeness mean / p95 | IoU side / front / top | skull / jaw roughness |
|---|---|---|---|---|
| loft from measured stations | 18.3 / 53.6 mm | 16.7 / 53.6 mm | 0.917 / 0.742 / 0.895 | 11.6 / 10.9 deg |
| earlier fit: one spacing, exact solve, no filter | 7.6 / 30.4 mm | 6.2 / 26.4 mm | 0.922 / 0.907 / 0.925 | 32.3 / 31.8 deg (crumpled) |
| earlier fit: normal rays only, median-filtered | 12.6 / 49.3 mm | 8.4 / 34.3 mm | 0.906 / 0.766 / 0.906 | 22.3 / 21.6 deg |
| **coarse to fine, robust, roughness budget (default)** | 9.4 / 33.1 mm | 8.8 / 34.1 mm | 0.903 / 0.872 / 0.886 | 21.1 / 19.7 deg |
| target itself | | | | 14.9 / 16.2 deg |

Limits: accuracy mean <= 4.4 mm and p95 <= 17.8 mm, completeness mean <= 8.9 mm, IoU >= 0.90, roughness <= 1.5x the
target. **The default run passes 7 of 11 checks**: the two smoothness checks pass now; it still fails both accuracy
limits and the front and top IoU.

Against the crumpled fit the default gives up 1.8 mm of mean accuracy and cuts roughness from 32 to 21 degrees; the
render (`proof/ref3d/ref3d_model.png`) is smooth, with none of the small creases. It is still **not a good copy** of
`ref3d_target.png`: the head is a smooth wedge, the horns are round and thick instead of tapering blades (a round tube
cannot follow a flat horn), and brow, nose and frill spikes are mostly missing. Lessons:

- **Silhouette IoU and Chamfer distance can be satisfied by a crumpled surface.** Add a smoothness metric (mean
  dihedral angle here) and look at the render. The distance-only fit scored 7.6 mm and looked wrong.
- **Per-vertex distance flatters the fit.** In an earlier run the skull's mean vertex-to-target distance after the
  fit was 2 mm while its sampled surface distance was 9.8 mm: the faces between vertices cross gaps. Always score
  sampled surface points.
- **What made it smooth** (each step measured): the median over neighbouring handles (neighbours hit different sheets:
  outer skin, inner mouth, a spike), a regularised solve instead of exact interpolation, wide handles first, and a
  roughness budget that rejects a rough iterate. Pulling along normals only looked smooth but barely moved the surface
  (mean vertex distance 25.8 to 19.3 mm): it passed smoothness by not fitting.
- The roughness budget is the binding limit: the skull's mean vertex distance was still falling (7.0, 5.9, 5.4,
  5.2 mm over the last stage) when the budget (22.4 deg) stopped it at 21.1 deg.
- A loft of superellipse cross-sections cannot make brow spikes, a nose horn or a frill; the residual step recovers
  part of that (completeness 16.7 to 8.8 mm) but as separate tubes, and floating ones must be rejected.
- My own scoring artefact: the loft's flat back cap closes a cut I made in the target, so those samples are excluded
  from accuracy (and reported as such in the script).

Next steps I would try: an elliptical horn section with a taper profile, a Laplacian pass on the displacement field
instead of a roughness cap, fitting against a smoothed copy of the target, and building the frill as a half-loft of its
own rather than as tubes.

## Using it from an agent

```python
import bpy, rb, prec
P0 = prec.coords(obj)
tree = prec.bvh_of(obj)
loc, nor, _ = prec.ray(tree, (0.07, -0.11, 0.4), (0, 0, -1))      # find the surface point
H = [loc, ...pins...]; D = [nor * 0.012, ..., zeros]               # raise it exactly 12 mm, pin the rest
F, residual, reg, U = prec.solve_stable(H, D, [0.022, ...], P0, max_disp=0.05)   # one stable linear solve
prec.set_coords(obj, P0 + U)
rep = prec.Report(); rep.num("brow", "height", measured_m, target_m, tol_mm)   # pass/fail table
# against a reference mesh: prec.target_mesh(), prec.chamfer(), prec.silhouette(view=...), prec.fit_to_target()
```

`characters/dragon_head.py` is the complete worked example; `characters/dragon_ref3d.py` shows measuring against a mesh.

## Honest limits

- The 2D reference is a skull drawing, not skin, and a side view only. The head matches it (IoU 0.981) but the
  width, the superellipse exponent and the horn cross-section are assumptions; the front and top views are not
  checked against anything. The render reads as a crocodile with cone horns.
- The 3D practice does not reproduce the target: silhouettes 0.87 to 0.91 IoU, a smooth but featureless wedge with
  round horns; teeth, tongue, frills and the open-mouth interior are not modelled.
- Texture bake covers skin only. Horns, teeth and eyes are plain materials.
- The scoring metrics I chose did not catch the crumpling by themselves; I added roughness only after looking at the
  render. Expect your own metrics to have the same blind spots.
- CPU only; the full 2D run takes about 4 minutes, the 3D practice 1 to 3 minutes on 4 vCPU. The Kelvinlet solve
  scales with handles squared, so a few thousand handles is the practical limit.
