# The control library, for the agent driving it

Everything here is measured on Isaac Sim 6.0.1 unless it says otherwise. The
numbers are the point: most of them cost a debugging session to find, and none
of them are guessable from the API.

---

## 1. Build cells out of real assets

The asset library ships **175 indexed props**, including **47 conveyor
sections** and **23 people**. A cell authored out of cubes and cylinders reads
as a mock-up no matter how good the physics under it is, and the first thing
anyone says about it is that it looks wrong.

```python
from simliverse_sim import list_props, find_prop, spawn_prop

list_props("conveyor")     # 47 sections
list_props("worker")       # people, under /Isaac/People/Characters
find_prop("pallet")        # 1.21 x 0.80 x 0.1425 m
```

**Search before you author.** The index carries `extent` and `physics` for
every entry, so `find_prop` answers "how big is it" without loading anything.
If a search comes back empty, say so — do not silently build the thing out of
primitives instead.

### Props that matter for a palletising cell

| what | key | note |
|---|---|---|
| conveyor | `conveyorbelt_a05` | 2.0 m section, **carries at 0.767 m** |
| pallet | `pallet` | 1.213 x 0.802 x 0.1425, **deck at 0.1425** |
| robot mount | `ur10_mount`, `stand` | 0.515 m / 0.618 m tall |
| worker | `male_adult_construction_01_new` | ~1.95 m |
| forklift, dolly, packing table | `forklift`, `dolly`, `packing_table` | dressing |

---

## 2. Placement traps, all of them measured

These are the bugs that do not raise. Every one cost real time.

**Props are placed by their centre and are large.** A pallet is 1.21 m long,
so `pallet_y=0.60` puts its near edge at **-0.005** — the arm's base is inside
the pallet, PhysX reports invalid transforms on seven links, and the scene is
quietly unusable. `spawn_prop` checks this and warns; `demo.ur10_palletizing.build`
records it as `cell["fouled"]`.

**A character's origin is not at its feet.** The bounding box bottom sits
**0.12 to 0.16 m below** the origin, so placing one at `z=0` buries it to the
shins and it reads as a short person, not a bug. Use
`guarding.spawn_operator`, which measures the bound and drops the figure.

**A conveyor prop's deck is not its bounding box.** `ConveyorBelt_A05` tops
out at 1.166 m because that includes the side frames; the **rollers carry at
0.767 m**. Aligning to the box puts the belt surface 0.4 m out.

**`size` means height, and only `Cube` has a `size` attribute.** For a
cylinder or capsule it maps to `CreateHeightAttr`. Before this was fixed, a
cylinder asked for 0.82 m came out at its default 2.0 and the figure it was a
leg of stood through the floor, silently.

**`scene.stop()` does not empty the stage.** Two cells built in one session
share it, and the older one is still solid — an escapement blade left from an
infeed experiment held a palletising cell's carton queue with 26 N while every
belt observable said the conveyor was fine. Call `scene.clear_world()`, which
keeps only the physics scene, its materials and the floor.

**`spawn_prop` faces world +X unless told otherwise.** Every indexed asset
keeps the local frame it was authored in, so a conveyor prop referenced onto a
belt running any other direction renders facing the wrong way — off the end
of the physics slab it is meant to cover, not on top of it. Pass
`orientation=[0, 0, yaw]`. `Conveyor.dress()` already does this.

---

## 3. Guarding: what makes it a cell rather than an arm on a floor

```python
from simliverse_sim import SafetyFence, spawn_pedestal, spawn_operator

fence = SafetyFence.build(
    "/World/Fence",
    centre=(0.0, 0.0), size=(6.0, 6.0),      # the guarded AREA, not a guess
    gate="south", gate_width=1.0,
    crossings=[{"side": "east", "centre": -0.4, "width": 0.7}],
)
fence.fits((0.0, 0.0), reach=2.70)   # does the arm stay inside the guarding?
```

- `size` is the **footprint**, because a cell is specified by the floor it
  occupies. Every entry point in `guarding` takes the area, never a centre and
  a half-extent.
- **Size each side by the larger of the equipment on it and the arm's reach
  envelope.** Sizing off equipment alone put a fence line 0.95 m from a 1.3 m
  arm — guarding inside the working envelope, on the one side nothing else
  stuck out of.
- A **crossing** is a gap for something that must pass through the line.
  Author it; a panel across a conveyor is edge-on from every camera angle
  anyone would take and stops every carton.
- Only cut a crossing **if the belt actually reaches the fence line**.
  Otherwise it is a doorway onto nothing.
- Panels are **static colliders and translucent**. Translucency needs a
  `UsdPreviewSurface` with an `opacity` input — `displayOpacity` authors
  cleanly, reports success, and RTX ignores it.

A pedestal is **structure, not scenery**: its top *is* the robot's base
height, so spawn the arm at `plinth["top"]`. Drawing a plinth under an arm
still on the floor gives a robot growing out of a crate.

An operator inside the guarding is reported (`inside_guarding`), not refused —
it is what you draw to illustrate a teach pendant, and it should be a choice.

---

## 4. Build straight from a drawing

```python
from simliverse_sim import fence_from_sketch, zones_from_sketch

built = fence_from_sketch(sketch_text)      # rect -> fence, arrow -> crossing
zones = zones_from_sketch(sketch_text)      # every shape, as placeable numbers
```

A `[LAYOUT SKETCH ...]` block is plan-view shapes in metres, taken off a grid
by hand. Treat the numbers as the requested layout — Isaac is Z-up, so the
canvas is already the XY plane and nothing needs rescaling or reprojecting.

- **A rectangle is the cell.** Picked by label first ("cell", "fence",
  "guard", ...), size only as a fallback — `chosen_by` in the result says
  which rule fired, and `"largest, unlabelled"` means nobody said and it
  guessed.
- **An arrow crossing the footprint becomes an opening.** One drawn wholly
  inside means travel direction and is left alone.
- **A circle labelled "operator"/"worker"/"person" decides which side the
  gate opens on** — nearest that circle. Leave `gate` unset for this to fire;
  passing `gate=` explicitly, `None` included, always wins outright. The
  default gate side used to be a hard-coded `"south"` no matter what was
  drawn — it only ever looked right because the worker happened to be drawn
  south of the cell, and the same default would have fired with the gate
  opening onto nothing had they been drawn anywhere else.

---

## 5. Conveyors

```python
belt = Conveyor.build(
    "/World/Belt", length=2.0, width=0.4,
    position=[x, y, 0.767],          # belt TOP, not centre
    direction=(1, 0, 0), speed=0.2,
    gate=True, gate_height=box + 0.03,
    guides=False,                    # see below
    dressing="conveyorbelt_a05",     # real prop over the physics slab
)
```

- `position` is the **belt top**, so it can be set to a working height without
  arithmetic.
- `dressing` references a real conveyor and hides the primitive slab. The slab
  stays as the physics — kinematic body, surface velocity, known deck, a stop
  — and stops being what anyone looks at.
- **Dressing rotates to the belt's own direction, and tiles to its full
  length — neither used to happen.** Every indexed prop keeps the local frame
  it was authored in; a belt built running -X still got a dressing prop
  facing world +X, which rendered off the far end of the physics slab it was
  meant to sit on. And `ConveyorBelt_A05` is 2.0 m regardless of belt length,
  so one section over a 6.4 m belt left 4.4 m visibly bare. Both are fixed in
  `dress()`; a caller does not need to think about either.
- **A halted belt is a sleeping belt.** PhysX does not wake a body because the
  surface under it started moving. `start()` nudges every tracked body; without
  it the trace reads "no carton settled at the stop" while
  `surfaceVelocityEnabled` is `True` and the cartons sit at v=0.
- **Cartons accumulate against the stop**, so the next one is touching the one
  being picked. Keep `max_grip_distance` well under the carton size or the cup
  seals on the neighbour.
- `guides=True` fits side rails. They stop a queue squirting cartons off the
  belt (117 mm off-centre at 1 kg, 609 mm at 1.5 kg on a 400 mm belt) **and**
  they currently cost placement accuracy, because the pick is not centred and
  a carton normally slides into line under the cup as it lifts. Off by default
  until the approach lands centred.
- **`from_prop("conveyorbelt_a09", position=...)` drives the real asset**, and
  its `position` means the centre of the belt deck's footprint — the assets
  author their origin wherever the artist left it (A09: the discharge end),
  and the library recentres so the belt lands where the layout says. Name a
  straight section: the fuzzy query "conveyor belt" resolves to A01, the
  90-degree curve. In a scene a customer looks at, prefer `from_prop` over a
  primitive `build()`.
- **A built belt stamps `describe()` onto its prim** (`simliverse:conveyor`),
  so `Conveyor.attach("/World/Sketch/r1")` with no other arguments works in a
  session that did not build it. Explicit arguments override the stamp; only a
  hand-authored belt still needs all four numbers.

---

## 6. Robots

| key | reach | note |
|---|---|---|
| `ur10` | 1.3 m | the measured cell |
| `kuka_kr210` | ~2.7 m | 150 kg payload, cuMotion config `Kuka_KR210` |

`HOME` in the demos is six joint angles measured on a UR10. Do not hand it to
another arm — it is either a shape error or, worse, a silent pose on a
different kinematic chain.

Motion: prefer `plan_to(...)` + `follow(...)` (cuMotion, collision-aware) for
long moves, `servo_to` for short refinements. Pass `robot_name=` to `plan_to`
or cuMotion cannot find a configuration.

Suction: force limits of 500 (Isaac's tutorial value) break the seal within
2 mm of any motion. Use `1.0e6`. **Writing any `isaac:*` attribute on a closed
gripper releases it**, so do not "test" limits mid-grip.

---

## 7. Look at the scene before believing it

```python
from simliverse_sim import vision
vision.look(centre=(0, 0, 0.6), scale=3.0)   # four views, not one
```

Four viewpoints by default. Every visual defect found in this cell was visible
from one direction and invisible from the others. `scale` backs the rig off —
the offsets are sized for a cell about a metre across, and a hero camera at
2.6 m stands *inside* a 3.7 m fence photographing a panel.

`looks_blank` per view catches an unlit stage and a camera aimed at nothing.

---

## 8. The measured palletising cell

`demo.ur10_palletizing.build()` — every default is the cell the numbers came
from. `demo.guarded_cell.build()` wraps it in guarding with a KR210 on a
plinth, a dressed conveyor and a worker at the gate.

Baseline: **2/2 placed, 3.0 and 5.1 mm, 58.69 s/carton.**

Working envelope, from `demo/scenario_sweep.py` (13 scenarios):

- carton **0.15–0.22 m at about 1 kg**
- belt **0.1–0.4 m/s**, deck **0.45–0.55 m**
- pallet **0.68–0.95 m** from the arm base

Outside that it does not currently work, and the sweep keeps those rows red
rather than dropping them. See `docs/CELL_SCENARIO_FINDINGS.md`.

**The pick has never been centred**, and the cell's own accuracy number hides
it: a carton sealed off-centre slides into line under the cup while lifted. The
3–5 mm above is partly the carton correcting the approach on its way up.
