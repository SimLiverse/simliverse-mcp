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

list_props("conveyor")  # 47 sections
list_props("worker")  # people, under /Isaac/People/Characters
find_prop("pallet")  # 1.21 x 0.80 x 0.1425 m
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
    centre=(0.0, 0.0),
    size=(6.0, 6.0),  # the guarded AREA, not a guess
    gate="south",
    gate_width=1.0,
    crossings=[{"side": "east", "centre": -0.4, "width": 0.7}],
)
fence.fits((0.0, 0.0), reach=2.70)  # does the arm stay inside the guarding?
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

built = fence_from_sketch(sketch_text)  # rect -> fence, arrow -> crossing
zones = zones_from_sketch(sketch_text)  # every shape, as placeable numbers
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
    "/World/Belt",
    length=2.0,
    width=0.4,
    position=[x, y, 0.767],  # belt TOP, not centre
    direction=(1, 0, 0),
    speed=0.2,
    gate=True,
    gate_height=box + 0.03,
    guides=False,  # see below
    dressing="conveyorbelt_a05",  # real prop over the physics slab
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
- **A curve is `CurvedConveyor.build_curve`**, generic over any bend:
  `build_curve(centre=[x,y], radius=1.55, start_angle=-90, turn=90, ...)`
  sweeps `turn` degrees from `start_angle` about `centre`. PhysX cannot drive
  one curved surface along an arc, so it is a fan of short flat chord slabs,
  each driven local-space along its own tangent; a carton crossing between
  them is handed a velocity turned a few degrees and rides the bend. Measured
  live: cartons followed a 1.2 m arc through the quadrant, on the deck.
  Trade-off: a carton spans ~1.5 slabs whose tangents differ, so it travels
  slower than belt speed — use fewer, longer segments and a higher `speed` for
  a brisk bend. `box_at_gate()` here finds the carton at the exit **angle**,
  and `load()` spaces cartons by **degrees** (`spacing_deg`), the natural
  coordinate on a circle. Dress it with the real curve prop:
  `build_curve(radius=1.55, dressing="conveyorbelt_a01")` (A01 is a two-tier
  90-degree quadrant, R~1.55, rollers at 0.74).
- **A halted belt is a sleeping belt.** PhysX does not wake a body because the
  surface under it started moving. `start()` nudges every tracked body; without
  it the trace reads "no carton settled at the stop" while
  `surfaceVelocityEnabled` is `True` and the cartons sit at v=0.
- **`pitch=` (degrees) tilts the belt so cartons ride UP to the stop.**
  Measured live: a 1 kg carton climbed 10-25 deg and settled against the
  raised stop, driven by a surface velocity along the slope (local space, not
  horizontal — otherwise it drives the carton into the deck). The deck
  friction must exceed `tan(pitch)` or it slides back; `build()` warns when it
  does not, and 0.9 is comfortable to ~25 deg. `position[2]` stays the deck at
  the belt centre, so a pitched belt pivots about its middle; the stop, the
  guides and the loaded cartons all tilt and rise with it, and `box_at_gate`
  measures the rest height at each carton's own point on the slope. **Also
  fix the slope-vs-run trap:** `length` is up the slope, but along-heading
  distances are horizontal, so `load` and `box_at_gate` use `half_run()`
  (`length/2 * cos(pitch)`) — at 30 deg a carton placed at the slope distance
  lands past the belt's end and falls (15 deg hid it).
- **Dress an incline with a real ramp prop, not a grey slab:**
  `build(pitch=30.7, dressing="conveyorbelt_a42")` puts the shipped roller
  ramp (blue frame, grey rollers) over the driven slab. The ramp props are
  fixed ~30 deg flat-ramp-flat models (`RAMPS`: A42 rollers 0.70->1.76, A37
  belt 0.74->1.78), placed flat and dropped so the low deck lands on the
  slab's low end; `dress()` warns if the belt's pitch does not match the
  prop's. A steep ramp is a feeder — it lifts a carton fast, so `box_at_gate`
  at the very top is unreliable; put a flat section at the top for a pick.
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

## 5b. Placing with a suction tool — measured on a KR210 (Isaac Sim 6.0.1)

Every number here came from a cell built out of a layout sketch and run
end to end; each one was a failure first.

- **The reachable annulus is 1.20–2.70 m** from the base, at a carton's
  release height, with the flange vertical. Both ends bite: inside 1.20 m the
  arm cannot fold in tight enough to point the tool down and IK has *no*
  solution, which looks nothing like "out of reach". An earlier outer figure
  of 1.90 m came from RMPflow servo failures and measured the policy, not the
  robot.
- **The annulus has a ceiling, and it drops with distance.** With the tool
  down, the KR210 on a 0.35 m pedestal can hold a slot 1.77 m out up to
  z = 0.82 m and one 1.93 m out only up to z = 0.66 m. A traverse at a fixed
  0.76 m had no IK solution over the far column; `pose_to(raise_on_fail=False)`
  returned *without moving*, the descent went part-way, and the cup opened
  0.55 m from the slot — reported as a placement error, which it was not. Ask
  before committing: `arm.can_reach(pos, DOWN)` solves without moving, and
  `arm.reach_ceiling(xy, DOWN, floor=release_z)` bisects the highest z the
  tool can be held down at. `build()` now returns `cell["reach"]` with a
  ceiling per slot and the unreachable ones named; a third layer on the far
  column is a layout error, not a cycle that will fail later. Never open the
  cup after a move whose `MotionResult.reached` is False — check every one.
- **Route the last centimetres, do not servo them.** RMPflow trades
  orientation against position and settles ~0.165 rad off vertical at the edge
  of reach — enough to set a carton on a corner. The same descent through
  `route_to` (exact IK, or no answer) arrives at 0.002 rad and places within
  8 mm.
- **Ask for the slot's yaw, accept the reach-facing one.** A 6-axis arm cannot
  present every orientation at every reach: a slot that solved reach-facing had
  no solution at its own yaw. Prefer square, fall back, and say which was used.
- **Lift straight up before moving away.** One move from the place pose to
  carry height lets the policy reconfigure the whole arm, dragging the cup —
  3 mm above the carton it just released — sideways through it.
- **Leave 50 mm between cartons, not 10.** Tight pitch transmits placement
  error along the row: cartons landing 16–137 mm off shouldered each other
  until one placed at 16 mm had drifted to 228 mm, untouched by the arm.
- **Verify the stack at the end, not each carton as it lands.** Checking one
  carton immediately proves only that it arrived. A run reported "4 placed and
  verified" while `verify_pallet` found three on the deck and one on the floor.

- **Route what has to arrive; servo only to refine.** `servo_to` is a local
  reactive policy, and where reaching a pose means turning the base most of the
  way round it stalls in a local minimum rather than going the long way. Two
  pick states each burned a 1001-frame budget that way, closing the cup on
  nothing and retrying a seal against thin air until the run looked hung --
  nothing was out of reach. `route_to` solves the goal configuration and drives
  to it, base rotation included.
- **A pose has two wrist solutions; `pose_to` now takes the near one.** a4
  and a6 turned by pi with a5 negated reach the same tool pose, and Lula
  returns whichever its seed falls nearest — from a pick configuration that is
  a coin toss. Three cycles to one slot solved `[.., 0.05, 2.17, 0.03]`, the
  fourth `[.., 3.08, -2.18, 3.10]`, and the 4.3 rad wrist swing on the way put
  the carried carton through two already on the pallet and threw one 1.28 m.
  `command_pose` re-solves from wrist- and base-flipped seeds and keeps the
  solution with the smallest single-joint move. If you drive joints yourself,
  compare the solution against the current configuration before applying it.
- **Seed IK from the arm's current pose, not from a home pose.** Lula returns
  the solution nearest its seed. A HOME seed asks for the configuration nearest
  *home*, so the arm swings out toward home and back on every single move; the
  symptom is a robot that visibly circles between picks, and placements that
  arrive in a configuration discontinuous with the one before.
- **Mount the tool on the flange face, and aim that axis at the floor.** The
  face normal is not tool Z on every arm -- a KR210's is tool X -- and a tool
  mounted on the wrong axis bolts to the side of the wrist. `approach_axis` is
  measured, stamped on the gripper prim, and read back on rebind, and
  `down_at_yaw` sends *that* axis to world -Z.
- **Release in the orientation you gripped in.** The cup fixes the carton
  relative to the flange, so an identical orientation returns it to the yaw it
  had on the belt. Correcting the yaw at the slot instead spun the wrist to
  arbitrary angles -- 21 and 41 degrees off, never settling on 0 or 90.
- **Meter the queue with `deadplate.Escapement`, do not stop the belt.** One
  carton released per cycle, blade back up behind it, line running throughout.

---

## 5c. Cables, hoses and dress packs

Isaac Sim 6.0.1 has no cable. `deformabletube_tube` is a static collision
mesh; `PhysxDeformableBodyAPI` is gone and its FEM replacement needs GPU
dynamics and a scene rebuild. `list_props("cable")`, `hose`, `wire`, `rope`
all come back empty — say so rather than spawning a grey cylinder.

```python
from simliverse_sim import Cable, verify_cable

cable = Cable.build(
    "/World/Dress",
    start=[0, 0, 1.5],
    end=[2, 0, 1.5],
    slack=0.10,
    radius=0.012,
    anchor_start="",
    anchor_end="/World/KUKA/link_6",
)
scene.settle(2.0)
print(verify_cable(cable))  # ends on their anchors, sag, settled
```

A chain of capsules on spherical joints, laid along a parabola with the
slack already in it. Measured in the live cell with GPU dynamics off:
sixteen 12.5 cm links over 2 m settle in 2 s to a 7.7 cm sag, no
self-collision, no drift. `anchor_*` is `""` for the world, a prim path to
ride on a body (a flange, a cabinet), or `None` to hang free. `slack` is
what makes it a cable: at 0 it is a bar.

---

## 5d. Mobile robots in a warehouse

```python
from demo.warehouse_amr import deliver

print(deliver(goal=[6, 3], waypoints=[[4, 0]], environment="Simple_Warehouse"))
```

`Robot.spawn("carter")` is a `WheeledRobot`; `jetbot` and `kaya` also exist.
`drive_to([x, y])` closes the loop on the base pose and returns whether it
arrived after braking (a base coasts, so it is measured stopped, not moving).
`verify_navigation(rover, goal, start_position=)` is the acceptance check —
physics running, moved under its own wheels, reached the goal — measured live:
a Carter drove ~4 m in the Simple_Warehouse and passed all three.

- **`drive_to` is not a planner** — it turns to face the goal and drives at
  it, into any rack in the way. `plan_path(waypoints)` smooths a route you
  supply but finds and avoids nothing. Hand `deliver` the via-points that keep
  the aisle; it drives them loosely (`VIA`), the goal to the arrival tolerance.
- **One tolerance for the drive and the verifier.** `drive_to(0.3)` reaching
  0.30 m while `verify_navigation(0.25)` calls it failed is two truths that
  disagree; `deliver` hands both the same number.
- **Reference the environment and add a light.** A `clear_world()` drops the
  warehouse's own lights and the scene renders black; `deliver` adds a dome.
  Spawn the base at z=0.3 and settle 20 steps or the first wheel command
  launches it off the interpenetrated floor.

---

## 5e. Other morphologies: drones, quadrupeds, humanoids

`list_robots()` carries a `morphology` and a `cartesian_control` flag per asset.
The library drives each body by the primitive that suits it, and refuses to fake
the ones that need more than it has.

- **Drones fly.** `Robot.spawn("quadcopter")` is an `AerialRobot`, thrust-
  controlled: `hover(steps=)`, `fly_to([x, y, z])` (a PD controller, not a
  planner - it flies straight through obstacles, so route it with waypoints),
  `apply_thrust`, `altitude`. A drone is a **rigid body, not an articulation** -
  keep the spawn handle after a timeline cycle; `Robot.attach` fails on it with
  "no articulation registered". Measured: a quadcopter flew a four-waypoint loop,
  each reached within 0.3 m, hover held to millimetres. `demo/drone_patrol.py`.
  Other airframes: `crazyflie`, NASA `ingenuity`.
- **Humanoids stand; walking needs a trained policy.** `Robot.spawn("h1")` is a
  `Humanoid` with `stand`, `walk`, `is_upright`, `limbs`. Measured: it spawned,
  stood, and `is_upright()` returned True (base at 1.04 m). `walk` is there, but
  a gait is a learned policy this library does not ship - a `walk()` with no
  policy holds a pose, it does not stride. Say so rather than teleporting the
  body, which is what `not_teleported` exists to catch. `g1`, `digit` likewise.
- **Quadrupeds spawn but their control is rough.** `spot`, `anymal_c/d`,
  `go1/go2`, `laikago` load, but some keys are mis-typed (measured: `spot` came
  back as a `Manipulator` and fell), and a 12-DOF gait needs a policy either
  way. Spawn and hold a pose; do not claim a walk you cannot run.
- **More AMRs and mobile manipulators, measured.** `turtlebot3` and `novacarter`
  drove ~1.8 m to a goal exactly like the Carter (section 5d); `dingo`, `jetbot`,
  `kaya` are the same primitive. `ridgebackfranka` / `ridgebackur` are arms on a
  driven base (`MobileManipulator`) - drive the base, then move the arm.

**Two harnesses test the whole fleet, not one cell.** `scenario_sweep`
(`generated_scenarios`) invents 100 palletising cells across arm x carton x
pattern x deck x dressing x guards, each reach- and deck-bounded so it is
placeable. `fleet_scenarios` (`fleet`, `run_fleet`) crosses the *other* bodies -
every mobile base x every warehouse floor, plus drone routes - by pointing each
scenario at the verified demo that runs it (`deliver`, `patrol`). Run them to
prove a change still works on a different body in a different environment, not
just on the one cell it was tuned on.

---

## 6. Robots

`list_robots()` discovers everything under `/Isaac/Robots/<Vendor>/<Model>`:
nine UR models (`ur3`..`ur30`), about eighty Fanuc (`crx10ia_l`, `lrmate200id*`,
`m20*`, `r2000ic_210f`...), `kuka_kr210`, Kawasaki `rs007l`..`rs080n`, Denso
`cobottapro900/1300`, Techman `tm12`, Franka/FR3 and more. Having the asset is
not the same as being able to drive it Cartesian: `pose_to` and `route_to` need
an RMPflow/Lula config, and only 21 arms have one. Check
`describe()["motion_config"]` before promising a cycle.

| key | reach | Cartesian (`pose_to`) | drive gains | note |
|---|---|---|---|---|
| `ur10` | 1.3 m | yes | 1e5 / 1e4 / 1e4 | the measured cell |
| `ur3`, `ur3e`, `ur5`, `ur5e`, `ur10e`, `ur16e` | 0.5–0.9 m | yes | UR10 values, <3 mm | `ur20`/`ur30` have no config: joint control only |
| `kuka_kr210` | ~2.7 m | yes | 1e5 / 1e4 / **1e6** | 150 kg payload; at 1e4 the first three joints barely move |
| `crx10ia_l` (Fanuc) | 1.2 m | yes (auto-matched now) | 1e6 | drives and points down (tool axis -Y); pick not yet completing — shoulder clearance is a known gap |
| `r2000ic_210f`, `lrmate200id`, other Fanuc | — | **no** | — | `MotionError: No RMPflow configuration` |
| Kawasaki, Denso, Techman, Franka, Flexiv | — | yes | untested | RMPflow configs ship for these |

**Clear the arm's body, not just its base point.** A Fanuc CRX-10iA/L's
shoulder housing reaches 0.30 m out at belt height; a UR10's about 0.15 m.
With the belt edge 0.23 m from the base axis the CRX's base joint stopped at
-43 degrees on every move toward the belt — the IK solution was right, the
drive caps were right, and the pick reported "hover not reached: 1.08 m
short". `build()` measures `arm_footprint(arm, below=deck + 0.25)` off the
link bounds and pushes the belt and the deck out past it (`cell["clearance"]`
records what moved); a failed move now says what the arm is `touching`. If
you lay a cell out by hand, keep every edge outside that radius and use
`capture_view` — the jam was obvious in one render and invisible in a page
of joint numbers.

Gains are per robot (`demo.ur10_palletizing.DRIVE_GAINS`), not one global
default: the UR family holds at `max_force=1e4`, everything heavier needs
`1e6` (a CRX's shoulder sat sagged at -0.82 rad at 1e4 while joints 3-6
tracked to the milliradian). Lay the cell out to the arm, not the arm into the UR10's cell:
`layout_for("ur5e")` scales the pick point, belt height and stack to the
arm's reach and swaps the 1.21 m pallet for a tote a small arm can span —
`build(**layout_for(robot))`. The pallet is the constraint: 0.80 m wide and
placed by its centre, it cannot get nearer than 0.65 m, past a UR5e's stack. `HOME` in the demos is six joint angles measured on a UR10. Do not
hand it to another arm — it is either a shape error or, worse, a silent pose
on a different kinematic chain.

Motion: prefer `plan_to(...)` + `follow(...)` (cuMotion, collision-aware) for
long moves, `servo_to` for short refinements. Pass `robot_name=` to `plan_to`
or cuMotion cannot find a configuration — and cuMotion ships configurations
for `franka` and `ur10` only, so on every other arm the long moves are
`route_to`/`pose_to` (Lula IK, no collision awareness).

Grippers: `arm.suction` (one cup, `holding`, `gripped_objects`) and `Gripper`
(parallel jaw, auto-detected from an asset's own finger joints — Franka hand,
`rs013n_onrobot_rg2`). A bare UR, KR210 or Fanuc ships with no jaw; bolt one
on:

```python
arm = Robot.spawn("ur10e")
fit = arm.attach_gripper("2f_85")  # stops the timeline; the articulation changes
scene.play()
scene.step(10)
arm = Robot.attach("/World/ur10e")  # rebuild the handle: arm.gripper is the jaw
arm.gripper.open()
arm.gripper.close(settle_steps=45)
arm.is_grasping(box)
```

The finger grippers in the library (`list_robots()` finds them under
`/Isaac/Robots/Robotiq` and `/Isaac/Robots/Schunk`; there is no OnRobot,
Zimmer or WSG on the server), read off their USDs:

| key | mechanism | driven joint | travel | note |
|---|---|---|---|---|
| `2f_85` | Robotiq 2F-85 linkage | `finger_joint` | 0–47° | mimic followers, 0.21 kg |
| `2f_140` | Robotiq 2F-140 linkage | `finger_joint` | 0–45° | every joint driven in the physics edit |
| `hand_e` | Robotiq Hand-E parallel | `Slider_1`, `Slider_2` | prismatic | **no drives** — `close()` refuses until `repair_drives()` |
| `egk_25` / `egu_50` / `ezu_35` | Schunk parallel | `Jaw_Drive` | 26.5 / 51 / 35 mm | one mimic follower |

`attach_gripper` references the asset **beside** the arm as `<arm>Gripper`,
puts its base on the flange along the measured tool axis (Z on a UR, X on a
KR210, -Y on a Fanuc CRX) from a standoff measured off the flange link's own
bound, drops the gripper's articulation root, joins it with a fixed joint
that stays *inside* the articulation, masks arm–gripper collisions and zeroes
the joint states — the Robot Assembler's recipe, and each step is there
because the Assembler documents what goes wrong without it. The fitted joints
are recorded on the arm's prim (`simliverse:gripper`), which is how a Hand-E's
`Slider_1` — a name no token matches — still ends up in `arm.gripper`.

Measured live, a 2F-85 on a UR10e (Isaac Sim 6.0.1): the articulation came
back as 12 DOF (6 + 6), `drive_health()` clean, the descent reached to 0.4 mm,
`close()` stopped at 0.73 rad against a 4 cm block with both inner fingers in
its contact list and `is_grasping` true, a 0.20 m lift carried it up 0.196 m,
and `open()` put it back on the table. Two numbers to plan with: the 2F-85's
pads close **about 0.155 m below the flange face**, so a grasp pose puts the
flange that far above the object's centre; and `finger_joint` read 2.20 rad
right after `open()` (its limit is 0.82) before closing normally — read the
grasp off `is_grasping` and the contact list, not off that joint.

**Palletising with a jaw instead of a cup.** `demo.ur10_palletizing.build(gripper="2f_85")`
runs the same pick/place loop on a finger jaw: one `_JawEE`/`_SuctionEE`
adapter behind `pick_waiting_box`/`place_on_slot`, so the cell does not branch
on the tool. Three things a jaw needs that a cup does not, each a real cell
lesson. A jaw grips a box's SIDES, so the datum is the box centre, not its top,
and the reach is measured from a per-asset table (`JAW_GEOMETRY`: a 2F-85 opens
~85 mm and its pads sit 0.155 m below the flange) rather than the joint limits,
which a linkage reports in radians. A jaw can only hold a box **narrower than it
opens** — `_JawEE.fits` refuses a 150 mm carton on an 85 mm jaw up front, so a
finger gripper is for small parcels and a cup is the tool for cartons. And the
pad centre is ~4 mm off the flange's tool axis, so the pick servos on
`gripper.pad_center()` to put the pads — not the flange — over the box before
closing, or a symmetric jaw shoves a light box out of the grasp. KNOWN GAP,
diagnosed not guessed: a top-down jaw pick off the belt mounts, orients, gates,
centres and closes on the parcel with both inner fingers in contact, but the
box does not survive the lift — and it is GEOMETRY, not grip force. A 2F is a
LINKAGE jaw whose fingers arc down-and-in as they close, so a top-down pick of a
box on a belt presses it into the deck (measured: the box's belt-contact force
rose to 132 N as the fingers closed) instead of squeezing its sides, and lifting
then knocks it off. Cranking box and pad friction to 1.4 made it worse, not
better, which is the proof the pinch is not the problem. A jaw palletising end
to end needs a different approach — a true parallel jaw without the down-arc, or
a side/horizontal grip — not more tuning of this one. Feed non-cube parcels with
`build(box=<footprint>, box_h=<height>)`.

The parallel jaw is the right instinct and half-proven: `build(gripper="egu_50")`
mounts the Schunk EGU-50 as a genuine PRISMATIC jaw (`gripper.is_linkage` is
False, two `Jaw_Drive` joints, span 51 mm), which closes straight in with no
down-arc. But its `attach_gripper` corrupts the physics scene on play — the belt
and its boxes launch to z ~ 21000 — where the 2F-85 attaches cleanly and a
0.04 m box on the same cell palletises fine under suction, so it is the Schunk
asset's articulation (its mimic `Jaw_Drive` followers, not zeroed the way the
2F's are), not the box or the cell. Fixing that attach is the next step toward a
jaw that palletises end to end; the prismatic geometry is already the answer to
the down-arc.

Suction: force limits of 500 (Isaac's tutorial value) break the seal within
2 mm of any motion. Use `1.0e6`. **Writing any `isaac:*` attribute on a closed
gripper releases it**, so do not "test" limits mid-grip.

---

## 7. Look at the scene before believing it

```python
from simliverse_sim import vision

vision.look(centre=(0, 0, 0.6), scale=3.0)  # four views, not one
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
