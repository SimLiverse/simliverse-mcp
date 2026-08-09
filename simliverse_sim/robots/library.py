"""
Robot catalogue and the spawn factory.

`spawn_robot` loads an asset and returns the handle whose control surface matches
the body — a `Manipulator` for an arm, a `WheeledRobot` for a rover, a `Humanoid`
for a humanoid. The class is chosen from the articulation's actual joint
structure after loading, so a robot that is not in the catalogue below still gets
the right controller.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from .._compat import add_reference, as_vec3, assets_root, get_stage
from .base import Morphology, Robot

logger = logging.getLogger("simliverse_sim.robots.library")


@dataclass(frozen=True)
class RobotAsset:
    key: str
    asset_path: str
    morphology: Morphology
    # RMPflow / Lula configuration name, where Isaac ships one. Without it,
    # Cartesian control is unavailable and joint control is the fallback.
    motion_config: str | None = None
    description: str = ""
    manufacturer: str = ""


# Asset paths are the 6.0 layout, where NVIDIA regrouped every robot under a
# vendor directory: Robots/Franka/franka.usd became
# Robots/FrankaRobotics/FrankaPanda/franka.usd. Nine of these seventeen entries
# still pointed at the old flat layout and 404'd — including the Franka, which
# is the robot every manipulation task reaches for first. Verified against the
# live asset server, not guessed.
CATALOGUE: dict[str, RobotAsset] = {
    # ── Manipulators ──────────────────────────────────────────────────────────
    "franka": RobotAsset(
        "franka", "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", Morphology.MANIPULATOR,
        "Franka", "7-DOF arm with a parallel gripper.",
    ),
    "fr3": RobotAsset(
        "fr3", "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd", Morphology.MANIPULATOR,
        "FR3", "Franka Research 3 — 7-DOF arm with a parallel gripper.",
    ),
    "ur10": RobotAsset(
        "ur10", "/Isaac/Robots/UniversalRobots/ur10/ur10.usd", Morphology.MANIPULATOR,
        "UR10", "6-DOF arm, no gripper by default.",
    ),
    "ur5": RobotAsset(
        "ur5", "/Isaac/Robots/UniversalRobots/ur5/ur5.usd", Morphology.MANIPULATOR,
        "UR5", "6-DOF arm, no gripper by default.",
    ),
    # The key said iiwa, the asset is a KR210, and the motion config named a
    # third robot — `Kuka_iiwa7`, which Isaac does not ship. The result was an
    # arm that spawned cleanly, classified as a Manipulator, advertised
    # `move_ee_to`, and then failed on the first Cartesian call with "No RMPflow
    # configuration matches". `Kuka_KR210` is supported; the asset was right all
    # along and only the config name was wrong.
    "kuka_kr210": RobotAsset(
        "kuka_kr210", "/Isaac/Robots/Kuka/KR210_L150/kr210_l150.usd", Morphology.MANIPULATOR,
        "Kuka_KR210", "6-DOF industrial arm, 150 kg payload. No gripper by default.",
    ),
    # No motion config: Lula ships 21 and `Kinova_Gen3` is not one of them. It
    # was seeded here anyway, so `list_robots()` advertised Cartesian control
    # this arm does not have and `move_ee_to` failed at runtime on a robot an
    # agent picked *because* the catalogue said it could reach.
    "kinova_gen3": RobotAsset(
        "kinova_gen3", "/Isaac/Robots/Kinova/Gen3/gen3n7_instanceable.usd", Morphology.MANIPULATOR,
        None, "7-DOF arm. Joint control only — no Cartesian motion config ships for it.",
    ),
    # ── Dexterous hands ───────────────────────────────────────────────────────
    "allegro_hand": RobotAsset(
        "allegro_hand", "/Isaac/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd",
        Morphology.DEXTEROUS_HAND, None, "16-DOF four-finger hand.",
    ),
    "shadow_hand": RobotAsset(
        "shadow_hand", "/Isaac/Robots/ShadowRobot/ShadowHand/shadow_hand_instanceable.usd",
        Morphology.DEXTEROUS_HAND, None, "24-DOF five-finger hand.",
    ),
    # ── Wheeled ───────────────────────────────────────────────────────────────
    "carter": RobotAsset(
        "carter", "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd", Morphology.WHEELED,
        None, "Differential-drive research AMR.",
    ),
    "jetbot": RobotAsset(
        "jetbot", "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd", Morphology.WHEELED,
        None, "Small two-wheel differential-drive robot.",
    ),
    "kaya": RobotAsset(
        "kaya", "/Isaac/Robots/NVIDIA/Kaya/kaya.usd", Morphology.WHEELED,
        None, "Three-wheel holonomic robot.",
    ),
    # ── Quadrupeds ────────────────────────────────────────────────────────────
    "anymal_c": RobotAsset(
        "anymal_c", "/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd", Morphology.QUADRUPED,
        None, "12-DOF quadruped. Locomotion needs a trained policy.",
    ),
    "unitree_go2": RobotAsset(
        "unitree_go2", "/Isaac/Robots/Unitree/Go2/go2.usd", Morphology.QUADRUPED,
        None, "12-DOF quadruped. Locomotion needs a trained policy.",
    ),
    "spot": RobotAsset(
        "spot", "/Isaac/Robots/BostonDynamics/spot/spot.usd", Morphology.QUADRUPED,
        None, "12-DOF quadruped. Locomotion needs a trained policy.",
    ),
    # ── Humanoids ─────────────────────────────────────────────────────────────
    "unitree_h1": RobotAsset(
        "unitree_h1", "/Isaac/Robots/Unitree/H1/h1.usd", Morphology.HUMANOID,
        None, "Full-size humanoid. Locomotion needs a trained policy.",
    ),
    "unitree_g1": RobotAsset(
        "unitree_g1", "/Isaac/Robots/Unitree/G1/g1.usd", Morphology.HUMANOID,
        None, "Compact humanoid. Locomotion needs a trained policy.",
    ),
    # ── Aerial ────────────────────────────────────────────────────────────────
    "quadcopter": RobotAsset(
        "quadcopter", "/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd", Morphology.AERIAL,
        None, "Small quadcopter, thrust-controlled.",
    ),
}


_CONTROLLERS: dict[Morphology, str] = {
    Morphology.MANIPULATOR: "Manipulator",
    Morphology.DEXTEROUS_HAND: "DexterousHand",
    Morphology.WHEELED: "WheeledRobot",
    Morphology.MOBILE_MANIPULATOR: "MobileManipulator",
    Morphology.QUADRUPED: "LeggedRobot",
    Morphology.HUMANOID: "Humanoid",
    Morphology.AERIAL: "AerialRobot",
}


def _controller_class(morphology: Morphology) -> type[Robot]:
    from . import aerial, legged, manipulator, mobile

    return {
        Morphology.MANIPULATOR: manipulator.Manipulator,
        Morphology.DEXTEROUS_HAND: manipulator.DexterousHand,
        Morphology.WHEELED: mobile.WheeledRobot,
        Morphology.MOBILE_MANIPULATOR: mobile.MobileManipulator,
        Morphology.QUADRUPED: legged.LeggedRobot,
        Morphology.HUMANOID: legged.Humanoid,
        Morphology.AERIAL: aerial.AerialRobot,
    }.get(morphology, Robot)


# ── Discovery ─────────────────────────────────────────────────────────────────
#
# The catalogue above is a seed, not the source of truth. Isaac Sim 6.0 regrouped
# every robot under a vendor directory and nine of seventeen hardcoded paths
# silently began returning 404 — including the Franka, which is the first robot
# any manipulation task reaches for. A hardcoded map cannot notice that; the
# asset server can be asked.
#
# Discovery also finds robots nobody listed. The 6.0 server ships vendors the
# catalogue never mentioned (Agility, Fourier, Galbot, Booster, 1X), so walking
# it surfaces humanoids that were simply invisible before.

# Directories that are not robots, and files that are parts rather than a robot.
_SKIP_DIRS = {"props", "materials", "detailedprops", "configuration", "instanceable_meshes"}
_SKIP_FILE_TOKENS = ("_part", "_mesh", "instanceable_meshes", "_props")

# Joint-name tokens that identify which RMPflow config an asset needs, where the
# asset's own name does not match the config name.
_MOTION_ALIASES = {"panda": "Franka", "fr3": "FR3"}

_DISCOVERED: dict[str, RobotAsset] | None = None


def _preferred_usd(files: list[str], model: str) -> str | None:
    """Pick the robot file when a model directory holds several.

    Prefers an exact model-name match, then the shortest name: `franka.usd` over
    `franka_alt_finger.usd`, and never a `*_part.usd` fragment.
    """
    usable = [
        f for f in files
        if f.endswith(".usd") and not any(t in f.lower() for t in _SKIP_FILE_TOKENS)
    ]
    if not usable:
        return None
    stem = model.lower().replace("_", "")
    exact = [f for f in usable if f.lower().replace("_", "").removesuffix(".usd") == stem]
    return sorted(exact or usable, key=len)[0]


def _supported_motion_configs() -> list[str]:
    """The RMPflow configurations Lula actually ships, or [] if it cannot be asked."""
    try:
        from .._compat import motion_generation

        pairs = motion_generation().interface_config_loader.get_supported_robot_policy_pairs()
        return list(pairs)
    except Exception:  # noqa: BLE001 — no Lula is a degraded mode, not an error
        logger.debug("Could not list RMPflow configurations", exc_info=True)
        return []


_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def _tokens(*parts: str) -> list[str]:
    """Split names the way a model number is actually written.

    On separators, on camelCase, and on letter/digit boundaries — so `KR210_L150`
    is {kr, 210, l, 150} and `UR3` is {ur, 3}. That last split is what keeps a
    UR30 from answering to the UR3 configuration: as whole strings "ur3" is a
    prefix of "ur30", but as tokens {ur, 3} is not a subset of {ur, 30}.
    """
    out: list[str] = []
    for part in parts:
        for chunk in re.split(r"[^0-9A-Za-z]+", part):
            for piece in _CAMEL.split(chunk):
                if piece:
                    out.append(piece.lower())
    return out


def _infer_motion_config(vendor: str, model: str, supported: list[str]) -> str | None:
    """Which RMPflow config, if any, belongs to `vendor/model`.

    The only thing the seeded catalogue still contributed over discovery was this
    name — and a hand-written table is exactly what went wrong with the Kuka,
    where the seed claimed `Kuka_iiwa7` for an asset that is a KR210 and a config
    Isaac does not ship. Lula publishes its own list; matching against that
    cannot name a config that does not exist.

    Every token of the config name must appear in the asset's vendor+model
    tokens **exactly**. Substring matching was tried and is not safe here: it
    gave `UR30` the UR3 configuration, which is a different arm with different
    link lengths, and would have surfaced as an arm that misses everything it
    reaches for rather than as a configuration error.

    Where several configs qualify, the more model-specific one wins — measured by
    how many of its matched tokens are *not* also in the vendor name. `FrankaFR3`
    matches both `Franka` and `FR3`, and "franka" is a brand token the vendor
    directory already carries, so FR3 is the specific one. A genuine tie returns
    None: no Cartesian control and a legible error beats a confident guess at
    which arm this is.
    """
    brand = set(_tokens(vendor))
    asset = set(_tokens(vendor, model))
    if not asset:
        return None

    ranked: list[tuple[tuple[int, int, int], str]] = []
    for config in supported:
        wanted = _tokens(config)
        if wanted and all(token in asset for token in wanted):
            specific = sum(1 for token in wanted if token not in brand)
            ranked.append(((specific, len(wanted), len(config)), config))
    if not ranked:
        return None

    ranked.sort(reverse=True)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        logger.info(
            "%s/%s matches several RMPflow configs (%s) equally well; leaving it "
            "without one rather than guessing.",
            vendor, model, ", ".join(c for _, c in ranked),
        )
        return None
    return ranked[0][1]


def discover_robots(refresh: bool = False) -> dict[str, RobotAsset]:
    """Walk the Isaac asset server and build the catalogue from what is there.

    Cached, because the walk is a few seconds of network listing. Falls back to
    the seeded `CATALOGUE` if the server is unreachable — an offline session
    should degrade to the known-good subset rather than report no robots at all.
    """
    global _DISCOVERED
    if _DISCOVERED is not None and not refresh:
        return _DISCOVERED

    try:
        import omni.client

        root = assets_root()
    except Exception as exc:
        logger.warning("Asset server unavailable (%s); using the seeded catalogue", exc)
        return dict(CATALOGUE)

    def ls(path: str) -> list[str]:
        result, entries = omni.client.list(root + path)
        if str(result) != "Result.OK":
            return []
        return sorted(e.relative_path for e in entries)

    found: dict[str, RobotAsset] = {}
    # Asked once: it loads Lula, and the walk visits a couple of hundred models.
    supported = _supported_motion_configs()
    base = "/Isaac/Robots"
    for vendor in ls(base):
        if vendor.startswith(".") or vendor.lower() in _SKIP_DIRS:
            continue
        for model in ls(f"{base}/{vendor}"):
            if model.startswith(".") or model.lower() in _SKIP_DIRS:
                continue
            model_dir = f"{base}/{vendor}/{model}"
            chosen = _preferred_usd(ls(model_dir), model)
            if chosen is None:
                continue
            key = model.lower().replace("-", "_").replace(" ", "_")
            found[key] = RobotAsset(
                key=key,
                asset_path=f"{model_dir}/{chosen}",
                # Morphology is classified from the articulation after loading —
                # a directory listing cannot know, and does not need to.
                morphology=Morphology.UNKNOWN,
                # Derived, not seeded — see `_infer_motion_config`. This is what
                # gives Cartesian control to arms nobody wrote an entry for.
                motion_config=_infer_motion_config(vendor, model, supported),
                description=f"{vendor} {model}",
                manufacturer=vendor,
            )

    if not found:
        logger.warning("Asset walk returned nothing; using the seeded catalogue")
        return dict(CATALOGUE)

    logger.info(
        "Matched RMPflow configs for %d of %d discovered robots",
        sum(1 for a in found.values() if a.motion_config), len(found),
    )

    # Seeded entries win on key collision: they carry a hand-checked morphology,
    # a motion config, and a written description that discovery cannot infer.
    merged = {**found, **CATALOGUE}

    # ...but "hand-checked" is a claim, not a guarantee, and this is where a
    # typo becomes a promise. `kinova_gen3` was seeded with the config
    # `Kinova_Gen3`, which Lula does not ship. `list_robots()` therefore
    # advertised Cartesian control for an arm that has none, and `move_ee_to`
    # failed at runtime on a robot an agent had picked *because* the catalogue
    # said it could reach.
    #
    # Inferred configs cannot be wrong this way — they are chosen from the
    # supported list. Only seeded ones can, so they are the ones checked.
    if supported:
        for key, asset in list(merged.items()):
            if asset.motion_config and asset.motion_config not in supported:
                logger.warning(
                    "%s claims motion config %r, which this Isaac Sim does not "
                    "ship (it has %d). Dropping the claim: the robot stays, but "
                    "it is reported as joint-control only rather than failing "
                    "later inside move_ee_to.",
                    key, asset.motion_config, len(supported),
                )
                merged[key] = replace(asset, motion_config=None)
    logger.info(
        "Robot catalogue: %d discovered, %d seeded, %d total",
        len(found), len(CATALOGUE), len(merged),
    )
    _DISCOVERED = merged
    return merged


def refresh_catalogue() -> int:
    """Re-walk the asset server. Returns the number of robots now known."""
    return len(discover_robots(refresh=True))


def resolve(robot_type: str) -> RobotAsset:
    """Look up a catalogue entry, tolerating partial names."""
    catalogue = discover_robots()
    key = robot_type.strip().lower().replace("-", "_").replace(" ", "_")
    if key in catalogue:
        return catalogue[key]
    matches = [k for k in catalogue if key in k or k in key]
    if matches:
        return catalogue[sorted(matches, key=len)[0]]
    raise ValueError(
        f"Unknown robot {robot_type!r}. Known robots: {', '.join(sorted(catalogue))}. "
        f"For anything else, load the USD yourself and call Robot.attach(prim_path)."
    )


def specialize(probe: Robot, **kwargs: Any) -> Robot:
    """Re-wrap a generic `Robot` as the subclass matching its actual structure."""
    from .base import classify_morphology

    morphology = classify_morphology(
        probe.joint_names, probe.groups, [str(l) for l in probe.links()]
    )
    controller = _controller_class(morphology)
    if controller is Robot:
        logger.info(
            "No specialised controller for %s (morphology=%s); joint-level control only.",
            probe.prim_path,
            morphology.value,
        )
        return probe
    return controller(probe.prim_path, scene=probe.scene, **kwargs)


def _register_articulation(scene: Any, prim_path: str) -> None:
    """Make PhysX parse a robot that has just been added to the stage.

    PhysX builds articulation metadata *when the timeline starts*, and only
    then. Until that has happened the prim looks perfectly healthy in USD while
    every handle built from it dies on

        AttributeError: 'NoneType' object has no attribute 'link_names'

    raised several frames inside isaacsim.core, naming nothing the caller did.
    Rigid bodies are parsed as they are added and have no such problem, which is
    why this bites only robots and reads as the library being broken.

    Both ways of getting there need fixing, because between them they cover
    every order a caller can write the code in:

      * **Timeline playing.** The robot arrived after the parse and was never
        seen. A stop/play cycle is the only repair, and it resets dynamic bodies
        to their spawn poses — hence the warning, and hence robots belonging at
        the start of a scene rather than in the middle of a task.
      * **Timeline stopped.** Nothing has been parsed yet at all. This is the
        *natural* order — configure, spawn, then play — so it has to work:
        `Robot.spawn(...)` followed by `scene.play()` is what anyone writes
        first, and it used to fail on the spawn, before the play it was about
        to do anyway. The timeline is started here and left playing; the
        caller's own `play()` is then a no-op.

    An agent that has to discover any of this empirically spends its budget
    doing so — one measured run burned six turns and a failed subagent on it.

    The cycle is repeated until the articulation actually resolves, rather than
    performed once and assumed to have worked. A robot arrives on the stage as
    a *reference*, and USD composes it asynchronously: cycle too early and PhysX
    parses a prim whose links do not exist yet, which fails exactly like never
    having cycled at all. One measured attempt did the right thing one frame too
    soon and reported the library as broken.
    """
    from .._compat import get_timeline, single_articulation, update_app

    def registered() -> bool:
        try:
            single_articulation(prim_path)
            return True
        except Exception:  # noqa: BLE001 — that is the question being asked
            return False

    try:
        playing = get_timeline().is_playing()
    except Exception:  # noqa: BLE001 — no timeline is not a reason to fail a spawn
        logger.debug("Could not read timeline state", exc_info=True)
        return

    if playing:
        logger.warning(
            "Spawned %s while the simulation was playing. PhysX only parses "
            "articulations at play time, so the timeline is being cycled to "
            "register it — dynamic objects will reset to their spawn poses.",
            prim_path,
        )

    for attempt in range(3):
        scene.stop()
        # Let USD finish composing the reference before PhysX looks at it.
        for _ in range(5):
            update_app()
        scene.play()
        scene.step(3)
        if registered():
            return
        logger.debug("Articulation %s not registered on attempt %d", prim_path, attempt + 1)

    logger.warning(
        "PhysX still has no articulation for %s after three timeline cycles. "
        "The asset may not have finished loading; building a handle now will "
        "fail with a link_names error.",
        prim_path,
    )


def spawn_robot(
    robot_type: str,
    *,
    prim_path: str | None = None,
    position: Any = (0.0, 0.0, 0.0),
    scene: Any = None,
    **kwargs: Any,
) -> Robot:
    """Load a robot and return a handle with the right control surface."""
    from ..scene import Scene as _Scene
    from .base import classify_morphology

    asset = resolve(robot_type)
    prim_path = prim_path or f"/World/{asset.key}"
    scene = scene or _Scene.get()

    add_reference(assets_root() + asset.asset_path, prim_path)

    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(get_stage().GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*as_vec3(position, name="position")))

    _register_articulation(scene, prim_path)

    if asset.morphology is Morphology.AERIAL:
        from .aerial import AerialRobot

        return AerialRobot(prim_path, scene=scene, **kwargs)

    # Load first, then classify from the real joint set — the catalogue's
    # morphology is a hint, but the articulation is the ground truth.
    probe = Robot(prim_path, scene=scene)
    morphology = classify_morphology(
        probe.joint_names, probe.groups, [str(l) for l in probe.links()]
    )
    controller = _controller_class(morphology)

    if controller is Robot:
        return probe
    if morphology in (Morphology.MANIPULATOR, Morphology.DEXTEROUS_HAND):
        kwargs.setdefault("rmp_config", asset.motion_config)
        # Record it on the prim as well. `spawn` knows the catalogue and
        # `attach` does not, so Cartesian control was being lost the moment a
        # robot was picked up again — and every controller picks its robot up
        # again with `attach` at INIT. A KR210 spawned with a working config
        # reported "No RMPflow configuration matches" one call later.
        if asset.motion_config:
            try:
                from pxr import Sdf

                prim = get_stage().GetPrimAtPath(prim_path)
                attr = prim.CreateAttribute("simliverse:motion_config", Sdf.ValueTypeNames.String)
                attr.Set(str(asset.motion_config))
            except Exception:  # noqa: BLE001 — a hint that cannot be stored is not fatal
                logger.debug("Could not record the motion config on %s", prim_path,
                             exc_info=True)
        if morphology is Morphology.DEXTEROUS_HAND:
            kwargs.pop("rmp_config", None)
    return controller(prim_path, scene=scene, **kwargs)


def list_robots() -> list[dict[str, str]]:
    """The catalogue, for an agent to choose from."""
    return [
        {
            "key": asset.key,
            "morphology": asset.morphology.value,
            "cartesian_control": "yes" if asset.motion_config else "no",
            "description": asset.description,
        }
        for asset in sorted(
            discover_robots().values(), key=lambda a: (a.morphology.value, a.key)
        )
    ]
