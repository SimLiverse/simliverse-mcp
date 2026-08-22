# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Validate that the adapter and handler structure is correct."""

import ast
import os

import pytest

EXTENSION_ROOT = os.path.join(os.path.dirname(__file__), "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension")


def _parse_file(path):
    with open(path) as f:
        return ast.parse(f.read())


def test_adapter_base_has_all_abstract_methods():
    """Verify the base adapter defines all required abstract methods."""
    tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "base.py"))
    methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                    methods.add(node.name)
                elif isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
                    methods.add(node.name)
    expected = {
        "get_stage",
        "get_assets_root_path",
        "discover_environments",
        "load_environment",
        "create_prim",
        "delete_prim",
        "add_reference_to_stage",
        "set_prim_transform",
        "get_prim_transform",
        "list_prims",
        "get_prim_info",
        "create_xform_prim",
        "create_articulation",
        # No discover_robots: robot discovery lives in simliverse_sim, which
        # walks the asset root directly. The adapters had a second copy that
        # answered differently, so the duplicate was removed rather than kept
        # in sync.
        "get_robot_joint_info",
        "set_joint_positions",
        "get_joint_positions",
        "create_world",
        "create_simulation_context",
        "create_physics_scene",
        "create_camera",
        "capture_camera_image",
        "create_lidar",
        "get_lidar_point_cloud",
        "create_pbr_material",
        "create_physics_material",
        "apply_material",
        "create_light",
        "modify_light",
        "clone_prim",
        "import_urdf",
        "play",
        "pause",
        "stop",
        "step",
        "execute_script",
        # Observability methods (issue #1)
        "get_simulation_state",
        "get_physics_state",
        "get_joint_config",
        "reload_script",
        # Dimensional data (issue #2)
        "get_prim_actual_size",
    }
    assert methods == expected, f"Missing: {expected - methods}, Extra: {methods - expected}"


def _abstract_methods():
    """Names every concrete adapter must supply, from base.py."""
    tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "base.py"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Name) and decorator.id == "abstractmethod") or (
                    isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod"
                ):
                    names.add(node.name)
    return names


def _methods_defined_in(*module_names):
    names = set()
    for module in module_names:
        tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", module))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names.add(node.name)
    return names


@pytest.mark.parametrize("version_module", ["v5.py", "v6.py"])
def test_adapter_implements_all_methods(version_module):
    """Every abstract method must be supplied somewhere on the adapter's chain.

    Resolved across `common.py` as well as the version module. An abstract
    method left unimplemented anywhere on that chain is a TypeError the moment
    `get_adapter()` instantiates the class, which happens at extension startup
    and takes the whole extension down with it.

    Checking only the version module — which this test used to do, and only for
    v5 — would now report the eighteen shared implementations as missing while
    still saying nothing at all about v6.
    """
    missing = _abstract_methods() - _methods_defined_in(version_module, "common.py")
    assert not missing, f"{version_module} has no implementation for: {sorted(missing)}"


def test_the_shared_layer_holds_nothing_version_specific():
    """`common.py` must not depend on an Isaac API that moved between 5.1 and 6.0.

    That move — `isaacsim.core.*` to `isaacsim.core.experimental.*` — is the
    whole reason two adapters exist. A method that needs either namespace is
    version-specific by definition and belongs in `v5.py` or `v6.py`, however
    similar its two implementations happen to look today.

    An import inside `try` is exempt, and deliberately: `execute_script` aliases
    the deprecated `omni.isaac` namespace and probes for `isaacsim.core.utils`
    so that user scripts written against either generation keep working. It
    degrades to `None` rather than raising, which is what makes it shared code
    instead of two implementations.
    """
    tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "common.py"))

    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for inner in ast.walk(node):
                guarded.add(id(inner))

    versioned = (
        "isaacsim.core.experimental",
        "isaacsim.core.api",
        "isaacsim.core.prims",
        "isaacsim.core.utils",
        "isaacsim.core.simulation_manager",
    )

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or id(node) in guarded:
            continue
        targets = [node.module or ""] if isinstance(node, ast.ImportFrom) else [
            a.name for a in node.names
        ]
        for target in targets:
            if any(target.startswith(v) for v in versioned):
                offenders.append(f"line {node.lineno}: {target}")

    assert not offenders, (
        f"common.py imports version-specific Isaac modules unguarded: {offenders}. "
        f"Those methods belong back in v5.py / v6.py."
    )


def test_all_handler_modules_have_register():
    """Verify every handler module exposes a register(registry, adapter) function."""
    handlers_dir = os.path.join(EXTENSION_ROOT, "handlers")
    handler_files = [
        "scene.py",
        "objects.py",
        "lighting.py",
        "robots.py",
        "sensors.py",
        "materials.py",
        "assets.py",
        "simulation.py",
    ]
    for filename in handler_files:
        filepath = os.path.join(handlers_dir, filename)
        assert os.path.exists(filepath), f"Handler file missing: {filename}"
        tree = _parse_file(filepath)
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert "register" in func_names, f"{filename} missing register() function"


def _handler_modules_from_init():
    """The handler modules `register_all_handlers` iterates over."""
    tree = _parse_file(os.path.join(EXTENSION_ROOT, "handlers", "__init__.py"))
    for node in ast.walk(tree):
        # `for module in [scene, objects, ...]: module.register(...)`
        if isinstance(node, ast.For) and isinstance(node.iter, ast.List):
            return [e.id for e in node.iter.elts if isinstance(e, ast.Name)]
    raise AssertionError("Could not find the handler list in handlers/__init__.py")


def test_every_handler_module_defines_register():
    """A handler missing `register` takes the whole extension down.

    `register_all_handlers` calls `module.register(...)` unconditionally, so one
    missing function raises AttributeError inside `on_startup` and Kit reports
    "Failed to startup python extension" — no MCP verbs at all, not just the
    ones from that module.

    This is not hypothetical. `register` was deleted from `robots.py` during a
    refactor and nothing caught it, because the already-running Isaac Sim kept
    the previous version of the module in memory. It surfaced only on the next
    container restart, as a total loss of the extension.
    """
    for name in _handler_modules_from_init():
        path = os.path.join(EXTENSION_ROOT, "handlers", f"{name}.py")
        assert os.path.isfile(path), f"handlers/__init__.py imports missing module {name!r}"
        functions = {
            node.name
            for node in ast.walk(_parse_file(path))
            if isinstance(node, ast.FunctionDef)
        }
        assert "register" in functions, (
            f"handlers/{name}.py has no register() — register_all_handlers calls it "
            f"unconditionally, so this breaks the entire extension at startup."
        )


def test_registered_handlers_are_defined():
    """Every function `register` wires into the registry must exist in its module."""
    for name in _handler_modules_from_init():
        tree = _parse_file(os.path.join(EXTENSION_ROOT, "handlers", f"{name}.py"))
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        register = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "register"
        )
        for call in (n for n in ast.walk(register) if isinstance(n, ast.Call)):
            target = call.func
            if isinstance(target, ast.Name) and target.id.islower():
                assert target.id in defined or target.id in dir(__builtins__), (
                    f"handlers/{name}.py register() wires {target.id}(), which is not "
                    f"defined in that module."
                )
