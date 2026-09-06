"""step_simulation refuses to step a playing timeline or an over-long burst.

The handler module is loaded from its file: importing the extension package
pulls in `carb`, which only exists inside Kit.
"""

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension")


def _load():
    root = types.ModuleType("isaac_sim_mcp_extension")
    root.__path__ = [PKG]
    sys.modules["isaac_sim_mcp_extension"] = root
    adapters = types.ModuleType("isaac_sim_mcp_extension.adapters")
    adapters.__path__ = [os.path.join(PKG, "adapters")]
    sys.modules["isaac_sim_mcp_extension.adapters"] = adapters
    spec = importlib.util.spec_from_file_location(
        "isaac_sim_mcp_extension.adapters.base", os.path.join(PKG, "adapters", "base.py")
    )
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    handlers = types.ModuleType("isaac_sim_mcp_extension.handlers")
    handlers.__path__ = [os.path.join(PKG, "handlers")]
    sys.modules["isaac_sim_mcp_extension.handlers"] = handlers
    spec = importlib.util.spec_from_file_location(
        "isaac_sim_mcp_extension.handlers.simulation", os.path.join(PKG, "handlers", "simulation.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Adapter:
    def __init__(self):
        self.calls = []

    def step(self, **kw):
        self.calls.append(kw)
        return {"observations": {}}


def _timeline(playing):
    mod = types.ModuleType("omni.timeline")
    mod.get_timeline_interface = lambda: types.SimpleNamespace(is_playing=lambda: playing)
    omni = sys.modules.get("omni") or types.ModuleType("omni")
    omni.timeline = mod
    sys.modules["omni"] = omni
    sys.modules["omni.timeline"] = mod


def test_a_playing_timeline_is_not_stepped():
    simulation = _load()
    _timeline(True)
    adapter = _Adapter()
    out = simulation.step(adapter, num_steps=7200)
    assert out["status"] == "error" and "already playing" in out["message"]
    assert adapter.calls == []


def test_a_paused_timeline_steps_but_not_more_than_the_cap():
    simulation = _load()
    _timeline(False)
    adapter = _Adapter()
    assert simulation.step(adapter, num_steps=7200)["status"] == "error"
    assert adapter.calls == []
    assert simulation.step(adapter, num_steps=120)["status"] == "success"
    assert adapter.calls[0]["num_steps"] == 120
