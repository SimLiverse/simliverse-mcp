"""No handler may advertise a parameter it never reads.

This is the single most expensive bug shape in the project, twice over.

`capture_view` took `position`/`look_at` from a client that sent `eye`/
`target`; the names were dropped, plausible pictures came back from the
default camera, and a whole session was spent reasoning about a scene
nobody was looking at. Then `capture_view` itself took a `camera_path`
alongside an aim and dropped the aim.

The same shape was sitting in three more handlers: `set_physics` accepted
`time_step` and `gpu_enabled` and set neither, so an agent asking for
1/240 s got 1/60 and a "success"; `objects.create` accepted `color` and
made grey cubes; `search_usd` accepted `position` and `scale` and left
everything at the origin.

What makes it costly is not the wrong behaviour -- it is that every
subsequent measurement agrees with the wrong behaviour. There is no
error, no anomaly, and nothing to notice. So the rule is mechanical: a
parameter in the signature is a promise, and it must appear in the body.

Kept as source analysis: these modules import Isaac at call time and
cannot be imported here.
"""

import ast
import os

HANDLERS = os.path.join(
    os.path.dirname(__file__), "..", "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension", "handlers",
)

#: Passed by the dispatcher, not by the caller.
PLUMBING = {"self", "adapter", "context", "kwargs", "args"}


def test_every_parameter_a_handler_advertises_is_read():
    dropped = []
    for name in sorted(os.listdir(HANDLERS)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(HANDLERS, name)) as f:
            tree = ast.parse(f.read())
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
                continue
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} - PLUMBING
            read = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            for missing in sorted(params - read):
                dropped.append(f"{name}:{fn.lineno} {fn.name}() never reads {missing!r}")
    assert not dropped, (
        "these handlers promise a parameter and ignore it, which is invisible "
        "to the caller:\n  " + "\n  ".join(dropped) + "\n"
        "Either honour it, or reject the call by name -- do not accept it "
        "and carry on."
    )
