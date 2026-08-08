"""Route B: WheelBasePoseController, waypoint by waypoint, with the modules purged."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcp

purge = """
import importlib, sys
for m in [k for k in sys.modules if k.startswith("simliverse_sim")]:
    del sys.modules[m]
importlib.invalidate_caches()
import simliverse_sim
print("purged; navigation has wheel-speed fix:",
      "wheel_radius * float(np.mean(wheels))" in
      open("/workspace/simliverse-mcp/simliverse_sim/robots/navigation.py").read())
"""
mcp.show(mcp.call("run_control", code=purge))

code = """
import numpy as np

scene = Scene.get()
scene.stop(); scene.play(); scene.step(30)
rover = Robot.attach("/World/Rover")

ROUTE = [[0.50, 0.45], [1.00, 0.80], [1.50, 0.45], [2.00, 0.00]]
driver = rover.pose_driver(max_linear=0.45, max_angular=1.2, position_tol=0.12)

bumped = set()
for leg, goal in enumerate(ROUTE):
    for tick in range(1200):
        if driver.step(goal):
            break
        for wall in ("/World/Wall1", "/World/Wall2"):
            if rover.touching(wall):
                bumped.add(wall)
        scene.step(1)
    p = np.asarray(rover.base_position, dtype=float)[:2]
    print("leg %d -> %s : reached %s  err %.3f" %
          (leg, goal, p.round(3).tolist(),
           float(np.linalg.norm(p - np.asarray(goal)))))

rover.drive(0.0, 0.0)
scene.step(60)
p = np.asarray(rover.base_position, dtype=float)[:2]
print()
print("final:", p.round(3).tolist(), "| goal [2.0, 0.0] | err",
      round(float(np.linalg.norm(p - np.array([2.0, 0.0]))), 3))
print("walls touched:", sorted(bumped) or "none")
for wall in ("/World/Wall1", "/World/Wall2"):
    print(wall, RigidObject(wall, scene=scene).position.round(3).tolist())
"""
r = mcp.call("run_control", code=code)
print(r.get("stdout") or r.get("message") or r)
