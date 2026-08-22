# plate_factory.py
# Spawns the 8 rectangular "plates" as real physics objects in PyBullet.
# These are plain boxes (no URDF file needed) built directly with
# createCollisionShape / createVisualShape / createMultiBody.
#
# Kept separate from arm_controller.py on purpose: arms and plates are
# different kinds of things (actuated vs. passive objects), so they get
# their own file. main.py wires them together.

import pybullet as p


class Plate:
    """One pyramid plate: a physics box the arms can detect/grip/move."""

    def __init__(self, name, half_extents, color, spawn_position, mass=0.3):
        self.name = name
        self.half_extents = half_extents
        self.picked = False  # will be used later once an arm grips it

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
        )

        self.body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=spawn_position,
        )

    def get_position(self):
        pos, _orn = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_status(self):
        x, y, z = self.get_position()
        return f"{self.name} size={self.half_extents} pos=({x:.2f},{y:.2f},{z:.2f}) picked={self.picked}"


def spawn_all_plates(plate_configs, positions=None):
    """plate_configs = the PLATES list from config.py.
    positions: optional list of [x,y,z], same length as plate_configs -
    when given, overrides each cfg's static "spawn_position" for this
    call. This is what lets rl_env.py randomize where plates spawn each
    episode instead of always using the same fixed spot."""
    plates = []
    for i, cfg in enumerate(plate_configs):
        spawn_position = positions[i] if positions is not None else cfg["spawn_position"]
        plate = Plate(
            name=cfg["name"],
            half_extents=cfg["half_extents"],
            color=cfg["color"],
            spawn_position=spawn_position,
        )
        plates.append(plate)
    return plates