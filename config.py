# config.py
# Central place that defines how many arms exist, where they stand,
# and what "suit component" each one is responsible for.
#
# SCOPE REDUCED: this project now trains and runs exactly ONE arm.
# The other 7 arm slots from the original 8-arm layout have been
# removed entirely (not just disabled) to cut training workload -
# the arm only ever needs to learn to detect and stack ITS 4 plates
# into a pyramid. If you want to scale back up later, add more
# entries back into ARMS following the same dict shape.

ARM_URDF = "franka_panda/panda.urdf"   # Franka Panda: 7-DOF arm + built-in 2-finger gripper
# NOTE: previously used kuka_iiwa/model.urdf, but that model has NO gripper at all -
# just a bare arm ending in nothing. Panda has real fingers we can open/close to grip plates.

ARMS = [
    {
        "name": "arm_1",
        "component": "pyramid_builder",
        "base_position": [0.5, 0.0, 0.0],
        "target_position": [0.0, 0.0, 0.3],  # build point, directly in front of the arm base
    },
]

SIMULATION_TIMESTEP = 1.0 / 240.0

# ---------------------------------------------------------------------
# PLATES: 4 rectangular boards, largest to smallest, used for the
# "pyramid stacking" pick-and-place task. The single arm learns to
# grab these and stack them biggest-on-bottom, smallest-on-top.
#
# Reduced from the original 8 down to 4 - fewer classes for the CNN
# classifier to learn, fewer objects for one arm to have to cycle
# through, smaller dataset to capture/train on.
#
# Each entry:
#   half_extents = [x, y, z] in meters (PyBullet boxes are defined by
#                   HALF the full width/depth/height)
#   color         = RGBA, just so they're visually distinguishable
#   spawn_position = where it starts, scattered on the ground for now
# ---------------------------------------------------------------------

PLATE_THICKNESS = 0.02  # full plate is 4cm thick

# 4 plates, spread out around the single arm's base so all are
# reachable and none overlap on spawn. Sizes step down evenly from
# largest (plate_1) to smallest (plate_4) so the pyramid has a clear
# base-to-tip taper.

PLATES = [
    {"name": "plate_1", "half_extents": [0.117, 0.117, PLATE_THICKNESS], "color": [0.8, 0.1, 0.1, 1], "spawn_position": [0.32, 0.0, 0.03]},
    {"name": "plate_2", "half_extents": [0.103, 0.103, PLATE_THICKNESS], "color": [0.9, 0.4, 0.1, 1], "spawn_position": [0.226, 0.226, 0.03]},
    {"name": "plate_3", "half_extents": [0.09, 0.09, PLATE_THICKNESS], "color": [0.9, 0.7, 0.1, 1], "spawn_position": [0.0, 0.32, 0.03]},
    {"name": "plate_4", "half_extents": [0.077, 0.077, PLATE_THICKNESS], "color": [0.6, 0.8, 0.1, 1], "spawn_position": [-0.226, 0.226, 0.03]},
]
# NOTE: the "spawn_position" values above are only a FALLBACK, used when
# randomize_plate_positions=False. By default rl_env.py now ignores them
# and instead samples a fresh random position per plate every episode -
# see WORKSPACE_* below - specifically so the arm learns to visually
# search for plates wherever they actually are, not memorize 4 fixed spots.

# Reachable workspace: an annulus (ring) around the arm's base, in meters.
# Panda's practical reach is ~0.8m; MAX_RADIUS is kept comfortably inside
# that so sampled positions are actually reachable, and MIN_RADIUS keeps
# plates from spawning right on top of the arm's own base.
WORKSPACE_MIN_RADIUS = 0.18
WORKSPACE_MAX_RADIUS = 0.55
WORKSPACE_MIN_CLEARANCE = 0.12  # minimum center-to-center distance kept between any 2 plates when randomizing, so they don't spawn overlapping/stacked

# ---------------------------------------------------------------------
# PYRAMID TARGET POSES: where each plate should end up, stacked largest
# on the bottom to smallest on top, centered at the build point.
# z is computed by stacking each plate directly on top of the previous
# one's top surface (running sum of full thicknesses).
# ---------------------------------------------------------------------

def _compute_pyramid_targets():
    targets = []
    z = 0.0
    for plate in PLATES:
        thickness = plate["half_extents"][2] * 2
        z += thickness / 2
        targets.append({"plate": plate["name"], "position": [0.0, 0.0, round(z, 4)]})
        z += thickness / 2
    return targets

PYRAMID_TARGETS = _compute_pyramid_targets()