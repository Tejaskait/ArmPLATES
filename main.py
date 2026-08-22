# main.py
# This is the "Layer 1 / Master Orchestrator" from the project doc.
# It reads config.py, spins up one ArmController per entry, and runs
# the shared simulation loop so all arms move together.

import time
import pybullet as p
import pybullet_data

from config import ARMS, ARM_URDF, SIMULATION_TIMESTEP, PLATES
from arm_controller import ArmController
from plate_factory import spawn_all_plates


def setup_world():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")


def build_arms():
    arms = []
    for cfg in ARMS:
        arm = ArmController(
            urdf_path=ARM_URDF,
            base_position=cfg["base_position"],
            name=cfg["name"],
            component=cfg["component"],
            target_position=cfg["target_position"],
        )
        arms.append(arm)
    return arms


def main():
    setup_world()
    arms = build_arms()
    plates = spawn_all_plates(PLATES)

    print(f"Loaded {len(arms)} arms:")
    for arm in arms:
        print("  -", arm.get_status())

    print(f"Loaded {len(plates)} plates:")
    for plate in plates:
        print("  -", plate.get_status())

    step_count = 0
    while True:
        for arm in arms:
            arm.step()

        p.stepSimulation()
        time.sleep(SIMULATION_TIMESTEP)

        step_count += 1
        if step_count % 240 == 0:  # roughly once per second
            for arm in arms:
                print(arm.get_status())
            for plate in plates:
                print(plate.get_status())


if __name__ == "__main__":
    main()