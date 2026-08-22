# evaluate.py
# Loads the single arm's trained policy and runs it with the GUI open so
# you can watch it reach for each of its 4 plates in turn. Also
# demonstrates how to pull a real camera image (fixes the blank
# "Synthetic Camera" panels - those are blank until getCameraImage() is
# actually called).

import os
# Windows-only fix: see train.py - avoids the libiomp5md.dll / libomp.dll
# double-init crash (OMP Error #15). Must be set before torch/pybullet import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import pybullet as p
from stable_baselines3 import PPO

from rl_env import ArmReachEnv
from config import PLATES


def get_camera_image(width=320, height=240):
    """Pulls one RGB + depth + segmentation frame from a camera looking
    at the workspace center. This is the 'or any other way' object
    detection path - later this feed can go into a YOLO model instead
    of using ground-truth plate positions."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.8, 0.8, 0.8],
        cameraTargetPosition=[0, 0, 0.1],
        cameraUpVector=[0, 0, 1],
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=3.0
    )
    img = p.getCameraImage(
        width, height, view_matrix, proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    return img  # (width, height, rgba, depth, segmentation)


def main(model_path="models/arm_1_reach_policy", episodes_per_plate=2, print_every=10):
    # IMPORTANT: these must match what train.py used, or you're evaluating
    # a policy under different conditions than it was trained on.
    env = ArmReachEnv(arm_index=0, render=True, max_steps=250,
                       randomize_plate_positions=True,
                       camera_width=64, camera_height=64)
    model = PPO.load(model_path)

    for plate_index in range(len(PLATES)):
        for ep in range(episodes_per_plate):
            obs, _ = env.reset(options={"target_plate_index": plate_index})
            done = False
            total_reward = 0.0
            info = {"distance": None, "success": False}
            step_num = 0
            frames_visible = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                step_num += 1

                visible = bool(obs[13] > 0.5)
                if visible:
                    frames_visible += 1
                if step_num % print_every == 0:
                    tx, ty, tz = obs[10], obs[11], obs[12]
                    print(f"  step {step_num:>3} | plate visible: {visible} | "
                          f"detected pos: ({tx:.3f}, {ty:.3f}, {tz:.3f}) | "
                          f"dist to eef: {info['distance']:.3f} | "
                          f"facing plate (1.0=dead-on): {info['alignment']:.2f}")

                get_camera_image()  # keeps the camera panels populated while watching
                time.sleep(1 / 60)  # smooth playback - remove/lower this if it feels too slow

            visibility_pct = 100 * frames_visible / step_num if step_num else 0
            print(f"{PLATES[plate_index]['name']} episode {ep+1}: reward={total_reward:.2f}, "
                  f"final_distance={info['distance']:.3f}, success={info['success']}, "
                  f"plate was visible {visibility_pct:.0f}% of steps")

    env.close()


if __name__ == "__main__":
    main()