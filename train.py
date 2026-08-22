# train.py
# Trains the single arm's policy to reach for its 4 plates in turn.
# (Previously this trained one SHARED policy across 8 arms by randomizing
# arm_index each episode. With only 1 arm left in config.ARMS that
# randomization is a no-op, so it's been dropped - every episode now
# just uses arm_index=0.)

import os
# Windows-only fix: PyTorch ships its own OpenMP runtime (libiomp5md.dll)
# and numpy/pybullet can pull in a second one (libomp.dll) - having both
# load in one process trips OMP Error #15 and aborts. This must be set
# BEFORE torch/numpy/pybullet import, so it's the very first line here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_env import ArmReachEnv


def main(total_timesteps=300_000):
    # --- env settings ---
    # max_steps: 150 -> 250. Plates now spawn at a random spot anywhere in
    #   the workspace annulus each episode (see config.WORKSPACE_*), not a
    #   handful of fixed nearby points - that's a bigger area to visually
    #   search, so the arm needs more room per episode to find and reach it.
    # camera_width/height: kept at 64 - still the full camera -> OpenCV ->
    #   CNN -> depth-unprojection pipeline every step, just cheaper to render.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = ArmReachEnv(arm_index=0, render=False, randomize_arm=False,
                       randomize_plate_positions=True,
                       max_steps=250, camera_width=64, camera_height=64)

    # Sanity-check the environment matches the Gym API correctly
    # before spending any time training on it.
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,  # smaller rollout buffer -> logs/updates happen sooner (default 2048 was
                      # taking a long time to print anything, given how heavy each step is:
                      # real physics step + camera render + OpenCV + CNN forward pass).
                      # Purely a feedback-cadence knob - does not change what the arm sees or
                      # how detection works, still the full camera -> OpenCV -> CNN -> depth
                      # unprojection pipeline every single step.
        device=device,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
    )

    # This is a real, longer run now (random-position search is a harder
    # task than the old fixed-spot version) - checkpoint periodically so a
    # crash/interrupt partway through doesn't lose all the progress. At
    # ~10 fps this is roughly every 15-20 min of training.
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, save_path="./models/checkpoints/",
        name_prefix="arm_1_reach_policy",
    )

    print(f"Training the single-arm policy on RANDOM plate positions for "
          f"{total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    os.makedirs("models", exist_ok=True)
    model_path = "models/arm_1_reach_policy"
    model.save(model_path)
    print(f"Saved trained policy to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()