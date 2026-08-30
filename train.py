# train.py
# Trains the single arm's policy to reach for its 4 plates in turn,
# using CURRICULUM LEARNING (see config.CURRICULUM_STAGES): trains on an
# easy small-radius workspace first, then widens it in stages, instead
# of the full random-position workspace from a cold start. This was a
# direct response to an observed failure mode: training on the full
# workspace directly produced a policy that could FIND plates (high
# visibility %) but never closed the final ~0.7-1.1m down to the 0.05m
# success threshold - "search a wide area" and "arrive with precision"
# were being learned simultaneously, and precision lost out.
#
# HOW TO ADVANCE STAGES: this script trains ONE stage per run. Change
# CURRENT_STAGE_INDEX below and re-run once evaluate.py looks solid for
# the current stage (rising success rate, distances actually reaching
# under ~0.05-0.1m, not just orbiting). Advancing is a manual decision,
# not automatic - there's no reliable way to detect "good enough" from
# inside the training loop alone.

import os
# Windows-only fix: PyTorch ships its own OpenMP runtime (libiomp5md.dll)
# and numpy/pybullet can pull in a second one (libomp.dll) - having both
# load in one process trips OMP Error #15 and aborts. This must be set
# BEFORE torch/numpy/pybullet import, so it's the very first line here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import glob
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_env import ArmReachEnv
from config import CURRICULUM_STAGES

# --- Which curriculum stage this run trains. Bump this by hand once the
# current stage's evaluate.py results look solid, then re-run. ---
CURRENT_STAGE_INDEX = 0

STEPS_PER_STAGE = 600_000  # bumped from 300k: stage 0 eval showed real progress (best visibility/reward yet) but distances plateaued around 0.4-0.9m with visible overshoot/oscillation near the target - not yet converged, needs more time on this same stage before advancing


def checkpoint_dir_for(stage_name):
    return f"./models/checkpoints/{stage_name}/"


def final_model_path_for(stage_name):
    return f"./models/arm_1_reach_policy_{stage_name}"


def find_latest_checkpoint(checkpoint_dir, prefix):
    """Looks for files like <prefix>_120000_steps.zip in checkpoint_dir
    and returns (path, steps_already_done) for the one with the highest
    step count, or (None, 0) if there aren't any yet."""
    pattern = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0

    def steps_from_name(path):
        match = re.search(r"_(\d+)_steps\.zip$", path)
        return int(match.group(1)) if match else -1

    latest = max(checkpoints, key=steps_from_name)
    return latest, steps_from_name(latest)


def main(stage_index=CURRENT_STAGE_INDEX, total_timesteps=STEPS_PER_STAGE):
    stage = CURRICULUM_STAGES[stage_index]
    stage_name = stage["name"]
    checkpoint_dir = checkpoint_dir_for(stage_name)
    checkpoint_prefix = f"arm_1_reach_policy_{stage_name}"
    final_model_path = final_model_path_for(stage_name)

    print(f"=== Curriculum stage {stage_index}: '{stage_name}' "
          f"(radius {stage['min_radius']}m - {stage['max_radius']}m) ===")

    # --- env settings ---
    # max_steps stays 400 across ALL stages deliberately - only the
    # workspace radius changes between stages. Changing episode horizon
    # mid-training previously caused real instability (miscalibrated value
    # function -> policy collapse); keeping max_steps constant avoids
    # repeating that mistake while still varying task difficulty via radius.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = ArmReachEnv(arm_index=0, render=False, randomize_arm=False,
                       randomize_plate_positions=True,
                       max_steps=400, camera_width=64, camera_height=64,
                       workspace_min_radius=stage["min_radius"],
                       workspace_max_radius=stage["max_radius"])

    # Sanity-check the environment matches the Gym API correctly
    # before spending any time training on it.
    check_env(env, warn=True)

    checkpoint_path, steps_done = find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)

    if checkpoint_path is not None:
        # Safe: resuming WITHIN a stage means the env hasn't changed at
        # all since this checkpoint was saved - same situation as any
        # normal interrupted-run resume.
        print(f"Found checkpoint at {checkpoint_path} ({steps_done} steps already done in this stage). Resuming...")
        model = PPO.load(checkpoint_path, env=env, device=device)
        remaining_steps = max(0, total_timesteps - steps_done)
        reset_num_timesteps = False
        if remaining_steps == 0:
            print(f"This stage's checkpoint already has {steps_done} >= target {total_timesteps} steps - nothing to do. "
                  f"Raise total_timesteps if you want to keep training this stage further, or advance CURRENT_STAGE_INDEX.")
            env.close()
            return
    else:
        prev_model_path = None
        if stage_index > 0:
            prev_stage_name = CURRICULUM_STAGES[stage_index - 1]["name"]
            candidate = final_model_path_for(prev_stage_name) + ".zip"
            if os.path.exists(candidate):
                prev_model_path = candidate

        if prev_model_path is not None:
            # Warm-start from the previous (easier) stage's finished
            # policy - NOT a raw "resume", since the workspace radius is
            # genuinely different now. reset_num_timesteps=True gives
            # this stage its own fresh TensorBoard run and its own
            # checkpoint counting, rather than pretending it's a
            # continuation of a run under different conditions.
            print(f"No checkpoint for this stage yet. Warm-starting from previous "
                  f"stage's policy: {prev_model_path}")
            model = PPO.load(prev_model_path, env=env, device=device)
            reset_num_timesteps = True
        else:
            print("No checkpoint and no previous-stage model found - starting fresh (random init).")
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=2048,  # standard PPO default - better advantage/gradient estimates than
                               # the 512 used in earlier runs (that was only for faster console
                               # feedback while debugging).
                device=device,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
            )
            reset_num_timesteps = True
        remaining_steps = total_timesteps

    # Checkpoint periodically so a crash/interrupt partway through a
    # stage doesn't lose all the progress in that stage.
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, save_path=checkpoint_dir,
        name_prefix=checkpoint_prefix,
    )

    print(f"Training stage '{stage_name}' for {remaining_steps} more steps "
          f"(stage target: {total_timesteps} total)...")
    model.learn(total_timesteps=remaining_steps, callback=checkpoint_callback,
                reset_num_timesteps=reset_num_timesteps)

    os.makedirs("models", exist_ok=True)
    model.save(final_model_path)
    print(f"Saved stage '{stage_name}' policy to {final_model_path}.zip")

    is_last_stage = stage_index == len(CURRICULUM_STAGES) - 1
    if not is_last_stage:
        next_stage = CURRICULUM_STAGES[stage_index + 1]["name"]
        print(f"\nNext step: check evaluate.py results for this stage. If they look solid, "
              f"set CURRENT_STAGE_INDEX = {stage_index + 1} in train.py (stage '{next_stage}') and re-run.")
    else:
        print("\nThis was the final curriculum stage (full workspace).")

    env.close()


if __name__ == "__main__":
    main()