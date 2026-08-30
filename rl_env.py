# rl_env.py
# A Gymnasium environment wrapping our PyBullet world.
#
# TASK: teach the ONE remaining arm to move its gripper close to
# whichever of its 4 plates it's currently been assigned, using ONLY
# what its own wrist camera (mounted between the finger clamps, see
# ArmController.get_camera_pose()) can see. The target position fed to
# the policy comes from cv_detector.detect_plates_from_arm(): OpenCV
# proposes candidate regions in the RGB frame -> the trained CNN
# (plate_classifier.py) classifies each one -> the depth buffer unprojects
# the match into a 3D point. At no point does this environment read the
# plate's real physics-engine position (Plate.get_position()) - the arm
# has to actually find the plate with its camera, exactly like it would
# have to on a real robot.
#
# Since there's only 1 arm now but 4 plates, the "which plate is my
# target this episode" job that used to belong to arm_index (1 arm <->
# 1 plate) is now handled by target_plate_index, picked at random each
# reset (or fixed via options={"target_plate_index": i}) - the arm has
# to learn to reach ANY of its 4 plates, which is the reach sub-skill
# it'll need for every step of building the pyramid.
#
# PLATE POSITIONS ARE RANDOMIZED EVERY EPISODE (randomize_plate_positions,
# default True): each reset samples a fresh spot for all 4 plates inside
# config.WORKSPACE_MIN_RADIUS..WORKSPACE_MAX_RADIUS around the arm base,
# keeping WORKSPACE_MIN_CLEARANCE between any two plates so they don't
# spawn overlapping. This is deliberate - the goal isn't an arm that's
# memorized 4 fixed coordinates, it's an arm that visually searches for
# and detects a plate wherever it happens to actually be placed.
#
# Observation (17 numbers):
#   - 7 arm joint angles
#   - 3 end-effector (x,y,z) position
#   - 3 detected target plate (x,y,z) position (camera + CNN, NOT ground truth)
#   - 1 "currently visible" flag (1.0 if the target was detected this
#     step, 0.0 if not - in which case the position is the last one seen)
#   - 3 gripper forward/approach unit vector (which way the clamp is facing)
#
# Action (7 numbers):
#   - target angle delta for each of the 7 arm joints, applied each step
#
# Reward:
#   - negative distance between end-effector and the DETECTED plate position
#   - extra penalty per step the target isn't visible at all, so the
#     policy is pushed to keep the plate in frame, not just swing blind
#   - alignment bonus for FACING the plate (gripper forward vector pointed
#     toward it, not just nearby) - this matters because "success" here
#     needs to leave the arm in a pose where closing the clamp would
#     actually grip the plate, not just be in its general vicinity
#   - bonus when close AND facing a currently-visible target

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from config import (ARM_URDF, ARMS, PLATES, SIMULATION_TIMESTEP,
                     WORKSPACE_MIN_RADIUS, WORKSPACE_MAX_RADIUS, WORKSPACE_MIN_CLEARANCE)
from arm_controller import ArmController
from plate_factory import spawn_all_plates
from cv_detector import detect_plates_from_arm


class ArmReachEnv(gym.Env):
    def __init__(self, arm_index=0, render=False, max_steps=500, randomize_arm=False,
                 randomize_target_plate=True, randomize_plate_positions=True,
                 camera_width=128, camera_height=128,
                 workspace_min_radius=WORKSPACE_MIN_RADIUS, workspace_max_radius=WORKSPACE_MAX_RADIUS):
        super().__init__()

        self.arm_index = arm_index          # which entry in config.ARMS this env controls (only 1 arm exists now, so always 0)
        self.randomize_arm = randomize_arm  # kept for API compatibility; with 1 arm in ARMS this is a no-op
        self.randomize_target_plate = randomize_target_plate  # if True, picks a random one of the 4 plates as the reach target each reset
        self.randomize_plate_positions = randomize_plate_positions  # if True, samples a fresh spot for every plate each reset instead of using config.py's fixed spawn_position
        self.workspace_min_radius = workspace_min_radius  # overridable so train.py can pass a curriculum stage's narrower range (see config.CURRICULUM_STAGES) instead of always the full workspace
        self.workspace_max_radius = workspace_max_radius
        self.render_mode = render
        self.max_steps = max_steps
        self.step_count = 0
        self.camera_width = camera_width
        self.camera_height = camera_height

        self.connection_mode = p.GUI if render else p.DIRECT
        self.physics_client = None  # set up in reset()

        # 7 joint angles + 3 eef pos + 3 detected target pos + 1 visibility flag + 3 gripper forward vector
        obs_dim = 7 + 3 + 3 + 1 + 3
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        # Action = delta angle applied to each of the 7 joints per step (radians)
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(7,), dtype=np.float32)

        self.success_threshold = 0.05  # meters - "close enough" to count as reached
        self.alignment_threshold = 0.85  # cosine similarity - how directly the gripper must face the plate to count as "success" (1.0 = perfectly facing, 0 = perpendicular). 0.85 ~= within ~32 degrees.
        self.not_visible_penalty = 0.5  # extra reward penalty per step target isn't detected (lowered from 1.0 - at that level it was comparable to or larger than the reward gained by closing another 0.1-0.2m of distance, which combined with near_field_radius rarely activating, incentivized "park at a safely-visible distance" over pushing closer)
        self.alignment_reward_weight = 0.3  # small - alignment is a tiebreaker near the target, never a substitute for closing distance (was 2.0, which let noisy far-away detections outweigh -distance entirely - a reward-hacking bug)
        self.near_field_radius = 0.6  # meters - widened from 0.3: eval showed policies converging to a 0.4-0.9m "safe distance" equilibrium and never entering the old 0.3m gate at all, so the fine-approach/hold incentive never activated for almost any episode. Widening it means getting closer starts paying off well before the final few centimeters, instead of only after crossing a threshold the policy was never reaching.

        self._last_seen_target_pos = np.zeros(3, dtype=np.float32)

    def _connect_physics(self):
        self.physics_client = p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _sample_plate_positions(self, arm_base_position):
        """Random [x,y,z] for each of the 4 plates, inside the reachable
        annulus around the arm base, rejecting any draw that lands too
        close to another already-placed plate (so they don't spawn
        overlapping/stacked on top of each other)."""
        base_x, base_y, _ = arm_base_position
        positions = []
        for cfg in PLATES:
            z = cfg["half_extents"][2]  # rest flat on the ground plane
            for _ in range(50):  # generous retry budget - annulus is roomy relative to clearance
                radius = self.np_random.uniform(self.workspace_min_radius, self.workspace_max_radius)
                angle = self.np_random.uniform(0, 2 * np.pi)
                x = base_x + radius * np.cos(angle)
                y = base_y + radius * np.sin(angle)
                if all(np.hypot(x - px, y - py) >= WORKSPACE_MIN_CLEARANCE for px, py, _ in positions):
                    positions.append([float(x), float(y), float(z)])
                    break
            else:
                # fell back after exhausting retries - still valid, just not clearance-checked
                positions.append([float(x), float(y), float(z)])
        return positions

    def _reset_world(self):
        """Clears all bodies and reloads gravity/ground, WITHOUT disconnecting
        the GUI window. Reconnecting p.GUI every episode is unreliable on
        Windows and caused 'Not connected to physics server' crashes."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.physics_client is None:
            self._connect_physics()
        self._reset_world()

        if self.randomize_arm:
            self.arm_index = self.np_random.integers(0, len(ARMS))

        arm_cfg = ARMS[self.arm_index]
        self.arm = ArmController(
            urdf_path=ARM_URDF,
            base_position=arm_cfg["base_position"],
            name=arm_cfg["name"],
            component=arm_cfg["component"],
            target_position=arm_cfg["target_position"],
        )

        if self.randomize_plate_positions:
            positions = self._sample_plate_positions(arm_cfg["base_position"])
            self.plates = spawn_all_plates(PLATES, positions=positions)
        else:
            self.plates = spawn_all_plates(PLATES)
        # target_plate_index matches the dataset_capture.py / plate_classifier.py
        # class index (plate_1 -> class 0, plate_2 -> class 1, etc). This is
        # just task assignment ("which plate to reach for this episode"), not
        # a location shortcut - the arm still has to find it visually.
        if options and "target_plate_index" in options:
            self.target_plate_index = options["target_plate_index"]
        elif self.randomize_target_plate:
            self.target_plate_index = int(self.np_random.integers(0, len(PLATES)))
        else:
            self.target_plate_index = 0

        self.step_count = 0
        self._last_seen_target_pos = np.zeros(3, dtype=np.float32)

        # let physics settle one step before returning first observation
        p.stepSimulation()

        return self._get_obs(), {}

    def _detect_target(self):
        """Runs the real detection pipeline through the arm's wrist camera and
        returns (position, visible_flag) for THIS arm's assigned plate class.
        If the plate isn't detected this step, falls back to the last known
        position and reports visible=0.0."""
        detections = detect_plates_from_arm(
            self.arm, width=self.camera_width, height=self.camera_height
        )

        best = None
        for det in detections:
            if det["plate_index"] != self.target_plate_index:
                continue
            if best is None or det["confidence"] > best["confidence"]:
                best = det

        if best is not None:
            pos = np.array(best["position"], dtype=np.float32)
            self._last_seen_target_pos = pos
            return pos, 1.0

        return self._last_seen_target_pos, 0.0

    def _get_obs(self):
        joint_angles = np.array(self.arm.get_joint_angles(), dtype=np.float32)
        eef_pos = np.array(self.arm.get_end_effector_position(), dtype=np.float32)
        target_pos, visible = self._detect_target()
        forward_vec = np.array(self.arm.get_gripper_forward_vector(), dtype=np.float32)
        return np.concatenate([
            joint_angles, eef_pos, target_pos, np.array([visible], dtype=np.float32), forward_vec
        ])

    def step(self, action):
        current_angles = self.arm.get_joint_angles()
        new_angles = [a + float(delta) for a, delta in zip(current_angles, action)]
        self.arm.move_to_pose(new_angles)

        p.stepSimulation()
        self.step_count += 1

        obs = self._get_obs()
        eef_pos = obs[7:10]
        target_pos = obs[10:13]
        # bool(...) here matters: obs[13] > 0.5 is a numpy.bool_, and SB3's
        # check_env() asserts terminated/truncated are plain Python bools -
        # numpy.bool_ silently poisons everything downstream of it (success,
        # terminated) and fails that assertion.
        visible = bool(obs[13] > 0.5)
        forward_vec = obs[14:17]
        distance = float(np.linalg.norm(eef_pos - target_pos))

        # Alignment: how directly the gripper's forward/approach axis
        # points AT the plate, via cosine similarity between "gripper
        # forward" and "direction from eef to target". 1.0 = dead-on,
        # 0 = perpendicular (wrist facing sideways to the plate),
        # negative = facing away entirely. This is what actually matters
        # for grasping later - being 2cm away facing the wrong way is
        # useless for a clamp.
        to_target = target_pos - eef_pos
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm > 1e-6:
            alignment = float(np.dot(forward_vec, to_target / to_target_norm))
        else:
            alignment = 1.0  # eef is essentially at the target already

        reward = -distance
        if not visible:
            reward -= self.not_visible_penalty
        elif distance < self.near_field_radius:
            reward += self.alignment_reward_weight * alignment

        success = bool(visible and distance < self.success_threshold and alignment > self.alignment_threshold)
        if success:
            reward += 10.0

        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps)

        info = {"distance": distance, "alignment": alignment, "success": success, "target_visible": visible}

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None