# arm_controller.py
# Wraps a single loaded arm (URDF body) in PyBullet: tracks which joints
# are "real" arm joints vs. gripper finger joints, exposes simple
# move/open/close commands, and reports its own status for logging.

import pybullet as p
import numpy as np


class ArmController:
    def __init__(self, urdf_path, base_position, name, component, target_position):
        self.name = name
        self.component = component
        self.target_position = target_position

        self.body_id = p.loadURDF(urdf_path, basePosition=base_position, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.body_id)

        JOINT_FIXED = p.JOINT_FIXED
        self.finger_joint_indices = []
        self.arm_joint_indices = []

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.body_id, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode("utf-8")

            if joint_type == JOINT_FIXED:
                continue

            if "finger" in joint_name.lower():
                self.finger_joint_indices.append(i)
            else:
                self.arm_joint_indices.append(i)

        self.has_gripper = len(self.finger_joint_indices) > 0
        self.state = "IDLE"

    def move_to_pose(self, joint_targets):
        for idx, angle in zip(self.arm_joint_indices, joint_targets):
            p.setJointMotorControl2(
                self.body_id, idx, p.POSITION_CONTROL,
                targetPosition=angle, force=200,
            )

    def open_gripper(self, width=0.04):
        if not self.has_gripper:
            return
        for idx in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.body_id, idx, p.POSITION_CONTROL,
                targetPosition=width, force=20,
            )

    def close_gripper(self):
        if not self.has_gripper:
            return
        for idx in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.body_id, idx, p.POSITION_CONTROL,
                targetPosition=0.0, force=20,
            )

    def step(self):
        if self.state == "IDLE":
            straight_up_pose = [0.0] * 7
            self.move_to_pose(straight_up_pose)
            self.open_gripper()
            self.state = "POSITIONING"
            return

        if self.state == "POSITIONING":
            # TODO: drive toward self.target_position here.
            return

    def _get_hand_rotation_matrix(self):
        hand_link_index = p.getJointInfo(self.body_id, self.finger_joint_indices[0])[16]
        hand_orientation = p.getLinkState(self.body_id, hand_link_index)[1]
        return p.getMatrixFromQuaternion(hand_orientation)

    def get_gripper_forward_vector(self):
        """Unit vector for the direction the gripper's fingers currently
        point/approach along - same 'forward' axis the wrist camera looks
        down. Used to reward the arm for actually FACING the plate with
        its clamp, not just being nearby in space (a wrist that's close
        but sideways/upside-down can't grip anything)."""
        rot = self._get_hand_rotation_matrix()
        forward = np.array([rot[2], rot[5], rot[8]], dtype=np.float32)
        norm = np.linalg.norm(forward)
        return forward / norm if norm > 1e-8 else forward

    def get_camera_pose(self, look_distance=0.15):
        """Returns (eye_position, look_at_position, up) for a camera mounted
        between the gripper's fingers, looking in the direction the hand
        is facing. Used to build a view matrix for a wrist-mounted camera."""
        if not self.has_gripper:
            raise RuntimeError(f"{self.name} has no gripper - can't mount a finger camera")

        finger_positions = [p.getLinkState(self.body_id, idx)[0] for idx in self.finger_joint_indices]
        eye_position = tuple(sum(coords) / len(coords) for coords in zip(*finger_positions))

        rot = self._get_hand_rotation_matrix()
        forward = (rot[2], rot[5], rot[8])
        up = (rot[0], rot[3], rot[6])

        look_at_position = tuple(eye_position[i] + forward[i] * look_distance for i in range(3))

        return (eye_position, look_at_position, up)

    def get_end_effector_position(self):
        last_arm_link = self.arm_joint_indices[-1]
        link_state = p.getLinkState(self.body_id, last_arm_link)
        return link_state[0]

    def get_joint_angles(self):
        return [p.getJointState(self.body_id, idx)[0] for idx in self.arm_joint_indices]

    def get_status(self):
        gripper_info = "has_gripper" if self.has_gripper else "NO GRIPPER FOUND"
        return f"{self.name} [{self.component}] -> state: {self.state} ({gripper_info})"