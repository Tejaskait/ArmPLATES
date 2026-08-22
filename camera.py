# camera.py
# Pure camera utilities - rendering a frame and turning a pixel + its
# depth-buffer value into a 3D world point. Nothing in this file knows
# which object is which; that's cv_detector.py's job (OpenCV region
# proposals + the trained CNN classifier). Keeping identity-blind camera
# math separate from object identification on purpose, so there's no
# temptation to shortcut through PyBullet's segmentation IDs again.

import numpy as np
import pybullet as p


def capture_frame(eye_position, target_position, up_vector,
                   width=128, height=128, fov=60, near=0.05, far=3.0):
    """Renders one frame from a camera at eye_position looking at target_position.
    Returns (rgba, depth_buffer, view_matrix_4x4, proj_matrix_4x4).
    Deliberately does NOT request/return the segmentation buffer - object
    identity has to come from the CNN classifier in cv_detector.py, not
    from PyBullet telling us which body_id each pixel belongs to."""
    view_matrix = p.computeViewMatrix(eye_position, target_position, up_vector)
    proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

    _, _, rgba, depth_buffer, _ = p.getCameraImage(
        width, height, view_matrix, proj_matrix,
        renderer=p.ER_TINY_RENDERER,  # software renderer, works headless (p.DIRECT) too
        flags=p.ER_NO_SEGMENTATION_MASK,
    )

    # PyBullet returns these as flat lists in DIRECT mode sometimes - normalize to arrays
    depth_buffer = np.reshape(depth_buffer, (height, width))

    view_matrix_4x4 = np.asarray(view_matrix).reshape([4, 4], order="F")
    proj_matrix_4x4 = np.asarray(proj_matrix).reshape([4, 4], order="F")

    return rgba, depth_buffer, view_matrix_4x4, proj_matrix_4x4


def pixel_to_world(pixel_x, pixel_y, depth_value, width, height, view_matrix_4x4, proj_matrix_4x4):
    """Converts one pixel + its depth-buffer value into a world-space (x,y,z) point.
    Standard PyBullet inverse-projection: undo the projection*view transform."""
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix_4x4, view_matrix_4x4))

    x_ndc = (2.0 * pixel_x - width) / width
    y_ndc = -(2.0 * pixel_y - height) / height
    z_ndc = 2.0 * depth_value - 1.0

    pixel_pos = np.array([x_ndc, y_ndc, z_ndc, 1.0])
    world_pos = np.matmul(tran_pix_world, pixel_pos)
    world_pos /= world_pos[3]  # homogeneous divide

    return world_pos[:3]