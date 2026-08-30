# cv_detector.py
# Real object detection for the finger cameras. Pipeline:
#   1. OpenCV finds candidate regions in the RGB frame (color-blob proposals)
#   2. each candidate crop is classified by the trained CNN (plate_classifier.py)
#      -> which plate it is, and how confident
#   3. the candidate's pixels are unprojected using the depth buffer
#      (camera.py's pixel_to_world, unchanged) -> 3D world position
#
# At NO point does this read body_id / segmentation ground truth. Only the
# RGB image and depth buffer - the same two things a real camera + a real
# trained model would give you.

import numpy as np
import cv2

from camera import capture_frame, pixel_to_world
from plate_classifier import load_model, predict, MODEL_PATH

CONFIDENCE_THRESHOLD = 0.6

# Sanity bound for unprojected detection positions: a plate physically
# cannot be farther than this from the arm's own base, since it's
# spawned inside config.WORKSPACE_MAX_RADIUS. A detection landing outside
# this (observed: several METERS away, even negative coordinates) means
# OpenCV's saturation mask + the CNN both confidently agreed on a false
# positive (checkerboard floor pattern, a shadow, etc) - low-res 64x64
# frames make this more likely. Treating these as "not detected" instead
# of trusting them matters a lot: an ungated garbage position was letting
# the policy farm reward from the alignment bonus by facing toward
# nonsense far-away points instead of actually approaching the plate.
MAX_PLAUSIBLE_DETECTION_RADIUS = 1.0  # meters from arm base; generous margin above WORKSPACE_MAX_RADIUS

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def find_candidate_boxes(rgb_image):
    """OpenCV region-proposal step: anything reasonably saturated (i.e. not
    the gray ground plane / gray robot links) is a candidate object."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    mask = (saturation > 60).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 4 and h >= 4:
            boxes.append((x, y, w, h))
    return boxes


def detect_plates_from_arm(arm_controller, width=256, height=256):
    """Runs the full pipeline using the arm's own wrist camera (between the
    finger clamps, see ArmController.get_camera_pose()).

    Returns a list of dicts:
      {"plate_index": int, "confidence": float, "position": (x, y, z), "bbox": (x, y, w, h)}
    One entry per detected plate above CONFIDENCE_THRESHOLD - could be zero,
    one, or several, exactly like a real detector."""
    model = _get_model()

    eye_position, look_at_position, up_vector = arm_controller.get_camera_pose()
    rgba, depth_buffer, view_matrix_4x4, proj_matrix_4x4 = capture_frame(
        eye_position, look_at_position, up_vector, width, height
    )
    rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)

    detections = []
    for (x, y, w, h) in find_candidate_boxes(rgb):
        crop = rgb[y:y + h, x:x + w]
        plate_index, confidence = predict(model, crop)
        if confidence < CONFIDENCE_THRESHOLD:
            continue  # not confident this candidate is a real plate

        # Unproject every pixel in the box using the depth buffer, average
        # to get a 3D centroid - same math as camera.py, just fed by an
        # OpenCV box instead of a segmentation mask.
        ys, xs = np.mgrid[y:y + h, x:x + w]
        world_points = np.array([
            pixel_to_world(px, py, depth_buffer[py, px], width, height, view_matrix_4x4, proj_matrix_4x4)
            for py, px in zip(ys.flatten(), xs.flatten())
        ])
        centroid = world_points.mean(axis=0)

        # Plausibility filter - see MAX_PLAUSIBLE_DETECTION_RADIUS comment above.
        dist_from_base = float(np.linalg.norm(centroid - np.array(arm_controller.base_position)))
        if dist_from_base > MAX_PLAUSIBLE_DETECTION_RADIUS:
            continue

        detections.append({
            "plate_index": plate_index,
            "confidence": confidence,
            "position": tuple(centroid),
            "bbox": (x, y, w, h),
        })

    return detections