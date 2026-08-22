# dataset_capture.py
# Generates labeled training crops for the plate classifier CNN.
#
# One plate spawned ALONE per scene (not all 8 together). This avoids the
# original bug: with all 8 plates in frame, overlapping/adjacent color
# blobs could merge into a single OpenCV contour, and the label taken
# from that merged box's center pixel would consistently favor whichever
# plate happened to be biggest/closest - starving small plates (plate_8
# got 0 crops) and wildly imbalancing the rest.
#
# With exactly one plate in the scene there's no ambiguity: the single
# largest saturated contour IS that plate, every time, and we control
# exactly how many crops each class gets.
#
# PyBullet ground truth is still used here, offline, only to know which
# plate is currently in the scene - that's the label. cv_detector.py at
# inference time never reads this; it only ever sees RGB + depth.

import os
import shutil
import uuid
import numpy as np
import cv2
import pybullet as p
import pybullet_data

from config import PLATES

OUT_DIR = "data/plate_dataset"
IMG_SIZE = 64  # crops are resized to 64x64 before saving
CROPS_PER_PLATE = 200

MIN_BOX_PIXELS = 3
IMAGE_SIZE = 256


def random_camera_pose(rng, plate_half_extent):
    """Eye position on a small dome centered on the plate, distance scaled
    to the plate's own size so every plate - tiny or large - reliably
    fills a reasonable chunk of the frame."""
    radius = rng.uniform(4.0, 7.0) * plate_half_extent
    radius = float(np.clip(radius, 0.08, 0.4))
    theta = rng.uniform(0, 2 * np.pi)
    elevation = rng.uniform(0.4, 1.2)  # radians above the plate, keeps some top-down angle
    eye = [
        radius * np.cos(theta) * np.cos(elevation),
        radius * np.sin(theta) * np.cos(elevation),
        radius * np.sin(elevation) + 0.02,
    ]
    look_at = [rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 0.02]
    return eye, look_at


def find_largest_box(rgb_image):
    """OpenCV region-proposal step, single-object scene: the largest
    saturated blob is the plate. Returns (x, y, w, h) or None."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    mask = (saturation > 60).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < MIN_BOX_PIXELS or h < MIN_BOX_PIXELS:
        return None
    return (x, y, w, h)


def capture_crops_for_plate(plate_cfg, rng, out_dir, target_count):
    saved = 0
    attempts = 0
    max_attempts = target_count * 20  # generous - some poses will miss the plate

    while saved < target_count and attempts < max_attempts:
        attempts += 1

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=plate_cfg["half_extents"])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=plate_cfg["half_extents"], rgbaColor=plate_cfg["color"])
        p.createMultiBody(baseMass=0.3, baseCollisionShapeIndex=col,
                           baseVisualShapeIndex=vis, basePosition=[0, 0, plate_cfg["half_extents"][2]])
        p.stepSimulation()

        plate_half_extent = plate_cfg["half_extents"][0]
        eye, look_at = random_camera_pose(rng, plate_half_extent)

        view_matrix = p.computeViewMatrix(eye, look_at, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.03, 3.0)
        _, _, rgba, _, _ = p.getCameraImage(IMAGE_SIZE, IMAGE_SIZE, view_matrix, proj_matrix,
                                             renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgba, (IMAGE_SIZE, IMAGE_SIZE, 4))[:, :, :3].astype(np.uint8)

        box = find_largest_box(rgb)
        if box is None:
            continue
        x, y, w, h = box
        crop = cv2.resize(rgb[y:y + h, x:x + w], (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(out_dir, f"{uuid.uuid4().hex}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        saved += 1

    return saved, attempts


def main():
    rng = np.random.default_rng(0)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)  # wipe stale/imbalanced data from previous runs

    counts = {}
    for i, cfg in enumerate(PLATES):
        class_dir = os.path.join(OUT_DIR, str(i))
        os.makedirs(class_dir, exist_ok=True)
        saved, attempts = capture_crops_for_plate(cfg, rng, class_dir, CROPS_PER_PLATE)
        counts[i] = saved
        print(f"plate_index {i} ({cfg['name']}): {saved}/{CROPS_PER_PLATE} crops "
              f"({attempts} attempts)")

    p.disconnect()

    print("\nFinal per-class crop counts:")
    empty = []
    for i, cfg in enumerate(PLATES):
        print(f"  plate_index {i} ({cfg['name']}): {counts[i]} crops")
        if counts[i] == 0:
            empty.append(i)
    if empty:
        print(f"\nWARNING: classes {empty} still got zero crops - widen the radius "
              f"range in random_camera_pose() for those sizes.")
    else:
        print("\nAll classes have crops. Ready for: python plate_classifier.py")


if __name__ == "__main__":
    main()