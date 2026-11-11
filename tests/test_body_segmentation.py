"""Body Segmentation Test!"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.body_segmentation import BodySegmenter  # noqa: E402


def make_figure(h=600, w=400):
    """A dark figure (head, torso, arms, legs) on a light noisy background, plus its true mask."""
    rng = np.random.default_rng(0)
    image = rng.integers(190, 240, (h, w, 3), dtype=np.uint8)
    truth = np.zeros((h, w), np.uint8)
    cx = w // 2
    cv2.circle(truth, (cx, 90), 45, 255, -1)                                   # head
    cv2.rectangle(truth, (cx - 15, 130), (cx + 15, 150), 255, -1)              # neck
    cv2.rectangle(truth, (cx - 80, 150), (cx + 80, 360), 255, -1)              # torso
    cv2.rectangle(truth, (cx - 115, 160), (cx - 85, 350), 255, -1)             # arms
    cv2.rectangle(truth, (cx + 85, 160), (cx + 115, 350), 255, -1)
    cv2.rectangle(truth, (cx - 70, 360), (cx - 10, 580), 255, -1)              # legs
    cv2.rectangle(truth, (cx + 10, 360), (cx + 70, 580), 255, -1)
    image[truth > 0] = rng.integers(30, 70, (int((truth > 0).sum()), 3), dtype=np.uint8)
    return image, truth


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max((a | b).sum(), 1)


def test_run_output_contract():
    image, _ = make_figure()
    result = BodySegmenter().run(image)
    assert set(result) == {"bbox", "body_mask", "torso_mask", "keypoints", "keypoint_source"}
    assert result["body_mask"].shape == image.shape[:2]
    assert result["body_mask"].dtype == np.uint8
    assert set(np.unique(result["body_mask"])) <= {0, 255}
    assert len(result["bbox"]) == 4


def test_body_mask_matches_figure():
    image, truth = make_figure()
    result = BodySegmenter().run(image)
    assert iou(result["body_mask"], truth) > 0.85


def test_torso_and_keypoints_are_plausible():
    image, truth = make_figure()
    result = BodySegmenter().run(image)
    kp = result["keypoints"]
    for name in ("neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        assert kp[name] is not None, name
    # Every keypoint name is present; undetected ones are None.
    from controller.body_segmentation import KEYPOINT_NAMES
    assert set(kp) == set(KEYPOINT_NAMES)
    assert kp["left_wrist"] is None  # no OpenPose model in tests
    # Person's left is image right.
    assert kp["left_shoulder"][0] > kp["right_shoulder"][0]
    assert kp["neck"][1] < kp["left_hip"][1]
    torso = result["torso_mask"] > 0
    assert torso.any()
    assert (torso & (truth == 0)).sum() / torso.sum() < 0.1  # torso stays on the body


def test_works_on_grayscale_and_upscaled_input():
    image, truth = make_figure(300, 200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = BodySegmenter().run(gray)
    assert result["body_mask"].shape == gray.shape
    assert iou(result["body_mask"], truth) > 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
