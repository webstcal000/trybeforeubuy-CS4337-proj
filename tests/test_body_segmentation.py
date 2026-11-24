"""Smoke tests for BodySegmenter on a synthetic person-shaped figure.

Run with:  python -m pytest tests   (or:  python tests/test_body_segmentation.py)
"""

import math
import os
import sys

import cv2
import numpy as np
import pytest

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


def make_person(right=(8, 4), left=(8, 4), arms=True, h=640, w=520, arm_length=1.0):
    """A front-facing person with jointed, skin-coloured arms, plus true keypoints.

    right/left are the PERSON's arms as (upper arm, forearm) angles in degrees from
    straight down, positive = away from the body. Real proportions: upper arm
    128 px, forearm 105 px, hand about three quarters of a forearm; arm_length
    scales the whole arm (people vary).
    """
    rng = np.random.default_rng(0)
    image = rng.integers(200, 240, (h, w, 3), dtype=np.uint8)
    skin, shirt, legs = (np.zeros((h, w), np.uint8) for _ in range(3))
    cx = w // 2
    cv2.ellipse(skin, (cx, 95), (38, 48), 0, 0, 360, 255, -1)                  # head
    cv2.rectangle(skin, (cx - 16, 135), (cx + 16, 165), 255, -1)                # neck
    cv2.fillPoly(shirt, [np.array([[cx - 78, 165], [cx + 78, 165], [cx + 64, 300], [cx + 70, 360],
                                   [cx - 70, 360], [cx - 64, 300]], np.int32)], 255)
    for dx in (-70, 70):
        cv2.circle(shirt, (cx + dx, 180), 22, 255, -1)                           # shoulders
    cv2.rectangle(legs, (cx - 68, 360), (cx - 6, h - 10), 255, -1)
    cv2.rectangle(legs, (cx + 6, 360), (cx + 68, h - 10), 255, -1)

    kp = {}
    for side, (up, fore) in (("right", right), ("left", left)):
        out = -1 if side == "right" else 1                  # person's right is image left
        joint = np.array([cx + out * 72, 182.0])
        elbow = joint + 128 * arm_length * np.array([out * math.sin(math.radians(up)), math.cos(math.radians(up))])
        wrist = elbow + 105 * arm_length * np.array([out * math.sin(math.radians(fore)), math.cos(math.radians(fore))])
        hand = wrist + 0.65 * (wrist - elbow)
        kp.update({f"{side}_shoulder": joint, f"{side}_elbow": elbow, f"{side}_wrist": wrist})
        if arms:
            p = [tuple(int(round(v)) for v in q) for q in (joint, elbow, wrist, hand)]
            cv2.line(skin, p[0], p[1], 255, 30)
            cv2.line(skin, p[1], p[2], 255, 24)
            cv2.line(skin, p[2], p[3], 255, 26)
            cv2.circle(skin, p[1], 14, 255, -1)
            sleeve_end = tuple(int(round(v)) for v in joint + 0.45 * (elbow - joint))
            cv2.line(shirt, p[0], sleeve_end, 255, 36)
    truth = ((skin | shirt | legs) > 0).astype(np.uint8) * 255
    image[legs > 0] = (60, 50, 40)
    image[skin > 0] = (120, 150, 200)
    image[shirt > 0] = (150, 70, 30)
    image = np.clip(image.astype(int) + rng.integers(-12, 12, image.shape), 0, 255).astype(np.uint8)
    return image, truth, kp


def segmenter_with_face():
    """A BodySegmenter that 'detects' the drawn face of make_person(), as Haar would on a photo."""
    seg = BodySegmenter()
    detect = seg.detect_person

    def detect_with_face(work):
        box = detect(work)
        k = seg.scale
        seg.face = (int(230 * k), int(68 * k), int(60 * k), int(62 * k))
        return box

    seg.detect_person = detect_with_face
    return seg


def arm_error(result, truth_kp, part):
    """Distance from the true joint, as a fraction of the shoulder width."""
    span = np.linalg.norm(truth_kp["left_shoulder"] - truth_kp["right_shoulder"])
    return np.linalg.norm(np.array(result["keypoints"][part]) - truth_kp[part]) / span


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
    assert kp["left_knee"] is None  # no OpenPose model in tests
    # Elbows are traced from the outline, so they land on the arms.
    assert 85 <= kp["right_elbow"][0] <= 115 and 285 <= kp["left_elbow"][0] <= 315
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


ARM_POSES = {
    "arms down": dict(right=(8, 4), left=(8, 4)),
    "A pose": dict(right=(35, 35), left=(35, 35)),
    "elbows bent": dict(right=(12, 60), left=(12, 60)),
}


# Real photos normally go through the face path. Without a face, the torso
# estimate itself is often wrong (and try_on refuses it), so that path is only
# checked on the plainest pose.
@pytest.mark.parametrize("pose,with_face", [(pose, True) for pose in ARM_POSES] + [("arms down", False)])
def test_elbows_and_wrists_follow_the_arms(pose, with_face):
    image, _, truth_kp = make_person(**ARM_POSES[pose])
    seg = segmenter_with_face() if with_face else BodySegmenter()
    result = seg.run(image)
    assert result["keypoint_source"] == ("face+mask" if with_face else "mask")
    for part in ("right_elbow", "left_elbow", "right_wrist", "left_wrist"):
        assert result["keypoints"][part] is not None, part
        assert arm_error(result, truth_kp, part) < 0.25, (part, result["keypoints"][part], truth_kp[part])


def test_shorter_arms_are_measured_not_assumed():
    """Arms much shorter than the body suggests: measure them, don't put the elbow at the wrist."""
    image, _, truth_kp = make_person(right=(35, 35), left=(35, 35), arm_length=0.55)
    result = segmenter_with_face().run(image)
    for part in ("right_elbow", "left_elbow", "right_wrist", "left_wrist"):
        assert result["keypoints"][part] is not None, part
        assert arm_error(result, truth_kp, part) < 0.25, (part, result["keypoints"][part], truth_kp[part])


def test_raised_arms_are_followed():
    image, _, truth_kp = make_person(right=(150, 160), left=(8, 4))
    result = segmenter_with_face().run(image)
    for part in ("right_elbow", "left_elbow", "left_wrist"):
        assert arm_error(result, truth_kp, part) < 0.25, part
    # The raised hand is out of the frame: no wrist rather than a guess.
    assert result["keypoints"]["right_wrist"] is None


def test_arms_out_of_frame_get_elbows_but_no_wrists():
    image, truth, truth_kp = make_person(right=(90, 90), left=(90, 90))      # T pose
    result = segmenter_with_face().run(image)
    for side in ("right", "left"):
        assert arm_error(result, truth_kp, f"{side}_elbow") < 0.25
        assert result["keypoints"][f"{side}_wrist"] is None
    # The arms reach past the person box but stay in the body mask.
    assert iou(result["body_mask"], truth) > 0.9


def test_no_visible_arms_means_no_arm_keypoints():
    image, _, _ = make_person(arms=False)
    for seg in (BodySegmenter(), segmenter_with_face()):
        kp = seg.run(image)["keypoints"]
        assert kp["left_shoulder"] is not None
        assert all(kp[p] is None for p in ("left_elbow", "right_elbow", "left_wrist", "right_wrist"))


def test_same_photo_gives_the_same_result():
    image, _, _ = make_person(right=(35, 35), left=(35, 35))
    a, b = BodySegmenter().run(image), BodySegmenter().run(image)
    assert np.array_equal(a["body_mask"], b["body_mask"])
    assert a["keypoints"] == b["keypoints"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
