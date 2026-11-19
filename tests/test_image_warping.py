"""Image Warping Tests!"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.garment_cutout import cut_out_garment  # noqa: E402
from controller.image_warping import (GarmentWarper, ThinPlateSpline,  # noqa: E402
                                      fit_affine, fit_perspective)
from model.data_model import Garment  # noqa: E402


def make_tshirt(h=300, w=240):
    """A red T-shirt on a grey background with its DeepFashion-style landmarks."""
    rng = np.random.default_rng(1)
    image = rng.integers(120, 140, (h, w, 3), dtype=np.uint8)
    poly = np.array([[95, 40], [145, 40], [200, 70], [185, 110], [165, 100],
                     [165, 260], [75, 260], [75, 100], [55, 110], [40, 70]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    image[mask > 0] = (40, 40, 200)
    landmarks = {"collar_l": (105, 42), "collar_r": (135, 42),
                 "sleeve_l": (48, 90), "sleeve_r": (192, 90),
                 "hem_l": (78, 258), "hem_r": (162, 258)}
    return image, mask, landmarks


def body_keypoints(cx=300, top=150, span=160):
    """Person's left is on the image right."""
    return {"neck": (cx, top - 20),
            "right_shoulder": (cx - span // 2, top), "left_shoulder": (cx + span // 2, top),
            "right_hip": (cx - 55, top + 220), "left_hip": (cx + 55, top + 220),
            "left_elbow": None, "right_elbow": None, "left_wrist": None, "right_wrist": None}


def make_garment(method_mask=True):
    image, mask, lm = make_tshirt()
    return Garment(0, "synthetic", "Tee", {}, clothes_type=1, landmarks=lm,
                   loader=lambda _: image, mask=mask if method_mask else None)


def test_fit_affine_recovers_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (6, 2))
    M = np.array([[1.2, 0.1, 5.0], [-0.2, 0.9, -3.0]])
    dst = src @ M[:, :2].T + M[:, 2]
    assert np.allclose(fit_affine(src, dst), M, atol=1e-6)


def test_fit_perspective_falls_back_to_affine_for_three_points():
    src = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
    H = fit_perspective(src, src * 2)
    assert H.shape == (3, 3) and np.allclose(H[2], [0, 0, 1])


def test_thin_plate_spline_interpolates_exactly():
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 1, (8, 2))
    dst = src + rng.normal(0, 0.05, src.shape)
    tps = ThinPlateSpline(src, dst)
    assert np.allclose(tps(src), dst, atol=1e-6)
    # Pure affine data stays affine everywhere.
    tps = ThinPlateSpline(src, src * 1.5 + 2)
    pts = rng.uniform(0, 1, (20, 2))
    assert np.allclose(tps(pts), pts * 1.5 + 2, atol=1e-6)


def test_correspondences_respect_left_right_conventions():
    _, _, lm = make_tshirt()
    src, dst, names = GarmentWarper().compute_correspondences(lm, body_keypoints())
    d = dict(zip(names, dst))
    # Garment "_l" is image-left, so it must land on the image-left side of the body.
    assert d["collar_l"][0] < 300 < d["collar_r"][0]
    assert d["hem_l"][0] < 300 < d["hem_r"][0]
    assert d["collar_l"][1] < d["hem_l"][1]
    # No elbows/wrists -> sleeves keep the garment's own proportions.
    assert d["sleeve_l"][0] < d["collar_l"][0] and d["sleeve_r"][0] > d["collar_r"][0]


def test_garment_is_sized_to_the_torso():
    warper = GarmentWarper(method="tps")
    _, mask = warper.warp(make_garment(), body_keypoints(), (600, 600, 3))
    # Shirt body (below the sleeves) should be about as wide as the torso: 0.92 * 160 px.
    row = np.nonzero(mask[300] > 127)[0]
    assert 120 < row.max() - row.min() < 175


@pytest.mark.parametrize("method,tol", [("affine", 25.0), ("perspective", 1.0), ("tps", 2.0)])
def test_warp_places_landmarks_on_targets(method, tol):
    warper = GarmentWarper(method=method)
    warped, mask = warper.warp(make_garment(), body_keypoints(), (600, 600, 3))
    assert warped.shape == (600, 600, 3) and mask.shape == (600, 600)
    err = np.linalg.norm(warper.last["warped_points"] - warper.last["dst"], axis=1)
    assert err.max() < tol, (method, err)
    # The warped garment covers the torso centre and stays near the body.
    assert mask[250, 300] > 0
    assert (mask > 0).sum() > 0 and mask[:, :100].max() == 0


def test_tps_uses_arm_keypoints_when_available():
    kp = body_keypoints()
    kp.update({"right_elbow": (190, 260), "left_elbow": (410, 260)})
    warper = GarmentWarper(method="tps")
    warper.warp(make_garment(), kp, (600, 600, 3))
    assert "sleeve_l" in warper.last["names"] and "sleeve_r" in warper.last["names"]
    err = np.linalg.norm(warper.last["warped_points"] - warper.last["dst"], axis=1)
    assert err.max() < 2.0


def test_blend_keeps_background_and_shows_garment():
    body = np.full((600, 600, 3), 200, np.uint8)
    body_mask = np.zeros((600, 600), np.uint8)
    cv2.rectangle(body_mask, (200, 100), (400, 500), 255, -1)
    warper = GarmentWarper(method="tps")
    warped, mask = warper.warp(make_garment(), body_keypoints(), body.shape)
    out = warper.blend(body, warped, mask, body_mask)
    assert out.shape == body.shape and out.dtype == np.uint8
    assert np.array_equal(out[:50, :50], body[:50, :50])         # untouched far away
    assert out[250, 300, 2] > 150 and out[250, 300, 0] < 100     # red shirt in the middle


def test_missing_shoulders_raise():
    _, _, lm = make_tshirt()
    kp = body_keypoints()
    kp["left_shoulder"] = None
    with pytest.raises(ValueError):
        GarmentWarper().compute_correspondences(lm, kp)


def test_collapsed_keypoints_raise():
    _, _, lm = make_tshirt()
    kp = {"right_shoulder": (305, 137), "left_shoulder": (311, 137),
          "right_hip": (305, 143), "left_hip": (311, 143)}
    with pytest.raises(ValueError):
        GarmentWarper().compute_correspondences(lm, kp)


def test_lower_body_garment_uses_hips():
    lm = {"waist_l": (60, 20), "waist_r": (140, 20), "hem_l": (50, 280), "hem_r": (150, 280)}
    src, dst, names = GarmentWarper().compute_correspondences(lm, body_keypoints(), clothes_type=2)
    d = dict(zip(names, dst))
    assert set(names) == {"waist_l", "waist_r", "hem_l", "hem_r"}
    assert abs(d["waist_l"][1] - 370) < 20 and d["hem_l"][1] > d["waist_l"][1]


def test_cutout_finds_shirt_and_drops_skin():
    image, truth, lm = make_tshirt()
    # Skin-coloured arms poking out of the sleeves.
    cv2.rectangle(image, (20, 100), (50, 200), (120, 160, 220), -1)
    cv2.rectangle(image, (190, 100), (220, 200), (120, 160, 220), -1)
    mask = cut_out_garment(image, (30, 35, 210, 265), lm, clothes_type=1)
    inter = ((mask > 0) & (truth > 0)).sum()
    union = ((mask > 0) | (truth > 0)).sum()
    assert inter / union > 0.85
    assert mask[150, 35] == 0   # arm is not part of the garment


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_cutout_light_shirt_on_white_product_shot():
    image, truth, lm = make_tshirt()
    image[:] = 250
    image[truth > 0] = 205                       # light grey shirt on white, like garment 265
    image[-3:] = 0                               # black letterbox bars, also like 265
    image[:3] = 0
    mask = cut_out_garment(image, (30, 35, 210, 265), lm, clothes_type=1)
    inter = ((mask > 0) & (truth > 0)).sum()
    union = ((mask > 0) | (truth > 0)).sum()
    assert inter / union > 0.95


def test_beige_clothing_is_not_treated_as_arms():
    h, w = 600, 600
    body = np.full((h, w, 3), 200, np.uint8)
    face = (140, 170, 225)                       # BGR skin tone
    beige = (150, 190, 200)                      # beige hoodie
    cv2.circle(body, (300, 80), 40, face, -1)
    cv2.rectangle(body, (180, 150), (420, 420), beige, -1)
    cv2.rectangle(body, (150, 160), (180, 380), face, -1)   # bare arm on the image left
    body_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(body_mask, (140, 40), (440, 560), 255, -1)
    kp = body_keypoints()
    kp.update({"head_top": (300, 40), "neck": (300, 130)})
    arms = GarmentWarper._arm_mask(body, body_mask, kp)
    assert arms[300, 165] == 1                   # the bare arm is found
    assert arms[300, 200] == 0                   # the beige hoodie is not
