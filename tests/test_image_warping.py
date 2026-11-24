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


# ---------------------------------------------------------------- bomber fixes

def test_back_view_labels_do_not_flip_the_garment():
    _, _, lm = make_tshirt()
    # Back-view photo: DeepFashion's left/right labels swap sides.
    swapped = {k[:-2] + ("_r" if k.endswith("_l") else "_l"): v for k, v in lm.items()}
    src, dst, names = GarmentWarper().compute_correspondences(swapped, body_keypoints())
    d = dict(zip(names, dst))
    assert d["hem_l"][1] > d["collar_l"][1] + 50      # still hangs down from the shoulders
    assert d["collar_l"][0] < d["collar_r"][0]


def test_worn_photo_is_sized_by_hem_corners_not_arms():
    image, mask, lm = make_tshirt()
    # Model's arms hanging beside the shirt make the mask much wider than the shirt.
    cv2.rectangle(mask, (20, 100), (75, 250), 255, -1)
    cv2.rectangle(mask, (165, 100), (220, 250), 255, -1)
    w = GarmentWarper()
    _, dst_flat, names = w.compute_correspondences(lm, body_keypoints(), garment_mask=mask, worn=False)
    _, dst_worn, _ = w.compute_correspondences(lm, body_keypoints(), garment_mask=mask, worn=True)
    i_l, i_r = names.index("hem_l"), names.index("hem_r")
    flat_w = dst_flat[i_r][0] - dst_flat[i_l][0]
    worn_w = dst_worn[i_r][0] - dst_worn[i_l][0]
    assert worn_w > 1.4 * flat_w                         # arms no longer shrink it
    assert 110 < worn_w < 160                           # about the torso width (0.92 * 160)


def test_cutout_drops_trousers_below_the_hem():
    image, truth, lm = make_tshirt()
    rng = np.random.default_rng(5)
    image[:] = rng.integers(60, 200, image.shape, dtype=np.uint8)   # busy, non-plain scene
    image[truth > 0] = (40, 40, 200)
    cv2.rectangle(image, (80, 262), (160, 299), (40, 40, 190), -1) # trousers close in colour
    mask = cut_out_garment(image, (30, 35, 210, 299), lm, clothes_type=1)
    assert mask[285, 120] == 0
    assert mask[200, 120] > 0


def test_body_proportion_check_rejects_keypoints_on_the_face():
    from controller.image_warping import check_body_proportions
    body = np.zeros((300, 205), np.uint8)
    body[:, 30:180] = 255
    bad = {"keypoint_source": "mask", "body_mask": body,
           "keypoints": {"left_shoulder": (102, 22), "right_shoulder": (71, 22),
                         "left_hip": (99, 55), "right_hip": (74, 55)}}
    with pytest.raises(ValueError):
        check_body_proportions(bad)
    good = dict(bad, keypoints={"left_shoulder": (140, 95), "right_shoulder": (70, 95),
                                "left_hip": (130, 190), "right_hip": (80, 190)})
    check_body_proportions(good)


def test_head_is_never_covered_when_face_was_found():
    body = np.full((600, 600, 3), 200, np.uint8)
    garment = np.zeros_like(body)
    mask = np.full((600, 600), 255, np.uint8)          # garment everywhere, including the head
    kp = body_keypoints()
    out = GarmentWarper(shading=0, keep_arms=False).blend(body, garment, mask, None, kp,
                                                          protect_head=True)
    assert out[60, 300].tolist() == [200, 200, 200]    # head untouched
    assert out[300, 300].tolist() == [0, 0, 0]         # torso covered



def test_side_on_worn_photo_is_not_blown_up():
    # Side-on model: hem corners almost touch, which used to make the garment huge.
    lm = {"collar_l": (110, 40), "collar_r": (130, 40), "hem_l": (112, 220), "hem_r": (128, 220)}
    _, dst, names = GarmentWarper().compute_correspondences(lm, body_keypoints(), worn=True)
    d = dict(zip(names, dst))
    torso = 220                                          # shoulder (150) to hip (370)
    garment_len = d["hem_l"][1] - d["collar_l"][1]
    assert 0.7 * torso - 1 <= garment_len <= 1.5 * torso + 1


def test_arms_only_kept_in_front_with_elbow_keypoints():
    h = w = 600
    body = np.full((h, w, 3), 200, np.uint8)
    cv2.circle(body, (300, 80), 40, (140, 170, 225), -1)
    cv2.rectangle(body, (150, 160), (180, 380), (140, 170, 225), -1)
    body_mask = np.full((h, w), 255, np.uint8)
    garment = np.zeros_like(body)
    mask = np.full((h, w), 255, np.uint8)
    kp = body_keypoints()
    kp.update({"head_top": (300, 40), "neck": (300, 130)})
    warper = GarmentWarper(shading=0)
    out = warper.blend(body, garment, mask, body_mask, kp)
    assert out[300, 165].tolist() == [0, 0, 0]           # no elbows: garment covers the arm
    kp.update({"right_elbow": (165, 270), "left_elbow": (435, 270)})
    out = warper.blend(body, garment, mask, body_mask, kp)
    assert out[300, 165].tolist() != [0, 0, 0]           # elbows known: arm stays in front


def test_quality_check_flags_bad_cutouts():
    from controller.garment_cutout import garment_quality
    image, truth, lm = make_tshirt()
    assert garment_quality(image, truth, lm)[0]
    assert not garment_quality(image, np.full_like(truth, 255), lm)[0]      # grabbed everything
    assert not garment_quality(image, np.zeros_like(truth), lm)[0]          # empty
    pale = image.copy()
    pale[:] = 250
    pale[truth > 0] = 238                                                     # white-on-white
    assert garment_quality(pale, truth, lm)[1] == "garment blends into the background"


def make_long_sleeve_shirt(h=300, w=260):
    """A red long-sleeved product shot: sleeves slant down to cuffs by the hem."""
    rng = np.random.default_rng(4)
    image = rng.integers(120, 140, (h, w, 3), dtype=np.uint8)
    poly = np.array([[105, 40], [155, 40], [195, 60], [235, 230], [210, 238], [178, 110],
                     [175, 260], [85, 260], [82, 110], [50, 238], [25, 230], [65, 60]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    image[mask > 0] = (40, 40, 200)
    lm = {"collar_l": (115, 42), "collar_r": (145, 42), "sleeve_l": (37, 234),
          "sleeve_r": (222, 234), "hem_l": (88, 258), "hem_r": (172, 258)}
    return Garment(0, "synthetic", "Shirt", {}, clothes_type=1, landmarks=lm,
                   loader=lambda _: image, mask=mask)


def arms_out_keypoints():
    """T pose: arms straight out sideways at shoulder height."""
    kp = body_keypoints()
    kp.update({"head_top": (300, 40), "neck": (300, 130),
               "right_elbow": (100, 165), "left_elbow": (500, 165),
               "right_wrist": (20, 170), "left_wrist": (580, 170)})
    return kp


def covered(mask, a, b, n=9):
    """Share of the points on segment a-b where the warped garment is."""
    pts = [np.asarray(a) + t * (np.asarray(b) - np.asarray(a)) for t in np.linspace(0.15, 0.85, n)]
    return np.mean([mask[int(p[1]), int(p[0])] > 127 for p in pts])


def test_long_sleeves_swing_onto_outstretched_arms():
    kp = arms_out_keypoints()
    warper = GarmentWarper()
    _, mask = warper.warp(make_long_sleeve_shirt(), kp, (600, 600, 3))
    for side in ("right", "left"):
        assert covered(mask, kp[f"{side}_shoulder"], kp[f"{side}_elbow"]) > 0.8, side
        assert covered(mask, kp[f"{side}_elbow"], kp[f"{side}_wrist"]) > 0.8, side
    # The cuffs end at the wrists.
    d = dict(zip(warper.last["names"], warper.last["warped_points"]))
    assert np.linalg.norm(d["sleeve_l"] - kp["right_wrist"]) < 10      # sleeve_l is image-left
    assert np.linalg.norm(d["sleeve_r"] - kp["left_wrist"]) < 10
    # Nothing is left hanging where the sleeves were in the catalog photo.
    assert mask[330, 205] == 0 and mask[330, 395] == 0


def test_moving_the_sleeves_leaves_the_torso_alone():
    kp = arms_out_keypoints()
    still = {k: (None if "elbow" in k or "wrist" in k else v) for k, v in kp.items()}
    warper = GarmentWarper()
    warper.warp(make_long_sleeve_shirt(), still, (600, 600, 3))
    before = dict(zip(warper.last["names"], warper.last["warped_points"]))
    warper.warp(make_long_sleeve_shirt(), kp, (600, 600, 3))
    after = dict(zip(warper.last["names"], warper.last["warped_points"]))
    for name in ("collar_l", "collar_r", "hem_l", "hem_r"):
        assert np.linalg.norm(before[name] - after[name]) < 2, name


def test_long_sleeve_bends_at_the_elbow():
    kp = body_keypoints()
    kp.update({"right_elbow": (190, 260), "left_elbow": (410, 260),      # elbows out...
               "right_wrist": (120, 200), "left_wrist": (480, 200)})     # ...forearms raised
    warper = GarmentWarper()
    _, mask = warper.warp(make_long_sleeve_shirt(), kp, (600, 600, 3))
    for side in ("right", "left"):
        assert covered(mask, kp[f"{side}_elbow"], kp[f"{side}_wrist"]) > 0.8, side
    # The straight-on line from the upper arm (where an unbent sleeve would go) stays clear.
    assert mask[340, 140] == 0 and mask[340, 460] == 0


def test_moved_sleeve_covers_the_bare_arm_but_not_the_hand():
    h = w = 600
    skin = (140, 170, 225)
    body = np.full((h, w, 3), 210, np.uint8)
    cv2.circle(body, (300, 80), 40, skin, -1)                              # face: skin reference
    kp = arms_out_keypoints()
    for side in ("right", "left"):
        cv2.line(body, tuple(kp[f"{side}_shoulder"]), tuple(kp[f"{side}_wrist"]), skin, 26)
    cv2.circle(body, (5, 170), 14, skin, -1)                               # a hand past the cuff
    seg = {"keypoints": kp, "body_mask": np.full((h, w), 255, np.uint8), "keypoint_source": "face+mask"}
    out = GarmentWarper(shading=0).try_on(body, seg, make_long_sleeve_shirt())
    upper_arm = (np.array(kp["right_shoulder"]) + np.array(kp["right_elbow"])) // 2
    b, g, r = out[upper_arm[1], upper_arm[0]].astype(int)
    assert r > 150 and b < 100                                             # red sleeve, not skin
    assert out[170, 5].tolist() == list(skin)                              # the hand still shows
