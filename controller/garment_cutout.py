"""Cut a garment out of a DeepFashion photo (where someone is wearing it).

GrabCut is seeded with:
  * the garment bounding box as "probably garment", everything outside as background,
  * the core of the garment (the polygon through its collar/waist and hem
    landmarks, shrunk toward its centre) as "definitely garment",
  * skin-coloured pixels and everything above the collar as "probably background",
    so arms, neck and face are not cut out with the clothes.
"""

import cv2
import numpy as np

CORE_POINTS = {
    1: ["collar_l", "collar_r", "hem_r", "hem_l"],
    2: ["waist_l", "waist_r", "hem_r", "hem_l"],
    3: ["collar_l", "collar_r", "hem_r", "hem_l"],
}


def skin_mask(image):
    """Rough skin detector in YCrCb space (255 = skin)."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, (0, 135, 85), (255, 175, 130))


def _core_polygon(landmarks, clothes_type, shrink=0.65):
    names = [n for n in CORE_POINTS.get(clothes_type, CORE_POINTS[1]) if n in landmarks]
    if len(names) < 3:
        return None
    pts = np.array([landmarks[n] for n in names], np.float32)
    centre = pts.mean(axis=0)
    return (centre + shrink * (pts - centre)).astype(np.int32)


def _clean(mask):
    k = max(3, (min(mask.shape) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    out = np.zeros_like(mask)
    cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return out


def plain_background(image, min_share=0.7, tolerance=12):
    """Background colour if the photo is a product shot on a plain backdrop, else None.

    Uses the dominant border colour, so a thin frame line or a cropped sleeve
    touching the edge doesn't disqualify an otherwise plain backdrop.
    """
    border = np.vstack([image[:4].reshape(-1, 3), image[-4:].reshape(-1, 3),
                        image[:, :4].reshape(-1, 3), image[:, -4:].reshape(-1, 3)]).astype(np.float32)
    bins = (border // 16).astype(np.int32)
    keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    top = values[np.argmax(counts)]
    candidate = np.median(border[keys == top], axis=0)
    close = np.abs(border - candidate).max(axis=1) <= tolerance
    return candidate if close.mean() >= min_share else None


def _cut_from_backdrop(image, bg, tolerance=14):
    diff = np.abs(image.astype(np.float32) - bg[None, None]).max(axis=2)
    mask = (diff > tolerance).astype(np.uint8) * 255
    return _clean(mask)


def cut_out_garment(image, bbox=None, landmarks=None, clothes_type=1, iterations=5):
    """Return a uint8 mask (255 = garment) the same size as ``image``."""
    h, w = image.shape[:2]
    bg = plain_background(image)
    if bg is not None:
        # Flat product shot: whatever isn't backdrop is the garment. More reliable
        # than GrabCut for light garments on white.
        mask = _cut_from_backdrop(image, bg)
        skin_share = (skin_mask(image)[mask > 0] > 0).mean() if mask.any() else 1.0
        if mask.mean() > 255 * 0.05 and skin_share < 0.03:
            return mask   # no person in the photo: a flat product shot
    landmarks = landmarks or {}
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pad_x, pad_y = int(0.03 * (x2 - x1)), int(0.03 * (y2 - y1))
        x1, y1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
        x2, y2 = min(x2 + pad_x, w - 1), min(y2 + pad_y, h - 1)
    else:
        x1, y1, x2, y2 = int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h)
    gc[y1:y2 + 1, x1:x2 + 1] = cv2.GC_PR_FGD

    # Neck/face above the collar (or the body above the waistband) is background.
    top_names = ["waist_l", "waist_r"] if clothes_type == 2 else ["collar_l", "collar_r"]
    tops = [landmarks[n][1] for n in top_names if n in landmarks]
    if tops:
        cut = int(min(tops))
        region = gc[:max(cut, 0)]
        region[region == cv2.GC_PR_FGD] = cv2.GC_PR_BGD
        # Well above the collar is never garment. Without this GrabCut keeps the
        # white backdrop of light-coloured product shots attached to the shoulders.
        hard = int(cut - 0.06 * (y2 - y1))
        if hard > 0:
            gc[:hard] = cv2.GC_BGD

    box = gc == cv2.GC_PR_FGD
    gc[box & (skin_mask(image) > 0)] = cv2.GC_PR_BGD
    if bg is not None:
        # Model photographed on a plain backdrop: the backdrop is certainly not garment.
        diff = np.abs(image.astype(np.float32) - bg[None, None]).max(axis=2)
        gc[diff <= 8] = cv2.GC_BGD

    core = _core_polygon(landmarks, clothes_type)
    if core is not None:
        cv2.fillPoly(gc, [core], cv2.GC_FGD)

    # GrabCut needs both some foreground and some background to model.
    if not np.any((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)):
        return np.zeros((h, w), np.uint8)
    if not np.any((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)):
        gc[:, :2] = gc[:, -2:] = cv2.GC_PR_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean(mask)
