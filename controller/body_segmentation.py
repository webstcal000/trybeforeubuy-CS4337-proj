"""Step 1: detect and segment the body region of the uploaded image with OpenCV."""

import os
import cv2
import numpy as np

# OpenPose COCO body model: heatmap index -> keypoint name.
OPENPOSE_COCO_POINTS = {
    0: "nose", 1: "neck",
    2: "right_shoulder", 3: "right_elbow", 4: "right_wrist",
    5: "left_shoulder", 6: "left_elbow", 7: "left_wrist",
    8: "right_hip", 9: "right_knee", 10: "right_ankle",
    11: "left_hip", 12: "left_knee", 13: "left_ankle",
    14: "right_eye", 15: "left_eye", 16: "right_ear", 17: "left_ear",
}
KEYPOINT_NAMES = ["head_top"] + list(OPENPOSE_COCO_POINTS.values())
DEFAULT_POSE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pose", "coco")

class BodySegmenter:
    """Finds the person in an image and produces a body mask and keypoints."""

    def __init__(self, work_height=512, grabcut_iters=5, pose_model_dir=DEFAULT_POSE_DIR, keypoint_threshold=0.1):
        self.work_height = work_height
        self.grabcut_iters = grabcut_iters
        self.keypoint_threshold = keypoint_threshold

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

        # The OpenPose model is optional (~200 MB); without it keypoints are estimated.
        self.pose_net = None
        proto = os.path.join(pose_model_dir, "pose_deploy_linevec.prototxt")
        weights = os.path.join(pose_model_dir, "pose_iter_440000.caffemodel")
        if os.path.exists(proto) and os.path.exists(weights):
            self.pose_net = cv2.dnn.readNetFromCaffe(proto, weights)

        self.scale = 1.0  # working size / original size, set by preprocess()
        self.face = None  # (x, y, w, h) in working pixels, set by detect_person()
    
    def preprocess(self, image):
        """Convert to 3-channel BGR, resize to the working height and denoise lightly."""
        if image is None:
            raise ValueError("image is None (bad path or unreadable file?)")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        self.scale = self.work_height / image.shape[0]
        work = cv2.resize(image, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)
        return cv2.bilateralFilter(work, d=5, sigmaColor=50, sigmaSpace=50)

    def detect_person(self, image):
        """Return the (x, y, w, h) box of the main person.
        Tries the HOG people detector first, then a box grown from a detected
        face, then a centered box (DeepFashion photos are tightly cropped).
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                    minSize=(w // 20, w // 20))
        self.face = tuple(max(faces, key=lambda f: f[2] * f[3])) if len(faces) else None

        # Pad so HOG can find people whose bodies touch the image border.
        pad = h // 4
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        boxes, weights = self.hog.detectMultiScale(padded, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(boxes):
            best = int(np.argmax(np.asarray(weights).ravel() * boxes[:, 2] * boxes[:, 3]))
            x, y, bw, bh = boxes[best]
            return self._clip_box((x - pad, y - pad, bw, bh), w, h)

        if self.face is not None:
            fx, fy, fw, fh = self.face
            cx = fx + fw / 2
            return self._clip_box((cx - 2.5 * fw, fy - 0.6 * fh, 5 * fw, 8 * fh), w, h)

        return self._clip_box((0.1 * w, 0.02 * h, 0.8 * w, 0.98 * h), w, h)


    def segment_body(self, image, bounding_box):
        """Separate the person from the background with GrabCut seeded by the box."""
        h, w = image.shape[:2]
        x, y, bw, bh = bounding_box

        mask = np.full((h, w), cv2.GC_BGD, np.uint8)
        mask[y:y + bh, x:x + bw] = cv2.GC_PR_FGD
        # Pixels on the top/left/right image edge are probably background
        # (the bottom edge is left alone: bodies are often cut off there).
        for edge in (mask[0, :], mask[:, 0], mask[:, -1]):
            edge[edge == cv2.GC_PR_FGD] = cv2.GC_PR_BGD

        # Seed a sure-foreground core so GrabCut has something to anchor on.
        cx = x + bw // 2
        mask[y + bh // 4:y + 3 * bh // 4, max(cx - bw // 12, 0):cx + bw // 12 + 1] = cv2.GC_FGD
        if self.face is not None:
            fx, fy, fw, fh = self.face
            mask[fy + fh // 4:fy + 3 * fh // 4, fx + fw // 4:fx + 3 * fw // 4] = cv2.GC_FGD

        # GrabCut needs at least some background pixels to model.
        if not np.any((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)):
            mask[:, :2] = mask[:, -2:] = cv2.GC_PR_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, self.grabcut_iters, cv2.GC_INIT_WITH_MASK)

        body = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return self.refine_mask(body)
    
    def refine_mask(self, mask):
        """Clean the mask: remove specks, close gaps, keep the largest blob, fill holes."""
        k = max(3, (mask.shape[0] // 100) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return mask
        refined = np.zeros_like(mask)
        cv2.drawContours(refined, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)

        # Smooth jagged edges.
        refined = cv2.GaussianBlur(refined, (k, k), 0)
        return np.where(refined > 127, 255, 0).astype(np.uint8)
    def detect_keypoints(self, image, body_mask=None):
        """Return ({name: (x, y)}, source) in working-image pixels.

        Uses OpenPose when its model files are present, otherwise estimates the
        torso keypoints from the face box and the body mask.
        """
        if self.pose_net is not None:
            points = self._openpose_keypoints(image)
            if {"left_shoulder", "right_shoulder"} <= points.keys():
                if body_mask is not None and "head_top" not in points:
                    head = self._estimate_keypoints(body_mask).get("head_top")
                    if head:
                        points["head_top"] = head
                return points, "openpose"
        if body_mask is None:
            return {}, "mask"
        return self._estimate_keypoints(body_mask), ("face+mask" if self.face is not None else "mask")
    def segment_torso(self, body_mask, keypoints=None):
        """Return the part of the body mask between the shoulders and the hips."""
        h, w = body_mask.shape
        torso = np.zeros_like(body_mask)
        needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

        if keypoints and all(keypoints.get(k) is not None for k in needed):
            ls, rs, lh, rh = (np.array(keypoints[k], np.float32) for k in needed)
            # Widen the polygon a bit so loose clothing at the sides is included.
            spread = 0.15 * np.linalg.norm(ls - rs)
            top_l, top_r = self._push_apart(ls, rs, spread)
            bot_l, bot_r = self._push_apart(lh, rh, spread)
            neck = keypoints.get("neck")
            if neck is None:
                neck = (ls + rs) / 2
            top_l[1] = top_r[1] = min(top_l[1], top_r[1], neck[1])
            polygon = np.array([top_l, top_r, bot_r, bot_l], np.int32)
            cv2.fillPoly(torso, [polygon], 255)
        else:
            ys = np.where(body_mask.any(axis=1))[0]
            if len(ys) == 0:
                return torso
            top, height = ys[0], ys[-1] - ys[0]
            torso[int(top + 0.2 * height):int(top + 0.6 * height), :] = 255

        return cv2.bitwise_and(torso, body_mask)

    def run(self, image):
        """Full pipeline. Takes a BGR image (as from cv2.imread); see the module docstring for the output."""
        orig_h, orig_w = image.shape[:2]
        work = self.preprocess(image)
        bbox = self.detect_person(work)
        body = self.segment_body(work, bbox)
        keypoints, source = self.detect_keypoints(work, body)
        torso = self.segment_torso(body, keypoints)

        s = 1.0 / self.scale
        return {
            "bbox": tuple(int(round(v * s)) for v in bbox),
            "body_mask": cv2.resize(body, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST),
            "torso_mask": cv2.resize(torso, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST),
            "keypoints": {k: (int(round(keypoints[k][0] * s)), int(round(keypoints[k][1] * s)))
                          if k in keypoints else None for k in KEYPOINT_NAMES},
            "keypoint_source": source,
        }
    @staticmethod
    def _clip_box(box, w, h):
        x, y, bw, bh = box
        x0, y0 = int(max(x, 0)), int(max(y, 0))
        x1, y1 = int(min(x + bw, w)), int(min(y + bh, h))
        return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)

    @staticmethod
    def _push_apart(a, b, amount):
        direction = (a - b) / (np.linalg.norm(a - b) + 1e-6)
        return a + direction * amount, b - direction * amount

    @staticmethod
    def _row_extent(mask, y, cx):
        """Left/right edge of the mask run containing cx on row y (or the whole row as fallback)."""
        row = mask[int(np.clip(y, 0, mask.shape[0] - 1))] > 0
        xs = np.where(row)[0]
        if len(xs) == 0:
            return None
        cx = int(np.clip(cx, 0, len(row) - 1))
        if not row[cx]:
            return xs[0], xs[-1]
        left = cx
        while left > 0 and row[left - 1]:
            left -= 1
        right = cx
        while right < len(row) - 1 and row[right + 1]:
            right += 1
        return left, right

    def _openpose_keypoints(self, image):
        h, w = image.shape[:2]
        in_w = int(368 * w / h) // 8 * 8
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (in_w, 368), (0, 0, 0), swapRB=False, crop=False)
        self.pose_net.setInput(blob)
        out = self.pose_net.forward()
        points = {}
        for idx, name in OPENPOSE_COCO_POINTS.items():
            heatmap = out[0, idx]
            _, conf, _, (px, py) = cv2.minMaxLoc(heatmap)
            if conf > self.keypoint_threshold:
                points[name] = (w * px / heatmap.shape[1], h * py / heatmap.shape[0])
        return points

    def _estimate_keypoints(self, body_mask):
        """Rough torso keypoints from body proportions (head height as the unit)."""
        ys, xs = np.where(body_mask > 0)
        if len(ys) == 0:
            return {}
        top, bottom = ys.min(), ys.max()
        h = body_mask.shape[0]

        if self.face is not None:
            fx, fy, fw, fh = self.face
            cx = fx + fw / 2
            head_h = fh * 1.25
            chin_y = fy + fh
        else:
            # Find the neck as the narrowest row in the upper part of the mask.
            cx = float(np.median(xs[ys < top + 0.3 * (bottom - top)]))
            widths = (body_mask > 0).sum(axis=1).astype(np.float32)
            lo, hi = int(top + 0.05 * (bottom - top)), int(top + 0.3 * (bottom - top))
            if hi - lo < 3:
                return {}
            neck_row = lo + int(np.argmin(widths[lo:hi] + (widths[lo:hi] == 0) * 1e9))
            head_h = max(neck_row - top, 1) * 1.05
            chin_y = neck_row

        neck_y = chin_y + 0.25 * head_h
        shoulder_y = chin_y + 0.5 * head_h
        hip_y = min(shoulder_y + 2.1 * head_h, bottom)
        max_half = 1.0 * head_h  # shoulders are ~2 head heights wide; stops arms widening them

        points = {"head_top": (cx, float(top)), "neck": (cx, neck_y)}
        ext = self._row_extent(body_mask, shoulder_y, cx)
        if ext:
            half = min(max(cx - ext[0], ext[1] - cx), max_half)
            # Image-left is the person's right.
            points["right_shoulder"] = (cx - half, shoulder_y)
            points["left_shoulder"] = (cx + half, shoulder_y)
        if hip_y < h:
            ext = self._row_extent(body_mask, hip_y, cx)
            if ext:
                half = min(max(cx - ext[0], ext[1] - cx), 0.8 * max_half)
                points["right_hip"] = (cx - half, hip_y)
                points["left_hip"] = (cx + half, hip_y)
        return points


def draw_debug(image, result):
    """Overlay the body mask, torso mask, bbox and keypoints for visual checking."""
    out = image.copy()
    tint = np.zeros_like(out)
    tint[result["body_mask"] > 0] = (0, 160, 0)
    tint[result["torso_mask"] > 0] = (0, 0, 200)
    out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
    x, y, w, h = result["bbox"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 2)
    r = max(3, image.shape[0] // 150)
    for name, point in result["keypoints"].items():
        if point is None:
            continue
        px, py = point
        cv2.circle(out, (px, py), r, (0, 255, 255), -1)
        cv2.putText(out, name, (px + r, py - r), cv2.FONT_HERSHEY_SIMPLEX, r / 10, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "keypoints: " + result["keypoint_source"], (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out

    


