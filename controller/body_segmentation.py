"""Step 1: detect and segment the body region of the uploaded image with OpenCV.

Output of BodySegmenter.run(image) -- this is the contract the warper relies on:

    {
        "bbox":            (x, y, w, h) of the person, in original image pixels
        "body_mask":       uint8 HxW, 255 = person, 0 = background
        "torso_mask":      uint8 HxW, 255 = torso (shoulders down to hips)
        "keypoints":       {name: (x, y) or None} in original image pixels; every name
                           in KEYPOINT_NAMES is present, None = not detected
        "keypoint_source": "openpose" | "face+mask" | "mask"
    }

Keypoint left/right is the PERSON's left/right (OpenPose convention), so for a
person facing the camera "left_shoulder" appears on the right side of the image.
The shoulders and hips are filled whenever a body is found. The estimated
fallback ("face+mask" / "mask") produces head_top, neck, the shoulders and hips,
and elbows and wrists traced along the arms in the body outline (see add_arms).
An arm that can't be seen in the outline -- held in front of the body, or a hand
out of the frame -- gets None rather than a guess. Knees, ankles and the face
points are None unless OpenPose is installed.
"""

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

    def __init__(self, work_height=512, grabcut_iters=5, pose_model_dir=DEFAULT_POSE_DIR,
                 keypoint_threshold=0.1):
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

    # ------------------------------------------------------------------ steps

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
        # Arms held out sideways or overhead reach past the person box, so the
        # area around it is only probably background. As sure background, the
        # arms' skin would also teach GrabCut that skin is background, and the
        # head would be cut off as well.
        px, py = int(0.4 * bw), int(0.15 * bh)
        mask[max(y - py, 0):y + bh + py, max(x - px, 0):x + bw + px] = cv2.GC_PR_BGD
        mask[y:y + bh, x:x + bw] = cv2.GC_PR_FGD
        # Pixels on the top/left/right image edge are probably background
        # (the bottom edge is left alone: bodies are often cut off there).
        for edge in (mask[0, :], mask[:, 0], mask[:, -1]):
            edge[edge == cv2.GC_PR_FGD] = cv2.GC_PR_BGD

        # Seed a sure-foreground core so GrabCut has something to anchor on.
        cx = x + bw // 2
        mask[y + bh // 4:y + 3 * bh // 4, max(cx - bw // 12, 0):cx + bw // 12 + 1] = cv2.GC_FGD
        # A thin line up through the neck to the head: without a detected face the
        # head (and so skin colour) is otherwise often lost to the background.
        thin = max(bw // 60, 1)
        mask[y + int(0.15 * bh):y + bh // 4, max(cx - thin, 0):cx + thin + 1] = cv2.GC_FGD
        if self.face is not None:
            fx, fy, fw, fh = self.face
            mask[fy + fh // 4:fy + 3 * fh // 4, fx + fw // 4:fx + 3 * fw // 4] = cv2.GC_FGD

        # GrabCut needs at least some background pixels to model.
        if not np.any((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)):
            mask[:, :2] = mask[:, -2:] = cv2.GC_PR_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(0)   # GrabCut starts from a random clustering; same photo, same result
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

    def add_arms(self, image, body_mask, keypoints, source="face+mask"):
        """Fill in elbows and wrists that the pose model didn't give, from the outline.

        Bare arms are thin and skin-coloured, and GrabCut often drops them, so
        first pixels of the person's own skin colour that connect to the body are
        added back to the mask. `source` is how the torso keypoints were found
        (see detect_keypoints); each way puts them at a different scale.
        Returns (body_mask, keypoints).
        """
        body_mask = self._recover_skin(image, body_mask, keypoints)
        keypoints = dict(keypoints)
        for name, point in self._estimate_arms(body_mask, keypoints, source).items():
            keypoints.setdefault(name, point)
        return body_mask, keypoints

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
        body, keypoints = self.add_arms(work, body, keypoints, source)
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

    # ---------------------------------------------------------------- helpers

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

    @staticmethod
    def _recover_skin(image, body_mask, points):
        """Add skin-coloured regions (bare arms, hands) that touch the body mask.

        Skin colour is sampled from the face, between the top of the head and the
        neck. If the additions would grow the body by more than a third, the
        match is probably the background (a beige wall, wood) and nothing is added.
        """
        needed = ("head_top", "neck", "right_shoulder", "left_shoulder")
        if any(points.get(k) is None for k in needed):
            return body_mask
        h, w = body_mask.shape
        head, neck = points["head_top"], points["neck"]
        span = abs(points["left_shoulder"][0] - points["right_shoulder"][0])
        y0, y1 = int(head[1] + 0.45 * (neck[1] - head[1])), int(neck[1])
        x0, x1 = int(max(neck[0] - 0.12 * span, 0)), int(min(neck[0] + 0.12 * span, w))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        face = lab[max(y0, 0):max(y1, 0), x0:x1][body_mask[max(y0, 0):max(y1, 0), x0:x1] > 0]
        if len(face) < 30:
            return body_mask
        ref = np.median(face, axis=0)
        skin = ((np.linalg.norm(lab[..., 1:] - ref[1:], axis=2) < 9)
                & (np.abs(lab[..., 0] - ref[0]) < 60)).astype(np.uint8) * 255
        k = max(3, (h // 100) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
        # Above the neck only raised arms are wanted: leave the head's column alone.
        skin[:int(neck[1]), int(max(neck[0] - 0.35 * span, 0)):int(neck[0] + 0.35 * span)] = 0

        merged = cv2.morphologyEx(body_mask | skin, cv2.MORPH_CLOSE, kernel)
        n, labels = cv2.connectedComponents(merged)
        on_body = np.unique(labels[body_mask > 0])
        grown = np.isin(labels, on_body[on_body > 0]) | (body_mask > 0)
        added = grown.sum() - (body_mask > 0).sum()
        if added <= 0 or added > 0.33 * (body_mask > 0).sum():
            return body_mask
        return grown.astype(np.uint8) * 255

    # Typical adult arm: shoulder-elbow 33 cm, elbow-wrist 26 cm, hand 19 cm.
    ELBOW_AT, WRIST_AT = 33 / 78, 59 / 78     # fractions of shoulder -> fingertips

    # Upper-arm length (33 cm) per unit of shoulder-to-hip distance and of shoulder
    # width, for each way the torso points are found. With the face estimate the
    # hips are 2.1 head heights below the shoulders and the shoulders at most 2
    # head heights wide; the outline-only estimate measures the head down to the
    # middle of the neck, so all its distances come out ~1.3x longer; OpenPose
    # marks the joints themselves (shoulder joint to hip joint ~50 cm, ~35 cm apart).
    ARM_SCALE = {"face+mask": (0.81, 0.85), "mask": (0.62, 0.65), "openpose": (0.66, 0.94)}

    def _estimate_arms(self, body_mask, points, source="face+mask"):
        """Elbows and wrists traced along the arms in the body outline.

        Each upper arm is the direction from the shoulder that stays inside the
        outline longest without entering the torso or head; the forearm is
        searched the same way from the elbow, so bent arms are followed. An arm
        that can't be seen in the outline (held in front of the body, or out of
        the frame) gets no points rather than a guess.
        """
        needed = ("head_top", "neck", "right_shoulder", "left_shoulder", "right_hip", "left_hip")
        if any(points.get(k) is None for k in needed):
            return {}
        sh = {"right": np.array(points["right_shoulder"], np.float64),   # image left
              "left": np.array(points["left_shoulder"], np.float64)}     # image right
        hip = {"right": np.array(points["right_hip"], np.float64),
               "left": np.array(points["left_hip"], np.float64)}
        span = sh["left"][0] - sh["right"][0]
        torso_len = (hip["right"][1] + hip["left"][1] - sh["right"][1] - sh["left"][1]) / 2
        rows = np.nonzero(body_mask.any(axis=1))[0]
        if span < 10 or len(rows) == 0 or torso_len < 0.18 * (rows[-1] - rows[0] + 1):
            return {}           # the torso estimate itself is implausible (see check_body_proportions)
        # Nominal lengths, only used where the arm's own length can't be measured
        # (it runs out of the frame or into the legs).
        per_torso, per_span = self.ARM_SCALE.get(source, self.ARM_SCALE["face+mask"])
        upper = 0.5 * (per_torso * torso_len + per_span * span)
        fore, hand = upper * 26 / 33, upper * 19 / 33
        total = upper + fore + hand
        arm_w = 0.2 * span

        # Torso and head, where an arm can't be told apart from the body. A
        # forearm crossing in front of the torso (hand on hip) isn't followed:
        # its wrist is left out rather than guessed.
        h, w = body_mask.shape
        blocked = np.zeros((h, w), np.uint8)
        inset, hip_inset = 0.12 * span, 0.05 * (hip["left"][0] - hip["right"][0])
        core = np.array([sh["right"] + (inset, 0), sh["left"] - (inset, 0),
                         hip["left"] - (hip_inset, 0), hip["right"] + (hip_inset, 0)], np.int32)
        cv2.fillPoly(blocked, [core], 255)
        # ...and the legs below the hips, which hanging hands sit beside.
        cv2.rectangle(blocked, (int(core[3][0]), int(min(core[2][1], core[3][1]))),
                      (int(core[2][0]), h - 1), 255, -1)
        cx = (sh["right"][0] + sh["left"][0]) / 2
        top = min(sh["right"][1], sh["left"][1])
        cv2.rectangle(blocked, (int(cx - 0.28 * span), 0), (int(cx + 0.28 * span), int(top)), 255, -1)
        free = (body_mask > 0) & (blocked == 0)

        out = {}
        for side, outward in (("right", -1.0), ("left", 1.0)):
            pivot = sh[side] - (outward * 0.1 * span, 0)
            # Upper arm: angles from straight down (0) through sideways (90) to up (180).
            angles = np.radians(np.arange(-25, 181, 3))
            dirs = np.stack([outward * np.sin(angles), np.cos(angles)], axis=1)
            d_up, score = self._best_direction(free, pivot, dirs, upper, arm_w)
            if d_up is None or score < 0.6:
                continue
            # A real arm has width. Without one, the search settles on the thin
            # strip of torso or leg just outside the blocked area.
            if self._limb_width(free, pivot, d_up, upper, arm_w) < 0.1 * span:
                continue
            reach, left_frame = self._reach(free, pivot, d_up, 1.3 * total)
            end = reach - 0.5 * arm_w          # where the straight part of the arm ends
            if end < 0.5 * upper:
                continue

            elbow = wrist = None
            if not left_frame and 0.7 * upper <= end <= 1.4 * upper:
                # The straight run may end at a bent elbow.
                bent = self._forearm(free, pivot, pivot + d_up * end, d_up, fore, hand, arm_w)
                if bent is not None:
                    elbow, wrist = pivot + d_up * end, bent[1]
            if elbow is None:
                elbow_d = min(upper, end)
                elbow = pivot + d_up * elbow_d
                found = self._forearm(free, pivot, elbow, d_up, fore, hand, arm_w, any_bend=True)
                wrist = found[1] if found is not None else None
                # No forearm found: if the arm still runs on well past any plausible
                # elbow, the nominal elbow was too far out on a straight arm; if it
                # stops about there, the forearm is just hidden (a hand on the hip).
                straight = (np.dot(found[0], d_up) > np.cos(np.radians(15)) if found is not None
                            else not left_frame and reach >= 1.3 * upper)
                if straight:
                    # A straight arm: if it ends in the frame, its own length beats
                    # the nominal one (people and photos vary). The forearm's reach
                    # is measured along the forearm, which follows the arm better.
                    d_fore, _, fore_reach, fore_left_frame = found or (d_up, None, 0.0, False)
                    # The straight run along the upper arm is a lower bound too.
                    length = max(elbow_d + fore_reach, reach)
                    if not fore_left_frame and upper < length < 1.25 * total:
                        elbow = pivot + d_up * (self.ELBOW_AT * length)
                        wrist = elbow + d_fore * ((self.WRIST_AT - self.ELBOW_AT) * length)

            for name, p in (("elbow", elbow), ("wrist", wrist)):
                if p is not None and 0 <= p[0] < w and 0 <= p[1] < h:
                    out[f"{side}_{name}"] = (float(p[0]), float(p[1]))
                elif name == "elbow":
                    break               # no wrist without its elbow
        return out

    def _forearm(self, free, pivot, elbow, d_up, fore, hand, arm_w, any_bend=False):
        """Search for a forearm leaving the elbow.

        Returns (direction, wrist, reach, left_frame) or None; reach is how far the
        forearm and hand run from the elbow, left_frame whether the edge stopped it.

        Unless any_bend, only a clear bend counts (35 degrees or more, with the
        forearm plainly visible); a forearm straight on from the upper arm is the
        caller's business.
        """
        h, w = free.shape
        base = np.arctan2(d_up[1], d_up[0])
        bends = np.radians(np.arange(-150, 151, 5))
        dirs = np.stack([np.cos(base + bends), np.sin(base + bends)], axis=1)
        # Don't let a folded forearm be found lying on the upper arm itself.
        near_upper = np.zeros((h, w), np.uint8)
        cv2.line(near_upper, tuple(int(v) for v in pivot), tuple(int(v) for v in elbow),
                 255, max(int(1.2 * arm_w), 3))
        free_fore = free & (near_upper == 0)
        d_fore, score = self._best_direction(free_fore, elbow, dirs, fore, arm_w,
                                             prefer=np.abs(np.degrees(bends)) * 0.0015)
        if d_fore is None or score < 0.6:
            return None
        d_fore = self._centre_line(free_fore, elbow, d_fore, fore, arm_w)
        bend = np.degrees(np.arccos(np.clip(np.dot(d_fore, d_up), -1, 1)))
        if not any_bend and (bend < 35 or score < 0.75):
            return None
        # Traced along the middle of the forearm (on the full outline: the elbow
        # itself sits inside near_upper), so a slightly-off direction doesn't
        # slip out of a thin forearm early.
        reach, left_frame = self._trace(free, elbow, d_fore, 1.3 * (fore + hand), arm_w)
        if reach < 0.3 * fore:
            return None
        if left_frame or reach >= 1.25 * (fore + hand):
            wrist_d = fore
        else:
            wrist_d = reach * 26 / 45            # forearm share of forearm + hand
        return d_fore, elbow + d_fore * wrist_d, reach, left_frame

    @staticmethod
    def _limb_width(free, start, d, length, arm_w):
        """Median width of the run of `free` pixels across the ray, 30-90% along it."""
        h, w = free.shape
        n = np.array([-d[1], d[0]])
        offs = np.arange(-2.0, 2.0001, 0.02) * arm_w
        c = len(offs) // 2
        widths = []
        for t in np.linspace(0.3, 0.9, 7) * length:
            pts = start + t * d + offs[:, None] * n
            x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            on = np.zeros(len(offs), bool)
            on[ok] = free[y[ok], x[ok]]
            if not on[c]:
                widths.append(0.0)
                continue
            lo = hi = c
            while lo > 0 and on[lo - 1]:
                lo -= 1
            while hi < len(on) - 1 and on[hi + 1]:
                hi += 1
            widths.append(offs[hi] - offs[lo])
        return float(np.median(widths))

    @staticmethod
    def _centre_line(free, start, d, length, arm_w):
        """Turn a limb direction from start towards the middle of the limb.

        The best ray can run along one edge of a limb. Across the ray at several
        distances, the middle of the limb's run of pixels is found; the returned
        direction points from start through their centre (start stays put).
        """
        h, w = free.shape
        n = np.array([-d[1], d[0]])
        offs = np.arange(-1.0, 1.0001, 0.05) * arm_w
        mids = []
        for t in np.linspace(0.3, 0.95, 10) * length:
            p = start + t * d
            pts = p + offs[:, None] * n
            x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            on = np.zeros(len(offs), bool)
            on[ok] = free[y[ok], x[ok]]
            if not on.any():
                continue
            # The run of limb pixels nearest the ray.
            idx = np.nonzero(on)[0]
            c = idx[np.argmin(np.abs(offs[idx]))]
            lo = hi = c
            while lo > 0 and on[lo - 1]:
                lo -= 1
            while hi < len(on) - 1 and on[hi + 1]:
                hi += 1
            if lo == 0 or hi == len(on) - 1:
                continue        # the run is wider than an arm here (merged with the body)
            mids.append(p + n * (offs[lo] + offs[hi]) / 2)
        if len(mids) < 4:
            return d
        new_d = np.mean(mids, axis=0) - start
        new_d /= max(np.linalg.norm(new_d), 1e-6)
        if np.dot(new_d, d) < np.cos(np.radians(20)):
            return d            # the fit disagrees wildly; trust the search
        return new_d

    @staticmethod
    def _best_direction(free, start, dirs, length, arm_w, prefer=None):
        """The direction from start whose arm-wide strip lies most inside `free`.

        Returns (unit direction, score in 0..1); score is the inside fraction minus
        any `prefer` penalty. The middle of the best-scoring run of angles is used,
        so a wide arm doesn't pull the answer to one edge.
        """
        h, w = free.shape
        ts = np.linspace(0.3, 0.95, 12) * length
        offsets = np.array([-0.25, 0.0, 0.25]) * arm_w
        scores = np.zeros(len(dirs))
        for i, d in enumerate(dirs):
            n = np.array([-d[1], d[0]])
            pts = start + ts[:, None, None] * d + offsets[None, :, None] * n
            pts = pts.reshape(-1, 2)
            x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if inside.mean() < 0.5:
                continue
            scores[i] = free[y[inside], x[inside]].mean()
        if prefer is not None:
            scores = scores - prefer
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            return None, 0.0
        lo = hi = best
        while lo > 0 and scores[lo - 1] >= scores[best] - 0.05:
            lo -= 1
        while hi < len(scores) - 1 and scores[hi + 1] >= scores[best] - 0.05:
            hi += 1
        d = dirs[lo:hi + 1].mean(axis=0)
        return d / max(np.linalg.norm(d), 1e-6), float(scores[best])

    @staticmethod
    def _trace(free, start, d, limit, arm_w):
        """Follow a limb from start, re-centring on it at every step.

        Like _reach, returns (distance, left_frame), but tolerates a direction
        that is a few degrees off. Where the limb merges with something wider
        (the body, the legs) it just carries straight on.
        """
        h, w = free.shape
        step = 2.0
        offs = np.arange(-1.0, 1.0001, 0.1) * arm_w
        p, d = np.asarray(start, np.float64), np.asarray(d, np.float64)
        last_p = p
        travelled, last_in, gap = 0.0, 0.0, 0
        gap_allowed = max(2, int(0.04 * limit / step))
        while travelled < limit:
            p = p + d * step
            travelled += step
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < w and 0 <= y < h):
                return last_in, True
            if not free[y, x]:
                gap += 1
                if gap > gap_allowed:
                    break
                continue
            last_in, last_p, gap = travelled, p, 0
            n = np.array([-d[1], d[0]])
            pts = p + offs[:, None] * n
            xs, ys = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            on = np.zeros(len(offs), bool)
            on[ok] = free[ys[ok], xs[ok]]
            c = len(offs) // 2
            lo = hi = c
            while lo > 0 and on[lo - 1]:
                lo -= 1
            while hi < len(on) - 1 and on[hi + 1]:
                hi += 1
            if lo > 0 and hi < len(on) - 1:          # a limb-sized run: ease towards its middle
                shift = float(np.clip((offs[lo] + offs[hi]) / 2, -0.1 * arm_w, 0.1 * arm_w))
                new_p = p + n * shift
                turn = new_p - (p - d * step)
                d = 0.85 * d + 0.15 * turn / max(np.linalg.norm(turn), 1e-6)
                d /= max(np.linalg.norm(d), 1e-6)
                p = last_p = new_p
        edge = max(3, int(0.01 * max(h, w)))     # the mask often stops just short of the edge
        near_edge = min(last_p[0], last_p[1], w - 1 - last_p[0], h - 1 - last_p[1]) < edge
        return last_in, bool(near_edge)

    @staticmethod
    def _reach(free, start, d, limit):
        """How far from start along d the outline continues (gaps under 4% of limit are bridged).

        Returns (distance, left_frame): left_frame is True when the image edge,
        not the end of the arm, stopped it.
        """
        h, w = free.shape
        gap_allowed = max(2, int(0.04 * limit))
        edge = max(3, int(0.01 * max(h, w)))   # the mask often stops just short of the edge
        last_in, gap = 0.0, 0
        for t in np.arange(0, limit, 1.0):
            x, y = int(round(start[0] + t * d[0])), int(round(start[1] + t * d[1]))
            if not (0 <= x < w and 0 <= y < h):
                return last_in, True
            if free[y, x]:
                last_in, gap = t, 0
            else:
                gap += 1
                if gap > gap_allowed:
                    break
        end = start + last_in * d
        near_edge = min(end[0], end[1], w - 1 - end[0], h - 1 - end[1]) < edge
        return last_in, bool(near_edge)


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
