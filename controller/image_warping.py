"""Step 2: warp a garment image to fit the segmented body and overlay it.

Inputs
  * a Garment (model/data_model.py): BGR image, mask (255 = garment) and
    landmarks named with IMAGE left/right, e.g. "collar_l" has the smaller x;
  * BodySegmenter.run() output: "keypoints" use the PERSON's left/right
    ("left_shoulder" is on the image's right), "body_mask" is 0/255.

How it works
  1. compute_correspondences() turns garment landmarks + body keypoints into
     matching point pairs (garment collar -> base of the neck, hem -> hip line,
     sleeve ends -> arms when elbow/wrist keypoints exist).
  2. The garment (with its mask as an alpha channel) is warped onto the body with
     one of three methods, each a fallback for the next:
       "affine"      least-squares affine fit (scale, rotate, shear, move)
       "perspective" homography, needs 4+ pairs (otherwise falls back to affine)
       "tps"         thin plate spline: passes exactly through every pair and
                     bends smoothly in between (implemented with numpy below)
     With elbow keypoints, each sleeve is cut out of the garment and swung onto
     its arm separately -- one piece for a short sleeve, two bending at the
     elbow for a long one -- so the torso of the garment isn't dragged along.
  3. blend() alpha-composites the warped garment with feathered edges.
"""

import cv2
import numpy as np

METHODS = ("affine", "perspective", "tps")


# --------------------------------------------------------------------------- fits
def fit_affine(src, dst):
    """2x3 matrix mapping src -> dst (least squares; similarity for 2 points)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if len(src) < 2:
        raise ValueError("need at least 2 point pairs")
    if len(src) == 2:
        # Similarity transform: x' = a*x - b*y + tx, y' = b*x + a*y + ty
        rows, rhs = [], []
        for (x, y), (u, v) in zip(src, dst):
            rows += [[x, -y, 1, 0], [y, x, 0, 1]]
            rhs += [u, v]
        a, b, tx, ty = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        return np.array([[a, -b, tx], [b, a, ty]])
    design = np.hstack([src, np.ones((len(src), 1))])
    sol = np.linalg.lstsq(design, dst, rcond=None)[0]   # 3x2
    return sol.T


def fit_perspective(src, dst):
    """3x3 homography src -> dst (least squares for >4 points); affine if <4 points."""
    src, dst = np.asarray(src, np.float32), np.asarray(dst, np.float32)
    if len(src) >= 4:
        H, _ = cv2.findHomography(src, dst, 0)
        if H is not None:
            return H
    return np.vstack([fit_affine(src, dst), [0, 0, 1]])


class ThinPlateSpline:
    """2-D thin plate spline mapping fitted points src -> dst.

    f(p) = A p + t + sum_i w_i U(|p - c_i|),  U(r) = r^2 log r
    ``regularization`` > 0 trades exactness for smoothness.
    """

    def __init__(self, src, dst, regularization=0.0):
        c = np.asarray(src, np.float64)
        d = np.asarray(dst, np.float64)
        n = len(c)
        if n < 3:
            raise ValueError("thin plate spline needs at least 3 point pairs")
        K = self._kernel(c, c) + regularization * np.eye(n)
        P = np.hstack([np.ones((n, 1)), c])
        L = np.zeros((n + 3, n + 3))
        L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
        rhs = np.vstack([d, np.zeros((3, 2))])
        params = np.linalg.lstsq(L, rhs, rcond=None)[0]
        self.centres = c
        self.weights, self.affine = params[:n], params[n:]

    @staticmethod
    def _kernel(a, b):
        r2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 0.5 * r2 * np.log(r2)
        return np.nan_to_num(k)

    def __call__(self, points):
        p = np.asarray(points, np.float64).reshape(-1, 2)
        out = np.empty_like(p)
        for start in range(0, len(p), 200_000):      # chunk to bound memory
            chunk = p[start:start + 200_000]
            k = self._kernel(chunk, self.centres)
            out[start:start + len(chunk)] = (self.affine[0] + chunk @ self.affine[1:]
                                             + k @ self.weights)
        return out


def _apply_matrix(M, pts):
    pts = np.asarray(pts, np.float64)
    if M.shape == (2, 3):
        return pts @ M[:, :2].T + M[:, 2]
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    return h[:, :2] / h[:, 2:3]


# ------------------------------------------------------------------- body helpers
def _body_point(keypoints, image_side, part):
    """Body keypoint on the IMAGE left ('l') or right ('r'). Person-right = image-left."""
    person = "right" if image_side == "l" else "left"
    p = keypoints.get(f"{person}_{part}")
    return None if p is None else np.asarray(p, np.float64)


def _normalise_sides(g, clothes_type):
    """Make "_l" landmarks the image-left ones (back-view photos have them swapped)."""
    left, right = ("waist_l", "waist_r") if clothes_type == 2 else ("collar_l", "collar_r")
    if left in g and right in g and g[left][0] > g[right][0]:
        swapped = {}
        for k, v in g.items():
            if k.endswith("_l"):
                swapped[k[:-2] + "_r"] = v
            elif k.endswith("_r"):
                swapped[k[:-2] + "_l"] = v
            else:
                swapped[k] = v
        return swapped
    return g


def check_body_proportions(segmentation):
    """Raise ValueError when the keypoints can't be the person's real torso.

    Guessed keypoints (no face or pose model) can land on the face: in one
    DeepFashion photo the shoulder-to-hip distance came out at 11% of the body's
    height. A real torso is roughly 30-60% of the visible body.
    """
    kp = segmentation.get("keypoints") or {}
    body = segmentation.get("body_mask")
    if body is None or segmentation.get("keypoint_source") == "openpose":
        return
    sh = [kp.get("left_shoulder"), kp.get("right_shoulder")]
    hp = [kp.get("left_hip"), kp.get("right_hip")]
    if None in sh or None in hp:
        return
    rows = np.nonzero((body > 0).any(axis=1))[0]
    if len(rows) == 0:
        return
    body_h = rows[-1] - rows[0] + 1
    torso = (hp[0][1] + hp[1][1]) / 2 - (sh[0][1] + sh[1][1]) / 2
    if torso < 0.18 * body_h:
        raise ValueError(f"torso keypoints look wrong (shoulder-to-hip is {torso:.0f}px of a "
                         f"{body_h}px tall body); body detection probably failed")


class GarmentWarper:
    """Aligns garment landmarks to body keypoints and blends the result."""

    def __init__(self, method="tps", feather=0.003, clip_margin=None, tps_regularization=1e-3,
                 shading=0.6, keep_arms="auto"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.feather = feather                  # edge blur, fraction of image diagonal
        self.clip_margin = clip_margin          # how far past the body mask a garment may reach
        self.tps_regularization = tps_regularization
        self.shading = shading                  # 0 = flat garment, 1 = full lighting from the photo
        # Keep bare arms in front of the garment. "auto": only when the pose has
        # elbow keypoints, because without them the sleeves stay in the catalog
        # model's pose and the wearer's arms show through as stripes.
        self.keep_arms = keep_arms
        self.last = {}                          # details of the most recent warp, for evaluation

    # ---------------------------------------------------------- correspondences
    def compute_correspondences(self, garment_landmarks, body_keypoints, clothes_type=1,
                                garment_mask=None, worn=False):
        """Return (src, dst, names): matching garment and body points, as Nx2 arrays.

        ``worn`` says the garment photo shows someone wearing it (arms hang beside
        the garment), as opposed to a flat product shot. With elbow keypoints a
        sleeve end's dst is on the arm; warp() moves the sleeve there.
        """
        src, dst, names, _, _ = self._correspondences(garment_landmarks, body_keypoints,
                                                      clothes_type, garment_mask, worn)
        return src, dst, names

    def _correspondences(self, garment_landmarks, body_keypoints, clothes_type=1,
                         garment_mask=None, worn=False):
        """compute_correspondences() plus (torso_dst, sleeves).

        torso_dst is where each point goes if the garment only follows the torso
        (sleeves stay in the catalog photo's pose); sleeves is {side: plan} for
        each sleeve to move onto an arm (see _sleeve_plan).
        """
        sleeves = {}
        g = _normalise_sides(
            {k: np.asarray(v, np.float64) for k, v in garment_landmarks.items() if v is not None},
            clothes_type)
        kp = {k: v for k, v in body_keypoints.items() if v is not None}
        sh = {s: _body_point(kp, s, "shoulder") for s in "lr"}
        hip = {s: _body_point(kp, s, "hip") for s in "lr"}

        if sh["l"] is None or sh["r"] is None:
            if hip["l"] is None or hip["r"] is None:
                raise ValueError("need both shoulders (or, for trousers, both hips)")
        if clothes_type != 2 and (sh["l"] is None or sh["r"] is None):
            raise ValueError("tops and dresses need both shoulder keypoints")

        pairs = []
        if clothes_type in (1, 3):
            span = np.linalg.norm(sh["r"] - sh["l"])
            if hip["l"] is not None and hip["r"] is not None:
                torso = np.linalg.norm((hip["l"] + hip["r"]) / 2 - (sh["l"] + sh["r"]) / 2)
                if span < 10 or torso < 10 or span < 0.25 * torso:
                    raise ValueError("shoulder keypoints are too close together to be right "
                                     f"({span:.0f}px apart); pose detection probably failed")
            u = (sh["r"] - sh["l"]) / max(span, 1e-6)          # across the body, image-left -> right
            n = np.array([-u[1], u[0]])                        # down the body
            sh_mid = (sh["l"] + sh["r"]) / 2
            if hip["l"] is not None and hip["r"] is not None:
                hip_mid = (hip["l"] + hip["r"]) / 2
                hip_w = np.linalg.norm(hip["r"] - hip["l"])
            else:                                              # rough proportions
                hip_mid = sh_mid + n * 1.3 * span
                hip_w = 0.8 * span
            torso_len = max(float(np.dot(hip_mid - sh_mid, n)), 1e-6)

            def along(d):  # point at distance d below the shoulder line, on the torso centre line
                return sh_mid + (hip_mid - sh_mid) * (d / torso_len)

            if "collar_l" not in g or "collar_r" not in g:
                raise ValueError("tops and dresses need both collar landmarks")
            g_collar = (g["collar_l"] + g["collar_r"]) / 2
            low = [g[k] for k in ("waist_l", "waist_r", "hem_l", "hem_r") if k in g]
            # "Down" the garment runs from the collar to the hem. Taking it from the
            # left/right labels instead flips the garment when they're swapped.
            if low:
                g_n = np.mean(low, axis=0) - g_collar
                g_n /= max(np.linalg.norm(g_n), 1e-6)
            else:
                g_n = np.array([0.0, 1.0])
            g_u = np.array([g_n[1], -g_n[0]])                  # across, image-left -> right
            g_len = max(float(np.dot(np.mean(low, axis=0) - g_collar, g_n)), 1e-6) if low else 100.0

            # Size the garment by its own torso width against the wearer's, keeping
            # the garment's proportions.
            #  * Flat product shot: measure across the mask below the armpits (the
            #    hem can curve in or flare out, so it's a poor reference there).
            #  * Worn photo: that measurement runs across the model's arms too,
            #    about twice too wide, so use the distance between the hem corners
            #    (waist corners for dresses).
            corners = None
            for a, b in (("waist_l", "waist_r"), ("hem_l", "hem_r")):
                if a in g and b in g:
                    corners = abs(float(np.dot(g[b] - g[a], g_u)))
                    break
            g_width = None if worn else self._garment_torso_width(garment_mask, g_collar, g_u, g_n, g_len)
            if g_width is None:
                g_width = corners if corners else 3 * np.linalg.norm(g["collar_r"] - g["collar_l"])
                if low:
                    # Hem corners collapse when the model stands side-on or tucks
                    # the shirt in; keep the width within believable proportions
                    # of the garment's own length.
                    lo, hi = (0.45, 0.9) if clothes_type == 1 else (0.25, 0.7)
                    g_width = float(np.clip(g_width, lo * g_len, hi * g_len))
            body_width = 0.92 * span          # shoulder-tip span is a bit wider than the torso
            scale = body_width / max(g_width, 1e-6)
            # Final safety net: the garment's length on the body must be believable
            # (a top ends somewhere around the hips, not at the knees).
            lo, hi = (0.7, 1.5) if clothes_type == 1 else (1.0, 2.6)
            scale = float(np.clip(scale, lo * torso_len / g_len, hi * torso_len / g_len))

            anchor = along(0.0)               # neckline sits on the shoulder line

            def on_torso(p):                  # garment point -> body, following the torso
                rel = np.asarray(p, np.float64) - g_collar
                return anchor + scale * (u * np.dot(rel, g_u) + n * np.dot(rel, g_n))

            for name, p in g.items():
                pairs.append((name, p, on_torso(p)))
            torso_dst = {name: t for name, _, t in pairs}

            # With elbow/wrist keypoints, the sleeve ends go onto the arms.
            arm = {name: t for name, _, t in self._sleeve_pairs(g, kp, sh)}
            pairs = [(name, p, arm.get(name, t)) for name, p, t in pairs]
            half_w = 0.5 * (g_width if g_width else 2 * np.linalg.norm(g["collar_r"] - g["collar_l"]))
            for side in "lr":
                plan = self._sleeve_plan(g, arm, kp, side, g_collar, g_u, g_n, g_len, half_w,
                                         on_torso)
                if plan is not None:
                    sleeves[side] = plan

        else:  # trousers / skirts
            if hip["l"] is None or hip["r"] is None:
                raise ValueError("trousers and skirts need both hip keypoints")
            hip_w = np.linalg.norm(hip["r"] - hip["l"])
            u = (hip["r"] - hip["l"]) / max(hip_w, 1e-6)
            n = np.array([-u[1], u[0]])
            c = (hip["l"] + hip["r"]) / 2 - n * 0.05 * hip_w
            w = 1.05 * hip_w
            if "waist_l" in g and "waist_r" in g:
                scale = w / max(np.linalg.norm(g["waist_r"] - g["waist_l"]), 1e-6)
                pairs.append(("waist_l", g["waist_l"], c - u * w / 2))
                pairs.append(("waist_r", g["waist_r"], c + u * w / 2))
                if "hem_l" in g and "hem_r" in g:
                    pairs += self._hem_pairs(g, "waist_l", "waist_r", c, u, n, scale)

        if len(pairs) < 2:
            raise ValueError("not enough matching garment landmarks and body keypoints")
        names = [p[0] for p in pairs]
        src = np.array([p[1] for p in pairs], np.float64)
        dst = np.array([p[2] for p in pairs], np.float64)
        if clothes_type in (1, 3):
            torso = np.array([torso_dst[name] for name in names], np.float64)
        else:
            torso = dst.copy()
        return src, dst, names, torso, sleeves

    @staticmethod
    def _garment_torso_width(mask, collar, g_u, g_n, length):
        """Width of the garment body across, measured on its mask ~60% down from the collar."""
        if mask is None:
            return None
        h, w = mask.shape[:2]
        widths = []
        for frac in (0.55, 0.6, 0.65):
            centre = collar + g_n * frac * length
            ts = np.arange(-w, w, 1.0)
            pts = centre[None] + ts[:, None] * g_u[None]
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            on = np.zeros(len(ts), bool)
            on[inside] = mask[pts[inside, 1].astype(int), pts[inside, 0].astype(int)] > 0
            mid = len(ts) // 2
            if not on[mid]:
                continue
            left = mid
            while left > 0 and on[left - 1]:
                left -= 1
            right = mid
            while right < len(ts) - 1 and on[right + 1]:
                right += 1
            widths.append(right - left + 1)
        return float(np.median(widths)) if widths else None

    @staticmethod
    def _hem_pairs(g, top_l, top_r, top_centre, u, n, scale):
        """Place the hem below the waistband, keeping the garment's own proportions."""
        g_top = (g[top_l] + g[top_r]) / 2
        g_u = (g[top_r] - g[top_l]) / max(np.linalg.norm(g[top_r] - g[top_l]), 1e-6)
        g_n = np.array([-g_u[1], g_u[0]])
        out = []
        for name in ("hem_l", "hem_r"):
            rel = g[name] - g_top
            target = top_centre + scale * (u * np.dot(rel, g_u) + n * np.dot(rel, g_n))
            out.append((name, g[name], target))
        return out

    @staticmethod
    def _sleeve_pairs(g, kp, sh):
        """Sleeve ends -> arms, only when the pose model gave elbow/wrist keypoints."""
        out = []
        for side in "lr":
            name = f"sleeve_{side}"
            if name not in g or sh[side] is None:
                continue
            elbow = _body_point(kp, side, "elbow")
            wrist = _body_point(kp, side, "wrist")
            if elbow is None:
                continue
            collar = g.get(f"collar_{side}")
            hem = g.get(f"hem_{side}", g.get(f"waist_{side}"))
            long_sleeve = (collar is not None and hem is not None
                           and g[name][1] - collar[1] > 0.6 * (hem[1] - collar[1]))
            if long_sleeve:
                target = wrist if wrist is not None else elbow + 0.9 * (elbow - sh[side])
            else:
                target = sh[side] + 0.5 * (elbow - sh[side])
            out.append((name, g[name], target))
        return out

    # Upper arm : forearm = 33 : 26, so a straight long sleeve bends this far along.
    SLEEVE_ELBOW_AT = 33 / 59

    @classmethod
    def _sleeve_plan(cls, g, arm_targets, kp, side, g_collar, g_u, g_n, g_len, half_w, on_torso):
        """How to move one sleeve onto its arm, or None to leave it with the torso.

        The sleeve turns about a pivot near the top of its seam (a fifth of the
        way from the shoulder point down to the armpit), so it stays attached to
        the warped torso. A short sleeve is one piece pointing down the upper
        arm; a long sleeve is two pieces that bend at the elbow.
        """
        name = f"sleeve_{side}"
        elbow = _body_point(kp, side, "elbow")
        if name not in g or name not in arm_targets or elbow is None:
            return None
        sign = -1.0 if side == "l" else 1.0
        g_sh = g_collar + sign * half_w * g_u + 0.05 * g_len * g_n
        g_ax = g_collar + sign * half_w * g_u + 0.28 * g_len * g_n
        pivot = g_sh + 0.2 * (g_ax - g_sh)
        cuff = g[name]
        if np.linalg.norm(cuff - pivot) < 1e-3:
            return None
        b_pivot = on_torso(pivot)
        collar = g.get(f"collar_{side}")
        hem = g.get(f"hem_{side}", g.get(f"waist_{side}"))
        long_sleeve = (collar is not None and hem is not None
                       and cuff[1] - collar[1] > 0.6 * (hem[1] - collar[1]))
        if long_sleeve:
            g_elbow = pivot + cls.SLEEVE_ELBOW_AT * (cuff - pivot)
            pieces = [(pivot, g_elbow, b_pivot, elbow), (g_elbow, cuff, elbow, arm_targets[name])]
        else:
            pieces = [(pivot, cuff, b_pivot, arm_targets[name])]
        return {"sign": sign, "collar": g_collar, "across": g_u, "half_w": half_w,
                "pivot": pivot, "cuff": cuff, "pieces": pieces}

    @staticmethod
    def _sleeve_mask(garment_mask, plan):
        """The sleeve's pixels in the garment photo: beyond the torso's side, joined to the cuff."""
        h, w = garment_mask.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        across = (xs - plan["collar"][0]) * plan["across"][0] + (ys - plan["collar"][1]) * plan["across"][1]
        beyond = plan["sign"] * across > plan["half_w"]
        region = ((garment_mask > 127) & beyond).astype(np.uint8)
        # Drop slivers of the torso's own edge that reach past the measured width.
        k = max(3, int(0.08 * plan["half_w"]) | 1)
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels = cv2.connectedComponents(region)
        if n <= 1:
            return np.zeros((h, w), np.uint8)
        cx, cy = (int(np.clip(v, 0, m - 1)) for v, m in zip(plan["cuff"], (w, h)))
        lab = labels[cy, cx]
        if lab == 0:                                   # cuff landmark just off the mask
            on = np.argwhere(labels > 0)
            near = on[np.argmin(np.hypot(on[:, 0] - cy, on[:, 1] - cx))]
            lab = labels[near[0], near[1]]
        return (labels == lab).astype(np.uint8) * 255

    @staticmethod
    def _piece_matrix(g_from, g_to, b_from, b_to, width_scale):
        """Affine map of a sleeve piece: g_from -> b_from, g_to -> b_to, width kept at width_scale."""
        g_axis, b_axis = g_to - g_from, b_to - b_from
        g_len, b_len = np.linalg.norm(g_axis), np.linalg.norm(b_axis)
        if g_len < 1e-6 or b_len < 1e-6:
            return None
        g_perp = np.array([-g_axis[1], g_axis[0]]) / g_len
        b_perp = np.array([-b_axis[1], b_axis[0]]) / b_len
        src = np.float32([g_from, g_to, g_from + 10 * g_perp])
        dst = np.float32([b_from, b_to, b_from + 10 * width_scale * b_perp])
        return cv2.getAffineTransform(src, dst)

    def _warp_sleeves(self, garment_image, garment_mask, sleeves, width_scale, output_shape):
        """Warp each planned sleeve piece. Returns (rgba layer, {side: matrices}) at output_shape."""
        h, w = output_shape[:2]
        layer = np.zeros((h, w, 4), np.float32)
        matrices = {}
        for side, plan in sleeves.items():
            sleeve = self._sleeve_mask(garment_mask, plan)
            if not sleeve.any():
                continue
            gh, gw = sleeve.shape
            ys, xs = np.mgrid[0:gh, 0:gw]
            pieces = plan["pieces"]
            matrices[side] = []
            # Draw the forearm piece first so the upper-arm piece covers the elbow joint.
            for i in reversed(range(len(pieces))):
                g_from, g_to, b_from, b_to = pieces[i]
                M = self._piece_matrix(g_from, g_to, b_from, b_to, width_scale)
                if M is None:
                    continue
                matrices[side].insert(0, M)
                axis = g_to - g_from
                t = ((xs - g_from[0]) * axis[0] + (ys - g_from[1]) * axis[1]) / max(np.dot(axis, axis), 1e-6)
                part = sleeve.copy()
                if len(pieces) > 1:
                    # Overlap the pieces a little at the elbow so no gap opens.
                    keep = (t <= 1.06) if i == 0 else (t >= -0.06)
                    part[~keep] = 0
                rgba = np.dstack([garment_image, part])
                out = cv2.warpAffine(rgba, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
                a = out[..., 3:] / 255.0
                layer[..., :3] = out[..., :3] * a + layer[..., :3] * (1 - a)
                layer[..., 3:] = 255.0 * a + layer[..., 3:] * (1 - a)
        return layer, matrices

    # ------------------------------------------------------------------ warps
    @staticmethod
    def _rgba(garment_image, garment_mask):
        if garment_mask is None:
            garment_mask = np.full(garment_image.shape[:2], 255, np.uint8)
        return np.dstack([garment_image, garment_mask])

    def affine_warp(self, garment_image, src_points, dst_points, output_shape=None, garment_mask=None):
        h, w = (output_shape or garment_image.shape)[:2]
        M = fit_affine(src_points, dst_points)
        out = cv2.warpAffine(self._rgba(garment_image, garment_mask), M.astype(np.float32), (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.last["forward"] = lambda p, M=M: _apply_matrix(M, p)
        return out[..., :3], out[..., 3]

    def perspective_warp(self, garment_image, src_points, dst_points, output_shape=None, garment_mask=None):
        h, w = (output_shape or garment_image.shape)[:2]
        H = fit_perspective(src_points, dst_points)
        out = cv2.warpPerspective(self._rgba(garment_image, garment_mask), H.astype(np.float64), (w, h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.last["forward"] = lambda p, H=H: _apply_matrix(H, p)
        return out[..., :3], out[..., 3]

    def tps_warp(self, garment_image, src_points, dst_points, output_shape=None, garment_mask=None):
        h, w = (output_shape or garment_image.shape)[:2]
        src = np.asarray(src_points, np.float64)
        dst = np.asarray(dst_points, np.float64)
        if len(src) < 3:
            return self.affine_warp(garment_image, src, dst, output_shape, garment_mask)

        # Anchor the far parts of the garment to the overall affine fit, so the
        # spline only bends the garment between the landmarks instead of flinging
        # the edges around.
        gh, gw = garment_image.shape[:2]
        ys, xs = np.nonzero(garment_mask) if garment_mask is not None else (np.array([0, gh - 1]), np.array([0, gw - 1]))
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        anchors = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1],
                            [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1],
                            [x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]], np.float64)
        span = max(x1 - x0, y1 - y0, 1)
        far = np.min(np.linalg.norm(anchors[:, None] - src[None], axis=2), axis=1) > 0.15 * span
        anchors = anchors[far]
        M = fit_affine(src, dst)
        all_src = np.vstack([src, anchors])
        all_dst = np.vstack([dst, _apply_matrix(M, anchors)])

        scale = max(span, 1.0)
        reg = self.tps_regularization
        inverse = ThinPlateSpline(all_dst / scale, all_src / scale, reg)   # output -> garment
        forward = ThinPlateSpline(all_src / scale, all_dst / scale, reg)
        self.last["forward"] = lambda p: forward(np.asarray(p, np.float64) / scale) * scale

        # Only evaluate the spline over where the garment can land.
        corners = _apply_matrix(M, [[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
        pad = 0.2 * (corners.max(0) - corners.min(0))
        bx0, by0 = np.maximum(np.floor(corners.min(0) - pad), 0).astype(int)
        bx1, by1 = np.minimum(np.ceil(corners.max(0) + pad), [w - 1, h - 1]).astype(int)
        rgba = np.zeros((h, w, 4), np.uint8)
        if bx1 <= bx0 or by1 <= by0:
            return rgba[..., :3], rgba[..., 3]
        gx, gy = np.meshgrid(np.arange(bx0, bx1 + 1), np.arange(by0, by1 + 1))
        grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
        mapped = inverse(grid / scale) * scale
        map_x = mapped[:, 0].reshape(gx.shape).astype(np.float32)
        map_y = mapped[:, 1].reshape(gx.shape).astype(np.float32)
        rgba[by0:by1 + 1, bx0:bx1 + 1] = cv2.remap(
            self._rgba(garment_image, garment_mask), map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rgba[..., :3], rgba[..., 3]

    def warp(self, garment, body_keypoints, output_shape, method=None):
        """Warp a Garment onto the body. Returns (warped_bgr, warped_mask) at output_shape."""
        method = method or self.method
        image = garment.load()
        mask = garment.get_mask()
        src, dst, names, torso_dst, sleeves = self._correspondences(
            garment.get_landmarks(), body_keypoints, garment.clothes_type, mask,
            worn=getattr(garment, "photo_kind", None) == "worn")
        fn = {"affine": self.affine_warp, "perspective": self.perspective_warp,
              "tps": self.tps_warp}[method]
        self.last = {"method": method, "names": names, "src": src, "dst": dst}
        if not sleeves:
            warped, warped_mask = fn(image, src, dst, output_shape=output_shape, garment_mask=mask)
            self.last["warped_points"] = self.last["forward"](src)
            self.last["warped_mask"] = warped_mask
            self.last["sleeve_mask"] = None
            return warped, warped_mask

        # Sleeves that follow the arms: the torso is warped as if the sleeves
        # stayed put, but without them; each sleeve is then cut out and swung
        # onto its arm in one piece (short) or two pieces bending at the elbow.
        # The first bit of each sleeve stays with the torso as a shoulder cap, so
        # no notch opens at the shoulder whichever way the arm points.
        sleeve_px = np.zeros_like(mask)
        ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        for plan in sleeves.values():
            axis = plan["cuff"] - plan["pivot"]
            t = ((xs - plan["pivot"][0]) * axis[0] + (ys - plan["pivot"][1]) * axis[1]) / max(np.dot(axis, axis), 1e-6)
            sleeve_px |= np.where(t > 0.12, self._sleeve_mask(mask, plan), 0).astype(np.uint8)
        torso_mask = np.where(sleeve_px > 0, 0, mask).astype(np.uint8)
        body, body_mask = fn(image, src, torso_dst, output_shape=output_shape, garment_mask=torso_mask)
        torso_forward = self.last["forward"]
        M = fit_affine(src, torso_dst)
        width_scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        layer, matrices = self._warp_sleeves(image, mask, sleeves, width_scale, output_shape)

        a = layer[..., 3:] / 255.0
        warped = np.clip(layer[..., :3] * a + body.astype(np.float32) * (1 - a) + 0.5, 0, 255).astype(np.uint8)
        warped_mask = np.clip(layer[..., 3] + body_mask.astype(np.float32) * (1 - a[..., 0]) + 0.5,
                              0, 255).astype(np.uint8)

        points = torso_forward(src)
        for i, name in enumerate(names):
            side = name[-1]
            if name.startswith("sleeve_") and matrices.get(side):
                points[i] = _apply_matrix(matrices[side][-1], src[i:i + 1])[0]

        self.last["forward"] = torso_forward     # the garment's torso; sleeves are in matrices
        self.last["sleeve_matrices"] = matrices
        self.last["warped_points"] = points
        self.last["warped_mask"] = warped_mask
        self.last["sleeve_mask"] = layer[..., 3].astype(np.uint8)
        return warped, warped_mask

    # ------------------------------------------------------------------ blend
    def blend(self, body_image, warped_garment, garment_mask, body_mask=None, keypoints=None,
              protect_head=False, sleeve_mask=None):
        """Composite the warped garment over the photo.

        * soft (feathered) garment edges;
        * broad shading from the photo is carried onto the garment, so it is not flat;
        * bare arms beside the torso stay in front of the garment (needs keypoints);
        * protect_head: never draw over the head (only use when the keypoints come
          from a detected face or pose model, so the head position is trustworthy);
        * sleeve_mask: where warp() moved a sleeve onto an arm; the sleeve covers
          the arm there instead of the arm showing through it.
        """
        h, w = body_image.shape[:2]
        alpha = garment_mask.astype(np.float32) / 255.0
        if protect_head and keypoints is not None:
            head = self._head_box(keypoints, (h, w))
            if head is not None:
                # An ellipse rather than a box, so a high collar gets a rounded
                # neckline instead of a square notch.
                x0, y0, x1, y1 = head
                keep = np.ones((h, w), np.float32)
                cx, ry, rx = (x0 + x1) / 2, (y1 - y0), (x1 - x0) / 2
                cv2.ellipse(keep, (int(cx), int(y0)), (int(rx), int(ry)), 0, 0, 360, 0.0, -1)
                alpha *= keep
        if body_mask is not None and self.clip_margin is not None:
            r = max(1, int(self.clip_margin * h))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            alpha *= cv2.dilate(body_mask, kernel) > 0
        use_arms = self.keep_arms is True or (
            self.keep_arms == "auto" and keypoints is not None
            and (keypoints.get("left_elbow") is not None or keypoints.get("right_elbow") is not None))
        if use_arms and keypoints is not None:
            arms = self._arm_mask(body_image, body_mask, keypoints)
            if arms is not None:
                if sleeve_mask is not None:
                    arms = arms * (sleeve_mask < 128)
                alpha *= 1.0 - arms
        sigma = self.feather * np.hypot(h, w)
        if sigma > 0.3:
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
        alpha = np.clip(alpha, 0, 1)

        garment = warped_garment.astype(np.float32)
        if self.shading > 0 and (alpha > 0.5).any():
            gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            broad = cv2.GaussianBlur(gray, (0, 0), 0.02 * np.hypot(h, w))
            ratio = np.clip(broad / max(float(broad[alpha > 0.5].mean()), 1.0), 0.7, 1.3)
            garment *= (1.0 + self.shading * (ratio - 1.0))[..., None]

        a = alpha[..., None]
        out = body_image.astype(np.float32) * (1 - a) + garment * a
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    @staticmethod
    def _arm_mask(body_image, body_mask, keypoints):
        """Bare-arm pixels beside the torso, as 0/1 float.

        Skin is defined by the person's own face colour (sampled between the top of
        the head and the neck), so beige or tan clothing isn't mistaken for arms.
        """
        sh = {s: _body_point(keypoints, s, "shoulder") for s in "lr"}
        hip = {s: _body_point(keypoints, s, "hip") for s in "lr"}
        head, neck = keypoints.get("head_top"), keypoints.get("neck")
        if any(v is None for v in (sh["l"], sh["r"], hip["l"], hip["r"], head, neck)):
            return None
        h, w = body_image.shape[:2]
        lab = cv2.cvtColor(body_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        span = np.linalg.norm(sh["r"] - sh["l"])
        cx = int(neck[0])
        y0, y1 = int(head[1] + 0.45 * (neck[1] - head[1])), int(neck[1])
        x0, x1 = max(cx - int(0.12 * span), 0), min(cx + int(0.12 * span), w)
        face = lab[max(y0, 0):max(y1, 0), x0:x1].reshape(-1, 3)
        if len(face) < 30:
            return None
        ref = np.median(face, axis=0)
        # Distance in a/b (colour) with a looser limit on L (brightness changes along the arm).
        dab = np.linalg.norm(lab[..., 1:] - ref[1:], axis=2)
        skin = (dab < 9) & (np.abs(lab[..., 0] - ref[0]) < 60)

        u = (sh["r"] - sh["l"]) / max(span, 1e-6)
        inset = 0.12 * span
        core = np.array([sh["l"] + u * inset, sh["r"] - u * inset,
                         hip["r"] - u * 0.5 * inset, hip["l"] + u * 0.5 * inset], np.int32)
        torso = np.zeros((h, w), np.uint8)
        cv2.fillPoly(torso, [core], 255)
        if body_mask is not None:
            skin &= body_mask > 0
        skin &= torso == 0
        skin[:int(min(sh["l"][1], sh["r"][1]))] = False
        k = max(3, (h // 100) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        arms = cv2.morphologyEx(skin.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        # Keep only sizeable blobs (real arms), not scattered matches.
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(arms)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 0.002 * h * w
        return keep[lbl].astype(np.float32)

    @staticmethod
    def _head_box(keypoints, shape):
        """(x0, y0, x1, y1) of the head: from the top of the image down to the neck."""
        neck = keypoints.get("neck")
        sh = [keypoints.get("left_shoulder"), keypoints.get("right_shoulder")]
        if neck is None or None in sh:
            return None
        h, w = shape
        span = abs(sh[0][0] - sh[1][0])
        half = 0.3 * span                       # a head is roughly 0.6 shoulder-widths wide
        x0, x1 = int(max(neck[0] - half, 0)), int(min(neck[0] + half, w))
        y1 = int(min(neck[1], h))
        return (x0, 0, x1, y1) if x1 > x0 and y1 > 0 else None

    def try_on(self, body_image, segmentation, garment, method=None):
        """Convenience: warp + blend using BodySegmenter.run() output.

        Raises ValueError when the body keypoints can't be right (see
        check_body_proportions) instead of shrinking the garment onto the face.
        """
        check_body_proportions(segmentation)
        warped, mask = self.warp(garment, segmentation["keypoints"], body_image.shape, method)
        trusted = segmentation.get("keypoint_source") in ("face+mask", "openpose")
        return self.blend(body_image, warped, mask, segmentation.get("body_mask"),
                          segmentation.get("keypoints"), protect_head=trusted,
                          sleeve_mask=self.last.get("sleeve_mask"))
