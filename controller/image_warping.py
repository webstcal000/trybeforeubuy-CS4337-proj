"""Step 2: warp a garment image to fit the segmented body and overlay it.

Inputs
  * a Garment (model/data_model.py): BGR image, mask (255 = garment) and
    landmarks named with IMAGE left/right, e.g. "collar_l" has the smaller x;
  * BodySegmenter.run() output: "keypoints" use the PERSON's left/right
    ("left_shoulder" is on the image's right), "body_mask" is 0/255.

How it works
  1. compute_correspondences() turns garment landmarks + body keypoints into
     matching point pairs (garment collar -> base of the neck, hem -> hip line,
     sleeves -> arms when elbow/wrist keypoints exist).
  2. The garment (with its mask as an alpha channel) is warped onto the body with
     one of three methods, each a fallback for the next:
       "affine"      least-squares affine fit (scale, rotate, shear, move)
       "perspective" homography, needs 4+ pairs (otherwise falls back to affine)
       "tps"         thin plate spline: passes exactly through every pair and
                     bends smoothly in between (implemented with numpy below)
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


class GarmentWarper:
    """Aligns garment landmarks to body keypoints and blends the result."""

    def __init__(self, method="tps", feather=0.003, clip_margin=None, tps_regularization=1e-3,
                 shading=0.6, keep_arms=True):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.feather = feather                  # edge blur, fraction of image diagonal
        self.clip_margin = clip_margin          # how far past the body mask a garment may reach
        self.tps_regularization = tps_regularization
        self.shading = shading                  # 0 = flat garment, 1 = full lighting from the photo
        self.keep_arms = keep_arms              # keep bare arms in front of the garment
        self.last = {}                          # details of the most recent warp, for evaluation

    # ---------------------------------------------------------- correspondences
    def compute_correspondences(self, garment_landmarks, body_keypoints, clothes_type=1, garment_mask=None):
        """Return (src, dst, names): matching garment and body points, as Nx2 arrays."""
        g = {k: np.asarray(v, np.float64) for k, v in garment_landmarks.items() if v is not None}
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
            g_u = g["collar_r"] - g["collar_l"]
            g_u /= max(np.linalg.norm(g_u), 1e-6)
            g_n = np.array([-g_u[1], g_u[0]])
            low = [g[k] for k in ("waist_l", "waist_r", "hem_l", "hem_r") if k in g]
            g_low = np.mean(low, axis=0) if low else g_collar + g_n * 100
            g_len = max(float(np.dot(g_low - g_collar, g_n)), 1e-6)

            # Size the garment by its own torso width (measured on its mask below
            # the armpits, where sleeves don't interfere) against the wearer's
            # torso width, and keep the garment's proportions. Hem width is a poor
            # size reference: many tops curve in or flare out at the hem.
            g_width = self._garment_torso_width(garment_mask, g_collar, g_u, g_n, g_len)
            if g_width is None:
                g_width = np.linalg.norm(low[1] - low[0]) if len(low) >= 2 else 3 * np.linalg.norm(
                    g["collar_r"] - g["collar_l"])
            body_width = 0.92 * span          # shoulder-tip span is a bit wider than the torso
            scale = body_width / max(g_width, 1e-6)

            anchor = along(-0.06 * span)      # base of the neck
            for name, p in g.items():
                rel = p - g_collar
                target = anchor + scale * (u * np.dot(rel, g_u) + n * np.dot(rel, g_n))
                pairs.append((name, p, target))

            # With elbow/wrist keypoints, pull the sleeve ends onto the arms.
            arm = dict((name, t) for name, _, t in self._sleeve_pairs(g, kp, sh))
            pairs = [(name, p, arm.get(name, t)) for name, p, t in pairs]

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
        return src, dst, names

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
        src, dst, names = self.compute_correspondences(
            garment.get_landmarks(), body_keypoints, garment.clothes_type, mask)
        fn = {"affine": self.affine_warp, "perspective": self.perspective_warp,
              "tps": self.tps_warp}[method]
        self.last = {"method": method, "names": names, "src": src, "dst": dst}
        warped, warped_mask = fn(image, src, dst, output_shape=output_shape, garment_mask=mask)
        self.last["warped_points"] = self.last["forward"](src)
        self.last["warped_mask"] = warped_mask
        return warped, warped_mask

    # ------------------------------------------------------------------ blend
    def blend(self, body_image, warped_garment, garment_mask, body_mask=None, keypoints=None):
        """Composite the warped garment over the photo.

        * soft (feathered) garment edges;
        * broad shading from the photo is carried onto the garment, so it is not flat;
        * bare arms beside the torso stay in front of the garment (needs keypoints).
        """
        h, w = body_image.shape[:2]
        alpha = garment_mask.astype(np.float32) / 255.0
        if body_mask is not None and self.clip_margin is not None:
            r = max(1, int(self.clip_margin * h))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            alpha *= cv2.dilate(body_mask, kernel) > 0
        if self.keep_arms and keypoints is not None:
            arms = self._arm_mask(body_image, body_mask, keypoints)
            if arms is not None:
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
        """Bare-skin pixels on the person outside the central torso (i.e. arms), as 0/1 float."""
        from controller.garment_cutout import skin_mask
        sh = {s: _body_point(keypoints, s, "shoulder") for s in "lr"}
        hip = {s: _body_point(keypoints, s, "hip") for s in "lr"}
        if any(v is None for v in (sh["l"], sh["r"], hip["l"], hip["r"])):
            return None
        span = np.linalg.norm(sh["r"] - sh["l"])
        u = (sh["r"] - sh["l"]) / max(span, 1e-6)
        inset = 0.12 * span
        core = np.array([sh["l"] + u * inset, sh["r"] - u * inset,
                         hip["r"] - u * 0.5 * inset, hip["l"] + u * 0.5 * inset], np.int32)
        torso = np.zeros(body_image.shape[:2], np.uint8)
        cv2.fillPoly(torso, [core], 255)
        skin = skin_mask(body_image) > 0
        if body_mask is not None:
            skin &= body_mask > 0
        skin &= torso == 0
        top = int(min(sh["l"][1], sh["r"][1]))
        skin[:top] = False                      # neck and face are handled by the collar
        k = max(3, (body_image.shape[0] // 120) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        arms = cv2.morphologyEx(skin.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        return (arms > 0).astype(np.float32)

    def try_on(self, body_image, segmentation, garment, method=None):
        """Convenience: warp + blend using BodySegmenter.run() output."""
        warped, mask = self.warp(garment, segmentation["keypoints"], body_image.shape, method)
        return self.blend(body_image, warped, mask, segmentation.get("body_mask"),
                          segmentation.get("keypoints"))
