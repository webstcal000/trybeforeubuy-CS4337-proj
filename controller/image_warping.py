"""Step 2: warp a garment image to fit the segmented body and overlay it."""


class GarmentWarper:
    """Aligns garment landmarks to body keypoints and blends the result."""

    def __init__(self, method="tps"):
        pass

    def compute_correspondences(self, garment_landmarks, body_keypoints):
        pass

    def affine_warp(self, garment_image, src_points, dst_points):
        pass

    def perspective_warp(self, garment_image, src_points, dst_points):
        pass

    def tps_warp(self, garment_image, src_points, dst_points):
        pass

    def warp(self, garment, body_keypoints, output_shape):
        pass

    def blend(self, body_image, warped_garment, garment_mask, body_mask):
        pass
