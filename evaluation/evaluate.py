"""Step 4: evaluate try-on quality and user experience."""


def segmentation_iou(predicted_mask, ground_truth_mask):
    pass


def ssim_score(result_image, reference_image):
    pass


def time_pipeline(controller, image_path, garment_id):
    pass


def collect_user_feedback(result_id, rating, comments):
    pass


def landmark_alignment_error(warped_points, target_points, normalizer=1.0):
    """Mean distance between where the garment landmarks landed and where they should be."""
    import numpy as np
    warped = np.asarray(warped_points, np.float64)
    target = np.asarray(target_points, np.float64)
    return float(np.linalg.norm(warped - target, axis=1).mean() / max(normalizer, 1e-6))


def garment_fit_scores(warp_info, segmentation):
    """Scores for one warp (GarmentWarper.last after warp()) against BodySegmenter.run() output.

    landmark_error  mean landmark miss, as a fraction of shoulder width (0 = exact)
    torso_coverage  share of the torso mask the garment covers (tops/dresses)
    spill           share of the garment that lands off the body mask
    """
    import numpy as np
    kp = segmentation["keypoints"]
    a, b = kp.get("left_shoulder"), kp.get("right_shoulder")
    if a is None or b is None:
        a, b = kp.get("left_hip"), kp.get("right_hip")
    span = np.linalg.norm(np.subtract(a, b)) if a is not None and b is not None else 1.0
    garment = warp_info["warped_mask"] > 0
    torso = segmentation["torso_mask"] > 0
    body = segmentation["body_mask"] > 0
    return {
        "landmark_error": landmark_alignment_error(warp_info["warped_points"], warp_info["dst"], span),
        "torso_coverage": float((garment & torso).sum() / max(torso.sum(), 1)),
        "spill": float((garment & ~body).sum() / max(garment.sum(), 1)),
    }


def run_evaluation(model, controller, samples):
    pass


if __name__ == "__main__":
    pass
