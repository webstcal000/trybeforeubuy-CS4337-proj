"""Step 1: detect and segment the body region of the uploaded image with OpenCV."""


class BodySegmenter:
    """Finds the person in an image and produces a body mask and keypoints."""

    def __init__(self):
        pass

    def preprocess(self, image):
        pass

    def detect_person(self, image):
        pass

    def segment_body(self, image, bounding_box):
        pass

    def segment_torso(self, body_mask):
        pass

    def detect_keypoints(self, image):
        pass

    def refine_mask(self, mask):
        pass

    def run(self, image):
        pass
