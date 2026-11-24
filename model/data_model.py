"""DataModel: stores attributes/properties, the DeepFashion dataset, and the uploaded image."""


class UserImage:
    """The image uploaded by the user."""

    def __init__(self, path):
        pass

    def load(self):
        pass

    def get_body_mask(self):
        pass

    def set_body_mask(self, mask):
        pass

    def get_keypoints(self):
        pass

    def set_keypoints(self, keypoints):
        pass


class Garment:
    """A single clothing item and its attributes.

    DeepFashion garments are photos of people wearing the clothes, so get_mask()
    cuts the garment out of the photo using its bounding box and landmarks.
    Landmark names use IMAGE left/right (see model/deepfashion.py).
    """

    def __init__(self, garment_id, image_path, category, attributes,
                 clothes_type=1, bbox=None, landmarks=None, visibility=None,
                 loader=None, mask=None):
        self.garment_id = garment_id
        self.image_path = image_path
        self.category = category
        self.attributes = attributes or {}
        self.clothes_type = clothes_type        # 1 upper, 2 lower, 3 full body
        self.bbox = bbox                        # (x1, y1, x2, y2) or None
        self.landmarks = dict(landmarks or {})  # {name: (x, y)}
        self.visibility = dict(visibility or {})
        self._loader = loader                   # callable(image_path) -> BGR image
        self.image = None
        self.mask = mask
        self.photo_kind = None                  # "product" or "worn", set by get_mask()

    def load(self):
        if self.image is None:
            if self._loader is not None:
                self.image = self._loader(self.image_path)
            else:
                import cv2
                self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError(f"could not read garment image {self.image_path}")
        return self.image

    def get_mask(self):
        from controller.garment_cutout import cut_out_garment, photo_kind
        if self.photo_kind is None:
            self.photo_kind = photo_kind(self.load())
        if self.mask is None:
            self.mask = cut_out_garment(self.load(), self.bbox, self.get_landmarks(),
                                        self.clothes_type)
        return self.mask

    def get_landmarks(self, visible_only=True):
        """Landmarks as {name: (x, y)}; occluded/cut-off ones dropped by default."""
        if not visible_only:
            return dict(self.landmarks)
        return {k: p for k, p in self.landmarks.items() if self.visibility.get(k, 0) == 0}


class DataModel:
    """Central store for the dataset, wardrobe, and current user image."""

    def __init__(self, dataset_root):
        pass

    def load_dataset(self):
        pass

    def set_user_image(self, path):
        pass

    def get_user_image(self):
        pass

    def get_wardrobe(self):
        pass

    def get_garment(self, garment_id):
        pass

    def filter_garments(self, category=None, style=None):
        pass

    def save_result(self, image, output_path):
        pass
