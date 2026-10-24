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
    """A single clothing item and its attributes."""

    def __init__(self, garment_id, image_path, category, attributes):
        pass

    def load(self):
        pass

    def get_mask(self):
        pass

    def get_landmarks(self):
        pass


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
