"""Loading the DeepFashion dataset and fetching extra resources from the DeepFashion API."""


class DeepFashionDataset:
    """Reads images, categories, attributes, and landmarks from the local dataset."""

    def __init__(self, root_dir):
        pass

    def load_annotations(self):
        pass

    def load_categories(self):
        pass

    def load_attributes(self):
        pass

    def load_landmarks(self):
        pass

    def get_item(self, index):
        pass

    def __len__(self):
        pass


class DeepFashionAPI:
    """Client for additional DeepFashion resources."""

    def __init__(self, base_url):
        pass

    def fetch_garment(self, garment_id):
        pass

    def search(self, query):
        pass
