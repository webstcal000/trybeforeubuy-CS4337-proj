"""Controller: runs segmentation and warping on the DataModel and hands results to the View."""

from controller.body_segmentation import BodySegmenter
from controller.image_warping import GarmentWarper


class TryOnController:
    """Connects the DataModel and the View."""

    def __init__(self, model, view):
        pass

    def handle_upload(self, image_path):
        pass

    def handle_garment_selected(self, garment_id):
        pass

    def handle_filter_changed(self, category=None, style=None):
        pass

    def try_on(self, garment_id):
        pass

    def try_on_outfit(self, garment_ids):
        pass

    def handle_save(self, output_path):
        pass
