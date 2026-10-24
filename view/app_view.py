"""View: basic UI for uploading a photo, browsing the wardrobe, and showing the result."""


class AppView:
    """Window that displays images and forwards user actions to the controller."""

    def __init__(self):
        pass

    def set_controller(self, controller):
        pass

    def build_layout(self):
        pass

    def on_upload_clicked(self):
        pass

    def on_garment_clicked(self, garment_id):
        pass

    def on_filter_changed(self):
        pass

    def on_save_clicked(self):
        pass

    def show_user_image(self, image):
        pass

    def show_wardrobe(self, garments):
        pass

    def show_result(self, image):
        pass

    def show_error(self, message):
        pass

    def run(self):
        pass
