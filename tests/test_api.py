import os
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import create_app  # noqa: E402

IMAGE_NAME = "img/Basic_Tee/img_00000001.jpg"
SAMPLE_NAME = "img/Basic_Tee/img_00000002.jpg"


def write_dataset(root):
    files = {
        "list_landmarks.txt": (
            "2\nimage_name clothes_type variation_type landmarks\n"
            f"{IMAGE_NAME} 1 0 046 030 0 074 030 0 020 060 0 100 060 0 040 110 0 080 112\n"
            f"{SAMPLE_NAME} 1 0 046 030 1 074 030 0 020 060 0 100 060 0 040 110 0 080 112\n"
        ),
        "list_category_cloth.txt": "1\ncategory_name category_type\nBasic_Tee 1\n",
        "list_category_img.txt": f"2\nimage_name category_label\n{IMAGE_NAME} 1\n{SAMPLE_NAME} 1\n",
        "list_bbox.txt": (
            f"2\nimage_name x_1 y_1 x_2 y_2\n{IMAGE_NAME} 010 020 110 118\n"
            f"{SAMPLE_NAME} 010 020 110 118\n"
        ),
        "list_eval_partition.txt": (
            f"2\nimage_name evaluation_status\n{IMAGE_NAME} train\n{SAMPLE_NAME} train\n"
        ),
    }
    for filename, contents in files.items():
        (root / filename).write_text(contents)
    image = np.full((160, 120, 3), (40, 100, 190), dtype=np.uint8)
    image_path = root / IMAGE_NAME
    image_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), image)
    sample_path = root / SAMPLE_NAME
    assert cv2.imwrite(str(sample_path), image)
    return image


def encoded_image(image, extension=".png"):
    ok, output = cv2.imencode(extension, image)
    assert ok
    return output.tobytes()


def test_health_and_asset_errors_are_explicit(tmp_path):
    app = create_app(dataset_dir=tmp_path, sample_person_path=tmp_path / "missing.jpg")
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["dataset_ready"] is False
        assert health.json()["sample_ready"] is False

        garments = client.get("/api/garments")
        assert garments.status_code == 503
        assert garments.json()["error"]["code"] == "dataset_unavailable"

        sample = client.get("/api/sample-person")
        assert sample.status_code == 503
        assert sample.json()["error"]["code"] == "sample_unavailable"


def test_catalog_is_filtered_and_images_are_allowlisted(tmp_path):
    write_dataset(tmp_path)
    app = create_app(dataset_dir=tmp_path)
    with TestClient(app) as client:
        catalog = client.get("/api/garments").json()
        assert catalog["categories"] == ["Basic_Tee"]
        assert len(catalog["garments"]) == 1
        assert catalog["garments"][0]["imageUrl"] == "/api/garments/0/image"

        image = client.get("/api/garments/0/image")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert client.get("/api/garments/999/image").status_code == 404
        assert client.get("/api/garments?category=unknown").json()["garments"] == []
        sample = client.get("/api/sample-person")
        assert sample.status_code == 200
        assert sample.headers["content-type"] == "image/jpeg"


def test_try_on_returns_png_and_rejects_bad_uploads(tmp_path, monkeypatch):
    image = write_dataset(tmp_path)
    app = create_app(dataset_dir=tmp_path)

    class FakeSegmenter:
        def run(self, uploaded_image):
            return {
                "body_mask": np.full(uploaded_image.shape[:2], 255, np.uint8),
                "keypoints": {"left_shoulder": (80, 40), "right_shoulder": (40, 40)},
            }

    class FakeWarper:
        def try_on(self, uploaded_image, segmentation, garment):
            assert garment.garment_id == 0
            return uploaded_image.copy()

    monkeypatch.setattr("api.app.BodySegmenter", FakeSegmenter)
    monkeypatch.setattr("api.app.GarmentWarper", FakeWarper)
    app_image = encoded_image(image)

    with TestClient(app) as client:
        response = client.post(
            "/api/try-on",
            files={"person": ("person.png", app_image, "image/png")},
            data={"garment_id": "0"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR).shape == image.shape

        unsupported = client.post(
            "/api/try-on",
            files={"person": ("person.gif", b"gif", "image/gif")},
            data={"garment_id": "0"},
        )
        assert unsupported.status_code == 400
        assert unsupported.json()["error"]["code"] == "unsupported_image_type"

        invalid = client.post(
            "/api/try-on",
            files={"person": ("person.png", b"\x89PNG\r\n\x1a\nnot an image", "image/png")},
            data={"garment_id": "0"},
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_image"

        mismatched = client.post(
            "/api/try-on",
            files={"person": ("person.png", encoded_image(image, ".jpg"), "image/png")},
            data={"garment_id": "0"},
        )
        assert mismatched.status_code == 400
        assert mismatched.json()["error"]["code"] == "image_type_mismatch"

        tiny = encoded_image(np.zeros((32, 32, 3), np.uint8))
        invalid_dimensions = client.post(
            "/api/try-on",
            files={"person": ("tiny.png", tiny, "image/png")},
            data={"garment_id": "0"},
        )
        assert invalid_dimensions.status_code == 422
        assert invalid_dimensions.json()["error"]["code"] == "invalid_dimensions"

        oversized = client.post(
            "/api/try-on",
            files={"person": ("large.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
            data={"garment_id": "0"},
        )
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "image_too_large"


def test_try_on_reports_missing_shoulders(tmp_path, monkeypatch):
    image = write_dataset(tmp_path)
    app = create_app(dataset_dir=tmp_path)

    class NoShouldersSegmenter:
        def run(self, uploaded_image):
            return {
                "body_mask": np.full(uploaded_image.shape[:2], 255, np.uint8),
                "keypoints": {"left_shoulder": None, "right_shoulder": None},
            }

    class UnusedWarper:
        def try_on(self, *args):
            raise AssertionError("the warper must not run without shoulders")

    monkeypatch.setattr("api.app.BodySegmenter", NoShouldersSegmenter)
    monkeypatch.setattr("api.app.GarmentWarper", UnusedWarper)

    with TestClient(app) as client:
        response = client.post(
            "/api/try-on",
            files={"person": ("person.png", encoded_image(image), "image/png")},
            data={"garment_id": "0"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "shoulders_not_found"
    assert "shoulders" in response.json()["error"]["message"]


def test_try_on_puts_the_sleeves_on_the_arms(tmp_path):
    """The real pipeline, end to end: a long-sleeved shirt on a person holding their arms out."""
    from test_body_segmentation import make_person, segmenter_with_face

    # A one-garment DeepFashion-style dataset: a red long-sleeved product shot.
    shirt = np.full((300, 260, 3), 245, np.uint8)
    poly = np.array([[105, 40], [155, 40], [195, 60], [235, 230], [210, 238], [178, 110],
                     [175, 260], [85, 260], [82, 110], [50, 238], [25, 230], [65, 60]], np.int32)
    cv2.fillPoly(shirt, [poly], (40, 40, 200))
    points = [(115, 42), (145, 42), (37, 234), (222, 234), (88, 258), (172, 258)]
    landmarks = " ".join(f"0 {x:03d} {y:03d}" for x, y in points)
    files = {
        "list_landmarks.txt": f"1\nimage_name clothes_type variation_type landmarks\n{IMAGE_NAME} 1 {landmarks}\n",
        "list_category_cloth.txt": "1\ncategory_name category_type\nBasic_Tee 1\n",
        "list_category_img.txt": f"1\nimage_name category_label\n{IMAGE_NAME} 1\n",
        "list_bbox.txt": f"1\nimage_name x_1 y_1 x_2 y_2\n{IMAGE_NAME} 020 035 240 265\n",
        "list_eval_partition.txt": f"1\nimage_name evaluation_status\n{IMAGE_NAME} train\n",
    }
    for filename, contents in files.items():
        (tmp_path / filename).write_text(contents)
    (tmp_path / IMAGE_NAME).parent.mkdir(parents=True)
    assert cv2.imwrite(str(tmp_path / IMAGE_NAME), shirt)

    person, _, kp = make_person(right=(35, 35), left=(35, 35))            # A pose
    app = create_app(dataset_dir=tmp_path, sample_person_path=tmp_path / "missing.jpg")
    with TestClient(app) as client:
        app.state.segmenter = segmenter_with_face()
        response = client.post(
            "/api/try-on",
            data={"garment_id": "0"},
            files={"person": ("person.png", encoded_image(person), "image/png")},
        )
    assert response.status_code == 200, response.text
    result = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

    def is_red(p):
        b, g, r = result[int(p[1]), int(p[0])].astype(int)
        return r > 140 and r > b + 60 and r > g + 60

    for side in ("right", "left"):
        upper = kp[f"{side}_shoulder"] + 0.6 * (kp[f"{side}_elbow"] - kp[f"{side}_shoulder"])
        fore = kp[f"{side}_elbow"] + 0.5 * (kp[f"{side}_wrist"] - kp[f"{side}_elbow"])
        assert is_red(upper) and is_red(fore), side          # the sleeve is on the arm
